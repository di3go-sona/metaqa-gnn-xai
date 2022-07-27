#%%
from typing import List

from regex import P
from dataset.dataloaders import EmbeddingsData, QAData
from pykeen.nn.modules import DistMultInteraction, TransEInteraction, ERMLPInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss

import wandb, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.nn import Embedding
from transformers import AutoTokenizer, BertModel
import torch_geometric
#%%
class JointQAModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, 
                 emb_size, 
                 hidden_sizes,
                 lr,
                 num_negs_per_pos,
                 bert_model = "prajjwal1/bert-mini",
                 fast=False) -> None:
        super().__init__()
        

        self.lr = lr
        self.num_negs_per_pos = num_negs_per_pos
        
        self.qa_interaction = DistMultInteraction()
        # self.qa_interaction = ERMLPInteraction(hidden_sizes[-1], 32)
        self.kge_interaction = TransEInteraction(1)
        
        self.qa_neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets), 
                                                num_negs_per_pos=self.num_negs_per_pos,
                                                corruption_scheme=('head',))
    
        
        self.kge_neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets), 
                                        num_negs_per_pos=self.num_negs_per_pos,
                                        corruption_scheme=('head','tail'))
        
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        
        self.text_model = BertModel.from_pretrained(bert_model)
        
        
        if self.text_model.config.hidden_size != hidden_sizes[-1]:
            self.text_linear = torch.nn.Linear(self.text_model.config.hidden_size, hidden_sizes[-1])
        else:
            self.text_linear = torch.nn.Identity()
            

        self.layers = len(hidden_sizes)
        if fast:
            self.rgcn1 = torch_geometric.nn.conv.FastRGCNConv(emb_size, hidden_sizes[0], kg_data.n_relations*2, aggr='mean')
        else:
            self.rgcn1 = torch_geometric.nn.conv.RGCNConv(emb_size, hidden_sizes[0], kg_data.n_relations*2, aggr='mean')
        
        if self.layers > 1:
            if fast:
                self.rgcn2 = torch_geometric.nn.conv.FastRGCNConv(hidden_sizes[0], hidden_sizes[1], kg_data.n_relations*2, aggr='mean')
            else:
                self.rgcn2 = torch_geometric.nn.conv.RGCNConv(hidden_sizes[0], hidden_sizes[1], kg_data.n_relations*2, aggr='mean')
            
        
        
        # Load triples
        triples = kg_data.triplets.cuda()
        itriples = torch.index_select(triples, 1, torch.tensor([2,1,0], dtype=torch.long, device=triples.device))
        itriples[:, 1]  += kg_data.n_relations
        self.triples = torch.cat((triples, itriples),0) 

        self.nodes_emb = Embedding(kg_data.n_nodes, emb_size, max_norm=1)
        self.relations_emb = Embedding(kg_data.n_relations, emb_size, max_norm=1)
  
        
        self.save_hyperparameters()
    
    def get_name(self):
        return f'{self.qa_interaction}|{self.kge_interaction}|{self.nodes_emb.embedding_dim}'
    
    def on_validation_batch_start(self, *args) -> None:
        self.triples = self.triples.to(self.device)
        x = self.nodes_emb(torch.arange(self.nodes_emb.num_embeddings, device=self.device))
        self.rgcn_nodes_emb = self.encode_nodes(x, self.triples)
        
    def on_train_batch_start(self, *args) -> None:
        self.triples = self.triples.to(self.device)
        x = self.nodes_emb(torch.arange(self.nodes_emb.num_embeddings, device=self.device))
        self.rgcn_nodes_emb = self.encode_nodes(x, self.triples)
    
    def encode_nodes(self, x, index):

        index = index.to(self.device)
        rgcn_nodes_emb = self.rgcn1(x, index[:,[0,2]].T, index[:,1])
        if self.layers > 1 :
            rgcn_nodes_emb = self.rgcn2(rgcn_nodes_emb.relu(), index[:,[0,2]].T, index[:,1])
        return rgcn_nodes_emb
    
    def qa_encode_question(self, question_toks):
        ''' Encodes the text tokens, input should have shape (seq_len, batch_size), return has shape (batch_size, embedding_size)'''

        enc = self.text_model(question_toks.T).last_hidden_state[:,0,:]
        enc_hat = self.text_linear(enc)
        return enc_hat
        
    def qa_encode_nodes(self, nodes: List[ torch.tensor ]):
        ''' Encodes the node tokens '''
        
        for n in nodes:
            yield self.rgcn_nodes_emb[n]
         
    def qa_encode(self, s, t, questions_emb):
        
        source_nodes_emb, target_nodes_emb = self.qa_encode_nodes((s, t))
        
        
        emb =  (
            source_nodes_emb ,
            questions_emb , 
            target_nodes_emb
        ) 

        return emb
    

                
        
    def qa_forward(self, s, t, questions_emb):
        return self.qa_interaction(*self.qa_encode(s, t, questions_emb))
    
    def qa_forward_triples(self, triples, questions_emb):
        s, _, t = triples.T
        return self.qa_interaction(*self.qa_encode(s, t, questions_emb))
    
    def training_step(self, batch, batch_idx) :
        qa_loss = self.qa_training_step(batch['qa'], batch_idx)
        kge_loss = self.kge_training_step(batch['kge'], batch_idx)
        return qa_loss #+ kge_loss
    
    def validation_step(self, batch, batch_idx) :
        self.qa_validation_step(batch['qa'], batch_idx)
        # self.kge_validation_step(batch['kge'], batch_idx)
        
    def qa_training_step(self, batch, batch_idx):
        triples, q_toks= batch
        
        
        questions_emb = self.qa_encode_question(q_toks)
        corr_triples = self.qa_neg_sampler.corrupt_batch(positive_batch=triples)

        
        pos_scores = self.qa_forward_triples(triples, 
                                  questions_emb)
        neg_scores = self.qa_forward_triples(corr_triples, 
                                  questions_emb)
        
        loss = self.loss_func.forward(pos_scores, neg_scores)

        self.log_dict({'train/qa_loss': loss.item()})
        
        return loss
    
    def forward(self, x, edge_index, src_idx, question, **kwargs ):
       
        

        # s,d,r = edge_index.coo()
        triples = torch.stack((edge_index[0],kwargs['relations'] ,edge_index[1])).T
        # print(x.shape, edge_index.shape, question.shape, triples.shape )
        qa_emb = self.qa_encode_question(question.unsqueeze(-1))
        nodes_emb = self.encode_nodes(x, triples)

        scores = self.qa_interaction(nodes_emb[src_idx].unsqueeze(0), qa_emb, nodes_emb)
        return scores.unsqueeze(-1)
    
    def qa_validation_step(self, batches, batch_idx):
        res = []
        for batch_type, batch in batches.items():
            src, true_targets_list, q_toks =  batch
            
            questions_emb = self.qa_encode_question(q_toks)
            candidate_targets = torch.arange(self.nodes_emb.num_embeddings,
                                    dtype=torch.long,
                                    device=self.device).unsqueeze(-1)
            
            batch_indices = torch.arange(
                        len(src),
                        dtype=torch.long,
                        device=self.device)

            scores = self.qa_forward(src, candidate_targets, questions_emb)
            scores[torch.stack( (src, batch_indices  ), 1).T.tolist() ] = -1
            pred_targets = scores.topk(100, dim=0)

            res.append(pred_targets)

            for true_targets, pred_target in zip(true_targets_list.T, pred_targets.indices.T):
                for k in [1,3,10,100]:
                    # print(type(pred_targets))
                    # print(type(true_targets))
                    # print(type(pred_targets.indices))
                    hits = torch.isin(pred_target[:k], true_targets).sum().item()
                    count = min(k, (true_targets > -1).sum().item())
                    
                    
                    perc_hits = hits / count
                    self.log(f'{batch_type}/hits@{k}', perc_hits, on_epoch=True)
                    
        return res
        
    def kge_encode(self, s,r,t ):

        emb = (
            self.nodes_emb(s), 
            self.relations_emb(r), 
            self.nodes_emb(t), 
        ) 
        

        return emb
        
    def kge_forward(self, s, r, t):
        return self.kge_interaction(*self.kge_encode(s, r, t))
        
    def kge_training_step(self, batch, batch_idx):
        
        corr_batch = self.kge_neg_sampler.corrupt_batch(positive_batch=batch)      
        

        pos_scores = self.kge_forward(*batch.T)
        neg_scores = self.kge_forward(*corr_batch.T)

        loss = self.loss_func.forward(pos_scores, neg_scores)
        
        self.log_dict({'train/kge_loss': loss.item()})
        return loss

    def configure_optimizers(self):
        to_train = [
            'nodes_emb.weight',
            
            'rgcn1.weight',
            'rgcn1.root',
            'rgcn1.bias',
            
            'rgcn2.weight',
            'rgcn2.root',
            'rgcn2.bias',
            
            'text_linear.weight',
            'text_linear.bias']
        
        params = []
        for n,v in self.named_parameters():
            
            if n in to_train or n.startswith('text') or n.startswith('qa_interaction'):
                params.append(v)
            else:
                v.requires_grad = False
        
        optim = torch.optim.Adam(params, lr=self.lr)
        return optim

#%%  
import click
from pytorch_lightning.trainer.supporters import CombinedLoader

@click.command()
@click.option('--emb-size', default=256, type=int)
@click.option('--hidden-size', default='128', type=str)
@click.option('--negs', default=1, type=int)
@click.option('--lr', default=0.0001, type=float)
@click.option('--ntm', default=False, is_flag=True)
@click.option('--kge-train-batch-size', default=1024, type=int)
@click.option('--qa-train-batch-size', default=128, type=int)
@click.option('--val-batch-size', default=128, type=int)
@click.option('--accumulate-train-batches', default=1, type=int)
@click.option('--limit-val-batches', default=100, type=int)
@click.option('--hops', default=1, type=int)
@click.option('--epochs', default=500, type=int)
def train( emb_size, hidden_size, negs, lr, ntm, kge_train_batch_size, qa_train_batch_size, val_batch_size, accumulate_train_batches, limit_val_batches, hops, epochs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    kge_data = EmbeddingsData('dataset', train_batch_size= kge_train_batch_size, val_batch_size= val_batch_size)
    qa_data = QAData('dataset', [hops], tokenizer, train_batch_size= qa_train_batch_size, val_batch_size= val_batch_size, use_ntm= ntm)

    hidden_size = [int(i) for i in hidden_size.split('|')]
    model = JointQAModel(kge_data, emb_size, hidden_size, lr, negs)

    wandb.init( entity='link-prediction-gnn', project="metaqa-qa", reinit=True)
    logger = WandbLogger(log_model=True)
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/qa/{hops}-hops/',
        filename=f'{model.get_name()}'+'|{epoch}'  
        )  
    
    trainer = pl.Trainer( 
        accelerator= 'gpu',
        gpus= -1,
        callbacks= [embeddings_checkpoint_callback],
        logger= logger, 
        log_every_n_steps= 1,
        limit_val_batches= limit_val_batches,
        accumulate_grad_batches= accumulate_train_batches,
        val_check_interval= 1.0, 
        check_val_every_n_epoch= 10,
        max_epochs= epochs)
    
    train_loader = CombinedLoader(
        {
            'qa': qa_data.train_dataloader(),
            'kge': kge_data.train_dataloader()
        }
    )
    val_loader = CombinedLoader(
        {
            'qa': qa_data.val_dataloader(),
            'kge': kge_data.val_dataloader()
        }
    )
    
    trainer.fit(model, 
                train_dataloaders= train_loader, 
                val_dataloaders= val_loader)
    wandb.finish()
    
if __name__ == '__main__':
    train()