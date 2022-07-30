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
                 aggr = 'max',
                 bert_model = "prajjwal1/bert-mini",
                 fast = False) -> None:
        super().__init__()
        

        self.lr = lr
        self.num_negs_per_pos = num_negs_per_pos
        
        self.qa_interaction = DistMultInteraction()
        self.kge_interaction = TransEInteraction(1)
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')

        # Text Model
        self.text_model = BertModel.from_pretrained(bert_model)
        
        if self.text_model.config.hidden_size != hidden_sizes[-1]:
            self.text_linear = torch.nn.Linear(self.text_model.config.hidden_size, hidden_sizes[-1])
        else:
            self.text_linear = torch.nn.Identity()
            

        # Rgcn
        self.layers = len(hidden_sizes)
        if fast:
            self.rgcn1 = torch_geometric.nn.conv.FastRGCNConv(emb_size, hidden_sizes[0], kg_data.n_relations*2, aggr=aggr)
        else:
            self.rgcn1 = torch_geometric.nn.conv.RGCNConv(emb_size, hidden_sizes[0], kg_data.n_relations*2, aggr=aggr)
        
        if self.layers > 1:
            if fast:
                self.rgcn2 = torch_geometric.nn.conv.FastRGCNConv(hidden_sizes[0], hidden_sizes[1], kg_data.n_relations*2, aggr=aggr)
            else:
                self.rgcn2 = torch_geometric.nn.conv.RGCNConv(hidden_sizes[0], hidden_sizes[1], kg_data.n_relations*2, aggr=aggr)
        if self.layers > 2:
            if fast:
                self.rgcn3 = torch_geometric.nn.conv.FastRGCNConv(hidden_sizes[1], hidden_sizes[2], kg_data.n_relations*2, aggr=aggr)
            else:
                self.rgcn3 = torch_geometric.nn.conv.RGCNConv(hidden_sizes[1], hidden_sizes[2], kg_data.n_relations*2, aggr=aggr)
            
        # Cache 
        self._z_cache = (None, None)
        self._q_cache = (None, None)
        
        # Edges and nodes
        self.edge_index = torch.nn.Parameter( kg_data.get_triples(), requires_grad = False)
        self.nodes_emb = Embedding(kg_data.n_nodes, emb_size, max_norm=1)
        self.relations_emb = Embedding(kg_data.n_relations, emb_size, max_norm=1)
        
        # Samplers
        self.qa_neg_sampler = BasicNegativeSampler(
                                mapped_triples= self.edge_index,
                                num_negs_per_pos= self.num_negs_per_pos,
                                corruption_scheme= ('head',))
        
        self.kge_neg_sampler = BasicNegativeSampler(
                                mapped_triples= self.edge_index,
                                num_negs_per_pos= self.num_negs_per_pos,
                                corruption_scheme= ('head','tail'))
        self.save_hyperparameters()
        
    
    def get_name(self):
        return f'{self.qa_interaction.__class__.__name__}|{self.nodes_emb.embedding_dim}'
        
    def encode_nodes(self, x, edge_index, cache_id = None):
        old_cache_id, old_z = self._z_cache
        if cache_id is not None and cache_id == old_cache_id:
            return old_z
        if x is None:
            x = self.nodes_emb(torch.arange(self.nodes_emb.num_embeddings, device=self.device))
        if edge_index is None:
            edge_index = self.edge_index
        
        z = self.rgcn1(x, edge_index[:,[0,2]].T, edge_index[:,1])
        
        if self.layers > 1 :
            z = self.rgcn2(z.relu(), edge_index[:,[0,2]].T, edge_index[:,1])
        if self.layers > 2 :
            z = self.rgcn3(z.relu(), edge_index[:,[0,2]].T, edge_index[:,1])
        
        self._z_cache = (cache_id, z)
        return z
    
    def encode_question(self, question_toks, cache_id = None):
        ''' Encodes the text tokens, input should have shape (seq_len, batch_size), return has shape (batch_size, embedding_size)'''
        old_cache_id, old_q = self._q_cache
        if cache_id is not None and cache_id == old_cache_id:
            return old_q
        

        q = self.text_model(question_toks.T).last_hidden_state[:,0,:]
        q = self.text_linear(q)
        
        self._q_cache = (cache_id, q)
        return q                
        
    def qa_forward(self, s, t, questions):
        z = self.encode_nodes(None, None)
        q = self.encode_question(questions)

        scores = self.qa_interaction(
            z[s], 
            q, 
            z[t]
        )
        
        return scores
    
    def qa_forward_triples(self, triples, questions):
        s, _, t = triples.T
        return self.qa_forward(s, t, questions)

    
    def training_step(self, batch, batch_idx) :
        qa_loss = self.qa_training_step(batch['qa'], batch_idx)
        kge_loss = self.kge_training_step(batch['kge'], batch_idx)
        return qa_loss #+ kge_loss
    
    def validation_step(self, batch, batch_idx) :
        self.qa_validation_step(batch['qa'], batch_idx)
        # self.kge_validation_step(batch['kge'], batch_idx)
        
    def qa_training_step(self, batch, batch_idx):
        triples, questions= batch
        
        
        # questions_emb = self.encode_question(questions)
        corr_triples = self.qa_neg_sampler.corrupt_batch(positive_batch=triples)

        
        pos_scores = self.qa_forward_triples(triples, 
                                  questions)
        neg_scores = self.qa_forward_triples(corr_triples, 
                                  questions)
        
        loss = self.loss_func.forward(pos_scores, neg_scores)

        self.log_dict({'train/qa_loss': loss.item()})
        
        return loss
    
    def forward(self, x, edge_index, src_idx, question, **kwargs ):
        triples = torch.stack((edge_index[0],kwargs['relations'] ,edge_index[1])).T

        qa_emb = self.encode_question(question.unsqueeze(-1))
        nodes_emb = self.encode_nodes(x, triples)

        scores = self.qa_interaction(nodes_emb[src_idx].unsqueeze(0), qa_emb, nodes_emb)
        return scores.unsqueeze(-1)
    
    def qa_validation_step(self, batches, batch_idx):
        res = []
        for batch_type, batch in batches.items():
            src, true_targets_list, questions =  batch
            
            # questions_emb = self.encode_question(q_toks)
            candidate_targets = torch.arange(self.nodes_emb.num_embeddings,
                                    dtype=torch.long,
                                    device=self.device).unsqueeze(-1)
            
            batch_indices = torch.arange(
                        len(src),
                        dtype=torch.long,
                        device=self.device)

            scores = self.qa_forward(src, candidate_targets, questions)
            scores[torch.stack( (src, batch_indices  ), 1).T.tolist() ] = -1
            pred_targets = scores.topk(100, dim=0)

            res.append(( scores, pred_targets))

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