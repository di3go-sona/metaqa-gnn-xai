#%%
from itertools import accumulate
from dataset.dataloaders import EmbeddingsData, QAData
from pykeen.nn.modules import DistMultInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss

import wandb, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.nn import Embedding
from transformers import AutoTokenizer, BertModel
from train_embeddings import KGEModel
import torch_geometric
#%%
class QAModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, 
                 emb_size, 
                 hidden_size,
                 lr,
                 num_negs_per_pos,
                 train_nodes,
                 nodes_emb= None, 
                 relations_emb= None) -> None:
        super().__init__()
        

        self.lr = lr
        self.num_negs_per_pos = num_negs_per_pos
        self.train_nodes = train_nodes
        
        self.interaction = DistMultInteraction()
        self.neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets), 
                                                num_negs_per_pos=self.num_negs_per_pos,
                                                corruption_scheme=('head',))
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        
        
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        
        if self.text_model.config.hidden_size != hidden_size:
            self.text_linear = torch.nn.Linear(self.text_model.config.hidden_size, hidden_size)
        else:
            self.text_linear = torch.nn.Identity()
            
        # if hidden_size != emb_size:
        #     self.emb_linear = torch.nn.Linear(emb_size, hidden_size)
        # else:
        #     self.emb_linear = torch.nn.Identity()
            
        self.rgcn1 = torch_geometric.nn.conv.RGCNConv(emb_size, hidden_size, kg_data.n_relations*2, aggr='max')
        
        triples = kg_data.triplets.cuda()
        itriples = torch.index_select(triples, 0, torch.tensor([2,1,0], dtype=torch.long, device=triples.device))
        itriples[:, 1]  += kg_data.n_relations
        
        self.triples = torch.cat((triples, itriples),0) 
        
        
        if nodes_emb is None:
            self.nodes_emb = Embedding(kg_data.n_nodes, emb_size)
        else:
            self.nodes_emb = nodes_emb
            
        if relations_emb is None:
            self.relations_emb = Embedding(kg_data.n_relations, emb_size)
        else:
            self.relations_emb = relations_emb           
        
        self.save_hyperparameters()
    
    def get_name(self):
        return f'{self.interaction}|{self.nodes_emb.embedding_dim}'
    
    
    def encode_text(self, q_toks):
        ''' Encodes the text tokens, returns has shape (batch_size, embedding_size)'''
        # enc = self.text_model(q_toks.T).pooler_output
        enc = self.text_model(q_toks.T).last_hidden_state[:,0,:]
        enc_hat = self.text_linear(enc)
        return enc_hat
         
    
    def encode(self, s, t, questions_emb):
        
        nodes_emb = self.nodes_emb(torch.arange(self.nodes_emb.num_embeddings, device=self.device))
        nodes_emb = self.rgcn1(nodes_emb, self.triples[:,[0,2]].T, self.triples[:,1])
        # nodes_emb
        # nodes_emb = Embedding.from_pretrained(nodes_emb)
        # print(nodes_emb.shape, s.shape,t.shape, nodes_emb[s].shape)
        
        emb =  (
            nodes_emb[s] ,
            questions_emb , 
            nodes_emb[t]
        ) 
        # print(emb[0].shape, questions_emb.shape)
        

        return emb
        
    def forward(self, s, t, questions_emb):
        return self.interaction(*self.encode(s, t, questions_emb))
    
    def forward_triples(self, triples, questions_emb):
        s, _, t = triples.T
        return self.interaction(*self.encode(s, t, questions_emb))
        
    def training_step(self, batch, batch_idx):
        triples, q_toks= batch
        
        
        questions_emb = self.encode_text(q_toks)
        corr_triples = self.neg_sampler.corrupt_batch(positive_batch=triples)

        
        pos_scores = self.forward_triples(triples, 
                                  questions_emb)
        neg_scores = self.forward_triples(corr_triples, 
                                  questions_emb)
        
        loss = self.loss_func.forward(pos_scores, neg_scores)

        self.log_dict({'train/loss': loss.item()})
        
        return loss
    
    
    def validation_step(self, batches, batch_idx):
        for batch_type, batch in batches.items():
            src, true_targets_list, q_toks =  batch
            
            questions_emb = self.encode_text(q_toks)
            candidate_targets = torch.arange(self.nodes_emb.num_embeddings,
                                    dtype=torch.long,
                                    device=self.device).unsqueeze(-1)
            
            batch_indices = torch.arange(
                        len(src),
                        dtype=torch.long,
                        device=self.device)

            scores = self.forward(src, candidate_targets, questions_emb)
            scores[torch.stack( (src, batch_indices  ), 1).T.tolist() ] = -1
            pred_targets = scores.topk(100, dim=0).indices



            for true_targets, pred_targets in zip(true_targets_list.T, pred_targets.T):
                for k in [1,3,10,100]:
                    hits = torch.isin(pred_targets[:k], true_targets).sum().item()
                    count = min(k, (true_targets > -1).sum().item())
                    
                    
                    perc_hits = hits / count
                    self.log(f'{batch_type}/hits@{k}', perc_hits, on_epoch=True)
        

    
    def configure_optimizers(self):
        to_train = [
                     
            'rgcn1.weight',
            'rgcn1.root',
            'rgcn1.bias',
            
            'text_linear.weight',
            'text_linear.bias']
        
        if self.train_nodes:
            to_train.append('nodes_emb.weight')
        
        params = []
        for n,v in self.named_parameters():
            
            if n in to_train or n.startswith('text_model'):
                params.append(v)
            else:
                v.requires_grad = False
        
        optim = torch.optim.Adam(params, lr=self.lr)
        return optim
      
import click
@click.command()
@click.option('--hidden-size', default=128, type=int)
@click.option('--negs', default=1, type=int)
@click.option('--lr', default=0.0001, type=float)
@click.option('--train-nodes', default=False, is_flag=True)
@click.option('--ntm', default=False, is_flag=True)
@click.option('--train-batch-size', default=1024, type=int)
@click.option('--train-batches', default=4, type=int)
@click.option('--val-batch-size', default=128, type=int)
@click.option('--limit-val-batches', default=100, type=int)
@click.option('--hops', default=1, type=int)
@click.option('--epochs', default=50, type=int)
def train(hidden_size, negs, lr, train_nodes, train_batches, ntm, train_batch_size, val_batch_size, limit_val_batches, hops, epochs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = QAData('dataset', [hops], tokenizer, train_batch_size=train_batch_size, val_batch_size=val_batch_size, use_ntm=ntm)
    kge_model = KGEModel.load_from_checkpoint('checkpoints/embeddings/DistMultInteraction()|768|epoch=49|.ckpt')
    model = QAModel(data, 
                    kge_model.nodes_emb.embedding_dim,
                    hidden_size,
                    lr,
                    negs,
                    train_nodes,
                    nodes_emb=kge_model.nodes_emb,  
                    relations_emb=kge_model.relations_emb)

    wandb.init( entity='link-prediction-gnn', project="metaqa-qa", reinit=True)
    logger = WandbLogger(log_model=True)
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/qa/{hops}-hops/',
        filename=f'{model.get_name()}'+'|{epoch}'  
        )  
    
    trainer = pl.Trainer( 
        accelerator='gpu',
        gpus=-1,
        callbacks=[embeddings_checkpoint_callback],
        logger= logger, 
        log_every_n_steps=10,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=train_batches,
        val_check_interval=0.1,
        max_epochs=epochs)
    
    trainer.fit(model, data)
    wandb.finish()



if __name__ == '__main__':
    train()
# %%
