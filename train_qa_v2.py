#%%
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
from train_embeddings_v2 import KGEModel

#%%
class QAModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, 
                 emb_size, 
                 lr,
                 num_negs_per_pos,
                 nodes_emb= None, 
                 relations_emb= None) -> None:
        super().__init__()
        

        self.lr = lr
        self.num_negs_per_pos = num_negs_per_pos
        
        self.interaction = DistMultInteraction()
        self.neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets), num_negs_per_pos=self.num_negs_per_pos)
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        
        
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_linear = torch.nn.Linear(self.text_model.config.hidden_size, emb_size)
        
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
        return 'ComplEx'
    
    
    def encode_text(self, q_toks):
        ''' Encodes the text tokens, returns has shape (batch_size, embedding_size)'''
        enc = self.text_model(q_toks.T).pooler_output
        enc_hat = self.text_linear(enc)
        return enc_hat
         
    def encode(self, triples, questions_emb):
        s, _, t = triples.T
        
        e =  (
            self.nodes_emb( s ),
            questions_emb,
            self.nodes_emb( t )
        ) 

        return e
        
    def forward(self, triples, questions_emb):
        return self.interaction(*self.encode(triples, questions_emb))
        
    def training_step(self, batch, batch_idx):
        triples, q_toks= batch
        
        
        questions_emb = self.encode_text(q_toks)
        corr_triples = self.neg_sampler.corrupt_batch(positive_batch=triples)
        
        # print(corr_triples.shape , questions_emb.shape, questions_emb.repeat(self.num_negs_per_pos,1).shape)
        pos_scores = self.forward(triples, questions_emb)
        neg_scores = self.forward(corr_triples.reshape(-1, 3), 
                                  questions_emb.repeat(self.num_negs_per_pos,1))


        loss = self.loss_func.forward(pos_scores, neg_scores.reshape(self.num_negs_per_pos, -1))
        # print(loss.shape)     
        self.log_dict({'train/loss': loss.item()})
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, dsts, q_toks =  batch
        
        questions_emb = self.encode_text(q_toks)
        
        for s, d, q in zip(src, dsts, questions_emb):
            triples = torch.stack((
                s.repeat(self.nodes_emb.num_embeddings),
                torch.zeros(self.nodes_emb.num_embeddings,
                        dtype=torch.long,
                        device=s.device),
                torch.arange(self.nodes_emb.num_embeddings,
                        dtype=torch.long,
                        device=s.device)
            )).T
            
            q_emb = q.repeat((self.nodes_emb.num_embeddings, 1))
            
            scores = self(triples, q_emb)

            top_ans = scores.topk(10).indices
            hits = torch.isin(d, top_ans).sum() / min(10, (d > -1).sum())
            mrr = 1/top_ans

            self.log_dict({'val/hits@10': hits}, on_epoch=True)
            self.log_dict({'val/mrr': mrr}, on_epoch=True)
            
            top_ans = (-scores).topk(10).indices
            hits = torch.isin(d, top_ans).sum() / min(10, (d > -1).sum())
            mrr = 1/top_ans

            self.log_dict({'val/inv_hits@10': hits}, on_epoch=True)
            self.log_dict({'val/inv_mrr': mrr}, on_epoch=True)
            
            
            
             
        
        

    
    def configure_optimizers(self):
        to_train = ['text_model.encoder.layer.10.attention.self.query.weight',
            'text_model.encoder.layer.10.attention.self.query.bias',
            'text_model.encoder.layer.10.attention.self.key.weight',
            'text_model.encoder.layer.10.attention.self.key.bias',
            'text_model.encoder.layer.10.attention.self.value.weight',
            'text_model.encoder.layer.10.attention.self.value.bias',
            'text_model.encoder.layer.10.attention.output.dense.weight',
            'text_model.encoder.layer.10.attention.output.dense.bias',
            'text_model.encoder.layer.10.attention.output.LayerNorm.weight',
            'text_model.encoder.layer.10.attention.output.LayerNorm.bias',
            'text_model.encoder.layer.10.intermediate.dense.weight',
            'text_model.encoder.layer.10.intermediate.dense.bias',
            'text_model.encoder.layer.10.output.dense.weight',
            'text_model.encoder.layer.10.output.dense.bias',
            'text_model.encoder.layer.10.output.LayerNorm.weight',
            'text_model.encoder.layer.10.output.LayerNorm.bias',
            'text_model.encoder.layer.11.attention.self.query.weight',
            'text_model.encoder.layer.11.attention.self.query.bias',
            'text_model.encoder.layer.11.attention.self.key.weight',
            'text_model.encoder.layer.11.attention.self.key.bias',
            'text_model.encoder.layer.11.attention.self.value.weight',
            'text_model.encoder.layer.11.attention.self.value.bias',
            'text_model.encoder.layer.11.attention.output.dense.weight',
            'text_model.encoder.layer.11.attention.output.dense.bias',
            'text_model.encoder.layer.11.attention.output.LayerNorm.weight',
            'text_model.encoder.layer.11.attention.output.LayerNorm.bias',
            'text_model.encoder.layer.11.intermediate.dense.weight',
            'text_model.encoder.layer.11.intermediate.dense.bias',
            'text_model.encoder.layer.11.output.dense.weight',
            'text_model.encoder.layer.11.output.dense.bias',
            'text_model.encoder.layer.11.output.LayerNorm.weight',
            'text_model.encoder.layer.11.output.LayerNorm.bias',
            'text_model.pooler.dense.weight',
            'text_model.pooler.dense.bias',
            'text_linear.weight',
            'text_linear.bias']
        
        params = []
        for n,v in self.named_parameters():
            
            if n in to_train:
                params.append(v)
            else:
                v.requires_grad = False
        
        optim = torch.optim.Adam(params, lr=self.lr)
        return optim
      
import click
@click.command()
@click.option('--embeddings', default=200, type=int)
@click.option('--lr', default=0.0001, type=float)
@click.option('--negs', default=1, type=int)
@click.option('--ntm', default=False, is_flag=True)
def train(embeddings, negs, lr, ntm):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = QAData('dataset', [1], tokenizer, train_batch_size=128, val_batch_size=16, use_ntm=ntm)
    kge_model = KGEModel.load_from_checkpoint('checkpoints/embeddings/ComplEx|epoch=99|mrr=0-v1.ckpt')
    model = QAModel(data, 
                    embeddings,
                    lr,
                    negs,
                    nodes_emb=kge_model.nodes_emb,  
                    relations_emb=kge_model.relations_emb)
    
    wandb.init( entity='link-prediction-gnn', project="metaqa-qa", reinit=True)
    logger = WandbLogger(log_model=True)
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/embeddings/',
        filename=f'{model.get_name()}'+'|{epoch}|{mrr}'  
        )  
    
    trainer = pl.Trainer( 
        accelerator='gpu',
        gpus=-1,
        callbacks=[embeddings_checkpoint_callback],
        logger= logger, 
        log_every_n_steps=1,
        # limit_val_batches=100,
        val_check_interval=1.0,
        max_epochs=100)
    
    trainer.fit(model, data)
    wandb.finish()



if __name__ == '__main__':
    train()
# %%
