#%%
from dataset.dataloaders import EmbeddingsData
from pykeen.nn.modules import DistMultInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss, BCEWithLogitsLoss


import wandb, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.nn import Embedding


class KGEModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, emb_size, lr, negs) -> None:
        super().__init__()
        
        self.lr = lr
        self.negs = negs
        self.interaction = DistMultInteraction()
        self.neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets), num_negs_per_pos=negs)
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        self.nodes_emb = Embedding(kg_data.n_nodes, emb_size, max_norm=1)
        self.relations_emb = Embedding(kg_data.n_relations, emb_size)
        

        
        self.save_hyperparameters()
    
    def get_name(self):
        return 'ComplEx'
        
    def encode(self, s,r,t ):

        emb = (
            self.nodes_emb(s), 
            self.relations_emb(r), 
            self.nodes_emb(t), 
        ) 
        

        return emb
        
    def forward(self, s, r, t):
        return self.interaction(*self.encode(s, r, t))
        
    def training_step(self, batch, batch_idx):
        

        
        corr_batch = self.neg_sampler.corrupt_batch(positive_batch=batch)      
        

        pos_scores = self.forward(*batch.T)
        neg_scores = self.forward(*corr_batch.T)

        loss = self.loss_func.forward(pos_scores, neg_scores)
        
        self.log_dict({'train/loss': loss.item()})
        return loss
             
    
    def validation_step(self, batch, batch_idx):
        src, rel, true_targets_list =  batch
        
        candidate_targets =  torch.arange(
                        self.nodes_emb.num_embeddings,
                        dtype=torch.long,
                        device=self.device).unsqueeze(-1)
        
        scores = self(src, rel, candidate_targets)
        top_targets = scores.topk(101, dim=0).indices[1:]
        
        for true_targets, pred_targets in zip(true_targets_list.T, top_targets.T):
            for k in [1,3,10,100]:
                hits = torch.isin(pred_targets[:k], true_targets).sum().item()
                count = min(k, ( (true_targets > -1).sum() > -1).sum().item())
                
                perc_hits = hits / count
                self.log(f'val/hits@{k}', perc_hits, on_epoch=True)

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim
      

# %%  
import click
@click.command()
@click.option('--embeddings', default=100, type=int)
@click.option('--lr', default=0.0001, type=float)
@click.option('--negs', default=1, type=int)
@click.option('--train-batch-size', default=4096, type=int)
def train(embeddings, negs, lr, train_batch_size):
    data = EmbeddingsData('dataset', train_batch_size=train_batch_size)
    model = KGEModel(data, embeddings, lr, negs)

    wandb.init( entity='link-prediction-gnn', project="metaqa-embeddings", reinit=True)
    logger = WandbLogger(log_model=True)    
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/embeddings/',
        filename=f'{model.get_name()}'+'|{epoch}|{mrr}'  
        )  
    
    trainer = pl.Trainer( 
        accelerator= 'gpu',
        gpus= -1,
        callbacks= [embeddings_checkpoint_callback],
        logger= logger, 
        log_every_n_steps= 10,
        check_val_every_n_epoch= 10,
        limit_val_batches=100,
        max_epochs=100)
    
    trainer.fit(model, data)
    wandb.finish()


# %%
if __name__ == '__main__':
    train()