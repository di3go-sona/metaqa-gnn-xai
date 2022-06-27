#%%
from dataset.dataloaders import EmbeddingsData
from pykeen.nn.modules import DistMultInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss


import wandb, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.nn import Embedding
import gc

class KGEModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, emb_size=200) -> None:
        super().__init__()
        
        self.interaction = DistMultInteraction()
        self.neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets))
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        self.nodes_emb = Embedding(kg_data.n_edges, emb_size)
        self.relations_emb = Embedding(kg_data.n_relations, emb_size)
        

        
        self.save_hyperparameters()
    
    def get_name(self):
        return 'ComplEx'
        
    def encode(self, triples):
        s,r,t = triples.T
        
        return (
            self.nodes_emb(s),
            self.nodes_emb(r),
            self.nodes_emb(t)
        ) 
        
    def forward(self, triples):
        return self.interaction(*self.encode(triples))
        
    def training_step(self, batch, batch_idx):
        
        corr_batch = self.neg_sampler.corrupt_batch(positive_batch=batch).squeeze(1)
        
        pos_scores = self.forward(batch)
        neg_scores = self.forward(corr_batch)

        loss = self.loss_func.forward(pos_scores, neg_scores)
        
        


        
        self.log_dict({'train/loss': loss.item()})
        return loss
    

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        return optim
      

# %%  
if __name__ == '__main__':

    data = EmbeddingsData(train_batch_size=2048)
    model = KGEModel(data)
    
    wandb.init( entity='link-prediction-gnn', project="metaqa-embeddings", reinit=True)
    logger = WandbLogger(log_model=True)
    # profiler = 
    
                
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
        max_epochs=100)
    
    trainer.fit(model, data)
    wandb.finish()


# %%
