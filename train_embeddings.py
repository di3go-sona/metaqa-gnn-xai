#%%
import wandb, os
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from dataset.dataloaders import EmbeddingsData
from datasets import *
from models import *
from models.EmbeddingsModel import EmbeddingsModel


config = {
    "embeddings_size": 128,
    "learning_rate":   0.001,
    "epochs":          50000,
    "regularization":  0.0,
    "batch_size":      512,
    "val_batch_size":  8,
    "decoder":          'DistmultDecoder', #
    'device': 'cuda'
}
    

if __name__ == '__main__':
    
    data  = EmbeddingsData(batch_size=config['batch_size'], val_batch_size=config['val_batch_size'])
    model = EmbeddingsModel(data.ds.n_nodes, data.ds.n_relations, config)
        
    # Initialize data and model for pre-training
    
    wandb.init( entity='link-prediction-gnn', project="metaqa-embeddings", reinit=True)
    wandb_logger = WandbLogger(log_model=True)
    
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/embeddings/',
        filename=f'{model.get_name()}'+'|{epoch}|{mrr}'  
        )  
    
    trainer = pytorch_lightning.Trainer( 
        ** {'gpus':-1} if config['device'] == 'cuda' else {},
        callbacks=[embeddings_checkpoint_callback],
        logger= wandb_logger, 
        log_every_n_steps=1,
        check_val_every_n_epoch=100,
        # limit_val_batches=1024,
        max_epochs=config['epochs'])
    
    trainer.fit(model, data)
    wandb.finish()


# %%
