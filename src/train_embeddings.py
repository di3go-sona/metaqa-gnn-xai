#%%

from dataloaders import EmbeddingsData
from models.embeddings import KGEModel
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint





# %%  
import click
@click.command()
@click.option('--emb-size' , default=32, type=int)
@click.option('--interaction', type=str, default='transe')
@click.option('--lr', default=0.0001, type=float)
@click.option('--negs', default=32, type=int)
@click.option('--train-batch-size', default=4096, type=int)
@click.option('--val-batch-size', default=64, type=int)
@click.option('--limit-val-batches', default=100, type=int)
@click.option('--epochs', default=1000, type=int)
def train(emb_size, interaction, negs, lr, train_batch_size, val_batch_size, limit_val_batches, epochs):
    
    data = EmbeddingsData('../data/metaqa', train_batch_size= train_batch_size, val_batch_size= val_batch_size)
    model = KGEModel(data, interaction, emb_size, lr, negs)

    wandb.init( entity='link-prediction-gnn', project="metaqa-embeddings", reinit=True)
    logger = WandbLogger(log_model=True)    
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/embeddings/',
        filename=f'{model.get_name()}'+'|{epoch}|'  
        )  
    
    trainer = pl.Trainer( 
        accelerator= 'gpu',
        gpus= -1,
        callbacks= [embeddings_checkpoint_callback],
        logger= logger, 
        log_every_n_steps= 10,
        check_val_every_n_epoch= 100,
        limit_val_batches=limit_val_batches,
        max_epochs=epochs)
    
    trainer.fit(model, data)
    wandb.finish()


# %%
if __name__ == '__main__':
    train()
# %%
