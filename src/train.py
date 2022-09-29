#%%
from dataloaders import QAData
from models.rgcnqa import RGCNQA, AutoTokenizer
import wandb
import pytorch_lightning as pl
import click

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.trainer.supporters import CombinedLoader


from globals import QA_CHECKPOINTS_PATH, METAQA_PATH
from utils import get_model_path

#%%  

@click.command()
@click.option('--layer-sizes', default='18|18|18', type=str)
@click.option('--concat-layers', is_flag=True)
@click.option('--concat-embeddings', type=click.Choice(['all', 'first', 'last']))
@click.option('--decompose', default=False, is_flag=True)
@click.option('--lr', default=0.01, type=float)
@click.option('--l2', default=0.1, type=float)
@click.option('--nobias', 'bias', default=True, is_flag=True)
@click.option('--noroot', 'root', default=True, is_flag=True)
@click.option('--nozeroed', 'zeroed', default=True, is_flag=True)
@click.option('--ntm', default=False, is_flag=True)
@click.option('--train-batch-size', default=64, type=int)
@click.option('--val-batch-size', default=128, type=int)
@click.option('--accumulate-batches', default=1, type=int)
@click.option('--limit-val-batches', default=10, type=int)
@click.option('--limit-train-batches', default=256, type=int)
@click.option('--hops', default=2, type=int)
@click.option('--aggr', default='mean', type=str)
@click.option('--fast', is_flag=True)
@click.option('--bert-model', default='prajjwal1/bert-mini', type=str)
@click.option('--patience', default=3, type=int)
@click.option('--epochs', default=10000, type=int)
@click.option('--resume-id', type=str)
def train( layer_sizes, 
            concat_layers,
            concat_embeddings,
            decompose,
            lr,
            l2,
            bias,
            root,
            zeroed,
            ntm,
            train_batch_size,
            val_batch_size,
            accumulate_batches,
            limit_val_batches,
            limit_train_batches,
            hops,
            aggr,
            fast,
            bert_model,
            patience,
            epochs,
            resume_id):
    layer_sizes = [int(i) for i in layer_sizes.split('|') ] if len (layer_sizes) > 0 else []

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    
    qa_data = QAData(METAQA_PATH, [hops], tokenizer, train_batch_size= train_batch_size, val_batch_size= val_batch_size, use_ntm= ntm)
    model = None
    if resume_id:
        run = wandb.init( id=resume_id, entity='link-prediction-gnn', project="metaqa-qa",  resume='must')
        model = RGCNQA.load_from_checkpoint(get_model_path(run.name))
    else:
        model = RGCNQA(qa_data, layer_sizes, decompose, lr, l2, bias, root, zeroed, aggr, bert_model, fast, concat_layers, concat_embeddings)
        run = wandb.init( name=model.get_name(), entity='link-prediction-gnn', project="metaqa-qa", reinit=True)
    
    logger = WandbLogger( log_model=True)
                
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{QA_CHECKPOINTS_PATH}/{hops}-hops/',
        filename=model.get_name()+'|{epoch}'  
        )  
    stopping_callback = EarlyStopping(monitor="val/hits_at_1", min_delta=0.00, patience=patience, verbose=False, mode="max")
    
    # additional_arg[] = {
    #     'ckpt_path': get_model_path(run.name) 
    # } if resume_id else {}
    
    trainer = pl.Trainer( 
        accelerator= 'gpu',
        gpus= -1,
        callbacks= [checkpoint_callback, stopping_callback],
        logger= logger, 
        log_every_n_steps= 1,
        limit_val_batches= limit_val_batches,
        limit_train_batches= limit_train_batches,
        accumulate_grad_batches= accumulate_batches,
        val_check_interval= 1.0, 
        check_val_every_n_epoch= 1,
        max_epochs= epochs,
        )
    
    train_loader = CombinedLoader(
        {
            'qa': qa_data.train_dataloader(),
        }, 'max_size_cycle'
    )
    val_loader = CombinedLoader(
        {
            'qa': qa_data.val_dataloader(),
        }, 'max_size_cycle'
    )
    
    
    trainer.fit(model, 
                train_dataloaders= train_loader, 
                val_dataloaders= val_loader,
                )
    wandb.finish()
    
if __name__ == '__main__':
    train()
# %%
