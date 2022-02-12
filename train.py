

from datetime import datetime
import wandb, os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import *
from models import *



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = {
    "learning_rate": 0.01,
    "reg": float(os.environ.get('REG', '0')),
    "epochs": int(os.environ.get('EPOCHS', '500')),
    "embeddings_size": int(os.environ.get('SIZE', '256')),
    "batch_size": int(os.environ.get('BATCH_SIZE', '1024')),  # int or -1 for 'full' 
    "n_layers": int(os.environ.get('LAYERS', '2')),
    'limit_val_batches': 100,
    'limit_train_batches': 100,
    'val_check_interval': 1,
    'device': 'cuda'
        }

    

if __name__ == '__main__':




    
    #initialize wandb logger
    wandb.init( tags=[f'{k}={str(v)}' for k,v in config.items()], config=config, entity='link-prediction-gnn' )
    wandb_logger = None or WandbLogger(project="metaqa",  entity="link-prediction-gnn")

    # Initialize data and model
    data = MetaQa.build_or_get(-1,200)
    print(data)
    model = AnswerPredictor( data.kb_n_nodes, data.kb_n_relations, len(data.qa_questions_vocab),dict(wandb.config))    
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb.run.dir,
        filename='{epoch}' \
            + '_reg=' + str(wandb.config['reg']) \
            + '_layers=' + str(wandb.config['n_layers']) \
            + '_size=' + str(wandb.config['embeddings_size']) \
            + '_time=' + str(datetime.now())
        )  
    # Initialize Trainer 
    trainer = pl.Trainer( ** {'gpus':-1, 'auto_select_gpus': True } if wandb.config['device'] == 'cuda' else {},
                        logger= wandb_logger, 
                        # limit_val_batches = wandb.config['limit_val_batches'],
                        # limit_train_batches = wandb.config['limit_train_batches'],
                        # val_check_interval = wandb.config['val_check_interval'],
                        log_every_n_steps=1, 
                        max_epochs=wandb.config['epochs'],
                        callbacks=[checkpoint_callback],
                        accumulate_grad_batches=1)
    # Train
    trainer.fit(model, data)


