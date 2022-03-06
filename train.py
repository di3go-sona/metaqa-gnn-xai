
import wandb, os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import *
from models import *



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

configs = [{
    "learning_rate": 0.0001,
    "reg": float(os.environ.get('REG', '1.0')),
    "epochs": int(os.environ.get('EPOCHS', '16')),
    "drop_embeddings": float(os.environ.get('EPOCHS', '0.5')),
    "drop_rgcn": float(os.environ.get('EPOCHS', '0.3')),
    "embeddings_size": int(os.environ.get('SIZE', f'256')),
    "batch_size": int(os.environ.get('BATCH_SIZE', '1024')),  # int or -1 for 'full' 
    "n_layers": int(os.environ.get('LAYERS', '0')),
    'device': 'cuda'
        } ]

    

if __name__ == '__main__':

    for config in configs:
    
        #initialize wandb logger
        wandb.init( project="metaqa",  entity="link-prediction-gnn", config=config, reinit=True)
        wandb_logger = None or WandbLogger(log_model=True)

        # Initialize data and model
        data = MetaQa.build_or_get(wandb.config['batch_size'])

        model = AnswerPredictor( data.kb_n_nodes, data.kb_n_relations, len(data.qa_questions_vocab),dict(wandb.config))    
        
        # checkpoint_callback = ModelCheckpoint(
        #     dirpath=wandb.run.dir,
        #     filename='{epoch}' \
        #         + '_reg=' + str(wandb.config['reg']) \
        #         + '_layers=' + str(wandb.config['n_layers']) \
        #         + '_size=' + str(wandb.config['embeddings_size']) \
        #         + '_time=' + str(datetime.now())
        #     )  
        
        # Initialize Trainer 
        trainer = pl.Trainer( ** {'gpus':-1, 'auto_select_gpus': True } if wandb.config['device'] == 'cuda' else {},
                            logger= wandb_logger, 
                            log_every_n_steps=1,
                            max_epochs=wandb.config['epochs'])
        # Train
        trainer.fit(model, data)


