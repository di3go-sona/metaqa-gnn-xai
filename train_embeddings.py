

from datetime import datetime
from gc import callbacks
import sys
from torch import embedding
import wandb, os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import *
from models import *



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = {
    "embeddings_size": int(os.environ.get("embeddings_size", 256)),
    "learning_rate":   float(os.environ.get("learning_rate", 0.0001)),
    "epochs":          int(os.environ.get("epochs", 300)),
    "dropout":         float(os.environ.get("dropout", 0.3)),
    "regularization":  float(os.environ.get("regularization", 0.1)),
    "batch_size":      int(os.environ.get("batch_size", 1024)),
    "val_batch_size":  int(os.environ.get("val_batch_size", 8)),
    "decoder":         str(os.environ.get("decoder", 'DistMultDecoder')),
    'device': 'cuda'
}



# configs = [{
#     "learning_rate": 0.0001,
#     "epochs": int(os.environ.get('EPOCHS', '300')),
#     "batch_size": int(os.environ.get('BATCH_SIZE', '1024')),
#     "val_batch_size": int(os.environ.get('BATCH_SIZE', '8')),
#     "dropout": float(os.environ.get('EPOCHS', '0.3')),
#     "learning_rate": 0.0001,
#     "epochs": int(os.environ.get('EPOCHS', '300')),
#     "reg": float(os.environ.get('REG', '1.0')),
#     "embeddings_size": int(os.environ.get('SIZE', f'256')),
#     "batch_size": int(os.environ.get('BATCH_SIZE', '1024')),  # int or -1 for 'full' 
#     "drop_rgcn": float(os.environ.get('EPOCHS', '0.3')),
#     "n_layers": int(os.environ.get('LAYERS', '1')),
#     'device': 'cuda'
#         } ]

    

if __name__ == '__main__':
    
    data  = MetaQaEmbeddings(config['batch_size'], config['val_batch_size'])
    model = EntitiesEmbeddings(data.ds.n_embeddings, data.ds.n_relations, config)


        
    # Initialize data and model for pre-training
    
    wandb.init( project="metaqa",  entity="link-prediction-gnn", config=config, reinit=True)
    wandb_logger = WandbLogger(log_model=True)
    
                
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/embeddings/',
        filename=f'{model.cname()}'+'|{epoch}|{hit@10}'  
        )  
    
    trainer = pl.Trainer( 
        ** {'gpus':-1, 'auto_select_gpus': True } if config['device'] == 'cuda' else {},
        callbacks=[embeddings_checkpoint_callback],
        logger= wandb_logger, 
        log_every_n_steps=1,
        check_val_every_n_epoch=5,
        limit_val_batches=256,
        max_epochs=config['epochs'])
    
    trainer.fit(model, data)
    wandb.finish()
    
        # if   '--load-embeddings' in sys.argv:
        #     embeddings_model = EntitiesEmbeddings.load_from_checkpoint( './checkpoints/embeddings/EntitiesEmbeddings-DistMultDecoder-n_embeddings=40152-epoch=299-hit@10=0.1171875.ckpt')
        # elif '--new-embeddings'  in sys.argv:
        #     embeddings_model = EntitiesEmbeddings(embeddings_data.ds.n_embeddings, config['embeddings_size'], embeddings_data.ds.n_relations, config['learning_rate'], config['dropout'])
        # else:
        #     embeddings_model = None
        
        # if '--train-embeddings' in sys.argv:
            
            
        #     wandb.init( project="metaqa",  entity="link-prediction-gnn", config=config, reinit=True)
        #     wandb_logger = WandbLogger(log_model=True)
            
        #     embeddings_checkpoint_callback = ModelCheckpoint(
        #         dirpath='checkpoints/embeddings/',
        #         filename=f'{embeddings_model.__class__.__name__}-{embeddings_model.decoder.__class__.__name__}-n_embeddings={embeddings_model.n_embeddings}'+'-{epoch}-{hit@10}'  
        #         )  
            
        #     trainer = pl.Trainer( 
        #         ** {'gpus':-1, 'auto_select_gpus': True } if config['device'] == 'cuda' else {},
        #         callbacks=[embeddings_checkpoint_callback],
        #         logger= wandb_logger, 
        #         log_every_n_steps=1,
        #         check_val_every_n_epoch=5,
        #         limit_val_batches=256,
        #         max_epochs=config['epochs'])
            
        #     trainer.fit(embeddings_model, embeddings_data)
        #     wandb.finish()


            
        # if '--load-model' in sys.argv:
        #     train_model = AnswerPredictor.load_from_checkpoint( 'checkpoints/qa/EntitiesEmbeddings-TransEDecoder-n_embeddings=0-epoch=9-hit@10=0.01708984375.ckpt',)
        # else:
        #     train_model = AnswerPredictor( train_data.kb_n_nodes, train_data.kb_n_relations, len(train_data.qa_questions_vocab),dict(config), embeddings = embeddings_model)
            

        # if '--train-model' in sys.argv:
        #     wandb.init( project="metaqa",  entity="link-prediction-gnn", config=config, reinit=True)
        #     wandb_logger = WandbLogger(log_model=True)
        
        #     embeddings_checkpoint_callback = ModelCheckpoint(
        #         dirpath='checkpoints/qa/',
        #         filename=f'{train_model.__class__.__name__}'+'-{epoch}-{val_acc}'  
        #         )  
            
        #     trainer = pl.Trainer( ** {'gpus':-1, 'auto_select_gpus': True } if config['device'] == 'cuda' else {},
        #                 logger= wandb_logger, 
        #                 log_every_n_steps=1,
        #                 max_epochs=config['epochs'])
        #     # Train
        #     trainer.fit(train_model, train_data)    
        #     wandb.finish()




