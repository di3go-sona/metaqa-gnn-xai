#%%
from itertools import accumulate
import wandb, os, glob, re
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.QaModel import QaModel, load_graph_encoder
from models.QuestionEncoder import QuestionEncoder
from models.EmbeddingsModel import EmbeddingsModel

from dataset.dataloaders import QuestionAnsweringData

if "CUDA_DEVICE_ORDER" not in os.environ: os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ: os.environ["CUDA_VISIBLE_DEVICES"] = "1"

configs = {
    "lr":              float(os.environ.get("lr", 0.0001)),
    "epochs":          int(os.environ.get("epochs", 15)),
    # "dropout":         float(os.environ.get("dropout", 0.2)),
    "val_batch_size":  int(os.environ.get("batch_size", 24)),
    "batch_size":      int(os.environ.get("batch_size", 24)),
    "n_layers":        int(os.environ.get("n_layers", 1)),
    "l2":              float(os.environ.get("reg", 0.01)),
    "hops":            os.environ.get("hops", [1]),
    'device':          'cuda'
}
    
    
data  = QuestionAnsweringData('dataset',
                              hops = configs['hops'],
                              batch_size = configs['batch_size'], 
                              val_batch_size = configs['val_batch_size'])
graph_index  = data.graph_index()
graph_encoder = load_graph_encoder()
question_encoder = QuestionEncoder(graph_encoder.embeddings_size)
model = QaModel(graph_encoder, question_encoder, graph_index, configs)

# b = next(iter(data.train_dataloader()))
# model.training_step(b,1)

logger = WandbLogger(log_model=True, entity='link-prediction-gnn', project='metaqa-qa')

trainer = pl.Trainer( 
        ** {'gpus':1, 'auto_select_gpus': True } if configs['device'] == 'cuda' else {},
        log_every_n_steps=1,
        val_check_interval=0.5,
        limit_val_batches=100,
        max_epochs=configs['epochs'], 
        logger=logger)



trainer.fit(model, data)

#%%
# if __name__ == '__main__':
    
#     data  = QA_Dataset('dataset')
#     model = QaModel(graph_encoder, question_encoder, configs)

        
#     # Initialize data and model for pre-training
    
#     wandb.init( project="metaqa",entity='link-prediction-gnn',reinit=True)
#     wandb_logger = WandbLogger(log_model=True)
    
#     # wandb_logger.watch(model, log="all", log_freq=100)
#     embeddings_checkpoint_callback = ModelCheckpoint(
#         dirpath='checkpoints/qa/',
#         filename=f'{model.cname()}'+'|{epoch}|{val_acc}'  
#         )  
    
#     trainer = pl.Trainer( 
#         ** {'gpus':1, 'auto_select_gpus': True } if config['device'] == 'cuda' else {},
#         callbacks=[embeddings_checkpoint_callback],
#         logger= wandb_logger, 
#         log_every_n_steps=1,
#         check_val_every_n_epoch=1,
#         max_epochs=config['epochs'])
    
#     trainer.fit(model, data)
#     wandb.finish()


# %%
