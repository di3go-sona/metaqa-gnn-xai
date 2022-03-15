import wandb, os, glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import *
from models.QaModel import QaModel
from models.EmbeddingsModel import EmbeddingsModel


embeddings_model = os.environ.get("model", None)
embeddings = None


# def str_to_class(classname):
#     return getattr(models.EmbeddingsDecoders, classname)

if embeddings_model == 'auto' or embeddings_model is None :
    embeddings_models = glob.glob('./checkpoints/embeddings/**')
    embeddings_models_scores = [re.search(EmbeddingsModel.metric+"=([0-9].[0-9]+)", m).group(1) for m in embeddings_models]
    embeddings_models = list(sorted(zip(embeddings_models_scores,embeddings_models )))
    score, path = (embeddings_models[0])
    embeddings_model = EmbeddingsModel.load_from_checkpoint(path)
    print(embeddings_model)
    embeddings = embeddings_model.embeddings


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

config = {
    "learning_rate":   float(os.environ.get("learning_rate", 0.0001)),
    "epochs":          int(os.environ.get("epochs", 1000)),
    "dropout":         float(os.environ.get("dropout", 0.2)),
    "batch_size":      int(os.environ.get("batch_size", 2048)),
    "decoder":         str(os.environ.get("decoder", 'DistMultDecoder')),
    "n_layers":        int(os.environ.get("n_layers", 1)),
    "reg":        int(os.environ.get("reg", 0.1)),
    'device': 'cuda'
}
    

if __name__ == '__main__':
    
    data  = MetaQa(2, config['batch_size'])
    model = QaModel(data.nodes_dataset.kb_n_nodes, 
        data.nodes_dataset.kb_n_relations, 
        data.qa_questions_datasets.qa_questions_vocab, 
        config, 
        embeddings)

        
    # Initialize data and model for pre-training
    
    wandb.init( project="metaqa", reinit=True)
    wandb_logger = WandbLogger(log_model=True)
    
    wandb_logger.watch(model, log="all", log_freq=50)
    embeddings_checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/qa/',
        filename=f'{model.cname()}'+'|{epoch}|{hit@10}'  
        )  
    
    trainer = pl.Trainer( 
        ** {'gpus':1, 'auto_select_gpus': True } if config['device'] == 'cuda' else {},
        callbacks=[embeddings_checkpoint_callback],
        logger= wandb_logger, 
        log_every_n_steps=1,
        check_val_every_n_epoch=50,
        limit_val_batches=1024,
        max_epochs=config['epochs'])
    
    trainer.fit(model, data)
    wandb.finish()

