import wandb, os, glob
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import *
from models.QaModel import QaModel
from models.EmbeddingsModel import EmbeddingsModel


embeddings_model = os.environ.get("embeddings_model", None)
embeddings_model = os.environ.get("model", None)

embeddings = None

config = {
    "learning_rate":   float(os.environ.get("learning_rate", 0.00001)),
    "epochs":          int(os.environ.get("epochs", 3)),
    "dropout":         float(os.environ.get("dropout", 0.2)),
    "batch_size":      int(os.environ.get("batch_size", 512)),
    "n_layers":        int(os.environ.get("n_layers", 2)),
    "reg":             float(os.environ.get("reg", 0.001)),
    "hops":            int(os.environ.get("hops", 3)),
    "n_pos":           int(os.environ.get("n_pos", 1)),
    "neg_ratio":       int(os.environ.get("neg_ratio", 10)),
    'device':          'cuda'
}
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

# if embeddings_model == 'auto' or embeddings_model is None :

if "CUDA_DEVICE_ORDER" not in os.environ: os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" not in os.environ: os.environ["CUDA_VISIBLE_DEVICES"]="1"
data  = MetaQa(config['hops'], config['batch_size'], config['n_pos'], config['neg_ratio'])
model = QaModel(data.nodes_dataset.kb_n_nodes, 
    data.nodes_dataset.kb_n_relations, 
    data.qa_questions_datasets.qa_questions_vocab, 
    config, 
    embeddings)

