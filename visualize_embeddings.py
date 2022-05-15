#%%
import glob
from matplotlib import pyplot as plt
import sklearn.decomposition, sklearn.cluster
import wandb, os
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datasets import *
from models import *
from models.EmbeddingsModel import EmbeddingsModel



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

def load_model():
    
    embeddings_models = glob.glob('./checkpoints/embeddings/**')
    embeddings_models_scores = [re.search(EmbeddingsModel.metric+"=([0-9].[0-9]+)", m).group(1) for m in embeddings_models]
    embeddings_models = list(sorted(zip(embeddings_models_scores,embeddings_models )))
    score, path = (embeddings_models[0])
    model = EmbeddingsModel.load_from_checkpoint(path, )
    return model 
    



# Load Data and Models
data  = MetaQaEmbeddings(config['batch_size'], config['val_batch_size'])
model = load_model()
embeddings = model.embeddings.detach().numpy()

# PCA
pca = sklearn.decomposition.PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# KMeans
clusters = {}
for i in tqdm(range(2,5)):
    clustering = sklearn.cluster.KMeans(n_clusters=i)
    res = clustering.fit_predict(embeddings)
    clusters[i] = res


print(reduced_embeddings)
print(embeddings.shape)
print(reduced_embeddings.shape)


# Get nodes names
nodes = data.ds.kb_nodes_vocab.lookup_tokens(range(data.ds.kb.number_of_nodes()))

# Get edges names
s_ids, d_ids, t_ids = zip(*data.ds.kb.edges(data='label'))
s_ids =  data.ds.kb_nodes_vocab.lookup_tokens([ int(i) for i in s_ids])
d_ids =  data.ds.kb_nodes_vocab.lookup_tokens([ int(i) for i in d_ids])
t_ids =  data.ds.kb_edges_vocab.lookup_tokens([ int(i) for i in t_ids])
edges = list(zip(s_ids,d_ids,t_ids))



# Create nx DiGraph
G = nx.classes.DiGraph()

# Adding nodes
for i, n in enumerate(tqdm(nodes, desc="Adding nodes")):
    props = {}
    for j in range(2,5):
        props[f'c_{j}'] = clusters[j][i]
    
    props['x'], props['y'] = reduced_embeddings[i]
    G.add_node(n, **props)

# Adding edges
for s,d,t in tqdm(edges, desc="Adding edges"):
    G.add_edge(s,d, label=t)



#%%
for s in ['will smith']:
    for n in [1,2,3]:
        edges = nx.dfs_edges(G.to_undirected(), source=s,depth_limit=n)
        
        # print(len(list(edges)))
        
        g = nx.classes.DiGraph()
        g.add_edges_from(edges)
        # for e in edges:
            
        nx.drawing.nx_pydot.write_dot(g,f'./dataset/kb_{s}_{n}.dot')


# %%
        
    # plt.figure(figsize=(200,200))
    # plt.scatter(*reduced_embeddings.T)
    # embeddings = embeddings_model.embeddings
            
    # Initialize data and model for pre-training
    
    # wandb.init( project="metaqa", reinit=True)
    # wandb_logger = WandbLogger(log_model=True)
    
                
    # embeddings_checkpoint_callback = ModelCheckpoint(
    #     dirpath='checkpoints/embeddings/',
    #     filename=f'{model.cname()}'+'|{epoch}|{hit@10}'  
    #     )  
    
    # trainer = pytorch_lightning.Trainer( 
    #     ** {'gpus':1, 'auto_select_gpus': True } if config['device'] == 'cuda' else {},
    #     callbacks=[embeddings_checkpoint_callback],
    #     logger= wandb_logger, 
    #     log_every_n_steps=1,
    #     check_val_every_n_epoch=50,
    #     limit_val_batches=1024,
    #     max_epochs=config['epochs'])
    
    # trainer.fit(model, data)
    # wandb.finish()


# %%
