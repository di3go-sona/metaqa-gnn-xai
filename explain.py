
# %%
from settings import *
from run import LinkPredictor, FB15KData

data = FB15KData()
model = LinkPredictor.load_from_checkpoint('None/version_None/checkpoints/epoch=199-step=52999.ckpt',
                                           num_nodes=data.num_nodes, 
                                           num_relations=data.num_relations, 
                                           config=wandb.config)

# edge_index, edge_type = data.train_data[...]
z = model.encode( *data.train_data[...])



# %%

(src, dst), type = data.test_data[0]
model.decode(z, src, type)

# %%
data.test_data[...]