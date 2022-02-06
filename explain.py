
# %%
from settings import *
from run import LinkPredictor, FB15KData

data = FB15KData()
model = LinkPredictor.load_from_checkpoint('checkpoints/model_1.ckpt',
                                           num_nodes=data.num_nodes, 
                                           num_relations=data.num_relations, 
                                           config=wandb.config)

# edge_index, edge_type = data.train_data[...]
index, type =  data.train_data[...]
z = model.encode(index.T, type)


# %%
i, nodes_cache = 0, {}
#%%
import torch
nodes = None
while i <len(data.test_data):
    (src, dst), type = data.test_data[i]
    # print((src, dst), type.unsqueeze(0) )
    scores = model.decode.score_objs(z, src.unsqueeze(0), type.unsqueeze(0)).squeeze()
    ranks = scores.argsort(descending=True)
    r = torch.arange(0,14541)[ranks == dst]
    print(r)
    i+=1
    if r < 3:
        nodes = [ n.item() for n in [src, dst] + [ ranks[j] for j in range(3) ] ]
        print(i, r )
        break
    
# %%
import torch_geometric
dataset = torch_geometric.datasets.RelLinkPredDataset('FB15k-237','FB15k-237')


# Load human readable names
import csv
with open('./FB15k-237/FB15k-237/raw/relations.dict') as fin:
    read_tsv = csv.reader(fin, delimiter="\t")
    relations_map = (dict(read_tsv))
with open('./FB15k-237/FB15k-237/raw/entities.dict') as fin:
    read_tsv = csv.reader(fin, delimiter="\t")
    entities_map = (dict(read_tsv))

import networkx as nx
G = nx.DiGraph()
for s,d,t  in zip(*dataset[0].train_edge_index.numpy(), dataset[0].train_edge_type):

    G.add_edge(s,d,id=t.item())
for s,d,t  in zip(*dataset[0].test_edge_index.numpy(), dataset[0].test_edge_type):

    G.add_edge(s,d,id=t.item())


# %%
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from io import BytesIO
def plot_graph(graph):
    graphdot = nx.drawing.nx_pydot.to_pydot(graph)
    fig, ax = plt.subplots(figsize=(24, 24))
    img = graphdot.create(['neato', '-Tpng', '-Goverlap=false'], 'png')
    plt.imshow( pimg.imread(BytesIO(img)) )
# %%

(src, dst), candidates  = nodes[:2], nodes[2:]
# src = random.randint(0,data.num_nodes)
# dst = random.randint(0,data.num_nodes)

edges = []
for node in [src, dst] + candidates :
    try:
        for N in G[node]:
            edges.append((node, N))  
    except:
        pass
    try:
        for N in nx.reverse_view(G)[node]:
            edges.append((N, node)) 
    except:
        pass
        


import requests, json



def node_name(node):
    return(str(node))
    if node in nodes_cache:
        return nodes_cache[node]
    
    id = entities_map[str(node)]
    args = { 'key':  'AIzaSyA1vYTArzSfaZLQ9ikIyh7RFmtOTos0v_k', 'ids': id}

    url = 'https://kgsearch.googleapis.com/v1/entities:search/'

    try:
        r = json.loads(requests.get(url, args).content)['itemListElement'][0]['result']['name']
        nodes_cache[node]=r
        return r
    except:
        print(id)
        return 'UNK'


def edge_name(edge):
    if isinstance(edge, tuple):
        s,d = edge
        return relations_map[str(G[s][d]['id'])].split('/')[-1]
    raise TypeError

g = nx.DiGraph()     

g.add_node(node_name(src), color='grey',style='filled')
g.add_node(node_name(dst), color='green',style='filled')
for c in  candidates:
    if c != dst:
        g.add_node(node_name(c), color='orange',style='filled')

        g.add_edge(node_name(src),node_name(c), label=edge_name((src,dst)), style='dotted')
    else:
        g.add_edge(node_name(src),node_name(c), label=edge_name((src,dst)), style='dotted',  color='green')
for e in edges:
    s,d = e
    g.add_edge(node_name(s),node_name(d), label=edge_name(e))

plot_graph(g) 
# %%
