# %%
import random
import networkx as nx
import matplotlib.pyplot as plt

import torch_geometric
import matplotlib.image as pimg
from io import BytesIO

def plot_graph(graph):
    graphdot = nx.drawing.nx_pydot.to_pydot(graph)
    fig, ax = plt.subplots(figsize=(24, 24))
    img = graphdot.create(['neato', '-Tpng', '-Goverlap=false'], 'png')
    plt.imshow( pimg.imread(BytesIO(img)) )
    

graph = nx.balanced_tree(3,3)
# nx.draw(graph, pos=nx.draw(graph))
plot_graph(graph)




# %%

# Load pytorch dataset 
dataset = torch_geometric.datasets.RelLinkPredDataset('FB15k-237','FB15k-237')
data = dataset[0]

# Load human readable names
import csv
with open('./FB15k-237/FB15k-237/raw/relations.dict') as fin:
    read_tsv = csv.reader(fin, delimiter="\t")
    relations_map = (dict(read_tsv))
with open('./FB15k-237/FB15k-237/raw/entities.dict') as fin:
    read_tsv = csv.reader(fin, delimiter="\t")
    entities_map = (dict(read_tsv))

    
G = nx.DiGraph()
for s,d,t  in zip(*data.train_edge_index.numpy(), data.train_edge_type):

    G.add_edge(s,d,id=t.item())
for s,d,t  in zip(*data.test_edge_index.numpy(), data.test_edge_type):

    G.add_edge(s,d,id=t.item())
# G.add_edges_from(data.train_edge_index.T.numpy(), id = data.train_edge_type)

# %%
r = random.randint(0,2000)
r=12
src, dst = data.test_edge_index[:,r].numpy()
type = data.test_edge_index[:,r].numpy()
# src = random.randint(0,data.num_nodes)
# dst = random.randint(0,data.num_nodes)

edges = []
for node in [src, dst]:
    for N in G[node]:
        edges.append((node, N))  

    for N in nx.reverse_view(G)[node]:
        edges.append((N, node))
        


import requests, json


nodes_cache = {}
def node_name(node):
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
g.add_node(node_name(src), color='green',style='filled')
g.add_node(node_name(dst), color='red',style='filled')
g.add_edge(node_name(src),node_name(dst), label=edge_name((src,dst)), style='dotted')
for e in edges:
    s,d = e
    g.add_edge(node_name(s),node_name(d), label=edge_name(e))

plot_graph(g) 
# %%




# nx.draw_networkx(g)
# G.add_nodes_from(list(range(dataset.num_nodes)))

# G.add_nodes_from(list(set(src_nodes_in + dst_nodes_in + src_nodes_out + dst_nodes_out)))
# G.add_edges_from(zip(src_nodes_in + src_nodes_out, dst_nodes_in + dst_nodes_out, rel_in + rel_out) )

# pos = nx.spring_layout(G)
# nx.draw(G, pos, node_size=50)

# edge_labels = nx.get_edge_attributes(G)
# nx.draw_networkx_edge_labels(G, pos, labels = edge_labels)

# nx.draw_networkx(G, node_size=50)
# [ relations_map[r].split('/')[-1] for r in list(src_rel) ]

# %%
