# %%
import random
import networkx as nx
import matplotlib.pyplot as plt

import torch_geometric
import pydot
import matplotlib.image as pimg
from io import BytesIO

def plot_graph(graph):
    graphdot = nx.drawing.nx_pydot.to_pydot(graph)
    fig, ax = plt.subplots(figsize=(24, 24))
    img = graphdot.create('neato', 'png')
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
with open('assets/relations.dict') as fin:
    read_tsv = csv.reader(fin, delimiter="\t")
    relations_map = (dict(read_tsv))

G = nx.DiGraph()
G.add_edges_from(data.train_edge_index.T.numpy())

# %%

node = random.randint(0,data.num_nodes)
edges = []

for N in G[node]:
    edges.append((N, node))
    # for n in  G[N]:
    #     edges.append((n, N))
        # edges.append((n,node))
        # for nn in G[n] :
        #     edges.append((n, nn))
        
g = nx.DiGraph(edges)       
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
