# %%
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch_geometric


G = nx.balanced_tree(3,3)
nx.draw(G, pos=nx.draw(G))
# %%


dataset = torch_geometric.datasets.RelLinkPredDataset('FB15k-237','FB15k-237')
data = dataset[0]
import csv

with open('relations.dict') as fin:
    read_tsv = csv.reader(fin, delimiter="\t")
    relations_map = (dict(read_tsv))
    print(relations_map)
    

data.test_edge_index
data.test_edge_type

# %%

node = random.randint(0,data.num_nodes)
# is_in = data.train_edge_index[1]  == node 
# is_out = data.train_edge_index[0]  == node
# nodes, edges = [], []

# src_nodes_in, dst_nodes_in =   data.train_edge_index.T[is_in].T.tolist()
# rel_in = data.train_edge_type[is_in].tolist()
# src_nodes_out, dst_nodes_out =   data.train_edge_index.T[is_out].T.tolist()
# rel_out = data.train_edge_type[is_out].tolist()



G = nx.DiGraph()
G.add_edges_from(data.train_edge_index.T.numpy())

edges = []
print(node)

for N in G[node], nx.reverse_view(G)[node]:
    for n in  N:
        edges.append((n,node))
        # for nn in G[n] :
        #     edges.append((n, nn))
g = nx.DiGraph(edges)
nx.draw_networkx(g)
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
