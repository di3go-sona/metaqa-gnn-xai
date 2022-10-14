#%%
import glob

import torch_geometric
from globals import *
import linecache
import os
import networkx as nx
import torch
def get_model_path(model_name):
    candidates = glob.glob(f'{QA_CHECKPOINTS_PATH}/**/{model_name}*')
    if len(candidates) > 1:
        assert f"Too many matches for model named {model_name}"
    return candidates[0]

def get_model_path(model_name):
    candidates = glob.glob('checkpoints/qa/**/' + model_name + '*')
    if len(candidates) > 1:
        assert f"Too many matches for model named {model_name}"
    return candidates[0]

def list_qa_models():
    candidates = glob.glob(f'{QA_CHECKPOINTS_PATH}/**/*.ckpt')
    return candidates

def list_embeddings_models():
    candidates = glob.glob(f'{EMBEDDINGS_CHECKPOINTS_PATH}/**.ckpt')
    return candidates



def get_qtype(q_id, q_hops, q_split):
    path = os.path.join(METAQA_PATH, f"{q_hops}-hop", f"qa_{q_split}_qtype.txt")
    line = linecache.getline(path, q_id+1).strip()
    line = line.split('_to_' )
    return line

def get_target_type(rel_id):
    dic = {
        
        0: ('genre', 'movie'), 
        9: ('movie', 'genre'),
        1: ('tags', 'movie'), 
        10: ('movie', 'tags'),
        2: ('director', 'movie'), 
        11: ('movie', 'director'),
        3: ('imdbrating', 'movie'), 
        12: ('movie', 'imdbrating'),
        4: ('actor', 'movie'), 
        13: ('movie', 'actor'),
        5: ('language', 'movie'), 
        14: ('movie', 'language'),
        6: ('year', 'movie'), 
        15: ('movie', 'year'),
        7: ('writer', 'movie'), 
        16: ('movie', 'writer'),
        8: ('imdbvotes', 'movie'),
        17: ('movie', 'imdbvotes')}
    
    return dic[rel_id]

def bfs_layers(G, sources):
    """Returns an iterator of all the layers in breadth-first search traversal.

    Parameters
    ----------
    G : NetworkX graph
        A graph over which to find the layers using breadth-first search.

    sources : node in `G` or list of nodes in `G`
        Specify starting nodes for single source or multiple sources breadth-first search

    Yields
    ------
    layer: list of nodes
        Yields list of nodes at the same distance from sources

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> dict(enumerate(nx.bfs_layers(G, [0, 4])))
    {0: [0, 4], 1: [1, 3], 2: [2]}
    >>> H = nx.Graph()
    >>> H.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    >>> dict(enumerate(nx.bfs_layers(H, [1])))
    {0: [1], 1: [0, 3, 4], 2: [2], 3: [5, 6]}
    >>> dict(enumerate(nx.bfs_layers(H, [1, 6])))
    {0: [1, 6], 1: [0, 3, 4, 2], 2: [5]}
    """
    if sources in G:
        sources = [sources]

    current_layer = list(sources)
    visited = set(sources)

    for source in current_layer:
        if source not in G:
            raise nx.NetworkXError(f"The node {source} is not in the graph.")

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    while current_layer:
        yield current_layer
        next_layer = list()
        for node in current_layer:
            for child in G[node]:
                if child not in visited:
                    visited.add(child)
                    next_layer.append(child)
        current_layer = next_layer


def get_golden_path(qa_data, question_id, dst_idx ):
    golden_nodes = []
    golden_edges = []
    
    src_idx, _, dst_idx, question = qa_data.val_ds_unflattened[question_id]
    # print(src_idx, _, dst_idx)
    # print( qa_data.tokenizer.decode(question) )
    raw_index = qa_data.get_triples()
    index = raw_index.T[[0,2]].cpu()
    relations = raw_index.T[1].cpu()

    subset, index, inv, edge_mask = torch_geometric.utils.k_hop_subgraph(src_idx, qa_data.hops[0], index)
    relations = relations[edge_mask]
    # subset, index, inv, edge_mask = torch_geometric.utils.k_hop_subgraph(dst_idx, qa_data.hops[0], index, flow='target_to_source')
    # relations = relations[edge_mask]
    # print(index)
    qtype = get_qtype(question_id, qa_data.hops[0], 'dev')
    # print(relations.shape, index.shape)
    # print(qtype)
    # print(index, relations)
    net = nx.DiGraph()
    
    for e, r in zip(index.T.tolist(), relations.tolist()):
        net.add_edge(*e, label=r)
    
    
    old_queue = [src_idx]
    queue = []
    for d in range(0, len(qtype)-1):
        for src in old_queue:
            for dst in net[src]:
                r = net.edges[(src, dst)]['label']
                src_type, dst_type = get_target_type((r+9)%18)
                
                
                # print( (src_type,dst_type))
                if (src_type == qtype[d]) and (dst_type == qtype[d+1]):
                    queue.append((dst))
                    golden_nodes.append(dst)
                    golden_edges.append((src, dst))
                    
        old_queue = queue
        queue = []
    
    # print((src_idx, qa_data.hops[0], torch.tensor(golden_edges).long().T))/
    # print(torch.tensor(golden_edges).T.long())
    if len(golden_edges) > 0:
        _, _, _, valid_edge_mask = torch_geometric.utils.k_hop_subgraph(dst_idx, qa_data.hops[0], index )
        # print(valid_edge_mask.shape, index.shape, valid_edge_mask)
        golden_edges = [ge for ge in golden_edges if ge in [ ge for ge in golden_edges if list(ge) in index.T[valid_edge_mask > 0.5].tolist() ] ]

    return golden_nodes, golden_edges

# golden_nodes, golden_edges = get_golden_path(qa_data, question_id)
# golden_nodes, golden_edges
# %%
