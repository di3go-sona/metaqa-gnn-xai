#%%
import glob
from globals import *
import linecache
import os
import networkx as nx
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
        0: 'genre', 
        1: 'tags', 
        2: 'writer', 
        3: 'imdbrating', 
        4: 'actor', 
        5: 'language', 
        6: 'year', 
        7: 'writer', 
        8: 'imdbvotes'}
    return dic.get(rel_id, 'movie')

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
# %%

def get_golden_path(index, relations, src_idx, question_id, hops, split):
    golden_nodes = []
    golden_edges = []
    qtype = get_qtype(question_id, hops, split)
    net = nx.DiGraph()
    
    for e, r in zip(index.T.tolist(), relations.tolist()):
        net.add_edge(*e, label=r)
    
    
    old_queue = [src_idx]
    queue = []
    for d in range(1, len(qtype)):
        golden_dst_type = qtype[d]
        for src in old_queue:
            for dst in net[src]:
                r = net.edges[(src, dst)]['label']
                dst_type = get_target_type(r)
                
                
                print(d, dst_type, golden_dst_type)
                if dst_type == golden_dst_type:
                    queue.append((dst))
                    golden_nodes.append(dst)
                    golden_edges.append((src, dst))
                    
        old_queue = queue
        queue = []

            
    return golden_nodes, golden_edges

