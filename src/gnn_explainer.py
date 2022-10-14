#%%
from train import *
from models.rgcnqa_explainer import RGCNQAExplainer
from torch_geometric.utils import k_hop_subgraph
import torch
from utils import *
import networkx as nx

MODEL_PATH = '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=39.ckpt'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = RGCNQA.load_from_checkpoint(MODEL_PATH, bias=False, fast=True)
qa_data = QAData('../data/metaqa', [model.hops], tokenizer, False)

#%%
QUESTION_ID = 0
ANSWER_ID = 0

x = model.nodes_emb
src_idx, _, dst_idx, question = qa_data.val_ds_unflattened[QUESTION_ID]

raw_index = model.edge_index
index = raw_index.T[[0,2]].cpu()
relations = raw_index.T[1].cpu()

subset, index, inv, edge_mask = k_hop_subgraph(src_idx, model.hops, index)
relations = relations[edge_mask]


coeffs = {
        'edge_size': 0.7,
        'edge_reduction': 'mean',
        'node_feat_size': 0.3  ,
        'node_feat_reduction': 'mean',
        'edge_ent': 0.3,
        'node_feat_ent': 0.3,
    }

explainer = RGCNQAExplainer(
    model,
    epochs=200,
    lr=0.02,
    return_type='raw',
    feat_mask_type='scalar',
    num_hops=model.hops,
    **coeffs)

node_feat_mask, edge_mask = explainer.explain_node(dst_idx[ANSWER_ID], 
                                                x,
                                                index,
                                                relations = relations,
                                                src_idx = src_idx,
                                                question = question,
                                                relabel_nodes = False)

# old_size = torch.concat([*index.T]).unique().size(0)
# new_size = (node_feat_mask > 0.5).sum()
# compression = new_size/old_size*100



g_nodes_index = torch.concat([*index.T]).unique().detach().tolist()
g_edge_index = index.T.detach().tolist()

nodes_mask = node_feat_mask.detach().tolist()
edges_mask = edge_mask.detach().tolist()
    
golden_nodes, golden_edges = get_golden_path(index, relations, src_idx, QUESTION_ID, model.hops, 'dev')


G = nx.DiGraph()

for w, n in zip( nodes_mask, g_nodes_index):
    # print(n, '->', w )
        
    attrs = {
        'label':  qa_data.entities_names[n],
        'size': 10, # + int(w/max(nodes_mask)*100),
    }

    if n in golden_nodes:
        attrs['color'] = '#f5d142'
    if n == dst_idx[ANSWER_ID]:
        attrs['color'] = 'green'
    if n == src_idx:
        attrs['color'] = 'black'

    G.add_node(n, **attrs)
    
from random import randint
rel_colors = {
    i: f"{randint(0, 128)},{randint(0, 128)},{randint(0, 128)}" for i in range(max(relations.tolist())+1)
}

for w, e, r in zip( edges_mask, g_edge_index, relations.tolist()) :
    # print(e, '->', w )
    attrs = {
        # 'color': f'rgba({rel_colors[r]}, {w})'
        'color': f'rgba(0,0,0, {w})'
        # 'color': f'rgba(0,0,0,1)'
    }
    if tuple(e) in golden_edges:
        attrs['color'] = f'rgba(145, 122, 23, {w})'
        # attrs['color'] = f'#f5d142'
    G.add_edge(*e, **attrs)

    
from pyvis.network import Network
net = Network(directed=True)
net.from_nx(G)
net.show('nx.html')



# %%
