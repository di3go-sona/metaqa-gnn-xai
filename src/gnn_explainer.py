#%%
from train import *
from models.rgcnqa_explainer import RGCNQAExplainer


MODEL_PATH = '../data/checkpoints/qa/2-hops/QA_RGCN|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=279.ckpt'
QUESTION_ID = 3253
ANSWER_ID = 0

def run():

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = QA_RGCN.load_from_checkpoint(MODEL_PATH, fast=True)
    qa_data = QAData('dataset', [model.hops], tokenizer, False)

    x = model.nodes_emb.weight
    src_idx, _, dst_idx, question = qa_data.val_ds_unflattened[QUESTION_ID]



    index = model.edge_index.T[[0,2]].cpu()
    relations = model.edge_index.T[1].cpu()

    subset, index, inv, edge_mask = k_hop_subgraph(src_idx, model.hops, index)
    relations = relations[edge_mask]


    explainer = RGCNQAExplainer(
        model,
        epochs=100,
        return_type='raw',
        feat_type='feature',
        num_hops=model.hops,)

    node_feat_mask, edge_mask = explainer.explain_node(dst_idx[ANSWER_ID], 
                                                    x,
                                                    index,
                                                    relations= relations,
                                                    src_idx= src_idx,
                                                    question= question)


    ax, G = explainer.visualize_subgraph(dst_idx[ANSWER_ID], index, edge_mask)


    for n in G.nodes():
        name = qa_data.entities_names[n]
        G.nodes[n]['label'] = name

    for e in G.edges():
        G.edges[e]['color'] = '#' + hex(int(G.edges[e]['att']*256))[2:] * 3

        
    from pyvis.network import Network
    net = Network(directed=True)
    net.from_nx(G)
    net.show('nx.html')

run()