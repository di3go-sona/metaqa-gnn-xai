#%%
from models.rgcnqa import *
from models.rgcnqa_explainer import *
from models.embeddings import *
from transformers import AutoTokenizer
from globals import *
from tqdm import tqdm 
import torch
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
checkpoints_set = {
    ('Funnel',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=10.ckpt',
#     ('Straight',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=19.ckpt',
#     ('Concatenated',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=13.ckpt',
#     ('Funnel',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|72>36>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=37.ckpt',
#     ('Straight',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=39.ckpt',
#     ('Concatenated',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=61.ckpt',
#     ('Straight|Bert-frozen',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|bert_frozen|epoch=129.ckpt',
    # ('Funnel',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|72>36>18>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=974.ckpt',
    # ('Straight',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|36>36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=437.ckpt',
    # ('Concatenated',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|36>36>36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=549.ckpt',
}




# Evaluate accuracy
for m, h in checkpoints_set:
    qa_data = QAData(METAQA_PATH, [h], tokenizer, train_batch_size= 128, val_batch_size= 16, use_ntm= True)
    qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)
    
    coeffs = {
        'edge_size': 1.0,
        'edge_reduction': 'mean',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 0.5,
        'node_feat_ent': 0.0,
    }

    explainer = RGCNQAExplainer(
        qa_model,
        epochs=100,
        lr=0.01,
        return_type='prob',
        feat_mask_type='scalar',
        num_hops=qa_model.hops,
        **coeffs)
    
    x = qa_model.nodes_emb
    
    for src_idx, _, dst_idxs, question in qa_data.val_ds_unflattened:
        for dst_idx in dst_idxs:

            

            index = qa_model.edge_index.T[[0,2]].cpu()
            relations = qa_model.edge_index.T[1].cpu()

            subset, index, inv, edge_mask = k_hop_subgraph(src_idx, qa_model.hops, index)
            relations = relations[edge_mask]

            node_feat_mask, edge_mask = explainer.explain_node(dst_idx, 
                                                            x,
                                                            index,
                                                            relations = relations,
                                                            src_idx = src_idx,
                                                            question = question,
                                                            relabel_nodes = False)

            old_size = torch.concat([*index.T]).unique().size(0)
            new_size = node_feat_mask.nonzer().sum().item()
            compression = old_size/(new_size+EPS)*100

            print(compression, old_size, new_size)
            
                        
            g_nodes_index = torch.concat([*index.T]).unique().detach().tolist()
            g_edge_index = index.T.detach().tolist()

            nodes_mask = node_feat_mask.detach().tolist()
            edges_mask = edge_mask.detach().tolist()


            import networkx as nx
            G = nx.DiGraph()
            G.add_nodes_from(g_nodes_index)


            for w, n in zip( nodes_mask, g_nodes_index):
                
                    
                attrs = {
                    'label':  qa_data.entities_names[n],
                    'size': 10 + int(w/max(nodes_mask)*100),
                }
                if n == src_idx:
                    attrs['color'] = 'yellow'
                if n == dst_idx:
                    attrs['color'] = 'green'
                G.add_node(n, **attrs)
                
            from random import randint
            rel_colors = {
                i: f"{randint(0, 128)},{randint(0, 128)},{randint(0, 128)}" for i in range(max(relations.tolist())+1)
            }

            for w, e, r in zip( edges_mask, g_edge_index, relations.tolist()) :
                print(w,e )
                attrs = {
                    'color': f'rgba({rel_colors[r]}, {w})'
                }
                G.add_edge(*e, **attrs)

                
            from pyvis.network import Network
            net = Network(directed=True)
            net.from_nx(G)
            net.show('nx.html')

    # hits_at_k = [ h  for batch in tqdm(list(qa_data.test_dataloader())) for h in qa_model.evaluate_batch(batch)]
    # print(m, h, sum(hits_at_k)/len(hits_at_k))

# %%
