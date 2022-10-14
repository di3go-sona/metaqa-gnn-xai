#%%
from train import *
from models.rgcnqa_explainer import RGCNQAExplainer
from torch_geometric.utils import k_hop_subgraph
import torch
from utils import *
import networkx as nx
from models.rgcnqa import *
from models.embeddings import *
from transformers import AutoTokenizer
from globals import *
from tqdm import tqdm
 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
from checkpoints import *


#%%
# Evaluate accuracy 

for m, h in checkpoints_set:
    xai_accuracy = []
    xai_size = []
    qa_data = QAData(METAQA_PATH, [h], tokenizer, train_batch_size= 128, val_batch_size= 4, use_ntm= False)
    items = []
    try:
        qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)
    except Exception:
        qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=True, verbose=False)
    for batch_id, batch in tqdm(enumerate(qa_data.test_dataloader())):
        for batch_subid, acc in enumerate(qa_model.evaluate_batch(batch)):
            if acc == 1:
                question_id = batch_id*4 + batch_subid

                
                src_idx, _, dst_idx, question = qa_data.val_ds_unflattened[question_id]
                
                golden_nodes, golden_edges = get_golden_path(qa_data, question_id,dst_idx[0])
                if len(golden_edges) == 0 : continue
                x = qa_model.nodes_emb
                
                raw_index = qa_model.edge_index
                index = raw_index.T[[0,2]].cpu()
                relations = raw_index.T[1].cpu()

                subset, index, inv, edge_mask = k_hop_subgraph(src_idx, qa_model.hops, index)
                relations = relations[edge_mask]


                coeffs = {
                        # 'edge_size': 0.7,
                        # 'edge_reduction': 'mean',
                        # 'node_feat_size': 0.3  ,
                        # 'node_feat_reduction': 'mean',
                        # 'edge_ent': 0.3,
                        # 'node_feat_ent': 0.3,
                    }

                explainer = RGCNQAExplainer(
                    qa_model,
                    epochs=200,
                    lr=0.01,
                    return_type='prob',
                    feat_mask_type='scalar',
                    num_hops=qa_model.hops,
                    **coeffs)

                node_feat_mask, edge_mask = explainer.explain_node(dst_idx[0], 
                                                                x,
                                                                index,
                                                                relations = relations,
                                                                src_idx = src_idx,
                                                                question = question,
                                                                relabel_nodes = False)

                initial_edges = index.size(1)
                initial_nodes = index.unique().size(0)
                xai_nodes = (node_feat_mask > 0.5).sum().item()
                xai_edges = (edge_mask > 0.5).sum().item()


                _, _, _, valid_edge_mask = torch_geometric.utils.k_hop_subgraph(dst_idx[0], qa_data.hops[0], index )
                # print(valid_edge_mask.shape, index.shape, valid_edge_mask)
                golden_edges = [ge for ge in golden_edges if ge in [ ge for ge in golden_edges if list(ge) in index.T[valid_edge_mask > 0.5].tolist() ] ]

                correct_golden_edges =  [ ge for ge in golden_edges if list(ge) in index.T[edge_mask > 0.5].tolist() ]
                
                
                # print( index.T[edge_mask > 0.5].tolist())
                # print(golden_edges)
                try:
                    items.append((initial_edges, xai_edges, len(golden_edges), len(correct_golden_edges)/len(golden_edges)))
                    print((initial_edges, xai_edges, len(golden_edges), len(correct_golden_edges)/len(golden_edges)))
                except:
                    continue
        if len(items) > 128:
            break
    initial_edges, xai_edges, golden_edges, correct_golden_edges = (sum(l)/len(l) for l in zip(*items))
    

                
                
    with open('../data/explainability.out', 'a') as fout:
        fout.write(f'\n{m}|{h}|{initial_edges}, {xai_edges}, {golden_edges}, {correct_golden_edges}, {len(items)}')

# cervedc
# %%
