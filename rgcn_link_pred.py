""""
Implements the link prediction task on the FB15k237 datasets according to the
`"Modeling Relational Data with Graph Convolutional Networks"
<https://arxiv.org/abs/1703.06103>`_ paper.

Caution: This script is executed in a full-batch fashion, and therefore needs
to run on CPU (following the experimental setup in the official paper).
"""
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn import GAE, RGCNConv
from torch_geometric.datasets import RelLinkPredDataset
import torch_scatter

data = RelLinkPredDataset('FB15k-237', 'FB15k-237')[0]
DEVICE='cuda'

class RGCNEncoder(torch.nn.Module):
    def __init__(self, n_nodes, hidden_channels, n_relations, n_layers=3):
        super().__init__()
        self.N = n_nodes
        self.R = n_relations
        self.E = hidden_channels
        self.L = n_layers

        self.embeddings = torch.nn.Parameter( torch.rand(n_nodes, hidden_channels, requires_grad=True) )
        self.rgnc_weights = torch.nn.ParameterList( [ torch.nn.Parameter(torch.rand(n_relations*2, hidden_channels, hidden_channels, requires_grad=True)) for _ in range(n_layers) ] )
        self.rgnc_biases = torch.nn.ParameterList( [ torch.nn.Parameter(torch.rand(n_relations*2, hidden_channels, requires_grad=True)) for _ in range(n_layers) ] )
        self.relu = torch.nn.ReLU()

    def get_messages(self, edge_index, edge_type, l):
        hidden = self.embeddings
        for r in range(self.R):
            for d in [0, 1]:
                dests = edge_index[d][edge_type == r]
                sources = edge_index[d][edge_type == r]
                messages = hidden[sources] @ self.rgnc_weights[l][r + self.R * d] + self.rgnc_biases[l][r + self.R * d]
                yield (messages, dests)
            # if d > 0:
                
            #     # layer_messages.append((messages, dests))
            #     yield (messages, dests)
            #     # output[:dests.max().item()+1] += torch_scatter.scatter_sum(messages, dests, dim=0)
            # else:
            #     messages = hidden[dests] @ self.rgnc_weights[l][r + self.R] + self.rgnc_biases[l][r + self.R] 
            #     # layer_messages.append((messages, sources))
            #     yield (messages, sources)
            #     # output[:sources.max().item()+1] += torch_scatter.scatter_sum(messages, sources, dim=0)

                        
                        
    def forward(self, edge_index, edge_type):
        edge_index = edge_index.to(DEVICE)
        edge_type = edge_type.to(DEVICE)
        
        output = self.embeddings


        for l in range(self.L):
            
            hidden = torch.zeros_like(output, device=DEVICE)
            msg_items = self.get_messages(edge_index, edge_type, l)
            messages, dests = zip(*msg_items)
            
            
            m = torch.vstack(messages)
            d = torch.hstack(dests)
            hidden[:d.max().item()+1] += torch_scatter.scatter_mean(m, d, dim=0)
            if l + 1 < self.L :
                hidden = self.relu(hidden)
            
            output = hidden           

                    # messages, dests =  zip(*(layer_messages ))
                    # print(dests.shape)
                    
                    # messages = torch.vstack(messages)
                    # dests = torch.hstack(dests)
                    # print(len(dests))
                    # print(len(messages))
                    
                    
                    # print(len(output))
                    # exit()
            
        return output



class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.rand(num_relations, hidden_channels, requires_grad=True))

    def score_objs(self, z, src_idx, rel_idx):
        z_src, z_dst = z[src_idx], z
        rel = self.rel_emb[rel_idx]
        return torch.sum(z_src * rel * z.unsqueeze(1), dim=-1)
    
    def forward(self, z, edge_index, edge_type):
        src, dst = edge_index
        # z_src, z_dst = z[src], z[dst]
        # rel = self.rel_emb[edge_type]
        return torch.sum(z[src] * self.rel_emb[edge_type] * z[dst], dim=1)




# @torch.no_grad()
# def test():
#     model.eval()
#     z = model.encode(data.edge_index, data.edge_type)

#     valid_mrr = compute_mrr(z, data.valid_edge_index, data.valid_edge_type)
#     test_mrr = compute_mrr(z, data.test_edge_index, data.test_edge_type)

#     return valid_mrr, test_mrr


# @torch.no_grad()
# def compute_mrr(z, edge_index: torch.Tensor, edge_type: torch.Tensor):
#     ranks = []
#     for i in tqdm(range(edge_type.numel())):
#         (src, dst), rel = edge_index[:, i], edge_type[i]

#         # Try all nodes as tails, but delete true triplets:
#         tail_mask = torch.ones(data.num_nodes, dtype=torch.bool)
#         for (heads, tails), types in [
#             (data.train_edge_index, data.train_edge_type),
#             (data.valid_edge_index, data.valid_edge_type),
#             (data.test_edge_index, data.test_edge_type),
#         ]:
#             tail_mask[tails[(heads == src) & (types == rel)]] = False

#         tail = torch.arange(data.num_nodes)[tail_mask]
#         tail = torch.cat([torch.tensor([dst]), tail])
#         head = torch.full_like(tail, fill_value=src)
#         eval_edge_index = torch.stack([head, tail], dim=0)
#         eval_edge_type = torch.full_like(tail, fill_value=rel)

#         out = model.decode(z, eval_edge_index, eval_edge_type)
#         perm = out.argsort(descending=True)
#         rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
#         ranks.append(rank + 1)

#         # Try all nodes as heads, but delete true triplets:
#         head_mask = torch.ones(data.num_nodes, dtype=torch.bool)
#         for (heads, tails), types in [
#             (data.train_edge_index, data.train_edge_type),
#             (data.valid_edge_index, data.valid_edge_type),
#             (data.test_edge_index, data.test_edge_type),
#         ]:
#             head_mask[heads[(tails == dst) & (types == rel)]] = False

#         head = torch.arange(data.num_nodes)[head_mask]
#         head = torch.cat([torch.tensor([src]), head])
#         tail = torch.full_like(head, fill_value=dst)
#         eval_edge_index = torch.stack([head, tail], dim=0)
#         eval_edge_type = torch.full_like(head, fill_value=rel)

#         out = model.decode(z, eval_edge_index, eval_edge_type)
#         perm = out.argsort(descending=True)
#         rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
#         ranks.append(rank + 1)

#     return (1. / torch.tensor(ranks, dtype=torch.float)).mean()

# from tqdm import tqdm 
# for epoch in tqdm(range(1, 1000)):
#     loss = train()
#     print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
#     if (epoch % 1) == 0:
#         valid_mrr, test_mrr = test()
#         print(f'Val MRR: {valid_mrr:.4f}, Test MRR: {test_mrr:.4f}')
