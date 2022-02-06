from email import message
import pytorch_lightning as pl 
import torch, torch_scatter
from torch.nn import Parameter, ParameterList, ReLU


class RGCNEncoder(torch.nn.Module):
    def __init__(self, n_nodes, hidden_channels, n_relations, n_layers=3):
        super().__init__()
        self.N = n_nodes
        self.R = n_relations
        self.E = hidden_channels
        self.L = n_layers

        self.embeddings = Parameter( torch.rand(n_nodes, hidden_channels, requires_grad=True) )
        self.rgnc_weights = ParameterList( [ Parameter(torch.rand(n_relations, hidden_channels, hidden_channels, requires_grad=True)) for _ in range(n_layers) ] )
        self.rgnc_biases = ParameterList( [ Parameter(torch.rand(n_relations, hidden_channels, requires_grad=True)) for _ in range(n_layers) ] )
        self.relu = ReLU()

    def get_messages(self, embeddings, edge_index, edge_type, l, ):
        
        
        for r in range(self.R):
            dests = edge_index[0][edge_type == r]
            sources = edge_index[1][edge_type == r]
            messages = embeddings[sources] @ self.rgnc_weights[l][r] + self.rgnc_biases[l][r]
            yield (messages, dests)
                        
                        
    def forward(self, edge_index, edge_type):
        edge_index = edge_index
        edge_type = edge_type
        
        output = self.embeddings


        for l in range(self.L):
            
            hidden = torch.zeros_like(output,)
            msg_items = self.get_messages(output, edge_index, edge_type, l, )
            messages, dests = zip(*msg_items)
            

            m = torch.vstack(messages).device(hidden)
            d = torch.hstack(dests).device(hidden)

            hidden += torch_scatter.scatter_sum(m, d, dim=0, out=hidden)
            if l + 1 < self.L :
                hidden = self.relu(hidden)
            
            output = hidden           
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

        return torch.sum(z[src] * self.rel_emb[edge_type] * z[dst], dim=1)




class LinkPredictor(pl.LightningModule):
    def __init__(self, num_nodes: int, num_relations: int, config: dict) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.config = config
        
        self.encode = RGCNEncoder(self.num_nodes, self.config['embeddings_size'], self.num_relations// 2, self.config['n_layers'] )
        self.decode = DistMultDecoder(self.num_relations // 2, hidden_channels=self.config['embeddings_size'])
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        decode_batch, encode_batch =  train_batch
        decode_batch_index, decode_batch_type = decode_batch
        encode_batch_index, encode_batch_type = encode_batch[...]

        
        z = self.encode(encode_batch_index.T, encode_batch_type)
        

        neg_edge_index = self.negative_sampling(decode_batch_index.T, self.num_nodes)
        pos_out = self.decode(z, decode_batch_index.T, decode_batch_type)
        pos_bce_loss = self.loss(pos_out, torch.ones_like(pos_out)) 
        
        neg_out = self.decode(z, neg_edge_index, decode_batch_type)
        neg_bce_loss = self.loss(neg_out, torch.zeros_like(neg_out)) 
        
        reg_loss = sum([ p.pow(2).mean() for p in self.parameters() ])

        loss = pos_bce_loss + neg_bce_loss + self.config['reg'] * reg_loss
        self.log("loss", loss.item())
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        (valid_edge_index, valid_edge_type), (train_edge_index, train_edge_type), (all_edge_index, all_edge_type) = batch
        
        if batch_idx == 0:
            self.z = self.encode(train_edge_index.T, train_edge_type)
            

        (valid_src_index, valid_dst_index) = valid_edge_index.T
        obj_scores = self.decode.score_objs(self.z, valid_src_index, valid_edge_type )

        ranks = obj_scores.argsort(0, descending=True)


        for k in [1,3,10]:
            hits = (ranks[:k]== valid_dst_index).float().mean()  * k
            self.log(f"hit@{k}", hits, on_epoch=True)
        
        
        # mrr = (1. / ranks.float().mean())
        # self.log(f"mrr", mrr, on_epoch=True)

        return 
    
    import torch 


    def negative_sampling(self, edge_index, num_nodes):
        # Sample edges by corrupting either the subject or the object of each edge.
        random_mask = torch.rand(edge_index.size(1)) < 0.5

        neg_edge_index = edge_index.clone()
        neg_edge_index[0, random_mask] = torch.randint(num_nodes, (random_mask.sum(), )).device(edge_index.device)
        neg_edge_index[1, ~random_mask] = torch.randint(num_nodes, ((~random_mask).sum(), )).device(edge_index.device)
        return neg_edge_index
