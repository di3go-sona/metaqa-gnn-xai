

import wandb, torch, torch_geometric

import pytorch_lightning as pl
from tqdm import tqdm 
from pytorch_lightning.loggers import WandbLogger
from rgcn_link_pred import GAE, RGCNEncoder, DistMultDecoder

wandb.config = {
    "dataset": 'FB15k-237',
    "learning_rate": 0.01,
    "reg": 0.01,
    "epochs": 100,
    "embeddings_size": 500,
    "n_layers": 2
        }

import torch_geometric
import pickle

dataset = torch_geometric.datasets.RelLinkPredDataset( wandb.config['dataset'],
                                                         wandb.config['dataset'])
data = dataset[0]

# with open('pickle', 'wb') as file:
#     t = (data.train_edge_index.T.numpy(), data.train_edge_type.numpy(), data.test_edge_index.T.numpy(), data.test_edge_type.numpy())
#     pickle.dump(t, file )
#     exit()
    
    
    
def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    mask_1 = torch.rand(edge_index.size(1)) < 0.5
    mask_2 = ~mask_1

    neg_edge_index = edge_index.clone()
    neg_edge_index[0, mask_1] = torch.randint(num_nodes, (mask_1.sum(), ))
    neg_edge_index[1, mask_2] = torch.randint(num_nodes, (mask_2.sum(), ))
    return neg_edge_index


class LinkPredictor(pl.LightningModule):
    def __init__(self, dataset, config) -> None:
        super().__init__()
        self.dataset = dataset
        self.data = dataset[0]
        self.config = config
        
        self.encode = RGCNEncoder(self.data.num_nodes, self.config['embeddings_size'], dataset.num_relations// 2, self.config['n_layers'] )
        self.decode = DistMultDecoder(dataset.num_relations // 2, hidden_channels=self.config['embeddings_size'])

        self.loss = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer
    
    # # if forward is called with 2 args it returns score for < subj, obj > for all rels
    # # otherwise it will return return score for subj and all obj
    # def forward(self, batch_subj_index, rel_index, batch_obj_index=None):
        
    #     embeddings = self.embeddings()
    #         # if idx < len(self.layers):
    #         #     embeddings = torch.nn.ReLU(embeddings)

    #     batch_subj_emb = embeddings[batch_subj_index]
    #     batch_rel_emb = self.relations[rel_index]

    #     if batch_obj_index is None:
    #         return  (batch_subj_emb * batch_rel_emb * embeddings.unsqueeze(1) ).sum(-1).T
    #     else:
    #         batch_obj_emb = embeddings[batch_obj_index]
    #         return  (batch_subj_emb * batch_rel_emb * batch_obj_emb ).sum(-1)
        
    def training_step(self, batch, batch_idx):
        edge_index, edge_type = batch
        edge_index = edge_index.T
        z = self.encode(edge_index, edge_type)
        neg_edge_index = negative_sampling(edge_index, self.data.num_nodes)
        pos_out = self.decode(z, edge_index, edge_type)
        neg_out = self.decode(z, neg_edge_index, edge_type)
        out = torch.cat([pos_out, neg_out])
        target = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
        cross_entropy_loss = self.loss(out, target)
        reg_loss = self.encode.embeddings.pow(2).mean()

        loss = cross_entropy_loss + self.config['reg'] * reg_loss
        self.log("loss", loss,)
        return loss
        # batch_index, batch_rel = batch
        # pos_subj_index, pos_obj_index = batch_index.T
        
        # neg_obj_index = torch.randint(self.num_nodes, ( self.config['batch_size'] * self.config['corrupted_obj_ratio'], ) )
        # neg_rel_index = torch.randint(self.num_relations // 2, ( self.config['batch_size'] * self.config['corrupted_rel_ratio'], ) )
        
        # pos_scores = self(pos_subj_index, batch_rel, pos_obj_index )
        # neg_scores_obj = self(pos_subj_index, batch_rel, neg_obj_index ) # corrupt target
        # neg_scores_rel = self(pos_subj_index, neg_rel_index, pos_obj_index ) # corrupt relation
        
        
        # scores = torch.stack((pos_scores, neg_scores_obj, neg_scores_rel)).view(-1)
        # targets = torch.stack((torch.ones_like(pos_scores), torch.zeros_like(neg_scores_obj), torch.zeros_like(neg_scores_rel))).view(-1)
        # print(scores.shape, targets.shape)
        # loss = self.criterion(scores, targets)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        edge_index, edge_type = batch[:256]
        z = self.encode(self.data.edge_index, self.data.edge_type)
        ranks = []
        for i in tqdm(range(edge_type.numel())[:1024]):
            (src, dst), rel = edge_index.T[:, i], edge_type[i]

            # Try all nodes as tails, but delete true triplets:
            tail_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
            for (heads, tails), types in [
                (self.data.train_edge_index, self.data.train_edge_type),
                (self.data.valid_edge_index, self.data.valid_edge_type),
                (self.data.test_edge_index, self.data.test_edge_type),
            ]:
                tail_mask[tails[(heads == src) & (types == rel)]] = False

            tail = torch.arange(self.data.num_nodes)[tail_mask]
            tail = torch.cat([torch.tensor([dst]), tail])
            head = torch.full_like(tail, fill_value=src)
            eval_edge_index = torch.stack([head, tail], dim=0)
            eval_edge_type = torch.full_like(tail, fill_value=rel)

            out = model.decode(z, eval_edge_index, eval_edge_type)
            perm = out.argsort(descending=True)
            rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
            ranks.append(rank + 1)

            # Try all nodes as heads, but delete true triplets:
            head_mask = torch.ones(self.data.num_nodes, dtype=torch.bool)
            for (heads, tails), types in [
                (self.data.train_edge_index, self.data.train_edge_type),
                (self.data.valid_edge_index, self.data.valid_edge_type),
                (self.data.test_edge_index, self.data.test_edge_type),
            ]:
                head_mask[heads[(tails == dst) & (types == rel)]] = False

            head = torch.arange(self.data.num_nodes)[head_mask]
            head = torch.cat([torch.tensor([src]), head])
            tail = torch.full_like(head, fill_value=dst)
            eval_edge_index = torch.stack([head, tail], dim=0)
            eval_edge_type = torch.full_like(head, fill_value=rel)

            out = model.decode(z, eval_edge_index, eval_edge_type)
            perm = out.argsort(descending=True)
            rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
            ranks.append(rank + 1)

        for k in [1,3,10]:
            hits = (torch.tensor(ranks, dtype=torch.float)< k+1).float().mean() 
            self.log(f"hit@{k}", hits, on_epoch=True)
        mrr = (1. / torch.tensor(ranks, dtype=torch.float)).mean()
        self.log(f"mrr", mrr, on_epoch=True)


        # batch_index, batch_rel = batch
        # hits_k  = [1,3,10]
        #     # Test cycle
                
        # batch_subj_index, batch_obj_index = batch_index.T.detach()
        # edge_scores = self( batch_subj_index, batch_rel )  # output has size batch_size * n_nodes
        # edge_scores[:,batch_subj_index] = 0
        # edge_ranks = edge_scores.argsort(-1, descending=True)
        # for k in hits_k:
        #     hits = (edge_ranks.T[:k] == batch_obj_index).float().mean() * k
        #     self.log(f"hit@{k}", hits, on_epoch=True)



data = dataset[0]

train_dataset = torch.utils.data.TensorDataset(data.train_edge_index.T, data.train_edge_type)
test_dataset = torch.utils.data.TensorDataset(data.test_edge_index.T, data.test_edge_type)


train_loader =  torch.utils.data.DataLoader( train_dataset, len(train_dataset), shuffle=True, drop_last=False)
test_loader =  torch.utils.data.DataLoader( test_dataset, len(test_dataset), drop_last=False)

# setup logger
wandb_logger = WandbLogger(project="simple-link-pred",  entity="link-prediction-gnn")

# init model
model = LinkPredictor(dataset, wandb.config)

# init trainer
trainer = pl.Trainer( auto_select_gpus= False, logger= wandb_logger, check_val_every_n_epoch= 3)
# train
trainer.fit(model, train_loader, test_loader)





