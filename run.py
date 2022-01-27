#%% 

from gc import callbacks
import torch, torch_geometric, pytorch_lightning as pl
import wandb

from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from rgcn_link_pred import RGCNEncoder, DistMultDecoder
from tqdm import tqdm 
from settings import *

    
def negative_sampling(edge_index, num_nodes):
    # Sample edges by corrupting either the subject or the object of each edge.
    random_mask = torch.rand(edge_index.size(1), device=DEVICE) < 0.5


    neg_edge_index = edge_index.clone()
    neg_edge_index[0, random_mask] = torch.randint(num_nodes, (random_mask.sum(), ), device=DEVICE)
    neg_edge_index[1, ~random_mask] = torch.randint(num_nodes, ((~random_mask).sum(), ), device=DEVICE)
    return neg_edge_index



class LinkPredictor(pl.LightningModule):
    def __init__(self, num_nodes: int, num_relations: int, config: wandb.config) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        
        # self.train_edge_index, self.train_edge_type = train_data.tensors
        # self.test_edge_index, self.test_edge_type = test_data.tensors
        # self.valid_edge_index, self.valid_edge_type = valid_data.tensors

        
        self.config = config
        
        self.encode = RGCNEncoder(self.num_nodes, self.config['embeddings_size'], self.num_relations// 2, self.config['n_layers'] )
        self.decode = DistMultDecoder(self.num_relations // 2, hidden_channels=self.config['embeddings_size'])

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
        
    def training_step(self, train_batch, batch_idx):
        train_edge_index, train_edge_type = train_batch
        train_edge_index = train_edge_index.T
        
        z = self.encode(train_edge_index, train_edge_type)
        

        neg_edge_index = negative_sampling(train_edge_index, self.num_nodes)
        pos_out = self.decode(z, train_edge_index, train_edge_type)
        pos_bce_loss = self.loss(pos_out, torch.ones_like(pos_out)) 
        
        neg_out = self.decode(z, neg_edge_index, train_edge_type)
        neg_bce_loss = self.loss(neg_out, torch.zeros_like(neg_out)) 
        
        # reg_loss = self.encode.embeddings.pow(2).mean() + sum([ p.pow(2).mean() for p in self.encode.rgnc_weights ])
        reg_loss = sum([ p.pow(2).mean() for p in self.parameters() ])

        loss = pos_bce_loss + neg_bce_loss + self.config['reg'] * reg_loss
        self.log("loss", loss.item())
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
    
    def on_validation_epoch_end(self) -> None:
        self.z = None
        
    def validation_step(self, batch, batch_idx):
        (valid_edge_index, valid_edge_type), (train_edge_index, train_edge_type), (all_edge_index, all_edge_type) = batch
        
        if batch_idx == 0:
            self.z = self.encode(train_edge_index.T, train_edge_type)
            
        # print(train_edge_index.shape, valid_edge_index.shape, all_edge_index.shape )
        (valid_src_index, valid_dst_index) = valid_edge_index.T
        obj_scores = self.decode.score_objs(self.z, valid_src_index, valid_edge_type )
        # print(obj_scores)
        ranks = obj_scores.argsort(0, descending=True)
        # ranks = perm[perm == valid_dst_index].float()
        # print(ranks)
        for k in [1,3,10]:
            hits = (ranks[:k]== valid_dst_index).float().mean()  * k
            self.log(f"hit@{k}", hits, on_epoch=True)
        mrr = (1. / ranks.float().mean())
        self.log(f"mrr", mrr, on_epoch=True)

        return 
        # print(src, dst, rel)
        # exit()
        ranks = []
        for i in tqdm(range(valid_edge_index.numel())[:1024]):
            (src, dst), rel = valid_edge_index.T[:, i], valid_edge_type[i]
            # Try all nodes as tails, but delete true triplets:
            tail_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=DEVICE)
            
            (heads, tails), types = all_edge_index.T, all_edge_type
            tail_mask[tails[(heads == src) & (types == rel)]] = False

            nodes = torch.arange(self.num_nodes, device=DEVICE)
            tail = nodes[tail_mask]
            print(dst)
            exit()
            tail = torch.cat([dst, tail])
            head = torch.full_like(tail, fill_value=src, device=DEVICE)
            eval_edge_index = torch.stack([head, tail], dim=0)
            eval_edge_type = torch.full_like(tail, fill_value=rel, device=DEVICE)

            out = model.decode(self.z, eval_edge_index, eval_edge_type)
            perm = out.argsort(descending=True)
            rank = int((perm == 0).nonzero(as_tuple=False).view(-1)[0])
            ranks.append(rank + 1)

            # Try all nodes as heads, but delete true triplets:
            head_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=DEVICE)
            (heads, tails), types = self.all_edge_index.T, self.all_edge_type
            head_mask[heads[(tails == dst) & (types == rel)]] = False

            head = nodes[head_mask]
            head = torch.cat([torch.tensor([src], device=DEVICE), head])
            tail = torch.full_like(head, fill_value=dst, device=DEVICE)
            eval_edge_index = torch.stack([head, tail], dim=0)
            eval_edge_type = torch.full_like(head, fill_value=rel, device=DEVICE)

            out = model.decode(self.z, eval_edge_index, eval_edge_type)
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


class FB15KData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        dataset = torch_geometric.datasets.RelLinkPredDataset( wandb.config['dataset'],
                                                wandb.config['dataset'])
        
        data = dataset[0]
        self.num_nodes = data.num_nodes
        self.num_relations = dataset.num_relations

        self.train_data = TensorDataset(data.train_edge_index.T, data.train_edge_type)
        self.test_data = TensorDataset(data.test_edge_index.T, data.test_edge_type)
        self.valid_data = TensorDataset(data.valid_edge_index.T, data.valid_edge_type)
        self.all_data = TensorDataset( torch.cat((data.train_edge_index.T, data.test_edge_index.T, data.valid_edge_index.T)),
                                    torch.cat((data.train_edge_type, data.test_edge_type, data.valid_edge_type)))

        
    def train_dataloader(self):
        return DataLoader( self.train_data,  
                        len(self.train_data) if wandb.config['batch_size'] == -1 else wandb.config['batch_size'], 
                        drop_last=True, 
                        shuffle=True)
        # return DataLoader( self.train_data,  len(self.train_data), drop_last=True, shuffle=True)
    def val_dataloader(self):
        return CombinedLoader([ 
            DataLoader( self.test_data, batch_size=64, shuffle=True),
            DataLoader( self.train_data, batch_size=len(self.train_data)),
            DataLoader( self.all_data, batch_size=len(self.all_data))
        ], 'max_size_cycle')
        

    def test_dataloader(self):
        return self.val_dataloader()
    

if __name__ == '__main__':

    data = FB15KData()
    wandb_logger = None or WandbLogger(project="simple-link-pred",  entity="link-prediction-gnn")

    # init model
    model = LinkPredictor(data.num_nodes, data.num_relations, wandb.config)

    gpu_args = {'gpus':-1, 'auto_select_gpus':True } if DEVICE == 'cuda' else {}
    # init trainer
    trainer = pl.Trainer( **gpu_args,
                        logger= wandb_logger, 
                        limit_val_batches = wandb.config['limit_val_batches'],
                        check_val_every_n_epoch = wandb.config['check_val_every_n_epoch'],
                        log_every_n_steps=1, 
                        max_epochs=wandb.config['epochs'],
                        callbacks=[checkpoint_callback])
    # train
    trainer.fit(model, data)





