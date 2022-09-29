       
import torch
import pytorch_lightning as pl

from pykeen.nn.modules import DistMultInteraction, ComplExInteraction, TransEInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss
from torch.nn import Embedding
from dataloaders import EmbeddingsData

class KGEModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, interaction: str, emb_size, lr, negs) -> None:
        super().__init__()
        
        self.lr = lr
        self.negs = negs
        if interaction.lower() == 'distmult':
            self.interaction = DistMultInteraction()
        elif interaction.lower() == 'complex':
            self.interaction = ComplExInteraction()
        elif interaction.lower() == 'transe-p1' or interaction.lower() == 'transe':
            self.interaction = TransEInteraction(1)
        elif interaction.lower() == 'transe-p2':
            self.interaction = TransEInteraction(2)
        else:
            raise Exception(f'Unknow interaction {interaction}')
        self.neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.get_triples()), num_negs_per_pos=negs)
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        self.nodes_emb = Embedding(kg_data.n_nodes, emb_size, max_norm=1)
        self.relations_emb = Embedding(kg_data.n_relations, emb_size)
        

        
        self.save_hyperparameters()
    
    def get_name(self):
        return f'{self.interaction}|{self.nodes_emb.embedding_dim}'
        
    def encode(self, s,r,t ):

        emb = (
            self.nodes_emb(s), 
            self.relations_emb(r), 
            self.nodes_emb(t), 
        ) 
        

        return emb
        
    def forward(self, s, r, t):
        return self.interaction(*self.encode(s, r, t))
        
    def training_step(self, batch, batch_idx):
        

        
        corr_batch = self.neg_sampler.corrupt_batch(positive_batch=batch)      
        

        pos_scores = self.forward(*batch.T)
        neg_scores = self.forward(*corr_batch.T)

        loss = self.loss_func.forward(pos_scores, neg_scores)
        
        self.log_dict({'train/loss': loss.item()})
        return loss
             
    
    def validation_step(self, batch, batch_idx):
        src, rel, true_targets_list =  batch
        
        candidate_indices =  torch.arange(
                        self.nodes_emb.num_embeddings,
                        dtype=torch.long,
                        device=self.device).unsqueeze(-1)
        
        batch_indices = torch.arange(
                        len(src),
                        dtype=torch.long,
                        device=self.device)
        
        scores = self(src, rel, candidate_indices)
        # print(scores.min(dim=0), torch.stack( (src, batch_indices  ).T, 1).shape)
        scores[torch.stack( (src, batch_indices  ), 1).T.tolist() ] = -1

        scores.shape
        # print(torch.stack( (src, batch_indices), 1))
        # print(scores.shape,torch.stack( (src, batch_indices), 1).T )
        top_targets = scores.topk(100, dim=0).indices

    
        for true_targets, pred_targets in zip(true_targets_list.T, top_targets.T):
            for k in [1,3,10,100]:
                hits = torch.isin(pred_targets[:k], true_targets).sum().item()
                count = min(k,  (true_targets > -1).sum().item())
                
                perc_hits = hits / count
                self.log(f'val/hits@{k}', perc_hits, on_epoch=True)

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim
    

      
