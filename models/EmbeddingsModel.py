       
import torch, pytorch_lightning as pl
import sys

from .GraphEncoder import * 
from .GraphDecoders import *
 
from datetime import datetime


class EmbeddingsModel(pl.LightningModule):
    def __init__(self, n_nodes: int, n_relations: int, config: dict ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        
        self.embeddings_size = config['embeddings_size']
        self.learning_rate = config['learning_rate']
        self.reg = config['regularization']
        
        # self.embeddings = torch.nn.Parameter(torch.rand(n_nodes, self.embeddings_size, requires_grad=True))
        # self.rel_embeddings = torch.nn.Parameter(torch.rand(n_relations, self.embeddings_size, requires_grad=True))
        # if config['decoder'] == 'DistmultDecoder':
        #     decoder_model = DistMultDecoder
        
        # elif config['decoder'] == 'TransEDecoder':
        #     decoder_model = TransEDecoder
        # else:
        #     assert(f'unknown decoder \'{config["decoder"]}\'')
        # decoder_model = 
        self.encoder = GraphEncoder(n_nodes, n_relations, self.embeddings_size)
        self.decoder = (DistMultDecoder if config['decoder'] == 'DistmultDecoder' else TransEDecoder)()
        self.save_hyperparameters()


    def get_name(self):
        return f"EmbeddingsModel|{self.decoder.__class__}|{datetime.now()}"
        
        
    def training_step(self, batch, batch_idx):
        

        triplets_index = batch
        src, dst, rel = triplets_index.T
        
        corrupted_src = torch.randint_like(src, self.n_nodes)
        corrupted_dst = torch.randint_like(dst, self.n_nodes)
        
        corrupted_src_triplets_index = torch.stack(
            (corrupted_src, dst, rel),1
        )
        
        corrupted_dst_triplets_index = torch.stack(
            (src, corrupted_dst, rel),1
        )
        
        # print((triplets_index.shape))
        # print((corrupted_src_triplets_index.shape))
        # print((corrupted_dst_triplets_index.shape))

        # neg_src, neg_dst = torch.round(torch.rand_like(src, dtype=float)*self.n_embeddings).long()-1, torch.round(torch.rand_like(dst, dtype=float)*self.n_embeddings).long()-1
        
        # src_emb, dst_emb = self.encode(src), self.encode(dst)
        # neg_src_emb, neg_dst_emb = self.encode(neg_src), self.encode(neg_dst)
        # type_emb = self.rel_encode(type)
        
        # pos_triplets = batch
        # corr_head_triplets = torch.stack((src_emb, neg_dst_emb, type_emb),2).T
        # __neg_triplets = torch.stack((neg_src_emb, dst_emb, type_emb),2).T
        
        # pos_loss = self.decoder.loss()
        # neg_src_loss = self.decoder.loss( self.encoder(triplets_index.T), self.encoder(corrupted_src_triplets_index.T))
        loss = self.decoder.loss( self.encoder(triplets_index.T), self.encoder(corrupted_dst_triplets_index.T))
        # loss = neg_src_loss + loss
        # self.log('train/neg_src_loss', neg_src_loss.item())
        self.log('train/loss', loss.item())
        # self.log('train/neg_src_loss', neg_src_loss.item())
        
        return  loss
        
    def validation_step(self, batch, batch_idx):


        triplets_index, mask = batch
        batch_size, _ = triplets_index.shape
        
        (src, dst, rel) = triplets_index.T
        
        val_src = src.repeat((self.n_nodes,1)).flatten()
        val_dst = torch.arange(0,self.n_nodes, device=triplets_index.device, dtype=torch.int32).repeat((1,batch_size)).flatten()
        val_rel = rel.repeat((self.n_nodes,1)).flatten()
        
        # print(val_src.shape, val_dst.shape, val_rel.shape)
        val_triplets =  torch.vstack((
            val_src,
            val_dst,
            val_rel
        ))
       


        Z = self.encoder(val_triplets)

        scores = self.decoder(Z).reshape(batch_size, -1).T
        
        

        # loss = self.decoder.loss( scores.T, mask)
        # self.log('val/loss', loss.item())
        
        scores[mask.T] = 0

        indices = scores.argsort(0, descending=True)#isinstance(self.decoder, DistMultDecoder))

        position = torch.arange(
            self.n_nodes,
            device=val_triplets.device).reshape(-1,1).repeat(1,batch_size) == dst.reshape(1,-1)

        ranks = indices[position]+1
        
        

        
        hits = [1,5,10,100]
        for k in hits:
            hit = (ranks <= k+1).float().mean().item()
            self.log(f'val/hit@{k}', hit)
        mrr = (1/ranks).mean()
        self.log(f'val/mrr', mrr)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        return optimizer

