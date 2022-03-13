       
import torch, pytorch_lightning as pl
import sys

from .EmbeddingsDecoders import * 



class EmbeddingsModel(pl.LightningModule):
    metric = 'hit@10'
        
    def __init__(self, n_embeddings: int, n_relations: int, config: dict ) -> None:
        super().__init__()
        self.n_embeddings = n_embeddings
        self.n_relations = n_relations
        self.embeddings_size = config['embeddings_size']
        self.learning_rate = config['learning_rate']
        self.p_dropout = config['dropout']
        self.dropout = torch.nn.Dropout(config['dropout']) 
        self.reg = config['regularization']
        
        self.embeddings = torch.nn.Parameter(torch.rand(n_embeddings, self.embeddings_size, requires_grad=True))
        self.rel_embeddings = torch.nn.Parameter(torch.rand(n_relations, self.embeddings_size, requires_grad=True))
        self.decoder = DistMultDecoder if config['decoder'] == 'DistmultDecoder' else TransEDecoder
        
        self.save_hyperparameters()


    def cname(self):
        params = ['embeddings_size', 'learning_rate', 'dropout', 'reg']
        params_string = '|'.join([f"{p}={getattr(self, p)}" for p in params])
        return f"{self.__class__.__name__}|{self.decoder.__class__.__name__}|{params_string}"
        
    def encode(self, edge_index):
        output = self.dropout(self.embeddings[edge_index])
        return output
    
    def rel_encode(self, edge_type):
        output = self.dropout(self.rel_embeddings[edge_type])
        return output
        
    def training_step(self, batch, batch_idx):

        src, dst, type = batch.long().T


        neg_src, neg_dst = torch.round(torch.rand_like(src, dtype=float)*self.n_embeddings).long()-1, torch.round(torch.rand_like(dst, dtype=float)*self.n_embeddings).long()-1
        src_emb, dst_emb = self.encode(src), self.encode(dst)
        neg_src_emb, neg_dst_emb = self.encode(neg_src), self.encode(neg_dst)
        type_emb = self.rel_encode(type)
        
        pos_triplets = torch.stack((src_emb, dst_emb, type_emb),2).T
        _neg_triplets = torch.stack((src_emb, neg_dst_emb, type_emb),2).T
        __neg_triplets = torch.stack((neg_src_emb, dst_emb, type_emb),2).T

        
        loss = self.decoder.loss(pos_triplets, _neg_triplets) + self.decoder.loss(pos_triplets, __neg_triplets)

        self.log('train_loss', loss.item())
        
        return loss 
        
    def validation_step(self, batch, batch_idx):


        src, dst, type = batch.long().T


        src_emb = self.encode(src)
        dst_embs = self.encode(torch.arange(self.n_embeddings))

        type_emb = self.rel_encode(type)
        batch_size = src_emb.shape[0]

        a,b,c = (
            src_emb.repeat(self.n_embeddings,1,1,).reshape(-1, self.embeddings_size),
            type_emb.repeat(self.n_embeddings,1,1,).reshape(-1, self.embeddings_size),
            dst_embs.unsqueeze(1).repeat(1,batch_size,1).reshape(-1, self.embeddings_size)
        )
        
        triplets = torch.stack((a.reshape(-1, self.embeddings_size),b.reshape(-1, self.embeddings_size),c.reshape(-1, self.embeddings_size)),)
        
        out = self.decoder(triplets).reshape(self.n_embeddings, batch_size)
        indices = out.argsort(0, descending=isinstance(self.decoder, DistMultDecoder))
        position = torch.arange(self.n_embeddings, device=triplets.device).reshape(-1,1).repeat(1,batch_size) == dst.reshape(1,-1)

        ranks = indices[position]+1
        
        hits = [1,5,10,100]
        for k in hits:
            hit = (ranks <= k).float().mean().item()
            self.log(f'hit@{k}', hit)
        mrr = (1/ranks).mean()
        self.log(f'mrr', mrr)
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        return optimizer

