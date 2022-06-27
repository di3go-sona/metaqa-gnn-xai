#%%
from dataset.dataloaders import EmbeddingsData, QAData
from pykeen.nn.modules import DistMultInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss


import wandb, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from torch.nn import Embedding
from transformers import AutoTokenizer, BertModel

#%%
class QAModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, emb_size=200) -> None:
        super().__init__()
        
        self.interaction = DistMultInteraction()
        self.neg_sampler = BasicNegativeSampler(mapped_triples=torch.tensor(kg_data.triplets))
        self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        
        
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_linear = torch.nn.Linear(self.text_model.config.hidden_size, emb_size)
        
        self.nodes_emb = Embedding(kg_data.n_edges, emb_size)
        self.relations_emb = Embedding(kg_data.n_relations, emb_size)
        
        
        
        self.save_hyperparameters()
    
    def get_name(self):
        return 'ComplEx'
    
    
    def _encode_text(self, text):
        toks = model.tokenizer(list(text), return_tensors='pt', padding=True)
        enc = self.text_model(**toks)['pooler_output']
        enc_hat = self.text_linear(enc)
        return enc_hat
         
    def encode(self, triples):
        s,q,t = list(zip(*triples))
        
        # print(s,q,t)
        # print(self.tokenizer(list(q)))
        e =  (
            self.nodes_emb( torch.tensor(s, device=self.device, dtype=torch.long)),
            self._encode_text(q),
            self.nodes_emb( torch.tensor(t, device=self.device, dtype=torch.long))
        ) 
        # print(e)
        return e
        
    def forward(self, triples):
        return self.interaction(*self.encode(triples))
        
    def training_step(self, batch, batch_idx):
        
        corr_batch = self.neg_sampler.corrupt_batch(positive_batch=batch).squeeze(1)
        
        pos_scores = self.forward(batch)
        neg_scores = self.forward(corr_batch)

        loss = self.loss_func.forward(pos_scores, neg_scores)
              
        self.log_dict({'train/loss': loss.item()})
        
        return loss
    

    
    def configure_optimizers(self):
        params = [v for n,v in self.named_parameters() if 'text_model' not in n]
        optim = torch.optim.Adam(params)
        return optim
      

# %%  
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = QAData('dataset', [1], tokenizer, train_batch_size=2048)
    model = QAModel(data)
    
    def collate(batch):
        s,q,t = zip(*batch)
        q = torch.nn.utils.rnn.pad_sequence( q ) 
        return s,q,t
    
    
    # wandb.init( entity='link-prediction-gnn', project="metaqa-embeddings", reinit=True)
    # logger = WandbLogger(log_model=True)

    
                
    # embeddings_checkpoint_callback = ModelCheckpoint(
    #     dirpath='checkpoints/embeddings/',
    #     filename=f'{model.get_name()}'+'|{epoch}|{mrr}'  
    #     )  
    
    # trainer = pl.Trainer( 
    #     accelerator='gpu',
    #     gpus=-1,
    #     callbacks=[embeddings_checkpoint_callback],
    #     logger= logger, 
    #     log_every_n_steps=1,
    #     max_epochs=100)
    
    # trainer.fit(model, data)
    # wandb.finish()




# %%
