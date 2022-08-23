#%%
from typing import List

from dataset.dataloaders import EmbeddingsData, QAData
from pykeen.nn.modules import DistMultInteraction, TransEInteraction, ERMLPInteraction
from pykeen.sampling import BasicNegativeSampler
from pykeen.losses import MarginRankingLoss

import wandb, torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.nn import Embedding
from transformers import AutoTokenizer, BertModel
import torch_geometric
from torch_geometric.utils import k_hop_subgraph

#%%
class JointQAModel(pl.LightningModule):
    def __init__(self, kg_data: EmbeddingsData, 
                 emb_size, 
                 hidden_sizes,
                 lr,
                 aggr = 'max',
                 bert_model = "prajjwal1/bert-mini",
                 fast = False) -> None:
        super().__init__()
        

        self.lr = lr

        # self.qa_interaction = DistMultInteraction()
        # self.kge_interaction = TransEInteraction(1)
        # self.loss_func = MarginRankingLoss(margin=1.0, reduction='mean')
        self.loss_func = torch.nn.CrossEntropyLoss()


        # Text Model
        self.text_model = BertModel.from_pretrained(bert_model)
        
        if self.text_model.config.hidden_size != emb_size:
            self.text_linear_emb = torch.nn.Linear(self.text_model.config.hidden_size, emb_size)
        else:
            self.text_linear_emb = torch.nn.Identity()
            

        
        # Rgcn
        self.hops = kg_data.hops[-1]
        self.layers = len(hidden_sizes) + 1
        if fast:
            self.rgcn1 = torch_geometric.nn.conv.FastRGCNConv(emb_size, int(self.layers == 1) or hidden_sizes[0], kg_data.n_relations*2, bias=False, aggr=aggr)
        else:
            self.rgcn1 = torch_geometric.nn.conv.RGCNConv(emb_size, int(self.layers == 1) or hidden_sizes[0], kg_data.n_relations*2, bias=False, aggr=aggr)
        
        if self.layers > 1:
            if fast:
                self.rgcn2 = torch_geometric.nn.conv.FastRGCNConv(hidden_sizes[0], int(self.layers == 2) or hidden_sizes[1], kg_data.n_relations*2, bias=False, aggr=aggr)
            else:
                self.rgcn2 = torch_geometric.nn.conv.RGCNConv(hidden_sizes[0], int(self.layers == 2) or hidden_sizes[1], kg_data.n_relations*2, bias=False, aggr=aggr)
        if self.layers > 2:
            if fast:
                self.rgcn3 = torch_geometric.nn.conv.FastRGCNConv(hidden_sizes[1], 1, kg_data.n_relations*2, bias=False, aggr=aggr)
            else:
                self.rgcn3 = torch_geometric.nn.conv.RGCNConv(hidden_sizes[1], 1, kg_data.n_relations*2, bias=False, aggr=aggr)
            
        # Cache 
        # self._z_cache = (None, None)
        # self._q_cache = (None, None, None)
        
        # Edges and nodes
        self.edge_index = torch.nn.Parameter( kg_data.get_triples(), requires_grad = False)
        self.nodes_emb = Embedding(kg_data.n_nodes, emb_size)#, max_norm=1)
        # self.src_node_emb = torch.nn.Parameter(torch.rand( (emb_size,)))
        # self.relations_emb = Embedding(kg_data.n_relations, emb_size, max_norm=1)
        
        # self.score_nodes = torch.nn.Linear(hidden_sizes[-1] if self.layers > 1 else emb_size, 1)
        
        # Samplers
        # self.qa_neg_sampler = BasicNegativeSampler(
        #                         mapped_triples= self.edge_index,
        #                         num_negs_per_pos= self.num_negs_per_pos,
        #                         corruption_scheme= ('head',))
        
        # self.kge_neg_sampler = BasicNegativeSampler(
        #                         mapped_triples= self.edge_index,
        #                         num_negs_per_pos= self.num_negs_per_pos,
        #                         corruption_scheme= ('head','tail'))
        self.save_hyperparameters()
        
    @property
    def n_nodes(self):
        return self.nodes_emb.num_embeddings
    
    @property
    def src_node_index(self):
        return torch.tensor( self.nodes_emb.num_embeddings , dtype = torch.long, device = self.device)
    
    def get_name(self):
        return f'QA_RGCN'
        
    def encode_nodes(self,
                        x: torch.Tensor,
                        question_emb: torch.Tensor,
                        edge_index: torch.Tensor,
                        src_index: int):

        if x is None:
            x = self.nodes_emb.weight.clone()
        if edge_index is None:
            edge_index = self.edge_index
        

        nodes_mask = (torch.arange(x.size(0), device = self.device, requires_grad=False) == src_index).unsqueeze(-1).float()
        

        # x =  (nodes_mask) * x +  (1-nodes_mask) * question_emb # reversed version
        # x = (1-nodes_mask) * x +  (nodes_mask) * question_emb # normal version
        x = (nodes_mask) * question_emb # zeroed out version
        

        subset, _edge_index, inv, edge_mask = k_hop_subgraph(src_index.item(), self.hops, edge_index[:,[0,2]].T)
        _edge_type = edge_index[:,1][edge_mask]
        z = self.rgcn1(x, _edge_index, _edge_type)
        
        if self.layers > 1 :
            subset, _edge_index, inv, edge_mask = k_hop_subgraph(src_index.item(), self.hops, edge_index[:,[0,2]].T)
            _edge_type = edge_index[:,1][edge_mask]
            z = self.rgcn2(z.relu(), _edge_index, _edge_type)
        if self.layers > 2 :
            subset, _edge_index, inv, edge_mask = k_hop_subgraph(src_index.item(), self.hops, edge_index[:,[0,2]].T)
            _edge_type = edge_index[:,1][edge_mask]
            z = self.rgcn3(z.relu(), _edge_index, _edge_type)
        
        # self._z_cache = (cache_id, z)
        return z
    
    def encode_question(self, question_toks):
        ''' Encodes the text tokens, input should have shape (seq_len, batch_size), return has shape (batch_size, embedding_size)'''
        
        should_squeeze = False
        if question_toks.dim() == 1:
            should_squeeze = True
            question_toks = question_toks.unsqueeze(-1)
            
        out = self.text_model(question_toks.T)
        out = self.text_linear_emb( out.last_hidden_state[:,0,:]) 
        
        if should_squeeze:
            out = out.squeeze()
            
        return out               
        
    def qa_forward(self, sources, questions):
        '''
        s: `(batch_size, )`
        t: `(batch_size, num_targets, )`
        questions: `(question_len, batch_size)`
        
        return -> ``(batch_size, num_targets, )`'''
        # if targets.dim() == 1:
        #     targets = targets.unsqueeze(0)
        _scores = []
        questions_emb = self.encode_question(questions)

        # print( f's: {sources.shape}, qe:{questions_emb.shape}, q: {questions.shape}')

        for s,  question_emb in zip(sources,  questions_emb):

            
            # question_emb = question_emb.expand((*self.nodes_emb.weight.shape,))
            # z = self.encode_nodes(torch.cat([self.nodes_emb.weight, question_emb] ,dim=-1), None, s)
            z = self.encode_nodes(self.nodes_emb.weight, question_emb, None, s)
            # print(z[t].squeeze(-1).shape,self.score_nodes(z[t]).squeeze(-1).shape)

            _scores.append(
                # self.score_nodes(z[t]).squeeze(-1)
                z.squeeze()
            )
        

        scores = torch.stack(_scores, 0)

        # print( f'scores: {scores.shape}, ({_scores[0].shape})')

        return scores
    

    def qa_forward_triples(self, triples: torch.Tensor, questions: torch.Tensor):
        s, _, _ = triples.T
        if triples.dim() == 3:
            s = s[0]
        return self.qa_forward(s, questions)

    
    def training_step(self, batch, batch_idx) :
        qa_loss = self.qa_training_step(batch['qa'], batch_idx)
        # kge_loss = self.kge_training_step(batch['kge'], batch_idx)
        return qa_loss #+ kge_loss
    
    def validation_step(self, batch, batch_idx) :
        self.qa_validation_step(batch['qa'], batch_idx)
        # self.kge_validation_step(batch['kge'], batch_idx)
        
    def qa_training_step(self, batch, batch_idx):
        triples, questions= batch
        
        # print(triples.shape, questions.shape)
        # corr_triples = self.qa_neg_sampler.corrupt_batch(positive_batch=triples)        
        # print( f"triples: {triples.shape}, corr_triples: {corr_triples.shape}" )
        scores = self.qa_forward( triples[:, 0], questions ).softmax(dim = -1)

        # neg_scores = self.qa_forward_triples(corr_triples, 
        #                         questions)
        # print(pos_scores.shape)

        loss = self.loss_func(scores, triples[:, 2])

        self.log_dict({'train/qa_loss': loss.item()})
        
        return loss
    
    def forward(self, x, edge_index, src_idx, question, **kwargs ):

        triples = torch.stack((edge_index[0], kwargs['relations'] ,edge_index[1])).T

        qa_emb = self.encode_question(question)
        scores = self.encode_nodes(x,qa_emb, triples, src_idx, )

        # scores = self.qa_interaction(nodes_emb[src_idx].unsqueeze(0), qa_emb, nodes_emb).unsqueeze(-1)
        return scores
    
    def qa_validation_step(self, batches, batch_idx):
        res = []
        for batch_type, batch in batches.items():
            src, true_targets_list, questions =  batch
            

            # candidate_targets = torch.arange(
            #                         self.nodes_emb.num_embeddings,
            #                         dtype=torch.long,
            #                         device=self.device
            #                         ).expand(
            #                             (*src.shape, self.nodes_emb.num_embeddings)
            #                             )

            
            scores = self.qa_forward(src,  questions)
            # batch_indices = torch.arange(
            #             len(src),
            #             dtype=torch.long,
            #             device=self.device)
            # scores[torch.stack( (batch_indices, src ), 1).T.tolist() ] = -1

            topk = scores.topk(100, dim=-1)

            res.append(( scores, topk))
            

            # print(f'true_targets: {true_targets_list.shape}, topk: {topk.indices.shape}')
            for true_indices, topk_indices in zip(true_targets_list.T, topk.indices):

                for k in [1,3,10,100]:

                    hits = torch.isin(topk_indices[:k], true_indices).sum().item()
                    count = min(k, (true_indices > -1).sum().item())
                    
                    perc_hits = hits / count
                    self.log(f'{batch_type}/hits_at_{k}', perc_hits, on_epoch=True)
                    
        return res
        
    def kge_encode(self, s,r,t ):

        emb = (
            self.nodes_emb(s), 
            self.relations_emb(r), 
            self.nodes_emb(t), 
        ) 
        

        return emb
        
    def kge_forward(self, s, r, t):
        return self.kge_interaction(*self.kge_encode(s, r, t))
        
    def kge_training_step(self, batch, batch_idx):
        
        corr_batch = self.kge_neg_sampler.corrupt_batch(positive_batch=batch)      
        

        pos_scores = self.kge_forward(*batch.T)
        neg_scores = self.kge_forward(*corr_batch.T)

        loss = self.loss_func(pos_scores, neg_scores)
        
        self.log_dict({'train/kge_loss': loss.item()})
        return loss

    def configure_optimizers(self):
        to_train = [
            'nodes_emb.weight',
            # 'src_node_emb'
            
            'rgcn1.weight',
            'rgcn1.root',
            'rgcn1.bias',
            
            'rgcn2.weight',
            'rgcn2.root',
            'rgcn2.bias',
            
            'rgcn3.weight',
            'rgcn3.root',
            'rgcn3.bias',]
        
        params = []
        for n,v in self.named_parameters():
            
            if n in to_train or n.startswith('text') or n.startswith('qa_interaction') or n.startswith('score_nodes'):
                params.append(v)
            else:
                v.requires_grad = False

        optim = torch.optim.Adam(params, lr=self.lr)
        return optim

#%%  
import click
from pytorch_lightning.trainer.supporters import CombinedLoader

@click.command()
@click.option('--emb-size', default=64, type=int)
@click.option('--hidden-size', default='32', type=str)
@click.option('--lr', default=0.0001, type=float)
@click.option('--ntm', default=False, is_flag=True)
@click.option('--kge-train-batch-size', default=1024, type=int)
@click.option('--qa-train-batch-size', default=4, type=int)
@click.option('--val-batch-size', default=128, type=int)
@click.option('--accumulate-train-batches', default=32, type=int)
@click.option('--limit-val-batches', default=100, type=int)
@click.option('--limit-train-batches', default=4096, type=int)
@click.option('--hops', default=1, type=int)
@click.option('--aggr', default='mean', type=str)
@click.option('--fast', is_flag=True)
@click.option('--patience', default=3, type=int)
@click.option('--epochs', default=500, type=int)
def train( emb_size, hidden_size, lr, ntm, kge_train_batch_size, qa_train_batch_size, val_batch_size, accumulate_train_batches, limit_val_batches, limit_train_batches, hops, aggr, fast, patience, epochs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    kge_data = EmbeddingsData('dataset', train_batch_size= kge_train_batch_size, val_batch_size= val_batch_size)
    qa_data = QAData('dataset', [hops], tokenizer, train_batch_size= qa_train_batch_size, val_batch_size= val_batch_size, use_ntm= ntm)

    hidden_size = [int(i) for i in hidden_size.split('|') ] if len (hidden_size) > 0 else []
    model = JointQAModel(qa_data, emb_size, hidden_size, lr, fast=fast, aggr=aggr)

    wandb.init( entity='link-prediction-gnn', project="metaqa-qa", reinit=True)
    logger = WandbLogger(log_model=True)
                
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/qa/{hops}-hops/',
        filename=f'{model.get_name()}|{">".join([str(i) for i in [emb_size, *hidden_size]])}>1'+'|{epoch}'  
        )  
    stopping_callback = EarlyStopping(monitor="val/hits_at_1", min_delta=0.00, patience=patience, verbose=False, mode="max")
    
    trainer = pl.Trainer( 
        accelerator= 'gpu',
        gpus= -1,
        callbacks= [checkpoint_callback, stopping_callback],
        logger= logger, 
        log_every_n_steps= 1,
        limit_val_batches= limit_val_batches,
        limit_train_batches= limit_train_batches,
        accumulate_grad_batches= accumulate_train_batches,
        val_check_interval= 1.0, 
        check_val_every_n_epoch= 1,
        max_epochs= epochs)
    
    train_loader = CombinedLoader(
        {
            'qa': qa_data.train_dataloader(),
            'kge': kge_data.train_dataloader()
        }, 'max_size_cycle'
    )
    val_loader = CombinedLoader(
        {
            'qa': qa_data.val_dataloader(),
            'kge': kge_data.val_dataloader()
        }, 'max_size_cycle'
    )
    
    trainer.fit(model, 
                train_dataloaders= train_loader, 
                val_dataloaders= val_loader)
    wandb.finish()
    
if __name__ == '__main__':
    train()
# %%
