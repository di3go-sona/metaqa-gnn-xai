
import torch
import  pytorch_lightning as pl
from transformers import BertModel
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn.conv import RGCNConv, FastRGCNConv

from dataloaders import QAData

#%%
class RGCNQA(pl.LightningModule):
    def __init__(self, 
                 kg_data: QAData, 
                 layer_sizes,
                 decompose,
                 lr,
                 l2,
                 root,
                 zeroed_nodes,
                 aggr,
                 bert_model,
                 fast, 
                 bias=False,
                 concat_layers = False,
                 concat_embeddings = None,
                 embeddings_model = None) -> None:
        super().__init__()
        
        assert(len(layer_sizes) > 0)
        

        self.lr = lr
        self.l2 = l2
        self.aggr = aggr
        self.bias = bias
        self.root = root
        self.zeroed = zeroed_nodes
        self.layers = layer_sizes
        self.decompose = decompose or None
        self.n_layers = len(layer_sizes)
        self.emb_size = self.layers[0]
        self.n_nodes  = kg_data.n_nodes
        self.hops = kg_data.hops[-1]
        self.concat_layers = concat_layers
        self.concat_embeddings = concat_embeddings
        
        self.embeddings_model = embeddings_model
        
        self.loss_func = torch.nn.CrossEntropyLoss()


        # TEXT MODEL
        self.text_model = BertModel.from_pretrained(bert_model)
        
        # if Bert last layer size is different from rgcn embedding size then insert an intermediate linear layer
        if self.text_model.config.hidden_size != self.emb_size:
            self.text_head = torch.nn.Linear(self.text_model.config.hidden_size, self.emb_size)
        else:
            self.text_head = torch.nn.Identity()
        
        # RGCN MODEL
        if self.zeroed : 
            self.nodes_emb =  torch.nn.Parameter( torch.zeros(self.n_nodes, self.emb_size), requires_grad=False )
        else: 
            self.nodes_emb = torch.nn.Parameter( torch.randn(self.n_nodes, self.emb_size) )
        
        if embeddings_model:
            self.nodes_pretrained_emb = embeddings_model.nodes_emb.weight
        
        if self.n_layers > 1:
            size_in = self.layers[0]
            size_out = self.layers[1]
            if concat_embeddings in ['all', 'all+head', 'first', ]:
                size_in += self.nodes_pretrained_emb.size(-1)
            if fast:
                self.rgcn1 = FastRGCNConv(size_in, size_out, kg_data.n_relations*2, bias=bias, num_bases= self.decompose and kg_data.n_relations, aggr=aggr)
            else:
                self.rgcn1 = RGCNConv(size_in, size_out, kg_data.n_relations*2, bias=bias, num_bases= self.decompose and kg_data.n_relations, aggr=aggr)
            
        if self.n_layers > 2:
            size_in = self.layers[1]
            size_out = self.layers[2]
            if self.concat_embeddings in ['all', 'all+head']:
                size_in += self.nodes_pretrained_emb.size(-1)
            if fast:
                self.rgcn2 = FastRGCNConv(size_in, size_out, kg_data.n_relations*2, bias=bias, num_bases= self.decompose and kg_data.n_relations, aggr=aggr)
            else:
                self.rgcn2 = RGCNConv(size_in, size_out, kg_data.n_relations*2, bias=bias, num_bases= self.decompose and kg_data.n_relations, aggr=aggr)
        
        if self.n_layers > 3:
            size_in = self.layers[2]
            size_out = self.layers[3]
            if self.concat_embeddings in ['all', 'all+head']:
                size_in += self.nodes_pretrained_emb.size(-1)
            if fast:
                self.rgcn3 = FastRGCNConv(size_in, size_out, kg_data.n_relations*2, bias=bias, num_bases= self.decompose and kg_data.n_relations, aggr=aggr)
            else:
                self.rgcn3 = RGCNConv(size_in, size_out, kg_data.n_relations*2, bias=bias, num_bases= self.decompose and kg_data.n_relations, aggr=aggr)
        
        # if Rgcn last layer size is different from 1 then insert an intermediate linear layer
        if self.concat_layers:
            if self.concat_embeddings in ['head', 'all+head']:
                self.rgcn_head = torch.nn.Linear(
                    sum(self.layers)
                    + 
                    self.nodes_pretrained_emb.size(-1), 1)  
            else:
                self.rgcn_head = torch.nn.Linear(sum(self.layers), 1)  
        elif size_out != 1:
            if self.concat_embeddings in ['head', 'all+head']:
                self.rgcn_head = torch.nn.Linear(
                    self.layers[-1]
                    + 
                    self.nodes_pretrained_emb.size(-1), 1)  
            else:
                self.rgcn_head = torch.nn.Linear (self.layers[-1], 1) 
        else:
            self.rgcn_head = torch.nn.Identity()
            
        # Copy edge index to save time on message passing 
        self.edge_index = torch.nn.Parameter( kg_data.get_triples(), requires_grad = False)
        self.save_hyperparameters()

    
    
    def get_name(self):
        name = 'RGCNQA'
        name += f'|{self.hops}_hops'
        name += '|'+'>'.join([str(i) for i in self.layers])
        if self.concat_layers:
            name += 'C'
        name += f'|lr={self.lr}'
        name += f'|l2={self.l2}'
        name += f'|{self.aggr}_pool'
        if self.zeroed:
            name += '|zeroed'
        if not self.bias:
            name += '|no_bias'
        if not self.root:
            name += '|no_root'
        if  self.decompose:
            name += '|decomposed'
        if  self.concat_embeddings:
            name += f'|concat={self.concat_embeddings}'
        return name
    
    def on_fit_start(self) -> None:
        self.nodes_emb = self.nodes_emb.to(self.device)
        
    def _encode_nodes(self,
                        x: torch.Tensor,
                        question_emb: torch.Tensor,
                        edge_index: torch.Tensor,
                        src_index: int):

        # Retrieve x, if zeroed flag is activated these are all zeros, otherwise nodes_emb
        if x is None:
            x = self.nodes_emb
        if edge_index is None:
            edge_index = self.edge_index
        
        
        # Compute a binary mask with all zeros except the src node index
        src_index_mask = (torch.arange(x.size(0), device = self.device, requires_grad=False) == src_index).unsqueeze(-1).float()
        # Sum question to x
        z = x + (src_index_mask * question_emb )
        if self.concat_layers:
            layers = [z]

        if  isinstance(src_index, torch.Tensor):
            src_index = src_index.item()
        subset, _edge_index, inv, edge_mask = k_hop_subgraph(src_index, self.hops, edge_index[:,[0,2]].T)
        _edge_type = edge_index[:,1][edge_mask]


        if self.n_layers > 1:

            if  self.concat_embeddings in ['all', 'all+head', 'first'] and self.layers[0] != 1:
                z = torch.concat( [z, self.nodes_pretrained_emb], axis=-1)

            z = self.rgcn1(z, _edge_index, _edge_type)

            if self.concat_layers:
                layers.append(z)
        
        if self.n_layers > 2 :
            if self.concat_embeddings in ['all', 'all+head'] and self.layers[1] != 1:
                z = torch.concat( [z, self.nodes_pretrained_emb], axis=-1)
            z = self.rgcn2(z.relu(), _edge_index, _edge_type)
            if self.concat_layers:
                layers.append(z)
            
        if self.n_layers > 3 :
            if self.concat_embeddings in ['all', 'all+head'] and self.layers[2] != 1:
                z = torch.concat( [z, self.nodes_pretrained_emb], axis=-1)
            z = self.rgcn3(z.relu(), _edge_index, _edge_type)
            if self.concat_layers:
                layers.append(z)
        
        if self.concat_embeddings in ['head', 'all+head']:
            z = torch.concat( [z, self.nodes_pretrained_emb], axis=-1)
            
        if self.concat_layers:
            out = self.rgcn_head(
                torch.concat( layers, -1)
            )
        else:
            out = self.rgcn_head(z)
            
        return out
    
    def _encode_question(self, question_toks):
        ''' Encodes the text tokens, input should have shape (seq_len, batch_size), return has shape (batch_size, embedding_size)'''
        
        should_squeeze = False
        if question_toks.dim() == 1:
            should_squeeze = True
            question_toks = question_toks.unsqueeze(-1)
            
        out = self.text_model(question_toks.T)
        out = self.text_head( out.last_hidden_state[:,0,:]) 
        
        if should_squeeze:
            out = out.squeeze()
            
        return out               
        
    def qa_forward(self, sources, questions):
        '''
        sources: `(batch_size, )`
        questions: `(question_len, batch_size)`
        
        return -> ``(batch_size, num_targets, )`'''

        _scores = []
        questions_emb = self._encode_question(questions)
        
        for s,  question_emb in zip(sources,  questions_emb):

            z = self._encode_nodes(self.nodes_emb, question_emb, None, s).squeeze()
            _scores.append( z )
        

        scores = torch.stack(_scores, 0)

        return scores

    # def qa_forward_triples(self, triples: torch.Tensor, questions: torch.Tensor):
    #     s, _, _ = triples.T
    #     if triples.dim() == 3:
    #         s = s[0]
    #     return self.qa_forward(s, questions)

    
        
    def training_step(self, batch, batch_idx):
        triples, questions = batch['qa']

        scores = self.qa_forward( triples[:, 0], questions ).softmax(dim = -1)

        loss = self.loss_func(scores, triples[:, 2])

        self.log_dict({'train/qa_loss': loss.item()})
        
        return loss
    
    def forward(self, x, edge_index,  **kwargs ):

        triples = torch.stack((edge_index[0], kwargs['relations'] ,edge_index[1])).T

        qa_emb = self._encode_question(kwargs['question'])
        scores = self._encode_nodes(x,qa_emb, triples, kwargs['src_idx'], )

        return scores
    
    def validation_step(self, batches, batch_idx):
        res = []
        # batch_type can be either train or val
        for batch_type, batch in batches['qa'].items():
            src, true_targets_list, questions =  batch
            
            # forward model
            scores = self.qa_forward(src,  questions)
            
            topk = scores.topk(100, dim=-1)

            res.append(( scores, topk))
            
            for true_indices, topk_indices in zip(true_targets_list.T, topk.indices):

                for k in [1,3,10,100]:

                    hits = torch.isin(topk_indices[:k], true_indices).sum().item()
                    count = min(k, (true_indices > -1).sum().item())
                    
                    perc_hits = hits / count
                    self.log(f'{batch_type}/hits_at_{k}', perc_hits, on_epoch=True)
        return res
    
    def evaluate_batch(self, batch, k=1):
        src, true_targets_list, questions =  batch
        
        # forward model
        scores = self.qa_forward(src,  questions)
        
        topk = scores.topk(100, dim=-1)

        # print(src.shape, true_targets_list.shape, questions.shape)
        
        for true_indices, topk_indices in zip(true_targets_list.T, topk.indices):
            
            # print(true_indices[true_indices>0].shape, topk_indices.shape)


            hits = torch.isin(topk_indices[:k], true_indices).sum().item()
            count = min(k, (true_indices > -1).sum().item())
            
            perc_hits = hits / count
            yield perc_hits

    def configure_optimizers(self):
        freeze = [ 
                  'nodes_emb'
                  'text_model'
                  ] if self.zeroed else []

        no_norm = [
                    'bias'
                    ]
        params = []
        
        for name , param in self.named_parameters():
            
            if any( w in name for w in freeze ):
                param.requires_grad = False
            
            elif any( w in name for w in no_norm ):
                params.append(
                    {
                        'params': param, 'weight_decay' : 0
                    }
                )

            else:
                params.append(
                    {
                        'params': param, 'weight_decay' : self.l2
                    }
                )
                
        return torch.optim.Adam(params, lr=self.lr)

# %%
