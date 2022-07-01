#%%
import os, glob, re

import torch, pytorch_lightning as pl 
from torch.nn import Linear
from torch.nn.modules.loss import BCEWithLogitsLoss

from models.QuestionEncoder import QuestionEncoder
from models.EmbeddingsModel import EmbeddingsModel
from models.GraphEncoder import GraphEncoder

from torch_geometric.nn.conv import RGCNConv
from torch.utils.data import WeightedRandomSampler



class WeightedFocalLoss(torch.nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets, weights=None):
        BCE_loss = BCEWithLogitsLoss( weights, reduction='none')(inputs, targets)
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
    
def load_graph_encoder():
    embeddings_models = glob.glob('./checkpoints/embeddings/**')
    embeddings_models_scores = [re.search("mrr=([0-9].[0-9]*)", m).group(1) for m in embeddings_models]
    embeddings_models = list(sorted(zip(embeddings_models_scores,embeddings_models )))
    _, path = (embeddings_models[0])
    embeddings_model = EmbeddingsModel.load_from_checkpoint(path)
    return embeddings_model.encoder



class QaModel(pl.LightningModule):
    def __init__(self, graph_encoder: GraphEncoder, question_encoder: QuestionEncoder, graph_index: torch.Tensor,  configs: dict) -> None:
        super().__init__()
        
        self.graph_encoder = graph_encoder
        self.question_encoder = question_encoder
        self.graph_embeddings = graph_encoder.nodes_embeddings(torch.arange(self.graph_encoder.n_nodes))
        self.configs = configs 
        
        self._graph_index = graph_index.to(self.configs['device'])
        
        self.hidden_size = 32
        
        self.rgcn_layers = torch.nn.ModuleList(
            [ RGCNConv(self.emb_size*3, self.emb_size, self.graph_encoder.n_relations, ), 
             RGCNConv(self.emb_size, self.hidden_size, self.graph_encoder.n_relations, )]
        )
            
        self.score_candidate = Linear(self.hidden_size, 1)
        self.loss = WeightedFocalLoss()
        self.save_hyperparameters()
        
        self.node_role_embedding = torch.nn.Embedding(2, self.emb_size)

    @property
    def graph_index(self):
        return (self._graph_index[:2],self._graph_index[2])
    
    @property
    def emb_size(self):
        return self.graph_embeddings.shape[1]
    
    def get_name(self):
        # params = ['learning_rate', 'p_dropout', 'reg', 'n_layers']
        # params_string = '|'.join([f"{p}={getattr(self, p)}" for p in params])
        hops = ''.join(self.configs['hops'])
        return f"{self.__class__.__name__}|hops={hops}"
    
    
    def configure_optimizers(self):
        
        no_decay = ["bias", "norm.weight",]
        freeze = [ "question_encoder.model", ]
        params = []
        for n,p in self.named_parameters():
            if any([stopword in n for stopword in freeze]):
                print(f'freeze: {n}')
            elif any([stopword in n for stopword in no_decay]):
                print(f'train_no_reg: {n}')
                params.append({
                    "params": p,
                    "weight_decay": 0,
                })
            else:
                print(f'train: {n}')
                params.append({
                    "params": p,
                    "weight_decay": self.configs['l2'],
                })
        
        optimizer = torch.optim.Adam(
            params, 
            lr=self.configs['lr'], 
            weight_decay=self.configs['l2'])
        
        return optimizer

    def training_step(self, batch, batch_idx):
        questions, roots, answers_mask, hops =  batch
        batch_size, n_nodes  = answers_mask.shape
        
        # Encode Answers
        Z = self(questions, roots, answers_mask)
        # Decode ( score ) Answers 
        scores = self.score_candidate(Z).view(batch_size, -1)
        
        
         
        # Compute weights for the loss
        w_pos =  torch.numel(answers_mask) / ( batch_size * answers_mask.sum() )
        w_neg = torch.numel(answers_mask) / ( batch_size * (1-answers_mask).sum() )
        weights = torch.zeros_like(answers_mask, dtype=torch.float, requires_grad=False)
        weights[answers_mask == 1] = w_pos
        weights[answers_mask == 0] = w_neg
        
        # Compute and log loss 
        # loss = torch.nn.BCEWithLogitsLoss(weights) ( scores, answers_mask.float(), )#.detach()
        
        
        # Compute and log loss 
        candidates_indices = list(WeightedRandomSampler(weights.view(-1), batch_size*50, replacement=True))        
        candidates_scores = scores.view(batch_size*n_nodes, -1)[candidates_indices]
        candidates_answers_mask = answers_mask.view(-1)[candidates_indices]

        # candidates_weights = weights.view(-1)[candidates_indices]
        loss = torch.nn.BCEWithLogitsLoss() ( candidates_scores.view(-1), candidates_answers_mask.float(), )
        
        # loss = self.loss ( scores.view(-1), answers_mask.view(-1).float(), weights.view(-1))
        
        self.log('train/loss', loss )
        
        
        # Compute and log prec 
        prec_pos = (scores[answers_mask == 0] < 0).sum() / (answers_mask == 0).sum()
        prec_neg = (scores[answers_mask == 1] >= 0).sum() / (answers_mask == 1).sum()
        prec_bal = (prec_pos + prec_neg)/2
        prec = (scores == answers_mask).float().mean()
        
        self.log('train/prec', prec)
        self.log('train/prec_bal', prec_bal)
        self.log('train/prec_pos', prec_pos)
        self.log('train/prec_neg', prec_neg)
        
        # Compute and log hits 
        if batch_idx % 10 == 0:
            indices = scores.argsort(1, descending=True)
            
            hits = {1:0, 10:0, 100:0, 1000:0}
            for i,a in zip(indices, answers_mask):
                candidate_hits = i[a.bool()]
                for h,v in hits.items():
                    
                    hits[h] = v + (len([c for c in candidate_hits if c < h]) / min(len(candidate_hits), h)) / batch_size
            
            self.log('train/loss', loss )
            for h,v in hits.items():
                self.log(f'train/hits@{h}', v)

        return loss
    
    def forward(self, questions, roots, answers_mask):
        
        # Encode Answers
        answers_emb = self.graph_encoder.nodes_embeddings(torch.arange(self.graph_encoder.n_nodes,  device=self.device, requires_grad=False))
        
        # Encode Questions
        questions_emb = self.question_encoder(list(questions))
        # Encode Roots
        batch_size, n_nodes = answers_mask.shape
        roots_mask = torch.zeros( (batch_size, n_nodes),dtype=torch.long, device=self.device)
        roots_mask[torch.arange(batch_size, device=self.device), roots] = 1
        node_role_emb = self.node_role_embedding(roots_mask)

        # Pass RGCN, one element at a time 
        _Z = []
        for i, (q_emb, r_emb) in enumerate(zip(questions_emb, node_role_emb),1):
            answers_emb_hat = torch.cat( 
                                        (answers_emb, 
                                         q_emb.repeat((self.graph_encoder.n_nodes,1)), 
                                         r_emb) ,1 )

            z = None
            for  i, rgcn in enumerate(self.rgcn_layers):
                z = answers_emb_hat if z is None else z
                z = rgcn(z, *self.graph_index )
                z = torch.nn.ReLU()(z)
            
            _Z.append(z)

        Z = torch.vstack(_Z)
        return Z
        
    def validation_step(self, batch, batch_idx):

        questions, roots, answers_mask, hops =  batch
        batch_size, n_nodes  = answers_mask.shape
        
        # Encode Answers
        Z = self(questions, roots, answers_mask)
        
        # Decode ( score ) Answers 
        scores = self.score_candidate(Z).view(batch_size, -1)
         
        # Compute weights for the loss
        w_pos =  torch.numel(answers_mask) / batch_size /answers_mask.sum()
        w_neg = torch.numel(answers_mask) / batch_size /(1-answers_mask).sum()
        weights = torch.zeros_like(answers_mask, dtype=torch.float, requires_grad=False)
        weights[answers_mask == 1] = w_pos
        weights[answers_mask == 0] = w_neg
        
        # Compute and log loss 
        # loss = torch.nn.BCEWithLogitsLoss() ( scores, answers_mask.float(), )
        # self.log('val/loss', loss )
        
        # Compute and log prec 
        prec_pos = (scores[answers_mask == 0] < 0).sum() / (answers_mask == 0).sum()
        prec_neg = (scores[answers_mask == 1] >= 0).sum() / (answers_mask == 1).sum()
        prec_bal = (prec_pos + prec_neg)/2
        prec = (scores == answers_mask).float().mean()

        
        self.log('val/prec', prec)
        self.log('val/prec_bal', prec_bal)
        self.log('val/prec_pos', prec_pos)
        self.log('val/prec_neg', prec_neg)
        
        # Compute and log hits 
        indices = scores.argsort(1, descending=True)
        
        hits = {1:0, 10:0, 100:0, 1000:0}
        for i,a in zip(indices, answers_mask):
            candidate_hits = i[a.bool()]
            for h,v in hits.items():
                hits[h] = v + (len([c for c in candidate_hits if c < h]) / min(len(candidate_hits), h)) / batch_size
        

        for h,v in hits.items():
            self.log(f'val/hits@{h}', v)

        # return loss

