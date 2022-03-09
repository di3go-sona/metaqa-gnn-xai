

import sys
from numpy import zeros_like
import pytorch_lightning as pl 
import torch, torch_scatter
from torch.nn import Parameter, ParameterList, ReLU, Embedding, Linear

class LSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,  stack='short-term'):
        super().__init__()

        self.embeddings = Embedding(vocab_size+1, embedding_size)
        self.stack = stack  
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)

        self.linear1 = torch.nn.Linear(64+(64 * (self.stack=='both')), 16)
        self.identity = torch.nn.Identity()
        
        self.forward_stack = torch.nn.Sequential(
          self.identity,
        )

    def forward(self, toks):
      hidden = tuple(torch.rand(1,toks.shape[0],self.hidden_size, device=toks.device) for _ in range(2))

      X = self.embeddings(toks)
      out, hidden = self.lstm(X, hidden)
      if self.stack == 'short-term':
        return self.forward_stack(hidden[0])
      if self.stack == 'long-term':
        return self.forward_stack(hidden[1])
      if self.stack == 'both':  
        return self.forward_stack(torch.concat(hidden,2))



class RGCNEncoder(torch.nn.Module):
    def __init__(self, n_nodes, n_relations, config, embeddings_model=None):
        super().__init__()
        self.N = n_nodes
        self.R = n_relations
        self.E = config['embeddings_size']
        self.L = config['n_layers']
        self.config = config


        self.drop_rgcn = torch.nn.Dropout(self.config['drop_rgcn'])
        self.embeddings = embeddings_model.embeddings if embeddings_model is not None else Parameter(torch.rand(self.N, self.E, requires_grad=True))
        self.rgnc_weights = ParameterList( [ Parameter(torch.rand(n_relations * 2, self.E, self.E, requires_grad=True)) for _ in range(self.L) ] )
        self.rgnc_biases = ParameterList( [ Parameter(torch.rand(n_relations * 2, self.E, requires_grad=True)) for _ in range(self.L) ] )
        self.relu = ReLU()

    def get_messages(self, embeddings_source, index_source, l, inverse=False):
        for r in range(self.R):
            messages = embeddings_source[index_source] @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
            yield messages
                        
                        
    def forward(self, edge_index, edge_type):
        edge_index = edge_index
        edge_type = edge_type
        output = self.embeddings # self.drop_embeddings(self.embeddings)


        for l in range(self.L):
            hidden = output.clone()
            
            for r in range(self.R):
                for inverse in [0,1]:
                    r_dests = edge_index[inverse-0][edge_type == r]
                    r_sources = edge_index[1-inverse][edge_type == r]
                    if l == 0:
                        messages = torch.nn.functional.embedding(r_sources, output) @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]

                    m = torch.vstack(tuple(messages)).to(hidden.device)
                    d = torch.hstack(tuple(r_dests)).to(hidden.device)
                    hidden += torch_scatter.scatter_mean(m, d, dim=0, out=hidden)

            if l + 1 < self.L :
                hidden = self.relu(hidden)
        
            output = self.drop_rgcn(hidden)
            
        
        return output



class TransEDecoder(torch.nn.Module):
    def __init__(self, margin=1.0) -> None:
        super().__init__()
        self.criterion = torch.nn.MarginRankingLoss(margin=margin)
    
    def loss(self, positive_triplets, negative_triplets):
        # assert positive_triplets.size()[1] == 3
        positive_distances = self._distance(positive_triplets)
        # print(positive_distances.shape)
        # assert negative_triplets.size()[1] == 3
        negative_distances = self._distance(negative_triplets)
        # print(negative_distances.shape)
        # return self.loss(positive_distances, negative_distances), positive_distances, negative_distances
        target = torch.tensor([-1], dtype=torch.long, device=positive_triplets.device)
        # print(self.criterion(positive_distances, negative_distances, target))
        # print(self.criterion(positive_distances , negative_distances, target))
        return self.criterion(positive_distances, negative_distances, target)

    
    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        # assert triplets.size()[1] == 3
        src, dst, type = triplets

        return (dst + type - src).norm(p=1, dim=1)
    
    def forward(self, triplets,):
        return self._distance(triplets)
    
    
class DistMultDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def loss(self, positive_triplets, negative_triplets):
        z = self.forward(positive_triplets)
        nz = self.forward(negative_triplets)
        return self.criterion(z, torch.ones_like(z)) + self.criterion(nz, torch.zeros_like(nz))
        
    def forward(self, triplets,):
        # print(triplets.shape)
        src, dst, type = triplets
        return (src*type*dst).sum(-1)


        
class EntitiesEmbeddings(pl.LightningModule):
    # def __init__(self, n_embeddings, embeddings_size, n_relations, lr=0.001, dropout=0.0) -> None:
    #     super().__init__()
    #     self.n_embeddings = n_embeddings
    #     self.embeddings_size = embeddings_size
    #     self.embeddings = torch.nn.Parameter(torch.rand(n_embeddings, embeddings_size, requires_grad=True))
    #     self.rel_embeddings = torch.nn.Parameter(torch.rand(n_relations, embeddings_size, requires_grad=True))
    #     self.lr = lr
    #     self.dropout = torch.nn.Dropout(dropout)
        
    #     self.decoder = DistMultDecoder()
        
    #     self.save_hyperparameters()
    
    def __init__(self, n_embeddings: int, n_relations: int, config: dict ) -> None:
        super().__init__()
        self.n_embeddings = n_embeddings
        self.n_relations = n_relations
        self.embeddings_size = config['embeddings_size']
        self.learning_rate = config['learning_rate']
        self.dropout = torch.nn.Dropout(config['dropout']) 
        self.reg = config['regularization']
        
        self.embeddings = torch.nn.Parameter(torch.rand(n_embeddings, self.embeddings_size, requires_grad=True))
        self.rel_embeddings = torch.nn.Parameter(torch.rand(n_relations, self.embeddings_size, requires_grad=True))
        self.decoder = getattr(sys.modules['models'],config['decoder'])() 
        
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
        # print(batch.shape)
        src, dst, type = batch.long().T
        # print(src, dst, type)

        neg_src, neg_dst = torch.round(torch.rand_like(src, dtype=float)*self.n_embeddings).long()-1, torch.round(torch.rand_like(dst, dtype=float)*self.n_embeddings).long()-1
        src_emb, dst_emb = self.encode(src), self.encode(dst)
        neg_src_emb, neg_dst_emb = self.encode(neg_src), self.encode(neg_dst)
        type_emb = self.rel_encode(type)
        
        pos_triplets = torch.stack((src_emb, dst_emb, type_emb),2).T
        _neg_triplets = torch.stack((src_emb, neg_dst_emb, type_emb),2).T
        __neg_triplets = torch.stack((neg_src_emb, dst_emb, type_emb),2).T
        # neg_triplets = torch.cat((_neg_triplets, __neg_triplets), dim=2)
        # print(neg_triplets.shape)
        # exit()
        
        loss = self.decoder.loss(pos_triplets, _neg_triplets) + self.decoder.loss(pos_triplets, __neg_triplets)

        
        # pos_prec = (pos_triplets.sigmoid() > 0.5).float().mean()
        # neg_prec = (_neg_triplets.sigmoid() < 0.5).float().mean()/2 + (__neg_triplets.sigmoid() < 0.5).float().mean()/2
        # acc = (pos_prec + neg_prec) / 2
        
        # self.log('train_acc', acc.item())
        self.log('train_loss', loss.item())
        # self.log('train_prec_pos', pos_prec.item())
        # self.log('train_prec_neg', neg_prec.item())

        
        return loss #+ self.reg*self.embeddings.square().sum(0).sqrt().mean()
        # kb_z = self.encode_kb(kb_index.T)
        
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

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg)
        return optimizer




class AnswerPredictor(pl.LightningModule):
    def __init__(self, n_nodes: int, n_relations: int, questions_vocab_size:int, config: dict, embeddings = None) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.config = config
        self.questions_vocab_size = questions_vocab_size

        self.encode_kb = RGCNEncoder(self.n_nodes+1, self.n_relations, self.config, embeddings )
        self.encode_question = LSTMEncoder(self.questions_vocab_size, 32, self.config['embeddings_size'], stack='short-term')
        
        self.linear1 =  Linear(self.config['embeddings_size']*3,self.config['embeddings_size'])
        self.linear2 =  Linear(self.config['embeddings_size'], 1)
        self.relu = ReLU()
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        
        self.decode_question =torch.nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2
            
        )
        

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([v for k,v in self.named_parameters() if k not in ['encode_kb.embeddings.embeddings', 'encode_kb.embeddings']], lr=self.config['learning_rate'])
        return optimizer

    def training_step(self, batch, batch_idx):
        kb,  ( qa_questions , qa_nodes) =  batch
        qa_root_nodes, qa_ans_nodes, qa_ans = qa_nodes.T
        kb_index, kb_type = kb

        kb_z = self.encode_kb(kb_index.T, kb_type)
        qa_question_z = self.encode_question(qa_questions)
        
        qa_answers_emb = kb_z[qa_ans_nodes.long()]
        root_emb = kb_z[qa_root_nodes.long()]
        qa_question_emb = qa_question_z.squeeze()

        out = self.decode( qa_answers_emb, qa_question_emb, root_emb).squeeze()


        pos_loss = self.loss(out[qa_ans == 1], qa_ans[qa_ans == 1])
        neg_loss = self.loss(out[qa_ans == 0], qa_ans[qa_ans == 0])
        prec_pos = (out.sigmoid()[qa_ans == 1]>0.5).float().mean()
        prec_neg = (out.sigmoid()[qa_ans == 0]<0.5).float().mean()
        acc = (prec_neg + prec_pos) / 2
        loss =  pos_loss + neg_loss
        
        
        
        self.log('train_acc', acc.item())
        self.log('train_loss', loss.item())
        self.log('train_prec_pos', prec_pos.item())
        self.log('train_prec_neg', prec_neg.item())

        return loss

        
    def validation_step(self, batch, batch_idx):

        kb,  ( qa_questions , qa_nodes) =  batch
        qa_root_nodes, qa_ans_nodes, qa_ans = qa_nodes.T
        kb_index, kb_type = kb

        kb_z = self.encode_kb(kb_index.T, kb_type)
        qa_question_z = self.encode_question(qa_questions)
        qa_answers_emb = kb_z[qa_ans_nodes.long()]
        root_emb = kb_z[qa_root_nodes.long()]
        qa_question_emb = qa_question_z.squeeze()

        out = self.decode( qa_answers_emb, qa_question_emb, root_emb).squeeze()



        pos_loss = self.loss(out[qa_ans == 1], qa_ans[qa_ans == 1])
        neg_loss = self.loss(out[qa_ans == 0], qa_ans[qa_ans == 0])
        prec_pos = (out.sigmoid()[qa_ans == 1]>0.5).float().mean()
        prec_neg = (out.sigmoid()[qa_ans == 0]<0.5).float().mean()
        acc = (prec_neg + prec_pos) / 2
        loss =  pos_loss  + neg_loss   + self.config['reg'] * self.encode_kb.embeddings.pow(2).mean().sqrt()
        
        
        
        self.log('val_acc', acc.item())
        self.log('val_loss', loss.item())
        self.log('val_prec_pos', prec_pos.item())
        self.log('val_prec_neg', prec_neg.item())
        return None

    def decode(self, qa_answers_emb, question_emb, root_emb):

        stacked_input = torch.cat((qa_answers_emb, question_emb, root_emb),dim=1)
        return self.decode_question(stacked_input)


