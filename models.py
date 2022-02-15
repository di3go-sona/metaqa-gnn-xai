
from numpy import zeros_like
import pytorch_lightning as pl 
import torch, torch_scatter
from torch.nn import Parameter, ParameterList, ReLU, Embedding, Linear

class LSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,  stack='short-term'):
        super().__init__()

        # self.add_module ('encoder', encoder)

        self.embeddings = Embedding(vocab_size+1, embedding_size)
        self.stack = stack  
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)

        self.linear1 = torch.nn.Linear(64+(64 * (self.stack=='both')), 16)
        self.identity = torch.nn.Identity()
        
        self.forward_stack = torch.nn.Sequential(
          self.identity,
        )


    
    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=0.01)    

    def forward(self, toks):
      hidden = tuple(torch.rand(1,toks.shape[0],self.hidden_size, device=toks.device) for _ in range(2))

      X = self.embeddings(toks)
      # for x in X:
      # print(X.shape)
      out, hidden = self.lstm(X, hidden)
      # print('out', out.shape)
      if self.stack == 'short-term':
        return self.forward_stack(hidden[0])
      if self.stack == 'long-term':
        return self.forward_stack(hidden[1])
      if self.stack == 'both':  
        # print(hidden[0].shape, hidden[1].shape, torch.concat(hidden,2).shape)
        return self.forward_stack(torch.concat(hidden,2))



class RGCNEncoder(torch.nn.Module):
    def __init__(self, n_nodes, n_relations, embeddings_size, n_layers=3):
        super().__init__()
        self.N = n_nodes
        self.R = n_relations
        self.E = embeddings_size
        self.L = n_layers

        self.drop = torch.nn.Dropout(p=0.75)
        # self.embeddings = torch.nn.Embedding(n_nodes, embeddings_size )
        self.embeddings = Parameter( torch.rand(n_nodes, embeddings_size, requires_grad=True)) 
        self.rgnc_weights = ParameterList( [ Parameter(torch.rand(n_relations * 2, embeddings_size, embeddings_size, requires_grad=True)) for _ in range(n_layers) ] )
        self.rgnc_biases = ParameterList( [ Parameter(torch.rand(n_relations * 2, embeddings_size, requires_grad=True)) for _ in range(n_layers) ] )
        self.relu = ReLU()

    def get_messages(self, embeddings_source, index_source, l, inverse=False):
        
        
        for r in range(self.R):
            # print(embeddings_source.shape, self.rgnc_weights[l][r + self.R * inverse].shape, self.rgnc_biases[l][r + self.R * inverse].shape)
            messages = embeddings_source[index_source] @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
            # print(messages.shape)
            yield messages
                        
                        
    def forward(self, edge_index, edge_type):
        edge_index = edge_index
        edge_type = edge_type
        output = self.embeddings


        for l in range(self.L):
            hidden = torch.zeros(self.N, self.E, device =edge_index.device)
            for r in range(self.R):
                for inverse in [0,1]:
                    r_dests = edge_index[inverse-0][edge_type == r]
                    r_sources = edge_index[1-inverse][edge_type == r]
                    if l == 0:
                        messages = torch.nn.functional.embedding(r_sources, output) @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
                        # print(output(r_sources).shape, r_sources.shape ,messages.shape)
                        # print(self.rgnc_weights[l][r + self.R * inverse].shape)
                        # print(self.rgnc_biases[l][r + self.R * inverse].shape)
                        # print(messages)

                    # else:
                    #     messages = output[r_sources] @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
                        # print(messages.shape)
                        # print(messages)

                    m = torch.vstack(tuple(messages)).to(hidden.device)
                    d = torch.hstack(tuple(r_dests)).to(hidden.device)
                    hidden += torch_scatter.scatter_mean(m, d, dim=0, out=hidden)

            if l + 1 < self.L :
                hidden = self.relu(hidden)
        
            output = hidden
            
        
        return self.drop(output)






class DistMultDecoder(torch.nn.Module):
    def __init__(self, n_relations, embeddings_size):
        super().__init__()
        self.rel_emb = Parameter(torch.rand(n_relations, embeddings_size, requires_grad=True))

    def score_heads(self, z, head_idx, rel_idx):
        z_tail, z_head = z[head_idx], z
        rel = self.rel_emb[rel_idx]
        return torch.sum(z_tail * rel * z.unsqueeze(1), dim=-1)

    def score_tails(self, z, head_idx, rel_idx):
        z_tail, z_head = z[head_idx], z
        rel = self.rel_emb[rel_idx]
        return torch.sum(z_head * rel * z.unsqueeze(1), dim=-1)
    
    def score_triplet(self, z, edge_index, edge_type):
        tail, dst = edge_index

        return torch.sum(z[tail] * self.rel_emb[edge_type] * z[dst], dim=1)

    def forward(self, z, edge_index, edge_type):
        return  self.score_triplet( z, edge_index, edge_type)



class AnswerPredictor(pl.LightningModule):
    def __init__(self, n_nodes: int, n_relations: int, questions_vocab_size:int, config: dict) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.config = config
        self.questions_vocab_size = questions_vocab_size

        self.encode_kb = RGCNEncoder(self.n_nodes+1, self.n_relations, self.config['embeddings_size'], self.config['n_layers'] )
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def training_step(self, batch, batch_idx):
        kb,  ( qa_questions, qa_questions_len, qa_nodes) =  batch
        qa_roots, qa_ans, qa_neg_ans = qa_nodes.T
        kb_index, kb_type = kb

        kb_z = self.encode_kb(kb_index.T, kb_type)
        qa_question_z = self.encode_question(qa_questions)
        
        qa_answers_emb = kb_z[qa_ans]
        qa_neg_answers_emb = kb_z[qa_neg_ans]
        root_emb = kb_z[qa_roots]
        qa_question_emb = qa_question_z.squeeze()

        out = self.decode( qa_answers_emb, qa_question_emb, root_emb)
        neg_out = self.decode( qa_neg_answers_emb, qa_question_emb, root_emb)

        loss = self.loss(out, torch.ones_like(out)) + self.loss(neg_out, torch.zeros_like(neg_out))
        acc =  ( (out>0.5)).float().mean()/2 + ( (neg_out<0.5)).float().mean()/2
        prec_pos = (out>0.5).float().mean()
        prec_neg = (neg_out<0.5).float().mean()
        
        self.log('train_acc', acc.item())
        self.log('train_loss', loss.item())
        self.log('train_prec_pos', prec_pos.item())
        self.log('train_prec_neg', prec_neg.item())

        return loss

        
    def validation_step(self, batch, batch_idx):

        kb,  ( qa_questions, qa_questions_len, qa_nodes) =  batch
        qa_roots, qa_ans, qa_neg_ans = qa_nodes.T
        kb_index, kb_type = kb

        kb_z = self.encode_kb(kb_index.T, kb_type)
        qa_question_z = self.encode_question(qa_questions)
        
        qa_answers_emb = kb_z[qa_ans]
        qa_neg_answers_emb = kb_z[qa_neg_ans]
        root_emb = kb_z[qa_roots]
        qa_question_emb = qa_question_z.squeeze()

        out = self.decode( qa_answers_emb, qa_question_emb, root_emb)
        # print('decode')
        neg_out = self.decode( qa_neg_answers_emb, qa_question_emb, root_emb)
        # print('decode')
        loss = self.loss(out, torch.ones_like(out)) + self.loss(neg_out, torch.zeros_like(neg_out))
        
        acc =  ( (out>0.5)).float().mean()/2 + ( (neg_out<0.5)).float().mean()/2
        self.log('val_acc', acc.item())
        self.log('val_loss', loss.item())
        return None

    def decode(self, qa_answers_emb, question_emb, root_emb):
        max_answers = qa_answers_emb.shape[0]
        stacked_input = torch.cat((qa_answers_emb, question_emb, root_emb),dim=-1)
        return self.decode_question(stacked_input)

