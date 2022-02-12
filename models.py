
from numpy import zeros_like
import pytorch_lightning as pl 
import torch, torch_scatter
from torch.nn import Parameter, ParameterList, ReLU, Embedding

class LSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,  stack='short-term'):
        super().__init__()

        # self.add_module ('encoder', encoder)

        self.embeddings = torch.nn.Embedding(vocab_size+2, embedding_size)
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
      hidden = tuple(torch.rand(1,toks.shape[0],self.hidden_size) for _ in range(2))

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

        self.embeddings = torch.nn.Embedding(n_nodes, embeddings_size )
        self.rgnc_weights = ParameterList( [ Parameter(torch.rand(n_relations * 2, embeddings_size, embeddings_size, requires_grad=True)) for _ in range(n_layers) ] )
        self.rgnc_biases = ParameterList( [ Parameter(torch.rand(n_relations * 2, embeddings_size, requires_grad=True)) for _ in range(n_layers) ] )
        self.relu = ReLU()

    def get_messages(self, embeddings_source, index_source, l, inverse=False):
        
        
        for r in range(self.R):
            print(embeddings_source.shape, self.rgnc_weights[l][r + self.R * inverse].shape, self.rgnc_biases[l][r + self.R * inverse].shape)
            messages = embeddings_source[index_source] @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
            print(messages.shape)
            yield messages
                        
                        
    def forward(self, edge_index, edge_type):
        edge_index = edge_index
        edge_type = edge_type
        
        output = self.embeddings

        # print( edge_index.shape, edge_type.shape)
        m_dests = edge_index[0]
        m_sources = edge_index[1]
        for l in range(self.L):
            hidden = torch.zeros(self.N, self.E)
            for r in range(self.R):
                for inverse in [0,1]:
                    r_dests = edge_index[inverse-0][edge_type == r]
                    r_sources = edge_index[1-inverse][edge_type == r]
                    if l == 0:
                        messages = output(r_sources) @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
                        # print(output(r_sources).shape, r_sources.shape ,messages.shape)
                        # print(self.rgnc_weights[l][r + self.R * inverse].shape)
                        # print(self.rgnc_biases[l][r + self.R * inverse].shape)
                        # print(messages)

                    else:
                        messages = output[r_sources] @ self.rgnc_weights[l][r + self.R * inverse] + self.rgnc_biases[l][r + self.R * inverse]
                        # print(messages.shape)
                        # print(messages)

                    m = torch.vstack(tuple(messages)).to(hidden.device)
                    d = torch.hstack(tuple(r_dests)).to(hidden.device)
                    hidden += torch_scatter.scatter_sum(m, d, dim=0, out=hidden)

            if l + 1 < self.L :
                hidden = self.relu(hidden)
        
            output = hidden  
        return output


        exit()

        m = torch.vstack(tuple(messages)).to(hidden.device)
        d = torch.hstack(tuple(dests)).to(hidden.device)

        print(m.shape, d.shape)

        hidden += torch_scatter.scatter_sum(m, d, dim=0, out=hidden)
        if l + 1 < self.L :
            hidden = self.relu(hidden)
        
        output = hidden           
        return output



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

        self.loss = torch.nn.BCEWithLogitsLoss()
        
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        return optimizer

    def training_step(self, batch, batch_idx):
        qa_batch, kb =  batch
 
        kb_index, kb_type = kb[...]
        qa_batch_question, qa_batch_answers, qa_batch_root = qa_batch

        # print(qa_batch_question.shape, qa_batch_answers.shape, qa_batch_root.shape)

        # qa_question, batch_qa_answers, batch_qa_root = (i.squeeze() for i in qa_batch)

        kb_z = self.encode_kb(kb_index.T, kb_type)
        qa_question_z = self.encode_question(qa_batch_question)
        # print(kb_z.shape, qa_z.shape)
        # print(qa_batch_question.shape)
        # print(kb_z[qa_batch_answers].shape, )
        # print(qa_batch_answers)
        # print(kb_z[qa_batch_answers][:,0])
        def negative_sampling():
            for qa_answers, qa_root in zip(qa_batch_answers, qa_batch_root):
                # print(qa_root, qa_answers)
                neg_qa_answers = torch.full_like(qa_answers, -1)
                valid_answers_mask  = qa_answers > -1
                incoming_sources = kb_index[(kb_index[:, 0] == qa_root)]
            
                # print(incoming_sources)
                neg_mask = (incoming_sources[:,1] == qa_answers.unsqueeze(-1)).sum(0) == 0
                
                # print(neg_qa_answers[:neg_mask.sum(-1)].shape, incoming_sources[:,1][neg_mask].shape )
                neg_qa_answers[:neg_mask.sum(-1)] = incoming_sources[:,1][neg_mask]
                # outgoing_sources = kb_index[((kb_index[:, 1] == qa_root).logical_and((kb_index[:, 1] != qa_answers.unsqueeze(-1)).sum(0)) )]
                # print(qa_answers, neg_qa_answers)#, outgoing_sources)
                yield(neg_qa_answers)
        # print(len(list(negative_sampling())))
        neg_qa_batch_answers = torch.vstack(tuple(negative_sampling()))
        # print(answers.shape)
            # print( kb_index[:,0]  != qa_answers.unsqueeze(-1) )

            # neg_qa_answers = kb_index[kb_index[:, 0] == qa_root.logical_and((kb_index[:,0]  != qa_answers.unsqueeze(-1)).sum(0) > 0 )][:,1]
            # print(neg_qa_answers)
            # neg_qa_answers = kb_index[kb_index[:, 1] == qa_root.logical_and((kb_index[:,1]  != qa_answers.unsqueeze(-1)).sum(0) > 0 )][:,0]
            # print(neg_qa_answers)

        neg_qa_answers_emb = kb_z[neg_qa_batch_answers].transpose(0,1)
        qa_answers_emb = kb_z[qa_batch_answers].transpose(0,1)

        question_emb = qa_question_z.squeeze().unsqueeze(0)
        root_emb = kb_z[qa_batch_root].unsqueeze(0)
        # print( qa_answers_emb.shape, question_emb.shape, root_emb.shape)

        out = (( qa_answers_emb * question_emb * root_emb )[qa_batch_answers.T > 0]).sum(0)
        neg_out = (( neg_qa_answers_emb * question_emb * root_emb )[neg_qa_batch_answers.T > 0]).sum(0)

        loss = self.loss(out, torch.ones_like(out)) + self.loss(neg_out, torch.zeros_like(neg_out))
        self.log("train_loss", loss.item())
        acc =  ( (out>0.5)).float().mean()/2 + ( (out<0.5)).float().mean()/2
        self.log('val_acc', acc.item())
        return loss
        print(out, neg_out)
        exit()
            # print(kb_index.shape)
        batch_neg_qa_answers = [ torch.hstack((
            kb_index[kb_index[:, 0] == qa_root.logical_and((kb_index[:,0]  != qa_answers.unsqueeze(-1)).sum(0) > 0 )][:,1], 
            kb_index[kb_index[:, 1] == qa_root.logical_and((kb_index[:,1]  != qa_answers.unsqueeze(-1)).sum(0) > 0 )][:,0] )) 
                for qa_answers, qa_root in zip(qa_batch_answers, qa_batch_root) ]

        print(batch_neg_qa_answers)

        neg_qa_answers_emb = kb_z[batch_neg_qa_answers]
        qa_answers_emb = kb_z[qa_batch_answers]

        print(neg_qa_answers_emb.shape, qa_answers_emb.shape)
            # print(neg_qa_answers.shape, neg_qa_answers)

        
            # tmp_answers = []
            # tmp_y = []
            # for qa_answer, qa_neg_answer in zip(qa_answers, neg_qa_answers):

                
            #     if qa_answer != -1:
            #         # print('id', qa_answer)
            #         # print(self.kb_z[qa_answer].requires_grad)
            #         # print(self.qa_z.squeeze().requires_grad)
            #         # print(self.kb_z[qa_root].requires_grad)
            #         # print( ( self.kb_z[qa_answer] * self.qa_z.squeeze() * self.kb_z[qa_root]).sum())
            #         tmp_answers.append( ( self.kb_z[qa_answer] * self.qa_z.squeeze() * self.kb_z[qa_root]).mean() )
            #         tmp_answers.append( ( self.kb_z[qa_neg_answer] * self.qa_z.squeeze() * self.kb_z[qa_root]).mean() )
            #         tmp_y.append(torch.tensor([1,0], dtype=torch.float))


            # tmp_batches.append(torch.vstack(tmp_answers))
        # out = torch.vstack(tmp_answers)
        # y = torch.hstack(tmp_y)
        # acc =  ( (out.detach() >0.5) == y).float().mean()
        # print(out.squeeze().shape,(y.shape))
        # loss = self.loss(out.squeeze(),y )
        loss = torch.tensor(0, requires_grad=True, dtype=torch.float)
        # print(loss)
        # print(loss)
        # print(self.kb_z.shape, self.qa_z.shape)
        self.log("train_loss", loss.item())
        # self.log('train_acc', acc.item())
        
        return loss

        neg_edge_index = self.negative_sampling(decode_batch_index.T, self.n_nodes)
        pos_out = self.decode(z, decode_batch_index.T, decode_batch_type)
        pos_bce_loss = self.loss(pos_out, torch.ones_like(pos_out)) 
        
        print(z)
        exit()

        neg_out = self.decode(z, neg_edge_index, decode_batch_type)
        neg_bce_loss = self.loss(neg_out, torch.zeros_like(neg_out)) 
        
        # reg_loss = sum([ p.pow(2).mean() for p in self.parameters() ])

        # loss = pos_bce_loss + neg_bce_loss + self.config['reg'] * reg_loss
        # self.log("loss", loss.item())
        
        # return loss
        
    def validation_step(self, batch, batch_idx):
        qa_batch, kb =  batch
 
        kb_index, kb_type = kb[...]
        qa_batch_question, qa_batch_answers, qa_batch_root = qa_batch

        # print(qa_batch_question.shape, qa_batch_answers.shape, qa_batch_root.shape)

        # qa_question, batch_qa_answers, batch_qa_root = (i.squeeze() for i in qa_batch)

        kb_z = self.encode_kb(kb_index.T, kb_type)
        qa_question_z = self.encode_question(qa_batch_question)
        # print(kb_z.shape, qa_z.shape)
        # print(qa_batch_question.shape)
        # print(kb_z[qa_batch_answers].shape, )
        # print(qa_batch_answers)
        # print(kb_z[qa_batch_answers][:,0])
        def negative_sampling():
            for qa_answers, qa_root in zip(qa_batch_answers, qa_batch_root):
                # print(qa_root, qa_answers)
                neg_qa_answers = torch.full_like(qa_answers, -1)
                valid_answers_mask  = qa_answers > -1
                incoming_sources = kb_index[(kb_index[:, 0] == qa_root)]
            
                # print(incoming_sources)
                neg_mask = (incoming_sources[:,1] == qa_answers.unsqueeze(-1)).sum(0) == 0
                
                # print(neg_qa_answers[:neg_mask.sum(-1)].shape, incoming_sources[:,1][neg_mask].shape )
                neg_qa_answers[:neg_mask.sum(-1)] = incoming_sources[:,1][neg_mask]
                # outgoing_sources = kb_index[((kb_index[:, 1] == qa_root).logical_and((kb_index[:, 1] != qa_answers.unsqueeze(-1)).sum(0)) )]
                # print(qa_answers, neg_qa_answers)#, outgoing_sources)
                yield(neg_qa_answers)
        # print(len(list(negative_sampling())))
        neg_qa_batch_answers = torch.vstack(tuple(negative_sampling()))
        # print(answers.shape)
            # print( kb_index[:,0]  != qa_answers.unsqueeze(-1) )

            # neg_qa_answers = kb_index[kb_index[:, 0] == qa_root.logical_and((kb_index[:,0]  != qa_answers.unsqueeze(-1)).sum(0) > 0 )][:,1]
            # print(neg_qa_answers)
            # neg_qa_answers = kb_index[kb_index[:, 1] == qa_root.logical_and((kb_index[:,1]  != qa_answers.unsqueeze(-1)).sum(0) > 0 )][:,0]
            # print(neg_qa_answers)

        neg_qa_answers_emb = kb_z[neg_qa_batch_answers].transpose(0,1)
        qa_answers_emb = kb_z[qa_batch_answers].transpose(0,1)

        question_emb = qa_question_z.squeeze().unsqueeze(0)
        root_emb = kb_z[qa_batch_root].unsqueeze(0)
        # print( qa_answers_emb.shape, question_emb.shape, root_emb.shape)

        out = (( qa_answers_emb * question_emb * root_emb )[qa_batch_answers.T > 0]).sum(0)
        neg_out = (( neg_qa_answers_emb * question_emb * root_emb )[neg_qa_batch_answers.T > 0]).sum(0)

        loss = self.loss(out, torch.ones_like(out)) + self.loss(neg_out, torch.zeros_like(neg_out))
        acc =  ( (out>0.5)).float().mean()/2 + ( (out<0.5)).float().mean()/2
        self.log('val_acc', acc.item())
        self.log('val_loss', loss.item())
        return None
        acc =  ( (out>0.5) == y).float().mean()
        self.log('val_acc', acc.item())
        return 
            

        (valid_src_index, valid_dst_index) = valid_edge_index.T
        obj_scores = self.decode.score_objs(self.z, valid_src_index, valid_edge_type )

        ranks = obj_scores.argsort(0, descending=True)


        for k in [1,3,10]:
            hits = (ranks[:k]== valid_dst_index).float().mean()  * k
            self.log(f"hit@{k}", hits, on_epoch=True)
        
        
        # mrr = (1. / ranks.float().mean())
        # self.log(f"mrr", mrr, on_epoch=True)

        return 
    
    import torch 


    def negative_sampling(self, edge_index, n_nodes):
        # Sample edges by corrupting either the subject or the object of each edge.
        random_mask = torch.rand(edge_index.size(1)) < 0.5

        neg_edge_index = edge_index.clone()
        neg_edge_index[0, random_mask] = torch.randint(n_nodes, (random_mask.sum(), )).device(edge_index.device)
        neg_edge_index[1, ~random_mask] = torch.randint(n_nodes, ((~random_mask).sum(), )).device(edge_index.device)
        return neg_edge_index
