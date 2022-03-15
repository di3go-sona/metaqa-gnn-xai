import torch, pytorch_lightning as pl 

from torch.nn import Linear, ReLU
from models.LSTMEncoder import LSTMEncoder
from models.RGCNEncoder import RGCNEncoder
from models.EmbeddingsDecoders import *

class QaModel(pl.LightningModule):
    def __init__(self, n_nodes: int, n_relations: int, questions_vocab_size:int, config: dict, embeddings = None) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.questions_vocab_size = questions_vocab_size
        
        self.n_layers = config['n_layers']
        self.reg = config['reg']
        self.p_dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        
        self.encode_kb = RGCNEncoder(embeddings, self.n_relations, config,  )
        self.encode_question = LSTMEncoder(len(self.questions_vocab_size), 32, embeddings.shape[1], stack='short-term')
        
        self.linear1 =  Linear(embeddings.shape[1]*3,embeddings.shape[1])
        self.linear2 =  Linear(embeddings.shape[1], 1)
        self.relu = ReLU()
        self.loss = torch.nn.BCEWithLogitsLoss()
        
        
        self.decode_question =torch.nn.Sequential(
            self.linear1,
            self.relu,
            self.linear2            
        )
        

        self.save_hyperparameters()
        
    def cname(self):
        params = ['learning_rate', 'p_dropout', 'reg', 'n_layers']
        params_string = '|'.join([f"{p}={getattr(self, p)}" for p in params])
        return f"{self.__class__.__name__}|{params_string}"

    def configure_optimizers(self):
        
        no_decay = ["bias", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.reg,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # print(list(self.named_parameters()))
        # optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters,
        #     lr=lr,
        #     weight_decay=weight_decay,
        # )
        
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate, weight_decay=self.reg)
        return optimizer

    def training_step(self, batch, batch_idx):
        kb,  ( qa_questions , qa_nodes) =  batch
        qa_root_nodes, qa_ans_nodes, qa_ans = qa_nodes.T
        kb_src, kb_dest, kb_type = kb.T
        
        qa_ans = qa_ans.float()
        
        

        kb_z = self.encode_kb(torch.vstack([kb_src, kb_dest]), kb_type)
        qa_question_z = self.encode_question(qa_questions)
        qa_answers_emb = kb_z[qa_ans_nodes.long()]
        root_emb = kb_z[qa_root_nodes.long()]
        qa_question_emb = qa_question_z.squeeze()

        out = self.decode( qa_answers_emb, qa_question_emb, root_emb).squeeze()


        # print(qa_ans.dtype, out.dtype)
        pos_loss = self.loss(out[qa_ans == 1], qa_ans[qa_ans == 1])
        neg_loss = self.loss(out[qa_ans == 0], qa_ans[qa_ans == 0])
        prec_pos = (out.sigmoid()[qa_ans == 1]>0.5).float().mean()
        prec_neg = (out.sigmoid()[qa_ans == 0]<0.5).float().mean()
        acc = (prec_neg + prec_pos) / 2
        loss =  pos_loss  + neg_loss 
        
        
        self.log('train_acc', acc.item())
        self.log('train_loss', loss.item())
        self.log('train_prec_pos', prec_pos.item())
        self.log('train_prec_neg', prec_neg.item())

        return loss

        
    def validation_step(self, batch, batch_idx):

        kb,  ( qa_questions , qa_nodes) =  batch
        qa_root_nodes, qa_ans_nodes, qa_ans = qa_nodes.T
        kb_src, kb_dest, kb_type = kb.T
        
        qa_ans = qa_ans.float()
        
        

        kb_z = self.encode_kb(torch.vstack([kb_src, kb_dest]), kb_type)
        qa_question_z = self.encode_question(qa_questions)
        qa_answers_emb = kb_z[qa_ans_nodes.long()]
        root_emb = kb_z[qa_root_nodes.long()]
        qa_question_emb = qa_question_z.squeeze()

        out = self.decode( qa_answers_emb, qa_question_emb, root_emb).squeeze()


        # print(qa_ans.dtype, out.dtype)
        pos_loss = self.loss(out[qa_ans == 1], qa_ans[qa_ans == 1])
        neg_loss = self.loss(out[qa_ans == 0], qa_ans[qa_ans == 0])
        prec_pos = (out.sigmoid()[qa_ans == 1]>0.5).float().mean()
        prec_neg = (out.sigmoid()[qa_ans == 0]<0.5).float().mean()
        acc = (prec_neg + prec_pos) / 2
        loss =  pos_loss  + neg_loss  
        
        
        
        self.log('val_acc', acc.item())
        self.log('val_loss', loss.item())
        self.log('val_prec_pos', prec_pos.item())
        self.log('val_prec_neg', prec_neg.item())
        return None

    def decode(self, qa_answers_emb, question_emb, root_emb):

        stacked_input = torch.cat((qa_answers_emb, question_emb, root_emb),dim=1)
        return self.decode_question(stacked_input)


