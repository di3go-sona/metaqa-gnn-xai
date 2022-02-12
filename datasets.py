
# %%
from itertools import repeat
import json
import os
import pickle
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset
import torch, torch_geometric,torchtext, wandb
import networkx as nx
import pytorch_lightning as pl
import nltk
from tqdm import tqdm 
from torch_geometric.utils.convert import from_networkx

class MetaQa(pl.LightningDataModule):
    def __init__(self, size, batch_size):
        super().__init__()
        self.train_qa = []
        self.test_qa = []

        self.max_question_toks = 128
        self.max_answers = 64
        self.size = size
        self.b_size = batch_size
        
        with open('data/1-hop/qa_train.txt') as fin:
            for t in fin.read().lower().strip().split('\n')[:self.size]:

                q, a = t.split('\t')
                root_start, root_end = q.find('['), q.find(']')
                root = q[root_start+1:root_end]
                question = q
                answers = a.split('|')
                self.train_qa.append({'root':root, 'question':question, 'answers':answers})
                

        with open('data/1-hop/qa_test.txt') as fin:
            for t in fin.read().lower().strip().split('\n')[:self.size]:

                q, a = t.split('\t')
                root_start, root_end = q.find('['), q.find(']')
                root = q[root_start+1:root_end]
                question = q
                answers = a.split('|')
                self.test_qa.append({'root':root, 'question':question, 'answers':answers})
                

        with open('data/kb.txt') as fin:
            l = fin.read().lower().strip().split('\n')
            self.kb = nx.classes.DiGraph()
            for i in l:
                src, type, dest = i.split( '|')
                self.kb.add_edge(src, dest, label=type)



        self.build_kb()
        self.build_questions()
        self.build_answers()
        self.build_datasets()

        with open(self.path, 'wb') as fout:
            pickle.dump(self, fout)
    

    @property
    def path(self):
        return f'data/MetaQa.{self.size}.cache'

    def build_questions(self):
        from mosestokenizer  import MosesTokenizer
        moses = MosesTokenizer()
        train_qa_toks = [ moses(t['question'])for t in tqdm(self.train_qa, desc='Build train_qa questions' )]
        test_qa_toks = [ moses(t['question'])for t in  tqdm(self.test_qa, desc='Build test_qa questions')]
        all_toks = train_qa_toks + test_qa_toks
        self.qa_questions_vocab = torchtext.vocab.build_vocab_from_iterator(all_toks)
        

        self.train_question_ids = torch.vstack([ torch.tensor(self.qa_questions_vocab(toks[:self.max_question_toks]) + [0]*(self.max_question_toks-len(toks)))
                for toks in tqdm(train_qa_toks, desc='Translate train_qa questions to ids')])
        self.test_question_ids = torch.vstack([ torch.tensor(self.qa_questions_vocab(toks[:self.max_question_toks]) + [0]*(self.max_question_toks-len(toks)) )
                for toks in tqdm(test_qa_toks, desc='Translate test_qa questions to ids')])
        
    def build_kb(self):
        kb =  from_networkx(self.kb)
        self.kb_index, kb_type = kb.edge_index, kb.label
        self.kb_edges_vocab = torchtext.vocab.build_vocab_from_iterator([kb_type])
        self.kb_type = torch.tensor([self.kb_edges_vocab.get_stoi()[t] for t in kb_type])
        self.kb_n_relations = len(self.kb_edges_vocab)
        self.kb_n_nodes = self.kb_index.max() + 1

    def build_datasets(self):
        self.kb_dataset = torch.utils.data.TensorDataset(self.kb_index.T, self.kb_type )
        self.train_qa_dataset = torch.utils.data.TensorDataset(self.train_question_ids, self.train_answers_ids, self.train_root_ids)
        print(self.test_question_ids.shape, self.test_answers_ids.shape, self.test_root_ids.shape)
        self.test_qa_dataset = torch.utils.data.TensorDataset(self.test_question_ids, self.test_answers_ids, self.test_root_ids)


    def build_answers(self, ):
        # train_answers_ids = [ t['answers'] for t in tqdm(self.train_qa )]
        # test_answers_ids = [ t['answers'] for t in  tqdm(self.test_qa)]
        self.train_answers_ids = []
        self.test_answers_ids = []
        self.train_root_ids = []
        self.test_root_ids = []
        import torch.nn.functional as F
        # from itertools import map
        # answer_ids = torch.zeros(self.train_qa, 64)


        toks_ids_map = { v:k for k,v in enumerate(list(self.kb.nodes())) } 

        answers = [  t['answers'] for t in self.train_qa ]
        roots = [  t['root'] for t in self.train_qa ]
        roots_ids = [toks_ids_map[tok] for tok in roots]
        answers_ids = [torch.tensor([toks_ids_map[tok] for tok in tokens[:64] ] + [-1 for i in range(max(0, 64-len(tokens)))]) for tokens in answers]

        self.train_answers_ids = torch.vstack([ i.unsqueeze(0) for i in answers_ids])
        self.train_root_ids = torch.tensor(roots_ids)

        answers = [  t['answers'] for t in self.test_qa ]
        roots = [  t['root'] for t in self.test_qa ]
        roots_ids = [toks_ids_map[tok] for tok in roots]

        answers_ids = [torch.tensor([toks_ids_map[tok] for tok in tokens[:64] ] + [-1 for i in range(max(0, 64-len(tokens)))]) for tokens in answers]

        self.test_answers_ids = torch.vstack([ i.unsqueeze(0) for i in answers_ids])
        self.test_root_ids = torch.tensor(roots_ids)



    def train_dataloader(self):
        return CombinedLoader([
            DataLoader( self.train_qa_dataset, batch_size=self.b_size, shuffle=True, drop_last=True),
            repeat(self.kb_dataset), ])

    def val_dataloader(self):

        return CombinedLoader([ 
            DataLoader( self.test_qa_dataset, batch_size=self.b_size, drop_last=True),
            repeat(self.kb_dataset) ])



    def test_dataloader(self):
        return self.val_dataloader()
    

    @staticmethod
    def build_or_get( size, batch_size):
        if os.path.exists(f'data/MetaQa.{size}.cache'):
            with open(f'data/MetaQa.{size}.cache', 'rb') as fin:
                metaqa:MetaQa =  pickle.load( fin)
                metaqa.b_size = batch_size
                return metaqa
        return MetaQa(size, batch_size)
        

if __name__ == '__main__':
    d = MetaQa(100, 24)
    next(iter(d.train_dataloader()))
# %%
