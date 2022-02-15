
# %%

from datetime import datetime
import itertools
import os
import pickle
import re
from pytorch_lightning.trainer.supporters import CombinedLoader
from mosestokenizer  import MosesTokenizer

from torch.utils.data import DataLoader, Dataset
import torch, torchtext
import networkx as nx
import pytorch_lightning as pl
from tqdm import tqdm 
from torch_geometric.utils.convert import from_networkx
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch):
    x_lens = []
    xx = []
    yy = []
    for x, y in batch:
        x_lens.append(len(x))
        xx.append(x)
        yy.append(y)
        # print(x,y)


    xx_pad = pad_sequence(xx, batch_first=True, padding_value=-1)


    return xx_pad,  x_lens, torch.vstack(yy)


class QaNodesDataset(Dataset):
    def __init__(self, kb, nodes_vocab, roots, answers, questions, max_answers=1) -> None:
        self.kb = kb
        self.nodes_vocab = nodes_vocab
        self.roots = roots
        self.answers = answers 
        self.max_answers = max_answers
        self.questions = questions

        

    def __getitem__(self, item):
        root = self.roots[item]
        answers = self.answers[item]
        # print(answers)
        root_name = self.nodes_vocab.lookup_token(root.item())
        neg_answers= list(self.kb.neighbors(root_name)) + list( nx.reverse_view(self.kb).neighbors(root_name))
        # print(neg_answers)
        neg_answers = self.nodes_vocab.lookup_indices(neg_answers)

        # neg_answers =list( nx.generators.ego.ego_graph(self.kb,root_name , radius=1, center=False, undirected=True).nodes())
        # print(neg_answers)
        # neg_answers = self.nodes_vocab(neg_answers)
        neg_answers = [ a for a in neg_answers if a not in answers]
        r = hash(datetime.now)
        pos_ans = answers[r % len(answers)]
        neg_ans = neg_answers[r % len(neg_answers)] if len(neg_answers) > 0 else torch.tensor(-1)
        return self.questions[item], torch.tensor([root, pos_ans, neg_ans])
        

    
    def __len__(self):
        return len(self.roots)


class MetaQa(pl.LightningDataModule):
    path = 'data/MetaQa.cache'
    def __init__(self, batch_size, max_answers=1, max_question_toks=32, strip_root=True):
        super().__init__()
        self.qa_data = {}


        self.max_answers = max_answers
        self.max_question_toks = max_question_toks
        self.strip_root = strip_root
        self.batch_size = batch_size

                




        self.build_kb()
        self.build_qa()
        self.build_datasets()

        with open(self.path, 'wb') as fout:
            pickle.dump(self, fout)
    


    def build_qa(self):
        ROOT_REGEX = r'\[(.*)\]'
        
        
        # Load Question and Answers
        for src in ['train', 'test']:
            with open(f'data/1-hop/qa_{src}.txt') as fin:
                tmp = []
                for t in fin.read().lower().strip().split('\n'):

                    question, answers = t.split('\t')
                    
                    root_match = re.search(ROOT_REGEX, question )
                    
                    answers = answers.split('|')
                    root = root_match.group(1)
                    
                    if self.strip_root:
                        question = re.sub(ROOT_REGEX, 'ROOT', question)
                    
                    tmp.append({'root':root, 'question':question, 'answers':answers})
                    
                self.qa_data[src] = tmp
                
                
        # Build Questions Tokens
        moses = MosesTokenizer()
        question_toks = { src : [ moses(self.qa_data[src][q]['question'])  for q, _ in tqdm(enumerate(self.qa_data[src]), desc='Build train_qa questions')]
                for src in ['train', 'test']    
        }
        
        # Build Vocab
        all_toks = list(itertools.chain(*question_toks.values()))
        self.qa_questions_vocab = torchtext.vocab.build_vocab_from_iterator(all_toks)
        
        # Translate question tokens to ids
        self.question_tok_ids = {}
        for src in ['train', 'test']:
            
            question_tok_ids = [ torch.tensor(self.qa_questions_vocab(toks))
                for toks in tqdm(question_toks[src], desc='Translate train_qa questions to ids') ]
            
            question_tok_ids = torch.nn.utils.rnn.pad_sequence(question_tok_ids, batch_first=True, padding_value=-1)
            self.question_tok_ids[src] = question_tok_ids
        
        # Translate question root nodes to ids
        self.root_nodes_ids = {}
        for src in ['train', 'test']:
            root_nodes_ids = [  t['root'] for t in self.qa_data[src] ]
            root_nodes_ids = self.kb_nodes_vocab(root_nodes_ids)
            self.root_nodes_ids[src] = torch.tensor(root_nodes_ids)

            
        self.answers_nodes_ids = {}
        for src in ['train', 'test']:
            answers_nodes_ids = [  t['answers'] for t in self.qa_data[src] ]

            answers_nodes_ids=[self.kb_nodes_vocab(list(a)) for a in answers_nodes_ids]
            answers_nodes_ids = [ torch.tensor(a) for a in answers_nodes_ids]
            self.answers_nodes_ids[src] = answers_nodes_ids

        
    def build_kb(self):
                # Load Knowledgebase Graph
        with open('data/kb.txt') as fin:
            l = fin.read().lower().strip().split('\n')
            self.kb = nx.classes.DiGraph()
            for i in l:
                src, type, dest = i.split( '|')
                self.kb.add_edge(src, dest, label=type)
                
            s, d, t  = zip(*self.kb.edges(data='label'))
            self.kb_nodes_vocab = torchtext.vocab.build_vocab_from_iterator([s,d])
            self.kb_edges_vocab = torchtext.vocab.build_vocab_from_iterator([t])

            s_id =  self.kb_nodes_vocab.lookup_indices(list(s))
            d_id =  self.kb_nodes_vocab.lookup_indices(list(d))
            t_id =  self.kb_edges_vocab.lookup_indices(list(t))



        self.kb_edge_index = torch.vstack([torch.tensor(s_id), torch.tensor(d_id)])
        self.kb_edge_type =torch.tensor(t_id)

        self.kb_n_relations = len(self.kb_edges_vocab)
        self.kb_n_nodes = self.kb_edge_index.max() + 1
        
        

    def build_datasets(self):
        self.kb_dataset = torch.utils.data.TensorDataset(self.kb_edge_index.T, self.kb_edge_type )
        self.train_qa_nodes_dataset = QaNodesDataset(self.kb, self.kb_nodes_vocab, self.root_nodes_ids['train'], self.answers_nodes_ids['train'],self.question_tok_ids['train'])
        self.test_qa_nodes_dataset = QaNodesDataset(self.kb, self.kb_nodes_vocab,self.root_nodes_ids['test'], self.answers_nodes_ids['test'],self.question_tok_ids['test'])
        # self.train_qa_question_dataset = self.question_tok_ids['train']
        # self.test_qa_question_dataset =self.question_tok_ids['test']



    def train_dataloader(self):
        return CombinedLoader([
            DataLoader(self.kb_dataset, batch_size=len(self.kb_dataset)),
            DataLoader(self.train_qa_nodes_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8,collate_fn=pad_collate),
            # DataLoader( self.train_qa_question_dataset, batch_size=self.batch_size, ),
             ], 'max_size_cycle')

    def val_dataloader(self):

        return CombinedLoader([ 
            DataLoader(self.kb_dataset, batch_size=len(self.kb_dataset)),
            DataLoader(self.test_qa_nodes_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8, collate_fn=pad_collate),           
            # DataLoader(self.test_qa_question_dataset, batch_size=self.batch_size, collate_fn=pad_collate),
             ], 'max_size_cycle')


    def test_dataloader(self):
        return self.val_dataloader()
    

    @staticmethod
    def build_or_get(batch_size):
        if os.path.exists(MetaQa.path):
            with open(MetaQa.path, 'rb') as fin:
                metaqa:MetaQa =  pickle.load( fin)
                metaqa.batch_size = batch_size
                return metaqa
        return MetaQa(batch_size)
        

if __name__ == '__main__':
    d = MetaQa(24)
    next(iter(d.train_dataloader()))
# %%
