
# %%

import itertools, math, os, pickle, random, re
import torch, torchtext, networkx as nx

from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning import LightningDataModule
from mosestokenizer  import MosesTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 


def pad_collate(batch):
    bX, by1, by2, by3 = zip(*batch)


    
    yy1 = torch.hstack(by1)
    yy2 = torch.hstack(by2)
    yy3 = torch.hstack(by3)
    
    X = torch.vstack(bX)
    y = torch.vstack([yy1, yy2, yy3]).T

    # print(X.shape, y.shape)
    return X, y


class QaNodesDataset(Dataset):
    def __init__(self, kb, nodes_vocab, roots, answers, questions, levels=1, mix=False, max_pos=2, neg_ratio=5) -> None:
        self.kb = kb
        self.nodes_vocab = nodes_vocab
        self.roots = roots
        self.answers = answers 
        self.questions = questions
        self.levels=levels
        self.mix = mix
        self.max_pos = max_pos
        self.neg_ratio = neg_ratio
        

    def __getitem__(self, item):
        root = self.roots[item]
        pos_answers = self.answers[item]
        root_name = self.nodes_vocab.lookup_token(root.item())
        nodes = [root_name]
        for _ in range(self.levels):

            next = []
            for n in nodes:
                next += list( self.kb.neighbors(n)) 
                next += list( nx.reverse_view(self.kb).neighbors(n))
                
            nodes = list(set(next))      

        neg_answers = [ a for a in  self.nodes_vocab.lookup_indices(nodes) if a not in pos_answers]
        negatives = min( len(neg_answers), self.max_pos*self.neg_ratio )
        positives = min( len(pos_answers), min(self.max_pos, math.ceil(negatives/self.neg_ratio)))


        tot_answers = negatives + positives
        question_t = self.questions[item].repeat((tot_answers, 1))
        root_node_t = torch.full((tot_answers,), root, dtype=torch.long )
        
        pos_ans_node_t = torch.tensor(random.choices(pos_answers, k=positives),dtype=torch.long) if positives >0 else torch.zeros(0,dtype=torch.long)
        neg_ans_node_t = torch.tensor(random.choices(neg_answers, k=negatives),dtype=torch.long)  if negatives >0 else torch.zeros(0,dtype=torch.long)
        pos_ans_t = torch.zeros(positives,dtype=torch.long)  if positives >0 else torch.zeros(0,)
        neg_ans_t = torch.ones(negatives,dtype=torch.long) if negatives >0 else torch.zeros(0,)
        
        ans_nodes_t = torch.cat((pos_ans_node_t, neg_ans_node_t ))
        ans_t = torch.cat((pos_ans_t, neg_ans_t ))


        
        return question_t, root_node_t, ans_nodes_t, ans_t
        

    
    def __len__(self):
        return len(self.roots)

class TripletsDataset(Dataset):
    def __init__(self, ) -> None:
        self.build_kb()

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

    def __getitem__(self, item):
       
        return torch.tensor([*self.kb_edge_index[:,item], self.kb_edge_type[item]])
    
    @property
    def n_embeddings(self):
        return torch.max(self.kb_edge_index)+2
     
    @property
    def n_relations(self):
        return torch.max(self.kb_edge_type)+2

    def __len__(self):
        return len(self.kb_edge_type)

class MetaQaEmbeddings(LightningDataModule):
    def __init__(self, batch_size=128, val_batch_size=None):
        super().__init__()
        self.ds = TripletsDataset()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds, self.batch_size, drop_last = True, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds, self.val_batch_size or self.batch_size, shuffle=True)
        
class MetaQa(LightningDataModule):
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
        self.qa_questions_vocab = torchtext.vocab.build_vocab_from_iterator(all_toks, special_first=True, specials=['pad', 'unk'])
        
        # Translate question tokens to ids
        self.question_tok_ids = {}
        for src in ['train', 'test']:
            
            question_tok_ids = [ torch.tensor(self.qa_questions_vocab(toks))
                for toks in tqdm(question_toks[src], desc='Translate train_qa questions to ids') ]
            
            question_tok_ids = torch.nn.utils.rnn.pad_sequence(question_tok_ids, batch_first=True, padding_value=0)
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
        return self

        
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
            DataLoader(self.train_qa_nodes_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=pad_collate),
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
    pass
# %%
