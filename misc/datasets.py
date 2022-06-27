
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

class QaQuestionsDataset(Dataset):
    def __init__(self, kb, question_toks, roots, answers, hops, max_pos, neg_ratio) -> None:
        self.kb = kb
        self.questions = question_toks
        self.roots = roots
        self.answers = answers
        self.hops = hops
        self.max_pos = max_pos
        self.neg_ratio = neg_ratio
        
    
    def __getitem__(self, item):
        root = self.roots[item]
        pos_answers = self.answers[item]
        # root_name = root.item()
        nodes = { 0: [root.item()] }
        # print(nodes, self.kb.nodes())
        for l in range(3):
            hits = 0
            next = []
            current_nodes = nodes[l]
            
            random.shuffle(current_nodes)

            
            for n in current_nodes:
                nodes_in = list( self.kb.neighbors(n))
                nodes_out = list(nx.reverse_view(self.kb).neighbors(n))
                _next = list(set(nodes_in + nodes_out ))
                
                
                
                if len(_next) > 0:
                    hits += 1
                    # random.shuffle(_next)
                    # _next = _next[:self.neg_ratio]
                next.extend(_next)
                if hits >= 3:
                    break
                
            nodes[l+1] = next
                
                
                # print(next)
                # next += random.choices(list(set(list( self.kb.neighbors(n)) + list( nx.reverse_view(self.kb).neighbors(n)))), k=self.neg_ratio)

            
        # pprint(nodes)
        nodes = nodes[1] + nodes[2] + nodes[3]

        neg_answers = [ a for a in  nodes if a not in  torch.hstack([pos_answers, root])]
        
        negatives = min( len(neg_answers), self.max_pos*self.neg_ratio )
        positives = min( len(pos_answers), self.max_pos)


        tot_answers = negatives + positives
        question_t = self.questions[item].repeat((tot_answers, 1))
        root_node_t = torch.full((tot_answers,), root, dtype=torch.long )
        
        pos_ans_node_t = torch.tensor(random.choices(pos_answers, k=positives),dtype=torch.long) if positives >0 else torch.ones(0,dtype=torch.long)
        neg_ans_node_t = torch.tensor(random.choices(neg_answers, k=negatives),dtype=torch.long)  if negatives >0 else torch.zeros(0,dtype=torch.long)
        pos_ans_t = torch.ones(positives,dtype=torch.long)  if positives >0 else torch.ones(0,)
        neg_ans_t = torch.zeros(negatives,dtype=torch.long) if negatives >0 else torch.zeros(0,)
        
        ans_nodes_t = torch.cat((pos_ans_node_t, neg_ans_node_t ))
        ans_t = torch.cat((pos_ans_t, neg_ans_t ))


        
        return question_t, root_node_t, ans_nodes_t, ans_t
    
    def __len__(self):
        return len(self.roots)

class QaQuestionsDatasetsProvider():
    def __init__(self, kb, nodes_vocab, hops=1, mix=False, max_pos=2, neg_ratio=10, strip_root=True) -> None:
        self.kb = kb
        self.nodes_vocab = nodes_vocab
        self.hops = hops
        self.mix = mix
        self.max_pos = max_pos
        self.neg_ratio = neg_ratio
        self.strip_root = True
        
        
        self.build_qa()
    
    def get_dataset(self, split):
        return QaQuestionsDataset( self.kb, self.question_tok_ids[split], self.root_nodes_ids[split], self.answers_nodes_ids[split],  self.hops, self.max_pos, self.neg_ratio)

    
    def build_qa(self):
        ROOT_REGEX = r'\[(.*)\]'
        SPLITS = ['train', 'test', 'dev']    
        self.qa_data = {}
        # Load Question and Answers
        for src in SPLITS:
            with open(f'data/{self.hops}-hops/qa_{src}.txt') as fin:
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
        question_toks = { src : [ moses(self.qa_data[src][q]['question'])  for q, _ in tqdm(enumerate(self.qa_data[src]), desc=f'Build {src} questions')]
                for src in SPLITS
        }
        
        # Build Vocab
        all_toks = list(itertools.chain(*question_toks.values()))
        self.qa_questions_vocab = torchtext.vocab.build_vocab_from_iterator(all_toks, special_first=True, specials=['pad', 'unk'])
        
        # Translate question tokens to ids
        self.question_tok_ids = {}
        for src in SPLITS:
            
            question_tok_ids = [ torch.tensor(self.qa_questions_vocab(toks))
                for toks in tqdm(question_toks[src], desc=f'Translate {src} questions to ids') ]
            
            question_tok_ids = torch.nn.utils.rnn.pad_sequence(question_tok_ids, batch_first=True, padding_value=0)
            self.question_tok_ids[src] = question_tok_ids
        
        # Translate question root nodes to ids
        self.root_nodes_ids = {}
        for src in SPLITS:
            root_nodes_ids = [  t['root'] for t in self.qa_data[src] ]
            root_nodes_ids = self.nodes_vocab(root_nodes_ids)
            self.root_nodes_ids[src] = torch.tensor(root_nodes_ids)

            
        self.answers_nodes_ids = {}
        for src in SPLITS:
            answers_nodes_ids = [  t['answers'] for t in self.qa_data[src] ]

            answers_nodes_ids=[self.nodes_vocab(list(a)) for a in answers_nodes_ids]
            answers_nodes_ids = [ torch.tensor(a) for a in answers_nodes_ids]
            self.answers_nodes_ids[src] = answers_nodes_ids
        return self
        
class NodesDataset(Dataset):
    def __init__(self, path='data/kb.txt') -> None:
        self.path=path
        self.build_kb()

    def build_kb(self, ):
        # Load Knowledgebase Graph
        with open(self.path) as fin:
            l = fin.read().lower().strip().split('\n')
            kb = nx.classes.DiGraph()
            
            for i in l:
                src, type, dest = i.split( '|')
                kb.add_edge(src, dest, label=type)
                
            s, d, t  = zip(*kb.edges(data='label'))
            self.kb_nodes_vocab = torchtext.vocab.build_vocab_from_iterator([s,d])
            self.kb_edges_vocab = torchtext.vocab.build_vocab_from_iterator([t])

            s_ids =  self.kb_nodes_vocab.lookup_indices(list(s))
            d_ids =  self.kb_nodes_vocab.lookup_indices(list(d))
            t_ids =  self.kb_edges_vocab.lookup_indices(list(t))

            self.kb = nx.classes.DiGraph()
            for s, d, t in zip(s_ids, d_ids, t_ids):
                self.kb.add_edge(s, d, label=t)


        self.kb_edge_index = torch.vstack([torch.tensor(s_ids), torch.tensor(d_ids)])
        self.kb_edge_type =torch.tensor(t_ids)

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
        self.ds = NodesDataset()
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds, self.batch_size, drop_last = True, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds, self.val_batch_size or self.batch_size, shuffle=True)
        
class MetaQa(LightningDataModule):
    path = 'data/MetaQa.cache'
    def __init__(self, hops, batch_size, max_pos=1, neg_ratio=10, max_question_toks=32,  strip_root=True):
        super().__init__()
        self.qa_data = {}

        self.hops = hops
        self.max_pos = max_pos
        self.neg_ratio = neg_ratio
        self.max_question_toks = max_question_toks
        self.strip_root = strip_root
        self.batch_size = batch_size
        
        self.nodes_dataset = NodesDataset()
        self.qa_questions_datasets = QaQuestionsDatasetsProvider(self.nodes_dataset.kb, self.nodes_dataset.kb_nodes_vocab, self.hops, False, self.max_pos, self.neg_ratio, True)
        
        
        # self.build_datasets()

        with open(self.path, 'wb') as fout:
            pickle.dump(self, fout)
    
        

    # def build_datasets(self):
        # self.kb_dataset = torch.utils.data.Triplets
        # self.qa_questions_datasets.get_dataset('train') = QaQuestionsDataset(self.kb, self.kb_nodes_vocab, self.root_nodes_ids['train'], self.answers_nodes_ids['train'],self.question_tok_ids['train'])
        # self.test_qa_nodes_dataset = QaQuestionsDataset(self.kb, self.kb_nodes_vocab,self.root_nodes_ids['test'], self.answers_nodes_ids['test'],self.question_tok_ids['test'])
        # self.train_qa_question_dataset = self.question_tok_ids['train']
        # self.test_qa_question_dataset =self.question_tok_ids['test']



    def train_dataloader(self):
        return CombinedLoader([
            DataLoader(self.nodes_dataset, batch_size=len(self.nodes_dataset)),
            DataLoader(self.qa_questions_datasets.get_dataset('train'), batch_size=self.batch_size or len(self.qa_questions_datasets.get_dataset('train')) ,prefetch_factor=4, shuffle=True, num_workers=8, collate_fn=pad_collate),
            # DataLoader( self.train_qa_question_dataset, batch_size=self.batch_size, ),
             ], 'max_size_cycle')

    def val_dataloader(self):

        
        return CombinedLoader([ 
            DataLoader(self.nodes_dataset, batch_size=len(self.nodes_dataset)),
            DataLoader(self.qa_questions_datasets.get_dataset('test'), batch_size=self.batch_size or len(self.qa_questions_datasets.get_dataset('test')) ,prefetch_factor=4, shuffle=False,num_workers=8, collate_fn=pad_collate),           
            # DataLoader(self.test_qa_question_dataset, batch_size=self.batch_size, collate_fn=pad_collate),
             ], 'max_size_cycle')


    def test_dataloader(self):
        return self.val_dataloader()

if __name__ == '__main__':
    nodes_dataset = NodesDataset()
    # print(len(nodes_dataset))
    qa_questions_dataset_provider = QaQuestionsDatasetsProvider(nodes_dataset.kb, nodes_dataset.kb_nodes_vocab, 2, False, 1)
    # print(qa_questions_dataset_provider)
    qa_questions_dataset_train = qa_questions_dataset_provider.get_dataset('train')
    qa_questions_dataset_test = qa_questions_dataset_provider.get_dataset('test')
    qa_questions_dataset_dev = qa_questions_dataset_provider.get_dataset('dev')
    # print(qa_questions_dataset_train, qa_questions_dataset_test, qa_questions_dataset_dev)
    print(qa_questions_dataset_train[1])
    print(qa_questions_dataset_test[1])
    print(qa_questions_dataset_dev[1])

# %%
