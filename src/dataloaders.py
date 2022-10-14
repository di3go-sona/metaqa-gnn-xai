#%%
import re
import torch, os
import pytorch_lightning as pl

from globals import *
from tqdm import tqdm 
from torch.utils.data import DataLoader
from pytorch_lightning.trainer.supporters import CombinedLoader

from joblib import Memory
memory = Memory('../data/cachedir', verbose=0)


@memory.cache(ignore=['parse_question'])
def _load_questions(path, hops, split, use_ntm, parse_question ):
        questions = []
        if use_ntm:
            for h in hops:

                p = os.path.join(path, f'{h}-hop','ntm', f'qa_{split}.txt' )
                with open(p) as fin:
                    for l in tqdm(fin):
                        questions.append(parse_question(l))
        else:
            for h in hops:

                p = os.path.join(path, f'{h}-hop','vanilla', f'qa_{split}.txt' )
                with open(p) as fin:
                    for l in tqdm(fin):
                        questions.append(parse_question(l))
               
        return questions
    
#%%
def qa_batch_collate(batch):
    _triplets, _q_toks =  zip(*batch)
    
    triplets = torch.tensor(_triplets, dtype= torch.long)    # batch_size, 3
    q_toks = torch.nn.utils.rnn.pad_sequence( _q_toks )      # seq_len, batch_size 
    
    return triplets, q_toks

def qa_batch_collate_unflattened(batch):
    _s, _q_id, _alist, _q_toks =  zip(*batch)
    
    _alist = [ torch.tensor(a, dtype=torch.long) for a in _alist]
    s = torch.tensor(_s, dtype=torch.long)
    tlist = torch.nn.utils.rnn.pad_sequence( _alist, padding_value=-1 ) 
    q_toks = torch.nn.utils.rnn.pad_sequence( _q_toks )      # seq_len, batch_size 
    
    return s, tlist, q_toks

def kge_batch_collate_unflattened(batch):
    _s, _r, _tlist =  zip(*batch)
    # print(_s, _r, _tlist)
    
    # _alist = [ torch.tensor(a, dtype=torch.long) for a in _alist]
    s = torch.tensor(_s, dtype=torch.long)
    r = torch.tensor(_r, dtype=torch.long)
    tlist = torch.nn.utils.rnn.pad_sequence( _tlist, padding_value=-1 ) 
    return s, r, tlist
    
#%%
class EmbeddingsData(pl.LightningDataModule):
    def __init__(self, path, train_batch_size=16, val_batch_size=16 ):
        super().__init__()
        
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.path = path
        
        with open(os.path.join(METAQA_PATH,'kb_entity_dict.txt')) as fin:
            entities = [ l.strip().split('\t') for l in fin.readlines()]
            self.entities_names = dict([ (int(k), v) for k,v in entities])
            self.entities_ids = dict([ ( v, int(k)) for k,v in entities])
            
        with open(os.path.join(METAQA_PATH,'kb_relations_dict.txt')) as fin:
            relations = [ l.strip().split('\t') for l in fin.readlines()]
            self.relations_names = dict([ (int(k), v) for k,v in relations])
            self.relations_ids = dict([ ( v, int(k)) for k,v in relations])

        with open(os.path.join(METAQA_PATH,'kb.txt')) as fin:
            edges = [ l.strip().split('|') for l in fin.readlines()]
            
        self._triples = torch.tensor( [ 
                                       [self.entities_ids[s], 
                                        self.relations_ids[r],
                                        self.entities_ids[d]] for s,r,d in edges] , dtype = torch.long)
        
        self.n_nodes = len(entities)
        self.n_edges = len(edges) 
        self.n_relations = len(relations) 

    def get_triples(self, add_inverse=True):
        if add_inverse:
            triples = self._triples
            inverse_triples = torch.index_select(triples, 1, torch.tensor([2,1,0], dtype=torch.long, device=triples.device))
            inverse_triples[:, 1]  += self.n_relations
            return torch.cat((triples, inverse_triples),0) 
        else: 
            return self._triples    
    
    def train_dataloader(self) :
        return  DataLoader(self._triples, self.train_batch_size, shuffle=True)
    
    def val_dataloader(self):
        pseudotriplets = {}
        
        for (s,r,t) in sorted(self._triples.tolist(), key=lambda x: x[:1]):
            key = (s,r)
            l = [*pseudotriplets.get(key, []), t]
            pseudotriplets[key] = l
        
        val_ds = [ (s,r,torch.tensor(tlist, dtype=torch.long)) for (s,r), tlist in pseudotriplets.items() ]
        return DataLoader(val_ds, self.val_batch_size, shuffle=True, collate_fn=kge_batch_collate_unflattened)
        
class QAData(EmbeddingsData):
    def __init__(self, path, hops, tokenizer, train_batch_size=16, val_batch_size=16, use_ntm=True ):
        super().__init__(path, train_batch_size=16, val_batch_size=16)
        
        self.hops = hops
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.use_ntm = use_ntm
        self.tokenizer = tokenizer

        self.train_ds = self.load_question_triplets('train')
        
        self.train_ds_unflattened = self.load_question_triplets('train', flatten=False)
        self.val_ds_unflattened = self.load_question_triplets('dev', flatten=False)
        # self.test_ds_unflattened = self.load_question_triplets('test', flatten=False)
    
    def parse_question(self, line):
        _question, answers = line.strip().split('\t')
        
        _root = re.search(r'\[(.*)\]', _question).group(1)
        _question = _question.strip().replace( f'[{_root}]', 'root')
        
        question_id = hash(_question)
        question_toks = self.tokenizer.encode( _question, return_tensors='pt')[0]
        
        root  = self.entities_ids[ _root ] 
        answers = [ self.entities_ids[a] for a in  answers.split('|') ]
        
        return root, question_id, answers, question_toks
    
    def load_questions(self, split ):                    
        return _load_questions(self.path, self.hops, split, self.use_ntm, self.parse_question)
        
        
    def load_question_triplets(self, split, flatten=True):
        questions = self.load_questions(split)
        if flatten:
            triplets = [ ((s,q_id,t), q_toks) for s, q_id, tlist, q_toks in questions for t in tlist]
        else:
            triplets = questions
        return triplets           
    
    def train_dataloader(self) :
        return  DataLoader(self.train_ds,
            self.train_batch_size,
            shuffle=True,
            collate_fn=qa_batch_collate )
    
    def val_dataloader(self):
        return CombinedLoader({
            'train': DataLoader( 
                self.train_ds_unflattened,
                self.val_batch_size,
                shuffle=True,
                collate_fn=qa_batch_collate_unflattened ),
         
            'val': DataLoader( 
                self.val_ds_unflattened,
                self.val_batch_size,
                shuffle=True,
                collate_fn=qa_batch_collate_unflattened ) })
    
    def test_dataloader(self):
        return DataLoader( 
                self.val_ds_unflattened,
                self.val_batch_size,
                shuffle=False,
                collate_fn=qa_batch_collate_unflattened )
          


