#%%
from train import QA_RGCN, QAData, AutoTokenizer
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

import pickle
import os

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_1 = QAData('dataset', [1], tokenizer)
data_2 = QAData('dataset', [2], tokenizer)
data_3 = QAData('dataset', [3], tokenizer)

# %%
def get_neighborhood_sizes(data, k):
    for (s,_, _), ans in tqdm(data.train_ds[:]):
        subset, _, _, _ = k_hop_subgraph(s, k, data.get_triples().T[[0,2]])
        yield subset


# %%
if not os.path.exists('neigh_1.pickle'):
    neigh_1 = [ s for s in get_neighborhood_sizes(data_1, 1)]
    with open('neigh_1.pickle', 'wb') as fout:
        pickle.dump(neigh_1, fout)
else:
    with open('neigh_1.pickle', 'rb') as fin:
        neigh_1 = pickle.load(fin)
        

if not os.path.exists('neigh_2.pickle'):
    neigh_2 = [ s for s in get_neighborhood_sizes(data_1, 2)]
    with open('neigh_2.pickle', 'wb') as fout:
        pickle.dump(neigh_2, fout)
else:
    with open('neigh_2.pickle', 'rb') as fin:
        neigh_2 = pickle.load(fin)

if not os.path.exists('neigh_3.pickle'):
    neigh_3 = [ s for s in get_neighborhood_sizes(data_1, 3)]
    with open('neigh_3.pickle', 'wb') as fout:
        pickle.dump(neigh_3, fout)
else:
    with open('neigh_3.pickle', 'rb') as fin:
        neigh_3 = pickle.load(fin)
    
    
    
    
# # %%
# import numpy as np
# def plot_loghist(x, bins=30):
#   hist, bins = np.histogram(x, bins=bins)
#   logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
#   plt.hist(x, bins=logbins)
#   plt.xscale('log')

# plt.figure(figsize=(10,3))
# plt.hist(size_3, bins=30)
# plt.hist(size_2, bins=30)
# plt.hist(size_1, bins=30)
# # %%
# plot_loghist(size_1)
# plot_loghist(size_2)
# plot_loghist(size_3)
# # %%

# %%
