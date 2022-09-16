#%%
from train import QA_RGCN, QAData, AutoTokenizer
from torch_geometric.utils import k_hop_subgraph
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_1 = QAData('dataset', [1], tokenizer)
data_2 = QAData('dataset', [2], tokenizer)
data_3 = QAData('dataset', [3], tokenizer)

#%%
from tqdm import tqdm
import matplotlib.pyplot as plt
# %%
def get_neighborhood_sizes(data, k):
    for (s,_, _), ans in tqdm(data.train_ds[:1024*4]):
        subset, _, _, _ = k_hop_subgraph(s, k, data.get_triples().T[[0,2]])
        yield len(subset)

size_1 = list(get_neighborhood_sizes(data_1, 1))
size_2 = list(get_neighborhood_sizes(data_1, 2))
size_3 = list(get_neighborhood_sizes(data_1, 3))

# %%
plt.hist(size_1)
plt.hist(size_2)
plt.hist(size_3)


# %%
