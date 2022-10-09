#%%

from random import choices

from cv2 import kmeans
from models.rgcnqa import *
from models.embeddings import *
from globals import *
#%%
embeddings_model = KGEModel.load_from_checkpoint('../data/checkpoints/embeddings/TransEInteraction()|32|epoch=4999|.ckpt')
data = EmbeddingsData(METAQA_PATH)
#%%

nodes = [45, 83, 97, 85, 118, 122, 786,  201, 1549, 1479] + choices(range(data.n_nodes), k=10)
nodes_t = torch.tensor(nodes)
z = embeddings_model.nodes_emb(nodes_t)


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
pca = PCA(2)
z_redux = pca.fit_transform(z.detach().numpy())
z_redux.shape



# tsne = TSNE(2)
# z_redux = tsne.fit_transform(z.detach().numpy())
# z_redux.shape

#%%
kmeans = KMeans(6)
clusters = kmeans.fit_predict(z.detach().numpy())
cmap = {
    0: 'r',
    1: 'g',
    2: 'b',
    3: 'c',
    4: 'm',
    5: 'y'
}
colors = [cmap[c] for c in clusters]

import matplotlib.pyplot as plt

margin = abs(z_redux[:,0].max() - z_redux[:,0].min())*0.1
plt.xlim(z_redux[:,0].min() - margin, z_redux[:,0].max() + 3 * margin)

plt.scatter(*z_redux.T, c=colors)


names =  [ data.entities_names[n] for n in nodes]
for i, txt in enumerate(names):
    plt.annotate(txt, (z_redux[i][0], z_redux[i][1]))
# %%
