import pickle, torch, torch_scatter
from tqdm import tqdm

with open('pickle', 'rb') as file:
    train_edge_index, train_edge_type, test_edge_index, test_edge_type = pickle.load(file)
    
train_edge_index, train_edge_type, test_edge_index, test_edge_type = [
    torch.tensor(e) for e in (train_edge_index, train_edge_type, test_edge_index, test_edge_type)
]


N = torch.vstack((train_edge_index, test_edge_index)).max() + 1
R = torch.hstack((train_edge_type, test_edge_type)).max() + 1
E = 128

embeddings = torch.randn(N, E, requires_grad=True)
W1 = torch.randn(R, E, E, requires_grad=True)
B1 = torch.randn(R, E, requires_grad=True)


relu = torch.nn.ReLU()

opt = torch.optim.Adam([embeddings, W1, B1])

while True:
    res = embeddings[train_edge_index[:,0]].clone()
    opt.zero_grad()

    for r in tqdm(range(R)):
        
        tmp = (embeddings[train_edge_index[:,0]][train_edge_type == r] @ W1[r] + B1[r] ).relu_() @ W1[r] + B1[r]
        res[train_edge_type == r] = tmp


    s = torch_scatter.scatter_sum(res, train_edge_index[:,1], dim=0)
    s.sum().backward()
    opt.step()