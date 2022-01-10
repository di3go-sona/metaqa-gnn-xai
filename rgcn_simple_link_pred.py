import pickle, torch, torch_scatter
from torch.nn.init import zeros_
from tqdm import tqdm

with open('pickle', 'rb') as file:
    train_edge_index, train_edge_type, test_edge_index, test_edge_type = pickle.load(file)
    
train_edge_index, train_edge_type, test_edge_index, test_edge_type = [
    torch.tensor(e) for e in (train_edge_index, train_edge_type, test_edge_index, test_edge_type)
]


N = torch.vstack((train_edge_index, test_edge_index)).max() + 1
R = torch.hstack((train_edge_type, test_edge_type)).max() + 1
E = 256
L = 3

embeddings = torch.randn(N, E, requires_grad=True)
rgnc_weights = [torch.randn(R, E, E, requires_grad=True) for _ in range(L) ]
rgnc_biases = [torch.randn(R, E, requires_grad=True) for _ in range(L) ]
relu = torch.nn.ReLU()

distmult_weights = torch.randn(N, E, requires_grad=True)


opt = torch.optim.Adam([embeddings, *rgnc_weights, *rgnc_biases])

while True:
    hidden = torch.zeros_like(embeddings)
    
    opt.zero_grad()

    for l in tqdm(range(L)):
        layer_messages = []
        for r in tqdm(range(R)):
            
            dests = train_edge_index[1][train_edge_type == r]
            sources = train_edge_index[0][train_edge_type == r]
            messages = embeddings[sources] @ rgnc_weights[l][r] + rgnc_biases[l][r]
            layer_messages.append((messages, dests))
            
        messages, dests =  zip(*layer_messages)
        messages = torch.vstack(messages)
        dests = torch.hstack(dests)
        hidden = relu(torch_scatter.scatter_sum(messages, dests, dim=0))
        output = hidden
        print(hidden.shape)
            # print(s.shape, N)
            # exit()
        #     rel_sources = train_edge_index[:,0][train_edge_type == r]
        #     hidden[rel_sources] += embeddings[rel_dests] @ rgnc_weights[l][r] + rgnc_biases[l][r]
        #     tmp = ([:,0] @ W1[r] + B1[r] ).relu_() @ W1[embeddings[rel_dests]r] + B1[r]
        #     res[train_edge_type == r] = tmp


        # s = torch_scatter.scatter_sum(res, train_edge_index[:,1], dim=0)
        # s.sum().backward()
        
        
    opt.step()