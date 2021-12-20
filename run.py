# %%
import wandb
import torch, torch_geometric
from tqdm import tqdm 
from models import *

run = wandb.init(project="simple-link-pred", entity="link-prediction-gnn", reinit=True)

wandb.config = {
    "dataset": 'FB15k-237',
    "learning_rate": 0.01,
    "epochs": 100,
    "embeddings_size": 64,
    "n_pos": 256,
    "neg_pos_ratio": 1,
    "model": "embeddings" # "embeddings" or "rgcn" 
        }



dataset = torch_geometric.datasets.RelLinkPredDataset(  wandb.config['dataset'],
                                                        wandb.config['dataset'])

Model = MODELS[wandb.config['model']]

model = Model(dataset.data.num_nodes, 
                wandb.config['embeddings_size'], 
                dataset.num_relations)

# wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['learning_rate'])
criterion = torch.nn.BCEWithLogitsLoss()


train_f = dataset.data['train_edge_type'] == 10
test_f = dataset.data['test_edge_type'] == 10
train_dataset = torch.utils.data.TensorDataset(dataset.data['train_edge_index'].T[train_f], dataset.data['train_edge_type'][train_f])
test_dataset = torch.utils.data.TensorDataset(dataset.data['test_edge_index'].T[test_f], dataset.data['test_edge_type'][test_f])

# train_dataset = torch.utils.data.TensorDataset(dataset.data['train_edge_index'].T, dataset.data['train_edge_type'])
# test_dataset = torch.utils.data.TensorDataset(dataset.data['test_edge_index'].T, dataset.data['test_edge_type'])


for epoch in range(wandb.config['epochs']):
    loss, hits, count = 0, 0, 0

    # Test cycle
    for i, (batch_index, batch_type) in enumerate(tqdm(torch.utils.data.DataLoader( test_dataset, wandb.config['n_pos']))):
        nodes_emb = model(None, batch_index.T, batch_type)
        test_batch_index = batch_index
        test_batch_emb = nodes_emb[test_batch_index[...,0]]

        nodes_emb = nodes_emb / nodes_emb.norm(dim=-1, keepdim=True)
        test_batch_emb = test_batch_emb / test_batch_emb.norm(dim=-1, keepdim=True)

        test_batch_scores = test_batch_emb @ nodes_emb.T
        test_batch_rank = test_batch_scores.argsort(-1, descending=True)
    

        hits += (test_batch_rank[...,:10].T == test_batch_index[..., 1]).sum()
        # print(test_batch_index[..., 1])
        # print(test_batch_rank[...,:10])
        # print((test_batch_rank[...,:10].T == test_batch_index[..., 1]).sum())
        # exit()
        count += len(batch_index)

        # early stop
        # if i == 5:
        #     break

    # Train cycle
    for batch_index, batch_type in tqdm(torch.utils.data.DataLoader( train_dataset, wandb.config['n_pos'], shuffle=True)):
        nodes_emb = model(None, batch_index.T, batch_type) 

        optimizer.zero_grad()

        pos_batch_index = batch_index
        neg_batch_index = torch.randint(dataset.data.num_nodes, ( wandb.config['n_pos'] * wandb.config['neg_pos_ratio'], 2), )

        pos_batch_emb = nodes_emb[pos_batch_index]
        neg_batch_emb = nodes_emb[neg_batch_index]

        pos_batch_emb = pos_batch_emb / pos_batch_emb.norm(dim=-1, keepdim=True)
        neg_batch_emb = neg_batch_emb / neg_batch_emb.norm(dim=-1, keepdim=True)

        pos_scores = pos_batch_emb.prod(1).sum(1)
        neg_scores = neg_batch_emb.prod(1).sum(1)
        
        # loss = - pos_scores.mean() + neg_scores.mean() 
        loss = criterion(pos_scores, torch.ones_like(pos_scores)) + criterion(neg_scores, torch.zeros_like(neg_scores))
        
        loss.backward()
        optimizer.step()

    
    wandb.log({"loss": loss, "hit@10": hits/count*100} )
    
        
run.finish()



