# %%
import wandb
import torch, torch_geometric
from tqdm import tqdm 
from models import *

run = wandb.init(project="simple-link-pred", entity="link-prediction-gnn", reinit=True)


wandb.config = {
    "dataset": 'FB15k-237',
    "learning_rate": 0.0001,
    "epochs": 100,
    "embeddings_size": 16,
    "n_pos": 256,
    "corrupted_obj_ratio": 1,
    "corrupted_rel_ratio": 1,
    "n_layers": 1,
    "model": "rgcn" # "embeddings" or "rgcn" 
        }



dataset = torch_geometric.datasets.RelLinkPredDataset(  wandb.config['dataset'],
                                                        wandb.config['dataset'])

Model = MODELS[wandb.config['model']]

train_dataset = torch.utils.data.TensorDataset(dataset.data['train_edge_index'].T, dataset.data['train_edge_type'])
test_dataset = torch.utils.data.TensorDataset(dataset.data['test_edge_index'].T, dataset.data['test_edge_type'])

model = Model(dataset.data.num_nodes, 
                wandb.config['embeddings_size'], 
                dataset.num_relations // 2, 
                train_dataset,
                wandb.config['n_layers'])


optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config['learning_rate'])
criterion = torch.nn.BCEWithLogitsLoss()

# wandb.watch(model)




for epoch in range(wandb.config['epochs']):
    epoch_loss = 0
    # Train cycle
    for pos_batch_index, batch_rel in tqdm(torch.utils.data.DataLoader( train_dataset, wandb.config['n_pos'], shuffle=True, drop_last=True)):
        neg_obj_index = torch.randint(dataset.data.num_nodes, ( wandb.config['n_pos'] * wandb.config['corrupted_obj_ratio'], ) )
        neg_rel_index = torch.randint(dataset.num_relations // 2, ( wandb.config['n_pos'] * wandb.config['corrupted_rel_ratio'], ) )

        optimizer.zero_grad()

        pos_subj_index, pos_obj_index = pos_batch_index.T
        
        pos_scores = model(pos_subj_index, batch_rel, pos_obj_index )
        neg_scores_obj = model(pos_subj_index, batch_rel, neg_obj_index ) # corrupt target
        neg_scores_rel = model(pos_subj_index, neg_rel_index, pos_obj_index ) # corrupt relation
        
        scores = torch.stack((pos_scores, neg_scores_obj, neg_scores_rel)).view(-1)
        targets = torch.stack((torch.ones_like(pos_scores), torch.zeros_like(neg_scores_obj), torch.zeros_like(neg_scores_rel))).view(-1)
        
        loss = criterion(scores, targets)
        epoch_loss += loss.detach()
        
        loss.backward()
        optimizer.step()

    if epoch % 3 == 0:
        hits, count = {1:0, 3:0, 10:0}, 0
        # Test cycle
        for i, (batch_index, batch_rel) in enumerate(tqdm(torch.utils.data.DataLoader( test_dataset, wandb.config['n_pos'],  shuffle=True))):
            
            batch_subj_index, batch_obj_index = batch_index.T.detach()
            edge_scores = model( batch_subj_index, batch_rel )  # output has size batch_size * n_nodes
            edge_scores[:,batch_subj_index] = 0
            edge_ranks = edge_scores.argsort(-1, descending=True)
            for k in hits.keys():
                hits[k] += (edge_ranks.T[:k] == batch_obj_index).sum()
            count += len(batch_index) 
    
        wandb.log({"loss": epoch_loss, **{f"hit@{k}": v/count for k,v in hits.items()} })
    else:
        wandb.log({"loss": epoch_loss} )
    
        
run.finish()



