#%%
from models.rgcnqa import *
from models.embeddings import *
from transformers import AutoTokenizer
from globals import *
from tqdm import tqdm 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
checkpoints_set = {
    ('Funnel',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=10.ckpt',
    ('Straight',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=19.ckpt',
    ('Concatenated',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=13.ckpt',
}
checkpoints_set = {
    ('Funnel',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|72>36>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=37.ckpt',
    ('Straight',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=39.ckpt',
    ('Concatenated',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=61.ckpt',
}
# checkpoints_set = {
#     ('Funnel',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|72>36>18>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=247.ckpt',
#     ('Straight',3): 'data/checkpoints/qa/3-hops/QA_RGCN|3_hops|36>36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=437.ckpt',
#     ('Concatenated',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|36>36>36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=549.ckpt',
# }


#%%
# Evaluate accuracy
# for m, h in checkpoints_set:
#     qa_data = QAData(METAQA_PATH, [h], tokenizer, train_batch_size= 128, val_batch_size= 16, use_ntm= False)
#     qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)

#     hits_at_k = [ h  for batch in tqdm(list(qa_data.test_dataloader())) for h in qa_model.evaluate_batch(batch)]
#     print(m, h, sum(hits_at_k)/len(hits_at_k))

for m, h in checkpoints_set:
    qa_data = QAData(METAQA_PATH, [h], tokenizer, train_batch_size= 128, val_batch_size= 16, use_ntm= False)
    qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)

    for batch in tqdm(list(qa_data.test_dataloader())):
        for sample in batch:
            print(sample)
            break
    x = None
    edge_index = None
    if x is None:
        x = qa_model.nodes_emb
    if edge_index is None:
        edge_index = qa_model.edge_index
    
    
    # Compute a binary mask with all zeros except the src node index
    src_index_mask = (torch.arange(x.size(0), device = qa_model.device, requires_grad=False) == src_index).unsqueeze(-1).float()
    # Sum question to x
    z = x + (src_index_mask * question_emb )
    if qa_model.concat_layers:
        layers = [z]


    subset, _edge_index, inv, edge_mask = k_hop_subgraph(src_index.item(), qa_model.hops, edge_index[:,[0,2]].T)
    _edge_type = edge_index[:,1][edge_mask]


    if qa_model.n_layers > 1:

        if  qa_model.concat_embeddings in ['all', 'all+head', 'first'] and qa_model.layers[0] != 1:
            z = torch.concat( [z, qa_model.nodes_pretrained_emb], axis=-1)

        z = qa_model.rgcn1(z, _edge_index, _edge_type)

        if qa_model.concat_layers:
            layers.append(z)
    
    if qa_model.n_layers > 2 :
        if qa_model.concat_embeddings in ['all', 'all+head'] and qa_model.layers[1] != 1:
            z = torch.concat( [z, qa_model.nodes_pretrained_emb], axis=-1)
        z = qa_model.rgcn2(z.relu(), _edge_index, _edge_type)
        if qa_model.concat_layers:
            layers.append(z)
        
    if qa_model.n_layers > 3 :
        if qa_model.concat_embeddings in ['all', 'all+head'] and qa_model.layers[2] != 1:
            z = torch.concat( [z, qa_model.nodes_pretrained_emb], axis=-1)
        z = qa_model.rgcn3(z.relu(), _edge_index, _edge_type)
        if qa_model.concat_layers:
            layers.append(z)
    
    if qa_model.concat_embeddings in ['head', 'all+head']:
        z = torch.concat( [z, qa_model.nodes_pretrained_emb], axis=-1)
        
    if qa_model.concat_layers:
        out = qa_model.rgcn_head(
            torch.concat( layers, -1)
        )
    else:
        out = qa_model.rgcn_head(z)

# test_dataloader = qa_data.val_dataloader()
# trainer.test(qa_qa_model, 
#              test_dataloaders = test_dataloader,
#              verbose = True
#              )


# %%
