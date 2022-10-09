#%%
from class_resolver import UnexpectedKeywordError
from models.rgcnqa import *
from models.embeddings import *
from transformers import AutoTokenizer
from globals import *
from tqdm import tqdm 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
checkpoints_set = {
#     ('Funnel',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=10.ckpt',
#     ('Straight',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=19.ckpt',
#     ('Concatenated',1): '../data/checkpoints/qa/1-hops/RGCNQA|1_hops|36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_bias|no_root|epoch=13.ckpt',
#     ('Funnel',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|72>36>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=37.ckpt',
#     ('Straight',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=39.ckpt',
#     ('Concatenated',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=61.ckpt',
#   ('Straight|Bert-frozen',2): '../data/checkpoints/qa/2-hops/RGCNQA|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|bert_frozen|epoch=129.ckpt',
#   ('Funnel',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|72>36>18>1|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=974.ckpt',
#   ('Straight',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|36>36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=437.ckpt',
  ('Concatenated',3): '../data/checkpoints/qa/3-hops/QA_RGCN|3_hops|36>36>36>36C|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=549.ckpt',
}

# Evaluate accuracy
for m, h in checkpoints_set:
    qa_data = QAData(METAQA_PATH, [h], tokenizer, train_batch_size= 128, val_batch_size= 4, use_ntm= True)
    try:
        qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)
    except UnexpectedKeywordError:
        qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)
    hits_at_k = [ h  for batch in tqdm(list(qa_data.test_dataloader())) for h in qa_model.evaluate_batch(batch)]

    with open('../data/robustness.out', 'a') as fout:
        hits_at_k = sum(hits_at_k)/len(hits_at_k)
        fout.write(f'{m}|{h}|{hits_at_k}')
# %%
