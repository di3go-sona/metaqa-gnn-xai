#%%
from models.rgcnqa import *
from models.embeddings import *
from transformers import AutoTokenizer
from globals import *
from tqdm import tqdm 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
from checkpoints import *

#%%
# Evaluate accuracy //TODO Fix the **** biasfor m, h in checkpoints_set:
for m, h in checkpoints_set:
    qa_data = QAData(METAQA_PATH, [h], tokenizer, train_batch_size= 128, val_batch_size= 4, use_ntm= True)
    try:
        qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=False, verbose=False)
    except Exception:
        qa_model = RGCNQA.load_from_checkpoint(checkpoints_set[( m, h)], bias=True, verbose=False)

    hits_at_k = [ h  for batch in tqdm(list(qa_data.test_dataloader())) for h in qa_model.evaluate_batch(batch)]
    with open('../data/robustness.out', 'a') as fout:
        hits_at_k = sum(hits_at_k)/len(hits_at_k)
        fout.write(f'\n{m}|{h}|{hits_at_k}')
# %%
