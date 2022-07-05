#%%
import torch
import transformers
from train_qa_v2 import QAModel, QAData
from transformers import AutoTokenizer, BertModel



# import click
# @click.command()
# @click.argument('model-path', type=str)
def run(model_path):
    hops = int (model_path.rsplit('/', 2)[1][0])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    data = QAData('dataset', [1], tokenizer, use_ntm=False)
    model = QAModel.load_from_checkpoint(model_path) 
    
    batch = next(iter(data.val_dataloader()))
    print(batch)
    # print(hops)
    
    
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# data = QAData('dataset', [1], tokenizer, train_batch_size=train_batch_size, val_batch_size=val_batch_size, use_ntm=ntm)
# kge_model = KGEModel.load_from_checkpoint(checkpoints/qa/3-hops/DistMultInteraction()|256|epoch=4.ckpt)
# model = QAModel.load_from_checkpoint()




#%%
if __name__ == '__main__':
    run()