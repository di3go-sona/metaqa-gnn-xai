#%%
import torch

class QuestionEncoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        
        # self.hidden_size = hidden_size
        self.output_size = output_size
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'prajjwal1/bert-medium')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'prajjwal1/bert-medium')
        
        self.linear = torch.nn.Linear(self.hidden_size, self.output_size)
        


    def forward(self, questions):
      tokens = torch.tensor(self.tokenizer(questions, add_special_tokens=True, padding=True)['input_ids'], device=self.model.device)
      
      hidden = self.model(tokens)['pooler_output']
      out = self.linear(hidden)
      return out
  
    @property
    def vocab_size(self):
        return self.model.config.vocab_size
    
    @property
    def hidden_size(self):
        return self.model.config.hidden_size
      
if __name__ == '__main__':
  from metaqa.dataset.datasets import QA_Dataset
  qa_ds = QA_Dataset('../dataset')
  q1 = qa_ds[0][0]
  q2 = qa_ds[0][0]
  model = QuestionEncoder(256)
  out = model([q1, q2])
  print(out.shape)




# class TextEncoder(torch.nn.Module):
#     def __init__(self, vocab_size, embedding_size, hidden_size,  stack='short-term'):
#         super().__init__()

#         self.embeddings = Embedding(vocab_size+1, embedding_size)
#         self.stack = stack  
#         self.hidden_size = hidden_size
        
#         self.lstm =LSTM(embedding_size, hidden_size, batch_first=True)

#         self.identity =Identity()
        
#         self.forward_stack =Sequential(
#           self.identity,
#         )

#     def forward(self, toks):
#       hidden = tuple(torch.rand(1,toks.shape[0],self.hidden_size, device=toks.device) for _ in range(2))

#       X = self.embeddings(toks)
#       out, hidden = self.lstm(X, hidden)
#       if self.stack == 'short-term':
#         return self.forward_stack(hidden[0])
#       if self.stack == 'long-term':
#         return self.forward_stack(hidden[1])
#       if self.stack == 'both':  
#         return self.forward_stack(torch.concat(hidden,2))

# %%
