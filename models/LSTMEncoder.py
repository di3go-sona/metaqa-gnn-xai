import torch
from torch.nn import Embedding, LSTM, Identity, Sequential

class LSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,  stack='short-term'):
        super().__init__()

        self.embeddings = Embedding(vocab_size+1, embedding_size)
        self.stack = stack  
        self.hidden_size = hidden_size
        
        self.lstm =LSTM(embedding_size, hidden_size, batch_first=True)

        self.identity =Identity()
        
        self.forward_stack =Sequential(
          self.identity,
        )

    def forward(self, toks):
      hidden = tuple(torch.rand(1,toks.shape[0],self.hidden_size, device=toks.device) for _ in range(2))

      X = self.embeddings(toks)
      out, hidden = self.lstm(X, hidden)
      if self.stack == 'short-term':
        return self.forward_stack(hidden[0])
      if self.stack == 'long-term':
        return self.forward_stack(hidden[1])
      if self.stack == 'both':  
        return self.forward_stack(torch.concat(hidden,2))
