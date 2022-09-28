import torch

class GraphEncoder(torch.nn.Module):
    def __init__(self, n_nodes, n_rels, embeddings_size) -> None:
        super().__init__()
        
        # self.n_embeddings = n_embeddings
        # self.embeddings_size = embeddings_size
        
        self.nodes_embeddings = torch.nn.Embedding(n_nodes, embeddings_size )
        self.rels_embeddings = torch.nn.Embedding(n_rels, embeddings_size )

        
    def forward(self, X):
        src, dst, rels = X
        return torch.cat((
            self.nodes_embeddings(src).unsqueeze(0),
            self.nodes_embeddings(dst).unsqueeze(0),
            self.rels_embeddings(rels).unsqueeze(0)
        ), 0)
        
    @property
    def n_nodes(self):
        return self.nodes_embeddings.num_embeddings
        
    @property
    def n_relations(self):
        return self.rels_embeddings.num_embeddings 
        
    @property
    def embeddings_size(self):
        return self.nodes_embeddings.embedding_dim