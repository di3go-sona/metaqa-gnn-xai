import torch, torch_scatter
from torch.nn import ParameterList, ReLU, Parameter


class RGCNEncoder(torch.nn.Module):
    def __init__(self, embeddings, n_relations, configs, freeze_embeddings=False):
        super().__init__()
        self.n_relations = n_relations
        self.n_nodes, self.embeddings_size = embeddings.shape
        self.n_layers = configs['n_layers']
        self.p_dropout = configs['dropout']

        self.dropout = torch.nn.Dropout(self.p_dropout)
        
        self.embeddings = embeddings if freeze_embeddings else Parameter(embeddings)
        self.rgcn_weights = ParameterList( [ Parameter(torch.rand(self.n_relations * 2 + 1, 
                                                                    self.embeddings_size,
                                                                    self.embeddings_size, requires_grad=True)) for _ in range(self.n_layers) ] )
        self.rgcn_biases = ParameterList( [ Parameter(torch.rand(self.n_relations * 2 + 1, 
                                                                    self.embeddings_size, requires_grad=True)) for _ in range(self.n_layers) ] )
        self.relu = ReLU()

    def get_messages(self, embeddings_source, index_source, l, inverse=False):
        for r in range(self.n_relations):
            messages = embeddings_source[index_source] @ self.rgcn_weights[l][r + self.n_relations * inverse] + self.rgcn_biases[l][r + self.n_relations * inverse]
            yield messages
                        
                        
    def forward(self, edge_index, edge_type):
        edge_index = edge_index
        edge_type = edge_type
        
        # print(edge_index.shape, edge_type.shape)
        
        
        output = self.dropout(self.embeddings) # self.drop_embeddings(self.embeddings)


        for l in range(self.n_layers):
            hidden = output.clone() 
            # print(hidden.shape)
            # print(self.rgcn_weights[l][-1].shape)
            # print(self.rgcn_biases[l][-1].shape)
            hidden =  (hidden.unsqueeze(1) @  self.rgcn_weights[l][-1]).squeeze() + self.rgcn_biases[l][-1]
            # print(hidden.shape)
            for r in range(self.n_relations):
                for inverse in [0,1]:
                    r_dests = edge_index[inverse-0][edge_type == r]
                    r_sources = edge_index[1-inverse][edge_type == r]
                    # if l == 0:
                    messages = torch.nn.functional.embedding(r_sources, output) @ self.rgcn_weights[l][r + self.n_relations * inverse] + self.rgcn_biases[l][r + self.n_relations * inverse]

                    m = torch.vstack(tuple(messages)).to(hidden.device)
                    d = torch.hstack(tuple(r_dests)).to(hidden.device)
                    
                    print(l,r,m.shape, d.shape)
                    hidden += torch_scatter.scatter_mean(m, d, dim=0, out=hidden)

            if l + 1 < self.n_layers :
                hidden = self.relu(hidden)
                hidden = self.dropout(hidden)
            output = hidden
            
        
        return output