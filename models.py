import torch, torch_geometric

class EmbeddingsModel(torch.nn.Module):
    def __init__(self, num_nodes, embedding_size, num_relations, *args ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.num_relations = num_relations


        embeddings = torch.rand(num_nodes, embedding_size, requires_grad=True)
        self._embeddings = torch.nn.Parameter( embeddings) 

        relations = torch.rand( num_relations, embedding_size, requires_grad=True)
        self.relations = torch.nn.Parameter( relations) 


    def embeddings(self):
        return self._embeddings
        
    # if forward is called with 2 args it returns score for < subj, obj > for all rels
    # otherwise it will return return score for subj and all obj
    def forward(self, batch_subj_index, rel_index, batch_obj_index=None):
        
        embeddings = self.embeddings()
        batch_subj_emb = embeddings[batch_subj_index]
        batch_rel_emb = self.relations[rel_index]

        if batch_obj_index is None:
            return  (batch_subj_emb * batch_rel_emb * embeddings.unsqueeze(1) ).sum(-1).T
        else:
            batch_obj_emb = embeddings[batch_obj_index]
            return  (batch_subj_emb * batch_rel_emb * batch_obj_emb ).sum(-1)



class RGCNModel(EmbeddingsModel):
    def __init__(self, num_nodes, embedding_size, num_relations, train_dataset, layers=1) -> None:
        super().__init__(num_nodes, embedding_size, num_relations)

        self.rgcn = torch_geometric.nn.RGCNConv(embedding_size, embedding_size, num_relations)
        self.edge_index, self.edge_type = train_dataset[...]
        self.layers = layers

    # if forward is called with 2 args it returns score for < subj, obj > for all rels
    # otherwise it will return return score for subj and all obj


    def embeddings(self):
        embeddings = self._embeddings
        for _ in range(self.layers):
            embeddings = self.rgcn(embeddings, self.edge_index.T, self.edge_type)
        return embeddings
    
    
    def forward(self, batch_subj_index, rel_index, batch_obj_index=None):
        
        embeddings = self.embeddings()
        batch_subj_emb = embeddings[batch_subj_index]
        batch_rel_emb = self.relations[rel_index]

        if batch_obj_index is None:
            return  (batch_subj_emb * batch_rel_emb * embeddings.unsqueeze(1) ).sum(-1).T
        else:
            batch_obj_emb = embeddings[batch_obj_index]
            return  (batch_subj_emb * batch_rel_emb * batch_obj_emb ).sum(-1)

        

MODELS = {
    "embeddings": EmbeddingsModel,
    "rgcn": RGCNModel
}