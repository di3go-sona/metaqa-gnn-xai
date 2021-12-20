import torch, torch_geometric

class EmbeddingsModel(torch.nn.Module):
    def __init__(self, num_nodes, embedding_size, num_relations) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.num_relations = num_relations

        embeddings = torch.rand((num_nodes, embedding_size), requires_grad=True)
        self.embeddings = torch.nn.Parameter( embeddings) 

    def forward(self, *args):
        return self.embeddings

MODELS = {
    "embeddings": EmbeddingsModel,
    "rgcn": torch_geometric.nn.RGCNConv
}