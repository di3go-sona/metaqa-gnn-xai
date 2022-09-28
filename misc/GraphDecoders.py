import torch 

class TransEDecoder(torch.nn.Module):
    def __init__(self, margin=1.0) -> None:
        super().__init__()
        self.criterion = torch.nn.MarginRankingLoss(margin=margin)
    
    def loss(self, positive_triplets, negative_triplets):

        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        
        target = torch.tensor([-1], dtype=torch.long, device=positive_triplets.device)

        return self.criterion(positive_distances, negative_distances, target)

    
    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        # assert triplets.size()[1] == 3
        src, dst, type = triplets

        return (dst + type - src).norm(p=1, dim=1)
    
    def forward(self, triplets,):
        return self._distance(triplets)
            


    
class DistMultDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def loss(self, positive_triplets, negative_triplets):
        z = self.forward(positive_triplets)
        nz = self.forward(negative_triplets)
        return self.criterion(z, torch.ones_like(z)) + self.criterion(nz, torch.zeros_like(nz))
        
    def forward(self, triplets,):
        # print(triplets.shape)
        src, dst, type = triplets
        return (src*type*dst).sum(-1)
    