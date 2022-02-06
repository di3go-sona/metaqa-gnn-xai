
from itertools import repeat
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset
import torch, torch_geometric, wandb
import pytorch_lightning as pl

class FB15KData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        dataset = torch_geometric.datasets.RelLinkPredDataset(  'data', 'FB15k-237' )
        
        data = dataset[0]
        self.num_nodes = data.num_nodes
        self.num_relations = dataset.num_relations

        self.train_data = TensorDataset(data.train_edge_index.T, data.train_edge_type)
        self.test_data = TensorDataset(data.test_edge_index.T, data.test_edge_type)
        self.valid_data = TensorDataset(data.valid_edge_index.T, data.valid_edge_type)
        self.all_data = TensorDataset( torch.cat((data.train_edge_index.T, data.test_edge_index.T, data.valid_edge_index.T)),
                                    torch.cat((data.train_edge_type, data.test_edge_type, data.valid_edge_type)))

        
    def train_dataloader(self):
        return CombinedLoader([
                        DataLoader( self.train_data,  
                        len(self.train_data) if wandb.config['batch_size'] == -1 else wandb.config['batch_size'], 
                        drop_last=True, 
                        shuffle=True), 
                        repeat(self.train_data)])
        # return DataLoader( self.train_data,  len(self.train_data), drop_last=True, shuffle=True)
    def val_dataloader(self):
        return CombinedLoader([ 
            DataLoader( self.test_data, batch_size=64, shuffle=True),
            DataLoader( self.train_data, batch_size=len(self.train_data)),
            DataLoader( self.all_data, batch_size=len(self.all_data))
        ], 'max_size_cycle')
        

    def test_dataloader(self):
        return self.val_dataloader()