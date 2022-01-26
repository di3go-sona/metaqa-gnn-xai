import wandb

wandb.config = {
    "dataset": 'FB15k-237',
    "learning_rate": 0.01,
    "reg": 0.01,
    "epochs": 1500,
    "embeddings_size": 500,
    "batch_size": 'full', # int of 'full' 
    "n_layers": 2,
    'limit_val_batches': 1,
    'check_val_every_n_epoch': 50,
    
        }



DEVICE='cuda'