from datetime import datetime
import wandb, os

config = {
    "dataset": 'FB15k-237',
    "learning_rate": 0.01,
    "reg": float(os.environ.get('REG', '0.01')),
    "epochs": int(os.environ.get('EPOCHS', '500')),
    "embeddings_size": int(os.environ.get('SIZE', '256')),
    "batch_size": int(os.environ.get('BATCH_SIZE', '1')),  # int or -1 for 'full' 
    "n_layers": int(os.environ.get('LAYERS', '1')),
    'limit_val_batches': 0.5,
    'check_val_every_n_epoch': 30,
        }

wandb.init( tags=[f'{k}={str(v)}' for k,v in config.items()], config=config, entity='link-prediction-gnn' )

DEVICE=os.environ.get('DEVICE', 'cpu')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath=wandb.run.dir,
    filename='{epoch}_{hit@10:.2f}' \
        + '_reg=' + str(wandb.config['reg']) \
        + '_layers=' + str(wandb.config['n_layers']) \
        + '_size=' + str(wandb.config['embeddings_size']) \
        + '_time=' + str(datetime.now())
    )