from datetime import datetime
import wandb, os

wandb.config = {
    "dataset": 'FB15k-237',
    "learning_rate": 0.01,
    "reg": float(os.environ.get('REG', '0.01')),
    "epochs": 500,
    "embeddings_size": 256,
    "batch_size": int(os.environ.get('BATCH_SIZE', '1')),  # int or -1 for 'full' 
    "n_layers": int(os.environ.get('LAYERS', '1')),
    'limit_val_batches': 0.5,
    'check_val_every_n_epoch': 50,
        }



DEVICE=os.environ.get('DEVICE', 'cpu')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',
    filename='{epoch}_{hit@10:.2f}' + f'_reg=' + str(wandb.config['reg']) + '_layers=' + str(wandb.config['n_layers']) + '_time=' + str(datetime.now())
    )