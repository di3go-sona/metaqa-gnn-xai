#%%

from models.rgcnqa import *
from models.embeddings import *

embeddings_model = KGEModel.load_from_checkpoint('../data/checkpoints/embeddings/TransEInteraction()|32|epoch=4999|.ckpt')

# qa_model = RGCNQA.load_from_checkpoint('../data/checkpoints/qa/2-hops/QA_RGCN|2_hops|36>36>36|lr=0.0001|l2=0.0|mean_pool|zeroed|no_root|epoch=279.ckpt')

# %%
    # trainer = pl.Trainer( 
    #     accelerator= 'gpu',
    #     gpus= -1,
    #     callbacks= [checkpoint_callback, stopping_callback],
    #     logger= logger, 
    #     log_every_n_steps= 1,
    #     limit_val_batches= limit_val_batches,
    #     limit_train_batches= limit_train_batches,
    #     accumulate_grad_batches= accumulate_batches,
    #     val_check_interval= 1.0, 
    #     check_val_every_n_epoch= 1,
    #     max_epochs= epochs,
    #     )