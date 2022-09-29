#%%
import glob
from globals import QA_CHECKPOINTS_PATH, EMBEDDINGS_CHECKPOINTS_PATH

def get_model_path(model_name):
    candidates = glob.glob(f'{QA_CHECKPOINTS_PATH}/**/{model_name}*')
    if len(candidates) > 1:
        assert f"Too many matches for model named {model_name}"
    return candidates[0]

def get_model_path(model_name):
    candidates = glob.glob('checkpoints/qa/**/' + model_name + '*')
    if len(candidates) > 1:
        assert f"Too many matches for model named {model_name}"
    return candidates[0]

def list_qa_models():
    candidates = glob.glob(f'{QA_CHECKPOINTS_PATH}/**/*.ckpt')
    return candidates

def list_embeddings_models():
    candidates = glob.glob(f'{EMBEDDINGS_CHECKPOINTS_PATH}/**.ckpt')
    return candidates

print(list_qa_models())
print(list_embeddings_models())
# %%
