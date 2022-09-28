import glob
def get_model_path(model_name):
    candidates = glob.glob('checkpoints/qa/**/' + model_name + '*')
    if len(candidates) > 1:
        assert f"Too many matches for model named {model_name}"
    return candidates[0]





