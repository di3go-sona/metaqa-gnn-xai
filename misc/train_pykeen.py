#%%
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
from pykeen.trackers import WANDBResultTracker

from typing import TYPE_CHECKING, Any, Mapping, Optional
from pykeen.utils import flatten_dictionary

from pprint import pprint

class CustomWANDBResultTracker(WANDBResultTracker):
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:  # noqa: D102
        
        pprint(metrics)
        pprint(step)
        
        if self.run is None:
            raise AssertionError("start_run must be called before logging any metrics")
        
        metrics = flatten_dictionary(dictionary=metrics, prefix=prefix)
        
        filtered_metrics = {k:v for k,v in metrics.items() if k in ['both.realistic.hits_at']}
        self.run.log(metrics, step=step)
    

    # def on_batch(self, epoch: int, batch, batch_loss: float):
    #     print(epoch, batch, batch_loss)
        
#%%
def load_triplets():
    relations_to_ids_path = 'dataset/kb_relations_dict.txt'
    entities_to_ids_path = 'dataset/kb_entity_dict.txt'

    df = pd.read_csv(relations_to_ids_path, sep='\t', header=None, index_col='id', names=['id', 'name'])
    relation_to_id = {row['name']: index for index, row in df.iterrows() }


    df = pd.read_csv(entities_to_ids_path, sep='\t', header=None, index_col='id', names=['id', 'name'])
    entity_to_id = {row['name']: index for index, row in df.iterrows() }
    
    triplets = TriplesFactory.from_path('dataset/kb.tsv', relation_to_id=relation_to_id, entity_to_id=entity_to_id)
    
    return triplets


triplets_factory = load_triplets()
training, testing = triplets_factory.split([.9, .1])

#%%
for model in ['DistMult']:
    pipeline_result = hpo_pipeline(
        model=model,
        training=training, 
        validation=testing, 
        testing=testing,
        result_tracker=CustomWANDBResultTracker,
        result_tracker_kwargs=dict(
            project='misc',
            entity='link-prediction-gnn'
        ),
        n_jobs= 1,
        # stopper='early',
        # stopper_kwargs=dict(frequency=15, patience=3, relative_delta=0.002),
        
        
        
    )
# %%
