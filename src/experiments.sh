# NO CONCATENATION - DIFFERENT SHAPES 
### 1-HOPS
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|1" --patience 3 --aggr mean --train-batch-size 42 --accumulate-batches 2 --hops 1 --noroot 
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|36" --patience 3 --aggr mean --train-batch-size 42  --accumulate-batches 2 --hops 1 --noroot 
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|36" --patience 3 --aggr mean --train-batch-size 42 --accumulate-batches 2 --hops 1 --noroot --concat-layers

### 2-HOPS
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "72|36|1" --patience 50 --aggr mean --train-batch-size 42 --accumulate-batches 2 --noroot 
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|36|36" --patience 50 --aggr mean --train-batch-size 42 --accumulate-batches 2 --noroot 
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|36|36" --patience 50 --aggr mean --train-batch-size 42 --accumulate-batches 2 --noroot --concat-layers

### 3-HOPS
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "72|36|18|1" --patience 1000 --aggr mean --train-batch-size --accumulate-batches 128 --limit-train-batches 4096 2 --val-batch-size 4 --hops 3 --noroot
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|36|36|1" --patience 1000 --aggr mean --train-batch-size --accumulate-batches 128 --limit-train-batches 4096 2 --val-batch-size 4 --hops 3 --noroot
# python3 train.py --fast --nobias --l2 0 --lr 0.0001 --layer-sizes "36|36|36|1" --patience 1000 --aggr mean --train-batch-size --accumulate-batches 128 --limit-train-batches 4096 2 --val-batch-size 4 --hops 3 --noroot

# WITH PRETRAINED EMB CONTATENATION - ONLY ONE SHAPE
### 2-hops
# python3 train.py --fast --l2 0 --lr 0.0001 --layer-sizes "36|36|36" --patience 100 --aggr mean --accumulate-batches 3 --noroot --train-batch-size 42 --concat-embeddings all        
# python3 train.py --fast --l2 0 --lr 0.0001 --layer-sizes "36|36|36" --patience 50 --aggr mean --accumulate-batches 3 --noroot --train-batch-size 42 --concat-embeddings all+head  
# python3 train.py --fast --l2 0 --lr 0.0001 --layer-sizes "36|36|36" --patience 50 --aggr mean --accumulate-batches 3 --noroot --train-batch-size 42 --concat-embeddings head

# WITH PRETRAINED EMB CONTATENATION - ONLY ONE SHAPE
