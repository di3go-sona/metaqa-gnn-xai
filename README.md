# rgcn-link-prediction
A question answering model based on Knowledge Graph Embeddings and Graph Neural Networks


Train the KG embeddings 

    python3 train_embeddings_v2.py --emb-size 256 --interaction complex --epochs 50

Train the QA model 

    python3 train_qa_v2.py --emb-size 256  --epochs 10
