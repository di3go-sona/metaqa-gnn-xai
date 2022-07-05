#%%
import matplotlib.pyplot as plt
from dataset.dataloaders import QuestionAnsweringData
import torch
import networkx as nx

data  = QuestionAnsweringData('dataset', [1])
questions_data = data.train_ds
kb_data = data.kb_ds

src, dst, rel = torch.stack(tuple(data.kb_ds)).T.tolist()

graph = nx.DiGraph()
graph.add_edges_from(zip(src, dst))



# %%
import gradio as gr

inputs = [
    gr.Number(value=1, precision=0,  max=len(questions_data), label='Question id')
]
outputs = [
    gr.Text(label='Question'),
    gr.HighlightedText(label='Answers'),
    gr.Plot(label='QA Graph'),
] 

def get_graph(graph, root_id, hops):
    subgraph_nodes = nx.dfs_preorder_nodes(graph.to_undirected(), root_id, hops)
    return nx.subgraph(graph, subgraph_nodes)
    

# %%
def get_question(question_id):
    _text, root_id, answers_mask, hops =  questions_data[question_id]
    root_name = questions_data.entities_names[root_id]
    answers_ids = torch.arange(len(questions_data.entities_names))[answers_mask.bool()]
    answers_names = [questions_data.entities_names[id] for id in answers_ids.tolist()]
    
    question_repr = _text.strip().replace('root', root_name) + ' ?'
    answers_repr = [(a,) for a in answers_names]
    
    fig = plt.figure(figsize=(16,16))
    subgraph = get_graph(graph, root_id, hops)
    pos = nx.kamada_kawai_layout(subgraph)
    lab = {k:questions_data.entities_names[k] for k in subgraph.nodes}
    
    nx.draw_networkx_nodes(subgraph, pos, node_size=1000, alpha=0.5)
    nx.draw_networkx_labels(subgraph, pos, lab)
    nx.draw_networkx_edges(subgraph, pos)
    
    return question_repr, answers_repr, fig


app = gr.Interface(fn=get_question, inputs=inputs, outputs=outputs)
app.launch()
# %%
