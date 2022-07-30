#%%
from re import sub
import torch, transformers
from train_joint import JointQAModel
from train_qa import QAModel, QAData
from transformers import AutoTokenizer, BertModel
import click, os, glob
import gradio as gr
import networkx as nx
import random
import matplotlib.pyplot as plt
from pyvis.network import Network

model = None 
hops = None
data = None
device = None
graph = None
#%%
# @click.command()
# @click.option('--cpu', default=False, show_default=True, is_flag=True)
# @click.option('--ckpt-folder', default='../checkpoints/*/*/*.ckpt', type=str)


def main(cpu, ckpt_folder, default_model='checkpoints/qa/2-hops/DistMultInteraction|256|epoch=499.ckpt'):
    global device
    
    device = 'cpu' if cpu else  'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", )
    # model = JointQAModel.load_from_checkpoint(default_model, map_location={'cuda:0': device, 'cpu': device })

    
    available_models = glob.glob(ckpt_folder)    
    
    def update_model(model_path):
        global model 
        global hops
        global data
        global device
        global graph
        
        model = JointQAModel.load_from_checkpoint(model_path, map_location={'cuda:0': device, 'cpu': device })
        
        if data is None or hops !=  int (model_path.rsplit('/', 2)[1][0]):
            hops = int (model_path.rsplit('/', 2)[1][0])
            data = QAData('dataset', [hops], tokenizer, use_ntm=False)
            

            
            raw_graph = [(data.entities_names[s], data.relations_names[r], data.entities_names[d]) for s,r,d in data.get_triples(add_inverse=False).tolist() ]
            graph = nx.DiGraph()
            for s,r,d in raw_graph:
                graph.add_edge(s,d, label=r)
            
        data_selector = gr.Slider.update(maximum=len(data.val_ds_unflattened), step=1,  label='Sample selector', visible=True )
        model_name_textbox = gr.Textbox.update(value=f"You selected model {model_path}", label='Selected Model', visible=True)
        
        return  data_selector

    def update_question(index):
        global model 
        global hops
        global data
        global device
        print(index)
        
        root, _, answers, question = data.val_ds_unflattened[int(index)]        
        root_name = data.entities_names[root]
        answers_names = [ data.entities_names[answer] for answer in answers ]
        question = data.tokenizer.decode(question).replace('root', root_name)

        question_selector = gr.Textbox.update(question, visible=True)
        answers_selector = gr.Radio.update(choices= answers_names, visible= True)
        
        predict_button = gr.Button.update(visible= True)
        plot_button = gr.Button.update(visible= True)
        
        return question_selector, answers_selector, predict_button, plot_button
    
    def predict(index):
        global data
        global device

        root, _, answers, question = data.val_ds_unflattened[int(index)]        
        preds = model.qa_validation_step(
            {'inference': 
                (torch.tensor(root, dtype=int, device=device).unsqueeze(-1), 
                torch.tensor(answers, dtype=int, device=device).unsqueeze(-1), 
                question.to(device).unsqueeze(-1)
                )
                }, -1)
        scores, topk = next(iter(preds))
        
        preds =  dict(
                    zip(
                        [ '✅ ' + data.entities_names[i] if i in answers  else '❌ ' + data.entities_names[i] for i in topk.indices.squeeze().tolist()  ], 
                        (topk.values.squeeze() / topk.values.max() ).tolist()
                        )
                    )
        predictions_box = gr.Label.update(value= preds, visible= True)
        return predictions_box
    
    def plot(index):
        global hops
        global graph
        
        root, _, answers, question = data.val_ds_unflattened[int(index)]
        preds = model.qa_validation_step(
            {'inference': 
                (torch.tensor(root, dtype=int, device=device).unsqueeze(-1), 
                torch.tensor(answers, dtype=int, device=device).unsqueeze(-1), 
                question.to(device).unsqueeze(-1)
                )
                }, -1)
        scores, topk = next(iter(preds))
        scores = scores.squeeze().tolist()
        
        root_name = data.entities_names[root]
        answers_names = [ data.entities_names[answer] for answer in answers ]

        subgraph = graph.subgraph(list(nx.traversal.dfs_preorder_nodes(graph.to_undirected(), root_name, depth_limit=2)))
        subgraph.nodes[root_name]['color'] = '#162347'

        for node_id in topk.indices.squeeze().tolist():
            node_name = data.entities_names[node_id]
            if node_name in subgraph:
                subgraph.nodes[node_name]['color'] = '#dd4b39'
        for answer_name in answers_names:
            subgraph.nodes[answer_name]['color'] = '#00ff1e'     
        for node_name in list(subgraph.nodes):
            node_id = data.entities_ids[node_name]
            subgraph.nodes[node_name]['value'] = scores[node_id]
        net = Network("600px", "1000px")
        net.from_nx(subgraph)
        html = net.generate_html().replace('\"', '\'')

    
        return f'''<iframe sandbox="allow-scripts" width="1100px" height="700px" srcdoc="{html}"></iframe>'''

        
        
    demo = gr.Blocks()

    with demo:
        gr.Markdown(
        """
        # Hello World!
        Start typing below to see the output.
        """)
        model_selector = gr.Radio(choices=[None, *available_models], value=None, placeholder="Select the model you want to use")
        # model_name_textbox = gr.Textbox(visible=False)
        question_selector = gr.Slider(interactive=True, visible=False)

        root_name_textbox = gr.Textbox(visible=False)
        answers_names_selector = gr.Radio([], visible=False, interactive=True)
        
        predict_button = gr.Button(value="Predict", variant="primary", visible=False)
        plot_button = gr.Button(value="Plot",  visible=False)
        
        predictions_box = gr.Label(num_top_classes=15,  visible=False)
        plot_box = gr.HTML()

        model_selector.change(fn=update_model, 
                inputs=model_selector, 
                outputs=[question_selector],
                )
        
        question_selector.change(fn=update_question, 
                inputs=question_selector, 
                outputs=[root_name_textbox, answers_names_selector, predict_button, plot_button]
                )
        
        predict_button.click(
            fn=predict, 
            inputs=question_selector,
            outputs=predictions_box
        )
        plot_button.click(
            fn=plot, 
            inputs=question_selector,
            outputs=plot_box
        )
        
        
        
        



    demo.launch()


if __name__ == '__main__':
    main(True, 'checkpoints/*/*/*.ckpt' )
    
    
# %%
