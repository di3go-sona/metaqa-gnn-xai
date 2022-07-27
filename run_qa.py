#%%
import torch, transformers
from train_joint import JointQAModel
from train_qa import QAModel, QAData
from transformers import AutoTokenizer, BertModel
import click, os, glob
import gradio as gr
import networkx as nx
import random
import matplotlib.pyplot as plt


model = None 
hops = None
data = None
device = None
graph = None
#%%
# @click.command()
# @click.option('--cpu', default=False, show_default=True, is_flag=True)
# @click.option('--ckpt-folder', default='../checkpoints/*/*/*.ckpt', type=str)

def main(cpu, ckpt_folder):
    global device
    device = 'cpu' if cpu else  'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", )
    
    available_models = glob.glob(ckpt_folder)    
    
    def update_model(model_path):
        global model 
        global hops
        global data
        global device
        global graph
        
        model = JointQAModel.load_from_checkpoint(model_path, map_location={'cuda:0': device }).to(device)
        model.on_validation_batch_start()
        
        if data is None or hops !=  int (model_path.rsplit('/', 2)[1][0]):
            hops = int (model_path.rsplit('/', 2)[1][0])
            data = QAData('dataset', [hops], tokenizer, use_ntm=False)
            
            raw_graph = [(data.entities_names[s], data.relations_names[r], data.entities_names[d]) for s,r,d in data.triplets.tolist() ]
            graph = nx.DiGraph()
            for s,r,d in raw_graph:
                graph.add_edge(s,r, label=d)
            
        data_selector = gr.Slider.update(maximum=len(data.val_ds_unflattened), step=1,  label='Sample selector', visible=True )
        model_name_textbox = gr.Textbox.update(value=f"You selected model {model_path}", label='Selected Model', visible=True)
        
        return model_name_textbox, data_selector

    def update_question(index):
        global model 
        global hops
        global data
        global device
        
        root, _, answers, question = data.val_ds_unflattened[int(index)]        
        root_name = data.entities_names[root]
        answers_names = [ data.entities_names[answer] for answer in answers ]
        question = data.tokenizer.decode(question).replace('root', root_name)

        question_selector = gr.Textbox.update(question, visible=True)
        answers_selector = gr.Radio.update(choices=answers_names, visible=True)
        
        return question_selector, answers_selector
    
    def predict(index):
        global data
        global device

        root, _, answers, question = data.val_ds_unflattened[int(index)]        
        print(root, answers, question)
        preds = model.qa_validation_step(
            {'inference': 
                (torch.tensor(root, dtype=int).to(device).unsqueeze(-1), 
                torch.tensor(answers, dtype=int).to(device).unsqueeze(-1), 
                question.to(device).unsqueeze(-1)
                )
                }, -1)
        topk = next(iter(preds))
        return dict(zip([ data.entities_names[i] for i in topk.indices.squeeze().tolist()], topk.values.squeeze().tolist()))
    
    def plot(index):
        root, _, answers, question = data.val_ds_unflattened[int(index)]
 
        root_name = data.entities_names[root]
        answers_names = [ data.entities_names[answer] for answer in answers ]
        raw_graph = [(data.entities_names[s], data.relations_names[r], data.entities_names[d]) for s,r,d in data.triplets.tolist() ]
        graph = nx.DiGraph()
        for s,r,d in raw_graph:
            graph.add_edge(s,d, label=r)
            if s == '1990' or s =='1951':
                print('found')
            
        paths = list(nx.simple_paths.all_simple_paths(graph.to_undirected(), root_name, answers_names, 2 ))
        # nx.draw_linear(random_graph)
        # list(graph.nodes())[:100]
        fig = plt.figure()
        nx.draw_spring(graph.subgraph(list(set([n for triplet in paths  for n in triplet ]))), with_labels=True)

        return fig
        
        
    demo = gr.Blocks()

    with demo:
        gr.Markdown(
        """
        # Hello World!
        Start typing below to see the output.
        """)
        model_selector = gr.Radio(choices=[None, *available_models], value=None, placeholder="Select the model you want to use")
        model_name_textbox = gr.Textbox(visible=False)
        question_selector = gr.Slider(interactive=True, visible=False)

        root_name_textbox = gr.Textbox(visible=False)
        answers_names_selector = gr.Radio([], visible=False, interactive=True)
        
        predict_button = gr.Button(value="Predict", variant="primary")
        plot_button = gr.Button(value="Plot")
        
        predictions_box = gr.outputs.Label(num_top_classes=15)
        plot_box = gr.Plot()

        model_selector.change(fn=update_model, 
                inputs=model_selector, 
                outputs=[model_name_textbox, question_selector],
                )
        
        question_selector.change(fn=update_question, 
                inputs=question_selector, 
                outputs=[root_name_textbox, answers_names_selector]
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
    main(False, 'checkpoints/*/*/*.ckpt' )
    
    
# %%
