#%%
from class_resolver import UnexpectedKeywordError
import torch
from models.rgcnqa_explainer import RGCNQAExplainer
from train import RGCNQA, QAData
from transformers import AutoTokenizer
import glob
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from torch_geometric.utils import k_hop_subgraph
from random import randint
from utils import *
model: RGCNQA = None 
data: QAData = None
hops: int = None
device: str = None
graph = None
subgraph = None


#%%
# @click.command()
# @click.option('--cpu', default=False, show_default=True, is_flag=True)
# @click.option('--ckpt-folder', default='../checkpoints/*/*/*.ckpt', type=str)
def main(cpu, ckpt_path):
    global device
    
    device = 'cpu' if cpu else  'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", )    
    available_models = glob.glob(ckpt_path)    
    
    def update_model(model_path):
        global model 
        global hops
        global data
        global device
        global graph
        try:
            model = RGCNQA.load_from_checkpoint(model_path, map_location={'cuda:0': device, 'cpu': device },bias=False)
        except UnexpectedKeywordError:
            model = RGCNQA.load_from_checkpoint(model_path, map_location={'cuda:0': device, 'cpu': device }, bias=False)

        if data is None or hops !=  int (model_path.rsplit('/', 2)[1][0]):
            hops = int (model_path.rsplit('/', 2)[1][0])
            data = QAData(METAQA_PATH, [hops], tokenizer, use_ntm=False)
            

            
            raw_graph = [(data.entities_names[s], data.relations_names[r], data.entities_names[d]) for s,r,d in data.get_triples(add_inverse=False).tolist() ]
            graph = nx.DiGraph()
            for s,r,d in raw_graph:
                graph.add_edge(s,d, label=r)
            
        data_selector = gr.Slider.update(maximum=len(data.val_ds_unflattened), step=1,  label='Sample selector', visible=True )
        model_name_textbox = gr.Textbox.update(value=f"You selected model {model_path}", label='Selected Model', visible=True)
        
        return  data_selector

    def update_question(question_index):
        global model 
        global hops
        global data
        global device
        
        

        
        root, _, answers, question = data.val_ds_unflattened[int(question_index)]        
        root_name = data.entities_names[root]
        answers_names = [ data.entities_names[answer] for answer in answers ]
        
        index = model.edge_index.T[[0,2]].cpu()
        relations = model.edge_index.T[1].cpu()
        subset, index, inv, edge_mask = k_hop_subgraph(root, model.hops, index)
        question = data.tokenizer.decode(question).replace('root', root_name)

        question_selector = gr.Textbox.update(question, visible=True)
        answers_selector = gr.Radio.update(choices= answers_names, visible= True)
        predict_button = gr.Button.update(visible= True)
        xai_button = gr.Button.update(visible= False)
        question_info = gr.Textbox.update(f"Nodes: {len(subset)}, Edges: {len(index)}", visible=True)
        
        return question_selector, answers_selector, predict_button, xai_button, question_info
    
    def predict(index):
        global data
        global device
        global subgraph
        
        root, _, answers, question = data.val_ds_unflattened[int(index)]        
        preds = model.validation_step(
            {'qa':
                {'inference': 
                    (torch.tensor(root, dtype=int, device=device).unsqueeze(-1), 
                    torch.tensor(answers, dtype=int, device=device).unsqueeze(-1), 
                    question.to(device).unsqueeze(-1)
                    )
                    }
            }, -1)
        scores, topk = next(iter(preds))
        
        preds =  dict(
                    zip(
                        [ '✅ ' + data.entities_names[i] if i in answers  else '❌ ' + data.entities_names[i] for i in topk.indices.squeeze().tolist()  ], 
                        (topk.values.squeeze() / topk.values.max() ).tolist()
                        )
                    )
        predictions_box = gr.Label.update(value= preds, visible= True)    

        scores = scores.squeeze().tolist()
        
        root_name = data.entities_names[root]
        answers_names = [ data.entities_names[answer] for answer in answers ]

        subgraph = graph.subgraph(list(nx.traversal.dfs_preorder_nodes(graph.to_undirected(), root_name, depth_limit=model.hops)))
        

        for position, node_id in enumerate(topk.indices.squeeze().tolist()):
            node_name = data.entities_names[node_id]
            if node_name in subgraph:
                if node_name in answers_names:
                    subgraph.nodes[node_name]['color'] = '#00ff1e'  
                elif position < len(answers_names):
                    subgraph.nodes[node_name]['color'] = '#dd4b39'
                    
        
        subgraph.nodes[root_name]['color'] = '#162347'
        for e in subgraph.edges():
            subgraph.edges[e]['color'] = 'black'
   
        # for node_name in list(subgraph.nodes):
        #     node_id = data.entities_ids[node_name]
        #     score =  scores[node_id]
        #     subgraph.nodes[node_name]['value'] = score
            
        #     if score > max([scores[a] for a in answers]):    
        #         subgraph.nodes[node_name]['color'] = '#dd4b39'
                
        # for answer_name in answers_names:
        #     subgraph.nodes[answer_name]['color'] = '#00ff1e'  
            
        net = Network("600px", "1000px", directed =True)
        net.from_nx(subgraph)
        html = net.generate_html().replace('\"', '\'')

        
        return predictions_box, f'''<iframe sandbox="allow-scripts" width="1100px" height="700px" srcdoc="{html}"></iframe>''', gr.Button.update(visible=True)

    def explain(question_index, answer_name, xai_epochs, xai_mask_type, xai_lr, xai_edge_size, xai_node_feat_size, xai_edge_ent, xai_node_feat_ent):
        if not answer_name:
            return None
        
        # question_index = 0
        dst_idx = data.entities_ids[answer_name]
        x = model.nodes_emb
        src_idx, _, _, question = data.val_ds_unflattened[question_index]
        # dst_idx = dst_idx[0]
        raw_index = model.edge_index
        index = raw_index.T[[0,2]].cpu()
        relations = raw_index.T[1].cpu()

        subset, index, inv, edge_mask = k_hop_subgraph(src_idx, model.hops, index)
        relations = relations[edge_mask]

        explainer = RGCNQAExplainer(model,
            epochs=int(xai_epochs),
            lr=xai_lr,
            return_type='prob',
            feat_type=xai_mask_type,
                        num_hops=model.hops,
            **{
                'xai_edge_size':xai_edge_size,
                'xai_node_feat_size':xai_node_feat_size,
                'xai_edge_ent':xai_edge_ent,
                'xai_node_feat_ent':xai_node_feat_ent
                })

        node_feat_mask, edge_mask = explainer.explain_node(dst_idx, 
                                                        x,
                                                        index,
                                                        relations = relations,
                                                        src_idx = src_idx,
                                                        question = question,
                                                        relabel_nodes = False)

        # old_size = torch.concat([*index.T]).unique().size(0)
        # new_size = (node_feat_mask > 0.5).sum()
        # compression = new_size/old_size*100
        


        # g_nodes_index = range(x.size(1))#subset.tolist()# torch.concat([*index.T]).unique().detach().tolist()
        g_edge_index = index.T.detach().tolist()

        nodes_mask = node_feat_mask.detach().tolist()
        edges_mask = edge_mask.detach().tolist()
            
        golden_nodes, golden_edges = get_golden_path(data, question_index,dst_idx) 


        G = nx.DiGraph()
        print(index.unique().detach().tolist())
        for n, w in zip( index.unique().detach().tolist(), nodes_mask ):
            print(n, '->', w )
            if n in index.unique().detach().tolist():
                attrs = {
                    'label':  data.entities_names[n],
                    'size': 10, # + int(w/max(nodes_mask)*100),
                }
                print(n, dst_idx)
                if n in golden_nodes:
                    attrs['color'] = '#f5d142'
                if n == dst_idx:
                    attrs['color'] = 'green'
                if n == src_idx:
                    attrs['color'] = 'black'

                G.add_node(n, **attrs)
            

        rel_colors = {
            i: f"{randint(0, 128)},{randint(0, 128)},{randint(0, 128)}" for i in range(max(relations.tolist())+1)
        }

        for w, e, r in zip( edges_mask, g_edge_index, relations.tolist()) :
            # print(e, '->', w )
            attrs = {
                # 'color': f'rgba({rel_colors[r]}, {w})'
                'color': f'rgba(0,0,0, {w})'
                # 'color': f'rgba(0,0,0,1)'
            }
            if tuple(e) in golden_edges:
                attrs['color'] = f'rgba(145, 122, 23, {w})'
                # attrs['color'] = f'#f5d142'
            G.add_edge(*e, **attrs)

            
        from pyvis.network import Network
        net = Network(directed=True)
        net.from_nx(G)

        html = net.generate_html().replace('\"', '\'')

    
        return f'''<iframe sandbox="allow-scripts" width="1100px" height="700px" srcdoc="{html}"></iframe>'''

    demo = gr.Blocks()

    with demo:
        gr.Markdown(
        """
        #  QA-RGCN
        Please select a model
        """)
        model_selector = gr.Radio(choices=[None, *available_models], value=None, placeholder="Select the model you want to use")

        question_selector = gr.Slider(interactive=True, visible=False)

        root_name_textbox = gr.Textbox(visible=False)
        question_info_textbox = gr.Textbox(visible=False)
        
        
        predict_button = gr.Button(value="Predict", variant="primary", visible=False)
        predictions_box = gr.Label(num_top_classes=10,  visible=False)
        
        xai_target_answer = gr.Radio([], visible=False, interactive=True, label='Xai Answer')
        xai_epochs = gr.Slider(minimum=0, maximum=10000, value=300, label='Xai epochs')
        xai_mask_type = gr.Radio(["scalar", "feature", "individual_feature" ], label='Xai Mask Type')
        xai_button = gr.Button(value="Explain",  visible=False)
        
        xai_lr = gr.Number(value= 0.01, log=True, label='xai_edge_lr')
        xai_edge_size = gr.Slider(minimum=0,  maximum=1, value= 0.005, label='xai_edge_size')
        xai_node_feat_size = gr.Slider(minimum=0, maximum=1,  value= 1.0, label='xai_node_feat_size')

        xai_edge_ent = gr.Slider(minimum=0, maximum=1,  value= 1.0, label='xai_edge_ent')
        xai_node_feat_ent = gr.Slider(minimum=0, maximum=1,  value= 0.1, label='xai_node_feat_ent')
        
        plot_box = gr.HTML()

        model_selector.change(
            fn=update_model, 
            inputs=model_selector, 
            outputs=[question_selector],
            )
        
        question_selector.change(
            fn=update_question, 
            inputs=question_selector, 
            outputs=[root_name_textbox, xai_target_answer, predict_button, xai_button, question_info_textbox]
            )
        
        predict_button.click(
            fn= predict, 
            inputs= [ question_selector],
            outputs=[ predictions_box, plot_box, xai_button]
            )
        
        xai_button.click(
            fn=explain, 
            inputs=[question_selector, xai_target_answer, xai_epochs, xai_mask_type, xai_lr, xai_edge_size, xai_node_feat_size, xai_edge_ent, xai_node_feat_ent],
            outputs=plot_box
            )
        
    demo.launch()


if __name__ == '__main__':
    main(True, '../data/checkpoints/*/*/*.ckpt' )
    
    
# %%
