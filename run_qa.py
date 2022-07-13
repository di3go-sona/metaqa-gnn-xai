#%%
import torch, transformers
from train_qa import QAModel, QAData
from transformers import AutoTokenizer, BertModel
import click
import gradio as gr
#%%




@click.command()
@click.argument('model-path', type=str)
@click.option('--no-gpu', default=False, show_default=True, is_flag=True)
def main(model_path, no_gpu):
    hops = int (model_path.rsplit('/', 2)[1][0])
    device = 'cpu' if no_gpu else 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", map_location={'cuda:0': device })
    
    data = QAData('dataset', [1], tokenizer, use_ntm=False)
    model = QAModel.load_from_checkpoint(model_path) 
    
    def run(question_id):
        data
        return f"Selected question {question_id}"

    slider = gr.Slider
    demo = gr.Interface(fn=run, inputs="text", outputs="text")

    demo.launch()

#%%
if __name__ == '__main__':
    main()
    
    