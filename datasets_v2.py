# %%
import torchtext.data

pos = torchtext.data.TabularDataset(
path='data/1-hop/qa_train.tsv', format='tsv',
fields=[('text', torchtext.data.Field()),
        ('labels', torchtext.data.Field())])
# %%

# sentiment = torchtext.data.TabularDataset(
#     path='data/sentiment/train.json', format='json',
#     fields={'sentence_tokenized': ('text', torchtext.data.Field(sequential=True)),
#              'sentiment_gold': ('labels', torchtext.data.Field(sequential=False))})
from mosestokenizer import MosesTokenizer
tokenizer = MosesTokenizer()
src = torchtext.data.Field(tokenize=tokenizer)
trg = torchtext.data.Field(tokenize=tokenizer)
# mt_train = torchtext.datasets.TranslationDataset(
#     path='data/mt/wmt16-ende.train', exts=('.en', '.de'),
#     fields=(src, trg))
# %%
