'''
ELMo usage example with pre-computed and cached context independent
token representations
Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import os

from src.models.bilm import dump_token_embeddings
from src import DATA_DIR

elmodir = os.path.join(DATA_DIR, 'elmo')

all_tokens = []

with open(os.path.join(DATA_DIR, 'snli', 'vocab_all.txt')) as f:
    for line in f:
        token, counter = line.split()
        all_tokens.append(token)

all_tokens = ['<S>', '</S>', '<UNK>'] + all_tokens[6:]

vocab_file = os.path.join(elmodir, 'vocab_elmo.txt')
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

options_file = os.path.join(elmodir, 'options.json')
weight_file = os.path.join(elmodir, 'lm_weights.hdf5')

# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = os.path.join(elmodir, 'elmo_token_embeddings.hdf5')
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)