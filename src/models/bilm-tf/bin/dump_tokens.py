import argparse
import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

def main(args):
    parser = argparse.ArgumentParser()

    datadir = args.elmodir
    vocab_file = os.path.join(datadir, 'vocab_elmo.txt')
    options_file = os.path.join(datadir, 'options.json')
    weight_file = os.path.join(datadir, 'lm_weights.hdf5')
    token_embedding_file = os.path.join(datadir, 'elmo_token_embeddings.hdf5')

    dump_token_embeddings(
        vocab_file, options_file, weight_file, token_embedding_file
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--elmodir')
    args = parser.parse_args()
    main(args)
