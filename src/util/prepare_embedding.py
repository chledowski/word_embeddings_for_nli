#!/usr/bin/env python3
"""
Simple model definitions
"""

import h5py
import logging
import numpy as np
import os

from src import DATA_DIR
from src.util import remove_mean_and_d_components, normalize_embeddings
from tqdm import trange

logger = logging.getLogger(__name__)

def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    """
    Random weights drawn from a Gaussian
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')


def prep_embedding_matrix(config, data, embedding_path=None):
    if config["norm_weight"]:
        target_matrix = norm_weight(data.vocab.size(), config["embedding_dim"])
    else:
        target_matrix = np.random.uniform(
                -0.1, 0.1, (data.vocab.size(), config["embedding_dim"]))

    if embedding_path:
        with h5py.File(embedding_path, 'r') as f:
            source_words = [w.encode() for w in f['words_flatten'][0].split('\n')]
            source_word_to_id = dict(zip(source_words, list(range(len(source_words)))))
            source_matrix = f['embedding']

            source_vocab_ids = []
            for i in range(data.vocab.size()):
                word = data.vocab.id_to_word(i)
                if word in source_word_to_id:
                    source_vocab_ids.append((source_word_to_id[word], i))

            source_vocab_ids.sort()
            source_ids, vocab_ids = map(list, zip(*source_vocab_ids))

            target_matrix[vocab_ids] = source_matrix[source_ids]

            logger.info("Found {} words from dictionary in embedding file. "
                        "Missing {} words.".format(
                    len(vocab_ids), data.vocab.size() - len(vocab_ids)))

            if config["D"] != 0:
                target_matrix = remove_mean_and_d_components(
                    target_matrix, config["D"], partial_whitening=config["whitening"])

    target_matrix[0, :] = 0
    if config["normalize"]:
        target_matrix = normalize_embeddings(target_matrix)
    return target_matrix
