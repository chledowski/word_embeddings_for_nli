#!/usr/bin/env python3
"""
Simple model definitions
"""

import h5py
import numpy as np
import os

from src import DATA_DIR
from src.util import remove_mean_and_d_components, normalize_embeddings


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


def prep_embedding_matrix(config, data):
    if config["embedding_name"] == "random_uniform":
        if config["norm_weight"]:
            embedding_matrix = norm_weight(data.vocab.size(), config["embedding_dim"])
        else:
            embedding_matrix = np.random.uniform(-0.1, 0.1, (data.vocab.size(), config["embedding_dim"]))
    else:
        embedding_file = h5py.File(os.path.join(DATA_DIR, 'embeddings', config["embedding_name"] + ".h5"), 'r')
        embedding_words = embedding_file['words_flatten'][0].split('\n')
        embedding_words = [word.encode() for word in embedding_words]
        embedding_word_to_id = dict(list(zip(embedding_words, list(range(len(embedding_words))))))  # word -> id
        embedding_matrix_all = embedding_file[list(embedding_file.keys())[0]][:]
        good = 0
        bad = 0

        if config["norm_weight"]:
            embedding_matrix = norm_weight(data.vocab.size(), config["embedding_dim"])
            for i in range(data.vocab.size()):
                word_lower = data.vocab.id_to_word(i)
                if word_lower in embedding_word_to_id:
                    good+=1
                    embedding_matrix[i] = embedding_matrix_all[embedding_word_to_id[word_lower]]
                else:
                    bad +=1
        else:
            embedding_matrix = []
            for i in range(data.vocab.size()):
                word_lower = data.vocab.id_to_word(i)
                if word_lower in embedding_word_to_id:
                    good += 1
                    embedding_matrix.append(embedding_matrix_all[embedding_word_to_id[word_lower]])
                else:
                    bad += 1
                    embedding_matrix.append(np.random.uniform(-0.1, 0.1, (embedding_matrix_all.shape[1],)))
            embedding_matrix = np.array(embedding_matrix)

        print("Found {} words in the dictionary. Missing {} words.".format(good, bad))

        if config["D"] != 0:
            embedding_matrix = remove_mean_and_d_components(embedding_matrix, config["D"], partial_whitening=config["whitening"])
    embedding_matrix[0, :] = 0
    if config["normalize"]:
        embedding_matrix = normalize_embeddings(embedding_matrix)

    return embedding_matrix
