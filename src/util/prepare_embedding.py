#!/usr/bin/env python3
"""
Simple model definitions
"""

import gc
import h5py
import numpy as np
import os

from src import DATA_DIR
from src.util import remove_mean_and_d_components, normalize_embeddings
from tqdm import trange


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


def prep_embedding_matrix(config, data, embedding_name):
    if config["embedding_name"] == "random_uniform":
        if config["norm_weight"]:
            embedding_matrix = norm_weight(data.vocab.size(), config["embedding_dim"])
        else:
            embedding_matrix = np.random.uniform(-0.1, 0.1, (data.vocab.size(), config["embedding_dim"]))
    else:
        embedding_file = h5py.File(os.path.join(DATA_DIR, 'embeddings', embedding_name + ".h5"), 'r')
        embedding_words = embedding_file['words_flatten'][0].split('\n')
        embedding_words = [word.encode() for word in embedding_words]
        embedding_word_to_id = dict(list(zip(embedding_words, list(range(len(embedding_words))))))  # word -> id
        embedding_matrix_all = embedding_file[list(embedding_file.keys())[0]][:]
        num_found = 0
        num_notfound = 0

        if config["norm_weight"]:
            embedding_matrix = norm_weight(data.vocab.size(), config["embedding_dim"])
            found_word_indices = []
            found_word_ids = []
            notfound_words = []
            for i in trange(data.vocab.size()):
                word_lower = data.vocab.id_to_word(i)
                if word_lower in embedding_word_to_id:
                    num_found += 1
                    found_word_indices.append(i)
                    found_word_ids.append(embedding_word_to_id[word_lower])
                    embedding_matrix[i] = embedding_matrix_all[embedding_word_to_id[word_lower]]
                else:
                    num_notfound += 1
                    notfound_words.append(word_lower)

            dumps_dir = os.path.join(DATA_DIR, 'dumps')
            os.makedirs(dumps_dir, exist_ok=True)
            np.save(os.path.join(dumps_dir, 'missing_words.npy'),
                    np.array(notfound_words))

        else:
            embedding_matrix = []
            for i in trange(data.vocab.size()):
                word_lower = data.vocab.id_to_word(i)
                if word_lower in embedding_word_to_id:
                    num_found += 1
                    embedding_matrix.append(embedding_matrix_all[embedding_word_to_id[word_lower]])
                else:
                    num_notfound += 1
                    embedding_matrix.append(np.random.uniform(-0.1, 0.1, (embedding_matrix_all.shape[1],)))
            embedding_matrix = np.array(embedding_matrix)

        print("Found {} words in the dictionary. Missing {} words.".format(num_found, num_notfound))

        if config["D"] != 0:
            embedding_matrix = remove_mean_and_d_components(embedding_matrix, config["D"],
                                                            partial_whitening=config["whitening"])

        del embedding_words, embedding_matrix_all
        embedding_file.close()


    embedding_matrix[0, :] = 0
    if config["normalize"]:
        embedding_matrix = normalize_embeddings(embedding_matrix)

    gc.collect()

    # print("Emb matrix norm", np.linalg.norm(embedding_matrix))
    return embedding_matrix
