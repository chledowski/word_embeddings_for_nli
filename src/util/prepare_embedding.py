#!/usr/bin/env python3
"""
Simple model definitions
"""

import h5py
import numpy as np
import os

from src import DATA_DIR
from src.util import remove_mean_and_d_components, normalize_embeddings

from keras.layers.embeddings import Embedding


def prep_embedding_matrix(config, vocab_size, data):

    # if not config["intersection_of_embedding_dicts"]:
    if config["embedding_name"] == "random_uniform":
        embedding_matrix = np.random.uniform(-0.1, 0.1, (vocab_size, config["embedding_dim"]))
        statistics = [0, vocab_size]
    else:
        embedding_matrix = []
        embedding_file = h5py.File(os.path.join(DATA_DIR, 'embeddings', config["embedding_name"] + ".h5"), 'r')
        embedding_words = embedding_file['words_flatten'][0].split('\n')
        embedding_words = [word.encode() for word in embedding_words]
        embedding_word_to_id = dict(list(zip(embedding_words, list(range(len(embedding_words))))))  # word -> id
        embedding_matrix_all = embedding_file[list(embedding_file.keys())[0]][:]
        good = 0
        bad = 0
        for i in range(vocab_size):
            word_lower = data.vocab.id_to_word(i)

            if word_lower in embedding_word_to_id:
                # print("good: " + word_lower)
                good+=1
                embedding_matrix.append(embedding_matrix_all[embedding_word_to_id[word_lower]])
            else:
                # if word_lower.lower() in embedding_word_to_id:
                    # print ("bad: " + word_lower)
                bad +=1
                embedding_matrix.append(np.random.uniform(-0.1, 0.1, size=(embedding_matrix_all.shape[1],)))
        print("Found {} words in the dictionary. Missing {} words.".format(good, bad))
        statistics = [good, bad]
        embedding_matrix = np.array(embedding_matrix)
        if config["D"] != 0:
            embedding_matrix = remove_mean_and_d_components(embedding_matrix, config["D"], partial_whitening=config["whitening"])
    embedding_matrix[0, :] = 0
    if config["normalize"]:
        embedding_matrix = normalize_embeddings(embedding_matrix)
    embed = Embedding(vocab_size, config["embedding_dim"],
                      weights=[embedding_matrix], mask_zero=True, trainable=config["train_embeddings"])

    return embed, embedding_matrix, statistics
