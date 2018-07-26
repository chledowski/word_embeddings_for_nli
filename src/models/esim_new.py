#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

import logging
import os

import keras.backend as K
from keras import optimizers
from keras.activations import softmax
from keras.layers import Subtract, Dense, Input, Dropout, TimeDistributed, Lambda, Bidirectional, \
    Dot, Permute, Multiply, Concatenate, Activation
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from src import DATA_DIR
from src.models.utils import *
from src.util.prepare_embedding import prep_embedding_matrix

logger = logging.getLogger(__name__)


def esim(config, data):

    vocabulary = {}
    with open(os.path.join(DATA_DIR, config["dataset"], 'vocab.txt')) as f:

        for line in f:
            (key, val) = line.split()
            vocabulary[key] = int(val)

    vocab_size = len(vocabulary)
    logger.info('Vocab size = {}'.format(vocab_size))

    logger.info('Using {} embedding'.format(config["embedding_name"]))
    embed, embedding_matrix, statistics = prep_embedding_matrix(config, vocab_size, data)

    a = Input(shape=(config['max_length'],), dtype='int32', name='premise')
    b = Input(shape=(config['max_length'],), dtype='int32', name='hypothesis')

    # ---------- Embedding layer ---------- #
    embedding = EmbeddingLayer(vocab_size, config["embedding_dim"],
                               embedding_matrix,
                               max_length=config['max_length']) #

    embedded_a = embedding(a)
    embedded_b = embedding(b)

    # ---------- Encoding layer ---------- #
    encoded_a = EncodingLayer(config["embedding_dim"],
                              config['max_length'],
                              dropout=config["dropout"])(embedded_a)
    encoded_b = EncodingLayer(config["embedding_dim"],
                              config['max_length'],
                              dropout=config["dropout"])(embedded_b)

    # ---------- Local inference layer ---------- #
    m_a, m_b = LocalInferenceLayer()([encoded_a, encoded_b])

    # ---------- Inference composition layer ---------- #
    composed_a = InferenceCompositionLayer(config["embedding_dim"],
                                           config['max_length'],
                                           dropout=config["dropout"])(m_a)
    composed_b = InferenceCompositionLayer(config["embedding_dim"],
                                           config['max_length'],
                                           dropout=config["dropout"])(m_b)

    # ---------- Pooling layer ---------- #
    pooled = PoolingLayer()([composed_a, composed_b])

    # ---------- Classification layer ---------- #
    prediction = MLPLayer(config["embedding_dim"], config["n_labels"],
                          dropout=config["dropout"], activations=(config["activation"], 'softmax'))(pooled)

    model = Model(inputs=[a, b], outputs=prediction)

    if config["optimizer"] == 'rmsprop':
        model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    elif config["optimizer"] == 'sgd':
        model.compile(optimizer=optimizers.SGD(lr=config["learning_rate"], momentum=0.9),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    elif config["optimizer"] == 'adam':
        model.compile(optimizer=optimizers.Adam(lr=config["learning_rate"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    print((model.summary()))

    return model
