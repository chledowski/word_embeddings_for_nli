#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

import logging
import os

import keras.backend as K
from keras.layers import concatenate, Dense, Input, Dropout, TimeDistributed, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Model
from keras.regularizers import l2

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from src import DATA_DIR
from src.util.data import SNLIData
from src.util.prepare_embedding import prep_embedding_matrix

logger = logging.getLogger(__name__)


def cbow(config, data):
    vocabulary = {}
    with open(os.path.join(DATA_DIR, config["dataset"], 'vocab.txt')) as f:

        for line in f:
            (key, val) = line.split()
            vocabulary[key] = int(val)

    vocab_size = len(vocabulary)
    logger.info('Vocab size = {}'.format(vocab_size))

    logger.info('Using {} embedding'.format(config["embedding_name"]))
    embedding_matrix = prep_embedding_matrix(config, vocab_size, data)

    embed = Embedding(vocab_size, config["embedding_dim"],
                      weights=[embedding_matrix],
                      input_length=config["sentence_max_length"],
                      trainable=config["train_embeddings"])

    AvgEmbeddings = Lambda(lambda x: K.mean(x, axis=1),
                           output_shape=(config["embedding_dim"],),
                           name="sum_emb")
    translate = TimeDistributed(Dense(config["embedding_dim"], activation='relu', name="translate"),
                                name="translate")

    configg = tf.ConfigProto()
    configg.gpu_options.allow_growth = True
    set_session(tf.Session(config=configg))

    premise = Input(shape=(config["sentence_max_length"],), dtype='int32')
    hypothesis = Input(shape=(config["sentence_max_length"],), dtype='int32')

    prem = AvgEmbeddings(translate(embed(premise)))
    hypo = AvgEmbeddings(translate(embed(hypothesis)))

    if config["batch_normalization"] == "True":
        prem = BatchNormalization()(prem)
        hypo = BatchNormalization()(hypo)

    joint = concatenate([prem, hypo])
    joint = Dropout(config['dropout'], name="pre_mlp_drop")(joint)

    for i in range(config['n_layers']):
        joint = Dense(2 * config["embedding_dim"], activation='relu',
                      kernel_regularizer=l2(4e-6),
                      name="dense_" + str(i))(joint)
        joint = Dropout(config['dropout'], name="dropout_" + str(i))(joint)

        if config["batch_normalization"] == "True":
            joint = BatchNormalization(name="bn_" + str(i))(joint)

    pred = Dense(config["n_labels"], activation='softmax', name="last_softmax")(joint)

    model = Model(inputs=[premise, hypothesis], outputs=pred)

    if config["optimizer"] == 'rmsprop':
        model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    print((model.summary()))
    return model
