#!/usr/bin/env pythonpl
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

import logging
import os

from keras.layers import merge, Dense, Input, Dropout, TimeDistributed, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2

from src import DATA_DIR
from src.util.data import SNLIData
from src.util.prepare_embedding import prep_embedding_matrix

logger = logging.getLogger(__name__)


def bilstm(config):

    logger.info("Loading data and vocabulary...")

    if config["dataset"]["name"] == "snli":
        data = SNLIData(os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"]["name"] == "mnli":
        data = SNLIData(os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"]["name"])

    vocabulary = {}
    with open(os.path.join(DATA_DIR, config["dataset"]["name"], 'vocab.txt')) as f:

        for line in f:
            (key, val) = line.split()
            vocabulary[key] = int(val)

    vocab_size = len(vocabulary)
    logger.info('Vocab size = {}'.format(vocab_size))

    logger.info('Using {} embedding'.format(config["embedding_name"]))
    embed, embedding_matrix, statistics = prep_embedding_matrix(config, vocab_size, data)

    premise = Input(shape=(None,), dtype='int32')
    hypothesis = Input(shape=(None,), dtype='int32')

    bilstm_layer = Bidirectional(LSTM(units=config["embedding_dim"]), merge_mode='concat', weights=None)
    # translate = TimeDistributed(Dense(config["embedding_dim"], activation='relu', name="translate"),
    #                             name="translate")

    prem = bilstm_layer(embed(premise))
    hypo = bilstm_layer(embed(hypothesis))

    if config["batch_normalization"] == "True":
        prem = BatchNormalization()(prem)
        hypo = BatchNormalization()(hypo)

    joint = merge([prem, hypo], mode='concat', name="pre_mlp_concat")
    joint = Dropout(config['dropout'], name="pre_mlp_drop")(joint)

    for i in range(config['n_layers']):
        joint = Dense(4 * config["embedding_dim"], activation='relu',
                      kernel_regularizer=l2(4e-6),
                      name="dense_" + str(i))(joint)
        joint = Dropout(config['dropout'], name="dropout_" + str(i))(joint)

        if config["batch_normalization"] == "True":
            joint = BatchNormalization(name="bn_" + str(i))(joint)

    pred = Dense(config["dataset"]["n_labels"], activation='softmax', name="last_softmax")(joint)

    model = Model(inputs=[premise, hypothesis], outputs=pred)
    print((model.summary()))
    return model, embedding_matrix, statistics
