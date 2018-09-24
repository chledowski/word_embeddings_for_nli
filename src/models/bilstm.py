#!/usr/bin/env pythonpl
# -*- coding: utf-8 -*-q
"""
Simple model definitions
"""

import logging
import os

import keras.backend as K

from keras import optimizers
from keras.layers import Concatenate, Dense, Input, Dropout, TimeDistributed, Bidirectional, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.regularizers import l2

from src import DATA_DIR
from src.util.data import NLIData
from src.util.prepare_embedding import prep_embedding_matrix
from src.models.elmo import ElmoEmbeddings

logger = logging.getLogger(__name__)


def bilstm(config, data):
    logger.info('Vocab size = {}'.format(data.vocab.size()))
    logger.info('Using {} embedding'.format(config["embedding_name"]))

    embedding_matrix = prep_embedding_matrix(config, data, config["embedding_name"])

    embed = Embedding(data.vocab.size(), config["embedding_dim"],
                      weights=[embedding_matrix],
                      input_length=config["sentence_max_length"],
                      trainable=config["train_embeddings"],
                      mask_zero=False)

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(config["sentence_max_length"],), dtype='int32', name='premise')
    premise_mask_input = Input(shape=(config["sentence_max_length"],), dtype='int32', name='premise_mask_input')
    hypothesis = Input(shape=(config["sentence_max_length"],), dtype='int32', name='hypothesis')
    hypothesis_mask_input = Input(shape=(config["sentence_max_length"],), dtype='int32', name='hypothesis_mask_input')

    if config['use_elmo']:
        elmo_embed = ElmoEmbeddings(config)
        premise_elmo_input = Input(shape=(config["sentence_max_length"],), dtype='int32', name='premise_elmo_input')
        hypothesis_elmo_input = Input(shape=(config["sentence_max_length"],), dtype='int32',
                                      name='hypothesis_elmo_input')

    premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)
    hypothesis_mask = Lambda(lambda x: K.cast(x, 'float32'))(hypothesis_mask_input)

    # [-1, Psize, 1]
    premise_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask)
    # [-1, Hsize, 1]
    hypothesis_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(hypothesis_mask)

    embed_p = embed(premise)  # [batchsize, Psize, Embedsize]
    embed_h = embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    if config['use_elmo']:
        elmo_p = elmo_embed([premise_elmo_input, premise_mask_exp], stage="pre_lstm", name='p')
        elmo_h = elmo_embed([hypothesis_elmo_input, hypothesis_mask_exp], stage="pre_lstm", name='h')

        embed_p = Concatenate(axis=2)([embed_p, elmo_p])
        embed_h = Concatenate(axis=2)([embed_h, elmo_h])

    bilstm_layer = Bidirectional(
        LSTM(units=config["embedding_dim"]), merge_mode='concat')

    prem = bilstm_layer(embed_p)
    hypo = bilstm_layer(embed_h)

    if config["batch_normalization"] == "True":
        prem = BatchNormalization()(prem)
        hypo = BatchNormalization()(hypo)

    joint = Concatenate()([prem, hypo])
    joint = Dropout(config['dropout'], name="pre_mlp_drop")(joint)

    for i in range(config['n_layers']):
        joint = Dense(4 * config["embedding_dim"], activation='relu',
                      kernel_regularizer=l2(4e-6),
                      name="dense_" + str(i))(joint)
        joint = Dropout(config['dropout'], name="dropout_" + str(i))(joint)

        if config["batch_normalization"] == "True":
            joint = BatchNormalization(name="bn_" + str(i))(joint)

    predictions = Dense(config["n_labels"], activation='softmax', name="last_softmax")(joint)

    model_input = [premise, premise_mask_input, hypothesis, hypothesis_mask_input]
    if config['use_elmo']:
        model_input += [premise_elmo_input, hypothesis_elmo_input]

    model = Model(inputs=model_input,
                  outputs=predictions)

    if config["optimizer"] == 'rmsprop':
        model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        raise ValueError("Unknown optimizer: %s" % config["optimizer"])

    print(model.summary())

    return model