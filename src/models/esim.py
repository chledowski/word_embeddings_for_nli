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
    Dot, Permute, Multiply, Concatenate, Activation, CuDNNLSTM
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from src import DATA_DIR
from src.util.data import SNLIData
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
    embedding_matrix = prep_embedding_matrix(config, vocab_size, data)

    embed = Embedding(vocab_size, config["embedding_dim"],
                      weights=[embedding_matrix],
                      input_length=config["sentence_max_length"],
                      trainable=config["train_embeddings"])

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(config["sentence_max_length"],), dtype='int32')
    hypothesis = Input(shape=(config["sentence_max_length"],), dtype='int32')

    embed_p = embed(premise)  # [batchsize, Psize, Embedsize]
    embed_h = embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    # 2, Encoder words with its surrounding context
    bilstm_encoder = Bidirectional(CuDNNLSTM(units=config["embedding_dim"], return_sequences=True))

    embed_p = Dropout(config["dropout"])(bilstm_encoder(embed_p))
    embed_h = Dropout(config["dropout"])(bilstm_encoder(embed_h))

    # 3, Score each words and calc score matrix Eph.
    F_p, F_h = embed_p, embed_h
    Eph = Dot(axes=(2, 2))([F_h, F_p])  # [batch_size, Hsize, Psize]
    Eh = Lambda(lambda x: softmax(x))(Eph)  # [batch_size, Hsize, Psize]
    Ep = Permute((2, 1))(Eph)  # [batch_size, Psize, Hsize)
    Ep = Lambda(lambda x: softmax(x))(Ep)  # [batch_size, Psize, Hsize]

    # 4, Normalize score matrix, encoder premesis and get alignment
    PremAlign = Dot((2, 1))([Ep, embed_h])  # [-1, Psize, dim]
    HypoAlign = Dot((2, 1))([Eh, embed_p])  # [-1, Hsize, dim]
    mm_1 = Multiply()([embed_p, PremAlign])
    mm_2 = Multiply()([embed_h, HypoAlign])
    sb_1 = Subtract()([embed_p, PremAlign])
    sb_2 = Subtract()([embed_h, HypoAlign])

    PremAlign = Concatenate()([embed_p, PremAlign, sb_1, mm_1])  # [batch_size, Psize, 2*unit]
    HypoAlign = Concatenate()([embed_h, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]
    PremAlign = Dropout(config["dropout"])(PremAlign)
    HypoAlign = Dropout(config["dropout"])(HypoAlign)

    translate = TimeDistributed(Dense(config["embedding_dim"],
                                      kernel_regularizer=l2(1e-5),
                                      bias_regularizer=l2(1e-5),
                                      activation='relu'),
                                name='translate')

    PremAlign = translate(PremAlign)
    HypoAlign = translate(HypoAlign)

    # 5, Final biLSTM < Encoder + Softmax Classifier
    bilstm_decoder = Bidirectional(CuDNNLSTM(units=300, return_sequences=True),
                                   name='finaldecoder')  # [-1,2*units]
    final_p = Dropout(config["dropout"])(bilstm_decoder(PremAlign))
    final_h = Dropout(config["dropout"])(bilstm_decoder(HypoAlign))

    AveragePooling = Lambda(lambda x: K.mean(x, axis=1))  # outs [-1, dim]
    MaxPooling = Lambda(lambda x: K.max(x, axis=1))  # outs [-1, dim]
    avg_p = AveragePooling(final_p)
    avg_h = AveragePooling(final_h)
    max_p = MaxPooling(final_p)
    max_h = MaxPooling(final_h)
    Final = Concatenate()([avg_p, max_p, avg_h, max_h])
    Final = Dropout(config["dropout"])(Final)
    Final = Dense(300,
                  kernel_regularizer=l2(1e-5),
                  bias_regularizer=l2(1e-5),
                  name='dense300_' + config["dataset"])(Final)
    if config["batch_normalization"]:
        Final = BatchNormalization()(Final)
    Final = Activation(config["activation"])(Final)
    Final = Dropout(config["dropout"] / 2)(Final)
    Final = Dense(3,
                  activation='softmax',
                  name='judge300_' + config["dataset"])(Final)
    model = Model(inputs=[premise, hypothesis], outputs=Final)

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