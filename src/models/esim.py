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
from keras.initializers import Orthogonal
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
                      trainable=config["train_embeddings"],
                      mask_zero=True)

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(config["sentence_max_length"],), dtype='int32')
    premise_mask_input = Input(shape=(config["sentence_max_length"],), dtype='float32')
    hypothesis = Input(shape=(config["sentence_max_length"],), dtype='int32')
    hypothesis_mask_input = Input(shape=(config["sentence_max_length"],), dtype='float32')
    KBph = Input(shape=(config["sentence_max_length"], config["sentence_max_length"], 5), dtype='float32')
    KBhp = Input(shape=(config["sentence_max_length"], config["sentence_max_length"], 5), dtype='float32')

    embed_p = embed(premise)  # [batchsize, Psize, Embedsize]
    embed_h = embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    # 2, Encoder words with its surrounding context
    bilstm_encoder = Bidirectional(
        LSTM(units=config["embedding_dim"],
             kernel_initializer='orthogonal',
             use_bias=False,
             return_sequences=True))

    embed_p = Dropout(config["dropout"])(bilstm_encoder(embed_p))
    embed_h = Dropout(config["dropout"])(bilstm_encoder(embed_h))

    # [-1, Psize, 1]
    premise_mask = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask_input)
    # [-1, Hsize, 1]
    hypothesis_mask = Lambda(lambda x: K.expand_dims(x, axis=-1))(hypothesis_mask_input)

    embed_p = Lambda(lambda x: x[0]*x[1])([embed_p, premise_mask])
    embed_h = Lambda(lambda x: x[0]*x[1])([embed_h, hypothesis_mask])

    # 3, Score each words and calc score matrix Eph.
    F_p, F_h = embed_p, embed_h
    Ehp = Dot(axes=(2, 2))([F_h, F_p])  # [batch_size, Hsize, Psize]
    Eh_soft = Lambda(lambda x: softmax(x))(Ehp)  # [batch_size, Hsize, Psize]
    Eph = Permute((2, 1))(Ehp)  # [batch_size, Psize, Hsize]
    Ep_soft = Lambda(lambda x: softmax(x))(Eph)  # [batch_size, Psize, Hsize]

    Ep_soft = Multiply()([Ep_soft, premise_mask])
    Eh_soft = Multiply()([Eh_soft, hypothesis_mask])

    # def l2_pairwise_distance(AB):
    #     A, B = AB
    #     r = K.sum(A * A, 2, keepdims=True)
    #     s = K.sum(B * B, 2, keepdims=True)
    #     return r - 2 * Dot(axes=(2, 2))([A, B]) + K.permute_dimensions(s, (0, 2, 1))
    #
    # Dph = Lambda(l2_pairwise_distance)([F_p, F_h]) # [batch_size, Psize, Hsize]
    # Dhp = Permute((2, 1))(Dph) # [batch_size, Hsize, Psize]

    # 4, Normalize score matrix, encoder premesis and get alignment
    PremAlign = Dot((2, 1))([Ep_soft, embed_h])  # [-1, Psize, dim]
    HypoAlign = Dot((2, 1))([Eh_soft, embed_p])  # [-1, Hsize, dim]
    mm_1 = Multiply()([embed_p, PremAlign])
    mm_2 = Multiply()([embed_h, HypoAlign])
    sb_1 = Subtract()([embed_p, PremAlign])
    sb_2 = Subtract()([embed_h, HypoAlign])

    if config['useitrick']:
        knowledge_lambda = config['lambda']
        i_1 = Lambda(lambda x: K.expand_dims(x[0]) * x[1])([Ep_soft, KBph])  # [-1, Psize, Hsize, 5]
        i_2 = Lambda(lambda x: K.expand_dims(x[0]) * x[1])([Eh_soft, KBhp])  # [-1, Hsize, Psize, 5]
        i_1 = Lambda(lambda x: knowledge_lambda * K.sum(x, axis=-2))(i_1)  # [-1, Psize, 5]
        i_2 = Lambda(lambda x: knowledge_lambda * K.sum(x, axis=-2))(i_2)  # [-1, Hsize, 5]
        PremAlign = Concatenate()([embed_p, PremAlign, sb_1, mm_1, i_1])  # [batch_size, Psize, 2*unit + 5]
        HypoAlign = Concatenate()([embed_h, HypoAlign, sb_2, mm_2, i_2])  # [batch_size, Hsize, 2*unit + 5]
    else:
        PremAlign = Concatenate()([embed_p, PremAlign, sb_1, mm_1])  # [batch_size, Psize, 2*unit]
        HypoAlign = Concatenate()([embed_h, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]

    PremAlign = Dropout(config["dropout"])(PremAlign)
    HypoAlign = Dropout(config["dropout"])(HypoAlign)

    translate = TimeDistributed(Dense(config["embedding_dim"],
                                      kernel_regularizer=l2(1e-5),
                                      bias_regularizer=l2(1e-5),
                                      activation='relu'),
                                name='translate')

    PremAlign = translate(PremAlign) # [-1, Psize, emb_size]
    HypoAlign = translate(HypoAlign) # [-1, Hsize, emb_size]

    if config['useitrick']:
        PremAlign = Concatenate()([PremAlign, i_1])
        HypoAlign = Concatenate()([HypoAlign, i_2])

    # PremAlign = Multiply()([PremAlign, premise_mask])
    # HypoAlign = Multiply()([HypoAlign, hypothesis_mask])

    # 5, Final biLSTM < Encoder + Softmax Classifier
    bilstm_decoder = Bidirectional(
            LSTM(units=300,
                 return_sequences=True,
                 use_bias=False,
                 kernel_initializer='orthogonal'),
            name='finaldecoder')  # [-1,2*units]
    final_p = Dropout(config["dropout"])(bilstm_decoder(PremAlign))
    final_h = Dropout(config["dropout"])(bilstm_decoder(HypoAlign))

    final_p = Multiply()([final_p, premise_mask])
    final_h = Multiply()([final_h, hypothesis_mask])

    AveragePooling = Lambda(lambda x: K.sum(x[0], axis=1) / K.sum(x[1], axis=-1, keepdims=True))  # outs [-1, dim]
    MaxPooling = Lambda(lambda x: K.max(x, axis=1))  # outs [-1, dim]
    avg_p = AveragePooling([final_p, premise_mask_input])
    avg_h = AveragePooling([final_h, hypothesis_mask_input])
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
    # model = Model(inputs=[premise, hypothesis], outputs=Final)
    model = Model(inputs=[premise, premise_mask_input,
                          hypothesis, hypothesis_mask_input,
                          KBph, KBhp], outputs=Final)

    if config["optimizer"] == 'rmsprop':
        model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"],
                                                   clipnorm=config["clip_gradient_norm"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    elif config["optimizer"] == 'sgd':
        model.compile(optimizer=optimizers.SGD(lr=config["learning_rate"],
                                               momentum=0.9,
                                               clipnorm=config["clip_gradient_norm"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    elif config["optimizer"] == 'adam':
        model.compile(optimizer=optimizers.Adam(lr=config["learning_rate"],
                                                clipnorm=config["clip_gradient_norm"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    print((model.summary()))

    return model