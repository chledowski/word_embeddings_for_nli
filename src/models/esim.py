#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

import logging
import numpy as np
import os

import keras.backend as K
from keras import optimizers
from keras.activations import softmax
from keras.layers import Add, Subtract, Dense, Dropout, Input, TimeDistributed, Lambda, Bidirectional, \
    Dot, Permute, Multiply, Concatenate, Activation, CuDNNLSTM
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.initializers import Orthogonal
from keras.regularizers import l2


from src import DATA_DIR
from src.util.prepare_embedding import prep_embedding_matrix
from src.models.utils import ScaledRandomNormal

logger = logging.getLogger(__name__)

def esim(config, data):
    logger.info('Vocab size = {}'.format(data.vocab.size()))
    logger.info('Using {} embedding'.format(config["embedding_name"]))

    embedding_matrix = prep_embedding_matrix(config, data, config["embedding_name"])

    embed = Embedding(data.vocab.size(), config["embedding_dim"],
                      weights=[embedding_matrix],
                      input_length=config["sentence_max_length"],
                      trainable=config["train_embeddings"],
                      mask_zero=False)

    if config["embedding_second_name"] != config["embedding_name"]:
        embedding_second_matrix = prep_embedding_matrix(config, data, config["embedding_second_name"])
    else:
        embedding_second_matrix = embedding_matrix

    max_norm_second = np.max(np.sum(embedding_second_matrix ** 2, axis=-1), axis=-1)
    print("max_norm_second: ", max_norm_second)

    logger.info('Using {} embedding'.format(config["embedding_second_name"]))

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(config["sentence_max_length"],), dtype='int32')
    premise_mask_input = Input(shape=(config["sentence_max_length"],), dtype='float32')
    hypothesis = Input(shape=(config["sentence_max_length"],), dtype='int32')
    hypothesis_mask_input = Input(shape=(config["sentence_max_length"],), dtype='float32')
    KBph = Input(shape=(config["sentence_max_length"], config["sentence_max_length"], 5), dtype='float32')
    KBhp = Input(shape=(config["sentence_max_length"], config["sentence_max_length"], 5), dtype='float32')

    a_lambda = config['a_lambda']
    KBatt = Lambda(lambda x: a_lambda * K.cast(K.greater(K.sum(x, axis=-1), 0.), K.floatx()))(KBph)

    embed_p = embed(premise)  # [batchsize, Psize, Embedsize]
    embed_h = embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    if config['knowledge_after_lstm'] in ['dot', 'euc']:
        embed_second = Embedding(data.vocab.size(), config["embedding_dim"],
                                  weights=[embedding_second_matrix],
                                  input_length=config["sentence_max_length"],
                                  trainable=config["train_embeddings"],
                                  mask_zero=False)
        embed_second_p = embed_second(premise)  # [batchsize, Psize, Embedsize]
        embed_second_h = embed_second(hypothesis)  # [batchsize, Hsize, Embedsize]

    # FIX(tomwesolowski): Add dropout
    embed_p = Dropout(config["dropout"])(embed_p)
    embed_h = Dropout(config["dropout"])(embed_h)

    # 2, Encoder words with its surrounding context
    bilstm_encoder = Bidirectional(
        CuDNNLSTM(units=config["embedding_dim"],
             # FIX(tomwesolowski): 26.07 Add Orthogonal and set use_bias = True
             kernel_initializer=Orthogonal(seed=config["seed"]),
             recurrent_initializer=Orthogonal(seed=config["seed"]),
             return_sequences=True)
    )

    # FIX(tomwesolowski): Remove dropout
    embed_p = bilstm_encoder(embed_p)
    embed_h = bilstm_encoder(embed_h)

    # [-1, Psize, 1]
    premise_mask = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask_input)
    # [-1, Hsize, 1]
    hypothesis_mask = Lambda(lambda x: K.expand_dims(x, axis=-1))(hypothesis_mask_input)

    embed_p = Multiply()([embed_p, premise_mask])
    embed_h = Multiply()([embed_h, hypothesis_mask])

    # 3, Score each words and calc score matrix Eph.
    F_p, F_h = embed_p, embed_h
    Eph = Dot(axes=(2, 2))([F_p, F_h])  # [batch_size, Psize, Hsize]

    # # FIX(tomwesolowski): Add attention lambda to words in relation
    if config['useatrick'] or config['fullkim']:
        Eph = Add()([Eph, KBatt])

    Ep_soft = Lambda(lambda x: softmax(x))(Eph)  # [batch_size, Psize, Hsize]

    Ehp = Permute((2, 1))(Eph)  # [batch_size, Hsize, Psize]
    Eh_soft = Lambda(lambda x: softmax(x))(Ehp)  # [batch_size, Hsize, Psize]

    # 4, Normalize score matrix, encoder premesis and get alignment
    PremAlign = Dot((2, 1))([Ep_soft, embed_h])  # [-1, Psize, dim]
    HypoAlign = Dot((2, 1))([Eh_soft, embed_p])  # [-1, Hsize, dim]
    mm_1 = Multiply()([embed_p, PremAlign])
    mm_2 = Multiply()([embed_h, HypoAlign])
    sb_1 = Subtract()([embed_p, PremAlign])
    sb_2 = Subtract()([embed_h, HypoAlign])

    knowledge_lambda = config['i_lambda']

    prem_knowledge_vector = [embed_p, PremAlign, sb_1, mm_1]
    hypo_knowledge_vector = [embed_h, HypoAlign, sb_2, mm_2]
    prem_i_vector = []
    hypo_i_vector = []

    if config['knowledge_after_lstm'] in ['dot', 'euc']:
        if config['knowledge_after_lstm'] == 'dot':
            Dph = Dot(axes=(2, 2))([embed_second_p, embed_second_h])  # [batch_size, Psize, Hsize]
        elif config['knowledge_after_lstm'] == 'euc':
            def l2_pairwise_distance(AB):
                A, B = AB
                r = K.sum(A * A, 2, keepdims=True)
                s = K.sum(B * B, 2, keepdims=True)
                return r - 2 * Dot(axes=(2, 2))([A, B]) + K.permute_dimensions(s, (0, 2, 1))
            Dph = Lambda(l2_pairwise_distance)([embed_second_p, embed_second_h])  # [batch_size, Psize, Hsize]
            Dph = Lambda(lambda x: 1 - x / (2*max_norm_second))(Dph)

        Dhp = Permute((2, 1))(Dph)  # [batch_size, Hsize, Psize]
        i_1 = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([Ep_soft, Dph])  # [batch_size, Psize, 1]
        i_2 = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([Eh_soft, Dhp])  # [batch_size, Hsize, 1]
        i_1 = Lambda(lambda x: knowledge_lambda * x)(i_1)  # [-1, Psize, 1]
        i_2 = Lambda(lambda x: knowledge_lambda * x)(i_2)  # [-1, Hsize, 1]
        prem_i_vector += [i_1]
        hypo_i_vector += [i_2]
    if config['useitrick'] or config['fullkim']:
        kim_i_1 = Lambda(lambda x: K.expand_dims(x[0]) * x[1])([Ep_soft, KBph])  # [-1, Psize, Hsize, 5]
        kim_i_2 = Lambda(lambda x: K.expand_dims(x[0]) * x[1])([Eh_soft, KBhp])  # [-1, Hsize, Psize, 5]
        kim_i_1 = Lambda(lambda x: knowledge_lambda * K.sum(x, axis=-2))(kim_i_1)  # [-1, Psize, 5]
        kim_i_2 = Lambda(lambda x: knowledge_lambda * K.sum(x, axis=-2))(kim_i_2)  # [-1, Hsize, 5]
        prem_i_vector += [kim_i_1]
        hypo_i_vector += [kim_i_2]

    PremAlign = Concatenate()(prem_knowledge_vector + prem_i_vector)
    HypoAlign = Concatenate()(hypo_knowledge_vector + hypo_i_vector)

    translate = TimeDistributed(
        Dense(config["embedding_dim"],
              # FIX(tomwesolowski): Remove regularizers
              # FIX(tomwesolowski): Add initializer
              kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
              activation='relu'),
        name='translate')

    PremAlign = translate(PremAlign) # [-1, Psize, emb_size]
    HypoAlign = translate(HypoAlign) # [-1, Hsize, emb_size]

    PremAlign = Dropout(config["dropout"])(PremAlign)
    HypoAlign = Dropout(config["dropout"])(HypoAlign)

    if config['useitrick'] \
            or config['knowledge_after_lstm'] in ['dot', 'euc'] \
            or config['fullkim']:
        PremAlign = Concatenate()([PremAlign] + prem_i_vector)
        HypoAlign = Concatenate()([HypoAlign] + hypo_i_vector)

    # 5, Final biLSTM < Encoder + Softmax Classifier
    bilstm_decoder = Bidirectional(
        CuDNNLSTM(units=300,
                 # FIX(tomwesolowski): 26.07 Add Orthogonal and set use_bias = True
                 kernel_initializer=Orthogonal(seed=config["seed"]),
                 recurrent_initializer=Orthogonal(seed=config["seed"]),
                 return_sequences=True),
            name='finaldecoder')  # [-1,2*units]

    final_p = bilstm_decoder(PremAlign)  # [-1, Psize, 600]
    final_h = bilstm_decoder(HypoAlign)  # [-1, Hsize, 600]

    final_p = Multiply()([final_p, premise_mask])
    final_h = Multiply()([final_h, hypothesis_mask])

    AveragePooling = Lambda(lambda x: K.sum(x[0], axis=1) / K.sum(x[1], axis=-1, keepdims=True))  # outs [-1, dim]
    MaxPooling = Lambda(lambda x: K.max(x, axis=1))  # outs [-1, dim]
    avg_p = AveragePooling([final_p, premise_mask_input])
    avg_h = AveragePooling([final_h, hypothesis_mask_input])
    max_p = MaxPooling(final_p)
    max_h = MaxPooling(final_h)

    pooling_vectors = [avg_p, max_p, avg_h, max_h]

    if config['usectrick'] or config['fullkim']:
        weight_aw_p = Lambda(lambda x: K.sum(K.expand_dims(x[0]) * x[1], axis=-2))(
            [Ep_soft, KBph])  # [-1, Psize, 5]
        weight_aw_h = Lambda(lambda x: K.sum(K.expand_dims(x[0]) * x[1], axis=-2))(
            [Eh_soft, KBhp])  # [-1, Hsize, 5]

        gate_kb = TimeDistributed(
            Dense(1,
                kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
                activation='relu'))
        weight_aw_p = gate_kb(weight_aw_p)  # [-1, Psize, 1]
        weight_aw_h = gate_kb(weight_aw_h)  # [-1, Hsize, 1]
        weight_aw_p = Lambda(lambda x: K.squeeze(x, axis=-1))(weight_aw_p)  # [-1, Psize]
        weight_aw_h = Lambda(lambda x: K.squeeze(x, axis=-1))(weight_aw_h)  # [-1, Hsize]

        weight_aw_p = Lambda(lambda x: x[1] * (K.exp(x[0] - K.max(x[0], axis=1, keepdims=True))))([weight_aw_p, premise_mask_input])
        weight_aw_h = Lambda(lambda x: x[1] * (K.exp(x[0] - K.max(x[0], axis=1, keepdims=True))))([weight_aw_h, hypothesis_mask_input])

        weight_aw_p = Lambda(lambda x: x / K.sum(x, axis=1, keepdims=True))(weight_aw_p)
        weight_aw_h = Lambda(lambda x: x / K.sum(x, axis=1, keepdims=True))(weight_aw_h)

        aw_p = Dot((1, 1))([weight_aw_p, final_p])  # [-1, Psize] + [-1, Psize, 600] = [-1, 600]
        aw_h = Dot((1, 1))([weight_aw_h, final_h])  # [-1, 600]

        pooling_vectors += [aw_p, aw_h]

    Final = Concatenate()(pooling_vectors)
    Final = Dropout(config["dropout"])(Final)
    Final = Dense(300,
                  # FIX(tomwesolowski): Remove regularizers
                  # FIX(tomwesolowski): Add initializer
                  kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
                  # FIX(tomwesolowski): Add tanh activation
                  activation='tanh',
                  name='dense300_' + config["dataset"])(Final)
    Final = Dropout(config["dropout"])(Final)
    Final = Dense(3,
                  # FIX(tomwesolowski): Add initializer
                  kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
                  activation='softmax',
                  name='judge300_' + config["dataset"])(Final)

    model_input = [premise, premise_mask_input, hypothesis, hypothesis_mask_input]

    if config['useitrick'] or config['useatrick'] or config['usectrick'] or config['fullkim']:
        model_input += [KBph, KBhp]

    model = Model(inputs=model_input, outputs=Final)

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