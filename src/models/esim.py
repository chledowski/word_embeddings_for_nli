#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple model definitions
"""

import logging
import numpy as np
import os
import tensorflow as tf

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
from src.models.elmo import ElmoEmbeddings, WeightElmoEmbeddings
from src.models.keras_utils import ScaledRandomNormal

logger = logging.getLogger(__name__)


def esim(config, data):
    logger.info('Vocab size = {}'.format(data.vocab.size()))
    logger.info('Using {} embedding'.format(config["embedding_name"]))

    embedding_matrix = prep_embedding_matrix(config, data, config["embedding_name"])

    embed = Embedding(data.vocab.size(), config["embedding_dim"],
                      weights=[embedding_matrix],
                      trainable=config["train_embeddings"],
                      name='embedding',
                      mask_zero=False)

    if config["embedding_second_name"] != config["embedding_name"]:
        embedding_second_matrix = prep_embedding_matrix(config, data, config["embedding_second_name"])
    else:
        embedding_second_matrix = embedding_matrix

    max_norm_second = np.max(np.sum(embedding_second_matrix ** 2, axis=-1), axis=-1)
    # print("max_norm_second: ", max_norm_second)

    logger.info('Using {} embedding'.format(config["embedding_second_name"]))

    # 1, Embedding the input and project the embeddings
    premise = Input(shape=(None,), dtype='int32', name='premise')
    premise_mask_input = Input(shape=(None,), dtype='int32', name='premise_mask_input')
    hypothesis = Input(shape=(None,), dtype='int32', name='hypothesis')
    hypothesis_mask_input = Input(shape=(None,), dtype='int32', name='hypothesis_mask_input')
    KBph = Input(shape=(None, None, 5), dtype='float32', name='KBph')
    KBhp = Input(shape=(None, None, 5), dtype='float32', name='KBhp')

    if config['use_elmo']:
        elmo_embed = ElmoEmbeddings(config)
        elmo_pre_weight = WeightElmoEmbeddings(config)
        elmo_post_weight = WeightElmoEmbeddings(config)
                                           # premise_placeholder = K.placeholder(shape=(None, None), dtype='int32')
        # hypothesis_placeholder = K.placeholder(shape=(None, None), dtype='int32')
        premise_elmo_input = Input(shape=(None,), dtype='int32', name='premise_elmo_input')
        hypothesis_elmo_input = Input(shape=(None,), dtype='int32',
                                      name='hypothesis_elmo_input')

    premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)
    hypothesis_mask = Lambda(lambda x: K.cast(x, 'float32'))(hypothesis_mask_input)

    # [-1, Psize, 1]
    premise_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask)
    # [-1, Hsize, 1]
    hypothesis_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(hypothesis_mask)

    a_lambda = config['a_lambda']
    KBatt = Lambda(lambda x: a_lambda * K.cast(K.greater(K.sum(x, axis=-1), 0.), K.floatx()))(KBph)

    embed_orig_p = embed(premise)  # [batchsize, Psize, Embedsize]
    embed_orig_h = embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    if config['use_elmo']:
        # TODO(tomwesolowski): Elmo - add L2 regularization with coef. 0.0001 to all layers
        # TODO(tomwesolowski): Elmo - add 50% dropout after attention layer
        # TODO(tomwesolowski): Elmo - add L2 reg. on elmo embeddings to the loss with coef. 0.001
        # TODO(tomwesolowski): Elmo - clip_gradient_norm=5.0
        elmo_p = elmo_embed([premise_elmo_input, premise_mask_exp])
        elmo_p = Dropout(config["dropout"])(elmo_p)
        weighted_elmo_p = elmo_pre_weight(elmo_p)

        elmo_h = elmo_embed([hypothesis_elmo_input, hypothesis_mask_exp])
        elmo_h = Dropout(config["dropout"])(elmo_h)
        weighted_elmo_h = elmo_post_weight(elmo_h)

        embed_p = Concatenate(axis=2)([embed_orig_p, weighted_elmo_p])
        embed_h = Concatenate(axis=2)([embed_orig_h, weighted_elmo_h])
    else:
        embed_p = embed_orig_p
        embed_h = embed_orig_h

    if config['knowledge_after_lstm'] in ['dot', 'euc']:
        embed_second = Embedding(data.vocab.size(), config["embedding_dim"],
                                  weights=[embedding_second_matrix],
                                  trainable=config["train_embeddings"],
                                  mask_zero=False)
        embed_second_p = embed_second(premise)  # [batchsize, Psize, Embedsize]
        embed_second_h = embed_second(hypothesis)  # [batchsize, Hsize, Embedsize]

    # TODO(tomwesolowski): Elmo - change to variational dropout
    # FIX(tomwesolowski): Add dropout
    embed_p = Dropout(config["dropout"])(embed_p)
    embed_h = Dropout(config["dropout"])(embed_h)

    if 'cudnn' not in config or config['cudnn']:
        lstm_layer = CuDNNLSTM
    else:
        lstm_layer = LSTM

    # 2, Encoder words with its surrounding context
    bilstm_encoder = Bidirectional(
        lstm_layer(
            units=config["embedding_dim"],
            # FIX(tomwesolowski): 26.07 Add Orthogonal and set use_bias = True
            kernel_initializer=Orthogonal(seed=config["seed"]),
            recurrent_initializer=Orthogonal(seed=config["seed"]),
            recurrent_regularizer=l2(config["l2_weight_regularization"]),
            bias_regularizer=l2(config["l2_weight_regularization"]),
            return_sequences=True),
        name='bilstm'
    )

    # FIX(tomwesolowski): Remove dropout
    embed_p = bilstm_encoder(embed_p)  # [-1, sen_len, 2*dim]
    embed_h = bilstm_encoder(embed_h)

    if config['residual_embedding']:
        if config['residual_embedding_mod_drop']:
            # x[0]: [-1, sen_len, 2*dim]
            # x[1]: [-1, sen_len, 2*dim]
            def _residual_embeds_mod_dropout(x):
                prob = tf.random_uniform(shape=tf.shape(x[0])[:2])
                prob = tf.expand_dims(prob, -1)
                return prob * x[0] + (1. - prob) * x[1]

            residual_embeds = Lambda(_residual_embeds_mod_dropout,
                                     name='residual_embeds')
        else:
            residual_embeds = Add(name='residual_embeds')
        embed_orig_p_twice = Concatenate(axis=2)([embed_orig_p, embed_orig_p])
        embed_orig_h_twice = Concatenate(axis=2)([embed_orig_h, embed_orig_h])
        embed_p = residual_embeds([embed_p, embed_orig_p_twice])
        embed_h = residual_embeds([embed_h, embed_orig_h_twice])

    if config['use_elmo'] and config['elmo_after_lstm']:
        weighted_elmo_post_p = elmo_post_weight(elmo_p)
        weighted_elmo_post_h = elmo_post_weight(elmo_h)

        embed_p = Concatenate(axis=2)([embed_p, weighted_elmo_post_p])
        embed_h = Concatenate(axis=2)([embed_h, weighted_elmo_post_h])
    embed_p = Multiply()([embed_p, premise_mask_exp])
    embed_h = Multiply()([embed_h, hypothesis_mask_exp])

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

    if config['use_elmo']:
        # TODO(tomwesolowski): Double check if properly implemented.
        PremAlign = Dropout(config["dropout"])(PremAlign)
        HypoAlign = Dropout(config["dropout"])(HypoAlign)

    translate = TimeDistributed(
        Dense(config["embedding_dim"],
              # FIX(tomwesolowski): Remove regularizers
              # FIX(tomwesolowski): Add initializer
              kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
              kernel_regularizer=l2(config["l2_weight_regularization"]),
              bias_regularizer=l2(config["l2_weight_regularization"]),
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
        lstm_layer(
            units=300,
            # FIX(tomwesolowski): 26.07 Add Orthogonal and set use_bias = True
            kernel_initializer=Orthogonal(seed=config["seed"]),
            recurrent_initializer=Orthogonal(seed=config["seed"]),
            recurrent_regularizer=l2(config["l2_weight_regularization"]),
            bias_regularizer=l2(config["l2_weight_regularization"]),
            return_sequences=True),
        name='finaldecoder')  # [-1,2*units]

    final_p = bilstm_decoder(PremAlign)  # [-1, Psize, 600]
    final_h = bilstm_decoder(HypoAlign)  # [-1, Hsize, 600]

    final_p = Multiply()([final_p, premise_mask_exp])
    final_h = Multiply()([final_h, hypothesis_mask_exp])

    AveragePooling = Lambda(lambda x: K.sum(x[0], axis=1) / K.sum(x[1], axis=-1, keepdims=True))  # outs [-1, dim]
    MaxPooling = Lambda(lambda x: K.max(x, axis=1))  # outs [-1, dim]
    avg_p = AveragePooling([final_p, premise_mask])
    avg_h = AveragePooling([final_h, hypothesis_mask])
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

        weight_aw_p = Lambda(lambda x: x[1] * (K.exp(x[0] - K.max(x[0], axis=1, keepdims=True))))(
            [weight_aw_p, premise_mask])
        weight_aw_h = Lambda(lambda x: x[1] * (K.exp(x[0] - K.max(x[0], axis=1, keepdims=True))))(
            [weight_aw_h, hypothesis_mask])

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
                  kernel_regularizer=l2(config["l2_weight_regularization"]),
                  bias_regularizer=l2(config["l2_weight_regularization"]),
                  # FIX(tomwesolowski): Add tanh activation
                  activation='relu' if config['use_elmo'] else 'tanh',
                  name='dense300_' + config["dataset"])(Final)
    Final = Dropout(config["dropout"])(Final)
    Final = Dense(3,
                  # FIX(tomwesolowski): Add initializer
                  kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
                  kernel_regularizer=l2(config["l2_weight_regularization"]),
                  bias_regularizer=l2(config["l2_weight_regularization"]),
                  activation='softmax',
                  name='judge300_' + config["dataset"])(Final)

    model_input = [premise, premise_mask_input, hypothesis, hypothesis_mask_input]

    if config['useitrick'] or config['useatrick'] or config['usectrick'] or config['fullkim']:
        model_input += [KBph, KBhp]

    if config['use_elmo']:
        model_input += [premise_elmo_input, hypothesis_elmo_input]

    model = Model(inputs=model_input, outputs=Final)

    # def elmo_loss(y_true, y_pred):
    #     elmo_embeddings = Concatenate()([elmo_p, elmo_h, elmo_after_p, elmo_after_h])
    #     return (K.categorical_crossentropy(y_true, y_pred) +
    #             config['l2_elmo_regularization'] * K.sum(elmo_embeddings ** 2))

    # if config['use_elmo']:
    #     loss = elmo_loss
    # else:
    loss = 'categorical_crossentropy'

    if config["optimizer"] == 'rmsprop':
        model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"],
                                                   clipnorm=config["clip_gradient_norm"]),
                      loss=loss, metrics=['accuracy'])

    elif config["optimizer"] == 'sgd':
        model.compile(optimizer=optimizers.SGD(lr=config["learning_rate"],
                                               momentum=0.9,
                                               clipnorm=config["clip_gradient_norm"]),
                      loss=loss, metrics=['accuracy'])

    elif config["optimizer"] == 'adam':
        model.compile(optimizer=optimizers.Adam(lr=config["learning_rate"],
                                                clipnorm=config["clip_gradient_norm"]),
                      loss=loss, metrics=['accuracy'])

    print(model.summary())

    return model