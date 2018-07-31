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

    embedding_matrix = prep_embedding_matrix(config, data)

    embed = Embedding(data.vocab.size(), config["embedding_dim"],
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

    attention_lambda = config['attention_lambda']
    KBatt = Lambda(lambda x: attention_lambda * K.cast(K.greater(K.sum(x, axis=-1), 0.), K.floatx()))(KBph)

    embed_p = embed(premise)  # [batchsize, Psize, Embedsize]
    embed_h = embed(hypothesis)  # [batchsize, Hsize, Embedsize]

    # FIX(tomwesolowski): Add dropout
    embed_p = Dropout(config["dropout"])(embed_p)
    embed_h = Dropout(config["dropout"])(embed_h)

    # 2, Encoder words with its surrounding context
    bilstm_encoder = Bidirectional(
        LSTM(units=config["embedding_dim"],
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
    # FIX(tomwesolowski): Add attention lambda to words in relation.
    if config['useitrick']:
        Eph = Add()([Eph, KBatt])
    Ep_soft = Lambda(lambda x: softmax(x))(Eph)  # [batch_size, Psize, Hsize]

    Ehp = Permute((2, 1))(Eph)  # [batch_size, Hsize, Psize]
    Eh_soft = Lambda(lambda x: softmax(x))(Ehp)  # [batch_size, Hsize, Psize]

    # Ep_soft = Multiply()([Ep_soft, premise_mask])
    # Eh_soft = Multiply()([Eh_soft, hypothesis_mask])

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
        knowledge_lambda = config['i_lambda']
        i_1 = Lambda(lambda x: K.expand_dims(x[0]) * x[1])([Ep_soft, KBph])  # [-1, Psize, Hsize, 5]
        i_2 = Lambda(lambda x: K.expand_dims(x[0]) * x[1])([Eh_soft, KBhp])  # [-1, Hsize, Psize, 5]
        i_1 = Lambda(lambda x: knowledge_lambda * K.sum(x, axis=-2))(i_1)  # [-1, Psize, 5]
        i_2 = Lambda(lambda x: knowledge_lambda * K.sum(x, axis=-2))(i_2)  # [-1, Hsize, 5]
        PremAlign = Concatenate()([embed_p, PremAlign, sb_1, mm_1, i_1])  # [batch_size, Psize, 2*unit + 5]
        HypoAlign = Concatenate()([embed_h, HypoAlign, sb_2, mm_2, i_2])  # [batch_size, Hsize, 2*unit + 5]
    else:
        PremAlign = Concatenate()([embed_p, PremAlign, sb_1, mm_1])  # [batch_size, Psize, 2*unit]
        HypoAlign = Concatenate()([embed_h, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]

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

    if config['useitrick']:
        PremAlign = Concatenate()([PremAlign, i_1])
        HypoAlign = Concatenate()([HypoAlign, i_2])

    # 5, Final biLSTM < Encoder + Softmax Classifier
    bilstm_decoder = Bidirectional(
        LSTM(units=300,
                 # FIX(tomwesolowski): 26.07 Add Orthogonal and set use_bias = True
                 kernel_initializer=Orthogonal(seed=config["seed"]),
                 recurrent_initializer=Orthogonal(seed=config["seed"]),
                 return_sequences=True),
            name='finaldecoder')  # [-1,2*units]

    final_p = bilstm_decoder(PremAlign)
    final_h = bilstm_decoder(HypoAlign)

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
    if config['useitrick']:
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