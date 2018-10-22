#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESIM Model
"""

import logging

import keras.backend as K
from keras.layers import Dropout, Input, Lambda, Concatenate
from keras.models import Model

from modules.esim import EmbeddingLayer, EncoderLayer, ExternalKnowledgeLayer, \
    FeedForwardLayer, InferenceLayer, PoolingLayer, ProjectionLayer
from modules.utils import get_regularizer, MaskMultiply

logger = logging.getLogger(__name__)


class ESIM(object):
    @classmethod
    def from_config(cls, config, embeddings):
        # 1, Embedding the input and project the embeddings
        premise_input = Input(shape=(None,), dtype='int32', name='sentence1')
        premise = Lambda(lambda x: x, name='premise')(premise_input)
        premise_mask_input = Input(shape=(None,), dtype='int32', name='sentence1_mask')
        premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)

        hypothesis_input = Input(shape=(None,), dtype='int32', name='sentence2')
        hypothesis = Lambda(lambda x: x, name='hypothesis')(hypothesis_input)
        hypothesis_mask_input = Input(shape=(None,), dtype='int32', name='sentence2_mask')
        hypothesis_mask = Lambda(lambda x: K.cast(x, 'float32'))(hypothesis_mask_input)

        premise_knowledge_input = Input(shape=(None, None, 5), dtype='float32', name='KBph')
        premise_knowledge = Lambda(lambda x: x)(premise_knowledge_input)
        hypothesis_knowledge_input = Input(shape=(None, None, 5), dtype='float32', name='KBhp')
        hypothesis_knowledge = Lambda(lambda x: x)(hypothesis_knowledge_input)

        main_embedding_layer = EmbeddingLayer.from_config(
            config=config['embeddings']['main'],
            embeddings=embeddings)

        embedded = dict()
        embedded['main'] = {
            'premise': main_embedding_layer(premise),
            'hypothesis': main_embedding_layer(hypothesis)
        }

        if config['embeddings'].get('residual') is not None:
            residual_embedding_layer = EmbeddingLayer.from_config(
                config=config['embeddings']['residual'],
                embeddings=embeddings)
            embedded['residual'] = {
                'premise': residual_embedding_layer(premise),
                'hypothesis': residual_embedding_layer(hypothesis)
            }

        premise = Dropout(config["dropout"])(embedded['main']['premise'])
        hypothesis = Dropout(config["dropout"])(embedded['main']['hypothesis'])

        input_encoder = EncoderLayer.from_config(config['encoder'])

        embedded['contextual'] = {
            'premise': input_encoder(premise),
            'hypothesis': input_encoder(hypothesis),
        }

        premise = embedded['contextual']['premise']
        hypothesis = embedded['contextual']['hypothesis']

        # TODO(tomwesolowski): Add Residual connections.

        premise = MaskMultiply()([premise, premise_mask])
        hypothesis = MaskMultiply()([hypothesis, hypothesis_mask])

        inference_layer = InferenceLayer.from_config(config['inference'])

        (premise_knowledge_vector, hypothesis_knowledge_vector,
         premise_soft_attention, hypothesis_soft_attention) = inference_layer(
            inputs=[premise, hypothesis],
            knowledge=[premise_knowledge, hypothesis_knowledge],
        )

        premise_external_knowledge_vector = None
        hypothesis_external_knowledge_vector = None

        if config.get('external_knowledge') is not None:
            external_knowledge_layer = ExternalKnowledgeLayer.from_config(
                config['external_knowledge'])

            external_knowledge_embedding = embedded[
                config['external_knowledge'].get('embedding', 'contextual')]

            (premise_external_knowledge_vector,
             hypothesis_external_knowledge_vector) = external_knowledge_layer(
                inputs=[external_knowledge_embedding['premise'],
                        external_knowledge_embedding['hypothesis']],
                knowledge=[premise_knowledge, hypothesis_knowledge],
                soft_attention=[premise_soft_attention, hypothesis_soft_attention]
            )

        if (premise_external_knowledge_vector is not None
                and hypothesis_external_knowledge_vector is not None):
            premise = Concatenate()(
                premise_knowledge_vector + [premise_external_knowledge_vector])
            hypothesis = Concatenate()(
                hypothesis_knowledge_vector + [hypothesis_external_knowledge_vector])

        projection = ProjectionLayer.from_config(config['projection'])

        premise = projection(premise)  # [-1, Psize, emb_size]
        hypothesis = projection(hypothesis)  # [-1, Hsize, emb_size]

        premise = Dropout(config["dropout"])(premise)
        hypothesis = Dropout(config["dropout"])(hypothesis)

        if (premise_external_knowledge_vector is not None
                and hypothesis_external_knowledge_vector is not None):
            premise = Concatenate()([premise, premise_external_knowledge_vector])
            hypothesis = Concatenate()([hypothesis, hypothesis_external_knowledge_vector])

        inference_encoder = EncoderLayer.from_config(config['inference_encoder'])

        embedded['inference'] = {
            'premise': inference_encoder(premise),
            'hypothesis': inference_encoder(hypothesis),
        }

        premise = embedded['inference']['premise']
        hypothesis = embedded['inference']['hypothesis']

        premise = MaskMultiply()([premise, premise_mask])
        hypothesis = MaskMultiply()([hypothesis, hypothesis_mask])

        pooling = PoolingLayer.from_config(config['pooling'])
        inference_final_vector = pooling(
            inputs=[premise, hypothesis],
            masks=[premise_mask, hypothesis_mask]
        )

        inference_final_vector = Dropout(config["dropout"])(inference_final_vector)

        for ff_config in config['feed_forwards']:
            ff_layer = FeedForwardLayer.from_config(ff_config)
            inference_final_vector = ff_layer(inference_final_vector)

        prediction = inference_final_vector

        model_input = [premise_input, premise_mask_input,
                       hypothesis_input, hypothesis_mask_input]

        if config['read_knowledge']:
            model_input += [premise_knowledge_input, hypothesis_knowledge_input]

        model = Model(inputs=model_input, outputs=prediction)

        print(model.summary())

        return model