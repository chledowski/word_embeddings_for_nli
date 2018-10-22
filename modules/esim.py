from abc import abstractmethod

import keras.backend as K
from keras.activations import softmax
from keras.layers import Add, Subtract, Dense, TimeDistributed, Lambda, Bidirectional, \
    Dot, Permute, Multiply, Concatenate, CuDNNLSTM
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from common.registrable import Registrable
from modules.utils import get_regularizer, ScaledRandomNormal


class EmbeddingLayer(object):
    def __init__(self, name, trainable, weights):
        self.model = Embedding(
            weights.shape[0],
            weights.shape[1],
            weights=[weights],
            trainable=trainable,
            name=name,
            mask_zero=False
        )

    def __call__(self, inputs):
        return self.model(inputs)

    @classmethod
    def from_config(cls, config, embeddings):
        return cls(
            name=config['name'],
            trainable=config['trainable'],
            weights=embeddings[config['name']].load()
        )


class EncoderLayer(object):
    def __init__(self, name, cudnn, units, initializer, regularizer, regularizer_strength):
        if cudnn:
            self._lstm = CuDNNLSTM
        else:
            self._lstm = LSTM

        self._regularizer = get_regularizer(regularizer, regularizer_strength)

        self.model = Bidirectional(
            name=name,
            layer=self._lstm(
                units=units,
                kernel_initializer=initializer,
                recurrent_initializer=initializer,
                recurrent_regularizer=self._regularizer,
                bias_regularizer=self._regularizer,
                return_sequences=True
            )
        )

    def __call__(self, input):
        return self.model(input)

    @classmethod
    def from_config(cls, config):
        return cls(
            name=config['name'],
            cudnn=config['cudnn'],
            units=config['units'],
            initializer=config['initializer'],
            regularizer=config['regularizer'],
            regularizer_strength=config['regularizer_strength'],
        )


class InferenceLayer(object):
    def __init__(self, similarity, kim_attention_boost=0.0):
        # 3, Score each words and calc score matrix Eph.
        self._similarity = similarity
        self._kim_attention_boost = kim_attention_boost

        if similarity == 'dot':
            self._attention_matrix = Dot(axes=(2, 2), name='attention_matrix')
        else:
            raise ValueError()

    def __call__(self, inputs, knowledge):
        """
        :param inputs: [premise, hypothesis]
        :return: [premise_knowledge_vector, hypothesis_knowledge_vector,
                  premise_attention_matrix, hypothesis_attention_matrix]
        """
        premise, hypothesis = inputs
        premise_knowledge, hypothesis_knowledge = knowledge

        premise_attention = self._attention_matrix(inputs)
        if self._kim_attention_boost != 0.0:
            attention_boost = self._kim_attention_boost
            boost = Lambda(
                lambda x: attention_boost * K.cast(K.greater(K.sum(x, axis=-1), 0.),
                                                   K.floatx()))
            premise_attention = Add()([premise_attention, boost(premise_knowledge)])
        hypothesis_attention = Permute((2, 1))(premise_attention)

        premise_soft_attention = Lambda(lambda x: softmax(x))(premise_attention)
        hypothesis_soft_attention = Lambda(lambda x: softmax(x))(hypothesis_attention)

        premise_aligned = Dot((2, 1))([premise_soft_attention, hypothesis])
        hypothesis_aligned = Dot((2, 1))([hypothesis_soft_attention, premise])

        premise_knowledge_vector = [
            premise,
            premise_aligned,
            Subtract()([premise, premise_aligned]),
            Multiply()([premise, premise_aligned])
        ]
        hypothesis_knowledge_vector = [
            hypothesis,
            hypothesis_aligned,
            Subtract()([hypothesis, hypothesis_aligned]),
            Multiply()([hypothesis, hypothesis_aligned])
        ]
        return [premise_knowledge_vector, hypothesis_knowledge_vector,
                premise_soft_attention, hypothesis_soft_attention]

    @classmethod
    def from_config(cls, config):
        return cls(
            similarity=config['similarity'],
            kim_attention_boost=config.get('kim_attention_boost', 0.0)
        )


class ExternalKnowledgeLayer(Registrable):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, inputs, knowledge, soft_attention):
        pass

    @classmethod
    def from_config(cls, config):
        return cls.by_name(config['name'])._load(config)


@ExternalKnowledgeLayer.register('kim')
class KIMExternalKnowledgeLayer(ExternalKnowledgeLayer):
    def __init__(self, ilambda):
        super(KIMExternalKnowledgeLayer, self).__init__()
        self._ilambda = ilambda

    def __call__(self, inputs, knowledge, soft_attention):
        premise_soft_attention, hypothesis_soft_attention = soft_attention
        premise_knowledge, hypothesis_knowledge = knowledge

        soften_knowledge = Lambda(lambda x: K.expand_dims(x[0]) * x[1])

        premise_vector = soften_knowledge([
            premise_soft_attention, premise_knowledge])  # [-1, Psize, Hsize, 5]
        hypothesis_vector = soften_knowledge([
            hypothesis_soft_attention, hypothesis_knowledge])  # [-1, Hsize, Psize, 5]

        ilambda = self._ilambda
        ilambda_multiply = Lambda(lambda x: ilambda * K.sum(x, axis=-2))

        premise_vector = ilambda_multiply(premise_vector)  # [-1, Psize, 5]
        hypothesis_vector = ilambda_multiply(hypothesis_vector)  # [-1, Hsize, 5]

        return [premise_vector, hypothesis_vector]

    @classmethod
    def _load(cls, config):
        return cls(
            ilambda=config['ilambda']
        )


@ExternalKnowledgeLayer.register('dot-i')
class DOTIExternalKnowledgeLayer(ExternalKnowledgeLayer):
    def __init__(self, ilambda):
        super(DOTIExternalKnowledgeLayer, self).__init__()
        self._ilambda = ilambda

    def __call__(self, inputs, knowledge, soft_attention):
        premise, hypothesis = inputs
        premise_soft_attention, hypothesis_soft_attention = soft_attention

        premise_dot = Dot(axes=(2, 2))([premise, hypothesis])  # [batch_size, Psize, Hsize]
        hypothesis_dot = Permute((2, 1))(premise_dot)  # [batch_size, Hsize, Psize]

        ilambda = self._ilambda
        align = Lambda(lambda x: ilambda * K.sum(x[0] * x[1], axis=-1, keepdims=True))

        premise_vector = align([premise_soft_attention, premise_dot])  # [batch_size, Psize, 1]
        hypothesis_vector = align([hypothesis_soft_attention, hypothesis_dot])  # [batch_size, Hsize, 1]

        return [premise_vector, hypothesis_vector]

    @classmethod
    def _load(cls, config):
        return cls(
            ilambda=config['ilambda']
        )


class ProjectionLayer(object):
    def __init__(self, units, activation, regularizer, regularizer_strength):
        self._regularizer = get_regularizer(regularizer, regularizer_strength)

        self.model = TimeDistributed(
            Dense(units=units,
                  kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
                  kernel_regularizer=self._regularizer,
                  bias_regularizer=self._regularizer,
                  activation=activation),
            name='projection'
        )

    def __call__(self, inputs):
        return self.model(inputs)

    @classmethod
    def from_config(cls, config):
        return cls(
            units=config['units'],
            activation=config['activation'],
            regularizer=config['regularizer'],
            regularizer_strength=config['regularizer_strength'],
        )


class PoolingLayer(object):
    def __init__(self, operations):
        self._operations = operations

    def __call__(self, inputs, masks):
        premise, hypothesis = inputs
        premise_mask, hypothesis_mask = masks

        pool_avg = Lambda(lambda x: K.sum(x[0], axis=1) / K.sum(x[1], axis=-1, keepdims=True))
        pool_max = Lambda(lambda x: K.max(x, axis=1))

        return Concatenate()([
            pool_avg([premise, premise_mask]),
            pool_max(premise),
            pool_avg([hypothesis, hypothesis_mask]),
            pool_max(hypothesis)
        ])

    @classmethod
    def from_config(cls, config):
        return cls(
            operations=config['operations'],
        )


class FeedForwardLayer(object):
    def __init__(self, name, units, activation, regularizer, regularizer_strength):
        self._regularizer = get_regularizer(regularizer, regularizer_strength)
        self.model = Dense(units,
                           kernel_initializer=ScaledRandomNormal(stddev=1.0, scale=0.01),
                           kernel_regularizer=self._regularizer,
                           bias_regularizer=self._regularizer,
                           activation=activation,
                           name=name)

    def __call__(self, inputs):
        return self.model(inputs)

    @classmethod
    def from_config(cls, config):
        return cls(
            name=config['name'],
            units=config['units'],
            activation=config['activation'],
            regularizer=config['regularizer'],
            regularizer_strength=config['regularizer_strength'],
        )


class ResidualLayer(object):
    pass
    # TODO(tomwesolowski): Refactor
    # if 'residual_embedding' in config and config['residual_embedding']['active']:
    #     if config['residual_embedding']['type'] == 'add':
    #         def _add_and_rotate(x, ortho_matrix):
    #             contextual, residual = x
    #             residual = K.dot(residual, K.constant(ortho_matrix, dtype='float32'))
    #             residual = Concatenate()([residual, residual])
    #             return Add()([contextual, residual])
    #         residual_connection = Lambda(_add_and_rotate, name='residual_embeds',
    #                                      arguments={
    #                                          'ortho_matrix': ortho_matrix
    #                                      })
    #     elif config['residual_embedding']['type'] == 'concat':
    #         residual_connection = Concatenate(name='residual_embeds')
    #     elif config['residual_embedding']['type'] == 'mod_drop':
    #         def _drop_mod(embeddings, normalize):
    #             def _mod_drop_train(contextual, residual):
    #                 keep_configs = K.constant([[0, 1],
    #                                            [1, 0],
    #                                            [1, 1]], dtype='float32')
    #
    #                 # scale by 1.0 / keep_prob
    #                 keep_configs_probs = K.mean(keep_configs, axis=0, keepdims=True)
    #                 keep_configs *= 1.0 / keep_configs_probs
    #
    #                 # [batch_size, sen_length]
    #                 selectors = K.random_uniform(K.shape(contextual)[:2], 0, 3, 'int32')
    #
    #                 # [batch_size, sen_length, 2, 1]
    #                 keep = K.expand_dims(K.gather(keep_configs, selectors))
    #
    #                 # [batch_size, sen_length, 2, 2*emb_dim]
    #                 stacked_embeddings = K.stack([contextual, residual], axis=2)
    #
    #                 # [batch_size, sen_length, 2*emb_dim]
    #                 return K.sum(keep * stacked_embeddings, axis=2)
    #
    #             def _mod_drop_test(contextual, residual):
    #                 return Add()([contextual, residual])
    #
    #             contextual, residual = embeddings
    #             # contextual: [batch, sen_length, 2*emb_dim]
    #             # residual: [batch, sen_length, emb_dim]
    #             residual = Concatenate()([residual, residual])
    #
    #             if normalize:
    #                 # [batch_size, sen_length, 1]
    #                 residual_norm = tf.norm(residual, axis=-1, keepdims=True)
    #                 # [batch_size, sen_length, 2*emb_dim]
    #                 unit_contextual = K.l2_normalize(contextual, axis=-1)
    #                 contextual = unit_contextual * residual_norm
    #
    #             return K.switch(K.learning_phase(),
    #                             _mod_drop_train(contextual, residual),
    #                             _mod_drop_test(contextual, residual))
    #
    #         residual_connection = Lambda(_drop_mod,
    #                                      name='residual_embeds',
    #                                      arguments={
    #                                          'normalize': config['residual_embedding']['normalize']
    #                                      })
    #     else:
    #         raise ValueError("Unknown residual conn. type:", config['residual_embedding']['type'])
    #
    #     embed_p = residual_connection([embed_p, embed_second_p])
    #     embed_h = residual_connection([embed_h, embed_second_h])
    #