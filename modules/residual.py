from abc import abstractmethod

import keras.backend as K
import tensorflow as tf
from keras.layers import Add, Concatenate, Lambda

from common.registrable import Registrable
from utils.algebra import ortho_weight


class ResidualLayer(Registrable):
    """
    Base class for residual connections.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, contextualized, residual):
        pass

    @classmethod
    def from_config(cls, config):
        return cls.by_name(config['name'])._load(config)


@ResidualLayer.register('add')
class AddResidualLayer(ResidualLayer):
    """
    Simple addition (with rotation, optionally) of contextualized and residual embeddings.
    """

    def __init__(self, rotate, embedding_dim):
        super(ConcatResidualLayer, self).__init__()
        self._embedding_dim = embedding_dim
        if rotate:
            self._orthogonal_matrix = ortho_weight(self._embedding_dim)

        def _add_and_rotate(x, ortho_matrix, rotate):
            contextual, residual = x
            if rotate:
                residual = K.dot(residual, K.constant(ortho_matrix, dtype='float32'))
            residual = Concatenate()([residual, residual])
            return Add()([contextual, residual])

        self.model = Lambda(_add_and_rotate, name='residual_embeds',
                             arguments={
                                 'ortho_matrix': self._orthogonal_matrix,
                                 'rotate': rotate
                             })

    def __call__(self, contextualized, residual):
        return self.model([contextualized, residual])

    @classmethod
    def _load(cls, config):
        return cls(
            rotate=config['rotate'],
            embedding_dim=config['embedding_dim']
        )


@ResidualLayer.register('concat')
class ConcatResidualLayer(ResidualLayer):
    """
    Simple concatenation of contextualized and residual embeddings.
    """

    def __init__(self):
        super(ConcatResidualLayer, self).__init__()
        self.model = Concatenate()

    def __call__(self, contextualized, residual):
        return self.model([contextualized, residual])

    @classmethod
    def _load(cls, config):
        return cls()


@ResidualLayer.register('mod-drop')
class ModDropResidualLayer(ResidualLayer):
    """
    Joins embeddings with modality dropout.
    """

    def __init__(self, normalize):
        super(ModDropResidualLayer, self).__init__()
        self.model = Lambda(self._mod_drop,
                            name='residual_embeds',
                            arguments={
                                'normalize': normalize
                            })

    def _mod_drop(self, embeddings, normalize):
        def _mod_drop_train(contextual, residual):
            keep_configs = K.constant([[0, 1],
                                       [1, 0],
                                       [1, 1]], dtype='float32')

            # scale by 1.0 / keep_prob
            keep_configs_probs = K.mean(keep_configs, axis=0, keepdims=True)
            keep_configs *= 1.0 / keep_configs_probs

            # [batch_size, sen_length]
            selectors = K.random_uniform(K.shape(contextual)[:2], 0, 3, 'int32')

            # [batch_size, sen_length, 2, 1]
            keep = K.expand_dims(K.gather(keep_configs, selectors))

            # [batch_size, sen_length, 2, 2*emb_dim]
            stacked_embeddings = K.stack([contextual, residual], axis=2)

            # [batch_size, sen_length, 2*emb_dim]
            return K.sum(keep * stacked_embeddings, axis=2)

        def _mod_drop_test(contextual, residual):
            return Add()([contextual, residual])

        contextual, residual = embeddings
        # contextual: [batch, sen_length, 2*emb_dim]
        # residual: [batch, sen_length, emb_dim]
        residual = Concatenate()([residual, residual])

        if normalize:
            # [batch_size, sen_length, 1]
            residual_norm = tf.norm(residual, axis=-1, keepdims=True)
            # [batch_size, sen_length, 2*emb_dim]
            unit_contextual = K.l2_normalize(contextual, axis=-1)
            contextual = unit_contextual * residual_norm

        return K.switch(K.learning_phase(),
                        _mod_drop_train(contextual, residual),
                        _mod_drop_test(contextual, residual))

    def __call__(self, contextualized, residual):
        return self.model([contextualized, residual])

    @classmethod
    def _load(cls, config):
        return cls(
            normalize=config['normalize']
        )