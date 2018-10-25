import keras
import keras.backend as K

from keras.layers import BatchNormalization, Dense, Dot, Lambda
from keras.regularizers import l2


def mean_squared_error_matrix(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[-2, -1])


keras.losses.mean_squared_error_matrix = mean_squared_error_matrix


class Restorer(object):
    """
    Restorer network
    """
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def restore(self, premise_contextualized, hypothesis_contextualized):
        premise_restored = self.__call__(premise_contextualized)
        hypothesis_restored = self.__call__(hypothesis_contextualized)

        dot_restored = Dot(axes=(2, 2))([premise_restored, hypothesis_restored])

        return dot_restored

    @classmethod
    def from_config(cls, config):
        # TODO(tomwesolowski): Parameterize by config.
        layers = []
        layers.append(Dense(units=600,
                            activation='relu',
                            kernel_regularizer=l2()))
        layers.append(BatchNormalization())
        layers.append(Dense(units=300,
                            name='final_transform',
                            kernel_regularizer=l2()))
        return cls(layers=layers)
