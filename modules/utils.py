import keras.backend as K

from keras.initializers import RandomNormal
from keras.layers import Lambda
from keras.regularizers import l1, l2


def get_regularizer(regularizer, regularizer_strength):
    if regularizer == 'l1':
        return l1(regularizer_strength)
    elif regularizer == 'l2':
        return l2(regularizer_strength)
    else:
        raise ValueError()


class MaskMultiply(object):
    def __init__(self):
        self.model = Lambda(
            lambda x: x[0] * K.expand_dims(x[1], axis=-1)
        )

    def __call__(self, inputs):
        return self.model(inputs)


class ScaledRandomNormal(RandomNormal):
    def __init__(self, mean=0., stddev=0.05, scale=1.0, seed=None):
        super(ScaledRandomNormal, self).__init__(mean=mean, stddev=stddev, seed=seed)
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return self.scale * super(ScaledRandomNormal, self).__call__(shape, dtype=dtype)
