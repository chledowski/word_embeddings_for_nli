from keras.initializers import RandomNormal


class ScaledRandomNormal(RandomNormal):
    def __init__(self, mean=0., stddev=0.05, scale=1.0, seed=None):
        super(ScaledRandomNormal, self).__init__(mean=mean, stddev=stddev, seed=seed)
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return self.scale * super(ScaledRandomNormal, self).__call__(shape, dtype=dtype)
