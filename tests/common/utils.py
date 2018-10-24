import numpy
import os

from keras.layers import InputLayer
from keras.models import Model

from common.paths import *


class RandomState(object):
    def shuffle(self, x):
        return x


def get_all_layer_outputs(model, stream):

    outputs = {}
    for layer in model.layers:
        if not type(layer) is InputLayer:
            for i in range(len(layer._inbound_nodes)):
                outputs['%s_%d' % (layer.name, i)] = layer.get_output_at(i)

    model = Model(inputs=model.inputs, outputs=list(outputs.values()))

    return (list(outputs.keys()),
            model.predict_generator(
                generator=stream,
                steps=1,
                use_multiprocessing=False,
                verbose=True
            )
    )


def save_outputs(dirpath, names, outputs):
    dirpath = os.path.join(DATA_DIR, dirpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    for name, output in zip(names, outputs):
        path = os.path.join(dirpath, '%s.npy' % name)
        numpy.save(path, output)
        print("Saved %s to: %s" % (name, path))