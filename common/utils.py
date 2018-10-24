import json
import warnings

import numpy
from keras.layers import InputLayer
from keras.models import Model
from numpy.random import RandomState
from numpy.random import seed

from common.paths import *

with warnings.catch_warnings():
    from tensorflow import set_random_seed


def prepare_environment(config):
    """
    Sets all seeds to the one provided in config.
    """
    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    return rng


def load_config(config_path):
    """
    Loads config from file.
    """
    # TODO(tomwesolowski): Add prepending DATA_DIR to all paths.
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_all_layer_outputs(model, stream):
    """
    Gets outputs of all intermediate layers of the model fed
    with one batch from stream.
    """
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