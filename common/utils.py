import json
import logging
import os

from numpy.random import seed
from numpy.random import RandomState

import warnings
with warnings.catch_warnings():
    from tensorflow import set_random_seed


def prepare_environment(config):
    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    return rng


def load_config(model_path):
    # TODO(tomwesolowski): Add prepending DATA_DIR to all paths.
    with open(os.path.join(model_path, 'config.json'), 'r') as f:
        config = json.load(f)
    return config