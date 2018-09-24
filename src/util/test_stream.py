#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import matplotlib

from src.configs.configs import baseline_configs
from src.models import build_model
from src.util.vegab import main, MetaSaver, AutomaticNamer
from src.util.training_loop import baseline_training_loop
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics

from numpy.random import seed
from numpy.random import RandomState
from tensorflow import set_random_seed


matplotlib.use('Agg')
logger = logging.getLogger(__name__)


def test_streams():
    config = baseline_configs['bilstm']

    config['dump_elmo'] = True
    config['dump_lemma'] = True
    config['use_elmo'] = True
    config['batch_sizes']['snli']['train'] = 1

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])

    next(streams['snli']['train'])


if __name__ == "__main__":
    test_streams()
