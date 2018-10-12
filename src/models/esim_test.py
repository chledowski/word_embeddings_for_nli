#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ELMO normalization & weighting test
Dumps elmo embeddings to compare them with bilm-tf-elmo embeddings
in notebook (notebooks/elmo_test)
"""

import glob
import keras.backend as K
import logging
import matplotlib
import numpy as np
import os
import re

from keras.layers import Concatenate, Dense, Input, Dropout, TimeDistributed, Bidirectional, Lambda
from keras.models import Model
from numpy.random import seed
from numpy.random import RandomState
from pathlib import Path
from tensorflow import set_random_seed

from src import DATA_DIR
from src.configs.configs import baseline_configs
from src.models import build_model
from src.models.utils import compute_mean_and_variance, normalize_layer, padarray, softmax
from src.util.vegab import main, MetaSaver, AutomaticNamer
from src.util.training_loop import baseline_training_loop
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics

matplotlib.use('Agg')
logger = logging.getLogger(__name__)


def build_model_and_stream(config):
    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])
    esim_model = build_model(config, datasets[config["dataset"]])

    model = Model(inputs=esim_model.inputs, outputs=[
        esim_model.get_layer('premise_out').output,
        esim_model.get_layer('hypothesis_out').output,
        esim_model.get_layer('attention_matrix').output
    ])

    return model, streams["snli"]["train"]


def get_model_output(config, model, stream):
    premises = []
    hypothesis = []
    attention = []

    for step in range(config['num_batches']):
        output = model.predict_generator(
            generator=stream,
            steps=1,
            use_multiprocessing=False,
            verbose=True
        )
        premises.append(output[0])  # [batch_size, sentence_len]
        hypothesis.append(output[1])  # [batch_size, sentence_len]
        attention.append(output[2])  # [batch_size, p_size, h_size]
        print("premises shape:", premises[-1].shape)
        print("hypothesis shape:", hypothesis[-1].shape)
        print("attention shape:", attention[-1].shape)

    return premises, hypothesis, attention


def save_to_file(premises,  hypothesis, attention):
    premises_path = os.path.join(
        DATA_DIR, 'dumps/attention_premises.npy')
    hypothesis_path = os.path.join(
        DATA_DIR, 'dumps/attention_hypothesis.npy')
    attention_path = os.path.join(
        DATA_DIR, 'dumps/attention.npy')

    for path, array in zip(
            [premises_path, hypothesis_path, attention_path],
            [premises,  hypothesis, attention]
    ):
        np.save(path, array)
        print("Saved to: %s" % path)


def prepare_config():
    config = dict(baseline_configs['esim'])

    num_batches = 1
    batch_size = 5

    config['use_multiprocessing'] = False
    config['num_batches'] = num_batches
    config['batch_size'] = batch_size
    config['batch_sizes']['snli']['train'] = batch_size
    config['seed'] = 12345
    return config


def test_esim():
    config = prepare_config()
    model, stream = build_model_and_stream(config)
    save_to_file(*get_model_output(config, model, stream))


if __name__ == "__main__":
    test_esim()
