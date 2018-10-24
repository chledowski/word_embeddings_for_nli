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
from keras.layers import InputLayer
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
        np.save(path, output)
        print("Saved %s to: %s" % (name, path))


def save_all():
    config = prepare_config('esim')

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])
    esim_model = build_model(config, datasets[config["dataset"]])

    names, outputs = get_all_layer_outputs(model=esim_model,
                                           stream=streams["snli"]['train'])

    save_outputs('dumps/old_esim', names, outputs)


def save_model_attention():
    config = prepare_config('esim')

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

    for stream_name in ['train', 'dev', 'test']:
        premises, hypothesis, attention = [], [], []

        for step in range(config['num_batches']):
            output = model.predict_generator(
                generator=streams["snli"][stream_name],
                steps=1,
                use_multiprocessing=False,
                verbose=True
            )
            premises.append(output[0])  # [batch_size, sentence_len]
            hypothesis.append(output[1])  # [batch_size, sentence_len]
            attention.append(output[2])  # [batch_size, p_size, h_size]

        premises_path = os.path.join(
            DATA_DIR, 'dumps/attention_%s_premises.npy' % stream_name)
        hypothesis_path = os.path.join(
            DATA_DIR, 'dumps/attention_%s_hypothesis.npy' % stream_name)
        attention_path = os.path.join(
            DATA_DIR, 'dumps/attention_%s.npy' % stream_name)

        for path, array in zip(
                [premises_path, hypothesis_path, attention_path],
                [premises, hypothesis, attention]
        ):
            np.save(path, array)
            print("Saved to: %s" % path)


def save_knowledge_matrix():
    config = prepare_config('esim-kim')

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])
    esim_model = build_model(config, datasets[config["dataset"]])

    model = Model(inputs=esim_model.inputs, outputs=[
        esim_model.get_layer('premise_out').output,
        esim_model.get_layer('hypothesis_out').output,
        esim_model.get_layer('kbph_out').output,
        esim_model.get_layer('kbhp_out').output,
    ])

    for stream_name in ['train', 'dev', 'test']:
        premises, hypothesis, kbph, kbhp = [], [], [], []

        for step in range(config['num_batches']):
            output = model.predict_generator(
                generator=streams["snli"][stream_name],
                steps=1,
                use_multiprocessing=False,
                verbose=True
            )
            premises.append(output[0])  # [batch_size, sentence_len]
            hypothesis.append(output[1])  # [batch_size, sentence_len]
            kbph.append(output[2])  # [batch_size, p_size, h_size]
            kbhp.append(output[3])  # [batch_size, p_size, h_size]

        premises_path = os.path.join(
            DATA_DIR, 'dumps/knowledge_matrix_%s_premises.npy' % stream_name)
        hypothesis_path = os.path.join(
            DATA_DIR, 'dumps/knowledge_matrix_%s_hypothesis.npy' % stream_name)
        kbph_path = os.path.join(
            DATA_DIR, 'dumps/knowledge_matrix_%s_kbph.npy' % stream_name)
        kbhp_path = os.path.join(
            DATA_DIR, 'dumps/knowledge_matrix_%s_kbhp.npy' % stream_name)

        for path, array in zip(
                [premises_path, hypothesis_path, kbph_path, kbhp_path],
                [premises, hypothesis, kbph, kbhp]
        ):
            np.save(path, array)
            print("Saved to: %s" % path)


def prepare_config(name):
    config = dict(baseline_configs[name])

    num_batches = 1
    batch_size = 5

    config['use_multiprocessing'] = False
    config['num_batches'] = num_batches
    config['batch_size'] = batch_size
    config['batch_sizes']['snli']['train'] = batch_size
    config['batch_sizes']['snli']['dev'] = batch_size
    config['batch_sizes']['snli']['test'] = batch_size
    config['seed'] = 12345
    return config


def test_esim():
    save_all()
    # save_knowledge_matrix()
    # save_model_attention()


if __name__ == "__main__":
    test_esim()
