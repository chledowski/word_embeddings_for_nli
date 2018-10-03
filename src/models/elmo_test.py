#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains a simple baseline on SNLI
Run like: python src/scripts/train_esim.py cc840 results/test_run
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
from src.models.elmo import ElmoEmbeddings
from src.util.vegab import main, MetaSaver, AutomaticNamer
from src.util.training_loop import baseline_training_loop
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics

matplotlib.use('Agg')
logger = logging.getLogger(__name__)


def get_latest_version_number():
    prefix_to_look_for = 'elmo_our_embeddings_'
    paths_and_times = []
    dirpath = os.path.join(DATA_DIR, 'elmo/%s*' % prefix_to_look_for)
    for name in glob.glob(dirpath):
        path = Path(name)
        paths_and_times.append((path.stat().st_mtime, name))

    paths_and_times = sorted(paths_and_times, key=lambda x: -x[0])
    if paths_and_times:
        _, paths_sorted = zip(*paths_and_times)
        latest_path = paths_sorted[0]
        latest_number = int(re.findall('%s(\d+)' % prefix_to_look_for, latest_path)[0])
    else:
        latest_number = -1
    return latest_number


def padarray(array, expected_shape):
    pad_width = []
    for dim, expected_dim in zip(array.shape, expected_shape):
        pad_width.append((0, expected_dim - dim))
    return np.pad(array, tuple(pad_width), mode='constant')


def build_elmo_extractor(config, data):
    elmo_embed = ElmoEmbeddings(config, all_stages=["pre_lstm"])

    premise = Input(shape=(None,), dtype='int32', name='premise')
    premise_mask_input = Input(shape=(None,), dtype='int32', name='premise_mask_input')
    premise_elmo_input = Input(shape=(None,), dtype='int32', name='premise_elmo_input')

    premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)
    # [-1, Psize, 1]
    premise_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask)

    elmo_p = elmo_embed([premise_elmo_input, premise_mask_exp], stage="pre_lstm", name='p')
    embeddings_p = elmo_embed.get_embeddings(stage='pre_lstm', name='p')

    model_input = [premise, premise_mask_input, premise_elmo_input]

    identity = Lambda(lambda x: x)

    model_output = [identity(premise_elmo_input), embeddings_p, elmo_p]

    model = Model(inputs=model_input, outputs=model_output)

    print(model.summary())

    return model


def test_elmo():
    config = baseline_configs['esim']

    num_batches = 2
    batch_size = 32

    config['use_elmo'] = True
    config['use_multiprocessing'] = False
    config['batch_sizes']['snli']['train'] = 32
    config['seed'] = 1234

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])
    model = build_elmo_extractor(config, datasets[config["dataset"]])

    def modified_stream(stream):
        for input, output in stream:
            x1, x1_mask, x2, x2_mask, x1_elmo, x2_elmo = input
            yield [x1, x1_mask, x1_elmo], output

    inputs = []
    embeddings = []
    weighted_embeddings = []

    for step in range(num_batches):
        output = model.predict_generator(
            generator=modified_stream(streams["snli"]["train"]),
            steps=1,
            use_multiprocessing=False,
            verbose=True
        )
        inputs.append(output[0])  # [batch_size, sentence_len]
        embeddings.append(output[1]) # [batch_size, 3, sentence_len, elmo_dim]
        weighted_embeddings.append(output[2]) # [batch_size, sentence_len, elmo_dim]
        print("input shape:", inputs[-1].shape)
        print("embeddings shape:", embeddings[-1].shape)
        print("weighted_embeddings shape:", weighted_embeddings[-1].shape)

    max_sentence_len = 0
    elmo_dim = 0

    for batch in inputs:
        for sentence in batch:
            max_sentence_len = max(max_sentence_len, sentence.shape[0])

    for layers in embeddings:
        for layer in layers:
            for sentence in layer:
                elmo_dim = sentence[0].shape[0]

    print("max_sentence_len", max_sentence_len)
    print("elmo_dim", elmo_dim)

    padded_inputs = np.concatenate(
        [padarray(input, [batch_size, max_sentence_len]) for input in inputs]
    )
    padded_embeddings = np.concatenate(
        [padarray(emb, [batch_size, 3, max_sentence_len, elmo_dim]) for emb in embeddings]
    )
    padded_weighted_embeddings = np.concatenate(
        [padarray(emb, [batch_size, max_sentence_len, elmo_dim]) for emb in weighted_embeddings]
    )

    # saving to file
    latest_version_number = get_latest_version_number()

    inputs_path = os.path.join(
        DATA_DIR, 'elmo/elmo_our_inputs_%d.npy' % (latest_version_number + 1))
    embeddings_path = os.path.join(
        DATA_DIR, 'elmo/elmo_our_embeddings_%d.npy' % (latest_version_number + 1))
    weighted_embeddings_path = os.path.join(
        DATA_DIR, 'elmo/elmo_our_weighted_embeddings_%d.npy' % (latest_version_number + 1))

    for path, array in zip(
        [inputs_path, embeddings_path, weighted_embeddings_path],
        [padded_inputs, padded_embeddings, padded_weighted_embeddings]
    ):
        np.save(path, array)
        print("Saved to: %s" % path)


if __name__ == "__main__":
    test_elmo()
