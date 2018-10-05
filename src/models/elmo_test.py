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
from src.models.elmo import ElmoEmbeddings, WeightElmoEmbeddings
from src.models.utils import compute_mean_and_variance, normalize_layer, padarray, softmax
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


def build_elmo_extractor(config, data):
    elmo_embed = ElmoEmbeddings(config)
    elmo_weight = WeightElmoEmbeddings(config, all_stages=["pre_lstm"])

    premise = Input(shape=(None,), dtype='int32', name='premise')
    premise_mask_input = Input(shape=(None,), dtype='int32', name='premise_mask_input')
    premise_elmo_input = Input(shape=(None,), dtype='int32', name='premise_elmo_input')

    premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)
    # [-1, Psize, 1]
    premise_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask)

    elmo_p = elmo_embed([premise_elmo_input, premise_mask_exp])
    weighted_elmo_p = elmo_weight(elmo_p, stage="pre_lstm")

    model_input = [premise, premise_mask_input, premise_elmo_input]

    identity = Lambda(lambda x: x)

    model_output = [identity(premise_elmo_input), identity(premise_mask_input), elmo_p, weighted_elmo_p]

    model = Model(inputs=model_input, outputs=model_output)

    print(model.summary())

    return model


def build_model_and_stream(config):
    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])
    model = build_elmo_extractor(config, datasets[config["dataset"]])

    def modified_stream(stream):
        for input, output in stream:
            x1, x1_mask, x2, x2_mask, x1_elmo, x2_elmo = input
            yield [x1, x1_mask, x1_elmo], output

    return model, modified_stream(streams["snli"]["train"])


def get_model_output(config, model, stream):
    inputs = []
    masks = []
    embeddings = []
    weighted_embeddings = []

    for step in range(config['num_batches']):
        output = model.predict_generator(
            generator=stream,
            steps=1,
            use_multiprocessing=False,
            verbose=True
        )
        inputs.append(output[0])  # [batch_size, sentence_len]
        masks.append(output[1])  # [batch_size, sentence_len]
        embeddings.append(output[2])  # [batch_size, 3, sentence_len, elmo_dim]
        weighted_embeddings.append(output[3])  # [batch_size, sentence_len, elmo_dim]
        print("input shape:", inputs[-1].shape)
        print("embeddings shape:", embeddings[-1].shape)
        print("weighted_embeddings shape:", weighted_embeddings[-1].shape)

    return inputs, masks, embeddings, weighted_embeddings


def pad_outputs(config, inputs, masks, embeddings, weighted_embeddings):
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
        [padarray(input, [config['batch_size'], max_sentence_len]) for input in inputs]
    )
    padded_masks = np.concatenate(
        [padarray(mask, [config['batch_size'], max_sentence_len]) for mask in masks]
    )
    padded_embeddings = np.concatenate(
        [padarray(emb, [config['batch_size'], 3, max_sentence_len, elmo_dim]) for emb in embeddings]
    )
    padded_weighted_embeddings = np.concatenate(
        [padarray(emb, [config['batch_size'], max_sentence_len, elmo_dim]) for emb in weighted_embeddings]
    )
    padded_masks_expanded = np.expand_dims(np.expand_dims(padded_masks, axis=-1), axis=1)
    padded_masks_expanded = np.tile(padded_masks_expanded, (1, 3, 1, elmo_dim))
    return padded_inputs, padded_masks_expanded, padded_embeddings, padded_weighted_embeddings


def save_to_file(padded_inputs, padded_embeddings, padded_weighted_embeddings):
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


def prepare_config():
    config = dict(baseline_configs['esim'])

    num_batches = 2
    batch_size = 2

    config['use_elmo'] = True
    config['use_multiprocessing'] = False
    config['num_batches'] = num_batches
    config['batch_size'] = batch_size
    config['batch_sizes']['snli']['train'] = batch_size
    config['seed'] = 1234
    config['elmo_use_layer_normalization'] = True
    config['elmo_initial_embeddings_weights'] = [0.5, 0.2, 0.3]
    config['elmo_initial_gamma'] = [0.5]
    return config


def test_elmo():
    config = prepare_config()

    model, stream = build_model_and_stream(config)
    inputs, masks, embeddings, weighted_embeddings = get_model_output(config, model, stream)
    inputs, masks, embeddings, weighted_embeddings = \
        pad_outputs(config, inputs, masks, embeddings, weighted_embeddings)

    num_samples = embeddings.shape[0]
    num_elmo_layers = embeddings.shape[1]

    # normalization test

    if config['elmo_use_layer_normalization']:
        expected_embeddings = normalize_layer(
            array=embeddings,
            axis=(-2, -1),
            mask=masks,
        )
    else:
        expected_embeddings = embeddings

    if not np.allclose(embeddings, expected_embeddings, atol=1e-5):
        for sample_id in range(num_samples):
            for layer_id in range(num_elmo_layers):
                real = embeddings[sample_id][layer_id]
                expected = expected_embeddings[sample_id][layer_id]
                mask = masks[sample_id][0]
                print("Layer", layer_id, "mean/var real",
                      compute_mean_and_variance(real, mask=mask, axis=(-2, -1)))
                print("Layer", layer_id, "mean/var expected",
                      compute_mean_and_variance(expected, mask=mask, axis=(-2, -1)))
        assert False

    # weighting test

    softmax_weights = softmax(config['elmo_initial_embeddings_weights'])
    expanded_weights = np.reshape(softmax_weights, [1, 3, 1, 1])
    expected_embeddings = np.sum(expected_embeddings * expanded_weights, axis=1)
    expected_embeddings = config['elmo_initial_gamma'] * expected_embeddings

    if not np.allclose(weighted_embeddings, expected_embeddings, atol=1e-5):
        real_embeddings = np.squeeze(weighted_embeddings)
        expected_embeddings = np.squeeze(expected_embeddings)
        real_norm = np.linalg.norm(real_embeddings, axis=1)
        expected_norm = np.linalg.norm(expected_embeddings, axis=1)
        print("Norms", real_norm, expected_norm)
        print("Diff avgs", np.mean(real_embeddings - expected_embeddings, axis=1))
        print("Cos. distances",
              np.sum(real_embeddings * expected_embeddings, axis=1) / (real_norm * expected_norm))
        assert False

    if not config['elmo_use_layer_normalization'] :
        save_to_file(inputs, embeddings, weighted_embeddings)


if __name__ == "__main__":
    test_elmo()
