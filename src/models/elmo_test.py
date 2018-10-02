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


def build_elmo_model(config, data):

    elmo_embed = ElmoEmbeddings(config)

    premise = Input(shape=(None,), dtype='int32', name='premise')
    premise_mask_input = Input(shape=(None,), dtype='int32', name='premise_mask_input')
    hypothesis = Input(shape=(None,), dtype='int32', name='hypothesis')
    hypothesis_mask_input = Input(shape=(None,), dtype='int32', name='hypothesis_mask_input')

    premise_elmo_input = Input(shape=(None,), dtype='int32', name='premise_elmo_input')
    hypothesis_elmo_input = Input(shape=(None,), dtype='int32',
                                  name='hypothesis_elmo_input')

    premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)
    hypothesis_mask = Lambda(lambda x: K.cast(x, 'float32'))(hypothesis_mask_input)

    # [-1, Psize, 1]
    premise_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(premise_mask)
    # [-1, Hsize, 1]
    hypothesis_mask_exp = Lambda(lambda x: K.expand_dims(x, axis=-1))(hypothesis_mask)

    elmo_p = elmo_embed([premise_elmo_input, premise_mask_exp], stage="pre_lstm", name='p')
    elmo_h = elmo_embed([hypothesis_elmo_input, hypothesis_mask_exp], stage="pre_lstm", name='h')

    embeddings_p = elmo_embed.get_embeddings(stage='pre_lstm', name='p')
    embeddings_h = elmo_embed.get_embeddings(stage='pre_lstm', name='h')

    model_input = [premise, premise_mask_input, hypothesis, hypothesis_mask_input,
                   premise_elmo_input, hypothesis_elmo_input]

    identity = Lambda(lambda x: x)

    model_output = [identity(premise_elmo_input), identity(hypothesis_elmo_input),
                    embeddings_p, embeddings_h]

    model = Model(inputs=model_input,
                  outputs=model_output)

    print(model.summary())

    return model


def test_elmo():
    config = baseline_configs['bilstm']

    # config['dump_elmo'] = True
    # config['dump_lemma'] = True
    config['use_elmo'] = True
    config['use_multiprocessing'] = False
    config['batch_sizes']['snli']['train'] = 32
    steps = 8
    config['seed'] = 1234

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=[config["dataset"]])
    model = build_elmo_model(config, datasets[config["dataset"]])

    def modified_stream(stream):
        for input, output in stream:
            x1, x1_mask, x2, x2_mask, x1_elmo, x2_elmo = input
            yield [x1, x1_mask, x2, x2_mask, x1_elmo, x2_elmo], output

    output = model.predict_generator(
        generator=modified_stream(streams["snli"]["train"]),
        steps=steps,
        use_multiprocessing=False,
        verbose=True
    )

    input = output[:2]
    output = output[2:]

    paths_and_times = []
    dirpath = os.path.join(DATA_DIR, 'elmo/elmo_our_out_*')
    for name in glob.glob(dirpath):
        path = Path(name)
        paths_and_times.append((path.stat().st_mtime, name))

    paths_and_times = sorted(paths_and_times, key=lambda x: -x[0])
    if paths_and_times:
        _, paths_sorted = zip(*paths_and_times)
        latest_path = paths_sorted[0]
        latest_number = int(re.findall('elmo_our_out_(\d+)', latest_path)[0])
    else:
        latest_number = 0

    out_path = os.path.join(DATA_DIR, 'elmo/elmo_our_out_%d.npy' % (latest_number+1))

    np.save(os.path.join(DATA_DIR, 'elmo/elmo_our_inp.npy'), input)
    np.save(out_path, output)

    print("Saved to: %s" % out_path)


if __name__ == "__main__":
    test_elmo()
