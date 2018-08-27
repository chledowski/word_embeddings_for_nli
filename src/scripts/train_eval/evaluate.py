#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains a simple baseline on SNLI
Run like: python src/scripts/train_esim.py cc840 results/test_run
"""

import argparse
import json
import os
import pprint as pp

from src.models import build_model
from src.util import modified_stream, evaluate_wv, load_embedding_from_h5
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics

from numpy.random import seed
from numpy.random import RandomState
from tensorflow import set_random_seed


def eval_model(batch_size=None):
    results_dict = {}

    with open(os.path.join('results', args.model_name, 'config.json'), 'r') as f:
        config = json.load(f)

    if batch_size:
        config['batch_sizes']['snli']['test'] = batch_size
        config['batch_sizes']['breaking']['test'] = batch_size

    pp.pprint(config)

    seed(config["seed"])
    set_random_seed(config["seed"])
    rng = RandomState(config["seed"])

    # To evaluate on more streams, add them here
    # config["batch_size"][stream] = ...

    datasets_to_load = list(set(["snli",
                                 "breaking",
                                 config["dataset"]]))
    datasets, streams = build_data_and_streams(config, rng, datasets_to_load=datasets_to_load)
    model = build_model(config, datasets[config["dataset"]])

    # Restore the best model found during validation
    model.load_weights(os.path.join('results', args.model_name, "best_model.h5"))

    metrics = compute_metrics(config, model, datasets, streams,
                              eval_streams=["test"])

    results_dict['accuracies'] = {}
    results_dict['losses'] = {}
    for stream_name, stream_metrics in metrics.items():
        loss, accuracy = stream_metrics
        print('{} loss / accuracy = {:.4f} / {:4f}'.format(stream_name, loss, accuracy))
        results_dict['accuracies'][stream_name] = accuracy
        results_dict['losses'][stream_name] = loss

    output_dir_path = 'results/%s_bs%d' % (args.model_name, batch_size)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    with open(os.path.join(output_dir_path, 'retrofitting_results.json'), 'w') as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)

    args = parser.parse_args()

    batch_sizes_to_check = [307, 2, 4]
    for batch_size in batch_sizes_to_check:
        eval_model(batch_size)
