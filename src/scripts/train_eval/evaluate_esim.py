#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains a simple baseline on SNLI
Run like: python src/scripts/train_esim.py cc840 results/test_run
"""

import argparse
import json
import os

from keras import optimizers

from src import DATA_DIR
from src.models import build_model
from src.util import modified_stream, evaluate_wv, load_embedding_from_h5
from src.util.data import SNLIData


def eval_model():

    if os.path.exists('results/retrofitting_results.json'):
        with open('results/retrofitting_results.json', 'r') as f:
            results_dict = json.load(f)
    else:
        results_dict = {}

    with open(os.path.join('results', args.model_name, 'config.json'), 'r') as f:
        config = json.load(f)

    if config["dataset"]["name"] == "snli":
        data = SNLIData(os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"]["name"] == "mnli":
        data = SNLIData(os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"]["name"])

    breaking_data = SNLIData(os.path.join(DATA_DIR, "snli"), "breaking")

    train = data.get_stream("train", batch_size=config["batch_size"])
    dev = data.get_stream("dev", batch_size=config["batch_size"])
    test = data.get_stream("test", batch_size=config["batch_size"])
    test_breaking = breaking_data.get_stream("test", batch_size=config["batch_size"])

    stream_train = modified_stream(train)()
    stream_dev = modified_stream(dev)()
    stream_test = modified_stream(test)()
    stream_test_breaking = modified_stream(test_breaking)()

    # Load model
    model, embedding_matrix, statistics = build_model(config)

    model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    # Restore the best model found during validation
    model.load_weights(os.path.join('results', args.model_name, "best_model.h5"))

    accuracies = {}
    losses = {}
    train_metrics = model.evaluate_generator(stream_train, 549364 / config["batch_size"])
    print('Train loss / train accuracy = {:.4f} / {:4f}'.format(train_metrics[0], train_metrics[1]))
    accuracies['train'] = train_metrics[0]
    losses['train'] = train_metrics[1]
    dev_metrics = model.evaluate_generator(stream_dev, 9842 / config["batch_size"])
    print('Dev loss / dev accuracy = {:.4f} / {:4f}'.format(dev_metrics[0], dev_metrics[1]))
    accuracies['dev'] = dev_metrics[0]
    losses['dev'] = dev_metrics[1]
    test_metrics = model.evaluate_generator(stream_test, 9824 / config["batch_size"])
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(test_metrics[0], test_metrics[1]))
    accuracies['test'] = test_metrics[0]
    losses['test'] = test_metrics[1]
    test_breaking_metrics = model.evaluate_generator(stream_test_breaking, 8193 / config["batch_size"])
    print('Breaking loss / breaking accuracy = {:.4f} / {:.4f}'.format(test_breaking_metrics[0], test_breaking_metrics[1]))
    accuracies['breaking'] = test_breaking_metrics[1]
    losses['breaking'] = test_breaking_metrics[0]

    results_dict[args.model_name]['accuracies'] = accuracies
    results_dict[args.model_name]['losses'] = losses

    _, _, wv = load_embedding_from_h5(args.model_name)
    results_dict[args.model_name]['backup'] = evaluate_wv(wv, simlex_only=False)

    with open('results/retrofitting_results.json', 'w') as f:
        json.dump(results_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default='wiki', type=str)

    args = parser.parse_args()
    eval_model()
