#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains a simple baseline on SNLI
Run like: python src/scripts/train_esim.py cc840 results/test_run
"""

import argparse
import json
import numpy as np
import os

from keras import optimizers

from src import DATA_DIR
from src.models import build_model
from src.util import modified_stream, evaluate_wv, load_embedding_from_h5
from src.util.data import SNLIData


def predict():
    results_dict = {}

    # TODO(tomwesolowski): Make it work again after config reorganization.

    with open(os.path.join('results', args.model_name, 'config.json'), 'r') as f:
        config = json.load(f)

    if config["dataset"] == "snli":
        data = SNLIData(os.path.join(DATA_DIR, "snli"), "snli")
    elif config["dataset"] == "mnli":
        data = SNLIData(os.path.join(DATA_DIR, "mnli"), "mnli")
    else:
        raise NotImplementedError('Dataset not supported: ' + config["dataset"])

    breaking_data = SNLIData(os.path.join(DATA_DIR, "snli"), "breaking")
    test_breaking = breaking_data.get_stream("breaking", batch_size=config["batch_size"])
    stream_test_breaking = modified_stream(test_breaking)()

    # Load model
    model, embedding_matrix, statistics = build_model(config, data)

    model.compile(optimizer=optimizers.RMSprop(lr=config["learning_rate"]),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    # Restore the best model found during validation
    model.load_weights(os.path.join('results', args.model_name, "best_model.h5"))

    test_breaking_predictions = model.predict_generator(stream_test_breaking, 8193 / config["batch_size"])
    np.save('results/%s/test_breaking_predictions.npy' % args.model_name, test_breaking_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default='wiki', type=str)

    args = parser.parse_args()
    predict()
