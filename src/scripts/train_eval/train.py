#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains a simple baseline on SNLI
Run like: python src/scripts/train_esim.py cc840 results/test_run
"""


import logging
import matplotlib

from src.configs.configs import baseline_configs
from src.models import build_model
from src.util.vegab import main, MetaSaver, AutomaticNamer
from src.util.training_loop import baseline_training_loop
from src.scripts.train_eval.utils import build_data_and_streams, compute_metrics

matplotlib.use('Agg')
logger = logging.getLogger(__name__)


def train_model(config, save_path):
    data_and_streams = build_data_and_streams(config)
    model = build_model(config, data_and_streams["data"])

    # Call training loop
    baseline_training_loop(model=model,
                           data_and_streams=data_and_streams,
                           save_path=save_path,
                           early_stopping=config["early_stopping"],
                           n_epochs=config["n_epochs"],
                           config=config)

    metrics = compute_metrics(config, model, data_and_streams, eval_streams=["dev", "test"])

    for stream_name, stream_metrics in metrics.items():
        loss, accuracy = stream_metrics
        logger.info('{} loss / accuracy = {:.4f} / {:4f}'.format(stream_name, loss, accuracy))

if __name__ == "__main__":
    main(baseline_configs, train_model,
         plugins=[MetaSaver(), AutomaticNamer(namer="timestamp_namer")])
