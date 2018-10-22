#!/usr/bin/env python

import logging

from common.experiment import Experiment
from common.paths import *
from common.utils import load_config, prepare_environment

from training.trainer import Trainer
from utils.vegab import MetaSaver, run_with_redirection

logger = logging.getLogger(__name__)


def train_from_config(config, serialization_dir):
    rng = prepare_environment(config)

    experiment = Experiment.from_config(config, rng=rng)
    experiment.print_configs()

    trainer = Trainer.from_params(
        config=experiment.config['trainer'],
        model=experiment.model,
        # TODO(tomwesolowski): Read it from config?
        train_stream=experiment.streams.train,
        dev_stream=experiment.streams.dev,
        serialization_dir=serialization_dir
    )

    trainer.train()


def train_from_parser(parser):
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--savedir", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    serialization_dir = os.path.join(DATA_DIR, args.savedir)

    if not os.path.exists(serialization_dir):
        os.makedirs(serialization_dir)

    def call_training_func(plugins):
        for p in plugins:
            p.on_before_call(config, serialization_dir)

        # TODO(tomwesolowski): Add config overriding from params.
        train_from_config(config=config,
                          serialization_dir=serialization_dir)

        for p in plugins:
            p.on_after_call(config, args.savedir)

    run_with_redirection(
        os.path.join(serialization_dir, 'stdout.txt'),
        os.path.join(serialization_dir, 'stderr.txt'),
        call_training_func)([MetaSaver()])