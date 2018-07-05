# -*- coding: utf-8 -*-
"""
Configs used in the project
"""

from src.util.vegab import ConfigRegistry

baseline_configs = ConfigRegistry()

baseline_configs.set_root_config({
    "activation": 'tanh',
    "dataset": {"name": "snli", "n_labels": 3},
    "model": 'esim',
    "n_layers": 3,
    "batch_normalization": False,
    "batch_size": 512,
    "early_stopping": True,
    "n_epochs": 40,
    "dropout": 0.1,
    "optimizer": 'rmsprop',
    "learning_rate": 10000,
    "lr_schedule": "[[1000, 0.001]]",
    "lr_schedule_type": "list_of_lists",
    "intersection_of_embedding_dicts": False,
    "D": 0,
    "whitening": False,
    "centering": False,
    "normalize": False,
    "embedding_dim": 300,
    "embedding_name": "cos",
    "train_on_fraction": 1
})