# -*- coding: utf-8 -*-
"""
Configs used in the project
"""

from src.util.vegab import ConfigRegistry

baseline_configs = ConfigRegistry()

baseline_configs.set_root_config({
    "dataset": "snli",
    "n_labels": 3,
    "model": 'cbow',
    "n_layers": 3,
    "batch_size": 32,
    'val_batch_size': 32,
    "early_stopping": True,
    "n_epochs": 40,
    "batch_normalization": False,
    "dropout": 0.1,
    "sentence_max_length": 100,
    "optimizer": 'rmsprop',
    "learning_rate": 10000,
    "lr_schedule": "[[1000, 0.001]]",
    "lr_schedule_type": "list_of_lists",
    "intersection_of_embedding_dicts": False,
    "D": 0,
    "whitening": False,
    "centering": False,
    "normalize": False,
    "norm_weight": False,
    "embedding_dim": 300,
    "embedding_name": "cos",
    "train_embeddings": False,
    "train_on_fraction": 1
})
