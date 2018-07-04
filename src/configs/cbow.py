# -*- coding: utf-8 -*-
"""
Configs used in the project
"""

from src.util.vegab import ConfigRegistry

baseline_configs = ConfigRegistry()

baseline_configs.set_root_config({
    "dataset": {"name": "snli", "n_labels": 3},
    "model": 'cbow',
    "n_layers": 3,
    "batch_size": 512,
    "early_stopping": True,
    "n_epochs": 40,
    "batch_normalization": True,
    "dropout": 0.1,
    "optimizer": 'rmsprop',
    "learning_rate": 10000,
    "lr_schedule": "[[1000, 0.001]]",
    "lr_schedule_type": "list_of_lists",
    "intersection_of_embedding_dicts": False,
    "D": 0,
    "whitening": False,
    "centering": False,
    "normalize": True,
    "embedding_dim": 300,
    "embedding_name": "cos",
    "train_on_fraction": 1
})


c = baseline_configs["root"]
c["embedding"] = {"name": "random_uniform", "dim": 300}
baseline_configs["uniform"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "common-crawl-42B", "dim": 300}
baseline_configs["cc42"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "common-crawl-840B", "dim": 300}
baseline_configs["cc840"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "conceptnet", "dim": 300}
baseline_configs["cnet"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "text8_0", "dim": 300}
baseline_configs["text0"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "text8_10", "dim": 300}
baseline_configs["text10"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "text8_30", "dim": 300}
baseline_configs["text30"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "pdc", "dim": 300}
baseline_configs["pdc"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "hdc", "dim": 300}
baseline_configs["hdc"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "lexvec", "dim": 300}
baseline_configs["lexvec"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "sg_googlenews", "dim": 300}
baseline_configs["sg"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "common-crawl-840B_retro_1", "dim": 300}
baseline_configs["840retro1"] = c