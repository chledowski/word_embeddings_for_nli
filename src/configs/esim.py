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
c["embedding"] = {"name": "wiki-6Bwordnet-synonyms+", "dim": 300}
baseline_configs["wiki_s_plus"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_eiu_q", "dim": 300}
baseline_configs["wiki_s_plus_eiu_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_adam_q", "dim": 300}
baseline_configs["wiki_s_plus_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_eiu_12__pca_yes_q", "dim": 300}
baseline_configs["wiki_s_plus_eiu_12_pca_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_eiu_2_10_pca_yes_q", "dim": 300}
baseline_configs["wiki_s_plus_eiu_2_10_pca_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wikitest", "dim": 300}
baseline_configs["wikitest"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_eiu_2_20_pca_yes_q", "dim": 300}
baseline_configs["wiki_s_plus_eiu_2_20_pca_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_eiu_12_0scale_normalize_pca_q", "dim": 300}
baseline_configs["wiki-6B_wordnet-synonyms+_eiu_12_0scale_normalize_pca_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_eiu_pca_yes_q", "dim": 300}
baseline_configs["wiki_s_plus_eiu_pca_q"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B", "dim": 300}
baseline_configs["wiki"] = c

c = baseline_configs["root"]
c["embedding"] = {"name": "wiki-6B_wordnet-synonyms+_original", "dim": 300}
baseline_configs["wiki_s_plus_original"] = c

