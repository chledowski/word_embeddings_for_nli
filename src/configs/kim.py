# -*- coding: utf-8 -*-
"""
Configs used in the project
"""
import os

from src import DATA_DIR
from src.util.vegab import ConfigRegistry

baseline_configs = ConfigRegistry()

baseline_configs.set_root_config({
    "model": "kim",
    "reload_": False,
    "dim_word": 300,
    "dim": 300,
    "patience": 7,
    "n_words": 110497,
    "n_words_lemma": 100360,
    "decay_c": 0.,
    "clip_c": 10.,
    "lrate": 0.0004,
    "optimizer": 'adam',
    "maxlen": 100,
    "batch_size": 32,
    "valid_batch_size": 32,
    "dispFreq": 100,
    "validFreq": int(549367/32+1),
    "saveFreq": int(549367/32+1),
    "use_dropout": True,
    "verbose": False,
    "datasets": [os.path.join(DATA_DIR, 'kim_data/premise_snli_1.0_train_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_snli_1.0_train_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/premise_snli_1.0_train_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_snli_1.0_train_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/label_snli_1.0_train.txt')],
    "valid_datasets": [os.path.join(DATA_DIR, 'kim_data/premise_snli_1.0_dev_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_snli_1.0_dev_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/premise_snli_1.0_dev_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_snli_1.0_dev_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/label_snli_1.0_dev.txt')],
    "test_datasets": [os.path.join(DATA_DIR, 'kim_data/premise_snli_1.0_test_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_snli_1.0_test_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/premise_snli_1.0_test_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_snli_1.0_test_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/label_snli_1.0_test.txt')],
    "breaking_datasets": [os.path.join(DATA_DIR, 'kim_data/premise_test_breaking_nli_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_test_breaking_nli_token.txt'),
                        os.path.join(DATA_DIR, 'kim_data/premise_test_breaking_nli_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/hypothesis_test_breaking_nli_lemma.txt'),
                        os.path.join(DATA_DIR, 'kim_data/label_test_breaking_nli.txt')],
    "dictionary": [os.path.join(DATA_DIR, 'kim_data/vocab_cased.pkl'),
                   os.path.join(DATA_DIR, 'kim_data/vocab_cased_lemma.pkl')],
    "kb_dicts": [os.path.join(DATA_DIR, 'kim_data/pair_features.pkl')],
    "embedding": '',
    "dim_kb": 5,
    "kb_inference": True,
    "kb_composition": False,
    "train_embeddings": True,
    "attention_lambda": 0
})