from src.util.vegab import ConfigRegistry

baseline_configs = ConfigRegistry()

baseline_configs['bilstm'] = {
    "dataset": "snli",
    "n_labels": 3,
    "model": 'bilstm',
    "n_layers": 3,
    "batch_sizes": {
        "train": 512,
        "dev": 512,
        "test": 512
    },
    "early_stopping": True,
    "n_epochs": 40,
    "batch_normalization": True,
    "dropout": 0.1,
    "optimizer": 'rmsprop',
    "learning_rate": 10000,
    "lr_schedule": "[[1000, 0.001]]",
    "lr_schedule_type": "list_of_lists",
    "l2_strength": 1e-5,
    "intersection_of_embedding_dicts": False,
    "D": 0,
    "whitening": False,
    "centering": False,
    "normalize": False,
    "norm_weight": False,
    "embedding_dim": 300,
    "embedding_name": "cos",
    "train_embeddings": True,
    "train_on_fraction": 1.0
}

baseline_configs['cbow'] = {
    "dataset": "snli",
    "n_labels": 3,
    "model": 'cbow',
    "n_layers": 3,
    "batch_sizes": {
        "train": 32,
        "dev": 32,
        "test": 32
    },
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
    "train_on_fraction": 1.0,

}

baseline_configs['esim'] = {
    'D': 0,
    'attention_lambda': 0.0,
    'batch_sizes': {'dev': 128, 'test': 2, 'train': 32},
    'centering': False,
    'clip_gradient_norm': 10.0,
    'dataset': 'snli',
    'dropout': 0.5,
    'early_stopping': True,
    'embedding_dim': 300,
    'embedding_name': 'gcc840',
    'i_lambda': 1.0,
    'intersection_of_embedding_dicts': False,
    'learning_rate': 0.0004,
    'lr_schedule': '[[1000, 0.0004]]',
    'lr_schedule_type': 'reduce_on_plateau',
    'model': 'esim',
    'n_epochs': 60,
    'n_labels': 3,
    'n_layers': 3,
    'norm_weight': True,
    'normalize': False,
    'optimizer': 'adam',
    'pair_features_pkl_path': 'pair_features.pkl',
    'pair_features_txt_path': 'kim_data/pair_features.txt',
    'seed': 1,
    'sentence_max_length': 100,
    'train_embeddings': False,
    'train_on_fraction': 1.0,
    'useitrick': 0,
    'whitening': False
}