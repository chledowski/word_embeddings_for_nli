
from src.util.vegab import ConfigRegistry

baseline_configs = ConfigRegistry()

baseline_configs['bilstm'] = {
    'model': 'bilstm',
    'batch_sizes': {
        'snli': {
            'train': 64,
            'dev': 133,
            'test': 307
        },
        'mnli': {
            'train': 32,
            'dev': 65,
            'test': 8
        },
        'breaking': {
            'test': 3
        }
    },
    'shuffle': {
        'snli': {
            'train': True,
        },
        'mnli': {
            'train': True,
        }
    },
    'dataset': 'snli',
    'dump_elmo': False,
    'dump_lemma': False,
    'early_stopping': True,
    'elmo_dir': 'elmo',
    'elmo_use_weighted_embeddings': True,
    'n_labels': 3,
    'n_layers': 0,
    'n_epochs': 40,
    'batch_normalization': True,
    'dropout': 0.1,
    'optimizer': 'rmsprop',
    'learning_rate': 10000,
    'lr_schedule': '[[1000, 0.001]]',
    'lr_schedule_type': 'list_of_lists',
    'l2_strength': 1e-5,
    'intersection_of_embedding_dicts': False,
    'D': 0,
    'whitening': False,
    'centering': False,
    'normalize': False,
    'norm_weight': False,
    'embedding_dim': 300,
    'embedding_name': 'gcc840',
    'train_embeddings': False,
    'train_on_fraction': 1.0,
    'seed': 1,
    'sentence_max_length': 90,
    'save_best_model': True,
    'save_model': False,
    'steps_per_epoch_scale': 1.0,
    'steps_per_epoch': -1,
    'fullkim': False,
    'useatrick': 0,
    'useitrick': 0,
    'usectrick': 0,
    'use_elmo': False,
}

baseline_configs['cbow'] = {
    'dataset': 'snli',
    'n_labels': 3,
    'model': 'cbow',
    'n_layers': 3,
    'batch_sizes': {
        'snli': {
            'train': 91,
            'dev': 133,
            'test': 307
        },
        'breaking': {
            'test': 3
        }
    },
    'shuffle': {
        'snli': {
            'train': True,
        }
    },
    'early_stopping': True,
    'n_epochs': 40,
    'batch_normalization': False,
    'dropout': 0.1,
    'sentence_max_length': 90,
    'optimizer': 'rmsprop',
    'learning_rate': 10000,
    'lr_schedule': '[[1000, 0.001]]',
    'lr_schedule_type': 'list_of_lists',
    'intersection_of_embedding_dicts': False,
    'D': 0,
    'whitening': False,
    'centering': False,
    'normalize': False,
    'norm_weight': False,
    'embedding_dim': 300,
    'embedding_name': 'gcc840',
    'train_embeddings': False,
    'train_on_fraction': 1.0,
    'seed': 1,
    'useitrick': 0,
}

baseline_configs['esim'] = {
    'D': 0,
    'batch_sizes': {
        'snli': {
            'train': 32,
            'dev': 133,
            'test': 307
        },
        'mnli': {
            'train': 32,
            'dev': 65,
            'test': 8
        },
        'breaking': {
            'test': 3
        }
    },
    'shuffle': {
        'snli': {
            'train': True,
        },
        'mnli': {
            'train': True,
        }
    },
    'centering': False,
    'clip_gradient_norm': 10.0,
    'dataset': 'snli',
    'dropout': 0.5,
    'early_stopping': True,
    'elmo_dir': 'elmo',
    'embedding_dim': 300,
    'embedding_name': 'gcc840',
    'embedding_second_name': 'gcc840',
    'fullkim': False,
    'a_lambda': 0.2,
    'i_lambda': 1.0,
    'intersection_of_embedding_dicts': False,
    'knowledge_after_lstm': 'none',
    'learning_rate': 0.0004,
    'l2_elmo_regularization': 0.0,
    'l2_weight_regularization': 0.0,
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
    'save_best_model': True,
    'save_model': False,
    'seed': 2,
    'sentence_max_length': 90,
    'steps_per_epoch_scale': 1.0,
    'train_embeddings': False,
    'train_on_fraction': 1.0,
    'use_elmo': False,
    'useatrick': 0,
    'useitrick': 0,
    'usectrick': 0,
    'whitening': False
}