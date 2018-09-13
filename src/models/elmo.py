import functools
import h5py
import json
import keras.backend as K
import os
import numpy as np


from keras.layers.embeddings import Embedding
from keras.layers import Add, Subtract, Dense, Dropout, Input, TimeDistributed, Lambda, Bidirectional, \
    Dot, Permute, Multiply, Concatenate, Activation, LSTM

from src import DATA_DIR
from src.models.bilm.data import Vocabulary


class Elmo:

    def __init__(self, config):
        elmo_dir = os.path.join(DATA_DIR, config['elmo_dir'])

        with h5py.File(os.path.join(elmo_dir, 'elmo_token_embeddings.hdf5'), 'r') as fin:
            # Have added a special 0 index for padding not present
            # in the original model.
            embed_weights = fin['embeddings'][...]
            self.embedding_dim = embed_weights.shape[1]
            weights = np.zeros(
                (embed_weights.shape[0] + 1, self.embedding_dim),
                dtype='float32'
            )
            weights[1:, :] = embed_weights

        options_file = os.path.join(elmo_dir, 'options.json')
        with open(options_file, 'r') as fin:
            self.options = json.load(fin)

        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm']['use_skip_connections']

        self.embed = Embedding(weights.shape[0], weights.shape[1],
                               weights=[weights],
                               input_length=config["sentence_max_length"],
                               trainable=False,
                               mask_zero=False)

        self.biltsms = []
        self.projections = []

        def clip_activation(act_fn, min_max_value, x):
            return K.clip(act_fn(x), -min_max_value, min_max_value)

        for i in range(2):

            self.bilstms[i] = Bidirectional(
                LSTM(
                    activity=functools.partial(clip_activation, K.tanh, cell_clip),
                    recurrent_activation='sigmoid',
                    units=lstm_dim,
                    return_sequences=True
                )
            )
            self.projections[i] = TimeDistributed(
                Dense(self.embedding_dim,
                      activation=functools.partial(clip_activation, K.hard_sigmoid, cell_clip))
            )


    def __call__(self, input, *args, **kwargs):
        input = self.embed(input)
        for i in range(2):
            output = self.biltsms[i](input)
            output = self.projections[i](output)
            if i > 0:
                input = Lambda(lambda x: x[0] + x[1])([input, output])
            else:
                input = output
