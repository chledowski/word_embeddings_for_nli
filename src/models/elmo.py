import functools
import h5py
import json
import keras.backend as K
import os
import numpy as np
import warnings

from keras.layers import *
from keras.layers.recurrent import LSTM, LSTMCell

from keras.layers.embeddings import Embedding
from keras.layers import Add, Subtract, Dense, Dropout, Input, TimeDistributed, Lambda, Bidirectional, \
    Dot, Permute, Multiply, Concatenate, Activation, LSTM

from src import DATA_DIR
from src.models.bilm.data import Vocabulary


class LSTMCellWithClipping(LSTMCell):

    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 cell_clip=None,
                 **kwargs):
        super(LSTMCellWithClipping, self).__init__(units,
                                                   activation,
                                                   recurrent_activation,
                                                   use_bias,
                                                   kernel_initializer,
                                                   recurrent_initializer,
                                                   bias_initializer,
                                                   unit_forget_bias,
                                                   kernel_regularizer,
                                                   recurrent_regularizer,
                                                   bias_regularizer,
                                                   kernel_constraint,
                                                   recurrent_constraint,
                                                   bias_constraint,
                                                   dropout,
                                                   recurrent_dropout,
                                                   implementation,
                                                   **kwargs)
        self.cell_clip = cell_clip

    def call(self, inputs, states, training=None):
        h, hc = super(LSTMCellWithClipping, self).call(inputs, states, training)
        _, c = hc
        c = K.clip(c, -self.cell_clip, self.cell_clip)
        return h, [h, c]


class LSTMWithClipping(LSTM):
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 cell_clip=None,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = LSTMCellWithClipping(units,
                                    activation=activation,
                                    recurrent_activation=recurrent_activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    recurrent_initializer=recurrent_initializer,
                                    unit_forget_bias=unit_forget_bias,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    recurrent_regularizer=recurrent_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    recurrent_constraint=recurrent_constraint,
                                    bias_constraint=bias_constraint,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout,
                                    implementation=implementation,
                                    cell_clip=cell_clip)

        super(LSTMCell, self).__init__(cell,
                                       return_sequences=return_sequences,
                                       return_state=return_state,
                                       go_backwards=go_backwards,
                                       stateful=stateful,
                                       unroll=unroll,
                                       **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)


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
                LSTMWithClipping(
                    recurrent_activation='sigmoid',
                    units=lstm_dim,
                    return_sequences=True,
                    cell_clip = cell_clip
                )
            )
            self.biltsms[i].trainable(False)
            self.projections[i] = TimeDistributed(
                Dense(self.embedding_dim,
                      activation=functools.partial(clip_activation, K.hard_sigmoid, proj_clip))
            )


    def __call__(self, input, *args, **kwargs):
        #TODO (chledows): add charcnn and finetuning.
        input = self.embed(input)
        elmo_embeddings = [input]
        for i in range(2):
            output = self.biltsms[i](input)
            output = self.projections[i](output)
            if i > 0:
                input = Lambda(lambda x: x[0] + x[1])([input, output])
            else:
                input = output
            elmo_embeddings.append(input)
        return elmo_embeddings

