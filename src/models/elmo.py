import functools
import h5py
import json
import keras.backend as K
import numpy as np
import os
import tensorflow as tf
import warnings

from keras.layers import *
from keras.layers.recurrent import LSTM, LSTMCell

from keras.layers.embeddings import Embedding
from keras.layers import Add, Subtract, Dense, Dropout, Input, TimeDistributed, Lambda, Bidirectional, \
    Dot, Permute, Multiply, Concatenate, Activation, LSTM

from src import DATA_DIR


class LSTMCellWithClippingAndProjection(Layer):
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
                 proj_clip=None,
                 projection_dim=None,
                 **kwargs):
        super(LSTMCellWithClippingAndProjection, self).__init__(**kwargs)
        self.units = units
        self.projected_units = projection_dim or self.units
        self.projection_dim = projection_dim
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.projected_units, self.units)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

        self.cell_clip = cell_clip
        self.proj_clip = proj_clip

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=(self.projected_units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.projection_dim is not None:
            self.projection_kernel = self.add_weight(
                shape=(self.units, self.projection_dim),
                name='projection_kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = self.recurrent_activation(x_i + K.dot(h_tm1_i,
                                                      self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1_f,
                                                      self.recurrent_kernel_f))
            c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1_c,
                                                            self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1_o,
                                                      self.recurrent_kernel_o))
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.kernel)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            z0 = z[:, :self.units]
            z1 = z[:, self.units: 2 * self.units]
            z2 = z[:, 2 * self.units: 3 * self.units]
            z3 = z[:, 3 * self.units:]

            i = self.recurrent_activation(z0)
            f = self.recurrent_activation(z1)
            c = f * c_tm1 + i * self.activation(z2)
            o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        # TODO(tomwesolowski): Clipping before or after dropout?
        if self.cell_clip is not None:
            c = K.clip(c, -self.cell_clip, self.cell_clip)

        if self.projection_dim is not None:
            h = K.dot(h, self.projection_kernel)
            if self.proj_clip is not None:
                h = K.clip(h, -self.proj_clip, self.proj_clip)

        return h, [h, c]


class LSTMWithClippingAndProjection(LSTM):
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
                 proj_clip=None,
                 projection_dim=None,
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

        cell = LSTMCellWithClippingAndProjection(units,
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
                                                 cell_clip=cell_clip,
                                                 proj_clip=proj_clip,
                                                 projection_dim=projection_dim)

        super(LSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)

        self.activity_regularizer = regularizers.get(activity_regularizer)


class ElmoEmbeddings(Layer):
    def __init__(self, config, **kwargs):
        super(ElmoEmbeddings, self).__init__(**kwargs)

        self.config = config
        self.elmo_dir = os.path.join(DATA_DIR, config['elmo_dir'])

        options_file = os.path.join(self.elmo_dir, 'options.json')
        with open(options_file, 'r') as fin:
            self.options = json.load(fin)

        self.num_layers = 2
        self.lstm_dim = self.options['lstm']['dim']
        self.projection_dim = self.options['lstm']['projection_dim']
        self.n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        self.cell_clip = self.options['lstm'].get('cell_clip')
        self.proj_clip = self.options['lstm'].get('proj_clip')
        self.use_skip_connections = self.options['lstm']['use_skip_connections']

        with h5py.File(os.path.join(self.elmo_dir, 'elmo_token_embeddings.hdf5'), 'r') as fin:
            self.vocab_size, self.embedding_dim = fin['embedding'].shape

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        trainables = []
        for sublayer in self.sublayers:
            trainables.extend(sublayer.trainable_weights)
        return trainables

    @property
    def non_trainable_weights(self):
        nontrainables = []
        for sublayer in self.sublayers:
            if not self.trainable:
                nontrainables.extend(sublayer.weights)
            else:
                nontrainables.extend(sublayer.trainable_weights)
        return nontrainables

    def compute_output_shape(self, input_shape):
        return input_shape[0] + (2*self.projection_dim,)

    def _load_embedding_weights(self):
        with h5py.File(os.path.join(self.elmo_dir, 'elmo_token_embeddings.hdf5'), 'r') as fin:
            # Have added a special 0 index for padding not present
            # in the original model.
            embed_weights = fin['embedding'][...]
            self.embedding_dim = embed_weights.shape[1]
            weights = np.zeros(
                (embed_weights.shape[0] + 1, self.embedding_dim),
                dtype='float32'
            )
            weights[1:, :] = embed_weights
        return weights

    def print_weights_shapes(self):
        for weight in self.lstms['forward'][0].get_weights():
            print(weight.shape)

    def load_weights(self):
        embedding_weights = self._load_embedding_weights()
        self.embed.set_weights([embedding_weights])

        weight_file = os.path.join(self.elmo_dir, 'lm_weights.hdf5')

        with h5py.File(weight_file, 'r') as fin:
            for i_layer in range(self.num_layers):
                for i_dir, direction in enumerate(['forward', 'backward']):
                    prefix = 'RNN_%d/RNN/MultiRNNCell/Cell%d/LSTMCell/' % (i_layer, i_dir)
                    self.lstms[direction][i_layer].set_weights([
                        fin[prefix + 'W_0'][:self.embedding_dim],  # kernel
                        fin[prefix + 'W_0'][self.embedding_dim:],  # recurrent kernel
                        fin[prefix + 'W_P_0'][...],  # projection
                        fin[prefix + 'B'][...]  # bias
                    ])

    def build(self, input_shapes):
        self.sublayers = []

        self.embed = Embedding(self.vocab_size + 1,
                               self.embedding_dim,
                               input_length=self.config["sentence_max_length"],
                               trainable=False,
                               mask_zero=False)
        self.sublayers.append(self.embed)
        self.lstms = {}

        for direction in ['forward', 'backward']:
            self.lstms[direction] = []
            for i in range(self.num_layers):
                lstm = LSTMWithClippingAndProjection(
                        recurrent_initializer='zeros',
                        recurrent_activation='sigmoid',
                        units=self.lstm_dim,
                        return_sequences=True,
                        cell_clip=self.cell_clip,
                        projection_dim=self.projection_dim,
                        proj_clip=self.proj_clip,
                        go_backwards=direction == 'backward',
                        trainable=False
                )
                self.lstms[direction].append(lstm)
                self.sublayers.append(lstm)

        self.built = True

    def call(self, inputs, weight_name, **kwargs):
        # TODO(chledows): add charcnn and finetuning.
        # TODO(tomwesolowski): After it's done, check if embeddings are the same as in dumped embeddings file.
        input_embeddings, mask = inputs
        input_embeddings = self.embed(input_embeddings)
        elmo_embeddings = {}
        print("num_layers", self.num_layers)
        for direction in ['forward', 'backward']:
            input = input_embeddings
            elmo_embeddings[direction] = [input]
            for i in range(self.num_layers):
                output = self.lstms[direction][i](input)  # [-1, maxlen, lstm_dim]
                if i > 0 and self.use_skip_connections:
                    # Residual connection between between first hidden and output layer.
                    input = Lambda(lambda x: x[0] + x[1])([input, output])
                else:
                    input = output
                elmo_embeddings[direction].append(input)  # [-1, maxlen, projection_dim]

        elmo_embeddings_both = []
        for forward_layer, backward_layer in zip(elmo_embeddings['forward'], elmo_embeddings['backward']):
            forward_layer = Lambda(lambda x: K.expand_dims(x, axis=1))(forward_layer)  # [-1, 1, maxlen, projection_dim]
            backward_layer = Lambda(lambda x: K.expand_dims(x, axis=1))(backward_layer)
            elmo_embeddings_both.append(
                Concatenate(axis=-1)([forward_layer, backward_layer])  # [-1, 1, maxlen, 2*projection_dim]
            )

        elmo_embeddings_concat = Concatenate(axis=1)(elmo_embeddings_both)  # [-1, num_layers, maxlen, 2*projection_dim]
        weight_layers_fn = functools.partial(self.weight_layers, weight_name)
        return Lambda(lambda x: weight_layers_fn(x)['weighted_op'])(
            [elmo_embeddings_concat, mask]
        )

    def weight_layers(self, name, lm_embeddings_and_mask, l2_coef=None,
                      use_top_only=False, do_layer_norm=False):
        '''
        Weight the layers of a biLM with trainable scalar weights to
        compute ELMo representations.
        For each output layer, this returns two ops.  The first computes
            a layer specific weighted average of the biLM layers, and
            the second the l2 regularizer loss term.
        The regularization terms are also add to tf.GraphKeys.REGULARIZATION_LOSSES
        Input:
            name = a string prefix used for the trainable variable names
            bilm_ops = the tensorflow ops returned to compute internal
                representations from a biLM.  This is the return value
                from BidirectionalLanguageModel(...)(ids_placeholder)
            l2_coef: the l2 regularization coefficient $\lambda$.
                Pass None or 0.0 for no regularization.
            use_top_only: if True, then only use the top layer.
            do_layer_norm: if True, then apply layer normalization to each biLM
                layer before normalizing
        Output:
            {
                'weighted_op': op to compute weighted average for output,
                'regularization_op': op to compute regularization term
            }
        '''
        lm_embeddings, mask = lm_embeddings_and_mask

        def _l2_regularizer(weights):
            if l2_coef is not None:
                return l2_coef * tf.reduce_sum(tf.square(weights))
            else:
                return tf.constant(0.0)

        n_lm_layers = int(lm_embeddings.get_shape()[1])
        lm_dim = int(lm_embeddings.get_shape()[3])

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            with tf.control_dependencies([lm_embeddings, mask]):
                # Cast the mask and broadcast for layer use.
                mask_float = tf.cast(mask, 'float32')
                broadcast_mask = tf.expand_dims(mask_float, axis=-1)

                def _do_ln(x):
                    # do layer normalization excluding the mask
                    x_masked = x * broadcast_mask
                    N = tf.reduce_sum(mask_float) * lm_dim
                    mean = tf.reduce_sum(x_masked) / N
                    variance = tf.reduce_sum(((x_masked - mean) * broadcast_mask) ** 2
                                             ) / N
                    return tf.nn.batch_normalization(
                        x, mean, variance, None, None, 1E-12
                    )

                if use_top_only:
                    layers = tf.split(lm_embeddings, n_lm_layers, axis=1)
                    # just the top layer
                    sum_pieces = tf.squeeze(layers[-1], squeeze_dims=1)
                    # no regularization
                    reg = 0.0
                else:
                    W = self.add_weight(
                        name='{}_ELMo_W'.format(name),
                        shape=(n_lm_layers,),
                        initializer='zeros',
                        regularizer=_l2_regularizer,
                        trainable=True,
                    )
                    # get the regularizer
                    reg = self._losses[-1]

                    # normalize the weights
                    normed_weights = tf.split(
                        tf.nn.softmax(W + 1.0 / n_lm_layers), n_lm_layers
                    )
                    # split LM layers
                    layers = tf.split(lm_embeddings, n_lm_layers, axis=1)

                    # compute the weighted, normalized LM activations
                    pieces = []
                    for w, t in zip(normed_weights, layers):
                        if do_layer_norm:
                            pieces.append(w * _do_ln(tf.squeeze(t, squeeze_dims=1)))
                        else:
                            pieces.append(w * tf.squeeze(t, squeeze_dims=1))
                    sum_pieces = tf.add_n(pieces)


                # scale the weighted sum by gamma
                gamma = self.add_weight(
                    name='{}_ELMo_gamma'.format(name),
                    shape=(1,),
                    initializer='ones',
                    regularizer=None,
                    trainable=True,
                )
                weighted_lm_layers = sum_pieces * gamma

                print(K.int_shape(weighted_lm_layers))

                ret = {'weighted_op': weighted_lm_layers, 'regularization_op': reg}

        return ret



