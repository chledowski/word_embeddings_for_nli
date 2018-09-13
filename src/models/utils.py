"""
Definition of the layers necessary for the ESIM model.
Inspired from the code on:
https://github.com/coetaur0/ESIM
"""

import keras.backend as K
import warnings

from keras.activations import softmax
from keras.initializers import RandomNormal
from keras.layers import *
from keras.layers.recurrent import LSTM, LSTMCell
from keras.layers import Dropout
from keras.models import Sequential


class LSTMCellWithClipping(LSTMCell):

    def call(self, inputs, states, training=None):
        h, hc = super(LSTMCellWithClipping, self).call(inputs, states, training)
        _, c = hc
        c = K.clip()


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
                 proj_clip=None,
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
                        proj_clip=proj_clip)
        super(LSTMCell, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)


class EmbeddingLayer(object):
    """
    Layer to transform words represented by indices to word embeddings.
    """

    def __init__(self, voc_size, output_dim, embedding_weights=None,
                 max_length=100, trainable=True, mask_zero=False):
        self.voc_size = voc_size
        self.output_dim = output_dim
        self.max_length = max_length

        if embedding_weights is not None:
            self.model = Embedding(voc_size, output_dim,
                                   weights=[embedding_weights],
                                   input_length=max_length,
                                   trainable=trainable, mask_zero=mask_zero,
                                   name='embedding')
        else:
            # If no pretrained embedding weights are passed to the initialiser,
            # the model is set to be trainable by default.
            self.model = Embedding(voc_size, output_dim,
                                   input_length=max_length, trainable=True,
                                   mask_zero=mask_zero, name='embedding')

    def __call__(self, input):
        return self.model(input)


class EncodingLayer(object):
    """
    Layer to encode variable length sentences with a BiLSTM.
    """

    def __init__(self, hidden_units, max_length=100, dropout=0.5,
                 activation='tanh', sequences=True):
        self.layer = Bidirectional(CuDNNLSTM(hidden_units, return_sequences=sequences),
                                   merge_mode='concat')
        #self.layer = Bidirectional(LSTM(hidden_units, return_sequences=sequences),
        #                           merge_mode='concat')
        self.dropout = dropout

    def __call__(self, input):
        return Dropout(self.dropout)(self.layer(input))

class LocalInferenceLayer(object):
    """
    Layer to compute local inference between two encoded sentences a and b.
    """

    def __call__(self, inputs, dropout):
        F_p = inputs[0]
        F_h = inputs[1]

        Eph = Dot(axes=(2, 2))([F_h, F_p])  # [batch_size, Hsize, Psize]
        Eh = Lambda(lambda x: softmax(x))(Eph)  # [batch_size, Hsize, Psize]
        Ep = Permute((2, 1))(Eph)  # [batch_size, Psize, Hsize)
        Ep = Lambda(lambda x: softmax(x))(Ep)  # [batch_size, Psize, Hsize]

        # #  Normalize score matrix, bilstm_encoder premesis and get alignment
        PremAlign = Dot((2, 1))([Ep, F_h])  # [-1, Psize, dim]
        HypoAlign = Dot((2, 1))([Eh, F_p])  # [-1, Hsize, dim]
        mm_1 = Multiply()([F_p, PremAlign])
        mm_2 = Multiply()([F_h, HypoAlign])
        sb_1 = Subtract()([F_p, PremAlign])
        sb_2 = Subtract()([F_h, HypoAlign])

        PremAlign = Concatenate()([F_p, PremAlign, sb_1, mm_1])  # [batch_size, Psize, 2*unit]
        HypoAlign = Concatenate()([F_h, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]


        PremAlign = Dropout(dropout)(PremAlign)
        HypoAlign = Dropout(dropout)(HypoAlign)

        return PremAlign, HypoAlign

'''
class LocalInferenceLayer(object):
    """
    Layer to compute local inference between two encoded sentences a and b.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        attention = Lambda(self._attention,
                           self._attention_output_shape)(inputs)

        align_a = Lambda(self._soft_alignment,
                         self._soft_alignment_output_shape)([attention, a])
        align_b = Lambda(self._soft_alignment,
                         self._soft_alignment_output_shape)([attention, b])

        # Enhancement of the local inference information obtained with the
        # attention mecanism and soft alignments.
        sub_a_align = Lambda(lambda x: x[0]-x[1])([a, align_a])
        sub_b_align = Lambda(lambda x: x[0]-x[1])([b, align_b])

        mul_a_align = Lambda(lambda x: x[0]*x[1])([a, align_a])
        mul_b_align = Lambda(lambda x: x[0]*x[1])([b, align_b])

        m_a = concatenate([a, align_a, sub_a_align, mul_a_align])
        m_b = concatenate([b, align_b, sub_b_align, mul_b_align])

        return m_a, m_b

    def _attention(self, inputs):
        """
        Compute the attention between elements of two sentences with the dot
        product.
        Args:
            inputs: A list containing two elements, one for the first sentence
                    and one for the second, both encoded by a BiLSTM.
        Returns:
            A tensor containing the dot product (attention weights between the
            elements of the two sentences).
        """
        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1],
                                                          pattern=(0, 2, 1)))
        return K.permute_dimensions(attn_weights, (0, 2, 1))

    def _attention_output_shape(self, inputs):
        input_shape = inputs[0]
        embedding_size = input_shape[1]
        return (input_shape[0], embedding_size, embedding_size)

    def _soft_alignment(self, inputs):
        """
        Compute the soft alignment between the elements of two sentences.
        Args:
            inputs: A list of two elements, the first is a tensor of attention
                    weights, the second is the encoded sentence on which to
                    compute the alignments.
        Returns:
            A tensor containing the alignments.
        """
        attention = inputs[0]
        sentence = inputs[1]

        # Subtract the max. from the attention weights to avoid overflows.
        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    def _soft_alignment_output_shape(self, inputs):
        attention_shape = inputs[0]
        sentence_shape = inputs[1]
        return (attention_shape[0], attention_shape[1], sentence_shape[2])

'''
class InferenceCompositionLayer(object):
    """
    Layer to compose the local inference information.
    """

    def __init__(self, hidden_units, max_length=100, dropout=0.5,
                 activation='tanh', sequences=True):
        self.hidden_units = hidden_units
        self.max_length = max_length
        self.dropout = dropout
        self.activation = activation
        self.sequences = sequences

    def __call__(self, input):
        composition = Dropout(self.dropout)(Bidirectional(CuDNNLSTM(self.hidden_units,
                                                                     return_sequences=self.sequences))(input))
        #composition = Dropout(self.dropout)(Bidirectional(LSTM(self.hidden_units,
        #                                                            return_sequences=self.sequences))(input))

        reduction = TimeDistributed(Dense(self.hidden_units,
                                          kernel_initializer='he_normal',
                                          activation='relu'))(composition)

        return Dropout(self.dropout)(reduction)


class PoolingLayer(object):
    """
    Pooling layer to convert the vectors obtained in the previous layers to
    fixed-length vectors.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        a_avg = GlobalAveragePooling1D()(a)
        a_max = GlobalMaxPooling1D()(a)

        b_avg = GlobalAveragePooling1D()(b)
        b_max = GlobalMaxPooling1D()(b)

        return concatenate([a_avg, a_max, b_avg, b_max])


class MLPLayer(object):
    """
    Multi-layer perceptron for classification.
    """

    def __init__(self, hidden_units, n_classes, dropout=0.5,
                 activations=('tanh', 'softmax')):
        self.model = Sequential()
        self.model.add(Dense(hidden_units, kernel_initializer='he_normal',
                             activation=activations[0],
                             input_shape=(4*hidden_units,)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(n_classes, kernel_initializer='zero',
                             activation=activations[1]))

    def __call__(self, input):
        return self.model(input)


class ScaledRandomNormal(RandomNormal):
    def __init__(self, mean=0., stddev=0.05, scale=1.0, seed=None):
        super(ScaledRandomNormal, self).__init__(mean=mean, stddev=stddev, seed=seed)
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return self.scale * super(ScaledRandomNormal, self).__call__(shape, dtype=dtype)
