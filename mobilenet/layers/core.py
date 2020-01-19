# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core Keras layers.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers


class Activation(tf.keras.layers.Layer):
    """Applies an activation function to an output.

    Arguments:
      activation: Activation function, such as `tf.nn.relu`, or string name of
        built-in activation function, such as "relu".

    Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

    Output shape:
      Same shape as input.
    """

    def __init__(self, activation, **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.supports_masking = True
        self.activation = activations.get(activation)

    def call(self, inputs):
        return self.activation(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'activation': activations.serialize(self.activation)}
        base_config = super(Activation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dense(tf.keras.layers.Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    Besides, layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).

    Example:

    ```python
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)

    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(32))
    ```

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.

    Input shape:
      N-D tensor with shape: `(batch_size, ..., input_dim)`.
      The most common situation would be
      a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(batch_size, ..., units)`.
      For instance, for a 2D input with shape `(batch_size, input_dim)`,
      the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Dense, self).__init__(**kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tf.compat.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(
            'kernel',
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=[self.units, ],
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        rank = len(inputs.shape)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not tf.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = tf.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                outputs = tf.sparse.sparse_tensor_dense_matmul(inputs, self.kernel)
            else:
                outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
