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
# pylint: disable=not-callable
# pylint: disable=redefined-builtin
"""Layers that can merge several inputs into one.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.python.keras import backend as K


class _Merge(tf.keras.layers.Layer):
    """Generic merge layer for elementwise merge functions.

    Used to implement `Sum`, `Average`, etc.

    Arguments:
        **kwargs: standard layer keyword arguments.
    """

    def __init__(self, **kwargs):
        super(_Merge, self).__init__(**kwargs)
        self.supports_masking = True
        self._supports_ragged_inputs = True

    def _merge_function(self, inputs):
        raise NotImplementedError

    def _compute_elemwise_op_output_shape(self, shape1, shape2):
        """Computes the shape of the resultant of an elementwise operation.

        Arguments:
            shape1: tuple or None. Shape of the first tensor
            shape2: tuple or None. Shape of the second tensor

        Returns:
            expected output shape when an element-wise operation is
            carried out on 2 tensors with shapes shape1 and shape2.
            tuple or None.

        Raises:
            ValueError: if shape1 and shape2 are not compatible for
                element-wise operations.
        """
        if None in [shape1, shape2]:
            return None
        elif len(shape1) < len(shape2):
            return self._compute_elemwise_op_output_shape(shape2, shape1)
        elif not shape2:
            return shape1
        output_shape = list(shape1[:-len(shape2)])
        for i, j in zip(shape1[-len(shape2):], shape2):
            if i is None or j is None:
                output_shape.append(None)
            elif i == 1:
                output_shape.append(j)
            elif j == 1:
                output_shape.append(i)
            else:
                if i != j:
                    raise ValueError(
                        'Operands could not be broadcast '
                        'together with shapes ' + str(shape1) + ' ' + str(shape2))
                output_shape.append(i)
        return tuple(output_shape)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A merge layer should be called '
                             'on a list of at least 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) > 1:
            raise ValueError(
                'Can not merge tensors with different '
                'batch sizes. Got tensors with shapes : ' + str(input_shape))
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        # If the inputs have different ranks, we have to reshape them
        # to make them broadcastable.
        if None not in input_shape and len(set(map(len, input_shape))) == 1:
            self._reshape_required = False
        else:
            self._reshape_required = True

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A merge layer should be called on a list of inputs.')
        if self._reshape_required:
            reshaped_inputs = []
            input_ndims = list(map(K.ndim, inputs))
            if None not in input_ndims:
                # If ranks of all inputs are available,
                # we simply expand each of them at axis=1
                # until all of them have the same rank.
                max_ndim = max(input_ndims)
                for x in inputs:
                    x_ndim = K.ndim(x)
                    for _ in range(max_ndim - x_ndim):
                        x = tf.expand_dims(x, axis=1)
                    reshaped_inputs.append(x)
                return self._merge_function(reshaped_inputs)
            else:
                # Transpose all inputs so that batch size is the last dimension.
                # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
                transposed = False
                for x in inputs:
                    x_ndim = K.ndim(x)
                    if x_ndim is None:
                        x_shape = tf.shape(x)
                        batch_size = x_shape[0]
                        new_shape = K.concatenate(
                            [x_shape[1:],
                             tf.expand_dims(batch_size, axis=-1)])
                        x_transposed = tf.reshape(
                            x,
                            tf.stack(
                                [batch_size, tf.math.reduce_prod(x_shape[1:])], axis=0))
                        x_transposed = tf.transpose(x_transposed, perm=(1, 0))
                        x_transposed = tf.reshape(x_transposed, new_shape)
                        reshaped_inputs.append(x_transposed)
                        transposed = True
                    elif x_ndim > 1:
                        dims = list(range(1, x_ndim)) + [0]
                        reshaped_inputs.append(tf.transpose(x, perm=dims))
                        transposed = True
                    else:
                        # We don't transpose inputs if they are 1D vectors or scalars.
                        reshaped_inputs.append(x)
                y = self._merge_function(reshaped_inputs)
                y_ndim = K.ndim(y)
                if transposed:
                    # If inputs have been transposed, we have to transpose the output too.
                    if y_ndim is None:
                        y_shape = tf.shape(y)
                        y_ndim = tf.shape(y_shape)[0]
                        batch_size = y_shape[y_ndim - 1]
                        new_shape = K.concatenate([
                            tf.expand_dims(batch_size, axis=-1), y_shape[:y_ndim - 1]
                        ])
                        y = tf.reshape(y, (-1, batch_size))
                        y = tf.transpose(y, perm=(1, 0))
                        y = tf.reshape(y, new_shape)
                    elif y_ndim > 1:
                        dims = [y_ndim - 1] + list(range(y_ndim - 1))
                        y = tf.transpose(y, perm=dims)
                return y
        else:
            return self._merge_function(inputs)

    def compute_output_shape(self, input_shape):
        if input_shape[0] is None:
            output_shape = None
        else:
            output_shape = input_shape[0][1:]
        for i in range(1, len(input_shape)):
            if input_shape[i] is None:
                shape = None
            else:
                shape = input_shape[i][1:]
            output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
        batch_sizes = {s[0] for s in input_shape if s is not None} - {None}
        if len(batch_sizes) == 1:
            output_shape = (list(batch_sizes)[0],) + output_shape
        else:
            output_shape = (None,) + output_shape
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, list):
            raise ValueError('`mask` should be a list.')
        if not isinstance(inputs, list):
            raise ValueError('`inputs` should be a list.')
        if len(mask) != len(inputs):
            raise ValueError('The lists `inputs` and `mask` '
                             'should have the same length.')
        if all(m is None for m in mask):
            return None
        masks = [tf.expand_dims(m, axis=0) for m in mask if m is not None]
        return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


class Add(_Merge):
    """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    Examples:

    ```python
        import keras

        input1 = keras.layers.Input(shape=(16,))
        x1 = keras.layers.Dense(8, activation='relu')(input1)
        input2 = keras.layers.Input(shape=(32,))
        x2 = keras.layers.Dense(8, activation='relu')(input2)
        # equivalent to `added = keras.layers.add([x1, x2])`
        added = keras.layers.Add()([x1, x2])
        out = keras.layers.Dense(4)(added)
        model = keras.models.Model(inputs=[input1, input2], outputs=out)
    ```
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output
