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
"""Convolutions layers.
"""
import tensorflow as tf
from tensorflow.python.keras import activations, initializers
from tensorflow.keras import backend


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Arguments:
        input_length: integer.
        filter_size: integer.
        padding: one of "same", "valid", "full", "causal"
        stride: integer.
        dilation: dilation rate, integer.

    Returns:
        The output length (integer).
    """
    if input_length is None:
        return None
    assert padding in {'same', 'valid', 'full', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if padding in ['same', 'causal']:
        output_length = input_length
    elif padding == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif padding == 'full':
        output_length = input_length + dilated_filter_size - 1
    return (output_length + stride - 1) // stride


def normalize_data_format(value):
    if value is None:
        value = backend.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


def normalize_tuple(value, n, name):
    """Transforms a single integer or iterable of integers into an integer tuple.

    Arguments:
      value: The value to validate and convert. Could an int, or any iterable of
        ints.
      n: The size of the tuple to be returned.
      name: The name of the argument being validated, e.g. "strides" or
        "kernel_size". This is only used to format error messages.

    Returns:
      A tuple of n integers.

    Raises:
      ValueError: If something else than an int/long or iterable thereof was
        passed.
    """
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except (ValueError, TypeError):
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
        return value_tuple


def normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'valid', 'same', 'causal'}:
        raise ValueError('The `padding` argument must be a list/tuple or one of '
                         '"valid", "same" (or "causal", only for `Conv1D). '
                         'Received: ' + str(padding))
    return padding


class Conv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 trainable=True,
                 name=None,
                 **kwargs):
        """
        Arguments:
            filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
            kernel_size: An integer or tuple/list of n integers, specifying the length of the convolution window.
            strides: An integer or tuple/list of n integers, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
            padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
            activation: Activation function to use. If you don't specify anything, no activation is applied.
            use_bias: Boolean, whether the layer uses a bias.
            trainable: Boolean, if `True` the weights of this layer will be marked as trainable (and listed in `layer.trainable_weights`).
            name: A string, the name of the layer.
        """
        super(Conv2D, self).__init__(trainable=trainable, name=name, **kwargs)
        rank = 2
        self.activation = activation
        self.filters = filters
        self.kernel_size = normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = normalize_tuple(strides, rank, 'strides')
        self.padding = normalize_padding(padding)
        self.activation = activation
        self.use_bias = use_bias
        self.data_format = normalize_data_format(data_format)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self._padding_op = self.get_padding_op()
        self.built = True

    def call(self, inputs):
        outputs = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self._padding_op)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space +
                                  [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i])
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters] +
                                  new_space)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(Conv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return 1
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding


class DepthwiseConv2D(Conv2D):
    """Depthwise separable 2D convolution.

    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    Arguments:
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `'valid'` or `'same'` (case-insensitive).
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel.
        The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be 'channels_last'.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied (
        see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix (
        see `keras.initializers`).
      bias_initializer: Initializer for the bias vector (
        see `keras.initializers`).

    Input shape:
      4D tensor with shape:
      `[batch_size, channels, rows, cols]` if data_format='channels_first'
      or 4D tensor with shape:
      `[batch_size, rows, cols, channels]` if data_format='channels_last'.

    Output shape:
      4D tensor with shape:
      `[batch_size, filters, new_rows, new_cols]` if data_format='channels_first'
      or 4D tensor with shape:
      `[batch_size, new_rows, new_cols, filters]` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.

    Returns:
      A tensor of rank 4 representing
      `activation(depthwiseconv2d(inputs, kernel) + bias)`.

    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        input_shape = tf.TensorShape(input_shape)
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel')

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        self._padding_op = self.get_padding_op()
        self.built = True

    def call(self, inputs):
        if self.data_format == 'channels_first':
            outputs = tf.nn.depthwise_conv2d(
                inputs,
                self.depthwise_kernel,
                strides=(1, ) + self.strides + (1, ),
                padding=self._padding_op,
                data_format="NCHW")
        else:
            outputs = tf.nn.depthwise_conv2d(
                inputs,
                self.depthwise_kernel,
                strides=(1, ) + self.strides + (1, ),
                padding=self._padding_op,
                data_format="NHWC")

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_output_length(rows, self.kernel_size[0],
                                  self.padding,
                                  self.strides[0])
        cols = conv_output_length(cols, self.kernel_size[1],
                                  self.padding,
                                  self.strides[1])
        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        return config


class ZeroPadding2D(tf.keras.layers.Layer):
    """Zero-padding layer for 2D input (e.g. picture).

    This layer can add rows and columns of zeros
    at the top, bottom, left and right side of an image tensor.

    Examples:

    >>> input_shape = (1, 1, 2, 2)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[0 1]
       [2 3]]]]
    >>> y = tf.keras.layers.ZeroPadding2D(padding=1)(x)
    >>> print(y)
    tf.Tensor(
      [[[[0 0]
         [0 0]
         [0 0]
         [0 0]]
        [[0 0]
         [0 1]
         [2 3]
         [0 0]]
        [[0 0]
         [0 0]
         [0 0]
         [0 0]]]], shape=(1, 3, 4, 2), dtype=int64)

    Arguments:
      padding: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        - If int: the same symmetric padding
          is applied to height and width.
        - If tuple of 2 ints:
          interpreted as two different
          symmetric padding values for height and width:
          `(symmetric_height_pad, symmetric_width_pad)`.
        - If tuple of 2 tuples of 2 ints:
          interpreted as
          `((top_pad, bottom_pad), (left_pad, right_pad))`
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, rows, cols)`

    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, padded_rows, padded_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, padded_rows, padded_cols)`
    """

    def __init__(self, padding=(1, 1), data_format=None, **kwargs):
        super(ZeroPadding2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = normalize_tuple(padding[0], 2,
                                             '1st entry of padding')
            width_padding = normalize_tuple(padding[1], 2,
                                            '2nd entry of padding')
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tf.TensorShape(
                [input_shape[0], input_shape[1], rows, cols])
        elif self.data_format == 'channels_last':
            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None
            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None
            return tf.TensorShape(
                [input_shape[0], rows, cols, input_shape[3]])

    def call(self, inputs):
        return backend.spatial_2d_padding(
            inputs, padding=self.padding, data_format=self.data_format)

    def get_config(self):
        config = {'padding': self.padding, 'data_format': self.data_format}
        base_config = super(ZeroPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
