#!/use/bin/python

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
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
        input_shape = tensor_shape.TensorShape(input_shape)
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

        self._padding_op = self._get_padding_op()
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
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                            [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return tensor_shape.TensorShape([input_shape[0], self.filters] +
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

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding
