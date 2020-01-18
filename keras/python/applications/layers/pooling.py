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
"""Pooling layers.
"""
import tensorflow as tf
from tensorflow.keras.backend import image_data_format


def normalize_data_format(value):
    if value is None:
        value = image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


class GlobalPooling2D(tf.keras.layers.Layer):
    """Abstract class for different global pooling 2D layers.
    """

    def __init__(self, data_format=None, **kwargs):
        super(GlobalPooling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self._supports_ragged_inputs = True

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            return tf.TensorShape([input_shape[0], input_shape[3]])
        else:
            return tf.TensorShape([input_shape[0], input_shape[1]])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(GlobalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalAveragePooling2D(GlobalPooling2D):
    """Global average pooling operation for spatial data.

    Arguments:
        data_format: A string,
          one of `channels_last` (default) or `channels_first`.
          The ordering of the dimensions in the inputs.
          `channels_last` corresponds to inputs with shape
          `(batch, height, width, channels)` while `channels_first`
          corresponds to inputs with shape
          `(batch, channels, height, width)`.
          It defaults to the `image_data_format` value found in your
          Keras config file at `~/.keras/keras.json`.
          If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      2D tensor with shape `(batch_size, channels)`.
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return tf.math.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        else:
            return tf.math.reduce_mean(inputs, axis=[2, 3], keepdims=False)


class GlobalMaxPooling2D(GlobalPooling2D):
    """Global max pooling operation for spatial data.

    Arguments:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      2D tensor with shape `(batch_size, channels)`.
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return tf.math.reduce_max(inputs, axis=[1, 2], keepdims=False)
        else:
            return tf.math.reduce_max(inputs, axis=[2, 3], keepdims=False)
