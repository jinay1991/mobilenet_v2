"""
Copyright (c) 2020. All Rights Reserved.
"""
import os
from datetime import datetime

import tensorflow as tf
import logging


class Conv2D(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, ksize, strides, activation=True):
        """
        filter_shape = [kernel_height, kernel_width, in_channel, out_channel]
        """
        super(Conv2D, self).__init__()
        self.weight = tf.Variable(tf.random.truncated_normal(shape=(ksize, ksize, in_ch, out_ch)))
        self.bias = tf.Variable(tf.zeros(shape=out_ch))
        self.strides = strides
        self.activation = activation

    def call(self, inputs):
        conv = tf.nn.conv2d(inputs, self.weight, strides=(1, self.strides, self.strides, 1), padding="SAME")
        conv = tf.add(conv, self.bias)
        if self.activation:
            conv = tf.nn.relu(conv)
        logging.info("  Conv2D: Input {} Output {}".format(inputs.get_shape(), conv.get_shape()))
        return conv


class DepthwiseConv2D(tf.keras.layers.Layer):
    def __init__(self, in_ch, ksize, strides):
        """
        filter_shape = [kernel_height, kernel_width, in_channel, out_channel]
        """
        super(DepthwiseConv2D, self).__init__()
        self.weight = tf.Variable(tf.random.truncated_normal(shape=(ksize, ksize, in_ch, 1)))
        self.bias = tf.Variable(tf.zeros(shape=in_ch))
        self.strides = strides

    def call(self, inputs):
        conv = tf.nn.depthwise_conv2d(inputs, self.weight, strides=(1, self.strides, self.strides, 1), padding="SAME")
        conv = tf.add(conv, self.bias)
        conv = tf.nn.relu(conv)
        logging.info("  DepthwiseConv2D: Input {} Output {}".format(inputs.get_shape(), conv.get_shape()))
        return conv


class Add(tf.keras.layers.Layer):
    def __init__(self):
        super(Add, self).__init__()

    def call(self, inputs):
        result = tf.add(inputs[0], inputs[1])
        logging.info("  Add: Input {} Output {}".format(inputs[0].get_shape(), result.get_shape()))
        return result


class Bottleneck(tf.keras.layers.Layer):
    def __init__(self, in_ch, out_ch, multiplier, repetitions, strides, enable_expansion=True):
        super(Bottleneck, self).__init__()
        input_channels = in_ch
        output_channels = out_ch
        self.depthwise_conv = []
        self.expand_conv = []
        self.project_conv = []
        self.add = []
        self.repetitions = repetitions
        self.enable_expansion = enable_expansion
        first_layer = True
        for i in range(0, self.repetitions):
            self.expand_conv.append(Conv2D(input_channels, input_channels * multiplier, ksize=1, strides=1))
            self.depthwise_conv.append(DepthwiseConv2D(input_channels * multiplier,
                                                       ksize=3, strides=strides if first_layer else 1))
            self.project_conv.append(Conv2D(input_channels * multiplier, output_channels,
                                            ksize=1, strides=1, activation=False))
            self.add.append(Add())
            first_layer = False
            input_channels = output_channels

    def call(self, inputs):
        prev_block = None
        block = inputs
        for i in range(0, self.repetitions):
            input_tensor = block
            if self.enable_expansion:
                block = self.expand_conv[i](block)
            block = self.depthwise_conv[i](block)
            block = self.project_conv[i](block)

            if prev_block is None:
                prev_block = block
            else:
                block = self.add[i]([prev_block, block])
                prev_block = None

            logging.info("Bottleneck: Input {} Output {}".format(input_tensor.get_shape(), block.get_shape()))

        return block


class AvgPool(tf.keras.layers.Layer):
    def __init__(self, strides):
        super(AvgPool, self).__init__()
        self.strides = strides

    def call(self, inputs):
        return tf.nn.avg_pool(inputs, ksize=(1, 3, 3, 1), strides=(1, self.strides, self.strides, 1), padding="SAME")


class Squeeze(tf.keras.layers.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        squeeze = tf.squeeze(inputs, axis=[1, 2])
        logging.info("Squeeze: Input {} Output {}".format(inputs.get_shape(), squeeze.get_shape()))
        return squeeze


class MobileNetV2(tf.keras.Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.conv2d_3x3 = Conv2D(3, 32, 3, 2)
        self.bottleneck_1 = Bottleneck(32, 16, 1, 1, 1, False)
        self.bottleneck_2 = Bottleneck(16, 24, 6, 2, 2)
        self.bottleneck_3 = Bottleneck(24, 32, 6, 3, 2)
        self.bottleneck_4 = Bottleneck(32, 64, 6, 4, 2)
        self.bottleneck_5 = Bottleneck(64, 96, 6, 3, 1)
        self.bottleneck_6 = Bottleneck(96, 160, 6, 3, 2)
        self.bottleneck_7 = Bottleneck(160, 320, 6, 1, 1)
        self.conv2d_1x1_1 = Conv2D(320, 1280, 1, 1)
        self.avgpool_7x7 = AvgPool(7)
        self.conv2d_1x1_2 = Conv2D(1280, 10, 1, 1, False)
        self.squeeze = Squeeze()

    def call(self, inputs):
        tensor = self.conv2d_3x3(inputs)
        tensor = self.bottleneck_1(tensor)
        tensor = self.bottleneck_2(tensor)
        tensor = self.bottleneck_3(tensor)
        tensor = self.bottleneck_4(tensor)
        tensor = self.bottleneck_5(tensor)
        tensor = self.bottleneck_6(tensor)
        tensor = self.bottleneck_7(tensor)
        tensor = self.conv2d_1x1_1(tensor)
        tensor = self.avgpool_7x7(tensor)
        tensor = self.conv2d_1x1_2(tensor)
        tensor = self.squeeze(tensor)
        return tensor


def format_example(image):
    image = tf.cast(image, tf.float32)
    image = tf.subtract(tf.divide(image, 127.5), 1)
    image = tf.image.resize(image, (224, 224))
    return image


def main(path, train=True):
    """ Main Function """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(32)

    x_val = format_example(x_train[-100:])
    y_val = tf.convert_to_tensor(y_train[-100:])
    x_train = format_example(x_train[:100])
    y_train = tf.convert_to_tensor(y_train[:100])

    if train:
        model = MobileNetV2()

        model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val), callbacks=[tensorboard_callback])
        model.save(path, save_format='tf')
    else:
        model = tf.keras.models.load_model('model')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    with open("model.tflite", "wb") as fp:
        fp.write(tflite_model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train Model")
    parser.add_argument("--save_model_to", help="Save Model to ", default="model")
    parser.add_argument("--verbose", action="store_true", help="Enable/Disable verbose printing")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.FATAL)

    main(args.save_model_to, args.train)
