"""
Copyright (c) 2020. All Rights Reserved.
"""
import os
from datetime import datetime

import tensorflow as tf
import logging


class MobileNetV2(tf.keras.Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        mu = 0
        sigma = 1e-2

        self.weight1 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 3, 32), mean=mu, stddev=sigma))
        self.bias1 = tf.Variable(tf.zeros(shape=(32)))
        self.weight2 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 32, 1), mean=mu, stddev=sigma))
        self.bias2 = tf.Variable(tf.zeros(shape=(32)))
        self.weight3 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 16), mean=mu, stddev=sigma))
        self.bias3 = tf.Variable(tf.zeros(shape=(16)))
        self.weight4 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 16, 96), mean=mu, stddev=sigma))
        self.bias4 = tf.Variable(tf.zeros(shape=(96)))
        self.weight5 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 96, 1), mean=mu, stddev=sigma))
        self.bias5 = tf.Variable(tf.zeros(shape=(96)))
        self.weight6 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 24), mean=mu, stddev=sigma))
        self.bias6 = tf.Variable(tf.zeros(shape=(24)))
        self.weight7 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144), mean=mu, stddev=sigma))
        self.bias7 = tf.Variable(tf.zeros(shape=(144)))
        self.weight8 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1), mean=mu, stddev=sigma))
        self.bias8 = tf.Variable(tf.zeros(shape=(144)))
        self.weight9 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 24), mean=mu, stddev=sigma))
        self.bias9 = tf.Variable(tf.zeros(shape=(24)))
        self.weight10 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144), mean=mu, stddev=sigma))
        self.bias10 = tf.Variable(tf.zeros(shape=(144)))
        self.weight11 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1), mean=mu, stddev=sigma))
        self.bias11 = tf.Variable(tf.zeros(shape=(144)))
        self.weight12 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 32), mean=mu, stddev=sigma))
        self.bias12 = tf.Variable(tf.zeros(shape=(32)))
        self.weight13 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192), mean=mu, stddev=sigma))
        self.bias13 = tf.Variable(tf.zeros(shape=(192)))
        self.weight14 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1), mean=mu, stddev=sigma))
        self.bias14 = tf.Variable(tf.zeros(shape=(192)))
        self.weight15 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 32), mean=mu, stddev=sigma))
        self.bias15 = tf.Variable(tf.zeros(shape=(32)))
        self.weight16 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192), mean=mu, stddev=sigma))
        self.bias16 = tf.Variable(tf.zeros(shape=(192)))
        self.weight17 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1), mean=mu, stddev=sigma))
        self.bias17 = tf.Variable(tf.zeros(shape=(192)))
        self.weight18 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 32), mean=mu, stddev=sigma))
        self.bias18 = tf.Variable(tf.zeros(shape=(32)))
        self.weight19 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192), mean=mu, stddev=sigma))
        self.bias19 = tf.Variable(tf.zeros(shape=(192)))
        self.weight20 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1), mean=mu, stddev=sigma))
        self.bias20 = tf.Variable(tf.zeros(shape=(192)))
        self.weight21 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 64), mean=mu, stddev=sigma))
        self.bias21 = tf.Variable(tf.zeros(shape=(64)))
        self.weight22 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        self.bias22 = tf.Variable(tf.zeros(shape=(384)))
        self.weight23 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        self.bias23 = tf.Variable(tf.zeros(shape=(384)))
        self.weight24 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64), mean=mu, stddev=sigma))
        self.bias24 = tf.Variable(tf.zeros(shape=(64)))
        self.weight25 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        self.bias25 = tf.Variable(tf.zeros(shape=(384)))
        self.weight26 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        self.bias26 = tf.Variable(tf.zeros(shape=(384)))
        self.weight27 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64), mean=mu, stddev=sigma))
        self.bias27 = tf.Variable(tf.zeros(shape=(64)))
        self.weight28 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        self.bias28 = tf.Variable(tf.zeros(shape=(384)))
        self.weight29 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        self.bias29 = tf.Variable(tf.zeros(shape=(384)))
        self.weight30 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64), mean=mu, stddev=sigma))
        self.bias30 = tf.Variable(tf.zeros(shape=(64)))
        self.weight31 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        self.bias31 = tf.Variable(tf.zeros(shape=(384)))
        self.weight32 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        self.bias32 = tf.Variable(tf.zeros(shape=(384)))
        self.weight33 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 96), mean=mu, stddev=sigma))
        self.bias33 = tf.Variable(tf.zeros(shape=(96)))
        self.weight34 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 576), mean=mu, stddev=sigma))
        self.bias34 = tf.Variable(tf.zeros(shape=(576)))
        self.weight35 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 576, 1), mean=mu, stddev=sigma))
        self.bias35 = tf.Variable(tf.zeros(shape=(576)))
        self.weight36 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 576, 96), mean=mu, stddev=sigma))
        self.bias36 = tf.Variable(tf.zeros(shape=(96)))
        self.weight37 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 576), mean=mu, stddev=sigma))
        self.bias37 = tf.Variable(tf.zeros(shape=(576)))
        self.weight38 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 576, 1), mean=mu, stddev=sigma))
        self.bias38 = tf.Variable(tf.zeros(shape=(576)))
        self.weight39 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 576, 96), mean=mu, stddev=sigma))
        self.bias39 = tf.Variable(tf.zeros(shape=(96)))
        self.weight40 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 576), mean=mu, stddev=sigma))
        self.bias40 = tf.Variable(tf.zeros(shape=(576)))
        self.weight41 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 576, 1), mean=mu, stddev=sigma))
        self.bias41 = tf.Variable(tf.zeros(shape=(576)))
        self.weight42 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 576, 160), mean=mu, stddev=sigma))
        self.bias42 = tf.Variable(tf.zeros(shape=(160)))
        self.weight43 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 160, 960), mean=mu, stddev=sigma))
        self.bias43 = tf.Variable(tf.zeros(shape=(960)))
        self.weight44 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 960, 1), mean=mu, stddev=sigma))
        self.bias44 = tf.Variable(tf.zeros(shape=(960)))
        self.weight45 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 960, 160), mean=mu, stddev=sigma))
        self.bias45 = tf.Variable(tf.zeros(shape=(160)))
        self.weight46 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 160, 960), mean=mu, stddev=sigma))
        self.bias46 = tf.Variable(tf.zeros(shape=(960)))
        self.weight47 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 960, 1), mean=mu, stddev=sigma))
        self.bias47 = tf.Variable(tf.zeros(shape=(960)))
        self.weight48 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 960, 160), mean=mu, stddev=sigma))
        self.bias48 = tf.Variable(tf.zeros(shape=(160)))
        self.weight49 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 160, 960), mean=mu, stddev=sigma))
        self.bias49 = tf.Variable(tf.zeros(shape=(960)))
        self.weight50 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 960, 1), mean=mu, stddev=sigma))
        self.bias50 = tf.Variable(tf.zeros(shape=(960)))
        self.weight51 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 960, 320), mean=mu, stddev=sigma))
        self.bias51 = tf.Variable(tf.zeros(shape=(320)))
        self.weight52 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 320, 1280), mean=mu, stddev=sigma))
        self.bias52 = tf.Variable(tf.zeros(shape=(1280)))
        self.weight54 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 1280, 10), mean=mu, stddev=sigma))
        self.bias54 = tf.Variable(tf.zeros(shape=(10)))

    def call(self, inputs):
        # [conv2d 3x3] Input 224x224x3 Output 112x112x32
        # t = ?, c = 32, n = 1, s = 2
        with tf.compat.v1.variable_scope("conv2d_3x3") as vs:
            conv1 = tf.nn.conv2d(inputs, self.weight1, strides=(1, 2, 2, 1), padding="SAME")
            conv1 = tf.add(conv1, self.bias1)
            conv1 = tf.nn.relu(conv1)
            logging.info("{}: Input {} Output {}".format(vs.name, inputs.get_shape(), conv1.get_shape()))

        # [bottleneck (1)] Input 112x112x32 Output 112x112x16
        # t = 1, c = 16, n = 1, s = 1
        with tf.compat.v1.variable_scope("bottleneck_1") as vs:
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv2 = tf.nn.depthwise_conv2d(conv1, self.weight2, strides=(1, 1, 1, 1), padding="SAME")
                conv2 = tf.add(conv2, self.bias2)
                conv2 = tf.nn.relu(conv2)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv1.get_shape(), conv2.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv3 = tf.nn.conv2d(conv2, self.weight3, strides=(1, 1, 1, 1), padding="SAME")
                conv3 = tf.add(conv3, self.bias3)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv2.get_shape(), conv3.get_shape()))

        # [bottleneck (2)] Input 112x112x16 Output 56x56x24
        # t = 6, c = 24, n = 2, s = 2
        with tf.compat.v1.variable_scope("bottleneck_2") as vs:
            # ------ n = 1 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv4 = tf.nn.conv2d(conv3, self.weight4, strides=(1, 1, 1, 1), padding="SAME")
                conv4 = tf.add(conv4, self.bias4)
                conv4 = tf.nn.relu(conv4)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv3.get_shape(), conv4.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv5 = tf.nn.depthwise_conv2d(conv4, self.weight5, strides=(1, 2, 2, 1), padding="SAME")
                conv5 = tf.add(conv5, self.bias5)
                conv5 = tf.nn.relu(conv5)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv4.get_shape(), conv5.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv6 = tf.nn.conv2d(conv5, self.weight6, strides=(1, 1, 1, 1), padding="SAME")
                conv6 = tf.add(conv6, self.bias6)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv5.get_shape(), conv6.get_shape()))

            # ------ n = 2 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv7 = tf.nn.conv2d(conv6, self.weight7, strides=(1, 1, 1, 1), padding="SAME")
                conv7 = tf.add(conv7, self.bias7)
                conv7 = tf.nn.relu(conv7)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv6.get_shape(), conv7.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv8 = tf.nn.depthwise_conv2d(conv7, self.weight8, strides=(1, 1, 1, 1), padding="SAME")
                conv8 = tf.add(conv8, self.bias8)
                conv8 = tf.nn.relu(conv8)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv7.get_shape(), conv8.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv9 = tf.nn.conv2d(conv8, self.weight9, strides=(1, 1, 1, 1), padding="SAME")
                conv9 = tf.add(conv9, self.bias9)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv8.get_shape(), conv9.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv9 = tf.add(conv9, conv6)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv9.get_shape(), conv9.get_shape()))

        # [bottleneck (3)] Input 56x56x24 Output 28x28x32
        # t = 6, c = 32, n = 3, s = 2
        with tf.compat.v1.variable_scope("bottleneck_3") as vs:
            # ------ n = 1 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv10 = tf.nn.conv2d(conv9, self.weight10, strides=(1, 1, 1, 1), padding="SAME")
                conv10 = tf.add(conv10, self.bias10)
                conv10 = tf.nn.relu(conv10)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv9.get_shape(), conv10.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv11 = tf.nn.depthwise_conv2d(conv10, self.weight11, strides=(1, 2, 2, 1), padding="SAME")
                conv11 = tf.add(conv11, self.bias11)
                conv11 = tf.nn.relu(conv11)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv10.get_shape(), conv11.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv12 = tf.nn.conv2d(conv11, self.weight12, strides=(1, 1, 1, 1), padding="SAME")
                conv12 = tf.add(conv12, self.bias12)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv11.get_shape(), conv12.get_shape()))

            # ------ n = 2 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv13 = tf.nn.conv2d(conv12, self.weight13, strides=(1, 1, 1, 1), padding="SAME")
                conv13 = tf.add(conv13, self.bias13)
                conv13 = tf.nn.relu(conv13)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv12.get_shape(), conv13.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv14 = tf.nn.depthwise_conv2d(conv13, self.weight14, strides=(1, 1, 1, 1), padding="SAME")
                conv14 = tf.add(conv14, self.bias14)
                conv14 = tf.nn.relu(conv14)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv13.get_shape(), conv14.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv15 = tf.nn.conv2d(conv14, self.weight15, strides=(1, 1, 1, 1), padding="SAME")
                conv15 = tf.add(conv15, self.bias15)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv14.get_shape(), conv15.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv15 = tf.add(conv15, conv12)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv15.get_shape(), conv15.get_shape()))

            # ------ n = 3 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv16 = tf.nn.conv2d(conv15, self.weight16, strides=(1, 1, 1, 1), padding="SAME")
                conv16 = tf.add(conv16, self.bias16)
                conv16 = tf.nn.relu(conv16)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv15.get_shape(), conv16.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv17 = tf.nn.depthwise_conv2d(conv16, self.weight17, strides=(1, 1, 1, 1), padding="SAME")
                conv17 = tf.add(conv17, self.bias17)
                conv17 = tf.nn.relu(conv17)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv16.get_shape(), conv17.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv18 = tf.nn.conv2d(conv17, self.weight18, strides=(1, 1, 1, 1), padding="SAME")
                conv18 = tf.add(conv18, self.bias18)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv17.get_shape(), conv18.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv18 = tf.add(conv18, conv15)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv18.get_shape(), conv18.get_shape()))

        # [bottleneck (4)] Input 28x28x32 Output 14x14x64
        # t = 6, c = 64, n = 4, s = 2
        with tf.compat.v1.variable_scope("bottleneck_4") as vs:
            # ------ n = 1 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv19 = tf.nn.conv2d(conv18, self.weight19, strides=(1, 1, 1, 1), padding="SAME")
                conv19 = tf.add(conv19, self.bias19)
                conv19 = tf.nn.relu(conv19)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv18.get_shape(), conv19.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv20 = tf.nn.depthwise_conv2d(conv19, self.weight20, strides=(1, 2, 2, 1), padding="SAME")
                conv20 = tf.add(conv20, self.bias20)
                conv20 = tf.nn.relu(conv20)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv19.get_shape(), conv20.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv21 = tf.nn.conv2d(conv20, self.weight21, strides=(1, 1, 1, 1), padding="SAME")
                conv21 = tf.add(conv21, self.bias21)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv20.get_shape(), conv21.get_shape()))

            # ------ n = 2 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv22 = tf.nn.conv2d(conv21, self.weight22, strides=(1, 1, 1, 1), padding="SAME")
                conv22 = tf.add(conv22, self.bias22)
                conv22 = tf.nn.relu(conv22)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv21.get_shape(), conv22.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv23 = tf.nn.depthwise_conv2d(conv22, self.weight23, strides=(1, 1, 1, 1), padding="SAME")
                conv23 = tf.add(conv23, self.bias23)
                conv23 = tf.nn.relu(conv23)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv22.get_shape(), conv23.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv24 = tf.nn.conv2d(conv23, self.weight24, strides=(1, 1, 1, 1), padding="SAME")
                conv24 = tf.add(conv24, self.bias24)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv23.get_shape(), conv24.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv24 = tf.add(conv24, conv21)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv24.get_shape(), conv24.get_shape()))

            # ------ n = 3 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv25 = tf.nn.conv2d(conv24, self.weight25, strides=(1, 1, 1, 1), padding="SAME")
                conv25 = tf.add(conv25, self.bias25)
                conv25 = tf.nn.relu(conv25)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv24.get_shape(), conv25.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv26 = tf.nn.depthwise_conv2d(conv25, self.weight26, strides=(1, 1, 1, 1), padding="SAME")
                conv26 = tf.add(conv26, self.bias26)
                conv26 = tf.nn.relu(conv26)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv25.get_shape(), conv26.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv27 = tf.nn.conv2d(conv26, self.weight27, strides=(1, 1, 1, 1), padding="SAME")
                conv27 = tf.add(conv27, self.bias27)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv26.get_shape(), conv27.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv27 = tf.add(conv27, conv24)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv24.get_shape(), conv24.get_shape()))

            # ------ n = 4 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv28 = tf.nn.conv2d(conv27, self.weight28, strides=(1, 1, 1, 1), padding="SAME")
                conv28 = tf.add(conv28, self.bias28)
                conv28 = tf.nn.relu(conv28)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv27.get_shape(), conv28.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv29 = tf.nn.depthwise_conv2d(conv28, self.weight29, strides=(1, 1, 1, 1), padding="SAME")
                conv29 = tf.add(conv29, self.bias29)
                conv29 = tf.nn.relu(conv29)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv28.get_shape(), conv29.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv30 = tf.nn.conv2d(conv29, self.weight30, strides=(1, 1, 1, 1), padding="SAME")
                conv30 = tf.add(conv30, self.bias30)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv29.get_shape(), conv30.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv30 = tf.add(conv30, conv27)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv30.get_shape(), conv30.get_shape()))

        # [bottleneck (5)] Input 14x14x64 Output 14x14x96
        # t = 6, c = 96, n = 3, s = 1
        with tf.compat.v1.variable_scope("bottleneck_5") as vs:
            # ------ n = 1 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv31 = tf.nn.conv2d(conv30, self.weight31, strides=(1, 1, 1, 1), padding="SAME")
                conv31 = tf.add(conv31, self.bias31)
                conv31 = tf.nn.relu(conv31)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv30.get_shape(), conv31.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv32 = tf.nn.depthwise_conv2d(conv31, self.weight32, strides=(1, 1, 1, 1), padding="SAME")
                conv32 = tf.add(conv32, self.bias32)
                conv32 = tf.nn.relu(conv32)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv31.get_shape(), conv32.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv33 = tf.nn.conv2d(conv32, self.weight33, strides=(1, 1, 1, 1), padding="SAME")
                conv33 = tf.add(conv33, self.bias33)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv32.get_shape(), conv33.get_shape()))

            # ------ n = 2 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv34 = tf.nn.conv2d(conv33, self.weight34, strides=(1, 1, 1, 1), padding="SAME")
                conv34 = tf.add(conv34, self.bias34)
                conv34 = tf.nn.relu(conv34)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv33.get_shape(), conv34.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv35 = tf.nn.depthwise_conv2d(conv34, self.weight35, strides=(1, 1, 1, 1), padding="SAME")
                conv35 = tf.add(conv35, self.bias35)
                conv35 = tf.nn.relu(conv35)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv34.get_shape(), conv35.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv36 = tf.nn.conv2d(conv35, self.weight36, strides=(1, 1, 1, 1), padding="SAME")
                conv36 = tf.add(conv36, self.bias36)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv35.get_shape(), conv36.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv36 = tf.add(conv36, conv33)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv36.get_shape(), conv36.get_shape()))

            # ------ n = 3 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv37 = tf.nn.conv2d(conv36, self.weight37, strides=(1, 1, 1, 1), padding="SAME")
                conv37 = tf.add(conv37, self.bias37)
                conv37 = tf.nn.relu(conv37)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv36.get_shape(), conv37.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv38 = tf.nn.depthwise_conv2d(conv37, self.weight38, strides=(1, 1, 1, 1), padding="SAME")
                conv38 = tf.add(conv38, self.bias38)
                conv38 = tf.nn.relu(conv38)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv37.get_shape(), conv38.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv39 = tf.nn.conv2d(conv38, self.weight39, strides=(1, 1, 1, 1), padding="SAME")
                conv39 = tf.add(conv39, self.bias39)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv38.get_shape(), conv39.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv39 = tf.add(conv39, conv36)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv39.get_shape(), conv39.get_shape()))

        # [bottleneck (6)] Input 14x14x96 Output 7x7x160
        # t = 6, c = 160, n = 3, s = 2
        with tf.compat.v1.variable_scope("bottleneck_6") as vs:
            # ------ n = 1 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv40 = tf.nn.conv2d(conv39, self.weight40, strides=(1, 1, 1, 1), padding="SAME")
                conv40 = tf.add(conv40, self.bias40)
                conv40 = tf.nn.relu(conv40)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv39.get_shape(), conv40.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv41 = tf.nn.depthwise_conv2d(conv40, self.weight41, strides=(1, 2, 2, 1), padding="SAME")
                conv41 = tf.add(conv41, self.bias41)
                conv41 = tf.nn.relu(conv41)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv40.get_shape(), conv41.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv42 = tf.nn.conv2d(conv41, self.weight42, strides=(1, 1, 1, 1), padding="SAME")
                conv42 = tf.add(conv42, self.bias42)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv41.get_shape(), conv42.get_shape()))

            # ------ n = 2 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv43 = tf.nn.conv2d(conv42, self.weight43, strides=(1, 1, 1, 1), padding="SAME")
                conv43 = tf.add(conv43, self.bias43)
                conv43 = tf.nn.relu(conv43)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv42.get_shape(), conv43.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv44 = tf.nn.depthwise_conv2d(conv43, self.weight44, strides=(1, 1, 1, 1), padding="SAME")
                conv44 = tf.add(conv44, self.bias44)
                conv44 = tf.nn.relu(conv44)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv43.get_shape(), conv44.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv45 = tf.nn.conv2d(conv44, self.weight45, strides=(1, 1, 1, 1), padding="SAME")
                conv45 = tf.add(conv45, self.bias45)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv44.get_shape(), conv45.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv45 = tf.add(conv45, conv42)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv45.get_shape(), conv45.get_shape()))

            # ------ n = 3 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv46 = tf.nn.conv2d(conv45, self.weight46, strides=(1, 1, 1, 1), padding="SAME")
                conv46 = tf.add(conv46, self.bias46)
                conv46 = tf.nn.relu(conv46)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv45.get_shape(), conv46.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv47 = tf.nn.depthwise_conv2d(conv46, self.weight47, strides=(1, 1, 1, 1), padding="SAME")
                conv47 = tf.add(conv47, self.bias47)
                conv47 = tf.nn.relu(conv47)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv46.get_shape(), conv47.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv48 = tf.nn.conv2d(conv47, self.weight48, strides=(1, 1, 1, 1), padding="SAME")
                conv48 = tf.add(conv48, self.bias48)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv47.get_shape(), conv48.get_shape()))

            with tf.compat.v1.variable_scope("add") as vsn:
                conv48 = tf.add(conv48, conv45)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv48.get_shape(), conv48.get_shape()))

        # [bottleneck (7)] Input 7x7x160 Output 7x7x320
        # t = 6, c = 320, n = 1, s = 1
        with tf.compat.v1.variable_scope("bottleneck_7") as vs:
            # ------ n = 1 ------
            with tf.compat.v1.variable_scope("expand") as vsn:
                conv49 = tf.nn.conv2d(conv48, self.weight49, strides=(1, 1, 1, 1), padding="SAME")
                conv49 = tf.add(conv49, self.bias49)
                conv49 = tf.nn.relu(conv49)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv48.get_shape(), conv49.get_shape()))

            with tf.compat.v1.variable_scope("depthwise") as vsn:
                conv50 = tf.nn.depthwise_conv2d(conv49, self.weight50, strides=(1, 1, 1, 1), padding="SAME")
                conv50 = tf.add(conv50, self.bias50)
                conv50 = tf.nn.relu(conv50)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv49.get_shape(), conv50.get_shape()))

            with tf.compat.v1.variable_scope("project") as vsn:
                conv51 = tf.nn.conv2d(conv50, self.weight51, strides=(1, 1, 1, 1), padding="SAME")
                conv51 = tf.add(conv51, self.bias51)
                logging.info("{}/{}: Input {} Output {}".format(vs.name, vsn.name, conv50.get_shape(), conv51.get_shape()))

        # [conv2d 1x1] Input 7x7x320 Output 7x7x1280
        # t = ?, c = 1280, n = 1, s = 1
        with tf.compat.v1.variable_scope("conv2d_1x1") as vs:
            conv52 = tf.nn.conv2d(conv51, self.weight52, strides=(1, 1, 1, 1), padding="SAME")
            conv52 = tf.add(conv52, self.bias52)
            conv52 = tf.nn.relu(conv52)
            logging.info("{}: Input {} Output {}".format(vs.name, conv51.get_shape(), conv52.get_shape()))

        # [avgpool 7x7] Input 7x7x1280 Output 1x1x1280
        # t = ?, c = ?, n = 1, s = ?
        with tf.compat.v1.variable_scope("avgpool_7x7") as vs:
            pool53 = tf.nn.avg_pool(conv52, ksize=(1, 3, 3, 1), strides=(1, 7, 7, 1), padding="SAME")
            logging.info("{}: Input {} Output {}".format(vs.name, conv52.get_shape(), pool53.get_shape()))

        # [conv2d 1x1] Input 1x1x1280 Output 1x1xk
        # t = ?, c = k, n = ?, s = ?
        with tf.compat.v1.variable_scope("conv2d_1x1") as vs:
            conv54 = tf.nn.conv2d(pool53, self.weight54, strides=(1, 1, 1, 1), padding="SAME")
            conv54 = tf.nn.bias_add(conv54, self.bias54)
            logging.info("{}: Input {} Output {}".format(vs.name, pool53.get_shape(), conv54.get_shape()))

        # [Squeeze] Input 1x1xk Output 1xk
        # t = ?, c = k, n = ?, s = ?
        with tf.compat.v1.variable_scope("squeeze") as vs:
            squeeze55 = tf.squeeze(conv54, axis=[1, 2])
            logging.info("{}: Input {} Output {}".format(vs.name, conv54.get_shape(), squeeze55.get_shape()))

        return squeeze55


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

    weights = model.get_weights()
    model.save_weights("weights.h5")
    
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

    main(args.save_model_to, args.train)
