#!/usr/bin/python

import tensorflow as tf


def Mobilenet_v2(features, mu=0, sigma=1e-2):
    """
    Mobilnet v2 Architecture with plain tf ops
    """

    print("*** MobileNet v2 Architecture ***")
    # Input ?x?x3 Output 224x224x3
    with tf.compat.v1.variable_scope('P0'):
        # p0 = tf.image.grayscale_to_rgb(features)
        p0 = tf.image.resize(features, (224, 224))
        print("P0: Input {} Output {}".format(features.get_shape(), p0.get_shape()))

    # [conv2d 3x3] Input 224x224x3 Output 112x112x32
    # t = ?, c = 32, n = 1, s = 2
    with tf.compat.v1.variable_scope("C0"):
        weight1 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 3, 32)))
        bias1 = tf.Variable(tf.zeros(shape=(32)))
        conv1 = tf.nn.conv2d(p0, weight1, strides=(1, 2, 2, 1), padding="SAME")
        conv1 = tf.add(conv1, bias1)
        conv1 = tf.nn.relu(conv1)
        print("C0: Input {} Output {}".format(p0.get_shape(), conv1.get_shape()))

    # [bottleneck (1)] Input 112x112x32 Output 112x112x16
    # t = 1, c = 16, n = 1, s = 1
    with tf.compat.v1.variable_scope("B1"):
        # n = 1
        weight2 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 32, 1)))
        bias2 = tf.Variable(tf.zeros(shape=(32)))
        conv2 = tf.nn.depthwise_conv2d(conv1, weight2, strides=(1, 1, 1, 1), padding="SAME")
        conv2 = tf.add(conv2, bias2)
        conv2 = tf.nn.relu(conv2)
        print("B1/1: Input {} Output {}".format(conv1.get_shape(), conv2.get_shape()))

        weight3 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 16)))
        bias3 = tf.Variable(tf.zeros(shape=(16)))
        conv3 = tf.nn.conv2d(conv2, weight3, strides=(1, 1, 1, 1), padding="SAME")
        conv3 = tf.add(conv3, bias3)
        # conv3 = tf.nn.relu(conv3)
        print("B1/2: Input {} Output {}".format(conv2.get_shape(), conv3.get_shape()))

    # [bottleneck (2)] Input 112x112x16 Output 56x56x24
    # t = 6, c = 24, n = 2, s = 2
    with tf.compat.v1.variable_scope("B2"):
        # ------ n = 1 ------
        # expand
        weight4 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 16, 96)))
        bias4 = tf.Variable(tf.zeros(shape=(96)))
        conv4 = tf.nn.conv2d(conv3, weight4, strides=(1, 1, 1, 1), padding="SAME")
        conv4 = tf.add(conv4, bias4)
        conv4 = tf.nn.relu(conv4)
        print("B2/1: Input {} Output {}".format(conv3.get_shape(), conv4.get_shape()))

        # depthwise
        weight5 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 96, 1)))
        bias5 = tf.Variable(tf.zeros(shape=(96)))
        conv5 = tf.nn.depthwise_conv2d(conv4, weight5, strides=(1, 2, 2, 1), padding="SAME")
        conv5 = tf.add(conv5, bias5)
        conv5 = tf.nn.relu(conv5)
        print("B2/2: Input {} Output {}".format(conv4.get_shape(), conv5.get_shape()))

        # conv
        weight6 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 24)))
        bias6 = tf.Variable(tf.zeros(shape=(24)))
        conv6 = tf.nn.conv2d(conv5, weight6, strides=(1, 1, 1, 1), padding="SAME")
        conv6 = tf.add(conv6, bias6)
        print("B2/3: Input {} Output {}".format(conv5.get_shape(), conv6.get_shape()))

        # ------ n = 2 ------
        # expand
        weight7 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144)))
        bias7 = tf.Variable(tf.zeros(shape=(144)))
        conv7 = tf.nn.conv2d(conv6, weight7, strides=(1, 1, 1, 1), padding="SAME")
        conv7 = tf.add(conv7, bias7)
        conv7 = tf.nn.relu(conv7)
        print("B2/4: Input {} Output {}".format(conv6.get_shape(), conv7.get_shape()))

        # depthwise
        weight8 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1)))
        bias8 = tf.Variable(tf.zeros(shape=(144)))
        conv8 = tf.nn.depthwise_conv2d(conv7, weight8, strides=(1, 1, 1, 1), padding="SAME")
        conv8 = tf.add(conv8, bias8)
        conv8 = tf.nn.relu(conv8)
        print("B2/5: Input {} Output {}".format(conv7.get_shape(), conv8.get_shape()))

        # conv
        weight9 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 24)))
        bias9 = tf.Variable(tf.zeros(shape=(24)))
        conv9 = tf.nn.conv2d(conv8, weight9, strides=(1, 1, 1, 1), padding="SAME")
        conv9 = tf.add(conv9, bias9)
        print("B2/6: Input {} Output {}".format(conv8.get_shape(), conv9.get_shape()))

        # add
        conv9 = tf.add(conv9, conv6)
        print("B2/7: Input {} Output {}".format(conv9.get_shape(), conv9.get_shape()))

    # [bottleneck (3)] Input 56x56x24 Output 28x28x32
    # t = 6, c = 32, n = 3, s = 2
    with tf.compat.v1.variable_scope("B3"):
        # ------ n = 1 ------
        # expand
        weight10 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144)))
        bias10 = tf.Variable(tf.zeros(shape=(144)))
        conv10 = tf.nn.conv2d(conv9, weight10, strides=(1, 1, 1, 1), padding="SAME")
        conv10 = tf.add(conv10, bias10)
        conv10 = tf.nn.relu(conv10)
        print("B3/1: Input {} Output {}".format(conv9.get_shape(), conv10.get_shape()))

        # depthwise
        weight11 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1)))
        bias11 = tf.Variable(tf.zeros(shape=(144)))
        conv11 = tf.nn.depthwise_conv2d(conv10, weight11, strides=(1, 2, 2, 1), padding="SAME")
        conv11 = tf.add(conv11, bias11)
        conv11 = tf.nn.relu(conv11)
        print("B3/2: Input {} Output {}".format(conv10.get_shape(), conv11.get_shape()))

        # conv
        weight12 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 32)))
        bias12 = tf.Variable(tf.zeros(shape=(32)))
        conv12 = tf.nn.conv2d(conv11, weight12, strides=(1, 1, 1, 1), padding="SAME")
        conv12 = tf.add(conv12, bias12)
        print("B3/3: Input {} Output {}".format(conv11.get_shape(), conv12.get_shape()))

        # ------ n = 2 ------
        # expand
        weight13 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192)))
        bias13 = tf.Variable(tf.zeros(shape=(192)))
        conv13 = tf.nn.conv2d(conv12, weight13, strides=(1, 1, 1, 1), padding="SAME")
        conv13 = tf.add(conv13, bias13)
        conv13 = tf.nn.relu(conv13)
        print("B3/4: Input {} Output {}".format(conv12.get_shape(), conv13.get_shape()))

        # depthwise
        weight14 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1)))
        bias14 = tf.Variable(tf.zeros(shape=(192)))
        conv14 = tf.nn.depthwise_conv2d(conv13, weight14, strides=(1, 1, 1, 1), padding="SAME")
        conv14 = tf.add(conv14, bias14)
        conv14 = tf.nn.relu(conv14)
        print("B3/5: Input {} Output {}".format(conv13.get_shape(), conv14.get_shape()))

        # conv
        weight15 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 32)))
        bias15 = tf.Variable(tf.zeros(shape=(32)))
        conv15 = tf.nn.conv2d(conv14, weight15, strides=(1, 1, 1, 1), padding="SAME")
        conv15 = tf.add(conv15, bias15)
        print("B3/6: Input {} Output {}".format(conv14.get_shape(), conv15.get_shape()))

        # add
        conv15 = tf.add(conv15, conv12)
        print("B3/7: Input {} Output {}".format(conv15.get_shape(), conv15.get_shape()))

        # ------ n = 3 ------
        # expand
        weight16 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192)))
        bias16 = tf.Variable(tf.zeros(shape=(192)))
        conv16 = tf.nn.conv2d(conv15, weight16, strides=(1, 1, 1, 1), padding="SAME")
        conv16 = tf.add(conv16, bias16)
        conv16 = tf.nn.relu(conv16)
        print("B3/8: Input {} Output {}".format(conv15.get_shape(), conv16.get_shape()))

        # depthwise
        weight17 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1)))
        bias17 = tf.Variable(tf.zeros(shape=(192)))
        conv17 = tf.nn.depthwise_conv2d(conv16, weight17, strides=(1, 1, 1, 1), padding="SAME")
        conv17 = tf.add(conv17, bias17)
        conv17 = tf.nn.relu(conv17)
        print("B3/9: Input {} Output {}".format(conv16.get_shape(), conv17.get_shape()))

        # conv
        weight18 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 32)))
        bias18 = tf.Variable(tf.zeros(shape=(32)))
        conv18 = tf.nn.conv2d(conv17, weight18, strides=(1, 1, 1, 1), padding="SAME")
        conv18 = tf.add(conv18, bias18)
        print("B3/10: Input {} Output {}".format(conv17.get_shape(), conv18.get_shape()))

        # add
        conv18 = tf.add(conv18, conv15)
        print("B3/11: Input {} Output {}".format(conv18.get_shape(), conv18.get_shape()))

    # [bottleneck (4)] Input 28x28x32 Output 14x14x64
    # t = 6, c = 64, n = 4, s = 2
    with tf.compat.v1.variable_scope("B4"):
        # ------ n = 1 ------
        # expand
        weight19 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192)))
        bias19 = tf.Variable(tf.zeros(shape=(192)))
        conv19 = tf.nn.conv2d(conv18, weight19, strides=(1, 1, 1, 1), padding="SAME")
        conv19 = tf.add(conv19, bias19)
        conv19 = tf.nn.relu(conv19)
        print("B4/1: Input {} Output {}".format(conv18.get_shape(), conv19.get_shape()))

        # depthwise
        weight20 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1)))
        bias20 = tf.Variable(tf.zeros(shape=(192)))
        conv20 = tf.nn.depthwise_conv2d(conv19, weight20, strides=(1, 2, 2, 1), padding="SAME")
        conv20 = tf.add(conv20, bias20)
        conv20 = tf.nn.relu(conv20)
        print("B4/2: Input {} Output {}".format(conv19.get_shape(), conv20.get_shape()))

        # conv
        weight21 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 64)))
        bias21 = tf.Variable(tf.zeros(shape=(64)))
        conv21 = tf.nn.conv2d(conv20, weight21, strides=(1, 1, 1, 1), padding="SAME")
        conv21 = tf.add(conv21, bias21)
        print("B4/3: Input {} Output {}".format(conv20.get_shape(), conv21.get_shape()))

        # ------ n = 2 ------
        # expand
        weight22 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384)))
        bias22 = tf.Variable(tf.zeros(shape=(384)))
        conv22 = tf.nn.conv2d(conv21, weight22, strides=(1, 1, 1, 1), padding="SAME")
        conv22 = tf.add(conv22, bias22)
        conv22 = tf.nn.relu(conv22)
        print("B4/4: Input {} Output {}".format(conv21.get_shape(), conv22.get_shape()))

        # depthwise
        weight23 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1)))
        bias23 = tf.Variable(tf.zeros(shape=(384)))
        conv23 = tf.nn.depthwise_conv2d(conv22, weight23, strides=(1, 1, 1, 1), padding="SAME")
        conv23 = tf.add(conv23, bias23)
        conv23 = tf.nn.relu(conv23)
        print("B4/5: Input {} Output {}".format(conv22.get_shape(), conv23.get_shape()))

        # conv
        weight24 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64)))
        bias24 = tf.Variable(tf.zeros(shape=(64)))
        conv24 = tf.nn.conv2d(conv23, weight24, strides=(1, 1, 1, 1), padding="SAME")
        conv24 = tf.add(conv24, bias24)
        print("B4/6: Input {} Output {}".format(conv23.get_shape(), conv24.get_shape()))

        # add
        conv24 = tf.add(conv24, conv21)
        print("B4/7: Input {} Output {}".format(conv24.get_shape(), conv24.get_shape()))

        # ------ n = 3 ------
        # expand
        weight25 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384)))
        bias25 = tf.Variable(tf.zeros(shape=(384)))
        conv25 = tf.nn.conv2d(conv24, weight25, strides=(1, 1, 1, 1), padding="SAME")
        conv25 = tf.add(conv25, bias25)
        conv25 = tf.nn.relu(conv25)
        print("B4/8: Input {} Output {}".format(conv24.get_shape(), conv25.get_shape()))

        # depthwise
        weight26 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1)))
        bias26 = tf.Variable(tf.zeros(shape=(384)))
        conv26 = tf.nn.depthwise_conv2d(conv25, weight26, strides=(1, 1, 1, 1), padding="SAME")
        conv26 = tf.add(conv26, bias26)
        conv26 = tf.nn.relu(conv26)
        print("B4/9: Input {} Output {}".format(conv25.get_shape(), conv26.get_shape()))

        # conv
        weight27 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64)))
        bias27 = tf.Variable(tf.zeros(shape=(64)))
        conv27 = tf.nn.conv2d(conv26, weight27, strides=(1, 1, 1, 1), padding="SAME")
        conv27 = tf.add(conv27, bias27)
        print("B4/10: Input {} Output {}".format(conv26.get_shape(), conv27.get_shape()))

        # add
        conv27 = tf.add(conv27, conv24)
        print("B4/11: Input {} Output {}".format(conv24.get_shape(), conv24.get_shape()))

        # ------ n = 4 ------
        # expand
        weight28 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384)))
        bias28 = tf.Variable(tf.zeros(shape=(384)))
        conv28 = tf.nn.conv2d(conv27, weight28, strides=(1, 1, 1, 1), padding="SAME")
        conv28 = tf.add(conv28, bias28)
        conv28 = tf.nn.relu(conv28)
        print("B4/12: Input {} Output {}".format(conv27.get_shape(), conv28.get_shape()))

        # depthwise
        weight29 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1)))
        bias29 = tf.Variable(tf.zeros(shape=(384)))
        conv29 = tf.nn.depthwise_conv2d(conv28, weight29, strides=(1, 1, 1, 1), padding="SAME")
        conv29 = tf.add(conv29, bias29)
        conv29 = tf.nn.relu(conv29)
        print("B4/13: Input {} Output {}".format(conv28.get_shape(), conv29.get_shape()))

        # conv
        weight30 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64)))
        bias30 = tf.Variable(tf.zeros(shape=(64)))
        conv30 = tf.nn.conv2d(conv29, weight30, strides=(1, 1, 1, 1), padding="SAME")
        conv30 = tf.add(conv30, bias30)
        print("B4/14: Input {} Output {}".format(conv29.get_shape(), conv30.get_shape()))

        # add
        conv30 = tf.add(conv30, conv27)
        print("B4/15: Input {} Output {}".format(conv30.get_shape(), conv30.get_shape()))

    # [bottleneck (5)] Input 14x14x64 Output 14x14x96
    # t = 6, c = 96, n = 3, s = 1
    with tf.compat.v1.variable_scope("B5"):
        # ------ n = 1 ------
        # expand
        weight31 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384)))
        bias31 = tf.Variable(tf.zeros(shape=(384)))
        conv31 = tf.nn.conv2d(conv30, weight31, strides=(1, 1, 1, 1), padding="SAME")
        conv31 = tf.add(conv31, bias31)
        conv31 = tf.nn.relu(conv31)
        print("B5/1: Input {} Output {}".format(conv30.get_shape(), conv31.get_shape()))

        # depthwise
        weight32 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1)))
        bias32 = tf.Variable(tf.zeros(shape=(384)))
        conv32 = tf.nn.depthwise_conv2d(conv31, weight32, strides=(1, 1, 1, 1), padding="SAME")
        conv32 = tf.add(conv32, bias32)
        conv32 = tf.nn.relu(conv32)
        print("B5/2: Input {} Output {}".format(conv31.get_shape(), conv32.get_shape()))

        # conv
        weight33 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 96)))
        bias33 = tf.Variable(tf.zeros(shape=(96)))
        conv33 = tf.nn.conv2d(conv32, weight33, strides=(1, 1, 1, 1), padding="SAME")
        conv33 = tf.add(conv33, bias33)
        print("B5/3: Input {} Output {}".format(conv32.get_shape(), conv33.get_shape()))

        # ------ n = 2 ------
        # expand
        weight34 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 576)))
        bias34 = tf.Variable(tf.zeros(shape=(576)))
        conv34 = tf.nn.conv2d(conv33, weight34, strides=(1, 1, 1, 1), padding="SAME")
        conv34 = tf.add(conv34, bias34)
        conv34 = tf.nn.relu(conv34)
        print("B5/4: Input {} Output {}".format(conv33.get_shape(), conv34.get_shape()))

        # depthwise
        weight35 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 576, 1)))
        bias35 = tf.Variable(tf.zeros(shape=(576)))
        conv35 = tf.nn.depthwise_conv2d(conv34, weight35, strides=(1, 1, 1, 1), padding="SAME")
        conv35 = tf.add(conv35, bias35)
        conv35 = tf.nn.relu(conv35)
        print("B5/5: Input {} Output {}".format(conv34.get_shape(), conv35.get_shape()))

        # conv
        weight36 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 576, 96)))
        bias36 = tf.Variable(tf.zeros(shape=(96)))
        conv36 = tf.nn.conv2d(conv35, weight36, strides=(1, 1, 1, 1), padding="SAME")
        conv36 = tf.add(conv36, bias36)
        print("B5/6: Input {} Output {}".format(conv35.get_shape(), conv36.get_shape()))

        # add
        conv36 = tf.add(conv36, conv33)
        print("B5/7: Input {} Output {}".format(conv36.get_shape(), conv36.get_shape()))

        # ------ n = 3 ------
        # expand
        weight37 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 576)))
        bias37 = tf.Variable(tf.zeros(shape=(576)))
        conv37 = tf.nn.conv2d(conv36, weight37, strides=(1, 1, 1, 1), padding="SAME")
        conv37 = tf.add(conv37, bias37)
        conv37 = tf.nn.relu(conv37)
        print("B5/8: Input {} Output {}".format(conv36.get_shape(), conv37.get_shape()))

        # depthwise
        weight38 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 576, 1)))
        bias38 = tf.Variable(tf.zeros(shape=(576)))
        conv38 = tf.nn.depthwise_conv2d(conv37, weight38, strides=(1, 1, 1, 1), padding="SAME")
        conv38 = tf.add(conv38, bias38)
        conv38 = tf.nn.relu(conv38)
        print("B5/9: Input {} Output {}".format(conv37.get_shape(), conv38.get_shape()))

        # conv
        weight39 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 576, 96)))
        bias39 = tf.Variable(tf.zeros(shape=(96)))
        conv39 = tf.nn.conv2d(conv38, weight39, strides=(1, 1, 1, 1), padding="SAME")
        conv39 = tf.add(conv39, bias39)
        print("B5/10: Input {} Output {}".format(conv38.get_shape(), conv39.get_shape()))

        # add
        conv39 = tf.add(conv39, conv36)
        print("B5/11: Input {} Output {}".format(conv39.get_shape(), conv39.get_shape()))

    # [bottleneck (6)] Input 14x14x96 Output 7x7x160
    # t = 6, c = 160, n = 3, s = 2
    with tf.compat.v1.variable_scope("B6"):
        # ------ n = 1 ------
        # expand
        weight40 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 576)))
        bias40 = tf.Variable(tf.zeros(shape=(576)))
        conv40 = tf.nn.conv2d(conv39, weight40, strides=(1, 1, 1, 1), padding="SAME")
        conv40 = tf.add(conv40, bias40)
        conv40 = tf.nn.relu(conv40)
        print("B6/1: Input {} Output {}".format(conv39.get_shape(), conv40.get_shape()))

        # depthwise
        weight41 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 576, 1)))
        bias41 = tf.Variable(tf.zeros(shape=(576)))
        conv41 = tf.nn.depthwise_conv2d(conv40, weight41, strides=(1, 2, 2, 1), padding="SAME")
        conv41 = tf.add(conv41, bias41)
        conv41 = tf.nn.relu(conv41)
        print("B6/2: Input {} Output {}".format(conv40.get_shape(), conv41.get_shape()))

        # conv
        weight42 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 576, 160)))
        bias42 = tf.Variable(tf.zeros(shape=(160)))
        conv42 = tf.nn.conv2d(conv41, weight42, strides=(1, 1, 1, 1), padding="SAME")
        conv42 = tf.add(conv42, bias42)
        print("B6/3: Input {} Output {}".format(conv41.get_shape(), conv42.get_shape()))

        # ------ n = 2 ------
        # expand
        weight43 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 160, 960)))
        bias43 = tf.Variable(tf.zeros(shape=(960)))
        conv43 = tf.nn.conv2d(conv42, weight43, strides=(1, 1, 1, 1), padding="SAME")
        conv43 = tf.add(conv43, bias43)
        conv43 = tf.nn.relu(conv43)
        print("B6/4: Input {} Output {}".format(conv42.get_shape(), conv43.get_shape()))

        # depthwise
        weight44 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 960, 1)))
        bias44 = tf.Variable(tf.zeros(shape=(960)))
        conv44 = tf.nn.depthwise_conv2d(conv43, weight44, strides=(1, 1, 1, 1), padding="SAME")
        conv44 = tf.add(conv44, bias44)
        conv44 = tf.nn.relu(conv44)
        print("B6/5: Input {} Output {}".format(conv43.get_shape(), conv44.get_shape()))

        # conv
        weight45 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 960, 160)))
        bias45 = tf.Variable(tf.zeros(shape=(160)))
        conv45 = tf.nn.conv2d(conv44, weight45, strides=(1, 1, 1, 1), padding="SAME")
        conv45 = tf.add(conv45, bias45)
        print("B6/6: Input {} Output {}".format(conv44.get_shape(), conv45.get_shape()))

        # add
        conv45 = tf.add(conv45, conv42)
        print("B6/7: Input {} Output {}".format(conv45.get_shape(), conv45.get_shape()))

        # ------ n = 3 ------
        # expand
        weight46 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 160, 960)))
        bias46 = tf.Variable(tf.zeros(shape=(960)))
        conv46 = tf.nn.conv2d(conv45, weight46, strides=(1, 1, 1, 1), padding="SAME")
        conv46 = tf.add(conv46, bias46)
        conv46 = tf.nn.relu(conv46)
        print("B6/8: Input {} Output {}".format(conv45.get_shape(), conv46.get_shape()))

        # depthwise
        weight47 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 960, 1)))
        bias47 = tf.Variable(tf.zeros(shape=(960)))
        conv47 = tf.nn.depthwise_conv2d(conv46, weight47, strides=(1, 1, 1, 1), padding="SAME")
        conv47 = tf.add(conv47, bias47)
        conv47 = tf.nn.relu(conv47)
        print("B6/9: Input {} Output {}".format(conv46.get_shape(), conv47.get_shape()))

        # conv
        weight48 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 960, 160)))
        bias48 = tf.Variable(tf.zeros(shape=(160)))
        conv48 = tf.nn.conv2d(conv47, weight48, strides=(1, 1, 1, 1), padding="SAME")
        conv48 = tf.add(conv48, bias48)
        print("B6/10: Input {} Output {}".format(conv47.get_shape(), conv48.get_shape()))

        # add
        conv48 = tf.add(conv48, conv45)
        print("B6/11: Input {} Output {}".format(conv48.get_shape(), conv48.get_shape()))

    # [bottleneck (7)] Input 7x7x160 Output 7x7x320
    # t = 6, c = 320, n = 1, s = 1
    with tf.compat.v1.variable_scope("B7"):
        # ------ n = 1 ------
        # expand
        weight49 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 160, 960)))
        bias49 = tf.Variable(tf.zeros(shape=(960)))
        conv49 = tf.nn.conv2d(conv48, weight49, strides=(1, 1, 1, 1), padding="SAME")
        conv49 = tf.add(conv49, bias49)
        conv49 = tf.nn.relu(conv49)
        print("B7/1: Input {} Output {}".format(conv48.get_shape(), conv49.get_shape()))

        # depthwise
        weight50 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 960, 1)))
        bias50 = tf.Variable(tf.zeros(shape=(960)))
        conv50 = tf.nn.depthwise_conv2d(conv49, weight50, strides=(1, 1, 1, 1), padding="SAME")
        conv50 = tf.add(conv50, bias50)
        conv50 = tf.nn.relu(conv50)
        print("B7/2: Input {} Output {}".format(conv49.get_shape(), conv50.get_shape()))

        # conv
        weight51 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 960, 320)))
        bias51 = tf.Variable(tf.zeros(shape=(320)))
        conv51 = tf.nn.conv2d(conv50, weight51, strides=(1, 1, 1, 1), padding="SAME")
        conv51 = tf.add(conv51, bias51)
        print("B7/3: Input {} Output {}".format(conv50.get_shape(), conv51.get_shape()))

    # [conv2d 1x1] Input 7x7x320 Output 7x7x1280
    # t = ?, c = 1280, n = 1, s = 1
    with tf.compat.v1.variable_scope("C8"):
        weight52 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 320, 1280)))
        bias52 = tf.Variable(tf.zeros(shape=(1280)))
        conv52 = tf.nn.conv2d(conv51, weight52, strides=(1, 1, 1, 1), padding="SAME")
        conv52 = tf.add(conv52, bias52)
        conv52 = tf.nn.relu(conv52)
        print("C8: Input {} Output {}".format(conv51.get_shape(), conv52.get_shape()))

    # [avgpool 7x7] Input 7x7x1280 Output 1x1x1280
    # t = ?, c = ?, n = 1, s = ?
    with tf.compat.v1.variable_scope("S9"):
        pool53 = tf.nn.avg_pool(conv52, ksize=(1, 3, 3, 1), strides=(1, 7, 7, 1), padding="SAME")
        print("S9: Input {} Output {}".format(conv52.get_shape(), pool53.get_shape()))

    # [conv2d 1x1] Input 1x1x1280 Output 1x1xk
    # t = ?, c = k, n = ?, s = ?
    with tf.compat.v1.variable_scope("C10"):
        weight54 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 1280, 1001)))
        bias54 = tf.Variable(tf.zeros(shape=(1001)))
        conv54 = tf.nn.conv2d(pool53, weight54, strides=(1, 1, 1, 1), padding="SAME")
        conv54 = tf.nn.bias_add(conv54, bias54)
        print("C10: Input {} Output {}".format(pool53.get_shape(), conv54.get_shape()))

    # [Squeeze] Input 1x1xk Output 1xk
    # t = ?, c = k, n = ?, s = ?
    with tf.compat.v1.variable_scope("M11"):
        squeeze55 = tf.squeeze(conv54, axis=[1, 2])
        print("M11: Input {} Output {}".format(conv54.get_shape(), squeeze55.get_shape()))

    return squeeze55


def normalize_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.subtract(tf.divide(image, 127.5), 1)
    image = tf.image.resize(image, (224, 224))
    return image


def load_image(path, normalize=True):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.expand_dims(image, 0)
    if normalize:
        image = tf.cast(image, tf.float32)
        image = tf.subtract(tf.divide(image, 127.5), 1)
    image = tf.image.resize(image, (224, 224))
    return image


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="./grace_hopper.jpg", help="Input Image path")
    args = parser.parse_args()

    image = load_image(args.image)

    classification = Mobilenet_v2(image)

    print(classification)
