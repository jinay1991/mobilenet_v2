"""
Copyright (c) 2020. All Rights Reserved.
"""

import tensorflow as tf

def mobilenet_v2(features):
    mu = 0
    sigma = 1e-2

    # Input ?x?x3 Output 224x224x3
    with tf.compat.v1.variable_scope('P0'):
        p0 = tf.image.resize(features, (224, 224))
        print("P0: Input {} Output {}".format(features.get_shape(), p0.get_shape()))

    # [Conv2d] Input 224x224x3 Output 112x112x32
    with tf.compat.v1.variable_scope("C1"):
        weight1 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 3, 32), mean=mu, stddev=sigma))
        bias1 = tf.Variable(tf.zeros(shape=(32)))
        conv1 = tf.nn.conv2d(p0, weight1, strides=(1, 2, 2, 1), padding="SAME")
        conv1 = tf.add(conv1, bias1)
        conv1 = tf.nn.relu(conv1)
        print("C1: Input {} Output {}".format(p0.get_shape(), conv1.get_shape()))

    # [bottleneck (1)] Input 112x112x32 Output 112x112x16
    # t = 1, c = 16, n = 1, s = 1
    with tf.compat.v1.variable_scope("B1"):
        # n = 1
        weight2 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 32, 1), mean=mu, stddev=sigma))
        bias2 = tf.Variable(tf.zeros(shape=(32)))
        conv2 = tf.nn.depthwise_conv2d(conv1, weight2, strides=(1, 1, 1, 1), padding="SAME")
        conv2 = tf.add(conv2, bias2)
        conv2 = tf.nn.relu(conv2)
        print("B1/1: Input {} Output {}".format(conv1.get_shape(), conv2.get_shape()))

        weight3 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 16), mean=mu, stddev=sigma))
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
        weight4 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 16, 96), mean=mu, stddev=sigma))
        bias4 = tf.Variable(tf.zeros(shape=(96)))
        conv4 = tf.nn.conv2d(conv3, weight4, strides=(1, 1, 1, 1), padding="SAME")
        conv4 = tf.add(conv4, bias4)
        conv4 = tf.nn.relu(conv4)
        print("B2/1: Input {} Output {}".format(conv3.get_shape(), conv4.get_shape()))

        # depthwise
        weight5 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 96, 1), mean=mu, stddev=sigma))
        bias5 = tf.Variable(tf.zeros(shape=(96)))
        conv5 = tf.nn.depthwise_conv2d(conv4, weight5, strides=(1, 2, 2, 1), padding="SAME")
        conv5 = tf.add(conv5, bias5)
        conv5 = tf.nn.relu(conv5)
        print("B2/2: Input {} Output {}".format(conv4.get_shape(), conv5.get_shape()))

        # conv
        weight6 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 96, 24), mean=mu, stddev=sigma))
        bias6 = tf.Variable(tf.zeros(shape=(24)))
        conv6 = tf.nn.conv2d(conv5, weight6, strides=(1, 1, 1, 1), padding="SAME")
        conv6 = tf.add(conv6, bias6)
        print("B2/3: Input {} Output {}".format(conv5.get_shape(), conv6.get_shape()))

        # ------ n = 2 ------
        # expand
        weight7 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144), mean=mu, stddev=sigma))
        bias7 = tf.Variable(tf.zeros(shape=(144)))
        conv7 = tf.nn.conv2d(conv6, weight7, strides=(1, 1, 1, 1), padding="SAME")
        conv7 = tf.add(conv7, bias7)
        conv7 = tf.nn.relu(conv7)
        print("B2/4: Input {} Output {}".format(conv6.get_shape(), conv7.get_shape()))

        # depthwise
        weight8 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1), mean=mu, stddev=sigma))
        bias8 = tf.Variable(tf.zeros(shape=(144)))
        conv8 = tf.nn.depthwise_conv2d(conv7, weight8, strides=(1, 1, 1, 1), padding="SAME")
        conv8 = tf.add(conv8, bias8)
        conv8 = tf.nn.relu(conv8)
        print("B2/5: Input {} Output {}".format(conv7.get_shape(), conv8.get_shape()))

        # conv
        weight9 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 24), mean=mu, stddev=sigma))
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
        weight10 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144), mean=mu, stddev=sigma))
        bias10 = tf.Variable(tf.zeros(shape=(144)))
        conv10 = tf.nn.conv2d(conv9, weight10, strides=(1, 1, 1, 1), padding="SAME")
        conv10 = tf.add(conv10, bias10)
        conv10 = tf.nn.relu(conv10)
        print("B3/1: Input {} Output {}".format(conv9.get_shape(), conv10.get_shape()))

        # depthwise
        weight11 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1), mean=mu, stddev=sigma))
        bias11 = tf.Variable(tf.zeros(shape=(144)))
        conv11 = tf.nn.depthwise_conv2d(conv10, weight11, strides=(1, 2, 2, 1), padding="SAME")
        conv11 = tf.add(conv11, bias11)
        conv11 = tf.nn.relu(conv11)
        print("B3/2: Input {} Output {}".format(conv10.get_shape(), conv11.get_shape()))

        # conv
        weight12 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 32), mean=mu, stddev=sigma))
        bias12 = tf.Variable(tf.zeros(shape=(32)))
        conv12 = tf.nn.conv2d(conv11, weight12, strides=(1, 1, 1, 1), padding="SAME")
        conv12 = tf.add(conv12, bias12)
        print("B3/3: Input {} Output {}".format(conv11.get_shape(), conv12.get_shape()))

        # ------ n = 2 ------
        # expand
        weight13 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192), mean=mu, stddev=sigma))
        bias13 = tf.Variable(tf.zeros(shape=(192)))
        conv13 = tf.nn.conv2d(conv12, weight13, strides=(1, 1, 1, 1), padding="SAME")
        conv13 = tf.add(conv13, bias13)
        conv13 = tf.nn.relu(conv13)
        print("B3/4: Input {} Output {}".format(conv12.get_shape(), conv13.get_shape()))

        # depthwise
        weight14 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1), mean=mu, stddev=sigma))
        bias14 = tf.Variable(tf.zeros(shape=(192)))
        conv14 = tf.nn.depthwise_conv2d(conv13, weight14, strides=(1, 1, 1, 1), padding="SAME")
        conv14 = tf.add(conv14, bias14)
        conv14 = tf.nn.relu(conv14)
        print("B3/5: Input {} Output {}".format(conv13.get_shape(), conv14.get_shape()))

        # conv
        weight15 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 32), mean=mu, stddev=sigma))
        bias15 = tf.Variable(tf.zeros(shape=(32)))
        conv15 = tf.nn.conv2d(conv14, weight15, strides=(1, 1, 1, 1), padding="SAME")
        conv15 = tf.add(conv15, bias15)
        print("B3/6: Input {} Output {}".format(conv14.get_shape(), conv15.get_shape()))

        # add
        conv15 = tf.add(conv15, conv12)
        print("B3/7: Input {} Output {}".format(conv15.get_shape(), conv15.get_shape()))

        # ------ n = 3 ------
        # expand
        weight16 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192), mean=mu, stddev=sigma))
        bias16 = tf.Variable(tf.zeros(shape=(192)))
        conv16 = tf.nn.conv2d(conv15, weight16, strides=(1, 1, 1, 1), padding="SAME")
        conv16 = tf.add(conv16, bias16)
        conv16 = tf.nn.relu(conv16)
        print("B3/8: Input {} Output {}".format(conv15.get_shape(), conv16.get_shape()))

        # depthwise
        weight17 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1), mean=mu, stddev=sigma))
        bias17 = tf.Variable(tf.zeros(shape=(192)))
        conv17 = tf.nn.depthwise_conv2d(conv16, weight17, strides=(1, 1, 1, 1), padding="SAME")
        conv17 = tf.add(conv17, bias17)
        conv17 = tf.nn.relu(conv17)
        print("B3/9: Input {} Output {}".format(conv16.get_shape(), conv17.get_shape()))

        # conv
        weight18 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 32), mean=mu, stddev=sigma))
        bias18 = tf.Variable(tf.zeros(shape=(32)))
        conv18 = tf.nn.conv2d(conv17, weight18, strides=(1, 1, 1, 1), padding="SAME")
        conv18 = tf.add(conv18, bias18)
        print("B3/10: Input {} Output {}".format(conv17.get_shape(), conv18.get_shape()))

        # add
        conv18 = tf.add(conv18, conv15)
        print("B3/11: Input {} Output {}".format(conv18.get_shape(), conv18.get_shape()))


    # [bottleneck (3)] Input 28x28x32 Output 14x14x64
    # t = 6, c = 64, n = 4, s = 2
    with tf.compat.v1.variable_scope("B4"):
        # ------ n = 1 ------
        # expand
        weight19 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 32, 192), mean=mu, stddev=sigma))
        bias19 = tf.Variable(tf.zeros(shape=(192)))
        conv19 = tf.nn.conv2d(conv18, weight19, strides=(1, 1, 1, 1), padding="SAME")
        conv19 = tf.add(conv19, bias19)
        conv19 = tf.nn.relu(conv19)
        print("B4/1: Input {} Output {}".format(conv18.get_shape(), conv19.get_shape()))

        # depthwise
        weight20 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 192, 1), mean=mu, stddev=sigma))
        bias20 = tf.Variable(tf.zeros(shape=(192)))
        conv20 = tf.nn.depthwise_conv2d(conv19, weight20, strides=(1, 2, 2, 1), padding="SAME")
        conv20 = tf.add(conv20, bias20)
        conv20 = tf.nn.relu(conv20)
        print("B4/2: Input {} Output {}".format(conv19.get_shape(), conv20.get_shape()))

        # conv
        weight21 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 192, 64), mean=mu, stddev=sigma))
        bias21 = tf.Variable(tf.zeros(shape=(64)))
        conv21 = tf.nn.conv2d(conv20, weight21, strides=(1, 1, 1, 1), padding="SAME")
        conv21 = tf.add(conv21, bias21)
        print("B4/3: Input {} Output {}".format(conv20.get_shape(), conv21.get_shape()))

        # ------ n = 2 ------
        # expand
        weight22 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        bias22 = tf.Variable(tf.zeros(shape=(384)))
        conv22 = tf.nn.conv2d(conv21, weight22, strides=(1, 1, 1, 1), padding="SAME")
        conv22 = tf.add(conv22, bias22)
        conv22 = tf.nn.relu(conv22)
        print("B4/4: Input {} Output {}".format(conv21.get_shape(), conv22.get_shape()))

        # depthwise
        weight23 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        bias23 = tf.Variable(tf.zeros(shape=(384)))
        conv23 = tf.nn.depthwise_conv2d(conv22, weight23, strides=(1, 1, 1, 1), padding="SAME")
        conv23 = tf.add(conv23, bias23)
        conv23 = tf.nn.relu(conv23)
        print("B4/5: Input {} Output {}".format(conv22.get_shape(), conv23.get_shape()))

        # conv
        weight24 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64), mean=mu, stddev=sigma))
        bias24 = tf.Variable(tf.zeros(shape=(64)))
        conv24 = tf.nn.conv2d(conv23, weight24, strides=(1, 1, 1, 1), padding="SAME")
        conv24 = tf.add(conv24, bias24)
        print("B4/6: Input {} Output {}".format(conv23.get_shape(), conv24.get_shape()))

        # add
        conv24 = tf.add(conv24, conv21)
        print("B4/7: Input {} Output {}".format(conv24.get_shape(), conv24.get_shape()))

        # ------ n = 3 ------
        # expand
        weight25 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        bias25 = tf.Variable(tf.zeros(shape=(384)))
        conv25 = tf.nn.conv2d(conv24, weight25, strides=(1, 1, 1, 1), padding="SAME")
        conv25 = tf.add(conv25, bias25)
        conv25 = tf.nn.relu(conv25)
        print("B4/8: Input {} Output {}".format(conv24.get_shape(), conv25.get_shape()))

        # depthwise
        weight26 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        bias26 = tf.Variable(tf.zeros(shape=(384)))
        conv26 = tf.nn.depthwise_conv2d(conv25, weight26, strides=(1, 1, 1, 1), padding="SAME")
        conv26 = tf.add(conv26, bias26)
        conv26 = tf.nn.relu(conv26)
        print("B4/9: Input {} Output {}".format(conv25.get_shape(), conv26.get_shape()))

        # conv
        weight27 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64), mean=mu, stddev=sigma))
        bias27 = tf.Variable(tf.zeros(shape=(64)))
        conv27 = tf.nn.conv2d(conv26, weight27, strides=(1, 1, 1, 1), padding="SAME")
        conv27 = tf.add(conv27, bias27)
        print("B4/10: Input {} Output {}".format(conv26.get_shape(), conv27.get_shape()))

        # add
        conv27 = tf.add(conv27, conv24)
        print("B4/11: Input {} Output {}".format(conv24.get_shape(), conv24.get_shape()))

        # ------ n = 4 ------
        # expand
        weight28 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 64, 384), mean=mu, stddev=sigma))
        bias28 = tf.Variable(tf.zeros(shape=(384)))
        conv28 = tf.nn.conv2d(conv27, weight28, strides=(1, 1, 1, 1), padding="SAME")
        conv28 = tf.add(conv28, bias28)
        conv28 = tf.nn.relu(conv28)
        print("B4/12: Input {} Output {}".format(conv27.get_shape(), conv28.get_shape()))

        # depthwise
        weight29 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 384, 1), mean=mu, stddev=sigma))
        bias29 = tf.Variable(tf.zeros(shape=(384)))
        conv29 = tf.nn.depthwise_conv2d(conv28, weight29, strides=(1, 1, 1, 1), padding="SAME")
        conv29 = tf.add(conv29, bias29)
        conv29 = tf.nn.relu(conv29)
        print("B4/13: Input {} Output {}".format(conv28.get_shape(), conv29.get_shape()))

        # conv
        weight30 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 384, 64), mean=mu, stddev=sigma))
        bias30 = tf.Variable(tf.zeros(shape=(64)))
        conv30 = tf.nn.conv2d(conv29, weight30, strides=(1, 1, 1, 1), padding="SAME")
        conv30 = tf.add(conv30, bias30)
        print("B4/14: Input {} Output {}".format(conv29.get_shape(), conv30.get_shape()))

        # add
        conv30 = tf.add(conv30, conv27)
        print("B4/15: Input {} Output {}".format(conv30.get_shape(), conv30.get_shape()))

    return conv30

def main():
    # prepare model to classify
    # features = tf.compat.v1.placeholder(tf.float32, (None, None, None, 3), name='features')
    # labels = tf.placeholder(tf.int64, None, name='labels')

    logits = mobilenet_v2(tf.random.truncated_normal(shape=[1, 227, 227, 3]))
    logits = tf.stop_gradient(logits)
    pass



if __name__ == "__main__":
    main()
