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
        conv6 = tf.nn.relu(conv6)
        print("B2/3: Input {} Output {}".format(conv5.get_shape(), conv6.get_shape()))

        # n = 2
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

        # FIXME: How to add conv9 with input_tensor {conv3}?
        # # add
        # conv9 = tf.add(conv9, conv3)
        # print("B2/7: Input {} Output {}".format(conv9.get_shape(), conv9.get_shape()))

    # [bottleneck (3)] Input 56x56x24 Output 28x28x32
    # t = 6, c = 32, n = 3, s = 2
    with tf.compat.v1.variable_scope("B3"):
        # expand
        weight10 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 24, 144), mean=mu, stddev=sigma))
        bias10 = tf.Variable(tf.zeros(shape=(144)))
        conv10 = tf.nn.conv2d(conv9, weight10, strides=(1, 1, 1, 1), padding="SAME")
        conv10 = tf.add(conv10, bias10)
        conv10 = tf.nn.relu(conv10)
        print("B2/1: Input {} Output {}".format(conv9.get_shape(), conv10.get_shape()))

        # depthwise
        weight5 = tf.Variable(tf.random.truncated_normal(shape=(3, 3, 144, 1), mean=mu, stddev=sigma))
        bias5 = tf.Variable(tf.zeros(shape=(144)))
        conv5 = tf.nn.depthwise_conv2d(conv4, weight5, strides=(1, 2, 2, 1), padding="SAME")
        conv5 = tf.add(conv5, bias5)
        conv5 = tf.nn.relu(conv5)
        print("B2/2: Input {} Output {}".format(conv4.get_shape(), conv5.get_shape()))

        # conv
        weight6 = tf.Variable(tf.random.truncated_normal(shape=(1, 1, 144, 32), mean=mu, stddev=sigma))
        bias6 = tf.Variable(tf.zeros(shape=(32)))
        conv6 = tf.nn.conv2d(conv5, weight6, strides=(1, 1, 1, 1), padding="SAME")
        conv6 = tf.add(conv6, bias6)
        conv6 = tf.nn.relu(conv6)
        print("B2/3: Input {} Output {}".format(conv5.get_shape(), conv6.get_shape()))

        # n = 2
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
        conv9 = tf.nn.relu(conv9)
        print("B2/6: Input {} Output {}".format(conv8.get_shape(), conv9.get_shape()))

    return conv2

def main():
    # prepare model to classify
    # features = tf.compat.v1.placeholder(tf.float32, (None, None, None, 3), name='features')
    # labels = tf.placeholder(tf.int64, None, name='labels')

    logits = mobilenet_v2(tf.random.truncated_normal(shape=[1, 227, 227, 3]))
    logits = tf.stop_gradient(logits)
    pass



if __name__ == "__main__":
    main()
