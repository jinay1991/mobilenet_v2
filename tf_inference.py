# Copyright (c) 2020. All Rights Reserved.

import os

import numpy as np

import tensorflow as tf
from mobilenet import MobileNetV2, decode_predictions
from PIL import Image

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='./data/grace_hopper.jpg',
        help='image to be classified')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    args = parser.parse_args()

    model = MobileNetV2(weights="imagenet", input_shape=(224, 224, 3), include_top=True)
    model.trainable = False
    model.summary()
    # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    # NxHxWxC, H:1, W:2
    img = Image.open(args.image).resize((224, 224))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    outputs = model.predict(input_data)

    top_k_results = decode_predictions(outputs)[0]
    for class_id, class_name, class_score in top_k_results:
        print("{} {} {}".format(class_id.encode("ascii"), class_name.encode("ascii"), class_score))
