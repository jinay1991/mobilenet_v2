# Copyright (c) 2020. All Rights Reserved

from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np

import tensorflow as tf
from mobilenet import decode_predictions
from PIL import Image
from tflite_converter import convert_mobilenet


class TFLiteInferenceEngine(object):
    def __init__(self):
        tflite_model = convert_mobilenet()

        self._interpreter = tf.lite.Interpreter(model_content=tflite_model)
        self._interpreter.allocate_tensors()

        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()

    def __call__(self, image_path, input_mean, input_std):
        # NxHxWxC, H:1, W:2
        height = self.input_details[0]['shape'][1]
        width = self.input_details[0]['shape'][2]
        img = Image.open(image_path).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        input_data = (np.float32(input_data) - input_mean) / input_std

        self._interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self._interpreter.invoke()

        return self._interpreter.get_tensor(self.output_details[0]['index'])


if __name__ == '__main__':
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

    tflite_inference_engine = TFLiteInferenceEngine()

    output_data = tflite_inference_engine(args.image, args.input_mean, args.input_std)

    top_k_results = decode_predictions(output_data)[0]
    for class_id, class_name, class_score in top_k_results:
        print("{} {} {}".format(class_id.encode("ascii"), class_name.encode("ascii"), class_score))
