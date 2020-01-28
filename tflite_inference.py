# Copyright (c) 2020. All Rights Reserved

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

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
        self.intermediate_details = self._interpreter.get_tensor_details()

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

    def save_intermediate_tensors(self, data_format, dirname):
        """
        Save intermediate tensors to files
        """
        for tensor_details in self.intermediate_details:
            tensor_name = tensor_details['name']
            tensor_index = tensor_details['index']
            tensor = self._interpreter.get_tensor(tensor_index)
            tensor = tensor.astype(np.uint8)

            channel_info = "#\n"
            if len(tensor.shape) == 4:
                if data_format == "channel_first":
                    tensor = np.transpose(tensor, (0, 3, 2, 1))
                    channel_order = "NCHW"
                else:
                    channel_order = "NHWC"
                channel_info = "# Note: Writing Matrix in {} format.\n#\n".format(channel_order)
            header = "#\n# Tensor Detail: \n#  name: {}\n#  type: {}\n#  shape: {}\n{}".format(
                tensor_name, tensor.dtype, tensor.shape, channel_info)

            if not os.path.exists(dirname):
                os.mkdir(dirname)
            fname = os.path.join(dirname, tensor_details['name'].replace("/", "_") + "_tensor.txt")
            with open(fname, "wb") as fp:
                fp.write("{}{}".format(header, np.array2string(tensor, threshold=sys.maxsize)))


def main(image, input_mean, input_std, data_format, save_to):
    tflite_inference_engine = TFLiteInferenceEngine()
    output_data = tflite_inference_engine(image, input_mean, input_std)

    top_k_results = decode_predictions(output_data)[0]
    for class_id, class_name, class_score in top_k_results:
        print("{} {} {}".format(class_id.encode("ascii"), class_name.encode("ascii"), class_score))

    if save_to is not None:
        tflite_inference_engine.save_intermediate_tensors(data_format, save_to)


if __name__ == '__main__':
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
    parser.add_argument(
        '--data_format',
        default="channel_first", type=str,
        help='data format (channel_first or channel_last)')
    parser.add_argument(
        '--save_to',
        default=None, type=str,
        help='directory name for saving intermediate_outputs')
    args = parser.parse_args()

    main(args.image, args.input_mean, args.input_std, args.data_format, args.save_to)
