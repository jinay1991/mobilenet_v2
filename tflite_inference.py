# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""label_image for tflite."""

from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np

import tensorflow as tf  # TF2
from PIL import Image
from mobilenet import decode_predictions


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--image',
        default='./data/grace_hopper.jpg',
        help='image to be classified')
    parser.add_argument(
        '-m',
        '--model_file',
        default='./data/mobilenet_v2_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '--input_mean',
        default=127.5, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=127.5, type=float,
        help='input standard deviation')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # ---------
    # intermediate_details = interpreter.get_tensor_details()
    # for tensor_detail in intermediate_details:
    #     print("[{}] {} {}".format(tensor_detail['index'], tensor_detail['name'], tensor_detail['dtype']))

    # dirname = "intermediate_layers_py"
    # if not os.path.exists(dirname):
    #     os.mkdir(dirname)

    # for tensor_detail in intermediate_details:
    #     tensor_index = tensor_detail['index']
    #     tensor_name = tensor_detail['name'].replace('/', '_')
    #     tensor = interpreter.get_tensor(tensor_index)
    #     tensor = tensor.astype(np.uint8)

    #     np.save(os.path.join(dirname, tensor_name), tensor)
    # # ---------

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    img = Image.open(args.image).resize((width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    top_k_results = decode_predictions(output_data)[0]
    for class_id, class_name, class_score in top_k_results:
        print("{} {} {}".format(class_id.encode("ascii"), class_name.encode("ascii"), class_score))
