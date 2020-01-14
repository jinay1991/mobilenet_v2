#!/usr/bin/python
import os

import numpy as np

import tensorflow as tf
from keras.python.datasets.imagenette import download_imagenette

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_keras_model(args.model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    converter.representative_dataset = download_imagenette
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    dirname = os.path.dirname(args.model)
    fname = args.model.split("/")[-1] + ".tflite"
    tflite_modelname = os.path.join(dirname,  fname)

    with open(tflite_modelname, "wb") as fp:
        fp.write(tflite_model)
