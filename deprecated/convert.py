#!/usr/bin/python
import os

import numpy as np

import tensorflow as tf


@tf.function
def normalize_and_resize(images):
    images = tf.cast(images, tf.float32) / 127.5
    images = tf.image.resize(images, (224, 224))
    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", required=True)
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    test_ds = tf.data.Dataset.from_tensor_slices((x_test)).batch(1)

    def representative_data_gen():
        for image in test_ds.take(100):
            yield [image]

    converter = tf.lite.TFLiteConverter.from_saved_model(args.saved_model_dir)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(os.path.join(args.saved_model_dir, "converted_model.tflite"), "wb") as fp:
        fp.write(tflite_model)
    print("Successfully saved model to tflite...")
