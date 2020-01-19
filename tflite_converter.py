# Copyright (c) 2020. All Rights Reserved.

import logging
import os
import pathlib

import numpy as np

import tensorflow as tf
from mobilenet import MobileNetV2


def download_imagenette():
    data_dir = tf.keras.utils.get_file(origin="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
                                       fname='imagenette2-160',
                                       untar=True)

    data_dir = pathlib.Path(data_dir)
    data_dir = data_dir.joinpath("val")
    image_count = len(list(data_dir.glob('*/*.JPEG')))
    logging.info("Dataset contains {} jpeg images".format(image_count))

    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                                                                      samplewise_std_normalization=True)

    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                         batch_size=1,
                                                         shuffle=True,
                                                         target_size=(224, 224),
                                                         classes=list(CLASS_NAMES))
    return train_data_gen


def convert_mobilenet():
    model = MobileNetV2(weights="imagenet")
    model.trainable = False
    model.summary()

    def representative_data_gen():
        data_dir = tf.keras.utils.get_file(origin="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
                                           fname='imagenette2-160',
                                           untar=True)

        data_dir = os.path.join(data_dir, "val")

        CLASS_NAMES = np.array([item for item in os.listdir(data_dir) if item != "LICENSE.txt"])
        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
                                                                          samplewise_std_normalization=True)

        train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                             batch_size=1,
                                                             shuffle=True,
                                                             target_size=(224, 224),
                                                             classes=list(CLASS_NAMES))
        yield [next(train_data_gen)[0]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    return tflite_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", default="saved_model", help="Path to use for saving converted *.tflite model")
    args = parser.parse_args()

    tflite_model = convert_mobilenet()

    dirname = args.save_to
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    fname = "mobilenet_v2_1.0_224_quant.tflite"
    with open(os.path.join(dirname,  fname), "wb") as fp:
        fp.write(tflite_model)
