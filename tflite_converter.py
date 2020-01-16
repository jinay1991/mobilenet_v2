#!/usr/bin/python
import os
import pathlib

import numpy as np

import tensorflow as tf
import logging


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="./data/mobilenet_v2_1.0_224_quant_frozen.pb")
    args = parser.parse_args()

    converter = tf.lite.TFLiteConverter.from_saved_model(args.model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    converter.representative_dataset = download_imagenette
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    dirname = os.path.dirname(args.model)
    fname = args.model.split("/")[-1] + "_converted.tflite"
    tflite_modelname = os.path.join(dirname,  fname)

    with open(tflite_modelname, "wb") as fp:
        fp.write(tflite_model)
