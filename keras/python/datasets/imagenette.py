#!/usr/bin/python
import logging
import pathlib

import numpy as np

import tensorflow as tf

FASTAI_IMAGENET = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def download_imagenette():
    data_dir = tf.keras.utils.get_file(origin=FASTAI_IMAGENET,
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
