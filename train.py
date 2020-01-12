import logging
import os
from datetime import datetime

import numpy as np

import tensorflow as tf
from mobilenet_keras_model import MobileNetV2
from vae_model import VariationalAutoEncoder


@tf.function
def normalize_and_resize(images):
    images = tf.cast(images, tf.float32) / 127.5
    images = tf.image.resize(images, (224, 224))
    return images


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", help="sets number of training epochs.", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="Enable/Disable verbose printing")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.FATAL)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    model = MobileNetV2()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(x_train, y_train,
              epochs=args.epochs,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard_callback])
    model.summary()
    model.trainable = False
    model.save("model", save_format="tf")
