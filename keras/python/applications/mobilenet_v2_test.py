# Copyright (c) 2020. All Rights Reserved.

import numpy as np

import tensorflow as tf
from mobilenet_v2 import MobileNetV2, decode_predictions
from PIL import Image
import unittest


class MobileNetV2Test(unittest.TestCase):
    def __init__(self):
        pass

    def test_image(self):
        pass


if __name__ == "__main__":
    model = MobileNetV2(weights="imagenet", include_top=True)
    model.trainable = False
    model.summary()

    img = np.array(Image.open("grace_hopper.jpg").resize((224, 224)))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = (img - 127.5) / 127.5

    output_data = model.predict(img)
    top_k = decode_predictions(output_data, top=5)
    for k in top_k[0]:
        print("{}".format(k))
