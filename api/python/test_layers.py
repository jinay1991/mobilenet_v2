
import numpy as np

import tensorflow as tf
from mobilenet_v2 import MobileNetV2, decode_predictions
from PIL import Image

# Create model.
model = MobileNetV2(weights="imagenet", include_top=True)
model.trainable = False
model.summary()

# Mobel Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

img = np.array(Image.open("grace_hopper.jpg").resize((224, 224)))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)
img = (img - 127.5) / 127.5

# add N dim

output_data = model.predict(img)
print("{}".format(decode_predictions(output_data, top=5)))
