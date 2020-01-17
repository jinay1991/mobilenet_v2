
import numpy as np

import tensorflow as tf
from mobilenet_v2 import MobileNetV2, decode_predictions
from PIL import Image

# Create model.
model = MobileNetV2(weights="imagenet", include_top=True)
model.trainable = False
model.summary()

# Mobel Compile
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

img = np.array(Image.open("grace_hopper.jpg").resize((224, 224)))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)
img = (img - 127.5) / 127.5

# add N dim

output_data = model.predict(img)
top_k = decode_predictions(output_data, top=5)
for k in top_k[0]:
    print("{}".format(k))

print("-=---------")

import pathlib


data_dir = tf.keras.utils.get_file(origin='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
                                   fname='imagenette2-160', untar=True)

data_dir = pathlib.Path(data_dir)
data_dir = data_dir.joinpath("val")
image_count = len(list(data_dir.glob('*/*.JPEG')))
print(image_count)

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=1,
                                                     shuffle=True,
                                                     target_size=(224, 224),
                                                     classes = list(CLASS_NAMES))
print(next(train_data_gen)[0])
def representative_data_gen():
    yield [next(train_data_gen)[0]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.experimental_new_converter = True
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()

with open("converted_model.tflite", "wb") as fp:
    fp.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------
intermediate_details = interpreter.get_tensor_details()
print("Input: {}".format(input_details))
print("Output: {}".format(output_details))
for t in intermediate_details:
    print("Intermediate ({}): {} {}".format(t["index"], t['name'], t["dtype"]))
# ---------

# check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32

# NxHxWxC, H:1, W:2
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
img = Image.open("grace_hopper.jpg").resize((width, height))

# add N dim
input_data = np.expand_dims(img, axis=0)

if floating_model:
    input_data = (np.float32(input_data) - 127.5) / 127.5

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
top_k = decode_predictions(output_data, top=5)
for k in top_k[0]:
    print("{}".format(k))
