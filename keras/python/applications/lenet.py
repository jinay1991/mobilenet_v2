# copyright (c) 2020. All Rights Reserved.

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def LeNet(input_shape=None,
          classes=10,
          **kwargs):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


if __name__ == "__main__":
    import os

    model = LeNet()
    # model.trainable = False
    # model.summary()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = tf.cast(train_images, tf.float32) / 255.0, tf.cast(test_images, tf.float32) / 255.0

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1,
              validation_data=(test_images, test_labels))

    dirname = "saved_model"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model.save(dirname, save_format="tf")

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)

    def representative_data_gen():
        dataset = tf.data.Dataset.from_tensor_slices((train_images)).batch(1)
        for inputs in dataset.take(100):
            inputs = tf.cast(inputs, tf.float32) / 255.0
            yield [inputs]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.experimental_new_converter = True
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(os.path.join(dirname, "lenet.tflite"), "wb") as fp:
        fp.write(tflite_model)
