# Image Classification with Mobilenet v2 using TensorFlow Low Level APIs

TensorFlow v2 provides `slim` and `keras` implementation of `Mobilenet v2`, though for some reasons, it is good to develop this model with Low Level APIs (i.e. `tf.nn.*`) providing us full control on how the layers are being formed.

This repository contains the code in Object oriented based bare-metal implementation which wraps all the operations within `tf.keras.Model`/`tf.keras.layers.Layers` allowing user to levarage high level APIs for training avoiding boiler plat code for this. Though all the operations are open and performed with low level APIs hence it still maintains full control of all the Weights/Biases and Operations.

Here, repository contains source code for inference only and it does not involve any of the training scripts due to the fact that it uses `ImageNet` weights from pre-trained MobileNetV2 with help of APIs `model.load_weights()`. Though if one want to use Model for transfer learning or retraining, they can still use the model definition and use Keras APIs such as `model.compile()` and `model.fit()` to train the model. 

The implementation initializes weights and biases with `glorot_uniform` and `zeros`, respectively. This can easily be changed if one requires to use some other random initializers by simply passing arguments to subsequent layer instants.

## Dependencies

To try these implementation, you might need `TensorFlow v2.0` which can be installed as explained in [TensorFlow Installation Guide](https://www.tensorflow.org/install)

```
python -m pip install -U pip
python -m pip install -U -r requirements.txt
```

## Build/Run

There are three programs this repository provides

1. TensorFlow Graph Inference: Use MobileNet V2 Model definition and perform inference on given image.

`python tf_inference.py -i <path/to/image>` or `bazel run //:tf_inference -- -i <path/to/image>`


2. TensorFlow Lite Model Conversion: Use MobileNet V2 Model definition and convert that to Integer TFLite Model.

`python tflite_converter.py --save_to <path/to/save/dir>` or `bazel run //:tflite_converter -- --save_to <path/to/save/dir>`


3. TensorFlow Lite Inference: Run inference for given image and provided TFLite Model.

`python tflite_inference.py -i <path/to/image>` or `bazel run //:tflite_inference -- -i <path/to/image>`

## Architecture

Details on MobileNet v2 Architecture and layer information can be found in `layers_to_tensor_map.xlsx`

Model Diagram can be found at `data/model.png`.

All the MobileNet V2 layer implementations with Low Level APIs from TensorFlow (such as `tf.nn.relu`, `tf.nn.conv2d` etc.) are implemented as custom layer classes (see `mobilenet/layers/*.py`) which is being modelled into MobileNet V2 Architecture (see `mobilenet/mobilenet_v2.py`).


## References

1. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
2. Architecture and layer definitions were referred or reused from [TensorFlow GitHub](https://github.com/tensorflow/tensorflow)
