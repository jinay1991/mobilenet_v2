# Image Classification with Mobilenet v2 using TensorFlow Low Level APIs

TensorFlow v2 provides `slim` and `keras` implementation of `Mobilenet v2`, though for some reasons, it is good to develop this model with Low Level APIs (i.e. `tf.nn.*`) providing us full control on how the layers are being formed.

This repository contains the code in two variants.

1. `Eager Execution` based `tf.function` which is bare-metal and identical to simple `numpy` based implementations.
2. Object oriented based bare-metal implementation which wraps all the operations within `tf.keras.Model`/`tf.keras.layers.Layers` allowing user to levarage high level APIs for training avoiding boiler plat code for this. Though all the operations are open and performed with low level APIs hence it still maintains full control of all the Weights/Biases and Operations.

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

## References

1. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
