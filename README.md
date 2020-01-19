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

To run this programs,

```
python tflite_inference.py -i <path/to/image>
```

or

```
bazel run //:tflite_inference
```

1. To infer TensorFlow graph with pre-trained weights, run `python tf_inference.py`.

2. To infer TensorFlow Lite Conversion with pre-trained imagenet graph, run `python tflite_converter.py`.

3. To infer TensorFlow Lite Model Inference with pre-trained imagenet weights, run `python tflite_inference.py`.


## Architecture

Details on MobileNet v2 Architecture and layer information can be found in `layers_to_tensor_map.xlsx`

## References

1. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
