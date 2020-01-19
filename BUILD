package(default_visibility = ["//visibility:public"])

filegroup(
    name = "testdata",
    srcs = glob([
        "data/*",
    ]),
)

py_binary(
    name = "tf_inference",
    srcs = ["tf_inference.py"],
    data = [":testdata"],
    python_version = "PY2",
)

py_binary(
    name = "tflite_converter",
    srcs = ["tflite_converter.py"],
    data = [":testdata"],
    python_version = "PY2",
)

py_binary(
    name = "tflite_inference",
    srcs = ["tflite_inference.py"],
    data = [
        ":testdata",
    ],
    python_version = "PY2",
)
