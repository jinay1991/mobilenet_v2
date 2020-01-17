package(default_visibility = ["//visibility:public"])

py_library(
    name = "mobilenet_v2",
    srcs = [
        "imagenet_utils.py",
        "mobilenet_v2.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//keras/python/layers"],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["mobilenet_v2_test.py"],
    data = ["//:testdata"],
    python_version = "PY2",
    deps = [
        ":mobilenet_v2",
    ],
)
