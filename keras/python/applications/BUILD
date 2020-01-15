package(default_visibility = ["//visibility:public"])

py_binary(
    name = "mobilenet_v2",
    srcs = [
        "imagenet_utils.py",
        "mobilenet_v2.py",
    ],
    data = [":testdata"],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["mobilenet_v2_test.py"],
    data = [":testdata"],
    deps = [
        ":mobilenet_v2",
    ],
)
