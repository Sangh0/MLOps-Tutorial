from enum import Enum


class ModelName(str, Enum):
    cnn = "cnn"
    cnn_with_bn = "cnn_with_bn"
    mlp = "mlp"
