from enum import Enum


class OptimizerName(str, Enum):
    sgd = "sgd"
    rmsprop = "rmsprop"
    adagrad = "adagrad"
    adam = "adam"
    adamw = "adamw"
