from enum import Enum


class DatasetName(str, Enum):
    mnist = "mnist"
    fashion_mnist = "fashion_mnist"
    cifar10 = "cifar10"
    cifar100 = "cifar100"
