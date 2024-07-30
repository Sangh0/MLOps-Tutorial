from enum import Enum


class DeviceName(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
