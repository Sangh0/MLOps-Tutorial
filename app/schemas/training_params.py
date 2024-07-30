from pydantic import BaseModel

from app.enums import ModelName, DatasetName, DeviceName, OptimizerName


class TrainingParams(BaseModel):
    model_name: ModelName = "cnn"
    dataset_name: DatasetName = "mnist"
    epochs: int = 10
    batch_size: int = 32
    valid_size: float = 0.2
    learning_rate: float = 0.01
    weight_decay: float = 0.005
    device: DeviceName = "mps"
    optimizer_name: OptimizerName = "sgd"
    exp_name: str = "exp1"
    seed: int = 42
