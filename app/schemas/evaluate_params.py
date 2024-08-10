from pydantic import BaseModel


class EvaluateParams(BaseModel):
    device: str
    model_name: str
    dataset_name: str
    weight_path: str
    batch_size: int
