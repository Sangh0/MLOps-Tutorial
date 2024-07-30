from pydantic import BaseModel


class EvaluateParams(BaseModel):
    model_name: str
    weight_path: str
    batch_size: int
