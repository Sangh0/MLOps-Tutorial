from pydantic import BaseModel


class InferenceParams(BaseModel):
    device: str
    model_name: str
    weight_path: str
    img_path: str
