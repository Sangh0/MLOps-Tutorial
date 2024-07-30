from pydantic import BaseModel


class InferenceParams(BaseModel):
    model_name: str
    weight_path: str
    img_path: str
