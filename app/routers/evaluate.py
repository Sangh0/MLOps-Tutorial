import torch

from fastapi import APIRouter, HTTPException

from app.schemas import EvaluateParams
from engine import evaluate
from models import CNN, CNNWithBN, MLP


router = APIRouter()


@router.post("/evaluate")
async def evaluate_model_endpoint(model_name: str, weight_path: str, batch_size: int):
    assert model_name in ("cnn", "cnn_with_bn", "mlp")

    params = EvaluateParams(
        model_name=model_name,
        weight_path=weight_path,
        batch_size=batch_size,
    )

    try:
        if model_name == "cnn":
            model = CNN()
        elif model_name == "cnn_with_bn":
            model = CNNWithBN()
        else:
            model = MLP()

        model.load_state_dict(torch.load(params.weight_path))
        evaluate(model, params.batch_size)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
