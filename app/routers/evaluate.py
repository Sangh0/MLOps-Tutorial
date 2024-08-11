import torch

from fastapi import APIRouter, HTTPException

from app.schemas import EvaluateParams
from app.enums import ModelName, DatasetName, DeviceName
from engine import evaluate
from models import CNN, CNNWithBN, MLP
from utils import load_dataloader


router = APIRouter()


@router.post("/evaluate")
async def evaluate_model_endpoint(
    device: DeviceName,
    model_name: ModelName,
    weight_path: str,
    batch_size: int,
    dataset_name: DatasetName,
):
    assert model_name in ("cnn", "cnn_with_bn", "mlp")

    params = EvaluateParams(
        device=device,
        model_name=model_name,
        dataset_name=dataset_name,
        weight_path=weight_path,
        batch_size=batch_size,
    )

    device = torch.device(params.device)

    test_loader = load_dataloader(
        dataset_name=params.dataset_name,
        batch_size=params.batch_size,
        evaluate=True,
    )

    try:
        if model_name == "cnn":
            model = CNN()
        elif model_name == "cnn_with_bn":
            model = CNNWithBN()
        else:
            model = MLP()

        model.load_state_dict(torch.load(params.weight_path))
        evaluate(device=device, model=model, test_loader=test_loader)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
