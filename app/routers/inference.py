import torch

from fastapi import APIRouter, HTTPException

from app.schemas import InferenceParams
from app.enums import ModelName, DeviceName
from engine import inference


router = APIRouter()


@router.post("/inference")
async def inference_model_endpoint(
    device: DeviceName,
    model_name: ModelName,
    weight_path: str,
    img_path: str,
):
    params = InferenceParams(
        device=device,
        model_name=model_name,
        weight_path=weight_path,
        img_path=img_path,
    )

    device = torch.device(params.device)

    try:
        result = inference(
            device=device,
            model_name=params.model_name,
            weight_path=params.weight_path,
            img_path=params.img_path,
        )
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
