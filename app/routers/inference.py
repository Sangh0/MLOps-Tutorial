from fastapi import APIRouter, HTTPException

from app.schemas import InferenceParams
from engine import inference


router = APIRouter()


@router.post("/inference")
async def inference_model_endpoint(model_name: str, weight_path: str, img_path: str):
    params = InferenceParams(
        model_name=model_name,
        weight_path=weight_path,
        img_path=img_path,
    )

    try:
        result = inference(weight_path=params.weight_path, img_path=params.img_path)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
