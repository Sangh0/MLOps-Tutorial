import os
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.routers import train, evaluate, inference


app = FastAPI(
    title="MNIST Training API",
    version="0.1",
    description="Tutorial API for training models such as CNN, CNN with BatchNorm, and MLP on various datasets such as MNIST, FashionMNIST, CIFAR10, and CIFAR100.",
)


app.include_router(train.router, prefix="/train", tags=["train"])
app.include_router(evaluate.router, prefix="/evaluate", tags=["evaluate"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CNN Training API. Visit /docs for API documentation."
    }


@app.get("/model_list")
def list_models():
    try:
        model_registry = "./runs/train"
        if not os.path.exists(model_registry):
            return {"models": []}
        models = os.listdir(model_registry)
        return {"models": models}

    except Exception as e:
        raise HTTPException(status_code=501, detail=str(e))


@app.get("/download_model")
def download_model(exp_name: str):
    model_registry = "./runs/train"
    weight_file_path = f"{model_registry}/{exp_name}/best_model.pth"
    if os.path.exists(weight_file_path):
        return FileResponse(weight_file_path, filename="best_model.pth")
    else:
        raise HTTPException(status_code=404, detail="Model file not found")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
