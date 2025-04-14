import uvicorn

from fastapi import FastAPI

from app.routers import train, evaluate, inference


app = FastAPI(
    title="Tutorial API",
    version="0.1",
    description="Tutorial API for training models such as CNN, CNN with BatchNorm, and MLP on various datasets such as MNIST, FashionMNIST, CIFAR10, and CIFAR100.",
)


app.include_router(train.router, prefix="", tags=["train"])
app.include_router(evaluate.router, prefix="", tags=["evaluate"])
app.include_router(inference.router, prefix="", tags=["inference"])


@app.post("/get_quantized_model")
def get_quantized_model(model_name: str, weight_dir: str, save_dir: str):
    from engine import Exporter

    exporter = Exporter(model_name=model_name, weight_dir=weight_dir, save_dir=save_dir)
    quantized_model = exporter()
    return {"message": f"Quantized model saved to {save_dir}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
