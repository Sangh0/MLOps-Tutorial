import uvicorn

from fastapi import FastAPI
from pyngrok import ngrok

from app.routers import train, evaluate, inference
from config import Config


app = FastAPI(
    title="Tutorial API",
    version="0.1",
    description="Tutorial API for training models such as CNN, CNN with BatchNorm, and MLP on various datasets such as MNIST, FashionMNIST, CIFAR10, and CIFAR100.",
)


app.include_router(train.router, prefix="", tags=["train"])
app.include_router(evaluate.router, prefix="", tags=["evaluate"])
app.include_router(inference.router, prefix="", tags=["inference"])


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the CNN Training API. Visit /docs for API documentation."
    }


@app.post("/get_mlflow_ui_url")
def get_mlflow_ui_url():
    ngrok.kill()

    NGROK_AUTH_TOKEN = Config.AUTH_TOKEN
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
    return {"MLflow Tracking UI": ngrok_tunnel.public_url}


def main():
    print(f"Authentication Key: {Config.AUTH_TOKEN}")
    print(f"Data Directory: {Config.DATA_DIR}")


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="0.0.0.0", port=8000)
