import mlflow
from fastapi import APIRouter, HTTPException

from app.schemas import TrainingParams
from app.enums import ModelName, DatasetName, DeviceName, OptimizerName
from models import CNN, CNNWithBN, MLP
from engine import Trainer
from utils import set_seed, load_dataloader


router = APIRouter()


@router.post("/train_model")
async def train_model_endpoint(
    model_name: ModelName = "cnn",
    dataset_name: DatasetName = "mnist",
    epochs: int = 10,
    learning_rate: float = 0.01,
    weight_decay: float = 0.005,
    batch_size: int = 32,
    valid_size: float = 0.2,
    device: DeviceName = "cpu",
    optimizer_name: OptimizerName = "sgd",
    exp_name: str = "experiment",
    seed: int = 42,
):
    set_seed(seed, deterministic=True)

    params = TrainingParams(
        model_name=model_name,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        valid_size=valid_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        optimizer_name=optimizer_name,
        exp_name=exp_name,
        seed=seed,
    )

    try:
        mlflow.set_experiment(exp_name)

        with mlflow.start_run() as mlflow_run:
            mlflow.log_params(params.dict())

            train_loader, valid_loader = load_dataloader(
                dataset_name=params.dataset_name,
                batch_size=params.batch_size,
                valid_size=params.valid_size,
            )

            if params.model_name == "cnn":
                model = CNN()
            elif params.model_name == "cnn_with_bn":
                model = CNNWithBN()
            else:
                model = MLP()

            trainer = Trainer(
                device=params.device,
                model=model,
                lr=params.learning_rate,
                weight_decay=params.weight_decay,
                epochs=params.epochs,
                optimizer_name=params.optimizer_name,
                exp_name=params.exp_name,
                mlflow_run=mlflow_run,
            )

            trainer.fit(train_loader, valid_loader)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
