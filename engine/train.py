import os
import logging
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

from utils.utils import cal_accuracy, load_optimizer


class Trainer(object):

    def __init__(
        self,
        device: str,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        epochs: int,
        optimizer_name: str,
        exp_name: str,
        mlflow_run: mlflow.ActiveRun,
    ):

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.optimizer = load_optimizer(
            optimizer_name=optimizer_name,
            model=self.model,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.model_registry = "./runs/"
        self.exp_name = exp_name
        self.mlflow_run = mlflow_run
        os.makedirs(self.model_registry + f"{self.exp_name}", exist_ok=True)

        self.logger = logging.getLogger("Training")
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(stream_handler)

        self.logger.info(
            f"##### Training Settings #####\n"
            f"{' ' * 3} - Device: {self.device}\n"
            f"{' ' * 3} - Model: {model.__class__.__name__}\n"
            f"{' ' * 3} - Optimizer: {optimizer_name}\n"
            f"{' ' * 3} - Learning Rate: {lr}\n"
            f"{' ' * 3} - Weight Decay: {weight_decay}\n"
            f"{' ' * 3} - Epochs: {epochs}\n"
            f"{' ' * 3} - Experiment Name: {exp_name}\n"
            f"#############################\n"
        )

    def train_on_batch(self, model, train_loader):
        model.train()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = model(images)
            acc = cal_accuracy(outputs, labels)
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()
            batch_acc += acc.item()

        return batch_loss / (batch + 1), batch_acc / (batch + 1)

    @torch.no_grad()
    def valid_on_batch(self, model, valid_loader):
        model.eval()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(valid_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)
            acc = cal_accuracy(outputs, labels)
            loss = self.loss_func(outputs, labels)

            batch_loss += loss.item()
            batch_acc += acc.item()

        return batch_loss / (batch + 1), batch_acc / (batch + 1)

    def fit(self, train_loader, valid_loader):
        best_loss = 9999999

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_on_batch(self.model, train_loader)
            valid_loss, valid_acc = self.valid_on_batch(self.model, valid_loader)

            self.logger.info(f"{'#'*50}")

            if best_loss > valid_loss:
                self.logger.info(
                    f"# loss decreased at epoch {epoch+1} ({best_loss} --> {valid_loss}), saving model..."
                )
                best_loss = valid_loss
                torch.save(
                    self.model.state_dict(),
                    self.model_registry + f"{self.exp_name}/best_model.pt",
                )

            # log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            mlflow.log_metric("valid_acc", valid_acc, step=epoch)
            mlflow.log_metric("lr", self.optimizer.param_groups[0]["lr"])

            self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            self.logger.info(
                f"Train loss: {train_loss:.3f}, Train accuracy: {train_acc:.3f}"
            )
            self.logger.info(
                f"Valid loss: {valid_loss:.3f}, Valid accuracy: {valid_acc:.3f}\n"
            )

        mlflow.pytorch.log_model(self.model, "models")
        # Register the model
        model_uri = f"runs:/{self.mlflow_run.info.run_id}/model"
        mlflow.register_model(model_uri, self.exp_name)

        # Get the latest model version
        model_version = mlflow.get_latest_versions(self.exp_name)[0].version

        # Set the model version to staging
        mlflow.tracking.MlflowClient().transition_model_version_stage(
            name=self.exp_name,
            version=model_version,
            stage="Staging",
        )