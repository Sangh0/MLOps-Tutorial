import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from pydantic import BaseModel

from app.utils.utils import cal_accuracy, load_optimizer


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
        os.makedirs(self.model_registry + f"{self.exp_name}", exist_ok=True)

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

            print(f"{'#'*50}")

            if best_loss > valid_loss:
                print(
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

            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Train loss: {train_loss:.3f}, Train accuracy: {train_acc:.3f}")
            print(f"Valid loss: {valid_loss:.3f}, Valid accuracy: {valid_acc:.3f}\n")

        mlflow.pytorch.log_model(self.model, "models")
        # 모델 등록
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, params.exp_name)
