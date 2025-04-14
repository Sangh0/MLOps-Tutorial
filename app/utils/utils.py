import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from typing import Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler


def cal_accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = torch.argmax(outputs, dim=1)
    correct = (outputs == labels).sum() / len(outputs)
    return correct


def load_optimizer(
    optimizer_name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
):
    assert optimizer_name in ("sgd", "rmsprop", "adagrad", "adam", "adamw")
    if optimizer_name == "sgd":
        opt = optim.SGD
    elif optimizer_name == "adagrad":
        opt = optim.Adagrad
    elif optimizer_name == "rmsprop":
        opt = optim.RMSprop
    elif optimizer_name == "adam":
        opt = optim.Adam
    else:
        opt = optim.AdamW

    optimizer = opt(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def load_dataloader(
    dataset_name: str = "mnist",
    batch_size: int = 32,
    valid_size: Optional[float] = 0.2,
    evaluate: bool = False,
):
    assert dataset_name in ("mnist", "fashion_mnist", "cifar10", "cifar100")

    if dataset_name == "mnist":
        dset = dsets.MNIST
    elif dataset_name == "fashion_mnist":
        dset = dsets.FashionMNIST
    elif dataset_name == "cifar10":
        dset = dsets.CIFAR10
    else:
        dset = dsets.CIFAR100

    if evaluate:
        test_set = dset(
            root=f"{dataset_name}/",
            train=False,
            transform=transforms.ToTensor(),
            download=True,
        )

        return DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    train_set = dset(
        root=f"{dataset_name}/",
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_set)),
        test_size=valid_size,
        shuffle=True,
        stratify=train_set.targets,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_idx),
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(valid_idx),
        drop_last=True,
    )
    return train_loader, valid_loader
