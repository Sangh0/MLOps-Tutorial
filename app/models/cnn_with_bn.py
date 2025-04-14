import torch
import torch.nn as nn


"""
Convolutional Neural Network with Batch Normalization
"""


class CNNWithBN(nn.Module):
    """
    Args:
        in_dim (int): Number of input channels
        hidden_dim (int): Number of hidden channels
        num_classes (int): Number of classes of dataset
    """

    def __init__(self, in_dim: int = 1, hidden_dim: int = 8, num_classes: int = 10):
        super(CNNWithBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
