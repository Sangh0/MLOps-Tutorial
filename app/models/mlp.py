import torch
import torch.nn as nn


"""
Multi Layer Perceptron
"""


class MLP(nn.Module):
    """
    Args:
        in_dim (int): Number of input channels
        hidden_dim (int): Number of hidden channels
        num_classes (int): Number of classes of dataset
    """

    def __init__(self, in_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
