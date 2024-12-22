import torch
from torch import nn

from utils import problem


class CrossEntropyLossLayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate Crossentropy loss based on (normalized) predictions and true categories/classes.

        For a single example (x is vector of predictions, y is correct class):

        cross_entropy(x, y) = -log(x[y])

        Args:
            y_pred (torch.Tensor): More specifically a torch.FloatTensor, with shape (n, c).
                Predictions of classes. Each row is normalized so that L-1 norm is 1 (Each row is proper probability vector).
                Input data.
            y_true (torch.Tensor): More specifically a torch.LongTensor, with shape (n,).
                Each element is an integer in range [0, c).
                Input data.

        Returns:
            torch.Tensor: More specifically a SINGLE VALUE torch.FloatTensor (i.e. with shape (1,)).
                Should be a mean over all examples.
                Result.
        """
        class_probs = y_pred[torch.arange(len(y_true)), y_true] + 1e-8
        return torch.mean(-torch.log(class_probs))
