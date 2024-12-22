import torch
from torch import nn

from utils import problem


class SoftmaxLayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a softmax calculation.
        Given a matrix x (n, d) on each element performs:

        softmax(x) = exp(x_ij) / sum_k=0^d exp(x_ik)

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape (n, d).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, also with shape (n, d).
                Each row has L-1 norm of 1, and each element is in [0, 1] (i.e. each row is a probability vector).
                Output data.
        """
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        numerator = torch.exp(x - x_max)
        denominator = torch.sum(numerator, dim=1, keepdim=True)
        return numerator / denominator
