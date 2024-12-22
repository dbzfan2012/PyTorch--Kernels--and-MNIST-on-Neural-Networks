from typing import Optional

import torch
from torch import nn

from utils import problem


class LinearLayer(nn.Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(
        self, dim_in: int, dim_out: int, generator: Optional[torch.Generator] = None
    ):
        """Linear Layer, which performs calculation of: x @ weight + bias

        Args:
            dim_in (int): Number of features in data input.
            dim_out (int): Number of features output data should have.
            generator (Optional[torch.Generator], optional): Generator to use when creating weight and bias.
                If defined it should be passed into torch.randn function.
                Defaults to None.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn((dim_in, dim_out), generator=generator, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(dim_out, generator=generator, dtype=torch.float))

        

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Actually perform multiplication x @ weight + bias

        Args:
            x (torch.Tensor): More specifically a torch.FloatTensor, with shape of (n, dim_in).
                Input data.

        Returns:
            torch.Tensor: More specifically a torch.FloatTensor, with shape of (n, dim_out).
                Output data.
        """
        return x @ self.weight + self.bias
