import torch

from utils import problem


class SGDOptimizer(torch.optim.Optimizer):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, params, lr: float) -> None:
        """Constructor for Stochastic Gradient Descent (SGD) Optimizer.

        Args:
            params: Parameters to update each step.
            lr (float): Learning Rate of the gradient descent.
        """
        super().__init__(params, {"lr": lr})

    @problem.tag("hw3-A")
    def step(self, closure=None):
        """
        Performs a step of gradient descent. You should loop through each parameter, and update it's value based on its gradient, value and learning rate.

        Args:
            closure (optional): Ignore this. We will not use in this class, but it is required for subclassing Optimizer.
                Defaults to None.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None: param.data -= group["lr"] * param.grad
