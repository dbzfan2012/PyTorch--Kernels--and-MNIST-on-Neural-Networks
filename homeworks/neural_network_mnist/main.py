# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create an F1 model that can output data via:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.h = h
        self.d = d
        self.k = k

        #W0: W = (d x h), b = (h,), alpha = 1 / sqrt(d)
        alpha0 = 1 / math.sqrt(d)
        #sets weights between [-alpha, alpha + 1), clamps to [-alpha, alpha] in case it exceeds alpha but is less than alpha + 1
        self.weights0 = Parameter(torch.empty(d, h).uniform_(-alpha0, alpha0))
        self.bias0 = Parameter(torch.empty(h).uniform_(-alpha0, alpha0))

        #W1: W = (h x k), b = (k,), alpha = 1 / sqrt(h)
        alpha1 = 1 / math.sqrt(h)
        #sets weights between [-alpha, alpha + 1), clamps to [-alpha, alpha] in case it exceeds alpha but is less than alpha + 1
        self.weights1 = Parameter(torch.empty(h, k).uniform_(-alpha1, alpha1))
        self.bias1 = Parameter(torch.empty(k).uniform_(-alpha1, alpha1))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(x @ self.weights0 + self.bias0) @ self.weights1 + self.bias1

class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model that can output data via:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        self.h0 = h0
        self.h1 = h1
        self.d = d
        self.k = k

        #W0: W = (d x h0), b = (h0,), alpha = 1 / sqrt(d)
        alpha0 = 1 / math.sqrt(d)
        self.weights0 = Parameter(torch.empty(d, h0).uniform_(-alpha0, alpha0))
        self.bias0 = Parameter(torch.empty(h0).uniform_(-alpha0, alpha0))

        #W1: W = (h0 x h1), b = (h1,), alpha = 1 / sqrt(h0)
        alpha1 = 1 / math.sqrt(h0)
        #sets weights between [-alpha, alpha + 1), clamps to [-alpha, alpha] in case it exceeds alpha but is less than alpha + 1
        self.weights1 = Parameter(torch.empty(h0, h1).uniform_(-alpha1, alpha1))
        self.bias1 = Parameter(torch.empty(h1).uniform_(-alpha1, alpha1))

        #W2: W = (h1 x k), b = (k,), alpha = 1 / sqrt(h1)
        alpha2 = 1 / math.sqrt(h1)
        #sets weights between [-alpha, alpha + 1), clamps to [-alpha, alpha] in case it exceeds alpha but is less than alpha + 1
        self.weights2 = Parameter(torch.empty(h1, k).uniform_(-alpha2, alpha2))
        self.bias2 = Parameter(torch.empty(k).uniform_(-alpha2, alpha2))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(relu(x @ self.weights0 + self.bias0) @ self.weights1 + self.bias1) @ self.weights2 + self.bias2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.

    Returns:
        List[float]: List containing average loss for each epoch.
    """

    num_epochs = 1
    train_losses = []
    current_accuracy = 0.0
    
    while current_accuracy < 0.99:
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            batch_loss = cross_entropy(logits, y_batch)

            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item() #* X_batch.size(0)

            predicted = torch.argmax(logits, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        current_accuracy = correct / total

        print(f"Epoch: {num_epochs}, Train Loss: {train_loss}, Train Accuracy: {current_accuracy}")
        num_epochs += 1
    return train_losses


@problem.tag("hw3-A", start_line=5)
def main():
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64)

    k, d = 10, 784
    lr = 0.001

    #F1 model
    h = 64
    model1 = F1(h, d, k)
    optimizer1 = Adam(model1.parameters(), lr)
    train_losses_1 = train(model1, optimizer1, train_loader)

    plt.figure()
    plt.plot(range(1, len(train_losses_1) + 1), train_losses_1, marker='o')
    plt.title("Training Losses vs Number of Epochs for F1")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()

    #Evaluate
    model1.eval()
    correct = 0
    total = 0
    test_loss_f1 = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model1(X_batch)
            batch_loss = cross_entropy(logits, y_batch)

            test_loss_f1 += batch_loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    test_loss_f1 /= len(test_loader)
    print(f"Test Loss F1: {test_loss_f1}, Accuracy F1: {accuracy}")



    #F2 model
    h0, h1 = 32, 32
    model2 = F2(h0, h1, d, k)
    optimizer2 = Adam(model2.parameters(), lr)
    train_losses_2 = train(model2, optimizer2, train_loader)

    plt.figure()
    plt.plot(range(1, len(train_losses_2) + 1), train_losses_2, marker='o')
    plt.title("Training Losses vs Number of Epochs for F2")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()

    #Evaluate
    model2.eval()
    correct = 0
    total = 0
    test_loss_f2 = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model2(X_batch)
            batch_loss = cross_entropy(logits, y_batch)

            test_loss_f2 += batch_loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    test_loss_f2 /= len(test_loader)
    print(f"Test Loss F2: {test_loss_f2}, Accuracy F2: {accuracy}")


if __name__ == "__main__":
    main()
