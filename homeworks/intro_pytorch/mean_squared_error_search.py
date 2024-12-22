if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].
    """
    all_observations = 0
    correct_observations = 0
    test = 1
    for observation, target in dataloader:
        predictions = model(observation)
        predictions_idx = torch.argmax(predictions, dim=1)
        true_idx = torch.argmax(target, dim=1)       

        #number of correct observations = sum of vector where element 1 means correct obs, 0 means incorrect
        correct_observations += torch.sum(true_idx == predictions_idx).item()
        all_observations += len(target) #number of true observations

    return correct_observations / all_observations

@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
    """
    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=16, shuffle=True)

    lr = 0.01

    #Increase number of epochs for smaller LR as it takes longer to converge
    num_epochs = 200

    results = {}

    #Linear Regression
    #size 2 input, output is final
    #passes into single linear layer
    print("Training the Linear layer")
    model1 = nn.Sequential(LinearLayer(2, 2))
    optimizer = SGDOptimizer(model1.parameters(), lr)
    loss_function = MSELossLayer()
    history = train(train_dataloader, model1, loss_function, optimizer, val_dataloader, num_epochs)
    results["Linear"] = {"train" : history["train"], "val": history["val"], "model": model1}

    

    #One HIdden Layer, size=2, sigmoid activation
    print("\n\nTraining the Hidden layer with Sigmoid")
    model2 = nn.Sequential(LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2))
    optimizer = SGDOptimizer(model2.parameters(), lr)
    loss_function = MSELossLayer()
    history = train(train_dataloader, model2, loss_function, optimizer, val_dataloader, num_epochs)
    results["OneHiddenSigmoid"] = {"train" : history["train"], "val": history["val"], "model": model2}

    #One hidden layer, size=2, ReLU activation
    print("\n\nTraining the Hidden layer with ReLU")
    model3 = nn.Sequential(LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2))
    optimizer = SGDOptimizer(model3.parameters(), lr)
    loss_function = MSELossLayer()
    history = train(train_dataloader, model3, loss_function, optimizer, val_dataloader, num_epochs)
    results["OneHiddenReLU"] = {"train" : history["train"], "val": history["val"], "model": model3}

    #Two hidden layers, size=2, sigmoid, then ReLU
    print("\n\nTraining the Two Hidden layers with Sigmoid then ReLU")
    model4 = nn.Sequential(LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2))
    optimizer = SGDOptimizer(model4.parameters(), lr)
    loss_function = MSELossLayer()
    history = train(train_dataloader, model4, loss_function, optimizer, val_dataloader, num_epochs)
    results["TwoHiddenSigmoidReLU"] = {"train" : history["train"], "val": history["val"], "model": model4}

    #Two hidden layers, size=2, ReLU then sigmoid
    print("\n\nTraining the Two Hidden layers with ReLU then Sigmoid")
    model5 = nn.Sequential(LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2))
    optimizer = SGDOptimizer(model5.parameters(), lr)
    loss_function = MSELossLayer()
    history = train(train_dataloader, model5, loss_function, optimizer, val_dataloader, num_epochs)
    results["TwoHiddenReLUSigmoid"] = {"train" : history["train"], "val": history["val"], "model": model5}

    return results


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )

    mse_configs = mse_parameter_search(dataset_train, dataset_val)

    best_model_name = None
    best_model = None
    best_val_loss = float('inf')
    best_epoch = None

    for name, results in mse_configs.items():
        #get minimum validation loss and epoch num
        min_val_loss = min(results["val"])
        epoch = results["val"].index(min_val_loss) + 1

        #get minimum val_loss
        if best_val_loss > min_val_loss: 
            best_val_loss = min_val_loss
            best_model = results["model"]
            best_epoch = epoch
            best_model_name = name

        epochs = range(1, len(results["train"]) + 1)
        plt.plot(epochs, results["train"], label = f"{name} Train")
        plt.plot(epochs, results["val"], label = f"{name} Val", linestyle='--')

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error Loss")
    plt.title("Training and Validation Loss for MSE")
    plt.show()

    print(f"The best model was {best_model_name} with a loss of {best_val_loss} on epoch {best_epoch}")

    test_dataloader = DataLoader(dataset_test, batch_size=16, shuffle=True)
    plot_model_guesses(test_dataloader, best_model, f"Model Guesses of {best_model_name}")
    accuracy_of_best = accuracy_score(best_model, test_dataloader)
    print(f"The best accuracy was: {accuracy_of_best}")

    print(len(list(best_model.parameters())))


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
