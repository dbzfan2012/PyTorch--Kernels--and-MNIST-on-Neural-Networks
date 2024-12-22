if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from losses import CrossEntropyLossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizers import SGDOptimizer
    from .losses import CrossEntropyLossLayer
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
def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    Trains and evaluates multiple different kinds of models

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
    """
    train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=False)
    val_dataloader = DataLoader(dataset_val, batch_size=64, shuffle=False)

    lr = 0.03

    #Increase number of epochs for smaller LR as it takes longer to converge
    num_epochs = 500

    results = {}

    #Linear Regression
    #size 2 input, output is final
    #passes into single linear layer
    print("Training the Linear layer")
    model1 = nn.Sequential(LinearLayer(2, 2), SoftmaxLayer())
    optimizer = SGDOptimizer(model1.parameters(), lr)
    loss_function = CrossEntropyLossLayer()
    history = train(train_dataloader, model1, loss_function, optimizer, val_dataloader, num_epochs)
    results["Linear"] = {"train" : history["train"], "val": history["val"], "model": model1}

    #One HIdden Layer, size=2, sigmoid activation
    print("\n\nTraining the Hidden layer with Sigmoid")
    model2 = nn.Sequential(LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), SoftmaxLayer())
    optimizer = SGDOptimizer(model2.parameters(), lr)
    loss_function = CrossEntropyLossLayer()
    history = train(train_dataloader, model2, loss_function, optimizer, val_dataloader, num_epochs)
    results["OneHiddenSigmoid"] = {"train" : history["train"], "val": history["val"], "model": model2}

    #One hidden layer, size=2, ReLU activation
    print("\n\nTraining the Hidden layer with ReLU")
    model3 = nn.Sequential(LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SoftmaxLayer())
    optimizer = SGDOptimizer(model3.parameters(), lr)
    loss_function = CrossEntropyLossLayer()
    history = train(train_dataloader, model3, loss_function, optimizer, val_dataloader, num_epochs)
    results["OneHiddenReLU"] = {"train" : history["train"], "val": history["val"], "model": model3}

    #Two hidden layers, size=2, sigmoid, then ReLU
    print("\n\nTraining the Two Hidden layers with Sigmoid then ReLU")
    model4 = nn.Sequential(LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SoftmaxLayer())
    optimizer = SGDOptimizer(model4.parameters(), lr)
    loss_function = CrossEntropyLossLayer()
    history = train(train_dataloader, model4, loss_function, optimizer, val_dataloader, num_epochs)
    results["TwoHiddenSigmoidReLU"] = {"train" : history["train"], "val": history["val"], "model": model4}

    #Two hidden layers, size=2, ReLU then sigmoid
    print("\n\nTraining the Two Hidden layers with ReLU then Sigmoid")
    model5 = nn.Sequential(LinearLayer(2, 2), ReLULayer(), LinearLayer(2, 2), SigmoidLayer(), LinearLayer(2, 2), SoftmaxLayer())
    optimizer = SGDOptimizer(model5.parameters(), lr)
    loss_function = CrossEntropyLossLayer()
    history = train(train_dataloader, model5, loss_function, optimizer, val_dataloader, num_epochs)
    results["TwoHiddenReLUSigmoid"] = {"train" : history["train"], "val": history["val"], "model": model5}

    return results


@problem.tag("hw3-A")
def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].
    """
    all_observations = 0
    correct_observations = 0
    with torch.no_grad():
        for observation, target in dataloader:
            predictions = model(observation)

            predictions_idx = torch.argmax(predictions, dim=1) 

            correct_observations += (target == predictions_idx).sum().item()
            all_observations += target.size(0)

    return correct_observations / all_observations


@problem.tag("hw3-A", start_line=7)
def main():
    """
    Main function of the Crossentropy problem.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)
    best_model_name = None
    best_model = None
    best_val_loss = float('inf')
    best_epoch = None

    for name, results in ce_configs.items():
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
        plt.plot(epochs, results["val"], label = f"{name} Val")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training and Validation Loss for Cross Entropy")
    plt.show()

    print(f"The best model was {best_model_name} with a loss of {best_val_loss} on epoch {best_epoch}")

    test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    plot_model_guesses(test_dataloader, best_model, f"Model Guesses of {best_model_name}")
    accuracy_of_best = accuracy_score(best_model, test_dataloader)
    print(f"The best accuracy was: {accuracy_of_best}")


if __name__ == "__main__":
    main()
