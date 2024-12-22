from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])
    """
    return (np.outer(x_i, x_j) + 1) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])
    """
    return np.exp(-gamma * ((x_i[:, None] - x_j) ** 2))


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    L(w) = ||K\alpha - y||_2^2 + \lambda\alpha^TK\alpha
    \alpha = (K + \lambda*I)^(-1)y

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = len(x)
    if kernel_function is poly_kernel:
        d = kernel_param
        kernel = poly_kernel(x, x, d)
        alpha = np.linalg.solve(kernel + _lambda * np.eye(n), y)

    if kernel_function is rbf_kernel:
        gamma = kernel_param
        kernel = rbf_kernel(x, x, gamma)
        alpha = np.linalg.solve(kernel + _lambda * np.eye(n), y)

    return alpha

@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    fold_size = len(x) // num_folds
    errors = []
    #loop through each fold, setting [i:i+fold_size] to be the current fold
    for i in range(num_folds - 1):
        validation_X = x[i:i + fold_size]
        validation_y = y[i:i + fold_size]
        X_train = np.concatenate([x[:i], x[i + fold_size:]])
        y_train = np.concatenate([y[:i], y[i + fold_size:]])

        alpha = train(X_train, y_train, kernel_function, kernel_param, _lambda)
        #kernel = (n x m), alpha = (n x 1)
        #kernel^T = (m x n) @ alpha (n x 1) = (m x 1), where m is length of validation set
        predictions = kernel_function(X_train, validation_X, kernel_param).T @ alpha

        MSE = np.mean((predictions - validation_y) ** 2)
        errors.append(MSE)

    return np.mean(errors)

@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.
    """
    random_iterations = 50
    all_pairs = pdist(x.reshape(len(x), 1), metric="euclidean")
    gamma = 1 / np.median(all_pairs)

    best_lambda = None
    min_error = float("inf")
    while random_iterations:
        current_lambda = 10 ** np.random.uniform(-5, -1)
        current_error = cross_validation(x, y, rbf_kernel, gamma, current_lambda, num_folds)

        if current_error < min_error:
            min_error = current_error
            best_lambda = current_lambda

        random_iterations -= 1

    return best_lambda, gamma

@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.
    """
    random_iterations = 50

    best_d = None
    best_lambda = None
    min_error = float("inf")
    while random_iterations:
        current_lambda = 10 ** np.random.uniform(-5, -1)
        current_d = np.random.randint(5, 26)
        current_error = cross_validation(x, y, poly_kernel, current_d, current_lambda, num_folds)
        
        if current_error < min_error:
            min_error = current_error
            best_lambda = current_lambda
            best_d = current_d

        random_iterations -= 1

    return best_lambda, best_d

@problem.tag("hw3-A", start_line=1)
def main():
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")

    fine_grid = np.linspace(0, 1, num=100)
    
    #len(x_30) for LOO, 10 for 10-fold CV
    num_folds = [10, len(x_30)]

    #rbf_kernel training
    _lambda, gamma = rbf_param_search(x_30, y_30, num_folds[0])
    print(f"For the RBF Kernel:\nIdeal Lambda: {_lambda}\nIdeal Gamma: {gamma}\n")
    alpha = train(x_30, y_30, rbf_kernel, gamma, _lambda)
    predictions = rbf_kernel(x_30, fine_grid, gamma).T @ alpha

    #plot predictions
    fine_grid = np.linspace(0, 1, num=100)
    plt.scatter(x_30, y_30, color='red', label="Training Data")
    plt.plot(fine_grid, predictions, color='blue', label="Model Predictions on Fine Grid")
    plt.ylim(-6, 6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("RBF Kernel Ridge Regression Predictions")

    plt.show()

    #poly_kernel training
    _lambda, d = poly_param_search(x_30, y_30, num_folds[0])
    print(f"For the Polynomial Kernel:\nIdeal Lambda: {_lambda}\nIdeal d: {d}\n")
    alpha = train(x_30, y_30, poly_kernel, d, _lambda)
    predictions = poly_kernel(x_30, fine_grid, d).T @ alpha

    #plot predictions
    plt.scatter(x_30, y_30, color='red', label="Training Data")
    plt.plot(fine_grid, predictions, color='blue', label="Model Predictions on Fine Grid")
    plt.ylim(-6, 6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Poly Kernel Ridge Regression Predictions")

    plt.show()


if __name__ == "__main__":
    main()
