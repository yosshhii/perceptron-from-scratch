"""Visualization: decision boundary and training error curve."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from perceptron.model import FloatArray, IntArray, Perceptron


def plot_decision_boundary(
    model: Perceptron,
    X: FloatArray,
    y: IntArray,
    title: str = "Decision boundary",
) -> Figure:
    """Plot 2D samples colored by class and the perceptron's decision boundary.

    The boundary is the line w0*x1 + w1*x2 + b = 0. A vertical line is drawn
    when w1 is zero; no line is drawn when both weights are zero.

    Args:
        model: Trained perceptron with two features.
        X: Feature matrix of shape (n_samples, 2).
        y: Class labels (0 or 1).
        title: Plot title.

    Returns:
        Figure with the samples and the decision boundary.

    Raises:
        ValueError: If X or the model does not have exactly 2 features.
    """
    if X.shape[1] != 2 or model.n_features != 2:
        raise ValueError("Decision boundary can only be plotted for 2 features")

    w0, w1 = model.weights
    b = model.bias

    fig, ax = plt.subplots()

    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="class 1")

    ax.set_xlabel("x1", color="gray")
    ax.set_ylabel("x2", color="gray")
    ax.set_title(title)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if w1 != 0:
        xs = np.linspace(xlim[0], xlim[1], 100)
        ys = -(w0 * xs + b) / w1
        ax.plot(xs, ys, label="decision boundary", color="black")

    elif w0 != 0:
        x_boundary = -b / w0
        ax.axvline(x_boundary, label="decision boundary", color="black")

    else:
        ax.set_title(f"{title} (no boundary: zero weights)")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend()

    return fig


def plot_errors(errors: list[int]) -> Figure:
    """Plot misclassified samples per epoch and return the figure."""
    fig, ax = plt.subplots()
    ax.plot(range(1, len(errors) + 1), errors, marker="o", color="red")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.4)
    ax.set_xlabel("Epoch", color="gray")
    ax.set_ylabel("Misclassified samples", color="gray")
    ax.set_title("Training errors")

    return fig
