"""Synthetic dataset generation and Iris loading."""

import numpy as np
from sklearn.datasets import load_iris

from perceptron.model import FloatArray, IntArray


def generate_two_clusters(
    n_samples: int,
    seed: int | None = None,
    offset: float = 5.0,
    spread: float = 1.0,
) -> tuple[FloatArray, IntArray]:
    """Generate two clusters of 2D points.

    Args:
        n_samples: Total number of points (split evenly between classes).
        seed: Random seed for reproducibility.
        offset: Each center is placed at (offset, offset) or
            (-offset, -offset).
        spread: Standard deviation of points around each center.
            Must be non-negative.

    Returns:
        Shuffled feature matrix of shape (n_samples, 2) and labels (0 or 1).

    Raises:
        ValueError: If n_samples is less than 2 or spread is negative.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    if spread < 0:
        raise ValueError("spread must be non-negative")

    n_first_cluster = n_samples // 2
    n_second_cluster = n_samples - n_first_cluster

    first_center = [offset, offset]  # class 0
    second_center = [-offset, -offset]  # class 1

    rng = np.random.default_rng(seed)

    first_cluster_samples = rng.normal(
        loc=first_center, scale=spread, size=(n_first_cluster, 2)
    )
    second_cluster_samples = rng.normal(
        loc=second_center, scale=spread, size=(n_second_cluster, 2)
    )

    samples = np.vstack([first_cluster_samples, second_cluster_samples])
    classes = np.concatenate(
        [
            np.zeros(n_first_cluster, dtype=np.int_),
            np.ones(n_second_cluster, dtype=np.int_),
        ]
    )

    indices = rng.permutation(n_samples)
    shuffled_samples = samples[indices]
    shuffled_classes = classes[indices]

    return shuffled_samples, shuffled_classes


def load_iris_binary() -> tuple[FloatArray, IntArray]:
    """Load Iris as a binary task: setosa vs the other species (versicolor, virginica).

    Uses two features: petal length and petal width (cm).

    Returns:
        Feature matrix of shape (150, 2) and labels (1 for setosa, 0 otherwise).
    """
    iris = load_iris()
    dataset = np.asarray(iris.data[:, [2, 3]], dtype=np.float64)
    target = np.asarray(iris.target == 0, dtype=np.int_)

    return dataset, target
