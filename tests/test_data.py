"""Tests for the data module."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from perceptron.data import generate_two_clusters, load_iris_binary


@pytest.mark.parametrize("n_samples", [2, 100, 101])
def test_generate_two_clusters_shape(n_samples: int) -> None:
    """Generator returns n_samples points with 2 features and one label each."""
    X, y = generate_two_clusters(n_samples)

    assert X.shape == (n_samples, 2)
    assert y.shape == (n_samples,)


def test_generate_two_clusters_labels_are_binary() -> None:
    """All generated labels are either 0 or 1."""
    _, y = generate_two_clusters(100)

    assert np.array_equal(np.unique(y), [0, 1])


def test_generate_two_clusters_classes_are_balanced() -> None:
    """Generator splits samples between classes as evenly as possible."""
    n_samples = 101
    _, y = generate_two_clusters(n_samples)

    counter_1 = np.sum(y == 1)

    assert counter_1 in [n_samples // 2, n_samples - n_samples // 2]


def test_generate_two_clusters_is_reproducible() -> None:
    """Same seed produces identical points and labels."""
    seed = 1
    n_samples = 101

    X1, y1 = generate_two_clusters(n_samples, seed=seed)
    X2, y2 = generate_two_clusters(n_samples, seed=seed)

    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_generate_two_clusters_different_seeds_differ() -> None:
    """Different seeds produce different points."""
    seed1 = 1
    seed2 = 2
    n_samples = 101

    X1, _ = generate_two_clusters(n_samples, seed=seed1)
    X2, _ = generate_two_clusters(n_samples, seed=seed2)

    assert not np.array_equal(X1, X2)


@pytest.mark.parametrize("n_samples", [-5, 0, 1])
def test_generate_two_clusters_raises_on_too_few_samples(n_samples: int) -> None:
    """Generator rejects fewer than 2 samples."""
    with pytest.raises(ValueError, match="samples"):
        generate_two_clusters(n_samples)


def test_generate_two_clusters_raises_on_negative_spread() -> None:
    """Generator rejects a negative spread."""
    spread = -5

    with pytest.raises(ValueError, match="spread"):
        generate_two_clusters(101, spread=spread)


def test_load_iris_binary_shapes() -> None:
    """Loader returns 150 samples with 2 features and one label each."""
    X, y = load_iris_binary()

    assert X.shape == (150, 2)
    assert y.shape == (150,)


def test_load_iris_binary_labels_are_binary() -> None:
    """All loaded labels are either 0 or 1."""
    _, y = load_iris_binary()

    assert np.array_equal(np.unique(y), [0, 1])


def test_load_iris_binary_marks_setosa_as_positive() -> None:
    """Exactly 50 samples (all setosa) are labeled 1."""
    _, y = load_iris_binary()

    # In the Iris dataset class 0 is setosa
    assert np.array_equal(y == 1, load_iris().target == 0)
