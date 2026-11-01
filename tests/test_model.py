"""Tests for the Perceptron model."""

import numpy as np
import pytest

from perceptron.model import Perceptron


@pytest.mark.parametrize(
    ("z", "expected"), [(0.0, 1), (-1.0, 0), (5.0, 1), (-0.1, 0), (-5.0, 0)]
)
def test_activate(z: float, expected: int) -> None:
    """Activate applies the step threshold."""
    assert Perceptron.activate(z) == expected


def test_init_weights_length() -> None:
    """Weights have one entry per feature."""
    p = Perceptron(n_features=3)
    assert len(p.weights) == 3


def test_init_stores_hyperparameters() -> None:
    """Constructor stores learning rate and epochs."""
    p = Perceptron(n_features=2, learning_rate=0.5, epochs=100)

    assert p.learning_rate == 0.5
    assert p.epochs == 100


def test_init_starts_from_zero() -> None:
    """Weights and bias start at zero."""
    p = Perceptron(n_features=5)

    assert np.all(p.weights == 0)
    assert p.bias == 0.0


def test_predict() -> None:
    """Predict uses the sign of w·x + b."""
    p = Perceptron(n_features=3)
    p.weights = np.array([1.0, 1.0, 1.0])
    p.bias = 0.0

    X = np.array(
        [
            [2.0, 2.0, 2.0],
            [-2.0, -2.0, -2.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ]
    )
    y = np.array([1, 0, 1, 1, 0])

    for i in range(len(X)):
        assert p.predict(X[i]) == y[i], f"failed on {X[i]}"


def test_train_separates_classes() -> None:
    """Training reaches perfect accuracy on linearly separable data."""
    p = Perceptron(n_features=3)

    X = np.array(
        [
            [5.0, 5.0, 5.0],
            [6.5, 6.0, 7.0],
            [10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0],
            [0.0, 0.0, 0.0],
            [-2.0, -2.0, -2.0],
        ]
    )
    y = np.array([1, 1, 1, 0, 0, 0])

    errors = p.train(X, y)

    assert p.score(X, y) == 1.0
    assert errors[-1] == 0


def test_score_raises_on_empty_input() -> None:
    """Score rejects an empty dataset."""
    p = Perceptron(n_features=3)

    X = np.array([])
    y = np.array([])

    with pytest.raises(ValueError, match="empty"):
        p.score(X, y)
