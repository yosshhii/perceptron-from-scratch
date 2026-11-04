"""Integration tests for the data and model modules."""

import pytest

from perceptron.data import generate_two_clusters, load_iris_binary
from perceptron.model import Perceptron


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_perceptron_separates_generated_clusters(seed: int) -> None:
    """Perceptron reaches perfect accuracy on two well-separated clusters."""
    p = Perceptron(n_features=2)
    X, y = generate_two_clusters(n_samples=1000, seed=seed)

    p.train(X, y)

    assert p.score(X, y) == 1.0


def test_perceptron_separates_iris_setosa() -> None:
    """Perceptron perfectly separates setosa from the other Iris species."""
    p = Perceptron(n_features=2)
    X, y = load_iris_binary()

    p.train(X, y)

    assert p.score(X, y) == 1.0
