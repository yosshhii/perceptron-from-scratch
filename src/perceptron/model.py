"""Single-layer perceptron for binary classification."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        epochs: int = 50,
    ) -> None:
        """Initialize the perceptron.

        Args:
            n_features: Number of input features.
            learning_rate: Step size for weight updates.
            epochs: Maximum number of passes over the training data.
        """
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: FloatArray = np.zeros(shape=n_features)
        self.bias: float = 0.0

    @staticmethod
    def activate(z: float) -> int:
        """Return 1 if z >= 0, otherwise 0."""
        return 1 if z >= 0 else 0

    def predict(self, x: FloatArray) -> int:
        """Return the predicted class (0 or 1) for a single sample."""
        z = float(self.weights @ x + self.bias)
        return self.activate(z)

    def train(self, X: FloatArray, y: IntArray) -> list[int]:
        """Train the perceptron with the perceptron learning rule.

        Stops early if an epoch finishes without errors.

        Returns:
            Number of misclassified samples in each epoch.
        """
        errors_list: list[int] = []

        for _ in range(self.epochs):
            error_counter = 0

            for i in range(len(X)):
                prediction = self.predict(X[i])
                target = y[i]

                if prediction != target:
                    update = self.learning_rate * (target - prediction)
                    self.weights = self.weights + update * X[i]
                    self.bias = self.bias + update
                    error_counter += 1

            errors_list.append(error_counter)

            if error_counter == 0:
                break

        return errors_list

    def score(self, X: FloatArray, y: IntArray) -> float:
        """Return the fraction of correctly classified samples."""
        if len(X) == 0:
            raise ValueError("X must not be empty")

        correct_counter = 0

        for i in range(len(X)):
            if self.predict(X[i]) == y[i]:
                correct_counter += 1

        return correct_counter / len(X)
