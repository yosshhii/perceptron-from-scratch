"""Command-line interface for training the perceptron."""

import argparse

import matplotlib.pyplot as plt

from perceptron.data import generate_two_clusters, load_iris_binary
from perceptron.model import FloatArray, IntArray, Perceptron
from perceptron.plots import plot_decision_boundary, plot_errors


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train a perceptron on synthetic clusters or the Iris dataset."
    )

    parser.add_argument(
        "--samples", type=int, default=200, help="number of points (clusters only)"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=50, help="seed for reproducibility")
    parser.add_argument(
        "--dataset",
        choices=["clusters", "iris"],
        default="clusters",
        help="dataset to train on",
    )
    parser.add_argument("--plot", action="store_true", help="show plots")

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check argument values and exit with a clear message if invalid."""
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    if args.lr <= 0:
        parser.error("--lr must be positive")

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")


def load_data(args: argparse.Namespace) -> tuple[FloatArray, IntArray]:
    """Generate clusters or load Iris depending on --dataset."""
    if args.dataset == "clusters":
        return generate_two_clusters(n_samples=args.samples, seed=args.seed)
    return load_iris_binary()


def train_model(
    args: argparse.Namespace, X: FloatArray, y: IntArray
) -> tuple[Perceptron, list[int]]:
    """Create a perceptron from CLI options and train it."""
    model = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X, y)

    return model, errors


def show_plots(
    model: Perceptron, errors: list[int], X: FloatArray, y: IntArray
) -> None:
    """Show the decision boundary and the training error curve."""
    plot_errors(errors)
    if X.shape[1] == 2 and model.n_features == 2:
        plot_decision_boundary(model=model, X=X, y=y)
    else:
        print("Decision boundary is only available for 2 features")

    plt.show()


def print_report(
    args: argparse.Namespace,
    model: Perceptron,
    errors: list[int],
    X: FloatArray,
    y: IntArray,
) -> None:
    """Print dataset info, training summary and accuracy."""
    print(f"Dataset: {args.dataset} ({X.shape[0]} samples, {X.shape[1]} features)")
    status = "converged" if errors[-1] == 0 else "not converged"
    epoch_word = "epoch" if len(errors) == 1 else "epochs"
    print(f"Training: {len(errors)} {epoch_word}, {status}")
    accuracy = model.score(X, y)
    print(f"Accuracy: {accuracy:.3f}")


def main(argv: list[str] | None = None) -> None:
    """Run the perceptron CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    X, y = load_data(args)
    model, errors = train_model(args, X, y)
    print_report(args, model, errors, X, y)

    if args.plot:
        show_plots(model, errors, X, y)
