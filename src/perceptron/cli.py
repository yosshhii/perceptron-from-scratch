"""Command-line interface for training the perceptron."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from perceptron.data import generate_two_clusters, load_iris_binary, train_test_split
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
    parser.add_argument(
        "--offset",
        type=float,
        default=5.0,
        help="cluster center offset (clusters only)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=1.0,
        help="cluster standard deviation (clusters only)",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="fraction of samples assigned to the test set (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for reproducibility")
    parser.add_argument(
        "--dataset",
        choices=["clusters", "iris"],
        default="clusters",
        help="dataset to train on",
    )
    parser.add_argument("--plot", action="store_true", help="show plots")
    parser.add_argument("--save-dir", type=Path, help="directory to save plots as PNG")

    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check argument values and exit with a clear message if invalid."""
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    if args.spread < 0:
        parser.error("--spread must be non-negative")

    if args.lr <= 0:
        parser.error("--lr must be positive")

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if not 0 < args.test_size < 1:
        parser.error("--test-size must be strictly between 0 and 1")


def load_data(args: argparse.Namespace) -> tuple[FloatArray, IntArray]:
    """Generate clusters or load Iris depending on --dataset."""
    if args.dataset == "clusters":
        return generate_two_clusters(
            n_samples=args.samples,
            seed=args.seed,
            offset=args.offset,
            spread=args.spread,
        )
    return load_iris_binary()


def split_data(
    args: argparse.Namespace, X: FloatArray, y: IntArray
) -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    """Split the dataset into training and test sets using CLI options."""
    return train_test_split(X, y, test_size=args.test_size, seed=args.seed)


def train_model(
    args: argparse.Namespace, X: FloatArray, y: IntArray
) -> tuple[Perceptron, list[int]]:
    """Create a perceptron from CLI options and train it."""
    model = Perceptron(n_features=X.shape[1], learning_rate=args.lr, epochs=args.epochs)
    errors = model.train(X, y)

    return model, errors


# def show_plots(
#     model: Perceptron, errors: list[int], X: FloatArray, y: IntArray
# ) -> None:
#     """Show the decision boundary and the training error curve."""
#     plot_errors(errors)
#     if X.shape[1] == 2 and model.n_features == 2:
#         plot_decision_boundary(model=model, X=X, y=y)
#     else:
#         print("Decision boundary is only available for 2 features")
#
#     plt.show()


def make_figures(
    model: Perceptron, errors: list[int], X: FloatArray, y: IntArray
) -> dict[str, Figure]:
    """Build plots and return them keyed by file name."""
    figures = {"errors": plot_errors(errors)}
    if X.shape[1] == 2 and model.n_features == 2:
        figures["boundary"] = plot_decision_boundary(
            model=model, X=X, y=y, title="Decision boundary on the test set"
        )
    else:
        print("Decision boundary is only available for 2 features")

    return figures


def save_figures(figures: dict[str, Figure], save_dir: Path) -> None:
    """Save figures as PNG files into save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        filename = f"{name}.png"
        file_path = save_dir / filename

        fig.savefig(file_path, bbox_inches="tight")


def print_report(
    args: argparse.Namespace,
    model: Perceptron,
    errors: list[int],
    X: FloatArray,
    y: IntArray,
) -> None:
    """Print dataset info, training summary and accuracy."""
    print(f"Test set: {args.dataset} ({X.shape[0]} samples, {X.shape[1]} features)")
    status = "converged" if errors[-1] == 0 else "not converged"
    epoch_word = "epoch" if len(errors) == 1 else "epochs"
    print(f"Training: {len(errors)} {epoch_word}, {status}")
    accuracy = model.score(X, y)
    print(f"Test accuracy: {accuracy:.3f}")


def main(argv: list[str] | None = None) -> None:
    """Run the perceptron CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    X, y = load_data(args)
    try:
        X_train, X_test, y_train, y_test = split_data(args, X, y)
    except ValueError as exc:
        parser.error(str(exc))
    model, errors = train_model(args, X_train, y_train)
    print_report(args, model, errors, X_test, y_test)

    if args.plot or args.save_dir:
        figures = make_figures(model, errors, X_test, y_test)

        if args.save_dir:
            save_figures(figures, args.save_dir)
            print(f"Saved plots to {args.save_dir}")

        if args.plot:
            plt.show()
