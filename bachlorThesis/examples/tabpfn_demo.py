"""Minimal demonstration of the TabPFN classifier on the Iris dataset."""

from __future__ import annotations

import argparse
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TabPFN Iris demo.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use the first CUDA device if available (defaults to CPU).",
    )
    return parser.parse_args()


def resolve_device(use_gpu: bool) -> str:
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --gpu but no CUDA device is available.")
        return "cuda"
    return "cpu"


def main() -> None:
    """Train and evaluate TabPFN on the classic Iris dataset."""

    args = parse_args()
    device = resolve_device(args.gpu)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data.astype("float32"),
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    model = TabPFNClassifier(device=device)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    target_names = [f"{iris.target_names[idx].title()}" for idx in range(len(iris.target_names))]

    print("TabPFN Iris classification demo")
    print("===============================")
    print(f"Accuracy: {accuracy:.3f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    main()
