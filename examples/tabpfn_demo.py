"""Minimal demonstration of the TabPFN classifier on the Iris dataset."""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier


def main() -> None:
    """Train and evaluate TabPFN on the classic Iris dataset."""

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data.astype("float32"),
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    model = TabPFNClassifier(
    device="cpu",
   )


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
