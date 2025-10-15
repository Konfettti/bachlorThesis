"""Minimal demonstration of XGBoost on the Iris dataset."""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main() -> None:
    """Train and evaluate XGBoost on the classic Iris dataset."""

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data.astype("float32"),
        iris.target,
        test_size=0.25,
        random_state=42,
        stratify=iris.target,
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(iris.target_names),
        eval_metric="mlogloss",
        max_depth=4,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    target_names = [f"{iris.target_names[idx].title()}" for idx in range(len(iris.target_names))]

    print("XGBoost Iris classification demo")
    print("===============================")
    print(f"Accuracy: {accuracy:.3f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    main()
