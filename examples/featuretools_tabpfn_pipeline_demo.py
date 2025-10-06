"""Demonstrate a full pipeline that combines Featuretools with TabPFN."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tabpfn import TabPFNClassifier

from featuretools_tabpfn_adapter import (
    FeaturetoolsTabPFNAdapter,
    FeaturetoolsTabPFNConfig,
)

import featuretools as ft


def build_retail_data() -> Mapping[str, pd.DataFrame]:
    rng = np.random.default_rng(seed=42)

    customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "join_date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-03",
                    "2023-01-08",
                    "2023-02-02",
                    "2023-02-09",
                    "2023-02-20",
                ]
            ),
            "country": ["DE", "US", "DE", "FR", "US", "DE"],
            "high_value": [1, 0, 1, 0, 0, 1],
        }
    )

    sessions = pd.DataFrame(
        {
            "session_id": list(range(100, 112)),
            "customer_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "session_start": pd.to_datetime(
                [
                    "2023-02-01 08:30",
                    "2023-02-01 16:30",
                    "2023-02-03 14:00",
                    "2023-02-04 09:15",
                    "2023-02-10 09:15",
                    "2023-02-12 16:45",
                    "2023-02-13 10:05",
                    "2023-02-14 11:20",
                    "2023-02-15 12:00",
                    "2023-02-16 13:15",
                    "2023-02-17 15:25",
                    "2023-02-18 09:45",
                ]
            ),
        }
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": list(range(1, 37)),
            "session_id": rng.choice(sessions["session_id"], size=36, replace=True),
            "amount": rng.gamma(shape=2.0, scale=15.0, size=36).round(2),
            "product": rng.choice(list("ABC"), size=36),
        }
    )

    return {
        "customers": customers,
        "sessions": sessions,
        "transactions": transactions,
    }


def build_entityset(dataframes: Mapping[str, pd.DataFrame]) -> ft.EntitySet:
    es = ft.EntitySet(id="retail")
    es = es.add_dataframe(
        dataframe_name="customers",
        dataframe=dataframes["customers"].drop(columns=["high_value"]),
        index="customer_id",
        time_index="join_date",
    )
    es = es.add_dataframe(
        dataframe_name="sessions",
        dataframe=dataframes["sessions"],
        index="session_id",
        time_index="session_start",
    )
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=dataframes["transactions"],
        index="transaction_id",
    )
    es = es.add_relationship(
        parent_dataframe_name="customers",
        parent_column_name="customer_id",
        child_dataframe_name="sessions",
        child_column_name="customer_id",
    )
    es = es.add_relationship(
        parent_dataframe_name="sessions",
        parent_column_name="session_id",
        child_dataframe_name="transactions",
        child_column_name="session_id",
    )
    return es


def main() -> None:
    dataframes = build_retail_data()
    labels = dataframes["customers"].set_index("customer_id")["high_value"]

    train_ids, test_ids = train_test_split(
        labels.index.to_numpy(),
        test_size=0.33,
        stratify=labels,
        random_state=42,
    )
    y_train = labels.loc[train_ids].to_numpy()
    y_test = labels.loc[test_ids].to_numpy()

    adapter = FeaturetoolsTabPFNAdapter(
        entityset_builder=build_entityset,
        config=FeaturetoolsTabPFNConfig(
            target_dataframe_name="customers",
            agg_primitives=["sum", "mean", "count"],
            trans_primitives=["month", "weekday"],
            max_depth=2,
            verbose=False,
        ),
    )

    pipeline = Pipeline(
        steps=[
            ("featuretools", adapter),
            (
                "tabpfn",
                TabPFNClassifier(device="cpu"),
            ),
        ]
    )

    pipeline.fit(
        {
            "dataframes": dataframes,
            "target_ids": train_ids,
        },
        y_train,
    )

    y_pred = pipeline.predict(
        {
            "dataframes": dataframes,
            "target_ids": test_ids,
        }
    )

    print("Featuretools + TabPFN pipeline demo")
    print("==================================")
    print(f"Train IDs: {sorted(train_ids.tolist())}")
    print(f"Test IDs: {sorted(test_ids.tolist())}\n")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Low value", "High value"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
