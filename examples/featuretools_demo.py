"""Minimal script to verify that Featuretools is installed and working.

The script builds a simple relational dataset, runs Deep Feature Synthesis (DFS),
and prints the generated feature matrix.
"""

from __future__ import annotations

import pandas as pd

import featuretools as ft


def build_sample_entityset() -> ft.EntitySet:
    """Create a small entity set with customers, sessions, and transactions."""
    customers_df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "join_date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-01"]),
            "country": ["DE", "US", "DE"],
        }
    )

    sessions_df = pd.DataFrame(
        {
            "session_id": [10, 11, 12, 13],
            "customer_id": [1, 2, 2, 3],
            "session_start": pd.to_datetime(
                [
                    "2023-02-01 08:30",
                    "2023-02-03 14:00",
                    "2023-02-10 09:15",
                    "2023-02-12 16:45",
                ]
            ),
        }
    )

    transactions_df = pd.DataFrame(
        {
            "transaction_id": [100, 101, 102, 103, 104, 105],
            "session_id": [10, 10, 11, 12, 13, 13],
            "amount": [10.0, 25.5, 12.0, 18.5, 9.0, 42.0],
            "product": ["A", "B", "A", "C", "B", "A"],
        }
    )

    es = ft.EntitySet(name="customer_data")
    es = es.add_dataframe(dataframe_name="customers", dataframe=customers_df, index="customer_id")
    es = es.add_dataframe(dataframe_name="sessions", dataframe=sessions_df, index="session_id")
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_df,
        index="transaction_id",
        time_index=None,
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


def run_deep_feature_synthesis(entityset: ft.EntitySet) -> pd.DataFrame:
    """Run DFS to generate features for the customers dataframe."""
    feature_matrix, feature_defs = ft.dfs(
        entityset=entityset,
        target_dataframe_name="customers",
        agg_primitives=["sum", "mean", "count"],
        trans_primitives=["month", "weekday"],
        max_depth=2,
        verbose=True,
    )

    print("Generated feature definitions:\n")
    for feature in feature_defs:
        print(f" - {feature.get_name()}")

    print("\nFeature matrix:\n")
    print(feature_matrix.head())

    return feature_matrix


def main() -> None:
    entityset = build_sample_entityset()
    run_deep_feature_synthesis(entityset)


if __name__ == "__main__":
    main()
