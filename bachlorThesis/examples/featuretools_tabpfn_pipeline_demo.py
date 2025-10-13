from __future__ import annotations
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="The provided callable <function mean")
warnings.filterwarnings("ignore", category=FutureWarning, message="The provided callable <function sum")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Could not infer format")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier

from ..featuretools_tabpfn_adapter import (
    FeaturetoolsTabPFNAdapter,
    FeaturetoolsTabPFNConfig,
)
import featuretools as ft


def make_data() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6],
        "join_date": pd.to_datetime([
            "2023-01-01","2023-01-05","2023-01-10",
            "2023-01-15","2023-02-01","2023-02-05"
        ]),
        "country": ["DE","US","DE","US","DE","US"],
        "churn":   [0,1,0,1,0,1],
    })

    sessions = pd.DataFrame({
        "session_id":  [10,11,12,13,14,15,16,17],
        "customer_id": [1, 2, 2, 3, 4, 4, 5, 6],
        "session_start": pd.to_datetime([
            "2023-02-01 08:30","2023-02-03 14:00","2023-02-10 09:15","2023-02-12 16:45",
            "2023-02-14 10:10","2023-02-14 18:22","2023-02-15 09:00","2023-02-16 12:30",
        ]),
    })

    transactions = pd.DataFrame({
        "transaction_id": [100,101,102,103,104,105,106,107,108],
        "session_id":     [10,10,11,12,13,14,15,16,17],
        "amount":         [10.0,25.5,12.0,18.5,9.0,42.0,7.5,13.0,22.0],
        "product":        ["A","B","A","C","B","A","C","C","B"],
    })

    return {
        "customers": customers,
        "sessions": sessions,
        "transactions": transactions,
    }


def builder(dataframes: dict[str, pd.DataFrame]) -> ft.EntitySet:
    es = ft.EntitySet(id="customer_data")
    es = es.add_dataframe(
        dataframe_name="customers",
        dataframe=dataframes["customers"],
        index="customer_id",
    )
    es = es.add_dataframe(
        dataframe_name="sessions",
        dataframe=dataframes["sessions"],
        index="session_id",
    )
    es = es.add_dataframe(
        dataframe_name="transactions",
        dataframe=dataframes["transactions"],
        index="transaction_id",
    )
    es = es.add_relationship("customers", "customer_id", "sessions", "customer_id")
    es = es.add_relationship("sessions", "session_id", "transactions", "session_id")
    return es


def main() -> None:
    dfs = make_data()
    y = dfs["customers"]["churn"].astype("int64")

    cfg = FeaturetoolsTabPFNConfig(
        target_dataframe_name="customers",
        agg_primitives=["sum", "mean", "count"],
        trans_primitives=["month", "weekday"],
        max_depth=2,
        verbose=False,
        dtype="float32",
    )
    adapter = FeaturetoolsTabPFNAdapter(entityset_builder=builder, config=cfg)

    X_map = {"dataframes": dfs}
    X_feat = adapter.fit_transform(X_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y.values, test_size=0.33, random_state=42, stratify=y.values
    )

    clf = TabPFNClassifier(device="cpu", n_estimators=32)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
