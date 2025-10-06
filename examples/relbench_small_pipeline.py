"""Run a small RelBench experiment on the rel-event dataset.

The script downloads the small user-attendance regression task from RelBench,
engineers lightweight user-level features by aggregating over a handful of
supporting tables, and trains a scikit-learn baseline on a capped number of
rows. The goal is to offer a reproducible starting point for running initial
experiments on RelBench with modest hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from relbench.base.table import Table
from relbench.tasks import BaseTask, get_task

pd.options.mode.copy_on_write = True


@dataclass
class PreparedSplits:
    """Container that holds feature matrices and targets for each split."""

    features: Dict[str, pd.DataFrame]
    targets: Dict[str, Optional[pd.Series]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight RelBench pipeline.")
    parser.add_argument(
        "--dataset",
        default="rel-event",
        help="RelBench dataset name (default: rel-event)",
    )
    parser.add_argument(
        "--task",
        default="user-attendance",
        help="Task name within the selected dataset (default: user-attendance)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5000,
        help="Maximum number of rows to keep per split (use None to keep all rows)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the random forest baseline (default: 200)",
    )
    return parser.parse_args()


def load_task(dataset_name: str, task_name: str) -> BaseTask:
    """Download (if required) and return the requested RelBench task."""

    print(f"Loading task '{task_name}' from dataset '{dataset_name}' ...")
    task = get_task(dataset_name, task_name, download=True)
    print("Task metadata:")
    print(f" - task type: {task.task_type}")
    print(f" - target column: {task.target_col}")
    print(f" - entity column: {task.entity_col}")
    return task


def fetch_split_tables(task: BaseTask) -> Dict[str, pd.DataFrame]:
    """Load train/val/test tables from disk as pandas DataFrames."""

    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        table: Table = task.get_table(split_name)
        df = table.df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=[task.entity_col])
        splits[split_name] = df
        print(f"Loaded {split_name} split with {len(df):,} rows")
    return splits


def limit_rows_per_split(splits: Dict[str, pd.DataFrame], max_rows: int | None) -> None:
    """Trim each split to at most ``max_rows`` examples for quicker experiments."""

    if not max_rows:
        return

    for split_name, df in splits.items():
        if len(df) > max_rows:
            splits[split_name] = df.sort_values("timestamp").head(max_rows).reset_index(drop=True)
            print(f" - trimmed {split_name} split to {max_rows:,} rows")


def collect_user_ids(splits: Dict[str, pd.DataFrame], entity_col: str) -> Set[int]:
    """Return the unique user identifiers across all splits."""

    user_ids: Set[int] = set()
    for df in splits.values():
        user_ids.update(df[entity_col].astype("Int64").dropna().astype(int).unique().tolist())
    print(f"Found {len(user_ids):,} unique entities in the capped splits")
    return user_ids


def compute_user_features(task: BaseTask, user_ids: Iterable[int]) -> pd.DataFrame:
    """Engineer lightweight user-level features from the relational database."""

    dataset = task.dataset
    db = dataset.get_db(upto_test_timestamp=True)
    tables = db.table_dict

    index = pd.Index(sorted(set(user_ids)), name=task.entity_col)
    features = pd.DataFrame(index=index)

    # Basic user metadata ---------------------------------------------------
    user_table = tables["users"].df
    user_subset = user_table[user_table["user_id"].isin(index)].copy()
    user_subset["joinedAt"] = pd.to_datetime(user_subset["joinedAt"], errors="coerce")
    reference_ts = pd.Timestamp(dataset.test_timestamp)
    user_subset["account_age_days"] = (reference_ts - user_subset["joinedAt"]).dt.days

    numeric_cols = ["birthyear", "timezone", "account_age_days"]
    features = features.join(
        user_subset.set_index("user_id")[numeric_cols].apply(pd.to_numeric, errors="coerce"),
        how="left",
    )

    cat_df = user_subset.set_index("user_id")["locale"].fillna("unknown").to_frame()
    top_locales = cat_df["locale"].value_counts().nlargest(5).index
    cat_df.loc[~cat_df["locale"].isin(top_locales), "locale"] = "other"
    locale_dummies = pd.get_dummies(cat_df["locale"], prefix="locale")

    gender_df = user_subset.set_index("user_id")["gender"].fillna("unknown")
    gender_dummies = pd.get_dummies(gender_df, prefix="gender")

    features = features.join(locale_dummies, how="left")
    features = features.join(gender_dummies, how="left")

    # User friendship network ------------------------------------------------
    friends_df = tables["user_friends"].df
    friend_counts = (
        friends_df[friends_df["user"].isin(index) & friends_df["friend"].notna()]
        .groupby("user")
        .size()
        .rename("friend_count")
    )
    features = features.join(friend_counts, how="left")

    # Event interest logs ----------------------------------------------------
    interest_df = tables["event_interest"].df
    interest_agg = (
        interest_df[interest_df["user"].isin(index)]
        .groupby("user")[["invited", "interested", "not_interested"]]
        .sum()
    )
    features = features.join(interest_agg, how="left")

    # Event attendance summaries --------------------------------------------
    attendees_df = tables["event_attendees"].df.dropna(subset=["user_id"])
    attendees_df = attendees_df[attendees_df["user_id"].isin(index)]
    if not attendees_df.empty:
        status_dummies = pd.get_dummies(attendees_df["status"], prefix="attend")
        status_agg = status_dummies.groupby(attendees_df["user_id"]).sum()
        features = features.join(status_agg, how="left")

    features = features.fillna(0.0)
    print(f"Constructed feature matrix with shape {features.shape}")
    return features


def assemble_model_ready_splits(
    task: BaseTask,
    splits: Dict[str, pd.DataFrame],
    features: pd.DataFrame,
) -> PreparedSplits:
    """Merge engineered features into each split and extract X/y arrays."""

    prepared_features: Dict[str, pd.DataFrame] = {}
    targets: Dict[str, Optional[pd.Series]] = {}
    reference_ts = pd.Timestamp(task.dataset.test_timestamp)

    for split_name, df in splits.items():
        enriched = df.copy()
        enriched["days_until_test"] = (reference_ts - enriched["timestamp"]).dt.days
        enriched = enriched.merge(
            features,
            left_on=task.entity_col,
            right_index=True,
            how="left",
        )
        enriched = enriched.drop(columns=["index"], errors="ignore").fillna(0.0)

        feature_columns = [
            col
            for col in enriched.columns
            if col not in {task.target_col, task.entity_col, "timestamp"}
        ]
        prepared_features[split_name] = enriched[feature_columns]
        if task.target_col in enriched.columns:
            targets[split_name] = enriched[task.target_col]
            target_shape = targets[split_name].shape
        else:
            targets[split_name] = None
            target_shape = None

        print(
            f"Prepared {split_name} split: X shape = {prepared_features[split_name].shape}, "
            f"y shape = {target_shape if target_shape is not None else 'unlabelled'}"
        )

    return PreparedSplits(prepared_features, targets)


def train_and_evaluate(
    prepared: PreparedSplits,
    n_estimators: int,
) -> None:
    """Fit a baseline model and report metrics for each split."""

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    model.fit(prepared.features["train"], prepared.targets["train"])

    for split_name in ("train", "val", "test"):
        X = prepared.features[split_name]
        y_pred = model.predict(X)
        y_true = prepared.targets[split_name]
        if y_true is not None:
            mse = mean_squared_error(y_true, y_pred)
            rmse = float(mse**0.5)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(
                f"{split_name.capitalize()} metrics -> RMSE: {rmse:.4f}, "
                f"MAE: {mae:.4f}, R^2: {r2:.4f}"
            )
        else:
            preview = ", ".join(f"{p:.3f}" for p in y_pred[:5])
            print(
                f"{split_name.capitalize()} split is unlabelled. Showing first five predictions: {preview}"
            )


def main() -> None:
    args = parse_args()
    task = load_task(args.dataset, args.task)
    splits = fetch_split_tables(task)
    limit_rows_per_split(splits, args.max_rows)
    user_ids = collect_user_ids(splits, task.entity_col)
    user_features = compute_user_features(task, user_ids)
    prepared = assemble_model_ready_splits(task, splits, user_features)
    train_and_evaluate(prepared, args.n_estimators)


if __name__ == "__main__":
    main()
