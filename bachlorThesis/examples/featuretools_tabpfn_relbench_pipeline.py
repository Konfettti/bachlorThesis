"""End-to-end pipeline that combines Featuretools DFS with TabPFN on RelBench tasks."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import featuretools as ft
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier, TabPFNRegressor

from relbench.base import TaskType
from relbench.base.database import Database
from relbench.base.table import Table
from relbench.tasks import BaseTask

from ..featuretools_tabpfn_adapter import (
    FeaturetoolsTabPFNAdapter,
    FeaturetoolsTabPFNConfig,
)
from .relbench_small_pipeline import load_task_or_dataset


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class TableSpec:
    """Metadata used to register a table with Featuretools."""

    index: str
    time_col: Optional[str]
    fkeys: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate relational features with Featuretools and train TabPFN on RelBench tasks.",
    )
    parser.add_argument("--dataset", default="rel-event", help="RelBench dataset name (e.g. rel-event, rel-fin)")
    parser.add_argument(
        "--preset",
        default="small",
        help=(
            "Dataset preset treated as suffix for RelBench 1.1 compatibility (e.g. rel-event-small). "
            "Use 'none' to skip suffix handling when a preset artefact is unavailable."
        ),
    )
    parser.add_argument(
        "--task",
        default="user-attendance",
        help="RelBench task name (required to obtain targets).",
    )
    parser.add_argument(
        "--agg-primitives",
        nargs="*",
        default=["sum", "mean", "count"],
        help="Aggregation primitives passed to Featuretools DFS.",
    )
    parser.add_argument(
        "--trans-primitives",
        nargs="*",
        default=["month", "weekday"],
        help="Transform primitives passed to Featuretools DFS.",
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum DFS depth.")
    parser.add_argument(
        "--max-base-rows",
        type=int,
        default=50000,
        help="Trim relational tables to at most this many rows for quicker experiments.",
    )
    parser.add_argument(
        "--max-observations",
        type=int,
        default=5000,
        help="Limit number of task observations per split (useful for quick runs).",
    )
    parser.add_argument("--n-estimators", type=int, default=32, help="Number of estimators for TabPFN ensemble.")
    parser.add_argument("--device", default="cpu", help="Torch device to use for TabPFN (default: cpu).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Featuretools output during DFS.",
    )
    return parser.parse_args()


def _ensure_index_column(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    if index_col in df.columns:
        return df
    df = df.reset_index(drop=True)
    df[index_col] = df.index.astype("int64")
    return df


def _coerce_time_column(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col and time_col in df.columns:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


def _coerce_foreign_keys(df: pd.DataFrame, fkeys: Iterable[str]) -> pd.DataFrame:
    if not fkeys:
        return df
    df = df.copy()
    for col in fkeys:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(-1)
                .astype("int64")
            )
    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes.astype("int64")
    return df


def prepare_base_tables(dataset, max_rows: Optional[int]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, TableSpec]]:
    """Convert RelBench database tables into DataFrames suitable for Featuretools."""

    db: Database = dataset.get_db()
    dataframes: Dict[str, pd.DataFrame] = {}
    specs: Dict[str, TableSpec] = {}

    for name, table in db.table_dict.items():
        df = table.df.copy()
        if max_rows is not None and len(df) > max_rows:
            df = df.head(max_rows).reset_index(drop=True)

        index_col = table.pkey_col or f"{name}__index"
        df = _ensure_index_column(df, index_col)
        df = _coerce_time_column(df, table.time_col)
        df = _coerce_foreign_keys(df, table.fkey_col_to_pkey_table.keys())
        df = _encode_categoricals(df)

        dataframes[name] = df
        specs[name] = TableSpec(index=index_col, time_col=table.time_col, fkeys=dict(table.fkey_col_to_pkey_table))

    return dataframes, specs


def prepare_observation_dataframe(
    table: Table,
    target_col: str,
    entity_col: str,
    max_rows: Optional[int],
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Prepare the task table as a Featuretools target dataframe."""

    df = table.df.copy()
    subset_cols = [entity_col]
    if target_col in df.columns:
        subset_cols.append(target_col)
    df = df.dropna(subset=subset_cols)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    df = df.reset_index(drop=True)

    # Ensure consistent typing
    df[entity_col] = (
        pd.to_numeric(df[entity_col], errors="coerce")
        .fillna(-1)
        .astype("int64")
    )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if target_col in df.columns:
        y: Optional[pd.Series] = df[target_col].copy()
        obs_df = df.drop(columns=[target_col])
    else:
        y = None
        obs_df = df
    if "index" in obs_df.columns:
        obs_df = obs_df.drop(columns=["index"])
    obs_df["observation_id"] = np.arange(len(obs_df), dtype="int64")
    obs_df = obs_df.set_index("observation_id", drop=False)
    obs_df = _encode_categoricals(obs_df)

    return obs_df, y


def build_entityset_builder(specs: Mapping[str, TableSpec]) -> Callable[[Mapping[str, pd.DataFrame]], ft.EntitySet]:
    """Create a builder that instantiates an EntitySet from provided dataframes."""

    def builder(dataframes: Mapping[str, pd.DataFrame]) -> ft.EntitySet:
        es = ft.EntitySet(id="relbench")
        for name, spec in specs.items():
            if name not in dataframes:
                continue
            df = dataframes[name].copy()
            df = _ensure_index_column(df, spec.index)
            df = _coerce_time_column(df, spec.time_col)
            df = _coerce_foreign_keys(df, spec.fkeys.keys())
            df = _encode_categoricals(df)

            add_kwargs = dict(dataframe_name=name, dataframe=df, index=spec.index)
            if spec.time_col and spec.time_col in df.columns:
                add_kwargs["time_index"] = spec.time_col
            es = es.add_dataframe(**add_kwargs)

        for child_name, spec in specs.items():
            if child_name not in dataframes:
                continue
            for fkey_col, parent_name in spec.fkeys.items():
                if parent_name not in dataframes:
                    continue
                parent_spec = specs[parent_name]
                if fkey_col not in dataframes[child_name].columns:
                    continue
                es = es.add_relationship(
                    parent_dataframe_name=parent_name,
                    parent_column_name=parent_spec.index,
                    child_dataframe_name=child_name,
                    child_column_name=fkey_col,
                )
        return es

    return builder


def encode_targets(y: pd.Series, task_type: TaskType, encoder: Optional[LabelEncoder]) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    if task_type == TaskType.REGRESSION:
        return y.to_numpy(dtype=np.float32), None

    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        enc = encoder or LabelEncoder()
        if encoder is None:
            enc.fit(y)
        transformed = enc.transform(y)
        return transformed, enc

    raise ValueError(f"Unsupported task type for TabPFN pipeline: {task_type}")


def make_model(task_type: TaskType, n_estimators: int, device: str):
    if task_type == TaskType.REGRESSION:
        return TabPFNRegressor(device=device, n_estimators=n_estimators)
    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        return TabPFNClassifier(device=device, n_estimators=n_estimators)
    raise ValueError(f"Unsupported task type for TabPFN: {task_type}")


def report_regression(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    print(
        f"[{split}] R2={r2_score(y_true, y_pred):.4f}  MAE={mean_absolute_error(y_true, y_pred):.4f}  RMSE={rmse:.4f}"
    )


def report_classification(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(
        f"[{split}] Accuracy={accuracy_score(y_true, y_pred):.4f}  F1(macro)={f1_score(y_true, y_pred, average='macro'):.4f}"
    )


def main() -> None:
    args = parse_args()

    task_or_dataset = load_task_or_dataset(args.dataset, args.preset, args.task)
    if not isinstance(task_or_dataset, BaseTask):
        raise RuntimeError("A concrete RelBench task is required to obtain targets for TabPFN training.")

    task = task_or_dataset
    dataset = task.dataset

    print("Preparing relational tables ...")
    base_dataframes, specs = prepare_base_tables(dataset, args.max_base_rows)

    print("Preparing task splits ...")
    splits: Dict[str, Tuple[pd.DataFrame, Optional[pd.Series]]] = {}
    label_status: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        try:
            table = task.get_table(split_name)
        except Exception as exc:  # pragma: no cover - defensive in case split missing
            print(f"   • Skipping split '{split_name}' ({exc}).")
            continue
        obs_df, y = prepare_observation_dataframe(table, task.target_col, task.entity_col, args.max_observations)
        if obs_df.empty:
            print(f"   • Split '{split_name}' is empty after preprocessing.")
            continue
        splits[split_name] = (obs_df, y)
        if y is None:
            label_status[split_name] = "labels hidden"
        elif y.isna().all():
            label_status[split_name] = "labels dropped during cleaning"
        else:
            label_status[split_name] = "labels available"
        note = label_status[split_name]
        print(f"   • {split_name}: {len(obs_df):,} observations ({note})")

    if "train" not in splits:
        raise RuntimeError("Training split could not be prepared.")

    observation_time_col = "timestamp" if "timestamp" in splits["train"][0].columns else None
    specs["observations"] = TableSpec(
        index="observation_id",
        time_col=observation_time_col,
        fkeys={task.entity_col: task.entity_table},
    )

    builder = build_entityset_builder(specs)
    cfg = FeaturetoolsTabPFNConfig(
        target_dataframe_name="observations",
        agg_primitives=args.agg_primitives,
        trans_primitives=args.trans_primitives,
        max_depth=args.max_depth,
        verbose=args.verbose,
        dtype="float32",
    )
    adapter = FeaturetoolsTabPFNAdapter(entityset_builder=builder, config=cfg)

    train_obs, y_train_series = splits["train"]
    if y_train_series is None:
        raise RuntimeError("Training targets are missing; cannot fit TabPFN model.")
    train_map = dict(base_dataframes)
    train_map["observations"] = train_obs

    print("Running Deep Feature Synthesis on training data ...")
    adapter.fit({"dataframes": train_map, "target_ids": train_obs["observation_id"]})
    X_train = adapter.transform({"dataframes": train_map, "target_ids": train_obs["observation_id"]})
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    encoder: Optional[LabelEncoder] = None
    y_train, encoder = encode_targets(y_train_series, task.task_type, encoder)

    model = make_model(task.task_type, args.n_estimators, args.device)
    print("Training TabPFN model ...")
    model.fit(X_train, y_train)

    feature_names = adapter.get_feature_names_out()
    print(f"Generated {len(feature_names)} features for observations.")

    if label_status:
        print(
            "Split label summary: "
            + ", ".join(f"{split} -> {status}" for split, status in label_status.items())
        )
        if any(status != "labels available" for status in label_status.values()):
            print(
                "Hint: Metrics are only computed for splits with available labels. "
                "RelBench frequently withholds test labels so you may only see train/val metrics."
            )

    for split_name, (obs_df, y_series) in splits.items():
        data_map = dict(base_dataframes)
        data_map["observations"] = obs_df
        X_split = adapter.transform({"dataframes": data_map, "target_ids": obs_df["observation_id"]})
        X_split = np.nan_to_num(X_split, nan=0.0, posinf=0.0, neginf=0.0)

        if task.task_type == TaskType.REGRESSION:
            if y_series is None:
                print(
                    f"[{split_name}] No targets provided for this split; skipping evaluation (RelBench often hides test labels)."
                )
                continue
            if y_series.isna().all():
                print(f"[{split_name}] Targets are entirely missing after preprocessing; skipping evaluation.")
                continue
            y_true = y_series.to_numpy(dtype=np.float32)
            y_pred = model.predict(X_split)
            report_regression(split_name, y_true, y_pred)
        else:
            if encoder is None:
                raise AssertionError("Encoder must be fitted for classification tasks.")
            if y_series is None:
                print(
                    f"[{split_name}] No targets provided for this split; skipping evaluation (RelBench often hides test labels)."
                )
                continue
            if y_series.isna().all():
                print(f"[{split_name}] Targets are entirely missing after preprocessing; skipping evaluation.")
                continue
            y_true = encoder.transform(y_series)
            y_pred = model.predict(X_split)
            report_classification(split_name, y_true, y_pred)


if __name__ == "__main__":
    main()
