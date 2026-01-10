"""End-to-end pipeline that combines Featuretools DFS with multiple models on RelBench tasks.

Command-line arguments
======================

``--dataset``
    RelBench dataset name (e.g. ``rel-event``, ``rel-fin``).

``--preset``
    Dataset preset treated as suffix for RelBench 1.1 compatibility (e.g. ``rel-event-small``).

``--task``
    RelBench task name used to obtain targets.

``--agg-primitives``
    Aggregation primitives passed to Featuretools Deep Feature Synthesis (DFS).

``--agg-dataframes``
    Optional list of dataframes that should receive aggregation primitives.

``--agg-primitive-options``
    JSON mapping of per-primitive options (e.g. ``{"sum": {"include_dataframes": ["transactions"]}}``).

``--agg-include-columns``
    Optional list of columns (format: ``table.column``) that should receive aggregation primitives.

``--agg-ignore-columns``
    Optional list of columns (format: ``table.column``) that should be excluded from aggregation primitives.

``--trans-primitives``
    Transform primitives passed to Featuretools DFS.

``--max-depth``
    Maximum depth for DFS feature generation.

``--training-window``
    Featuretools DFS ``training_window`` (e.g. ``30d``, ``90d``). If not set, DFS uses all available history.

``--max-base-rows``
    Trim relational tables to at most this many rows for quicker experiments.

``--max-observations``
    Limit number of task observations per split (useful for quick runs).

``--model``
    Model backend used after feature generation. Choose between ``tabpfn`` (default),
    ``xgboost``, ``lightgbm`` or ``realmlp``.

``--n-estimators``
    Number of estimators used by the selected model. For ``tabpfn`` this controls the
    ensemble size, while for ``xgboost`` and ``lightgbm`` it maps to the number of boosting
    rounds.

``--device``
    Torch device used for TabPFN/RealMLP. Overridden by ``--gpu`` when provided.

``--gpu``
    If set, train supported models on the first available CUDA device.

``--verbose``
    Enable verbose Featuretools output during DFS.

``--realmlp-n-epochs``
    Number of epochs when training RealMLP-TD (default: 256).

``--realmlp-n-cv``
    Number of CV folds for bagging/ensembling with RealMLP-TD (default: 1).

``--realmlp-verbosity``
    Verbosity level for RealMLP-TD training (default: 1).
"""

from __future__ import annotations

import argparse
import json
import sys
import types
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from time import perf_counter
import tracemalloc

import numpy as np
import pandas as pd
from pandas.api.types import is_extension_array_dtype, is_integer_dtype

import torch
import featuretools as ft
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from tabpfn import TabPFNClassifier, TabPFNRegressor

try:  # Optional dependency: PyTabKit RealMLP
    from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
except Exception:  # pragma: no cover - optional dependency may be missing
    RealMLP_TD_Classifier = None
    RealMLP_TD_Regressor = None

try:  # Optional dependency: xgboost
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency may be missing
    XGBClassifier = None
    XGBRegressor = None

try:  # Optional dependency: lightgbm
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover - optional dependency may be missing
    LGBMClassifier = None
    LGBMRegressor = None

def _ensure_datasets_exceptions_module() -> None:
    """Ensure datasets.exceptions is importable for relbench compatibility."""
    if "datasets.exceptions" in sys.modules:
        return

    class DatasetNotFoundError(FileNotFoundError):
        """Fallback DatasetNotFoundError for datasets 3+."""

    module = types.ModuleType("datasets.exceptions")
    module.DatasetNotFoundError = DatasetNotFoundError
    sys.modules["datasets.exceptions"] = module


_ensure_datasets_exceptions_module()

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


def resolve_device(use_gpu: bool, requested_device: str) -> str:
    if use_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested via --gpu but no CUDA device is available.")
        return "cuda"
    return requested_device


def _profile_with_memory(target: Dict[str, Dict[str, float]], label: str, func):
    tracemalloc.start()
    start = perf_counter()
    result = func()
    duration = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    target[label] = {
        "runtime_sec": duration,
        "peak_memory_mb": peak / (1024 * 1024),
    }
    return result


def _profile_step(target: Dict[str, float], label: str, func):
    start = perf_counter()
    result = func()
    target[label] = perf_counter() - start
    return result


@dataclass(frozen=True)
class TableSpec:
    """Metadata used to register a table with Featuretools."""

    index: str
    time_col: Optional[str]
    fkeys: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate relational features with Featuretools and train a model on RelBench tasks.",
    )
    parser.add_argument("--dataset", default="rel-event", help="RelBench dataset name (e.g. rel-event, rel-fin)")
    parser.add_argument(
        "--preset",
        default="small",
        help="Dataset preset treated as suffix for RelBench 1.1 compatibility (e.g. rel-event-small)",
    )
    parser.add_argument(
        "--task",
        default="user-attendance",
        help="RelBench task name (required to obtain targets).",
    )
    parser.add_argument(
        "--agg-primitives",
        nargs="*",
        default=["count"],
        help="Aggregation primitives passed to Featuretools DFS.",
    )
    parser.add_argument(
        "--agg-dataframes",
        nargs="*",
        default=None,
        help="Restrict aggregation primitives to these dataframes (default: apply to all).",
    )
    parser.add_argument(
        "--agg-primitive-options",
        default=None,
        help="JSON mapping of per-primitive options passed to Featuretools primitive_options.",
    )
    parser.add_argument(
        "--agg-include-columns",
        nargs="*",
        default=None,
        help="Restrict aggregation primitives to these columns (format: table.column).",
    )
    parser.add_argument(
        "--agg-ignore-columns",
        nargs="*",
        default=None,
        help="Exclude these columns from aggregation primitives (format: table.column).",
    )
    parser.add_argument(
        "--trans-primitives",
        nargs="*",
        default=["month", "weekday"],
        help="Transform primitives passed to Featuretools DFS.",
    )
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum DFS depth.")
    parser.add_argument(
        "--training-window",
        default=None,
        help="Featuretools DFS training_window (e.g. 30d, 90d). If unset, use full history.",
    )
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
    parser.add_argument(
        "--model",
        choices=("tabpfn", "xgboost", "lightgbm", "realmlp"),
        default="tabpfn",
        help="Model backend to use for supervised learning (default: tabpfn).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=32,
        help="Number of estimators for the selected model (ensemble size or boosting rounds).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for TabPFN/RealMLP (overridden to 'cuda' when --gpu is set).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use the first CUDA device for supported models (TabPFN, RealMLP, XGBoost, LightGBM).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose Featuretools output during DFS.",
    )
    parser.add_argument("--realmlp-n-epochs", type=int, default=256, help="Number of epochs for RealMLP-TD.")
    parser.add_argument(
        "--realmlp-n-cv",
        type=int,
        default=1,
        help="Number of CV folds for bagging/ensembling with RealMLP-TD.",
    )
    parser.add_argument(
        "--realmlp-verbosity",
        type=int,
        default=1,
        help="Verbosity level for RealMLP-TD training.",
    )
    return parser.parse_args()


def _ensure_index_column(df: pd.DataFrame, index_col: str) -> pd.DataFrame:
    if index_col in df.columns:
        df = df.copy()
    else:
        df = df.reset_index(drop=True)
        df[index_col] = df.index
    df[index_col] = pd.to_numeric(df[index_col], errors="coerce").astype("Int64")
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
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _coerce_array_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in df.columns:
        if df[column].dtype != object:
            continue
        series = df[column]
        sample = series.dropna().head(1)
        if sample.empty:
            continue
        value = sample.iloc[0]
        if isinstance(value, (list, tuple, np.ndarray)):
            df[column] = series.map(
                lambda item: tuple(item.tolist())
                if isinstance(item, np.ndarray)
                else tuple(item)
                if isinstance(item, (list, tuple))
                else item
            )
    return df


def _parse_column_specs(items: Optional[Iterable[str]]) -> Optional[Dict[str, List[str]]]:
    if not items:
        return None
    mapping: Dict[str, List[str]] = {}
    for item in items:
        if "." not in item:
            raise ValueError(f"Expected column spec 'table.column', got '{item}'.")
        table, column = item.split(".", 1)
        mapping.setdefault(table, []).append(column)
    return mapping


def _parse_primitive_options(raw: Optional[str]) -> Optional[Dict[str, Dict[str, object]]]:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --agg-primitive-options: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--agg-primitive-options must decode to a JSON object.")
    return parsed


def _compute_nan_replacements(features: np.ndarray) -> np.ndarray:
    """Return column-wise replacements for NaN values.

    RealMLP-TD does not allow NaNs in continuous columns. We impute missing values
    with the column-wise mean computed on the training feature matrix and fall
    back to zeros if a column is entirely NaN.
    """

    if features.size == 0:
        return np.zeros(features.shape[1], dtype=features.dtype)
    valid_mask = ~np.isnan(features)
    counts = valid_mask.sum(axis=0)
    sums = np.where(valid_mask, features, 0).sum(axis=0)
    col_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts != 0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    return col_means.astype(features.dtype, copy=False)


def _fill_missing(features: np.ndarray, replacements: np.ndarray) -> np.ndarray:
    """Replace NaNs in ``features`` using the provided per-column values."""

    return np.where(np.isnan(features), replacements, features)


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
        dataframes[name] = df
        specs[name] = TableSpec(index=index_col, time_col=table.time_col, fkeys=dict(table.fkey_col_to_pkey_table))

    return dataframes, specs


def prepare_observation_dataframe(
    table: Table,
    target_col: str,
    entity_col: str,
    max_rows: Optional[int],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare the task table as a Featuretools target dataframe."""

    df = table.df.copy()
    subset_cols = [entity_col]
    if target_col in df.columns:
        subset_cols.append(target_col)
    df = df.dropna(subset=subset_cols)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0)
    df = df.reset_index(drop=True)

    # Ensure consistent typing
    df[entity_col] = pd.to_numeric(df[entity_col], errors="coerce").astype("Int64")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if target_col in df.columns:
        y = df[target_col].copy()
        obs_df = df.drop(columns=[target_col])
    else:
        y = pd.Series(index=df.index, dtype="float32", name=target_col)
        obs_df = df
    if "index" in obs_df.columns:
        obs_df = obs_df.drop(columns=["index"])
    obs_df["observation_id"] = np.arange(len(obs_df), dtype="int64")
    obs_df = obs_df.set_index("observation_id", drop=False)
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
            df = _coerce_array_like_columns(df)
            add_kwargs = dict(dataframe_name=name, dataframe=df, index=spec.index)
            if spec.time_col and spec.time_col in df.columns:
                add_kwargs["time_index"] = spec.time_col
            logical_types = {}
            if spec.index in df.columns:
                logical_types[spec.index] = "IntegerNullable"
            for fkey in spec.fkeys.keys():
                if fkey in df.columns:
                    logical_types[fkey] = "IntegerNullable"
            for col in df.columns:
                if col in logical_types:
                    continue
                dtype = df[col].dtype
                if is_integer_dtype(dtype) and is_extension_array_dtype(dtype):
                    logical_types[col] = "IntegerNullable"
            if logical_types:
                add_kwargs["logical_types"] = logical_types
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


def make_model(
    task_type: TaskType,
    model_name: str,
    n_estimators: int,
    device: str,
    realmlp_n_epochs: int,
    realmlp_n_cv: int,
    realmlp_verbosity: int,
    use_gpu: bool,
):
    if model_name == "tabpfn":
        if task_type == TaskType.REGRESSION:
            return TabPFNRegressor(device=device, n_estimators=n_estimators)
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return TabPFNClassifier(device=device, n_estimators=n_estimators)
        raise ValueError(f"Unsupported task type for TabPFN: {task_type}")

    if model_name == "xgboost":
        if XGBClassifier is None or XGBRegressor is None:
            raise ImportError("xgboost is required but not installed. Please install xgboost to use this model.")
        tree_method = "gpu_hist" if use_gpu else "hist"
        predictor = "gpu_predictor" if use_gpu else "auto"
        if task_type == TaskType.REGRESSION:
            return XGBRegressor(
                n_estimators=n_estimators,
                objective="reg:squarederror",
                random_state=0,
                tree_method=tree_method,
                predictor=predictor,
            )
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return XGBClassifier(
                n_estimators=n_estimators,
                random_state=0,
                use_label_encoder=False,
                eval_metric="logloss",
                tree_method=tree_method,
                predictor=predictor,
            )
        raise ValueError(f"Unsupported task type for xgboost: {task_type}")

    if model_name == "lightgbm":
        if LGBMClassifier is None or LGBMRegressor is None:
            raise ImportError("lightgbm is required but not installed. Please install lightgbm to use this model.")
        device_type = "gpu" if use_gpu else "cpu"
        if task_type == TaskType.REGRESSION:
            return LGBMRegressor(n_estimators=n_estimators, random_state=0, device=device_type)
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return LGBMClassifier(n_estimators=n_estimators, random_state=0, device=device_type)
        raise ValueError(f"Unsupported task type for lightgbm: {task_type}")

    if model_name == "realmlp":
        if RealMLP_TD_Classifier is None or RealMLP_TD_Regressor is None:
            raise ImportError("PyTabKit is required but not installed. Please install pytabkit and torch to use RealMLP.")
        if task_type == TaskType.REGRESSION:
            return RealMLP_TD_Regressor(
                device=device,
                n_epochs=realmlp_n_epochs,
                n_cv=realmlp_n_cv,
                random_state=0,
                verbosity=realmlp_verbosity,
            )
        if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return RealMLP_TD_Classifier(
                device=device,
                n_epochs=realmlp_n_epochs,
                n_cv=realmlp_n_cv,
                random_state=0,
                verbosity=realmlp_verbosity,
            )
        raise ValueError(f"Unsupported task type for RealMLP: {task_type}")

    raise ValueError(f"Unknown model backend: {model_name}")


def report_regression(split: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return f"[{split}] R2={r2_score(y_true, y_pred):.4f}  MAE={mean_absolute_error(y_true, y_pred):.4f}  RMSE={rmse:.4f}"


def _probability_metrics(y_true: np.ndarray, y_prob: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
    if y_prob is None:
        return None, None

    try:
        if y_prob.ndim == 1 or y_prob.shape[1] == 1:
            positive_scores = y_prob.ravel()
        elif y_prob.shape[1] == 2:
            positive_scores = y_prob[:, 1]
        else:
            positive_scores = None

        if positive_scores is not None and len(np.unique(y_true)) == 2:
            auc = average_precision_score(y_true, positive_scores)
            roc_auc = roc_auc_score(y_true, positive_scores)
        else:
            auc = average_precision_score(y_true, y_prob, average="macro")
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:  # pragma: no cover - defensive around metric compatibility
        return None, None

    return float(auc), float(roc_auc)


def report_classification(
    split: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]
) -> str:
    auc, roc_auc = _probability_metrics(y_true, y_prob)
    metrics = [
        f"Accuracy={accuracy_score(y_true, y_pred):.4f}",
        f"F1(macro)={f1_score(y_true, y_pred, average='macro'):.4f}",
    ]
    if auc is not None:
        metrics.append(f"AUC={auc:.4f}")
    if roc_auc is not None:
        metrics.append(f"ROC-AUC={roc_auc:.4f}")

    return f"[{split}] " + "  ".join(metrics)


def main() -> None:
    args = parse_args()
    resolved_device = resolve_device(args.gpu, args.device)
    result_lines: list[str] = []

    task_or_dataset = load_task_or_dataset(args.dataset, args.preset, args.task)
    if not isinstance(task_or_dataset, BaseTask):
        raise RuntimeError("A concrete RelBench task is required to obtain targets for TabPFN training.")

    task = task_or_dataset
    dataset = task.dataset
    step_timings: Dict[str, float] = {}
    model_metrics: Dict[str, Dict[str, float]] = {}

    base_dataframes, specs = _profile_step(
        step_timings, "prepare_base_tables_seconds", lambda: prepare_base_tables(dataset, args.max_base_rows)
    )

    splits: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for split_name in ("train", "val", "test"):
        try:
            table = task.get_table(split_name, mask_input_cols=False)
        except Exception as exc:  # pragma: no cover - defensive in case split missing
            result_lines.append(f"[{split_name}] Skipped split ({exc})")
            continue
        obs_df, y = prepare_observation_dataframe(
            table,
            task.target_col,
            task.entity_col,
            args.max_observations,
        )
        if obs_df.empty:
            result_lines.append(f"[{split_name}] No observations available after preprocessing")
            continue
        splits[split_name] = (obs_df, y)

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
        agg_primitive_dataframes=args.agg_dataframes,
        agg_primitive_options=_parse_primitive_options(args.agg_primitive_options),
        agg_primitive_include_columns=_parse_column_specs(args.agg_include_columns),
        agg_primitive_ignore_columns=_parse_column_specs(args.agg_ignore_columns),
        trans_primitives=args.trans_primitives,
        max_depth=args.max_depth,
        training_window=args.training_window,
        verbose=args.verbose,
        dtype="float32",
    )
    adapter = FeaturetoolsTabPFNAdapter(entityset_builder=builder, config=cfg)

    train_obs, y_train_series = splits["train"]
    train_map = dict(base_dataframes)
    train_map["observations"] = train_obs

    fit_payload = {"dataframes": train_map, "target_ids": train_obs["observation_id"]}
    if args.training_window:
        if observation_time_col is None:
            raise RuntimeError("training_window requires a timestamp column on observations.")
        fit_payload["cutoff_time"] = train_obs[["observation_id", observation_time_col]]

    _profile_step(step_timings, "dfs_fit_seconds", lambda: adapter.fit(fit_payload))
    transform_payload = {"dataframes": train_map, "target_ids": train_obs["observation_id"]}
    if args.training_window:
        transform_payload["cutoff_time"] = train_obs[["observation_id", observation_time_col]]
    X_train = _profile_step(
        step_timings,
        "dfs_transform_train_seconds",
        lambda: adapter.transform(transform_payload),
    )
    nan_replacements: Optional[np.ndarray] = None
    if args.model == "realmlp":
        nan_replacements = _compute_nan_replacements(X_train)
        X_train = _fill_missing(X_train, nan_replacements)

    encoder: Optional[LabelEncoder] = None
    y_train, encoder = encode_targets(y_train_series, task.task_type, encoder)

    model = make_model(
        task.task_type,
        args.model,
        args.n_estimators,
        resolved_device,
        args.realmlp_n_epochs,
        args.realmlp_n_cv,
        args.realmlp_verbosity,
        args.gpu,
    )
    _profile_with_memory(model_metrics, "train", lambda: model.fit(X_train, y_train))

    feature_names = adapter.get_feature_names_out()
    feature_count = len(feature_names)

    for split_name, (obs_df, y_series) in splits.items():
        data_map = dict(base_dataframes)
        data_map["observations"] = obs_df
        transform_payload = {"dataframes": data_map, "target_ids": obs_df["observation_id"]}
        if args.training_window:
            transform_payload["cutoff_time"] = obs_df[["observation_id", observation_time_col]]
        X_split = _profile_step(
            step_timings,
            f"dfs_transform_{split_name}_seconds",
            lambda: adapter.transform(transform_payload),
        )
        if nan_replacements is not None:
            X_split = _fill_missing(X_split, nan_replacements)

        if task.task_type == TaskType.REGRESSION:
            if y_series.isna().all():
                result_lines.append(f"[{split_name}] No targets available; skipping evaluation.")
                continue
            y_true = y_series.to_numpy(dtype=np.float32)
            y_pred = _profile_with_memory(model_metrics, f"predict_{split_name}", lambda: model.predict(X_split))
            result_lines.append(report_regression(split_name, y_true, y_pred))
        else:
            if encoder is None:
                raise AssertionError("Encoder must be fitted for classification tasks.")
            if y_series.isna().all():
                result_lines.append(f"[{split_name}] No targets available; skipping evaluation.")
                continue
            y_true = encoder.transform(y_series)
            y_pred = _profile_with_memory(model_metrics, f"predict_{split_name}", lambda: model.predict(X_split))
            y_prob: Optional[np.ndarray] = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = _profile_with_memory(
                        model_metrics, f"predict_proba_{split_name}", lambda: np.asarray(model.predict_proba(X_split))
                    )
                except Exception as exc:  # pragma: no cover - optional probability output
                    result_lines.append(f"[{split_name}] predict_proba failed: {exc}")
            result_lines.append(report_classification(split_name, y_true, y_pred, y_prob))

    summary_lines = [f"Synthesized features: {feature_count}"]
    if result_lines:
        summary_lines.append("Results:")
        summary_lines.extend(result_lines)

    if step_timings or model_metrics:
        summary_lines.append("Performance:")
        for label, duration in step_timings.items():
            summary_lines.append(f"   • {label}: {duration:.3f} s")
        for label, metrics in model_metrics.items():
            runtime = metrics.get("runtime_sec")
            peak_mb = metrics.get("peak_memory_mb")
            summary_lines.append(f"   • model[{label}]: {runtime:.3f} s runtime, {peak_mb:.2f} MB peak memory")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
