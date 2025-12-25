"""Run a lightweight RelBench experiment on a built-in dataset (no Kaggle required)."""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Set

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from relbench.base.table import Table
from relbench.datasets import get_dataset
from relbench.tasks import get_task, BaseTask

pd.options.mode.copy_on_write = True


@dataclass
class PreparedSplits:
    """Container for feature matrices and targets."""
    features: Dict[str, pd.DataFrame]
    targets: Dict[str, Optional[pd.Series]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight RelBench experiment.")
    parser.add_argument("--dataset", default="rel-event", help="RelBench dataset (e.g. rel-fin, rel-movies)")
    parser.add_argument(
        "--preset",
        default="small",
        help=(
            "Dataset preset name treated as a suffix (e.g., small -> rel-fin-small). "
            "Use 'none' to skip suffix handling when no preset artefact exists."
        ),
    )
    parser.add_argument("--task", default="user-attendance", help="Task name (default: user-attendance for rel-event)")
    parser.add_argument("--max-rows", type=int, default=5000, help="Limit rows per split for speed")
    parser.add_argument("--n-estimators", type=int, default=200, help="Number of trees in the random forest")
    return parser.parse_args()


def _normalized_preset(preset: str | None) -> Optional[str]:
    if not preset:
        return None
    normalized = preset.strip().lower()
    if normalized in {"", "base", "default", "none"}:
        return None
    return normalized


def _load_dataset_relbench_110(dataset_name: str, preset: str):
    """
    relbench 1.1.0's get_dataset(...) does NOT accept a 'preset' kwarg.
    Strategy:
      1) Try hyphenated name: '{dataset_name}-{preset}' (e.g., 'rel-fin-small')
      2) Fall back to base name: '{dataset_name}'
    """
    # 1) Try hyphenated variant
    normalized = _normalized_preset(preset)
    hyphenated = f"{dataset_name}-{normalized}" if normalized else dataset_name
    if hyphenated != dataset_name:
        try:
            print(f"Trying dataset alias: '{hyphenated}' ...")
            return get_dataset(hyphenated, download=True)
        except Exception as e:
            print(f"   • Alias '{hyphenated}' not available ({e}). Falling back to '{dataset_name}'.")
            if normalized:
                print(
                    "   • Hint: This RelBench release may only ship the full dataset. "
                    "Use --max-base-rows/--max-observations flags to downsample for quicker runs."
                )

    # 2) Fallback to base dataset name
    print(f"Loading base dataset name: '{dataset_name}' ...")
    return get_dataset(dataset_name, download=True)


def load_task_or_dataset(dataset_name: str, preset: str, task_name: Optional[str]) -> BaseTask | object:
    """Try to load a RelBench task or fallback to raw dataset (relbench 1.1.0 compatible)."""
    preset_note = (
        "treated as a name suffix; use --preset none to skip aliasing"
        if _normalized_preset(preset)
        else "loaded directly"
    )
    print(f"Loading dataset '{dataset_name}' (preset='{preset}' is {preset_note}) ...")

    dataset = _load_dataset_relbench_110(dataset_name, preset)

    if task_name:
        # get_task signature in 1.1.0 supports download=True
        try:
            task = get_task(dataset_name, task_name, download=True)
        except TypeError:
            # Be defensive if local build differs
            task = get_task(dataset_name, task_name)
        print(f"Loaded specific task '{task_name}'.")
        return task
    else:
        print(f"Loaded dataset '{dataset_name}' directly (no task).")
        return dataset


def fetch_split_tables(dataset) -> Dict[str, pd.DataFrame]:
    """Extract available splits (train/val/test) as pandas DataFrames."""
    splits: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        df = None
        if hasattr(dataset, "get_table"):
            try:
                table: Table = dataset.get_table(split_name)
                if table is not None and hasattr(table, "df"):
                    df = table.df.copy()
            except Exception:
                df = None
        if df is None and hasattr(dataset, "splits") and split_name in getattr(dataset, "splits", {}):
            df = dataset.splits[split_name].copy()
        if df is None:
            continue

        # Ensure a timestamp column exists and is parsed (if present)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            df["timestamp"] = pd.NaT

        splits[split_name] = df
        print(f"Loaded {split_name} split with {len(df):,} rows")
 
    if not splits:
        raise RuntimeError("No splits found on the loaded dataset. Check dataset name or version.")
    return splits


def limit_rows_per_split(splits: Dict[str, pd.DataFrame], max_rows: int | None) -> None:
    """Trim each split to at most `max_rows`."""
    if not max_rows:
        return
    for split_name, df in list(splits.items()):
        if len(df) > max_rows:
            splits[split_name] = df.head(max_rows).reset_index(drop=True)
            print(f"Trimmed {split_name} to {max_rows:,} rows.")


def collect_entity_ids(splits: Dict[str, pd.DataFrame], entity_col: str) -> Set[int]:
    """Return unique entity IDs across splits."""
    ids: Set[int] = set()
    for df in splits.values():
        if entity_col in df:
            ids.update(pd.to_numeric(df[entity_col], errors="coerce").dropna().astype(int).unique().tolist())
    print(f"Found {len(ids):,} unique entities across splits.")
    return ids


def simple_feature_engineering(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Example: simple feature aggregation for demonstration."""
    print("Running simple feature engineering ...")
    if entity_col not in df.columns:
        raise KeyError(f"Entity column '{entity_col}' not found in the provided split.")
    numeric_cols = df.select_dtypes("number").columns.difference([entity_col])
    if len(numeric_cols) == 0:
        # Create a dummy numeric column to avoid empty aggregations
        df = df.copy()
        df["_dummy_num"] = 1.0
        numeric_cols = ["_dummy_num"]

    features = (
        df.groupby(entity_col, dropna=False)[numeric_cols]
        .agg(["mean", "sum", "count"])
    )
    features.columns = ["_".join(c).strip() for c in features.columns]
    return features


def assemble_model_ready_splits(
    splits: Dict[str, pd.DataFrame],
    features: pd.DataFrame,
    entity_col: str,
    target_col: Optional[str] = None,
) -> PreparedSplits:
    """Join features and extract X/y matrices."""
    X_dict: Dict[str, pd.DataFrame] = {}
    y_dict: Dict[str, Optional[pd.Series]] = {}

    for split_name, df in splits.items():
        if entity_col not in df.columns:
            # If the entity column is missing, create it as NaN to allow the merge (will produce NaN features)
            df = df.copy()
            df[entity_col] = pd.NA

        enriched = df.merge(features, on=entity_col, how="left")

        # Build feature matrix by excluding identifier and target
        excluded = {entity_col, "timestamp", "index"}
        if target_col:
            excluded.add(target_col)
        feature_cols = [c for c in enriched.columns if c not in excluded]

        X = enriched[feature_cols]
        y = enriched[target_col] if target_col and (target_col in enriched.columns) else None

        X_dict[split_name] = X
        y_dict[split_name] = y

        y_shape = y.shape if y is not None else None
        print(f"{split_name}: X={X.shape}, y={y_shape}")

    return PreparedSplits(X_dict, y_dict)


def train_and_evaluate(prepared: PreparedSplits, n_estimators: int) -> None:
    """Train and evaluate a baseline model. Skips training if y is missing."""
    y_train = prepared.targets.get("train")
    if y_train is None:
        print("⚠️  No target available for training (did you omit --task?). Skipping model training.")
        return

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0, n_jobs=-1)
    model.fit(prepared.features["train"], y_train)

    for split in ("train", "val", "test"):
        X = prepared.features.get(split)
        y_true = prepared.targets.get(split)
        if X is None:
            continue
        y_pred = model.predict(X)
        if y_true is not None:
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"{split.capitalize()}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        else:
            print(f"{split.capitalize()} predictions (unlabeled) preview: {y_pred[:5]}")


def main() -> None:
    args = parse_args()
    dataset_or_task = load_task_or_dataset(args.dataset, args.preset, args.task)

    # Use unified handling (works for both BaseTask and Dataset)
    if isinstance(dataset_or_task, BaseTask):
        entity_col = dataset_or_task.entity_col
        target_col = dataset_or_task.target_col
        splits = fetch_split_tables(dataset_or_task)
    else:
        dataset = dataset_or_task
        # Try common attribute names for the entity id column
        entity_col = getattr(dataset, "primary_key", None)
        if entity_col is None:
            entity_col = getattr(dataset, "primary_table", None)
        if entity_col is None:
            # Fall back to a best-effort guess
            # prefer a column literally named 'entity_id' or 'id'
            splits_probe = fetch_split_tables(dataset)
            example_df = next(iter(splits_probe.values()))
            candidate = None
            for col in ("entity_id", "id", "company_id", "user_id"):
                if col in example_df.columns:
                    candidate = col
                    break
            if candidate is None:
                raise RuntimeError("Could not infer entity column. Please specify a task or adjust the code.")
            entity_col = candidate
            splits = splits_probe
        else:
            splits = fetch_split_tables(dataset)
        target_col = None  # raw dataset path (no task) has no target

    limit_rows_per_split(splits, args.max_rows)
    _ = collect_entity_ids(splits, entity_col)

    example_split = next(iter(splits.values()))
    features = simple_feature_engineering(example_split, entity_col)
    prepared = assemble_model_ready_splits(splits, features, entity_col, target_col)

    train_and_evaluate(prepared, args.n_estimators)


if __name__ == "__main__":
    main()





