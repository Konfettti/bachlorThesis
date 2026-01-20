"""Build a DuckDB database for rel-event manual baseline SQL."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import sys
import types


def _ensure_datasets_exceptions_module() -> None:
    if "datasets.exceptions" in sys.modules:
        return

    class DatasetNotFoundError(FileNotFoundError):
        """Fallback DatasetNotFoundError for datasets 3+."""

    module = types.ModuleType("datasets.exceptions")
    module.DatasetNotFoundError = DatasetNotFoundError
    sys.modules["datasets.exceptions"] = module


_ensure_datasets_exceptions_module()

from relbench.datasets import get_dataset
from relbench.tasks import get_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DuckDB database for rel-event baselines.")
    parser.add_argument(
        "--output",
        default="event/event.db",
        help="Path to write the DuckDB database (default: event/event.db).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(name="rel-event")
    task = get_task("rel-event", "user-attendance", download=True)

    conn = duckdb.connect(str(output_path))

    for name, table in dataset.get_db().table_dict.items():
        df = table.df
        conn.register(f"tmp_{name}", df)
        conn.execute(f"create or replace table {name} as select * from tmp_{name}")

    for split in ("train", "val", "test"):
        df = task.get_table(split).df
        conn.register("tmp_user_attendance", df)
        conn.execute(f"create or replace table user_attendance_{split} as select * from tmp_user_attendance")

    conn.close()


if __name__ == "__main__":
    main()