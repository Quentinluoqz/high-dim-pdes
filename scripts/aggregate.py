#!/usr/bin/env python3
"""Validate and aggregate atomic run metrics with optional Slurm accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

VALID_STATUSES = {"completed", "diverged", "failed", "timeout", "not_reached"}
REQUIRED = {
    "schema_version",
    "run_id",
    "method",
    "equation",
    "dim",
    "seed",
    "precision",
    "status",
    "config_hash",
    "source_commit",
}


def _flatten(prefix: str, value: Any, row: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, child, row)
    elif isinstance(value, list):
        row[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        row[prefix] = value


def load_metric(path: Path) -> dict[str, Any]:
    """Load one metric document and enforce the stable schema boundary."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"metrics document is not an object: {path}")
    missing = REQUIRED - set(raw)
    if missing:
        raise ValueError(f"missing fields {sorted(missing)} in {path}")
    if raw["status"] not in VALID_STATUSES:
        raise ValueError(f"unknown status {raw['status']!r} in {path}")
    row: dict[str, Any] = {"metrics_path": str(path)}
    _flatten("", raw, row)
    return row


def aggregate(results_root: Path, ledger_path: Path | None = None) -> pd.DataFrame:
    paths = sorted(results_root.glob("**/metrics.json"))
    rows = [load_metric(path) for path in paths]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    duplicates = frame[frame.duplicated("run_id", keep=False)]
    if not duplicates.empty:
        detail = duplicates[["run_id", "metrics_path"]].to_dict("records")
        raise ValueError(f"duplicate run identities: {detail}")
    if ledger_path and ledger_path.is_file():
        ledger = pd.read_csv(ledger_path)
        key = "run_id" if "run_id" in ledger.columns else "JobName"
        if key not in frame.columns:
            raise ValueError(f"cannot merge ledger on {key}")
        frame = frame.merge(ledger, on=key, how="left", suffixes=("", "_slurm"))
    return frame.sort_values(["equation", "method", "dim", "seed", "precision"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = aggregate(args.results, args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    args.out.with_suffix(".json").write_text(
        json.dumps(frame.to_dict("records"), indent=2, default=str) + "\n", encoding="utf-8"
    )
    completed = int((frame.get("status", pd.Series(dtype=str)) == "completed").sum())
    print(f"aggregated={len(frame)} completed={completed} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
