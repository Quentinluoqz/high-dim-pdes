#!/usr/bin/env python3
"""Expand an experiment sweep into a deterministic run manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import yaml

FIELDS = [
    "array_index",
    "run_id",
    "campaign",
    "method",
    "equation",
    "dim",
    "seed",
    "precision",
    "target_eps",
    "world_size",
    "cluster",
    "partition",
    "qos",
    "cpus_per_task",
    "gpus",
    "time_limit",
    "overrides",
    "command",
]


def _mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _matches(row: dict[str, Any], condition: dict[str, Any]) -> bool:
    return all(
        row.get(key) in value if isinstance(value, list) else row.get(key) == value
        for key, value in condition.items()
    )


def _cluster(row: dict[str, Any], route: dict[str, Any]) -> str:
    for rule in route.get("rules", []):
        if _matches(row, rule.get("when", {})):
            return str(rule["cluster"])
    return str(route["default"])


def expand(experiment: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, str | int]]:
    axes = experiment.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("experiment.axes must be a non-empty mapping")
    required = {"method", "equation", "dim", "seed", "precision"}
    if not required.issubset(axes):
        raise ValueError(f"missing axes: {sorted(required - set(axes))}")
    for key, values in axes.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis {key!r} must be a non-empty list")

    keys = list(axes)
    exclusions = experiment.get("exclude", [])
    route = experiment.get("route", {})
    clusters = profile.get("clusters", {})
    rows: list[dict[str, str | int]] = []
    for combination in itertools.product(*(axes[key] for key in keys)):
        raw = dict(zip(keys, combination, strict=True))
        if any(_matches(raw, exclusion) for exclusion in exclusions):
            continue
        cluster_name = _cluster(raw, route)
        if cluster_name not in clusters:
            raise ValueError(f"cluster {cluster_name!r} is absent from profile")
        resources = clusters[cluster_name]
        overrides = [str(item) for item in experiment.get("overrides", [])]
        if "target_eps" in raw:
            overrides.append(f"method.params.target_eps={raw['target_eps']}")
        if "world_size" in raw:
            overrides.append(f"method.params.world_size={raw['world_size']}")
        identity = {
            **raw,
            "campaign": experiment.get("name", "experiment"),
            "overrides": overrides,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        run_id = (
            f"{raw['method']}-{raw['equation']}-d{raw['dim']}-s{raw['seed']}-"
            f"{raw['precision']}-{digest}"
        )
        args = [
            "python",
            "scripts/run_one.py",
            "--method",
            str(raw["method"]),
            "--equation",
            str(raw["equation"]),
            "--dim",
            str(raw["dim"]),
            "--seed",
            str(raw["seed"]),
            "--precision",
            str(raw["precision"]),
            "--resume",
        ]
        for item in overrides:
            args.extend(("--override", item))
        rows.append(
            {
                "array_index": len(rows) + 1,
                "run_id": run_id,
                "campaign": str(experiment.get("name", "experiment")),
                "method": str(raw["method"]),
                "equation": str(raw["equation"]),
                "dim": int(raw["dim"]),
                "seed": int(raw["seed"]),
                "precision": str(raw["precision"]),
                "target_eps": str(raw.get("target_eps", "")),
                "world_size": str(raw.get("world_size", "")),
                "cluster": cluster_name,
                "partition": str(resources["partition"]),
                "qos": str(resources.get("qos", "")),
                "cpus_per_task": int(resources["cpus_per_task"]),
                "gpus": int(raw.get("world_size", resources.get("gpus", 0))),
                "time_limit": str(resources["time_limit"]),
                "overrides": json.dumps(overrides, separators=(",", ":")),
                "command": shlex.join(args),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, nargs="?")
    parser.add_argument("--config", dest="experiment_config", type=Path)
    parser.add_argument("--output", "--out", dest="output", type=Path, default=Path("runs.csv"))
    parser.add_argument("--defaults", type=Path, default=Path("infra/cluster_defaults.yaml"))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiment_path = args.experiment_config or args.experiment
    if experiment_path is None:
        parser.error("provide EXPERIMENT or --config PATH")
    if args.experiment_config is not None and args.experiment is not None:
        parser.error("provide only one of EXPERIMENT or --config PATH")
    profile = _mapping(args.defaults)
    if args.preflight:
        profile = _deep_merge(profile, _mapping(args.preflight))
    rows = expand(_mapping(experiment_path), profile)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["partition"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"runs": len(rows), "partitions": counts}, sort_keys=True))
    if args.dry_run:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
