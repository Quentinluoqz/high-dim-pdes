from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(run_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "method": "deep_bsde_orig",
        "equation": "e5_heat",
        "dim": 2,
        "seed": 0,
        "precision": "fp32",
        "status": "completed",
        "config_hash": "abc",
        "source_commit": "def",
        "rel_error": 0.1,
        "wall_clock_sec": 1.0,
        "throughput": 10.0,
        "hardware": {"world_size": 1},
    }


def test_aggregate_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    module = _load_script("aggregate")
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "metrics.json").write_text(json.dumps(_metric("same")), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        module.aggregate(tmp_path)


def test_plots_degrade_to_explicit_missing_panels(tmp_path: Path) -> None:
    module = _load_script("plots")
    frame = pd.DataFrame([_metric("one")])
    for stem, builder in module.FIGURES:
        module._save(builder(frame), tmp_path, stem)
        assert (tmp_path / f"{stem}.png").is_file()
        assert (tmp_path / f"{stem}.pdf").is_file()
