from __future__ import annotations

import json
from dataclasses import replace

from hdpde.runner import run_atomic
from hdpde.types import RunStatus
from hdpde.utils.config import load_run_config


def test_atomic_smoke_writes_frozen_artifacts(tmp_path) -> None:
    config = load_run_config(
        ["configs/smoke.yaml"],
        [
            f"output.root={tmp_path / 'results'}",
            "method.params.iterations=2",
            "method.params.compute_reference=false",
            "source_commit=test-commit",
        ],
    )
    metrics = run_atomic(config)
    assert metrics.status == RunStatus.COMPLETED
    run_dirs = list((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1
    assert json.loads((run_dirs[0] / "metrics.json").read_text())["source_commit"] == "test-commit"
    assert (run_dirs[0] / "config.json").is_file()
    assert (run_dirs[0] / "train.log").is_file()


def test_unknown_method_is_retained_as_failure(tmp_path) -> None:
    config = load_run_config(
        ["configs/smoke.yaml"],
        [f"output.root={tmp_path / 'results'}", "source_commit=test-commit"],
    )
    config = replace(config, method=replace(config.method, name="unknown"))
    metrics = run_atomic(config)
    assert metrics.status == RunStatus.FAILED
    assert "unknown method" in (metrics.message or "")
