from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("HDPDE_RUN_DDP_TEST") != "1",
    reason="set HDPDE_RUN_DDP_TEST=1 where local rendezvous sockets are permitted",
)
def test_two_process_cpu_ddp_atomic_run(tmp_path: Path) -> None:
    results = tmp_path / "results"
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        "scripts/run_one.py",
        "--method",
        "deep_bsde_orig",
        "--equation",
        "e5",
        "--dim",
        "2",
        "--seed",
        "0",
        "--precision",
        "fp32",
        "--device",
        "cpu",
        "--override",
        "method.params.iterations=1",
        "--override",
        "method.params.steps=2",
        "--override",
        "method.params.world_size=2",
        "--override",
        "method.params.global_batch_size=4",
        "--override",
        "method.params.compute_reference=false",
        "--override",
        f"output.root={results}",
        "--override",
        "source_commit=ddp-test",
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    metrics_paths = list(results.glob("*/metrics.json"))
    assert len(metrics_paths) == 1
    metrics = json.loads(metrics_paths[0].read_text(encoding="utf-8"))
    assert metrics["status"] == "completed"
    assert metrics["hardware"]["world_size"] == 2
    assert metrics["hardware"]["global_batch_size"] == 4
