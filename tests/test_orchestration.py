from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args], cwd=ROOT, check=True, capture_output=True, text=True
    )


def test_manifest_campaign_counts() -> None:
    expected = {"main.yaml": 120, "rank.yaml": 45, "prec.yaml": 36, "scale.yaml": 6}
    for name, count in expected.items():
        result = _run("scripts/make_manifest.py", f"configs/experiments/{name}", "--dry-run")
        assert json.loads(result.stdout)["runs"] == count


def test_preflight_overrides_slurm_defaults(tmp_path) -> None:
    manifest = tmp_path / "runs.csv"
    _run(
        "scripts/make_manifest.py",
        "configs/experiments/rank.yaml",
        "--output",
        str(manifest),
    )
    profile = {
        "clusters": {
            "cpu": {
                "partition": "LIVE-CPU",
                "cpus_per_task": 7,
                "throttle": 3,
            }
        }
    }
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    generated = tmp_path / "slurm"
    _run(
        "scripts/make_slurm.py",
        str(manifest),
        "--preflight",
        str(profile_path),
        "--output-dir",
        str(generated),
    )
    text = (generated / "rank_cpu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --partition=LIVE-CPU" in text
    assert "#SBATCH --ntasks=1" in text
    assert "#SBATCH --cpus-per-task=7" in text
    assert "#SBATCH --array=" in text and "%3" in text


def test_ledger_matches_array_index() -> None:
    spec = importlib.util.spec_from_file_location(
        "collect_sacct", ROOT / "scripts/collect_sacct.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = [{"array_index": "3", "run_id": "run-three"}]
    record = dict.fromkeys(module.SACCT_FIELDS, "")
    record.update({"JobIDRaw": "8123_3", "JobName": "hdpde-main", "State": "COMPLETED"})
    fields, rows = module.merge_ledger(manifest, [record], "hdpde-")
    assert "sacct_state" in fields
    assert rows[0]["run_id"] == "run-three"
    assert rows[0]["sacct_state"] == "COMPLETED"
    assert rows[0]["manifest_match"] == "true"


def test_manifest_command_uses_atomic_cli(tmp_path) -> None:
    output = tmp_path / "rank.csv"
    _run(
        "scripts/make_manifest.py",
        "configs/experiments/rank.yaml",
        "--output",
        str(output),
    )
    with output.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["command"].startswith("python scripts/run_one.py --method tt_cross")
    assert "--resume" in row["command"]
    assert "method.params.target_eps=0.01" in row["command"]


def test_scale_generates_separate_one_and_two_gpu_arrays(tmp_path) -> None:
    manifest = tmp_path / "scale.csv"
    _run(
        "scripts/make_manifest.py",
        "configs/experiments/scale.yaml",
        "--output",
        str(manifest),
    )
    generated = tmp_path / "slurm"
    _run(
        "scripts/make_slurm.py",
        str(manifest),
        "--allow-defaults",
        "--output-dir",
        str(generated),
    )
    one = (generated / "scale_a100_g1.sbatch").read_text(encoding="utf-8")
    two = (generated / "scale_a100_g2.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in one and "%2" in one
    assert "#SBATCH --gres=gpu:2" in two and "%1" in two
    assert "torchrun" in two and "--nproc_per_node={world_size}" in two
