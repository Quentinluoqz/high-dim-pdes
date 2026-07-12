#!/usr/bin/env python3
"""Generate partition-specific Slurm arrays from a run manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def _load_profile(defaults: Path, preflight: Path | None) -> dict[str, Any]:
    with defaults.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if preflight is None:
        return profile
    with preflight.open(encoding="utf-8") as handle:
        override = yaml.safe_load(handle)
    for key, value in override.items():
        if key == "clusters":
            for name, settings in value.items():
                profile.setdefault("clusters", {}).setdefault(name, {}).update(settings)
        else:
            profile[key] = value
    return profile


def _script(
    manifest: Path,
    campaign: str,
    cluster_name: str,
    array_indices: list[int],
    profile: dict[str, Any],
    requested_gpus: int,
) -> str:
    cluster = profile["clusters"][cluster_name]
    modules = "\n".join(f"module load {item}" for item in profile.get("modules", []))
    qos = f"#SBATCH --qos={cluster['qos']}\n" if cluster.get("qos") else ""
    gres = f"#SBATCH --gres=gpu:{requested_gpus}\n" if requested_gpus else ""
    manifest_path = str(manifest)
    python_executable = profile.get("python", "python")
    array_spec = ",".join(str(index) for index in array_indices)
    throttle = max(1, int(cluster["throttle"]) // max(1, requested_gpus))
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=hdpde-{campaign}-{cluster_name}
#SBATCH --partition={cluster["partition"]}
{qos}#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cluster["cpus_per_task"]}
{gres}#SBATCH --time={cluster["time_limit"]}
#SBATCH --array={array_spec}%{throttle}
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --signal=B:TERM@120

set -euo pipefail
MANIFEST="${{MANIFEST:-{manifest_path}}}"
REQUEUE_ON_TERM="${{REQUEUE_ON_TERM:-0}}"
CHILD_PID=""

on_term() {{
  if [[ -n "$CHILD_PID" ]]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" || true
  fi
  if [[ "$REQUEUE_ON_TERM" == "1" ]]; then
    scontrol requeue "$SLURM_JOB_ID"
  fi
}}
trap on_term TERM

mkdir -p logs
{modules}
source "{profile.get("venv", ".venv/bin/activate")}"

readarray -d '' -t RUN_ARGS < <(\
  "{python_executable}" - "$MANIFEST" "${{SLURM_ARRAY_TASK_ID}}" "{cluster_name}" <<'PY'
import csv
import json
import sys

manifest, index, cluster = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(manifest, newline='', encoding='utf-8') as handle:
    matches = [
        row for row in csv.DictReader(handle)
        if int(row['array_index']) == index and row['cluster'] == cluster
    ]
if len(matches) != 1:
    message = f'expected one row for array index {{index}} and {{cluster}}; got {{len(matches)}}'
    raise SystemExit(message)
row = matches[0]
world_size = int(row.get('world_size') or '1')
launcher = (
    ['torchrun', '--standalone', f'--nproc_per_node={{world_size}}']
    if world_size > 1 else [sys.executable]
)
args = launcher + [
    'scripts/run_one.py', '--method', row['method'],
    '--equation', row['equation'], '--dim', row['dim'], '--seed', row['seed'],
    '--precision', row['precision'], '--resume',
]
for override in json.loads(row['overrides']):
    args.extend(['--override', override])
for arg in args:
    sys.stdout.write(arg + '\\0')
PY
)

"${{RUN_ARGS[@]}}" &
CHILD_PID=$!
wait "$CHILD_PID"
CHILD_PID=""
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("slurm/generated"))
    parser.add_argument("--defaults", type=Path, default=Path("infra/cluster_defaults.yaml"))
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--allow-defaults",
        action="store_true",
        help="permit template generation without a verified cloud preflight",
    )
    args = parser.parse_args()
    if args.preflight is None and not args.allow_defaults:
        parser.error(
            "--preflight is required for real jobs (or use --allow-defaults for templates)"
        )
    profile = _load_profile(args.defaults, args.preflight)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest contains no runs")
    groups: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["cluster"], int(row.get("gpus", "0") or "0"))
        groups.setdefault(key, []).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign = rows[0]["campaign"]
    gpu_variants: dict[str, set[int]] = {}
    for cluster_name, requested_gpus in groups:
        gpu_variants.setdefault(cluster_name, set()).add(requested_gpus)
    for (cluster_name, requested_gpus), selected in groups.items():
        if cluster_name not in profile.get("clusters", {}):
            raise ValueError(f"cluster {cluster_name!r} is absent from profile")
        indices = [int(row["array_index"]) for row in selected]
        suffix = cluster_name
        if len(gpu_variants[cluster_name]) > 1 or requested_gpus > 1:
            suffix = f"{cluster_name}_g{requested_gpus}"
        output = args.output_dir / f"{campaign}_{suffix}.sbatch"
        output.write_text(
            _script(
                args.manifest,
                campaign,
                cluster_name,
                indices,
                profile,
                requested_gpus,
            ),
            encoding="utf-8",
        )
        output.chmod(0o755)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
