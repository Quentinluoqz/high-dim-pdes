#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  printf 'usage: %s MANIFEST CLUSTER\n' "$0" >&2
  exit 2
fi
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

MANIFEST="$1"
CLUSTER="$2"
CHILD_PID=""

on_term() {
  if [[ -n "$CHILD_PID" ]]; then
    kill -TERM "$CHILD_PID" 2>/dev/null || true
    wait "$CHILD_PID" || true
  fi
  if [[ "${REQUEUE_ON_TERM:-0}" == "1" ]]; then
    scontrol requeue "$SLURM_JOB_ID"
  fi
}
trap on_term TERM

readarray -d '' -t RUN_ARGS < <(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" "$CLUSTER" <<'PY'
import csv
import json
import sys

manifest, index, cluster = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(manifest, newline='', encoding='utf-8') as handle:
    matches = [row for row in csv.DictReader(handle)
               if int(row['array_index']) == index and row['cluster'] == cluster]
if len(matches) != 1:
    raise SystemExit(f'expected one row for index {index} and {cluster}, got {len(matches)}')
row = matches[0]
world_size = int(row.get('world_size') or '1')
launcher = (['torchrun', '--standalone', f'--nproc_per_node={world_size}']
            if world_size > 1 else [sys.executable])
args = launcher + ['scripts/run_one.py', '--method', row['method'],
        '--equation', row['equation'], '--dim', row['dim'], '--seed', row['seed'],
        '--precision', row['precision'], '--resume']
for override in json.loads(row['overrides']):
    args.extend(['--override', override])
for arg in args:
    sys.stdout.write(arg + '\0')
PY
)

"${RUN_ARGS[@]}" &
CHILD_PID=$!
wait "$CHILD_PID"
CHILD_PID=""
