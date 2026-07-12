# Agent guidance

## Architecture

- `src/hdpde` is portable algorithm code and must not import from `scripts`, `slurm`,
  `experiments`, or `infra`.
- Configuration is strict. Unknown YAML and CLI keys are errors.
- All randomness passes through an explicit seed. Never use global, unrecorded seeds.
- All atomic executions go through `scripts/run_one.py`.
- Raw outputs, checkpoints, logs, tokens, and machine-specific paths are never committed.

## Scientific rules

- Never report an error without a `ReferenceResult` carrying provenance.
- Preserve `diverged`, `failed`, `timeout`, and `not_reached` outcomes as data.
- TF32 means float32 tensors plus CUDA TF32 matmul; it never means AMP or FP16.
- A resumed checkpoint must match the complete configuration hash.
- Do not make Tier-2 solver code part of this repository without a recorded license review.

## Required gates

```bash
uv run ruff format --check src tests experiments scripts
uv run ruff check src tests experiments scripts
uv run mypy src
uv run pytest -m "not slow and not gpu"
uv run pre-commit run --all-files
```

Formatting or auto-fix commands may be run only after inspecting the affected scope and diff.
