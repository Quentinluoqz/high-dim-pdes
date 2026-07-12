# high-dim-pdes

Reproducible comparison of Deep BSDE and tensor-train methods for parabolic PDEs
in dimensions 5--100. The project separates portable science code, experiment
manifests, USTC 107 Slurm execution, and provenance-checked result integration.

## What is implemented

- Five benchmark equations behind one batched PyTorch interface.
- Analytic E1/E2/E5 references, batched Cole--Hopf Monte Carlo for E3, and the
  canonical literature reference boundary for E4.
- Deep BSDE M1a (per-time-step networks) and M1b (shared network), including
  fp32/TF32/fp64 policy and exact checkpoint resume state.
- Tier-1 TT-cross representation experiments with `teneva`; Tier 2 remains an
  explicit external feasibility gate because it depends on xerus/SALSA.
- Atomic run output, manifest expansion, Slurm array generation, accounting,
  aggregation, figures, and a local-to-cloud evidence handoff.

The original scientific design is retained in
[`hdpde_project_plan.md`](hdpde_project_plan.md). The platform training PDF is
intentionally ignored and is not redistributed.

## Local setup

Python 3.12 and [`uv`](https://docs.astral.sh/uv/) are the supported baseline.

```bash
uv sync --extra dev
uv run pytest -m "not slow and not gpu"
```

The local machine is used for unit tests and reduced CPU smoke runs. Scientific
high-dimensional acceptance runs belong on allocated compute nodes.

```bash
uv run python scripts/run_one.py \
  --method deep_bsde_orig --equation e5 --dim 2 --seed 0 --precision fp32 \
  --device cpu --override method.params.iterations=20 \
  --override method.params.compute_reference=false
```

Every run writes `config.json`, `metrics.json`, `train.log`, and resumable
checkpoints under a content-bound `results/<run_id>/` directory. Raw results are
ignored by Git; stable summaries belong under `docs/results/`.

## Quality gates

```bash
uv run ruff format --check src tests experiments scripts
uv run ruff check src tests experiments scripts
uv run mypy src
uv run pytest -m "not slow and not gpu"
uv run pre-commit run --all-files
```

Slow and GPU tests are explicit pytest markers and are never presented as locally
verified when the required hardware is absent.

## Experiment workflow

1. Edit a grid under `configs/experiments/`.
2. Expand and inspect it before submission:

   ```bash
   uv run python scripts/make_manifest.py --config configs/experiments/main.yaml --dry-run
   uv run python scripts/make_manifest.py --config configs/experiments/main.yaml --out runs/main.csv
   ```

3. Run `scripts/cloud_preflight.sh` on the USTC login node and retain its output.
4. Generate partition-specific arrays from the approved manifest; submit pilots before
   the full campaign.
5. Return metrics and `sacct` evidence, then aggregate and plot locally:

   ```bash
   uv run python scripts/aggregate.py --results results --ledger ledger.csv
   uv run python scripts/plots.py --summary results/summary.csv --out figs
   uv run python docs/numbers.py --summary results/summary.csv
   ```

The full cloud handoff and security rules are in
[`docs/platform/ustc107.md`](docs/platform/ustc107.md). Never commit or send a
Slurm JWT; REST access is read-only and reads the token from the process environment.

## Repository layout

```text
src/hdpde/          equations, references, solvers, metrics, utilities
configs/            equations, methods, and immutable campaign definitions
scripts/            atomic execution, orchestration, accounting, aggregation
slurm/              generated/reviewed USTC 107 templates
infra/              local/remote environment policy
tests/              fast numerical, resume, TT, and orchestration checks
docs/               platform runbook, evidence, findings, report and demo
figs/               selected publication figures (raw runs remain ignored)
```

## Scientific integrity rules

- No defensible reference means no reported relative error.
- Diverged, failed, timed-out, and target-not-reached runs remain in the ledger.
- Config hash and source commit must match before results are aggregated.
- E3's hard target is less than 1% relative error; 0.5% is a stretch target.
- Static training-deck QoS values are defaults only; live Slurm output wins.

## Citations

See [`docs/references.md`](docs/references.md) and [`THIRD_PARTY.md`](THIRD_PARTY.md).
