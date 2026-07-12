from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hdpde.solvers.deep_bsde import (
    DeepBSDEConfig,
    DeepBSDEModel,
    DeepBSDESolver,
    DeepBSDETrainer,
    SharedZNetwork,
    StepwiseZNetwork,
    configure_precision,
)
from hdpde.types import RunMetrics, RunStatus
from hdpde.utils.config import EquationSpec, MethodSpec, OutputSpec, RunConfig


class HeatEquation:
    T = 0.25

    def __init__(self, dimension: int) -> None:
        self.x0 = torch.zeros(dimension)

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)

    def diffusion(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del t, x
        return torch.tensor(1.0)

    def nonlinearity(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        del t, x, z
        return torch.zeros_like(y)

    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x.square(), dim=1)

    def reference_solution(self) -> object:
        return {"value": self.x0.numel() * self.T}


class NonFiniteEquation(HeatEquation):
    def terminal(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), torch.nan, device=x.device, dtype=x.dtype)


@pytest.mark.parametrize("variant", ["m1a", "m1b"])
def test_variants_have_trainable_u0_z0_and_expected_network(variant: str) -> None:
    model = DeepBSDEModel(dimension=3, steps=4, variant=variant)
    assert model.u0.requires_grad
    assert model.z0.shape == (3,)
    if variant == "m1a":
        assert isinstance(model.z_network, StepwiseZNetwork)
        assert len(model.z_network.networks) == 3
        first_layer = model.z_network.networks[0][0]
        assert (first_layer.in_features, first_layer.out_features) == (3, 13)
        assert isinstance(model.z_network.networks[0][1], torch.nn.BatchNorm1d)
        assert isinstance(model.z_network.networks[0][2], torch.nn.ReLU)
        assert isinstance(model.z_network.networks[0][4], torch.nn.BatchNorm1d)
        assert isinstance(model.z_network.networks[0][5], torch.nn.ReLU)
    else:
        assert isinstance(model.z_network, SharedZNetwork)
        first_layer = model.z_network.network[0]
        assert (first_layer.in_features, first_layer.out_features) == (4, 13)
        assert isinstance(model.z_network.network[1], torch.nn.Tanh)


@pytest.mark.parametrize(
    ("precision", "expected_dtype", "expected_tf32"),
    [
        ("fp32", torch.float32, False),
        ("tf32", torch.float32, True),
        ("fp64", torch.float64, False),
    ],
)
def test_precision_modes_are_distinct(
    precision: str, expected_dtype: torch.dtype, expected_tf32: bool
) -> None:
    dtype = configure_precision(precision)  # type: ignore[arg-type]
    assert dtype == expected_dtype
    assert torch.backends.cuda.matmul.allow_tf32 is expected_tf32
    assert torch.backends.cudnn.allow_tf32 is expected_tf32


@pytest.mark.parametrize("variant", ["m1a", "m1b"])
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_training_smoke_preserves_precision_and_returns_metrics(
    variant: str, precision: str
) -> None:
    config = DeepBSDEConfig(
        dimension=2,
        steps=3,
        batch_size=8,
        iterations=2,
        variant=variant,  # type: ignore[arg-type]
        precision=precision,  # type: ignore[arg-type]
        seed=9,
    )
    trainer = DeepBSDETrainer(HeatEquation(2), config)
    result = trainer.train()
    assert result.status == "completed"
    assert result.iterations_completed == 2
    assert result.final_loss is not None
    assert result.estimate is not None
    assert result.parameter_count > 0
    assert next(trainer.model.parameters()).dtype == config.dtype


def test_checkpoint_resume_restores_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    config = DeepBSDEConfig(
        dimension=2,
        steps=3,
        batch_size=5,
        iterations=4,
        learning_rate=2.0e-3,
        scheduler_gamma=0.9,
        seed=123,
    )

    uninterrupted = DeepBSDETrainer(HeatEquation(2), config)
    uninterrupted_result = uninterrupted.train()

    interrupted = DeepBSDETrainer(HeatEquation(2), config)
    partial_result = interrupted.train(checkpoint_path=path, stop_after=2)
    assert partial_result.iterations_completed == 2

    resumed = DeepBSDETrainer(HeatEquation(2), config)
    resumed.load_checkpoint(path)
    resumed_result = resumed.train()

    assert uninterrupted_result.history == resumed_result.history
    assert uninterrupted.scheduler.state_dict() == resumed.scheduler.state_dict()
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert (
        uninterrupted.optimizer.state_dict()["param_groups"]
        == (resumed.optimizer.state_dict()["param_groups"])
    )


def test_resume_rejects_configuration_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    config = DeepBSDEConfig(dimension=2, steps=2, iterations=1)
    trainer = DeepBSDETrainer(HeatEquation(2), config)
    trainer.train(checkpoint_path=path)

    mismatched = DeepBSDETrainer(HeatEquation(2), replace(config, seed=1))
    with pytest.raises(ValueError, match="configuration mismatch"):
        mismatched.load_checkpoint(path)


def test_nonfinite_loss_is_recorded_as_diverged() -> None:
    config = DeepBSDEConfig(dimension=2, steps=2, batch_size=4, iterations=2)
    result = DeepBSDESolver(NonFiniteEquation(2)).run(config)
    assert result.status == "diverged"
    assert result.iterations_completed == 0
    assert result.error == "non-finite loss"


def test_bad_equation_shape_is_recorded_as_failed() -> None:
    class BadEquation(HeatEquation):
        def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            del t
            return torch.zeros(x.shape[0], x.shape[1] + 1)

    config = DeepBSDEConfig(dimension=2, steps=2, batch_size=4, iterations=1)
    result = DeepBSDESolver(BadEquation(2)).run(config)
    assert result.status == "failed"
    assert result.error is not None
    assert "unsupported drift shape" in result.error


def test_public_run_config_maps_to_run_metrics(tmp_path: Path) -> None:
    config = RunConfig(
        equation=EquationSpec(name="E5", dim=2),
        method=MethodSpec(
            name="M1b",
            params={
                "steps": 2,
                "batch_size": 4,
                "iterations": 1,
                "compute_reference": False,
            },
        ),
        seed=11,
        precision="fp64",
        device="cpu",
        output=OutputSpec(root=str(tmp_path), checkpoint_every=1),
        source_commit="abc123",
    )
    metrics = DeepBSDESolver(HeatEquation(2)).run(config)
    assert isinstance(metrics, RunMetrics)
    assert metrics.status is RunStatus.COMPLETED
    assert metrics.method == "M1b"
    assert metrics.config_hash == config.config_hash
    assert metrics.source_commit == "abc123"
    assert metrics.n_params is not None
    assert metrics.reference is None
    assert Path(metrics.artifacts["checkpoint"]).is_file()


def test_public_shared_method_name_selects_m1b() -> None:
    config = RunConfig(
        equation=EquationSpec(name="E5", dim=2),
        method=MethodSpec(
            name="deep_bsde_shared",
            params={"iterations": 1, "compute_reference": False},
        ),
        device="cpu",
    )
    internal, _, _ = DeepBSDESolver._from_run_config(config)
    assert internal.variant == "m1b"
