"""Tests for exact and Monte Carlo reference metadata."""

from __future__ import annotations

import math

import pytest
import torch

from hdpde.equations import HJBLQ
from hdpde.reference import ReferenceResult, cole_hopf_reference


def test_reference_result_is_json_compatible() -> None:
    result = ReferenceResult(1.0, (0.9, 1.1), "mc", "test", 100)
    assert result.to_dict() == {
        "value": 1.0,
        "ci95": (0.9, 1.1),
        "kind": "mc",
        "source": "test",
        "sample_count": 100,
    }


def test_cole_hopf_constant_terminal_is_exact() -> None:
    constant = 1.25

    def terminal(x: torch.Tensor) -> torch.Tensor:
        return torch.full(x.shape[:-1] + (1,), constant, dtype=x.dtype)

    result = cole_hopf_reference(
        terminal,
        x=torch.zeros(2),
        remaining_time=1.0,
        diffusion_scale=math.sqrt(2.0),
        lambda_=1.0,
        sample_count=1_001,
        batch_size=64,
        seed=9,
        source="unit test",
    )
    assert result.value == pytest.approx(constant, abs=1e-12)
    assert result.ci95 is not None
    assert result.ci95[0] == pytest.approx(constant, abs=1e-12)
    assert result.ci95[1] == pytest.approx(constant, abs=1e-12)


def test_e3_reference_is_seeded_batched_and_contains_value() -> None:
    equation = HJBLQ(1)
    small_batches = equation.reference_solution(sample_count=20_000, batch_size=257, seed=123)
    repeated = equation.reference_solution(sample_count=20_000, batch_size=257, seed=123)
    assert small_batches.value == repeated.value
    assert small_batches.ci95 == repeated.ci95
    assert small_batches.ci95 is not None
    assert small_batches.ci95[0] <= small_batches.value <= small_batches.ci95[1]
    assert small_batches.sample_count == 20_000


def test_e3_d1_confidence_interval_covers_quadrature_value() -> None:
    equation = HJBLQ(1)
    result = equation.reference_solution(sample_count=200_000, batch_size=10_000, seed=7)
    # E[2/(1+2 Z^2)] has this closed one-dimensional Gaussian integral.
    transformed_mean = math.sqrt(math.pi) * math.exp(0.25) * math.erfc(0.5)
    exact = -math.log(transformed_mean)
    assert result.ci95 is not None
    assert result.ci95[0] <= exact <= result.ci95[1]


def test_e3_reference_at_terminal_time_recovers_terminal_value() -> None:
    equation = HJBLQ(2)
    x = torch.tensor([1.0, -0.5], dtype=torch.float64)
    result = equation.reference_solution(t=equation.T, x=x, sample_count=100, batch_size=17, seed=0)
    expected = float(equation.terminal(x))
    assert result.value == pytest.approx(expected, abs=1e-12)
