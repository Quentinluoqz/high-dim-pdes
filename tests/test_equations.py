"""Tests for E1--E5 definitions and tensor contracts."""

from __future__ import annotations

import math

import pytest
import torch

from hdpde.equations import (
    HJBLQ,
    AllenCahn,
    BlackScholesBarenblatt,
    HeatEquation,
    LQControl,
    make_equation,
)


@pytest.mark.parametrize(
    "equation_type",
    [
        BlackScholesBarenblatt,
        LQControl,
        HJBLQ,
        AllenCahn,
        HeatEquation,
    ],
)
@pytest.mark.parametrize("dim", [1, 5, 50])
def test_batch_contract_preserves_dtype_and_device(equation_type: type[object], dim: int) -> None:
    equation = equation_type(dim)  # type: ignore[call-arg]
    x = torch.randn(4, dim, dtype=torch.float64)
    y = torch.randn(4, 1, dtype=torch.float64)
    z = torch.randn_like(x)

    assert equation.drift(0.0, x).shape == x.shape  # type: ignore[attr-defined]
    assert equation.diffusion(0.0, x).shape == (4, dim, dim)  # type: ignore[attr-defined]
    assert equation.terminal(x).shape == (4, 1)  # type: ignore[attr-defined]
    assert equation.nonlinearity(0.0, x, y, z).shape == y.shape  # type: ignore[attr-defined]
    for result in (
        equation.drift(0.0, x),  # type: ignore[attr-defined]
        equation.diffusion(0.0, x),  # type: ignore[attr-defined]
        equation.terminal(x),  # type: ignore[attr-defined]
        equation.nonlinearity(0.0, x, y, z),  # type: ignore[attr-defined]
    ):
        assert result.dtype == x.dtype
        assert result.device == x.device


def test_e1_exact_solution_matches_terminal_and_formula() -> None:
    equation = BlackScholesBarenblatt(5)
    x = torch.ones(2, 5, dtype=torch.float64)
    torch.testing.assert_close(equation.exact_solution(1.0, x), equation.terminal(x))
    expected = 5.0 * math.exp(0.05 + 0.4**2)
    assert equation.reference_solution().value == pytest.approx(expected)


def test_e2_riccati_solution_matches_terminal() -> None:
    equation = LQControl(4)
    x = torch.randn(7, 4, dtype=torch.float64)
    torch.testing.assert_close(equation.exact_solution(equation.T, x), equation.terminal(x))


def test_e3_and_e4_default_data() -> None:
    x = torch.zeros(2, 3, dtype=torch.float64)
    e3 = HJBLQ(3)
    e4 = AllenCahn(3)
    torch.testing.assert_close(e3.terminal(x), torch.full((2, 1), -math.log(2.0), dtype=x.dtype))
    torch.testing.assert_close(e4.terminal(x), torch.full((2, 1), 0.5, dtype=x.dtype))
    assert e4.reference_solution().value is None
    canonical = AllenCahn(100).reference_solution()
    assert canonical.value == pytest.approx(0.052802)


def test_e5_closed_form_matches_terminal() -> None:
    equation = HeatEquation(3)
    x = torch.randn(5, 3, dtype=torch.float64)
    torch.testing.assert_close(equation.exact_solution(equation.T, x), equation.terminal(x))
    expected = (3.0 / 5.0) ** 1.5
    assert equation.reference_solution().value == pytest.approx(expected)


def test_factory_and_validation() -> None:
    assert isinstance(make_equation("e1", 2), BlackScholesBarenblatt)
    with pytest.raises(ValueError, match="unknown equation"):
        make_equation("E9", 2)
    with pytest.raises(ValueError, match="shape"):
        HeatEquation(2).terminal(torch.zeros(3, 3))
