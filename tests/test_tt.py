from __future__ import annotations

import importlib.util

import pytest

from hdpde.equations import make_equation
from hdpde.solvers.tt import TTConfig, run_tt_cross
from hdpde.solvers.tt.tier2_wrapper import diagnose_tier2


def test_tt_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="dim"):
        TTConfig(dim=0, target_eps=1e-3)
    with pytest.raises(ValueError, match="target_eps"):
        TTConfig(dim=3, target_eps=0.0)


def test_tier2_absence_is_an_explicit_diagnostic(tmp_path) -> None:
    result = diagnose_tier2(tmp_path / "not-vendored")
    assert not result.ready
    assert not result.source_present
    assert any("not vendored" in blocker for blocker in result.blockers)


@pytest.mark.skipif(importlib.util.find_spec("teneva") is None, reason="teneva is optional")
def test_heat_equation_d3_has_low_tt_rank() -> None:
    # The heat-kernel solution is a product of one-dimensional Gaussian factors,
    # hence has true TT rank one. Rank <=5 leaves room for cross/truncation noise.
    equation = make_equation("E5", 3)
    result = run_tt_cross(
        equation,
        TTConfig(
            dim=3,
            equation_id="E5",
            target_eps=1e-3,
            grid_size=16,
            max_sweeps=10,
            max_samples=50_000,
            validation_samples=500,
            grid_test_samples=2_000,
            continuous_test_samples=500,
            seed=7,
        ),
    )
    assert result.status == "completed"
    assert result.grid_rel_l2 <= 1e-3
    assert result.max_rank <= 5
