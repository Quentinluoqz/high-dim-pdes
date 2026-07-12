from __future__ import annotations

from pathlib import Path

import pytest

from hdpde.utils.config import load_run_config, parse_overrides, run_config_from_dict


def test_smoke_config_loads_and_hash_is_stable() -> None:
    config = load_run_config(["configs/smoke.yaml"])
    same = load_run_config(["configs/smoke.yaml"])
    assert config.equation.name == "e5_heat"
    assert config.equation.dim == 2
    assert config.method.name == "deep_bsde_orig"
    assert config.config_hash == same.config_hash


def test_later_files_and_dotted_overrides_win() -> None:
    config = load_run_config(
        [
            "configs/defaults.yaml",
            "configs/equations/e1_bsb.yaml",
            "configs/methods/deep_bsde_orig.yaml",
        ],
        ["equation.dim=9", "method.params.batch_size=7", "precision=fp64"],
    )
    assert config.equation.dim == 9
    assert config.method.params["batch_size"] == 7
    assert config.precision == "fp64"


def test_unknown_top_level_key_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown run keys"):
        run_config_from_dict(
            {
                "equation": {"name": "e5_heat"},
                "method": {"name": "deep_bsde_orig"},
                "typo": True,
            }
        )


def test_missing_inherited_file_is_reported(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    child.write_text("inherits: missing.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_run_config([child])


def test_override_syntax() -> None:
    assert parse_overrides(["a=1", "b=true", "c=[x, y]"]) == {
        "a": 1,
        "b": True,
        "c": ["x", "y"],
    }
    with pytest.raises(ValueError):
        parse_overrides(["missing-separator"])
