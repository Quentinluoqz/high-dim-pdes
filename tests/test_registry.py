from __future__ import annotations

import pytest

from hdpde.utils.registry import (
    available_equations,
    build_equation,
    clear_registries,
    register_equation,
)


def setup_function() -> None:
    clear_registries()


def teardown_function() -> None:
    clear_registries()


def test_registry_builds_and_rejects_duplicates() -> None:
    @register_equation("demo")
    class Demo:
        def __init__(self, dim: int) -> None:
            self.dim = dim

    assert build_equation("demo", dim=7).dim == 7
    assert available_equations() == ("demo",)
    with pytest.raises(ValueError, match="duplicate"):
        register_equation("demo")(Demo)


def test_unknown_registry_key_lists_available() -> None:
    with pytest.raises(KeyError, match="available"):
        build_equation("missing")
