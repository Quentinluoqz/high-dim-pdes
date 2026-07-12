"""Fail-closed registries for equations and methods."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_EQUATIONS: dict[str, Callable[..., Any]] = {}
_METHODS: dict[str, Callable[..., Any]] = {}


def _register[T](store: dict[str, Callable[..., Any]], name: str, factory: T) -> T:
    if not name or name in store:
        raise ValueError(f"duplicate or empty registry key: {name!r}")
    if not callable(factory):
        raise TypeError("registered factory must be callable")
    store[name] = factory  # type: ignore[assignment]
    return factory


def register_equation[T](name: str) -> Callable[[T], T]:
    """Register an equation class or factory."""

    return lambda factory: _register(_EQUATIONS, name, factory)


def register_method[T](name: str) -> Callable[[T], T]:
    """Register a solver class or factory."""

    return lambda factory: _register(_METHODS, name, factory)


def build_equation(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered equation."""

    try:
        return _EQUATIONS[name](**kwargs)
    except KeyError as exc:
        raise KeyError(f"unknown equation {name!r}; available={sorted(_EQUATIONS)}") from exc


def build_method(name: str, **kwargs: Any) -> Any:
    """Instantiate a registered method."""

    try:
        return _METHODS[name](**kwargs)
    except KeyError as exc:
        raise KeyError(f"unknown method {name!r}; available={sorted(_METHODS)}") from exc


def available_equations() -> tuple[str, ...]:
    return tuple(sorted(_EQUATIONS))


def available_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def clear_registries() -> None:
    """Clear global registries. Intended for isolated tests only."""

    _EQUATIONS.clear()
    _METHODS.clear()
