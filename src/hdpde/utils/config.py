"""Strict YAML configuration loading and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(slots=True)
class EquationSpec:
    """A registered equation and its constructor parameters."""

    name: str
    dim: int = 5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MethodSpec:
    """A registered method and its solver parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutputSpec:
    """Run artifact policy."""

    root: str = "results"
    checkpoint_every: int = 100
    keep: tuple[str, ...] = ("best", "last")


@dataclass(slots=True)
class RunConfig:
    """Fully resolved configuration for one atomic run."""

    equation: EquationSpec
    method: MethodSpec
    seed: int = 0
    precision: str = "fp32"
    device: str = "auto"
    output: OutputSpec = field(default_factory=OutputSpec)
    resume: bool = False
    source_commit: str = "unknown"
    tags: dict[str, str] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.equation.dim < 1:
            raise ValueError("equation.dim must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.precision not in {"fp32", "tf32", "fp64"}:
            raise ValueError(f"unsupported precision: {self.precision}")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"unsupported device: {self.device}")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-ready representation."""

        return asdict(self)

    @property
    def config_hash(self) -> str:
        """Return a short content hash for provenance and resume checks."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"top-level YAML object must be a mapping: {path}")
    parents = raw.pop("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise TypeError(f"inherits must be a string or list of strings: {path}")
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(merged, _load_yaml_tree(path.parent / parent, (*stack, path)))
    return _deep_merge(merged, raw)


def _set_dotted(raw: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid override path: {key}")
    node = raw
    for part in parts[:-1]:
        current = node.setdefault(part, {})
        if not isinstance(current, dict):
            raise TypeError(f"override crosses non-mapping value at {part}")
        node = current
    node[parts[-1]] = value


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse repeatable ``key=value`` strings using YAML scalar semantics."""

    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must use key=value syntax: {item}")
        key, value = item.split("=", 1)
        parsed[key] = yaml.safe_load(value)
    return parsed


def _check_keys(raw: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {context} keys: {sorted(unknown)}")


def run_config_from_dict(raw: dict[str, Any]) -> RunConfig:
    """Construct a strict :class:`RunConfig` from a merged mapping."""

    _check_keys(
        raw,
        {
            "schema_version",
            "equation",
            "method",
            "seed",
            "precision",
            "device",
            "output",
            "resume",
            "source_commit",
            "tags",
        },
        "run",
    )
    eq = raw.get("equation")
    method = raw.get("method")
    output = raw.get("output", {})
    if not isinstance(eq, dict) or not isinstance(method, dict) or not isinstance(output, dict):
        raise TypeError("equation, method, and output must be mappings")
    _check_keys(eq, {"name", "dim", "params"}, "equation")
    _check_keys(method, {"name", "params"}, "method")
    _check_keys(output, {"root", "checkpoint_every", "keep"}, "output")
    if "name" not in eq or "name" not in method:
        raise ValueError("equation.name and method.name are required")
    if not isinstance(eq.get("params", {}), dict) or not isinstance(method.get("params", {}), dict):
        raise TypeError("equation.params and method.params must be mappings")
    keep = output.get("keep", ["best", "last"])
    if not isinstance(keep, (list, tuple)) or not all(isinstance(item, str) for item in keep):
        raise TypeError("output.keep must be a list of strings")
    return RunConfig(
        equation=EquationSpec(
            name=str(eq["name"]), dim=int(eq.get("dim", 5)), params=dict(eq.get("params", {}))
        ),
        method=MethodSpec(name=str(method["name"]), params=dict(method.get("params", {}))),
        seed=int(raw.get("seed", 0)),
        precision=str(raw.get("precision", "fp32")),
        device=str(raw.get("device", "auto")),
        output=OutputSpec(
            root=str(output.get("root", "results")),
            checkpoint_every=int(output.get("checkpoint_every", 100)),
            keep=tuple(keep),
        ),
        resume=bool(raw.get("resume", False)),
        source_commit=str(raw.get("source_commit", "unknown")),
        tags={str(k): str(v) for k, v in dict(raw.get("tags", {})).items()},
        schema_version=int(raw.get("schema_version", 1)),
    )


def load_run_config(paths: list[str | Path], overrides: list[str] | None = None) -> RunConfig:
    """Merge YAML files in order, apply overrides, and validate the result."""

    if not paths:
        raise ValueError("at least one configuration path is required")
    raw: dict[str, Any] = {}
    for path in paths:
        raw = _deep_merge(raw, _load_yaml_tree(Path(path)))
    for key, value in parse_overrides(overrides or []).items():
        _set_dotted(raw, key, value)
    return run_config_from_dict(raw)
