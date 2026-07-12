"""Atomic artifact writing and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hdpde.utils.config import RunConfig


def source_commit(repo_root: str | Path = ".") -> str:
    """Return Git HEAD, bundle witness, or an explicit environment override."""

    if value := os.getenv("HDPDE_SOURCE_COMMIT"):
        return value.strip()
    witness = Path(repo_root) / "SOURCE_COMMIT"
    if witness.is_file():
        return witness.read_text(encoding="utf-8").strip()
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def make_run_id(config: RunConfig) -> str:
    """Build a human-readable, content-bound run identity."""

    def safe(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)

    return "__".join(
        [
            safe(config.method.name),
            safe(config.equation.name),
            f"d{config.equation.dim}",
            f"s{config.seed}",
            config.precision,
            config.config_hash,
        ]
    )


def prepare_run_dir(config: RunConfig) -> Path:
    root = Path(config.output.root)
    path = root / make_run_id(config)
    path.mkdir(parents=True, exist_ok=True)
    (path / "ckpt").mkdir(exist_ok=True)
    return path


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON through a same-directory temporary file and atomic replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=target.parent, delete=False, encoding="utf-8"
    ) as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
