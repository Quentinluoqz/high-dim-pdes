"""Diagnostics-only boundary for the optional external Tier-2 solver.

No upstream code is vendored. Users must separately review its license, pin a
commit, and build Xerus before an adapter may execute it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from shutil import which
from typing import Any


@dataclass(frozen=True, slots=True)
class Tier2Diagnostics:
    external_root: str
    source_present: bool
    git_commit: str | None
    license_files: tuple[str, ...]
    xerus_python_available: bool
    xerus_binary_available: bool
    ready: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_tier2(external_root: str | Path = "external/PDE-backward-solver") -> Tier2Diagnostics:
    """Perform a read-only readiness check for the external ICML 2021 code."""
    root = Path(external_root)
    licenses = (
        tuple(
            str(path.relative_to(root))
            for pattern in ("LICENSE*", "COPYING*")
            for path in sorted(root.glob(pattern))
            if path.is_file()
        )
        if root.is_dir()
        else ()
    )
    commit_file = root / "PINNED_COMMIT"
    commit = commit_file.read_text(encoding="utf-8").strip() if commit_file.is_file() else None
    xerus_python = find_spec("xerus") is not None
    xerus_binary = which("xerus") is not None
    blockers: list[str] = []
    if not root.is_dir():
        blockers.append("external solver source is absent (intentionally not vendored)")
    if root.is_dir() and not licenses:
        blockers.append("upstream solver license has not been identified")
    if root.is_dir() and not commit:
        blockers.append("PINNED_COMMIT is missing")
    if not (xerus_python or xerus_binary):
        blockers.append("Xerus is not available in Python or PATH")
    return Tier2Diagnostics(
        external_root=str(root),
        source_present=root.is_dir(),
        git_commit=commit,
        license_files=licenses,
        xerus_python_available=xerus_python,
        xerus_binary_available=xerus_binary,
        ready=not blockers,
        blockers=tuple(blockers),
    )


def require_tier2(external_root: str | Path = "external/PDE-backward-solver") -> Tier2Diagnostics:
    """Return diagnostics or fail without modifying external solver code."""
    diagnostics = diagnose_tier2(external_root)
    if not diagnostics.ready:
        details = "; ".join(diagnostics.blockers)
        raise RuntimeError(f"TT Tier 2 is unavailable: {details}")
    return diagnostics
