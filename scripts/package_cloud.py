#!/usr/bin/env python3
"""Build a secret-safe cloud source bundle with commit and SHA256 metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "results",
    "outputs",
    "checkpoints",
    "external",
    "preflight",
}
EXCLUDED_NAMES = {".env", ".DS_Store"}
EXCLUDED_SUFFIXES = {".jwt", ".token", ".pem", ".key", ".pdf"}


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        not EXCLUDED_PARTS.intersection(relative.parts)
        and path.name not in EXCLUDED_NAMES
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def tracked_files(root: Path) -> list[Path]:
    """Return only Git-tracked files and reject an uncommitted source tree."""

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("cloud bundles require a clean Git working tree")
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in listed.stdout.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("dist/hdpde-cloud.tar.gz"))
    args = parser.parse_args()
    root = args.root.resolve()
    files = sorted(path for path in tracked_files(root) if path.is_file() and _included(path, root))
    commit = source_commit(root)
    checksums = {str(path.relative_to(root)): _sha256(path) for path in files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        metadata = Path(temporary)
        (metadata / "SOURCE_COMMIT").write_text(commit + "\n", encoding="utf-8")
        (metadata / "SHA256SUMS.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with tarfile.open(args.output, "w:gz") as archive:
            for path in files:
                archive.add(path, arcname=Path("hdpde") / path.relative_to(root))
            archive.add(metadata / "SOURCE_COMMIT", arcname="hdpde/SOURCE_COMMIT")
            archive.add(metadata / "SHA256SUMS.json", arcname="hdpde/SHA256SUMS.json")
    sidecar_path = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar_path.write_text(f"{_sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(args.output)
    print(sidecar_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
