#!/usr/bin/env python3
"""Thin script wrapper around :mod:`hdpde.cli`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hdpde.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
