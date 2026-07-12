from __future__ import annotations

import json

from hdpde.utils.config import EquationSpec, MethodSpec, OutputSpec, RunConfig
from hdpde.utils.io import atomic_write_json, make_run_id, prepare_run_dir


def test_run_id_and_atomic_json(tmp_path) -> None:
    config = RunConfig(
        equation=EquationSpec("e5_heat", 2),
        method=MethodSpec("deep_bsde_orig"),
        output=OutputSpec(root=str(tmp_path / "results")),
    )
    assert "e5_heat" in make_run_id(config)
    run_dir = prepare_run_dir(config)
    target = run_dir / "metrics.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}
    assert (run_dir / "ckpt").is_dir()
