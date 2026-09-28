"""Every retained closeout can supply all evidence to its existing validator."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from blueprint_pipeline.materializer_cli import build_parser, call_arguments

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("closeouts", ROOT / "scripts/retain_appearance_native_closeouts.py")
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)


@pytest.mark.parametrize("name", sorted(MODULE.STEPS))
def test_closeout_cli_binds_every_required_input(name, tmp_path):
    step = MODULE.STEPS[name]
    assert set(step.params) == set(inspect.signature(step.materialize).parameters)
    argv = [name]
    expected = {}
    for keyword, param in step.params.items():
        assert param.required
        value = str(tmp_path / keyword)
        if param.json_file:
            Path(value).write_text('[{"evidence":"retained"}]')
            expected[keyword] = [{"evidence": "retained"}]
        else:
            expected[keyword] = value
        argv.extend([param.flag, value])
    assert call_arguments(step, build_parser(MODULE.STEPS).parse_args(argv)) == expected
