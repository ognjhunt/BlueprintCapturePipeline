from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/bootstrap_measurement_drake_development.py"
SCHEMA = ROOT / "docs/schemas/measurement_drake_development_environment.v1.schema.json"


def _script_namespace() -> dict:
    return runpy.run_path(str(SCRIPT), run_name="measurement_drake_bootstrap_test")


def test_drake_bootstrap_builds_digest_bound_nonproduction_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = _script_namespace()
    bootstrap = namespace["bootstrap"]
    python = tmp_path / "python3.13"
    uv = tmp_path / "uv"
    python.write_text("python", encoding="utf-8")
    uv.write_text("uv", encoding="utf-8")
    environment = tmp_path / "drake-environment"
    commands: list[list[str]] = []

    def fake_run(argv: list[str]) -> None:
        commands.append(list(argv))
        if argv[1] == "venv":
            worker = environment / "bin/python"
            worker.parent.mkdir(parents=True)
            worker.write_text("python", encoding="utf-8")

    monkeypatch.setitem(bootstrap.__globals__, "_run", fake_run)
    monkeypatch.setattr(
        bootstrap.__globals__["subprocess"],
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"drake_version": "1.55.0", "python_version": "3.13.5"}),
            stderr="",
        ),
    )
    receipt = bootstrap(python=python, environment=environment, uv=uv)
    jsonschema.validate(receipt, json.loads(SCHEMA.read_text(encoding="utf-8")))
    supplied = receipt["bootstrap_receipt_digest"]
    unsigned = dict(receipt)
    unsigned.pop("bootstrap_receipt_digest")
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert supplied == expected
    assert receipt["development_only"] is True
    assert receipt["production_route_eligible"] is False
    assert receipt["r7_admission"] is False
    assert commands[0][1:3] == ["venv", "--python"]
    assert commands[1][-1] == "drake==1.55.0"


def test_drake_bootstrap_refuses_existing_environment(tmp_path: Path) -> None:
    namespace = _script_namespace()
    environment = tmp_path / "existing"
    environment.mkdir()
    python = tmp_path / "python3.13"
    uv = tmp_path / "uv"
    python.write_text("python", encoding="utf-8")
    uv.write_text("uv", encoding="utf-8")
    with pytest.raises(
        namespace["DrakeBootstrapError"],
        match="environment_must_be_new_explicit_path",
    ):
        namespace["bootstrap"](python=python, environment=environment, uv=uv)
