"""Doc contract: the flags and console scripts promised by ``docs/LIVE_PIPELINE_SETUP.md``
must stay wired to the live-pipeline entrypoints.

The setup doc instructs operators to drive intake with a fixed set of ``--stage-*`` /
input flags and to run a fixed set of ``blueprint-*`` console scripts. If a CLI flag is
renamed or a console-script entry point is dropped from ``pyproject.toml`` the documented
runbook silently breaks. This test pins both contracts.

The argparse parsers are constructed inline inside ``main()`` and are not exposed as
objects, so we invoke ``main(["--help"])`` (which prints help and raises ``SystemExit(0)``
before doing any work) and assert the documented flags appear in the captured help text.
``pyproject.toml`` is read-only here -- we never mutate it (it is shared with another
agent); we only grep its text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import blueprint_pipeline
import blueprint_pipeline.live_pipeline_input_intake as input_intake
import blueprint_pipeline.live_pipeline_proof_audit as proof_audit

REPO_ROOT = Path(blueprint_pipeline.__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Intake CLI flags documented in docs/LIVE_PIPELINE_SETUP.md.
DOCUMENTED_INTAKE_FLAGS = (
    "--stage-webapp-request",
    "--stage-arena-results",
    "--stage-policy-package",
    "--stage-real-robot-pov",
    "--real-robot-pov",
    "--policy-package",
)

# Console scripts the setup runbook tells operators to invoke.
DOCUMENTED_CONSOLE_SCRIPTS = (
    "blueprint-audit-live-pipeline-setup",
    "blueprint-run-live-pipeline-control-plane",
    "blueprint-audit-live-pipeline-proof-boundary",
    "blueprint-intake-live-pipeline-inputs",
    "blueprint-live-pipeline-intake-service",
)


def _help_text(main, capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_intake_cli_recognizes_documented_flags(capsys: pytest.CaptureFixture[str]) -> None:
    help_text = _help_text(input_intake.main, capsys)
    for flag in DOCUMENTED_INTAKE_FLAGS:
        assert flag in help_text, flag


def test_proof_audit_cli_recognizes_require_live_ready(
    capsys: pytest.CaptureFixture[str],
) -> None:
    help_text = _help_text(proof_audit.main, capsys)
    assert "--require-live-ready" in help_text


def test_pyproject_declares_documented_console_scripts() -> None:
    # Read-only: never edit pyproject.toml (shared with a concurrent agent).
    text = PYPROJECT.read_text(encoding="utf-8")
    for script in DOCUMENTED_CONSOLE_SCRIPTS:
        # Match the entry-point key at the start of an assignment line.
        assert f"\n{script} = " in text, script
