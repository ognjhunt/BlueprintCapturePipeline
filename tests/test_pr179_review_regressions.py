"""Regression guards for the defects raised in review of PR #179.

Each test pins one specific failure mode that review caught, so a later refactor
that reintroduces it fails here rather than in production.
"""

from __future__ import annotations

import argparse
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import vast_provider_adapter as vast

# -- P1: the CLI option must be accepted by the function main() forwards to ---


def test_adapter_accepts_the_gpu_selection_policy_keyword() -> None:
    """`main()` forwards `vars(args)`, so every CLI option needs a parameter.

    Without this, every console/module invocation of the Vast adapter raised
    TypeError before it could write even a dry-run result.
    """

    parameters = inspect.signature(vast.run_vast_provider_adapter).parameters
    assert "gpu_selection_policy" in parameters


def test_every_adapter_cli_option_maps_to_a_run_parameter(monkeypatch) -> None:
    """Generalises the above so the next added flag cannot reintroduce it.

    The parser is built inline in ``main()``, so it is captured by intercepting
    ``parse_args`` rather than skipped -- the full test lane treats any skip as
    missing evidence.
    """

    captured: list[argparse.ArgumentParser] = []
    real_parse_args = argparse.ArgumentParser.parse_args

    def _capture(self, *parse_args, **parse_kwargs):
        captured.append(self)
        return real_parse_args(self, *parse_args, **parse_kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _capture)
    with pytest.raises(SystemExit):
        vast.main(["--help"])

    assert captured, "adapter parser was never constructed"
    parser = captured[0]
    parameters = set(inspect.signature(vast.run_vast_provider_adapter).parameters)
    # main() renames a few dests before forwarding; mirror that mapping.
    renames = {
        "allow_vast_instance_launch": "allow_instance_launch",
        "machine_avoidlist": "machine_avoidlist_path",
        "allowed_machine_id": "allowed_machine_ids",
        "session_budget_ledger": "session_budget_ledger_path",
    }
    missing = sorted(
        {
            renames.get(action.dest, action.dest)
            for action in parser._actions
            if action.dest != "help"
        }
        - parameters
    )
    assert not missing, f"CLI options with no run parameter: {missing}"


def test_adapter_dry_run_completes(tmp_path: Path) -> None:
    """End-to-end proof the forwarding path works, not just the signature."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.vast_provider_adapter",
            "--job-dir",
            str(tmp_path / "job"),
            "--mode",
            "dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "dry_run_ready" in completed.stdout


# -- P1: a failed startup must not leak the worker process ------------------


_FAILING_WORKER = textwrap.dedent(
    """
    import json, sys, time
    # Announce a payload the client must reject, then stay alive holding the
    # "GPU" so a leak is observable.
    print(json.dumps({"schema_version": "oscar_resident_worker_ready.v1",
                      "status": "ready", "model_load_seconds": 1.0}))
    sys.stdout.flush()
    time.sleep(300)
    """
)




