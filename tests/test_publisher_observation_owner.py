"""A worker started as ``__main__`` keeps its observation owner (capture-447ae4f4, 2026-11-01).

Covers (for impacted-test selection):
  src/blueprint_pipeline/control_plane_registered_reference_gate.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_registered_reference_gate as gate
from blueprint_pipeline.control_plane_lane_experiment_errors import OwnerTargetVersionError

DISPATCHER = "blueprint_pipeline.task_evaluation_launch_dispatcher"


def _defined_as_main(source: str, *, spec_name: str | None) -> dict:
    """Execute ``source`` the way ``runpy.run_module(name, run_name="__main__")`` does."""
    namespace = {"__name__": "__main__",
                 "__spec__": importlib.util.find_spec(spec_name) if spec_name else None,
                 "gate": gate}
    exec(compile(source, "<worker>", "exec"), namespace)  # noqa: S102 - fixed test source
    return namespace


_DISPATCH = (
    "@gate._launch_admission_observation\n"
    "def dispatch_launch_request():\n"
    "    gate._finish_publisher_admission()\n"
    "    return 'admitted'\n"
)


def test_a_dispatcher_run_as_main_owns_and_finishes_its_admission() -> None:
    """The supervisor runs workers with run_name="__main__"; the check compared that name."""
    namespace = _defined_as_main(_DISPATCH, spec_name=DISPATCHER)
    function = namespace["dispatch_launch_request"]
    assert function.__module__ == "__main__"
    assert gate._owner_identity(function.__wrapped__) == (DISPATCHER, "dispatch_launch_request")
    assert function() == "admitted"


def test_a_function_without_the_dispatchers_identity_still_cannot_finish() -> None:
    for spec_name in (None, "blueprint_pipeline.task_evaluation_launch_reconciler"):
        function = _defined_as_main(_DISPATCH, spec_name=spec_name)["dispatch_launch_request"]
        with pytest.raises(OwnerTargetVersionError, match="experiment_publisher_input_limit"):
            function()


def test_the_dispatcher_module_run_by_runpy_finishes_its_admission(tmp_path: Path) -> None:
    """End to end through the real module, exactly as the supervisor starts it."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import inspect, runpy, sys\n"
        "namespace = runpy.run_module(%r, run_name='__main__probe__')\n"
        "function = namespace['dispatch_launch_request']\n"
        "from blueprint_pipeline import control_plane_registered_reference_gate as gate\n"
        "original = inspect.unwrap(function)\n"
        "assert original.__module__ == '__main__probe__', original.__module__\n"
        "assert gate._owner_identity(original) == (%r, 'dispatch_launch_request'), gate._owner_identity(original)\n"
        "print('ok')\n" % (DISPATCHER, DISPATCHER))
    source = Path(gate.__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120,
                               env={"PYTHONPATH": str(source), "PATH": "/usr/bin:/bin"}, check=False)
    assert completed.returncode == 0 and completed.stdout.strip() == "ok", completed.stderr[-2000:]
