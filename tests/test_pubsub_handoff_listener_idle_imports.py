"""An idle listener tick does not load the pipeline it has no handoff for.

The listener starts every minute and almost always acknowledges nothing. It
imported ``run_e2e`` at module load -- the capture orchestrator, trimesh and
scipy, about 2.5 s of CPU per start compiled from source -- although only a
handoff ever calls it. The entrypoint is now imported on first use, and the
runtime env guard proves it imports at unit start instead.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import blueprint_pipeline.pubsub_handoff_listener as listener
import blueprint_pipeline.run_e2e as run_e2e
from blueprint_pipeline.production_runtime_env_guard import CONTROL_PLANE_ENTRYPOINTS

ROOT = Path(__file__).resolve().parents[1]


def test_importing_the_listener_leaves_the_pipeline_unloaded() -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import blueprint_pipeline.pubsub_handoff_listener;"
        "loaded = sorted(m for m in ('blueprint_pipeline.run_e2e', 'blueprint_pipeline.capture_orchestrator',"
        " 'trimesh', 'scipy') if m in sys.modules);"
        "assert not loaded, loaded"
    )
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT / "src")],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert child.returncode == 0, child.stderr


def test_a_handoff_still_runs_the_pipeline_entrypoint(monkeypatch) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(run_e2e, "run_end_to_end", lambda **kwargs: calls.append(kwargs) or {"status": "ok"})

    assert inspect.signature(listener.process_handoff_payload).parameters["run_e2e"].default is (
        listener.run_end_to_end
    )
    assert listener.run_end_to_end(capture_root="/captures/a", provider="openai") == {"status": "ok"}
    assert calls == [{"capture_root": "/captures/a", "provider": "openai"}]


def test_the_runtime_guard_still_proves_the_pipeline_imports() -> None:
    assert "blueprint_pipeline.run_e2e" in CONTROL_PLANE_ENTRYPOINTS
    assert "blueprint_pipeline.pubsub_handoff_listener" in CONTROL_PLANE_ENTRYPOINTS
