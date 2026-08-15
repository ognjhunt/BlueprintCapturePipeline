"""The hold trace is only evidence if the worker can import it and the runtime records it."""

from __future__ import annotations

from pathlib import Path

from blueprint_pipeline import adp009d_native_microcheck_bundle as bundle_module
from blueprint_pipeline import adp009d_isaac_runtime as isaac_runtime


def test_bundle_ships_the_hold_trace_module() -> None:
    """The runtime imports it by flat name, so the bundle must carry the file.

    A module wired into the runtime but missing from the bundle dies as a
    ModuleNotFoundError on the worker, after provisioning has already been paid
    for.
    """

    source = Path(bundle_module.__file__).read_text(encoding="utf-8")

    assert '"adp009d_hold_trace.py"' in source


def test_runtime_records_a_per_step_hold_trace() -> None:
    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert "extract_arm_sample(" in source
    assert "extract_arm_effort_limits(" in source
    assert "classify_arm_hold_trace(" in source
    assert "hold_settle_decision(" in source
    assert "max(max_hold_frames, DEFAULT_MAX_HOLD_SETTLE_SAMPLES)" in source
    assert 'if backend == "newton"' in source
    assert "except HoldTraceError" in source


def test_hold_drift_diagnostics_carry_the_trace() -> None:
    """The blocked path is the one that needs the trace most, so it must retain it."""

    source = Path(isaac_runtime.__file__).read_text(encoding="utf-8")

    assert "hold_trace=hold_trace" in source
    assert '"hold_trace"' in source
    assert '"status": "unavailable"' in source
    assert '"camera_warmup_frames": warmup_frames' in source
