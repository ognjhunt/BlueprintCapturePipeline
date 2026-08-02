from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.measurement_adapter_execution import (
    _safe_environment,
    _subprocess_failure_codes,
)
from blueprint_pipeline.measurement_dlo_lab_cable_adapter import (
    _classified_native_failure,
)


def test_native_worker_failure_is_categorized_without_persisting_content(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text("OMP: Error libgomp terminate called\n", encoding="utf-8")
    assert _subprocess_failure_codes(stderr, -6) == [
        "worker_exit_nonzero:-6",
        "worker_signal:6",
        "worker_stderr_native_termination",
        "worker_stderr_openmp_runtime_failure",
    ]


def test_safe_environment_synthesizes_isolated_home_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", "/operator-home-must-not-leak")
    monkeypatch.setenv("XDG_CACHE_HOME", "/operator-cache-must-not-leak")

    environment = _safe_environment(tmp_path)

    assert environment["HOME"] == str(tmp_path)
    assert environment["XDG_CACHE_HOME"] == str(tmp_path / ".cache")
    assert environment["TMPDIR"] == str(tmp_path)


def test_dlo_supervisor_classifies_abort_without_returning_stderr() -> None:
    assert _classified_native_failure(
        b"qt.qpa: could not load the Qt platform plugin xcb; terminate called\n",
        -6,
    ) == [
        "dlo_lab_adapter_native_termination",
        "dlo_lab_adapter_qt_platform_failure",
        "dlo_lab_adapter_supervised_worker_exit_nonzero:-6",
        "dlo_lab_adapter_supervised_worker_signal:6",
    ]


def test_dlo_supervisor_classifies_native_system_error_without_returning_stderr() -> None:
    assert _classified_native_failure(
        b"terminate called after throwing std::system_error\nwhat(): Invalid argument\n",
        -6,
    ) == [
        "dlo_lab_adapter_native_invalid_argument",
        "dlo_lab_adapter_native_system_error",
        "dlo_lab_adapter_native_termination",
        "dlo_lab_adapter_supervised_worker_exit_nonzero:-6",
        "dlo_lab_adapter_supervised_worker_signal:6",
    ]
