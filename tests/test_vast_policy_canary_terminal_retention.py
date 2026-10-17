"""Execute the generated output archiver at the observed V25 aggregate size."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import blueprint_pipeline.vast_provider_adapter as adapter
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


pytestmark = pytest.mark.slow
RESULT_NAME = "native_task_arena_policy_canary_session_result.v1.json"
OBSERVED_V25_BYTES = 252_230_383


def _archive_script(kind: str, *, cells: int = 10) -> str:
    shell = adapter._probe_shell_script(
        "https://heartbeat.invalid",
        enable_blueprint_bundle=True,
        provider_bundle_kind=kind,
        policy_cell_count=cells,
    )
    scripts = re.findall(r"\$RUNTIME_PY - <<'PY'\n(.*?)\nPY\n", shell, re.S)
    return next(script for script in scripts if "output_zip = work_dir / 'adp_arena_provider_runtime_output.zip'" in script)


def _archive(tmp_path: Path, kind: str = "native_task_arena_policy_canary_session", *, cells: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", _archive_script(kind, cells=cells)],
        env={
            **os.environ,
            "BLUEPRINT_ADP_ARENA_OUTPUT_DIR": str(tmp_path / "runtime_output"),
            "BLUEPRINT_VAST_WORK_DIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_generated_archive_retains_observed_large_terminal_result_and_reader_uses_it(tmp_path: Path) -> None:
    output = tmp_path / "runtime_output"
    output.mkdir()
    terminal = output / RESULT_NAME
    # Valid JSON plus whitespace reproduces the exact observed size without
    # retaining a large fixture or building the entire payload in memory.
    header = b'{"status":"blocked","blockers":["retained_terminal_failure"]}'
    with terminal.open("wb") as destination:
        destination.write(header)
        remaining = OBSERVED_V25_BYTES - len(header)
        block = b" " * 1024**2
        while remaining:
            count = min(remaining, len(block))
            destination.write(block[:count])
            remaining -= count
    child = output / "cell_runs/00" / RESULT_NAME
    child.parent.mkdir(parents=True)
    child.write_text('{"status":"completed"}')
    with (output / "unrelated-large.bin").open("wb") as destination:
        destination.truncate(100_000_001)

    completed = _archive(tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN" in completed.stdout
    archive_path = tmp_path / "adp_arena_provider_runtime_output.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.getinfo(RESULT_NAME).file_size == OBSERVED_V25_BYTES
        assert "unrelated-large.bin" not in archive.namelist()
        with archive.open(RESULT_NAME) as archived, terminal.open("rb") as original:
            assert hashlib.file_digest(archived, "sha256").digest() == hashlib.file_digest(original, "sha256").digest()
    summary = inspect_provider_runtime_output_zip(archive_path)
    assert summary["runtime_result_member"] == RESULT_NAME
    assert summary["runtime_result_status"] == "blocked"
    assert summary["runtime_result_blockers"] == ["retained_terminal_failure"]


@pytest.mark.parametrize("size", [None, 0, 512 * 1024**2 + 1])
def test_generated_archive_refuses_missing_empty_or_oversize_required_result(tmp_path: Path, size: int | None) -> None:
    output = tmp_path / "runtime_output"
    output.mkdir()
    if size is not None:
        with (output / RESULT_NAME).open("wb") as destination:
            destination.truncate(size)
    completed = _archive(tmp_path)
    assert completed.returncode != 0
    assert "policy_canary_required_terminal_result_" in completed.stderr
    assert "BLUEPRINT_VAST_PROVIDER_OUTPUT_ZIP_WRITTEN" not in completed.stdout
    assert not (tmp_path / "adp_arena_provider_runtime_output.zip").exists()


def test_generic_arena_keeps_existing_limit_without_requiring_canary_result(tmp_path: Path) -> None:
    output = tmp_path / "runtime_output"
    output.mkdir()
    (output / "diagnostic.json").write_text('{"status":"blocked"}')
    with (output / RESULT_NAME).open("wb") as destination:
        destination.truncate(100_000_001)
    completed = _archive(tmp_path, "native_task_arena")
    assert completed.returncode == 0, completed.stderr
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert archive.namelist() == ["diagnostic.json"]


def test_full_session_archiver_retains_the_scaled_observed_terminal_size(tmp_path):
    output = tmp_path / "runtime_output"
    output.mkdir()
    terminal = output / RESULT_NAME
    size = OBSERVED_V25_BYTES * 10  # 200 rows at the observed twenty-row size.
    # Sparse bytes exercise the retention boundary without allocating 2.5 GB
    # of disk. JSON/schema validity is covered independently by the worker and
    # the observed-size valid-JSON reader regression above.
    with terminal.open("wb") as stream:
        stream.truncate(size)
    assert terminal.stat().st_blocks * 512 < 1024**2
    completed = _archive(tmp_path, cells=200)
    assert completed.returncode == 0, completed.stderr
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert archive.getinfo(RESULT_NAME).file_size == size
        with archive.open(RESULT_NAME) as archived, terminal.open("rb") as original:
            assert hashlib.file_digest(archived, "sha256").digest() == hashlib.file_digest(original, "sha256").digest()


def test_full_session_archiver_refuses_above_its_scaled_bound(tmp_path):
    output = tmp_path / "runtime_output"
    output.mkdir()
    with (output / RESULT_NAME).open("wb") as stream:
        stream.truncate(5 * 1024**3 + 1)
    completed = _archive(tmp_path, cells=200)
    assert completed.returncode != 0
    assert "policy_canary_required_terminal_result_size_invalid" in completed.stderr
    assert not (tmp_path / "adp_arena_provider_runtime_output.zip").exists()


@pytest.mark.parametrize("count", [0, 201, True, "200"])
def test_terminal_size_allowance_refuses_unbounded_cell_counts(count):
    with pytest.raises(ValueError, match="policy_canary_terminal_cell_count_invalid"):
        _archive_script("native_task_arena_policy_canary_session", cells=count)
