from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.retained_gpu_session_lifecycle import (
    JOURNAL_NAME,
    STATES,
    record_retained_gpu_state,
)


def test_retained_gpu_lifecycle_records_required_state_path(tmp_path: Path) -> None:
    path = (
        "allocated",
        "container_starting",
        "image_pulling",
        "model_downloading",
        "model_loading",
        "healthy",
        "retained_owned",
        "refresh_in_progress",
        "experiment_running",
        "terminal_success",
        "teardown_requested",
        "provider_absent",
    )

    for state in path:
        manifest = record_retained_gpu_state(tmp_path, state, evidence={"state": state})

    assert set(path) == set(STATES) - {"terminal_failure"}
    assert manifest["state"] == "provider_absent"
    assert manifest["terminal"] is True
    rows = [json.loads(line) for line in (tmp_path / JOURNAL_NAME).read_text().splitlines()]
    assert len(rows) == len(path)
    assert rows[0]["previous_record_sha256"] == "0" * 64
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_record_sha256"] == previous["record_sha256"]


def test_retained_gpu_lifecycle_rejects_invalid_transition(tmp_path: Path) -> None:
    record_retained_gpu_state(tmp_path, "allocated")

    with pytest.raises(ValueError, match="invalid_retained_gpu_transition"):
        record_retained_gpu_state(tmp_path, "provider_absent")


def test_retained_gpu_lifecycle_rejects_unknown_state(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported_retained_gpu_state"):
        record_retained_gpu_state(tmp_path, "episode_dispatched_continuing_spend")
