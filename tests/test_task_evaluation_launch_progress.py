from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from blueprint_pipeline.task_evaluation_launch_progress import (
    PROGRESS_SCHEMA_VERSION,
    build_launch_progress,
)

NOW = datetime(2026, 8, 11, 16, 0, 0, tzinfo=timezone.utc)
REQUEST = {
    "launch_id": "adp009d-840313-diagnostic-web-1",
    "run_id": "adp009d-840313-diagnostic-run-1",
    "request_digest": "sha256:" + "a" * 64,
}


def _phase_log(run_root: Path, rows: list[dict]) -> None:
    path = run_root / "adp009d-job" / "attempts" / "attempt_001" / "vast_provider_run"
    path.mkdir(parents=True, exist_ok=True)
    (path / "vast_runtime_phase_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_progress_reports_the_latest_phase_and_this_lane_only(tmp_path: Path) -> None:
    """The website showed nothing for the ~25 minutes before a terminal receipt.
    Progress must surface the phase the worker last recorded, and must attribute
    only this lane's instance so a concurrent operator's spend is never shown as
    ours."""
    _phase_log(
        tmp_path,
        [
            {"phase": "vast_instance_create_requested", "status": "completed"},
            {"phase": "vast_heartbeat_started", "status": "running"},
        ],
    )
    guard = {
        "instances": [
            {"name": "blueprint-native-deformable-asset-9", "state": "running",
             "age_seconds": 900, "cost_per_hr_usd": 1.2},
            {"name": "blueprint-adp009d-1786", "state": "running",
             "age_seconds": 600, "cost_per_hr_usd": 0.6},
        ]
    }

    progress = build_launch_progress(
        run_root=tmp_path, request=REQUEST, guard=guard,
        elapsed_seconds=612.5, observed_at=NOW,
    )

    assert progress["schema_version"] == PROGRESS_SCHEMA_VERSION
    assert progress["launch_id"] == REQUEST["launch_id"]
    assert progress["request_digest"] == REQUEST["request_digest"]
    assert progress["phase"] == "vast_heartbeat_started"
    assert progress["phase_status"] == "running"
    assert progress["elapsed_seconds"] == 612.5
    assert progress["observed_at_iso"] == NOW.isoformat()
    # This lane's instance, not the concurrent operator's.
    assert progress["provider"]["instance_age_seconds"] == 600.0
    assert progress["provider"]["estimated_cost_usd"] == round(600 * 0.6 / 3600, 6)

    # Observational only: a progress record must never carry a terminal claim.
    serialized = json.dumps(progress)
    for forbidden in ("receipt_digest", "terminal", "control_passed", "success"):
        assert forbidden not in serialized


def test_progress_degrades_rather_than_raising_without_evidence(tmp_path: Path) -> None:
    """Progress is best effort. A run that has not written a phase log yet, or a
    guard with no instance for this lane, must still produce a usable record."""
    progress = build_launch_progress(
        run_root=tmp_path, request=REQUEST, guard={}, elapsed_seconds=0.0, observed_at=NOW
    )

    assert progress["phase"] == "starting"
    assert progress["phase_status"] == "running"
    assert "provider" not in progress

    _phase_log(tmp_path, [{"not_a_phase": 1}])
    assert build_launch_progress(
        run_root=tmp_path, request=REQUEST, guard=None,
        elapsed_seconds=1.0, observed_at=NOW,
    )["phase"] == "starting"


def test_malformed_phase_lines_are_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "adp009d-job" / "attempts" / "attempt_001" / "vast_provider_run"
    path.mkdir(parents=True, exist_ok=True)
    (path / "vast_runtime_phase_log.jsonl").write_text(
        '{"phase": "vast_offer_selected", "status": "completed"}\nnot json at all\n',
        encoding="utf-8",
    )

    progress = build_launch_progress(
        run_root=tmp_path, request=REQUEST, guard={}, elapsed_seconds=5.0, observed_at=NOW
    )

    assert progress["phase"] == "vast_offer_selected"
    assert progress["phase_status"] == "completed"
