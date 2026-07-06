"""TOCTOU: the buyer status projection must re-read consent LIVE at emit time.

The projection derived "revocation required" only from the (possibly stale)
data-package export manifest. A consent revocation that lands after that manifest
is written but before the buyer projection is emitted would ship a clean, buyer-
facing "completed" state. The live re-read closes that window (fail-closed: a
live read can only ADD a revocation, never clear an inherited one).
"""

from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.robot_eval_job_orchestrator import (
    _capture_root_from_job_dir,
    _webapp_robot_eval_status_projection,
)


def _job_dir(tmp_path: Path, *, consent_status: str) -> Path:
    capture_root = tmp_path / "scenes" / "s" / "captures" / "c"
    job_dir = capture_root / "pipeline" / "robot_eval_jobs" / "job1"
    job_dir.mkdir(parents=True)
    (capture_root / "raw").mkdir(parents=True)
    (capture_root / "raw" / "rights_consent.json").write_text(
        json.dumps(
            {
                "consent_status": consent_status,
                **(
                    {"consent_revoked": True, "consent_revoked_at": "2026-07-04T00:00:00Z"}
                    if consent_status == "revoked"
                    else {}
                ),
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def _project(job_dir: Path):
    return _webapp_robot_eval_status_projection(
        job_dir=job_dir,
        job_id="job1",
        scene_id="s",
        capture_id="c",
        status="completed",
        blockers=[],
        request={},
        scenario_eval_matrix={},
        simulator_result={},
        copied_artifacts={},
        robot_pov_manifest={},
        policy_manifest={},
        policy_execution_manifest={},
        evaluation_result={},
        proof_boundary={},
        live_closure={},
        data_package_export={},  # CLEAN inherited manifest — says nothing about revocation
        generated_at="2026-07-06T00:00:00Z",
    )


def test_capture_root_from_job_dir_finds_capture(tmp_path):
    assert _capture_root_from_job_dir(Path("/nonexistent/x/y")) is None
    job_dir = _job_dir(tmp_path, consent_status="documented")
    found = _capture_root_from_job_dir(job_dir)
    assert found == tmp_path / "scenes" / "s" / "captures" / "c"


def test_projection_blocks_on_live_revocation_despite_clean_manifest(tmp_path):
    job_dir = _job_dir(tmp_path, consent_status="revoked")
    proj = _project(job_dir)
    assert proj["state"] == "blocked"
    assert proj["buyer_display_state"] == "blocked_consent_revoked_takedown_required"


def test_projection_not_blocked_when_consent_documented(tmp_path):
    job_dir = _job_dir(tmp_path, consent_status="documented")
    proj = _project(job_dir)
    assert proj["buyer_display_state"] != "blocked_consent_revoked_takedown_required"
