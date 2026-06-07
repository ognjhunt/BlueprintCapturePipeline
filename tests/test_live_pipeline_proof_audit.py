from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.live_pipeline_control_plane import run_live_pipeline_control_plane
from blueprint_pipeline.live_pipeline_proof_audit import (
    LIVE_PIPELINE_PROOF_AUDIT_SCHEMA_VERSION,
    build_live_pipeline_proof_audit,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path, *, with_webapp_ids: bool = True) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    descriptor: dict[str, object] = {"scene_id": "scene-1", "capture_id": "capture-1"}
    if with_webapp_ids:
        descriptor.update(
            {
                "site_submission_id": "site-submission-1",
                "request_id": "request-1",
                "buyer_request_id": "buyer-request-1",
                "capture_job_id": "capture-job-1",
            }
        )
    _write_json(capture_root / "capture_descriptor.json", descriptor)
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def test_live_pipeline_proof_audit_passes_when_external_inputs_are_blocked(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["schema_version"] == LIVE_PIPELINE_PROOF_AUDIT_SCHEMA_VERSION
    assert audit["status"] == "passed_external_inputs_blocked"
    assert audit["internal_blockers"] == []
    assert audit["external_blockers"] == [
        "webapp_upstream_truth",
        "isaac_lab_arena_owner_evidence",
    ]
    assert audit["live_readiness"]["webapp_upstream_truth_ready"] is False
    assert audit["live_readiness"]["owner_arena_evidence_ready"] is False
    assert audit["proof_boundary"]["simulator_execution_proven"] is False
    assert Path(str(audit["output_path"])).is_file()


def test_live_pipeline_proof_audit_can_require_live_ready_inputs(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )

    audit = build_live_pipeline_proof_audit(
        manifest_path=output_path,
        require_live_ready=True,
    )

    assert audit["status"] == "failed_live_ready_required"
    assert "required_live_inputs_missing" in audit["internal_blockers"]


def test_live_pipeline_proof_audit_fails_on_proof_overclaim(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    packet_path = tmp_path / "control" / "live_pipeline_external_input_packet.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["proof_boundary"]["robot_readiness_proven"] = True
    packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    audit = build_live_pipeline_proof_audit(manifest_path=output_path)

    assert audit["status"] == "failed"
    assert "forbidden_proof_boundary_upgrade" in audit["internal_blockers"]
    assert audit["proof_violations"][0]["field"] == "proof_boundary.robot_readiness_proven"


def test_live_pipeline_proof_audit_module_cli(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path, with_webapp_ids=False)
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "inbox",
        load_local_env=False,
        output_path=output_path,
    )
    env = os.environ.copy()
    src_root = Path.cwd() / "src"
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_proof_audit",
            "--manifest-path",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=passed_external_inputs_blocked" in completed.stdout
    audit = json.loads(
        (tmp_path / "control" / "live_pipeline_proof_boundary_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "passed_external_inputs_blocked"
