from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.live_pipeline_control_plane import run_live_pipeline_control_plane
from blueprint_pipeline.live_pipeline_input_intake import (
    LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION,
    build_live_pipeline_input_intake,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(capture_root / "raw" / "manifest.json", {"scene_id": "scene-1"})
    return capture_root


def _control_manifest(tmp_path: Path, capture_root: Path) -> Path:
    output_path = tmp_path / "control" / "live_pipeline_control_plane_manifest.json"
    run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=output_path,
    )
    return output_path


def _webapp_request(capture_root: Path, *, job_id: str = "webapp-job-1") -> dict[str, object]:
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "site_package": {
                "capture_root": str(capture_root),
                "site_id": "site-1",
                "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
            },
            "source": {
                "system": "Blueprint-WebApp",
                "site_submission_id": "site-submission-1",
                "request_id": "request-1",
                "buyer_request_id": "buyer-request-1",
                "capture_job_id": "capture-job-1",
            },
        },
    }


def _webapp_site_library_request(
    capture_root: Path, *, job_id: str = "webapp-job-1"
) -> dict[str, object]:
    buyer_request_id = "buyer-request-1"
    return {
        "queue_contract": "robot_eval_job_request_inbox.v1",
        "status": "queued_for_pipeline",
        "job_id": job_id,
        "job_request": {
            "schema_version": "robot_eval_job_request.v1",
            "job_id": job_id,
            "buyer_request_id": buyer_request_id,
            "site_package": {
                "capture_root": str(capture_root),
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
                "buyer_request_id": buyer_request_id,
                "package_uri": "gs://local-blueprint/scenes/scene-1/captures/capture-1/pipeline",
            },
            "owner_system": {
                "name": "Blueprint-WebApp",
                "request_id": job_id,
                "buyer_request_id": buyer_request_id,
                "site_submission_id": "site-submission-1",
                "capture_job_id": "capture-job-1",
            },
            "source": {
                "system": "Blueprint-WebApp",
                "selection_state": {
                    "buyer_request_id": buyer_request_id,
                    "site_submission_id": "site-submission-1",
                    "capture_job_id": "capture-job-1",
                },
            },
        },
    }


def _arena_results(results_dir: Path) -> Path:
    _write_json(
        results_dir / "rollout_manifest.json",
        {
            "episodes": [
                {
                    "episode_id": "episode-1",
                    "scenario_id": "scenario-1",
                    "status": "success",
                    "success": True,
                }
            ]
        },
    )
    return results_dir


def test_live_pipeline_input_intake_validates_inputs_without_staging(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    results_dir = _arena_results(tmp_path / "arena-results")
    _write_json(request_path, _webapp_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        arena_results_dir=results_dir,
    )

    assert result["schema_version"] == LIVE_PIPELINE_INPUT_INTAKE_SCHEMA_VERSION
    assert result["status"] == "ready_for_control_plane"
    assert result["webapp_job_request"]["status"] == "ready"
    assert result["arena_results"]["status"] == "ready_for_ingest"
    assert result["webapp_staging"]["status"] == "not_requested"
    assert result["proof_boundary"]["simulator_execution_proven"] is False
    assert Path(str(result["output_path"])).is_file()


def test_live_pipeline_input_intake_stages_valid_webapp_request(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    target_path = Path(str(result["webapp_staging"]["target_path"]))
    assert result["status"] == "staged_for_control_plane"
    assert result["webapp_staging"]["performed"] is True
    assert result["staged_inputs"]["status"] == "staged"
    assert Path(str(result["staged_inputs"]["path"])).is_file()
    assert target_path.is_file()
    assert json.loads(target_path.read_text(encoding="utf-8"))["job_id"] == "webapp-job-1"


def test_live_pipeline_input_intake_accepts_webapp_site_library_id_locations(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_site_library_request(capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    assert result["status"] == "staged_for_control_plane"
    assert result["webapp_job_request"]["missing_fields"] == []
    assert result["webapp_job_request"]["fields_present"] == {
        "site_submission_id": True,
        "request_id": True,
        "buyer_request_id": True,
        "capture_job_id": True,
    }


def test_live_pipeline_input_intake_staged_arena_results_feed_control_plane(
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    results_dir = _arena_results(tmp_path / "arena-results")

    intake = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        arena_results_dir=results_dir,
        stage_arena_results=True,
    )
    rerun = run_live_pipeline_control_plane(
        capture_root=capture_root,
        job_request_inbox=tmp_path / "webapp-inbox",
        load_local_env=False,
        output_path=manifest_path,
    )
    packet = json.loads(
        Path(rerun["external_input_packet"]["path"]).read_text(encoding="utf-8")
    )
    required_input_ids = {item["id"] for item in packet["required_inputs"]}

    assert intake["status"] == "staged_for_control_plane"
    assert intake["staged_inputs"]["arena_results_staged"] is True
    assert rerun["staged_inputs"]["arena_results_ready"] is True
    assert rerun["setup_status"] == "local_ready_live_external_blocked"
    assert required_input_ids == {"webapp_upstream_truth"}
    assert "Isaac Lab-Arena" not in " ".join(rerun["next_inputs_needed"])


def test_live_pipeline_input_intake_rejects_mismatched_capture_root(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    other_capture_root = tmp_path / "other" / "captures" / "capture-2"
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_request(other_capture_root))

    result = build_live_pipeline_input_intake(
        manifest_path=manifest_path,
        webapp_job_request=request_path,
        stage_webapp_request=True,
    )

    assert result["status"] == "blocked"
    assert "webapp:request_capture_root_does_not_match_control_plane" in result[
        "input_blockers"
    ]
    assert result["webapp_staging"]["performed"] is False


def test_live_pipeline_input_intake_module_cli(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    manifest_path = _control_manifest(tmp_path, capture_root)
    request_path = tmp_path / "incoming" / "webapp-job-1.json"
    _write_json(request_path, _webapp_request(capture_root))
    env = os.environ.copy()
    src_root = Path.cwd() / "src"
    env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(src_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "blueprint_pipeline.live_pipeline_input_intake",
            "--manifest-path",
            str(manifest_path),
            "--webapp-job-request",
            str(request_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert "status=ready_for_control_plane" in completed.stdout
    audit = json.loads(
        (tmp_path / "control" / "live_pipeline_input_intake_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "ready_for_control_plane"
