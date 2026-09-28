# Covers (for impacted-test selection):
#   src/blueprint_pipeline/wam_provider_output.py
#   src/blueprint_pipeline/vast_structured_policy_canary_inspection.py
#   src/blueprint_pipeline/vast_provider_adapter.py
"""The path-free archive inspections answer exactly what their path wrappers do."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import blueprint_pipeline.vast_provider_adapter as vpa
from blueprint_pipeline.vast_structured_policy_canary_inspection import (
    STRUCTURED_POLICY_CANARY_MEMBER,
    inspect_structured_policy_canary_archive,
)
from blueprint_pipeline.wam_provider_output import (
    inspect_provider_runtime_output_archive,
    inspect_provider_runtime_output_zip,
)

TOP = "native_task_arena_policy_canary_session_result.v1.json"


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def _result(status: str, **extra) -> bytes:
    return json.dumps({"schema_version": TOP.removesuffix(".json"), "status": status,
                       "blockers": [], **extra}).encode()


_ARCHIVES = {
    "top_level_over_nested_cell": {
        "cell_runs/00/" + TOP: _result("runtime_selected_cell_completed_pending_aggregation"),
        TOP: _result("blocked", blockers=["policy_canary_worker_failed_without_result"]),
        "provider_entrypoint_diagnostic.json": json.dumps({"status": "blocked"}).encode(),
        "cell_runs/00/episodes/media/e0/external.mp4": b"video-bytes",
        "cell_runs/00/episodes/media/e0/wrist.mp4": b"more-video",
    },
    "unreadable_top_level_result": {TOP: b"{not json", "cell_runs/00/" + TOP: _result("completed")},
    "source_calibration_identity_mismatch": {
        "adp009d_source_calibration_gpu_render_result.v1.json": json.dumps(
            {"schema_version": "other", "render_scope": "retained_scene"}).encode(),
    },
    "no_result_at_all": {"logs/worker.log": b"step ok\n"},
}


def _probe(path: Path) -> dict[str, object]:
    # Deliberately path-free, so two extraction directories compare equal.
    return {"status": "completed" if path.read_bytes() else "blocked", "frame_count": 4,
            "duration_seconds": 1.0}


@pytest.mark.parametrize("case", sorted(_ARCHIVES))
@pytest.mark.parametrize("videos", [False, True])
def test_archive_inspection_equals_the_path_inspection(tmp_path, case, videos) -> None:
    path = _zip(tmp_path / "vast_provider_runtime_output.zip", _ARCHIVES[case])
    options = {"expected_video_count": 2, "video_probe": _probe}

    by_path = inspect_provider_runtime_output_zip(
        path, video_extract_dir=tmp_path / "path_videos" if videos else None, **options)
    with zipfile.ZipFile(path) as archive:
        by_archive = inspect_provider_runtime_output_archive(
            archive, zip_path=str(path.resolve()), zip_size_bytes=path.stat().st_size,
            video_extract_dir=tmp_path / "archive_videos" if videos else None, **options)

    assert by_archive == by_path
    if videos and case == "top_level_over_nested_cell":
        assert by_path["video_smoke_proven"] is True
        assert sorted(p.name for p in (tmp_path / "path_videos").iterdir()) == sorted(
            p.name for p in (tmp_path / "archive_videos").iterdir())


def test_a_failure_inside_the_archive_is_the_path_inspection_s_blocked_shape(tmp_path) -> None:
    path = _zip(tmp_path / "output.zip", _ARCHIVES["top_level_over_nested_cell"])

    def exploding_probe(_path: Path) -> dict[str, object]:
        raise OSError("probe exploded")

    by_path = inspect_provider_runtime_output_zip(
        path, video_extract_dir=tmp_path / "a", expected_video_count=2, video_probe=exploding_probe)
    with zipfile.ZipFile(path) as archive:
        by_archive = inspect_provider_runtime_output_archive(
            archive, zip_path=str(path.resolve()), zip_size_bytes=path.stat().st_size,
            video_extract_dir=tmp_path / "b", expected_video_count=2, video_probe=exploding_probe)

    assert by_archive == by_path == {
        "status": "blocked", "zip_path": str(path.resolve()), "zip_present": True,
        "zip_size_bytes": path.stat().st_size,
        "blockers": ["provider_runtime_output_zip_invalid:OSError"], "video_smoke_proven": False,
    }
    (tmp_path / "not-a-zip.zip").write_bytes(b"not a zip archive")
    assert inspect_provider_runtime_output_zip(tmp_path / "not-a-zip.zip")["blockers"] == [
        "provider_runtime_output_zip_invalid:BadZipFile"]


def _structured_payload() -> dict[str, object]:
    native_action = [[float(row * 8 + column) for column in range(8)] for row in range(32)]
    receipt = {"native_action_shape": [32, 8], "wam_prefix_action_shape": [16, 8],
               "executed_prefix_steps": 8, **{key: str(position) * 64 for position, key in enumerate((
                   "server_identity_sha256", "observation_sha256", "native_action_sha256",
                   "wam_prefix_action_sha256", "executed_prefix_action_sha256",
                   "commanded_next_state_sha256", "receipt_sha256"), start=1)}}
    return {
        "status": "passed", "native_action": native_action, "wam_prefix_action": native_action[:16],
        "executed_action": native_action[:8], "commanded_next_joint_position": native_action[7][:7],
        "commanded_next_gripper_position": [native_action[7][7]],
        "policy_endpoint_evidence": {"identity_verified": True, "request_count": 1,
                                     "server_metadata": {"policy_id": "model/policy",
                                                         "model_revision": "2" * 40}},
        "policy_request_receipt": receipt,
    }


@pytest.mark.parametrize("case", ["passing", "failing_shape", "member_missing", "member_invalid"])
def test_structured_canary_archive_inspection_equals_the_adapter_wrapper(tmp_path, case) -> None:
    payload = _structured_payload()
    if case == "failing_shape":
        payload["native_action"] = payload["native_action"][:31]
    members = {
        "passing": {STRUCTURED_POLICY_CANARY_MEMBER: json.dumps(payload).encode()},
        "failing_shape": {STRUCTURED_POLICY_CANARY_MEMBER: json.dumps(payload).encode()},
        "member_missing": {"other.json": b"{}"},
        "member_invalid": {STRUCTURED_POLICY_CANARY_MEMBER: b"\xff{not json"},
    }[case]
    path = _zip(tmp_path / "policy-output.zip", members)

    with zipfile.ZipFile(path) as archive:
        by_archive = inspect_structured_policy_canary_archive(archive)

    assert by_archive == vpa._inspect_structured_policy_canary_output(path)
    assert by_archive["status"] == ("passed" if case == "passing" else "blocked")
    assert by_archive["raw_secret_values_recorded"] is False


def test_structured_canary_wrapper_keeps_its_missing_and_unreadable_codes(tmp_path) -> None:
    missing = vpa._inspect_structured_policy_canary_output(tmp_path / "absent.zip")
    assert missing == inspect_structured_policy_canary_archive(None)
    assert missing["blockers"] == ["structured_policy_canary_output_zip_missing"]
    assert vpa._inspect_structured_policy_canary_output(None) == missing

    (tmp_path / "corrupt.zip").write_bytes(b"not a zip archive")
    corrupt = vpa._inspect_structured_policy_canary_output(tmp_path / "corrupt.zip")
    assert corrupt["status"] == "blocked"
    assert corrupt["blockers"] == ["structured_policy_canary_member_invalid"]
