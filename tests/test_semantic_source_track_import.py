"""Contract tests for compact, source-bound persistent mask-track imports."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement.semantic_gaussian_lifting import (
    canonical_json_digest,
)
from blueprint_pipeline.scene_placement.semantic_source_track_import import (
    MASK_ENCODING,
    PROVIDER_RESULT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    import_semantic_source_tracks,
)
from blueprint_pipeline.semantic_source_track_stage import (
    main as semantic_source_track_stage_main,
    run_semantic_source_track_stage,
    run_semantic_source_track_terminal_reimport,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64


def _fixture() -> tuple[dict, dict]:
    frames = [
        {
            "source_frame_id": f"frame-{index}",
            "source_frame_digest": "sha256:" + str(index) * 64,
            "retained_video_digest": SHA_B,
            "decoded_pts_seconds": float(index),
            "sync_map_row_digest": SHA_C,
            "camera_record_digest": "sha256:" + str(index + 2) * 64,
            "encoder_retained": True,
            "width": 4,
            "height": 2,
        }
        for index in (1, 2)
    ]
    profile = {
        "method_id": "blueprint.local.sam_track_import_fixture",
        "method_version": "1.0.0",
        "runtime_digest": SHA_C,
        "model_digest": SHA_D,
        "persistent_track_ids": True,
        "mask_encoding": MASK_ENCODING,
        "model_self_grading_forbidden": True,
        "execution_mode": "local",
        "customer_data_training_allowed": False,
    }
    profile["profile_digest"] = canonical_json_digest(profile)
    provider = {
        "schema_version": PROVIDER_RESULT_SCHEMA_VERSION,
        "bindings": {
            "capture_digest": SHA_A,
            "retained_video_digest": SHA_B,
            "camera_solution_digest": SHA_C,
            "frame_registry_digest": canonical_json_digest(frames),
        },
        "profile_digest": profile["profile_digest"],
        "model_digest": SHA_D,
        "runtime_digest": SHA_C,
        "tracks": [
            {
                "track_id": "track-chair-1",
                "label": "chair",
                "label_source": "model_inferred",
                "observations": [
                    {
                        "source_frame_id": frame["source_frame_id"],
                        "source_frame_digest": frame["source_frame_digest"],
                        "decoded_pts_seconds": frame["decoded_pts_seconds"],
                        "camera_record_digest": frame["camera_record_digest"],
                        "width": 4,
                        "height": 2,
                        "mask_encoding": MASK_ENCODING,
                        "runs": [
                            {"start": index, "length": 2, "probability": 0.95}
                        ],
                    }
                    for index, frame in enumerate(frames)
                ],
            }
        ],
    }
    provider["result_digest"] = canonical_json_digest(provider)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "bindings": {
            "capture_digest": SHA_A,
            "retained_video_digest": SHA_B,
            "camera_solution_digest": SHA_C,
            "frame_registry_digest": canonical_json_digest(frames),
            "provider_result_digest": provider["result_digest"],
        },
        "frame_registry": frames,
        "provider_profile": profile,
        "allowed_evidence_uses": ["semantic_analysis"],
    }
    return request, provider


def _rebind_provider(request: dict, provider: dict) -> None:
    provider["result_digest"] = canonical_json_digest(
        {key: value for key, value in provider.items() if key != "result_digest"}
    )
    request["bindings"]["provider_result_digest"] = provider["result_digest"]


def _stage_paths(tmp_path: Path, request: dict, provider: dict) -> dict[str, Path]:
    provider_path = tmp_path / "provider-result.json"
    provider_path.write_text(json.dumps(provider), encoding="utf-8")
    request["input_artifacts"] = {
        "provider_result": {
            "sha256": "sha256:" + hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            "size_bytes": provider_path.stat().st_size,
        }
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return {
        "request": request_path,
        "provider": provider_path,
        "output": tmp_path / "result.json",
    }


def test_imports_compact_tracks_bound_to_retained_frames_pts_and_cameras() -> None:
    request, provider = _fixture()
    result = import_semantic_source_tracks(request, provider)

    assert result["status"] == "completed"
    assert result["claim_ceiling"] == "source_bound_2d_mask_tracks_only"
    assert result["track_registry"] == [
        {
            "track_id": "track-chair-1",
            "label": "chair",
            "label_source": "model_inferred",
            "mask_model_digest": SHA_D,
            "track_evidence_digest": result["track_registry"][0][
                "track_evidence_digest"
            ],
            "supporting_frame_ids": ["frame-1", "frame-2"],
            "observation_count": 2,
            "semantic_authority": "inferred_candidate",
        }
    ]
    assert len(result["frame_masks"]) == 2
    assert result["frame_masks"][0]["mask_encoding"] == MASK_ENCODING
    assert result["frame_masks"][0]["track_masks"][0]["runs"] == [
        {"start": 0, "length": 2, "probability": 0.95}
    ]
    assert result["bindings"]["track_registry_digest"] == canonical_json_digest(
        result["track_registry"]
    )
    assert result["bindings"]["frame_masks_digest"] == canonical_json_digest(
        result["frame_masks"]
    )
    assert result["directly_observed_object_fact"] is False
    assert result["canonical_object_geometry"] is False
    assert result["metric_box_ready"] is False
    assert result["collision_ready"] is False
    assert result["physics_ready"] is False
    assert result["physical_task_success_established"] is False
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"


def test_rejects_stale_pts_camera_and_nonretained_frame_bindings() -> None:
    request, provider = _fixture()
    provider["tracks"][0]["observations"][0]["decoded_pts_seconds"] = 7.0
    provider["tracks"][0]["observations"][1]["camera_record_digest"] = SHA_A
    request["frame_registry"][0]["encoder_retained"] = False
    request["bindings"]["frame_registry_digest"] = canonical_json_digest(
        request["frame_registry"]
    )
    provider["bindings"]["frame_registry_digest"] = request["bindings"][
        "frame_registry_digest"
    ]
    _rebind_provider(request, provider)

    result = import_semantic_source_tracks(request, provider)

    assert result["status"] == "blocked"
    assert "frame_registry_encoder_retention_not_proven:frame-1" in result["blockers"]
    assert "track_observation_pts_mismatch:track-chair-1:frame-1" in result["blockers"]
    assert (
        "track_observation_camera_digest_mismatch:track-chair-1:frame-2"
        in result["blockers"]
    )


def test_rejects_overlapping_or_out_of_bounds_probability_runs() -> None:
    request, provider = _fixture()
    provider["tracks"][0]["observations"][0]["runs"] = [
        {"start": 2, "length": 3, "probability": 0.8},
        {"start": 4, "length": 5, "probability": 1.2},
    ]
    _rebind_provider(request, provider)

    result = import_semantic_source_tracks(request, provider)

    assert result["status"] == "blocked"
    assert (
        "mask_run_bounds_or_probability_invalid:track-chair-1:frame-1"
        in result["blockers"]
    )


def test_provider_profile_requires_persistent_ids_use_permission_and_no_self_grading() -> None:
    request, provider = _fixture()
    request["provider_profile"]["persistent_track_ids"] = False
    request["provider_profile"]["model_self_grading_forbidden"] = False
    request["provider_profile"]["profile_digest"] = canonical_json_digest(
        {
            key: value
            for key, value in request["provider_profile"].items()
            if key != "profile_digest"
        }
    )
    provider["profile_digest"] = request["provider_profile"]["profile_digest"]
    request["allowed_evidence_uses"] = []
    _rebind_provider(request, provider)

    result = import_semantic_source_tracks(request, provider)

    assert result["status"] == "blocked"
    assert "provider_profile_persistent_track_ids_required" in result["blockers"]
    assert "provider_profile_self_grading_boundary_missing" in result["blockers"]
    assert "semantic_analysis_use_not_permitted" in result["blockers"]


def test_empty_provider_result_abstains_without_inventing_objects() -> None:
    request, provider = _fixture()
    provider["tracks"] = []
    _rebind_provider(request, provider)

    result = import_semantic_source_tracks(request, provider)

    assert result["status"] == "abstained"
    assert result["track_registry"] == []
    assert len(result["frame_masks"]) == 2
    assert all(row["track_masks"] == [] for row in result["frame_masks"])
    assert all(
        row["mask_artifact_digest"] == canonical_json_digest([])
        for row in result["frame_masks"]
    )
    assert result["claim_ceiling"] == "no_source_tracks_detected"
    assert "provider_returned_no_tracks" in result["warnings"]
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"


def test_retains_valid_source_frame_with_no_above_threshold_track_mask() -> None:
    request, provider = _fixture()
    provider["tracks"][0]["observations"] = provider["tracks"][0]["observations"][:1]
    _rebind_provider(request, provider)

    result = import_semantic_source_tracks(request, provider)

    assert result["status"] == "completed"
    assert [row["source_frame_id"] for row in result["frame_masks"]] == [
        "frame-1",
        "frame-2",
    ]
    assert result["frame_masks"][0]["track_masks"]
    assert result["frame_masks"][1]["track_masks"] == []
    assert result["frame_masks"][1]["mask_artifact_digest"] == canonical_json_digest(
        []
    )


def test_track_and_observation_order_does_not_change_normalized_registries() -> None:
    request, provider = _fixture()
    second = copy.deepcopy(provider["tracks"][0])
    second["track_id"] = "track-table-1"
    second["label"] = "table"
    provider["tracks"].append(second)
    _rebind_provider(request, provider)
    first_result = import_semantic_source_tracks(request, provider)

    reversed_request = copy.deepcopy(request)
    reversed_provider = copy.deepcopy(provider)
    reversed_provider["tracks"].reverse()
    for track in reversed_provider["tracks"]:
        track["observations"].reverse()
    _rebind_provider(reversed_request, reversed_provider)
    second_result = import_semantic_source_tracks(
        reversed_request, reversed_provider
    )

    assert first_result["track_registry"] == second_result["track_registry"]
    assert first_result["frame_masks"] == second_result["frame_masks"]
    assert first_result["bindings"]["track_registry_digest"] == second_result[
        "bindings"
    ]["track_registry_digest"]
    assert first_result["bindings"]["frame_masks_digest"] == second_result[
        "bindings"
    ]["frame_masks_digest"]


def test_file_stage_hash_verifies_provider_and_writes_compact_result(tmp_path) -> None:
    request, provider = _fixture()
    paths = _stage_paths(tmp_path, request, provider)

    result = run_semantic_source_track_stage(
        request_path=paths["request"],
        provider_result_path=paths["provider"],
        output_path=paths["output"],
    )

    assert result["status"] == "completed"
    assert result["transport_profile"] == "bounded_compact_probability_rle.v1"
    assert result["stage_input_artifacts"]["provider_result"]["sha256"] == (
        request["input_artifacts"]["provider_result"]["sha256"]
    )
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result


def test_file_stage_tampering_returns_terminal_blocked_artifact(tmp_path) -> None:
    request, provider = _fixture()
    paths = _stage_paths(tmp_path, request, provider)
    paths["provider"].write_text(json.dumps({**provider, "tracks": []}), encoding="utf-8")

    result = run_semantic_source_track_stage(
        request_path=paths["request"],
        provider_result_path=paths["provider"],
        output_path=paths["output"],
    )

    assert result["status"] == "blocked"
    assert "input_artifact_size_mismatch:provider_result" in result["blockers"]
    assert "input_artifact_sha256_mismatch:provider_result" in result["blockers"]
    assert result["physics_ready"] is False
    assert result["comparative_policy_ranking_verdict"] == "thesis_not_supported"


def test_file_stage_refuses_input_overwrite_and_symlink_output(tmp_path) -> None:
    request, provider = _fixture()
    paths = _stage_paths(tmp_path, request, provider)
    with pytest.raises(ValueError, match="output_path_must_not_overwrite_an_input"):
        run_semantic_source_track_stage(
            request_path=paths["request"],
            provider_result_path=paths["provider"],
            output_path=paths["provider"],
        )

    real_output = tmp_path / "real-output.json"
    real_output.write_text("{}", encoding="utf-8")
    symlink_output = tmp_path / "output-link.json"
    symlink_output.symlink_to(real_output)
    with pytest.raises(ValueError, match="output_symlink_forbidden"):
        run_semantic_source_track_stage(
            request_path=paths["request"],
            provider_result_path=paths["provider"],
            output_path=symlink_output,
        )


def _terminal_runtime_result_with_legacy_missing_empty_frame() -> dict:
    request, provider = _fixture()
    provider["tracks"][0]["observations"] = provider["tracks"][0]["observations"][:1]
    _rebind_provider(request, provider)
    normalized = import_semantic_source_tracks(request, provider)
    normalized["frame_masks"] = normalized["frame_masks"][:1]
    normalized["bindings"]["frame_masks_digest"] = canonical_json_digest(
        normalized["frame_masks"]
    )
    normalized["result_digest"] = canonical_json_digest(
        {key: value for key, value in normalized.items() if key != "result_digest"}
    )
    runtime = {
        "schema_version": "semantic_sam31_vast_source_track_result.v1",
        "status": "passed",
        "source_track_import_request": request,
        "provider_result": provider,
        "normalized_source_tracks": normalized,
        "blockers": [],
        "raw_secret_values_recorded": False,
    }
    runtime["runtime_result_digest"] = canonical_digest(
        runtime, digest_field="runtime_result_digest"
    )
    return runtime


def test_terminal_reimport_cli_restores_valid_empty_frame_and_retains_receipt(
    tmp_path,
) -> None:
    runtime_path = tmp_path / "provider-runtime-result.json"
    runtime = _terminal_runtime_result_with_legacy_missing_empty_frame()
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    output_path = tmp_path / "normalized-current.json"
    receipt_path = tmp_path / "terminal-reimport-receipt.json"

    assert (
        semantic_source_track_stage_main(
            [
                "--terminal-runtime-result",
                str(runtime_path),
                "--source-commit-sha",
                "a" * 40,
                "--output",
                str(output_path),
                "--receipt-output",
                str(receipt_path),
            ]
        )
        == 0
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert [row["source_frame_id"] for row in result["frame_masks"]] == [
        "frame-1",
        "frame-2",
    ]
    assert result["frame_masks"][1]["track_masks"] == []
    assert result["frame_masks"][1]["mask_artifact_digest"] == canonical_json_digest(
        []
    )
    assert result["terminal_reimport"]["paid_resource_allocated"] is False
    assert receipt["status"] == "ready"
    assert receipt["normalized_result"]["frame_count"] == 2
    assert receipt["normalized_result"]["result_digest"] == result["result_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_terminal_reimport_rejects_tampered_runtime_receipt(tmp_path) -> None:
    runtime_path = tmp_path / "provider-runtime-result.json"
    runtime = _terminal_runtime_result_with_legacy_missing_empty_frame()
    runtime["status"] = "failed"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal_reimport_runtime_result_invalid"):
        run_semantic_source_track_terminal_reimport(
            terminal_runtime_result_path=runtime_path,
            source_commit_sha="a" * 40,
            output_path=tmp_path / "normalized-current.json",
            receipt_output_path=tmp_path / "terminal-reimport-receipt.json",
        )
