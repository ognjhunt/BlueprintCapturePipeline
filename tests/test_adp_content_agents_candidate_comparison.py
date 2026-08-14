from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_content_agents_candidate_comparison import (
    ContentAgentsCandidateComparisonError,
    materialize_content_agents_candidate_comparison,
    validate_content_agents_candidate_comparison,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


BACKENDS = ("earthtojake_text_to_cad", "pan_chera_multi_agent_cad")


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _file(path: Path, payload: bytes) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    import hashlib

    return "sha256:" + hashlib.sha256(payload).hexdigest(), len(payload)


def _candidate(root: Path, *, slot: int, backend: str, website: bool = False) -> dict:
    item = root / f"slot-{slot}-{backend}"
    bundle_digest, bundle_size = _file(item / "bundle.zip", b"bundle-" + backend.encode())
    receipt = _write(
        item / "bundle.json",
        {
            "schema_version": "adp_content_agents_provider_bundle.v1",
            "status": "ready",
            "blockers": [],
            "input_variant": "agent_cad_v1",
            "bundle_path": str(item / "bundle.zip"),
            "bundle_sha256": bundle_digest,
            "bundle_size_bytes": bundle_size,
            "input_variant_bindings": {
                "replacement_slot": slot,
                "task_id": f"task-{slot}",
                "asset_id": f"asset-{slot}",
                "cad_agent_backend_id": backend,
            },
            "input_usd_normalization": {"mesh_count": slot * 10},
        },
    )
    profile_value = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": f"profile-{slot}-{backend}",
        "allocator": {
            "subcommand": "gpu-canary",
            "argv": ["--probe-kind", "adp-usd-content-agents"],
        },
        "immutable_inputs": [
            {
                "name": "source_bundle_manifest",
                "path": "/host/input/bundle.json",
                "digest": "sha256:" + __import__("hashlib").sha256(receipt.read_bytes()).hexdigest(),
            }
        ],
    }
    profile_value["profile_digest"] = canonical_digest(
        profile_value, digest_field="profile_digest"
    )
    profile = _write(item / "profile.json", profile_value)
    allocator = _write(
        item / "allocator.json",
        {
            "schema_version": "adp_content_agents_vast_run.v1",
            "status": "completed",
            "blockers": [],
            "bundle_sha256": bundle_digest,
            "retry_cap": 0,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "raw_secret_values_recorded": False,
            "estimated_cost_usd": 0.1 * slot,
            "hard_cap_usd": 1.0,
        },
    )
    review_paths = []
    files = []
    for index in range(2):
        frame = item / f"review-{index}.png"
        sha, size = _file(frame, f"frame-{slot}-{backend}-{index}".encode())
        review_paths.append(str(frame))
        files.append(
            {
                "relative_path": f"immutable_execution/texture_workdir/renders/render_0_{index}_0.png",
                "sha256": sha,
                "size_bytes": size,
            }
        )
    files += [
        {
            "relative_path": "immutable_execution/texture_workdir/output/textured_output.usd",
            "sha256": "sha256:" + "1" * 64,
            "size_bytes": 10,
        },
        {
            "relative_path": "immutable_execution/physics_workdir/physics/source_asset_physics.usda",
            "sha256": "sha256:" + "2" * 64,
            "size_bytes": 20,
        },
    ]
    manifest_value = {
        "schema_version": "task_evaluation_artifact_manifest.v1",
        "status": "completed",
        "blockers": [],
        "files": files,
    }
    manifest_value["manifest_digest"] = canonical_digest(
        manifest_value, digest_field="manifest_digest"
    )
    manifest = _write(item / "manifest.json", manifest_value)
    cleanup = _write(
        item / "cleanup.json",
        {
            "schema_version": "wam_provider_object_store_cleanup.v1",
            "status": "completed",
            "blockers": [],
            "all_objects_absent": True,
            "raw_secret_values_recorded": False,
        },
    )
    teardown = _write(
        item / "teardown.json",
        {
            "schema_version": "vast_teardown_manifest.v1",
            "status": "completed",
            "continuing_spend_from_this_run": False,
            "runner_gpu_teardown_completed": True,
            "raw_secret_values_recorded": False,
        },
    )
    zero_value = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "blockers": [],
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "automatic_retry_performed": False,
    }
    if website:
        zero_value.update(
            {
                "launch_id": f"launch-{slot}-{backend}",
                "run_id": f"run-{slot}-{backend}",
                "request_digest": "sha256:" + "3" * 64,
            }
        )
    zero_value["provider_zero_receipt_digest"] = canonical_digest(
        zero_value, digest_field="provider_zero_receipt_digest"
    )
    zero = _write(item / "zero.json", zero_value)
    sync = None
    if website:
        sync_value = {
            "schema_version": "task_evaluation_launch_webapp_sync_result.v1",
            "status": "succeeded",
            "launch_id": zero_value["launch_id"],
            "run_id": zero_value["run_id"],
            "request_digest": zero_value["request_digest"],
            "receipt_digest": "sha256:" + "4" * 64,
            "response": {
                "schema_version": "task_evaluation_launch_web_sync_receipt.v1",
                "status": "completed",
                "launch_id": zero_value["launch_id"],
                "run_id": zero_value["run_id"],
                "request_digest": zero_value["request_digest"],
                "receipt_digest": "sha256:" + "4" * 64,
            },
        }
        sync_value["sync_result_digest"] = canonical_digest(
            sync_value, digest_field="sync_result_digest"
        )
        sync = _write(item / "sync.json", sync_value)
    return {
        "bundle_receipt_path": str(receipt),
        "launch_profile_path": str(profile),
        "allocator_result_path": str(allocator),
        "artifact_manifest_path": str(manifest),
        "object_store_cleanup_path": str(cleanup),
        "teardown_manifest_path": str(teardown),
        "provider_zero_path": str(zero),
        "review_frame_paths": review_paths,
        "webapp_sync_path": str(sync) if sync else None,
    }


def _matrix(tmp_path: Path) -> list[dict]:
    return [
        _candidate(tmp_path, slot=slot, backend=backend, website=slot == 2)
        for slot in (1, 2)
        for backend in BACKENDS
    ]


def test_comparison_reopens_terminal_and_review_evidence(tmp_path: Path) -> None:
    comparison = materialize_content_agents_candidate_comparison(
        candidates=_matrix(tmp_path),
        output_path=tmp_path / "comparison.json",
        generated_at="2026-08-14T00:00:00Z",
    )

    assert comparison["status"] == "completed_candidates_ready_for_within_task_visual_review"
    assert comparison["candidate_count"] == 4
    assert comparison["replacement_slot_count"] == 2
    assert comparison["aggregate_estimated_cost_usd"] == 0.6
    assert [row["cad_agent_backend_id"] for row in comparison["items"]] == [
        "earthtojake_text_to_cad",
        "pan_chera_multi_agent_cad",
        "earthtojake_text_to_cad",
        "pan_chera_multi_agent_cad",
    ]
    assert [row["website_trigger_proven"] for row in comparison["items"]] == [
        False,
        False,
        True,
        True,
    ]
    assert validate_content_agents_candidate_comparison(comparison) == comparison


def test_comparison_rejects_missing_backend(tmp_path: Path) -> None:
    with pytest.raises(
        ContentAgentsCandidateComparisonError, match="comparison_backend_coverage_invalid"
    ):
        materialize_content_agents_candidate_comparison(
            candidates=_matrix(tmp_path)[:-1], output_path=tmp_path / "out.json"
        )


def test_comparison_rejects_tampered_review_frame(tmp_path: Path) -> None:
    candidates = _matrix(tmp_path)
    Path(candidates[0]["review_frame_paths"][0]).write_bytes(b"tampered")
    with pytest.raises(
        ContentAgentsCandidateComparisonError, match="comparison_review_frames_invalid"
    ):
        materialize_content_agents_candidate_comparison(
            candidates=candidates, output_path=tmp_path / "out.json"
        )


def test_comparison_rejects_nonterminal_or_nonzero_candidate(tmp_path: Path) -> None:
    candidates = _matrix(tmp_path)
    allocator_path = Path(candidates[0]["allocator_result_path"])
    allocator = json.loads(allocator_path.read_text())
    allocator["status"] = "blocked"
    _write(allocator_path, allocator)
    with pytest.raises(
        ContentAgentsCandidateComparisonError, match="comparison_allocator_result_invalid"
    ):
        materialize_content_agents_candidate_comparison(
            candidates=candidates, output_path=tmp_path / "out.json"
        )


def test_validator_rejects_source_or_receipt_drift(tmp_path: Path) -> None:
    comparison = materialize_content_agents_candidate_comparison(
        candidates=_matrix(tmp_path), output_path=tmp_path / "comparison.json"
    )
    comparison["items"][0]["mesh_count"] = 999
    comparison["receipt_digest"] = canonical_digest(
        comparison, digest_field="receipt_digest"
    )
    with pytest.raises(
        ContentAgentsCandidateComparisonError, match="comparison_content_invalid"
    ):
        validate_content_agents_candidate_comparison(comparison)


def test_comparison_rejects_profile_that_does_not_bind_bundle_receipt(
    tmp_path: Path,
) -> None:
    candidates = _matrix(tmp_path)
    profile_path = Path(candidates[0]["launch_profile_path"])
    profile = json.loads(profile_path.read_text())
    profile["immutable_inputs"][0]["digest"] = "sha256:" + "f" * 64
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    _write(profile_path, profile)
    with pytest.raises(
        ContentAgentsCandidateComparisonError, match="comparison_launch_profile_invalid"
    ):
        materialize_content_agents_candidate_comparison(
            candidates=candidates, output_path=tmp_path / "out.json"
        )
