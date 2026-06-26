from __future__ import annotations

import json
import struct
from hashlib import sha256
from pathlib import Path

import pytest

from blueprint_pipeline.marble_sim_assets import build_marble_sim_assets
from blueprint_pipeline import production_handoff_readiness as readiness
from blueprint_pipeline.production_handoff_readiness import build_production_handoff_readiness
from blueprint_pipeline.simulation_automation import build_simulation_automation


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _uri(path: str) -> str:
    return f"gs://local-blueprint/scenes/scene-1/captures/capture-1/{path}"


def _capture_root(tmp_path: Path) -> Path:
    root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(root / "capture_descriptor.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    _write_json(
        root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1", "site_type": "warehouse"},
    )
    return root


def _write_minimal_glb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "worldlabs_collider", "mesh": 0}],
        "meshes": [{"name": "scene_collider", "primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [
            {
                "type": "VEC3",
                "componentType": 5126,
                "count": 8,
                "min": [-1.0, -1.0, 0.0],
                "max": [3.0, 3.0, 1.5],
            }
        ],
    }
    raw_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    raw_json += b" " * ((4 - (len(raw_json) % 4)) % 4)
    total_length = 12 + 8 + len(raw_json)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<II", len(raw_json), 0x4E4F534A)
        + raw_json
    )


def _write_privacy_and_worldlabs(root: Path) -> None:
    (root / "privacy").mkdir(parents=True, exist_ok=True)
    (root / "privacy" / "final_walkthrough.mov").write_bytes(b"privacy-safe-video")
    (root / "pipeline" / "worldlabs_input").mkdir(parents=True, exist_ok=True)
    (root / "pipeline" / "worldlabs_input" / "worldlabs_input.mp4").write_bytes(
        b"prepared-worldlabs-input"
    )
    source_checksum = sha256(b"privacy-safe-video").hexdigest()
    output_checksum = sha256(b"prepared-worldlabs-input").hexdigest()
    selected_uri = _uri("pipeline/worldlabs_input/worldlabs_input.mp4")
    collider_path = root / "pipeline" / "worldlabs_assets" / "worldlabs_collider.glb"
    _write_minimal_glb(collider_path)

    _write_json(
        root / "pipeline" / "privacy_processing_manifest.json",
        {
            "schema_version": "v1",
            "status": "person_removed",
            "fail_closed": True,
            "privacy_processed_video_uri": _uri("privacy/final_walkthrough.mov"),
            "world_model_video_uri": _uri("privacy/final_walkthrough.mov"),
            "depth_source": "depth_anything",
        },
    )
    _write_json(root / "pipeline" / "privacy_verification_report.json", {"status": "passed"})
    _write_json(
        root / "pipeline" / "worldlabs_input" / "worldlabs_input_manifest.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "selected_video_source_id": "privacy_processed_video_uri",
            "selected_video_uri": _uri("privacy/final_walkthrough.mov"),
            "output_video_uri": selected_uri,
            "input_labeling": {"privacy_safe_input": True, "raw_video_bypass_used": False},
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_input_audit.json",
        {
            "schema_version": "v1",
            "status": "ready",
            "selected_video_source_id": "privacy_processed_video_uri",
            "selected_video_uri": _uri("privacy/final_walkthrough.mov"),
            "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
            "source_checksum_sha256": source_checksum,
            "source_is_final_walkthrough": True,
            "derivative_of_final_walkthrough": True,
            "privacy_safe_input": True,
            "raw_video_bypass_used": False,
            "output_video_uri": selected_uri,
            "output_checksum_sha256": output_checksum,
        },
    )
    _write_json(
        root / "pipeline" / "site_package" / "canonical_site_package.json",
        {
            "conditioning": {
                "rgb_video": {
                    "privacy_safe_world_model_input": {
                        "uri": selected_uri,
                        "checksum_sha256": output_checksum,
                    }
                }
            }
        },
    )
    _write_json(
        root
        / "pipeline"
        / "site_package"
        / "provider_adapter_inputs"
        / "world_labs_marble.json",
        {
            "status": "ready",
            "conditioning_inputs": {
                "rgb_video": {
                    "uri": selected_uri,
                    "privacy_safe": True,
                    "input_audit_uri": _uri("pipeline/worldlabs_input_audit.json"),
                    "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
                    "checksum_sha256": output_checksum,
                    "source_checksum_sha256": source_checksum,
                }
            },
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_request_manifest.json",
        {
            "schema_version": "v1",
            "provider_name": "world_labs",
            "provider_model": "marble-1.1",
            "status": "ready_for_generation",
            "selected_video_source_id": "privacy_safe_world_model_input",
            "selected_video_uri": selected_uri,
            "selected_input_checksum_sha256": output_checksum,
            "source_input_checksum_sha256": source_checksum,
            "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
            "worldlabs_input_audit_uri": _uri("pipeline/worldlabs_input_audit.json"),
            "privacy_safe_input": True,
            "input_labeling": {"privacy_safe_input": True, "raw_video_bypass_used": False},
            "input_audit": {
                "privacy_safe_input": True,
                "raw_video_bypass_used": False,
                "source_manifest_uri": _uri("pipeline/privacy_processing_manifest.json"),
                "output_video_uri": selected_uri,
                "output_checksum_sha256": output_checksum,
                "source_checksum_sha256": source_checksum,
            },
        },
    )
    _write_json(root / "pipeline" / "worldlabs_operation_manifest.json", {"status": "ready"})
    _write_json(
        root / "pipeline" / "worldlabs_world_manifest.json",
        {
            "world_id": "world-1",
            "assets": {
                "mesh": {"collider_mesh_url": "https://cdn.example/collider.glb"},
                "splats": {},
            },
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_assets" / "materialized_assets_manifest.json",
        {
            "schema_version": "worldlabs_asset_materialization.v1",
            "status": "complete",
            "download_count": 1,
            "downloads": [
                {
                    "kind": "collider_mesh_glb",
                    "local_path": str(collider_path),
                    "sha256": sha256(collider_path.read_bytes()).hexdigest(),
                }
            ],
        },
    )
    _write_json(
        root / "pipeline" / "worldlabs_export_manifest.json",
        {
            "schema_version": "worldlabs_export_manifest.v1",
            "output_collider_mesh_path": str(collider_path),
            "remote_collider_mesh_glb_url": "https://cdn.example/collider.glb",
        },
    )
    _write_json(root / "pipeline" / "provider_preview_status.json", {"status": "ready"})
    _write_json(root / "pipeline" / "provider_run_manifest.json", {"status": "ready"})
    _write_json(
        root / "pipeline" / "webapp_sync_result.json",
        {
            "status": "succeeded",
            "latest_stage": "qualification",
            "syncs": {
                "qualification": {
                    "status": "succeeded",
                    "response": {"ok": True, "requestId": "request-1"},
                    "attachment_payload": {
                        "scene_id": "scene-1",
                        "capture_id": "capture-1",
                        "site_submission_id": "site-submission-1",
                        "request_id": "request-1",
                        "buyer_request_id": "buyer-request-1",
                        "capture_job_id": "capture-job-1",
                        "upstream_links_verified": True,
                        "missing_upstream_links": [],
                    },
                    "buyer_access_check": {
                        "buyer_access_checked": False,
                        "buyer_accessible": False,
                    },
                }
            },
        },
    )


def test_production_handoff_readiness_is_ready_except_owner_gpu(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)

    result = build_production_handoff_readiness(capture_root=root, mode="production")
    manifest = _read_json(root / "pipeline" / "production_handoff_readiness_manifest.json")

    assert result["status"] == "ready_except_owner_gpu_simulator_execution"
    assert result["owner_gpu_simulator_execution_is_only_unproven_step"] is True
    assert result["remaining_unproven_steps"] == ["owner_gpu_simulator_execution_not_run"]
    assert result["proof_summary"]["privacy_safe_worldlabs_input"] is True
    assert result["proof_summary"]["webapp_sync_succeeded"] is True
    assert result["proof_summary"]["webapp_upstream_links_verified"] is True
    assert result["proof_summary"]["worldlabs_generation_manifested"] is True
    assert result["proof_summary"]["cpu_preflight_ready_for_owner_gpu"] is True
    assert result["proof_summary"]["arena_environment_packet_manifested"] is True
    assert result["proof_summary"]["rank_fidelity_result_proven"] is False
    assert result["artifacts"]["arena_environment_packet"]["exists"] is True
    assert manifest["status"] == result["status"]


def test_production_handoff_readiness_accepts_post_owner_gpu_proof(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    automation_dir = root / "pipeline" / "simulation_automation"
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "ready_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": True,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_blocked_manifest.v1",
            "status": "resolved",
            "blocker_id": "owner_gpu_simulator_execution_not_run",
        },
    )

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "ready_after_owner_gpu_simulator_execution"
    assert result["owner_gpu_simulator_execution_is_only_unproven_step"] is False
    assert result["remaining_unproven_steps"] == []
    assert result["proof_summary"]["owner_gpu_simulator_execution_proven"] is True
    assert result["claim_boundary"]["owner_gpu_simulator_execution_proven"] is True
    assert "gpu_handoff_missing_owner_gpu_blocker" not in result["blockers"]


def test_production_handoff_readiness_blocks_isaac_packet_with_generic_owner_proof(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    automation_dir = root / "pipeline" / "simulation_automation"
    _write_json(
        root / "pipeline" / "first_gpu_e2e_run_packet" / "first_gpu_run_packet.json",
        {
            "schema_version": "first_gpu_run_packet.v1",
            "simulator": "isaac_sim",
        },
    )
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "ready_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": True,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "simulator_backend": "mujoco",
            "isaac_sim_execution_proven": False,
            "isaac_robot_asset_execution_proven": False,
            "unitree_g1_asset_spawned": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_blocked_manifest.v1",
            "status": "resolved",
            "blocker_id": "owner_gpu_simulator_execution_not_run",
        },
    )

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_after_owner_gpu_handoff"
    assert "isaac_sim_unitree_g1_execution_not_proven" in result["blockers"]
    assert result["proof_summary"]["expected_owner_simulator"] == "isaac_sim"
    assert result["proof_summary"]["generic_owner_gpu_simulator_execution_proven"] is True
    assert result["proof_summary"]["owner_gpu_simulator_execution_proven"] is False
    assert result["proof_summary"]["isaac_unitree_g1_execution_proven"] is False
    assert result["claim_boundary"]["owner_gpu_simulator_execution_proven"] is False
    assert result["claim_boundary"]["isaac_sim_execution_proven"] is False


def test_production_handoff_readiness_accepts_mujoco_packet_with_mujoco_g1_proof(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    automation_dir = root / "pipeline" / "simulation_automation"
    _write_json(
        root / "pipeline" / "first_gpu_e2e_run_packet" / "first_gpu_run_packet.json",
        {
            "schema_version": "first_gpu_run_packet.v1",
            "simulator": "mujoco",
        },
    )
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "ready_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": True,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "simulator_backend": "mujoco",
            "mujoco_g1_asset_execution_proven": True,
            "mujoco_g1_asset_spawned": True,
            "isaac_sim_execution_proven": False,
            "isaac_robot_asset_execution_proven": False,
            "unitree_g1_asset_spawned": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_blocked_manifest.v1",
            "status": "resolved",
            "blocker_id": "owner_gpu_simulator_execution_not_run",
        },
    )

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "ready_after_owner_gpu_simulator_execution"
    assert result["remaining_unproven_steps"] == []
    assert result["proof_summary"]["expected_owner_simulator"] == "mujoco"
    assert result["proof_summary"]["generic_owner_gpu_simulator_execution_proven"] is True
    assert result["proof_summary"]["mujoco_unitree_g1_execution_proven"] is True
    assert result["proof_summary"]["selected_simulator_execution_proven"] is True
    assert result["proof_summary"]["owner_gpu_simulator_execution_proven"] is True
    assert result["claim_boundary"]["owner_gpu_simulator_execution_proven"] is True
    assert result["claim_boundary"]["mujoco_g1_asset_execution_proven"] is True
    assert result["claim_boundary"]["mujoco_g1_asset_spawned"] is True
    assert result["claim_boundary"]["rank_fidelity_result_proven"] is False
    assert "isaac_sim_unitree_g1_execution_not_proven" not in result["blockers"]
    assert "mujoco_g1_execution_not_proven" not in result["blockers"]


def test_production_handoff_readiness_blocks_mujoco_packet_without_mujoco_g1_proof(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    automation_dir = root / "pipeline" / "simulation_automation"
    _write_json(
        root / "pipeline" / "first_gpu_e2e_run_packet" / "first_gpu_run_packet.json",
        {
            "schema_version": "first_gpu_run_packet.v1",
            "simulator": "mujoco",
        },
    )
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "ready_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": True,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "simulator_backend": "mujoco",
            "mujoco_g1_asset_execution_proven": False,
            "mujoco_g1_asset_spawned": False,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_blocked_manifest.v1",
            "status": "resolved",
            "blocker_id": "owner_gpu_simulator_execution_not_run",
        },
    )

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_after_owner_gpu_handoff"
    assert "mujoco_g1_execution_not_proven" in result["blockers"]
    assert result["proof_summary"]["expected_owner_simulator"] == "mujoco"
    assert result["proof_summary"]["generic_owner_gpu_simulator_execution_proven"] is True
    assert result["proof_summary"]["mujoco_unitree_g1_execution_proven"] is False
    assert result["proof_summary"]["selected_simulator_execution_proven"] is False
    assert result["proof_summary"]["owner_gpu_simulator_execution_proven"] is False
    assert result["claim_boundary"]["owner_gpu_simulator_execution_proven"] is False
    assert result["claim_boundary"]["mujoco_g1_asset_execution_proven"] is False


def test_production_handoff_readiness_blocks_before_gpu_when_worldlabs_missing(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    (root / "pipeline" / "worldlabs_world_manifest.json").unlink()
    build_simulation_automation(capture_root=root)

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_before_owner_gpu_handoff"
    assert result["owner_gpu_simulator_execution_is_only_unproven_step"] is False
    assert "worldlabs_world_manifest_missing" in result["blockers"]


def test_production_handoff_readiness_requires_webapp_upstream_link_truth(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    webapp_sync = _read_json(root / "pipeline" / "webapp_sync_result.json")
    qualification = webapp_sync["syncs"]["qualification"]  # type: ignore[index]
    attachment = qualification["attachment_payload"]  # type: ignore[index]
    attachment["buyer_request_id"] = ""  # type: ignore[index]
    attachment["missing_upstream_links"] = ["buyer_request_id"]  # type: ignore[index]
    attachment["upstream_links_verified"] = False  # type: ignore[index]
    _write_json(root / "pipeline" / "webapp_sync_result.json", webapp_sync)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_before_owner_gpu_handoff"
    assert result["owner_gpu_simulator_execution_is_only_unproven_step"] is False
    assert result["proof_summary"]["privacy_safe_worldlabs_input"] is True
    assert "missing_webapp_buyer_request_id" in result["blockers"]
    assert result["proof_summary"]["webapp_sync_succeeded"] is True
    assert result["proof_summary"]["webapp_upstream_links_verified"] is False


def test_production_handoff_readiness_helper_string_list_edges() -> None:
    marker = object()

    assert readiness._string_list(None) == []
    assert readiness._string_list("one") == ["one"]
    assert readiness._string_list(["one", "one", "two", ""]) == ["one", "two"]
    assert readiness._string_list(marker) == [str(marker)]


def test_production_handoff_readiness_reports_missing_artifacts_when_qa_is_skipped(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)

    result = build_production_handoff_readiness(
        capture_root=root,
        mode="",
        run_provider_preview_qa=False,
    )

    assert result["mode"] == "production"
    assert result["status"] == "blocked_before_owner_gpu_handoff"
    assert "provider_preview_qa_missing" in result["blockers"]
    assert "worldlabs_request_manifest_missing" in result["blockers"]
    assert "worldlabs_operation_manifest_missing" in result["blockers"]
    assert "worldlabs_asset_materialization_manifest_missing" in result["blockers"]
    assert "worldlabs_export_manifest_missing" in result["blockers"]
    assert "scene_asset_inventory_missing" in result["blockers"]
    assert "scene_asset_preflight_missing" in result["blockers"]
    assert "cpu_preflight_manifest_missing" in result["blockers"]
    assert "arena_environment_packet_missing" in result["blockers"]
    assert "gpu_handoff_packet_missing" in result["blockers"]
    assert "gpu_owner_system_proof_schema_missing" in result["blockers"]
    assert "owner_gpu_blocked_manifest_missing" in result["blockers"]


def test_production_handoff_readiness_reports_malformed_artifacts(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    pipeline_dir = root / "pipeline"
    automation_dir = pipeline_dir / "simulation_automation"

    _write_json(pipeline_dir / "worldlabs_request_manifest.json", {"privacy_safe_input": False})
    _write_json(pipeline_dir / "worldlabs_operation_manifest.json", {"status": "blocked"})
    _write_json(pipeline_dir / "worldlabs_world_manifest.json", {"status": "blocked", "world_id": "world-1"})
    _write_json(
        pipeline_dir / "worldlabs_assets" / "materialized_assets_manifest.json",
        {"status": "complete", "download_count": 0},
    )
    _write_json(automation_dir / "scene_asset_inventory.json", {"asset_count": 0})
    _write_json(automation_dir / "cpu_preflight_manifest.json", {"ready_for_owner_gpu_preflight": False})
    _write_json(
        automation_dir / "arena_environment_packet.json",
        {"simulator_execution_proven": True, "rank_fidelity_result_proven": True},
    )
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "status": "not_ready",
            "rank_fidelity_result_proven": True,
            "blockers": ["custom_blocker"],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json",
        {"blocker_id": "wrong_blocker"},
    )

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_before_owner_gpu_handoff"
    assert "worldlabs_request_not_privacy_safe" in result["blockers"]
    assert "worldlabs_operation_not_current_for_privacy_safe_input" in result["blockers"]
    assert "worldlabs_world_not_current_for_privacy_safe_input" in result["blockers"]
    assert "worldlabs_materialized_asset_download_missing" in result["blockers"]
    assert "scene_asset_inventory_empty" in result["blockers"]
    assert "cpu_preflight_not_ready_for_owner_gpu" in result["blockers"]
    assert "arena_packet_illegally_marks_simulator_execution" in result["blockers"]
    assert "arena_packet_illegally_marks_rank_fidelity" in result["blockers"]
    assert "gpu_handoff_packet_not_ready" in result["blockers"]
    assert "gpu_handoff_illegally_marks_rank_fidelity" in result["blockers"]
    assert "custom_blocker" in result["blockers"]
    assert "gpu_handoff_missing_owner_gpu_blocker" in result["blockers"]
    assert "owner_gpu_blocked_manifest_wrong_blocker" in result["blockers"]


def test_production_handoff_readiness_reports_missing_world_id(tmp_path: Path) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    _write_json(root / "pipeline" / "worldlabs_world_manifest.json", {"status": "ready"})

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_before_owner_gpu_handoff"
    assert "worldlabs_world_id_missing" in result["blockers"]


def test_production_handoff_readiness_preserves_post_proof_handoff_blockers(
    tmp_path: Path,
) -> None:
    root = _capture_root(tmp_path)
    _write_privacy_and_worldlabs(root)
    build_marble_sim_assets(capture_root=root)
    build_simulation_automation(capture_root=root)
    automation_dir = root / "pipeline" / "simulation_automation"
    _write_json(
        automation_dir / "gpu_handoff_packet.json",
        {
            "schema_version": "gpu_handoff_packet.v1",
            "status": "ready_for_owner_gpu_preflight_handoff",
            "ready_for_owner_gpu_preflight": True,
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": ["post_proof_blocker"],
        },
    )
    _write_json(
        automation_dir / "owner_gpu_simulator_execution_proof_manifest.json",
        {
            "schema_version": "owner_gpu_simulator_execution_proof_manifest.v1",
            "status": "accepted",
            "owner_gpu_simulator_execution_proven": True,
            "simulator_execution_proven": True,
            "rank_fidelity_result_proven": False,
            "public_claim_upgrade_allowed": False,
            "blockers": [],
        },
    )

    result = build_production_handoff_readiness(capture_root=root, mode="production")

    assert result["status"] == "blocked_after_owner_gpu_handoff"
    assert result["blockers"] == ["post_proof_blocker"]


def test_production_handoff_readiness_main_returns_success_blocked_and_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        readiness,
        "build_production_handoff_readiness",
        lambda **_: {"status": "ready_except_owner_gpu_simulator_execution", "blockers": []},
    )
    assert readiness.main(["--capture-root", str(tmp_path), "--mode", "advisory", "--skip-provider-preview-qa"]) == 0
    assert "[production-handoff-readiness] status=ready_except_owner_gpu_simulator_execution" in capsys.readouterr().out

    monkeypatch.setattr(
        readiness,
        "build_production_handoff_readiness",
        lambda **_: {"status": "blocked_before_owner_gpu_handoff", "blockers": ["missing"]},
    )
    assert readiness.main(["--capture-root", str(tmp_path)]) == 1
    blocked_output = capsys.readouterr().out
    assert "[production-handoff-readiness] status=blocked_before_owner_gpu_handoff" in blocked_output
    assert "[production-handoff-readiness] blockers=missing" in blocked_output

    def raise_value_error(**_: object) -> dict[str, object]:
        raise ValueError("bad capture")

    monkeypatch.setattr(readiness, "build_production_handoff_readiness", raise_value_error)
    assert readiness.main(["--capture-root", str(tmp_path)]) == 2
    assert "[production-handoff-readiness] status=error reason=bad capture" in capsys.readouterr().out
