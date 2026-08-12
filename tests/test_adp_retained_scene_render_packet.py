from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.adp_retained_scene_render_packet import (
    RetainedSceneRenderPacketError,
    build_retained_scene_gpu_render_bundle,
    build_retained_scene_gpu_render_request,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.adp_retained_scene_render_vast import (
    _authority_environment,
    run_retained_scene_render_vast,
    validate_retained_scene_render_bundle,
    validate_retained_scene_render_paid_attempt_authority,
)
from blueprint_pipeline.gaussian_splat_decode import (
    SplatData,
    write_standard_3dgs_ply,
    write_standard_3dgs_ply_subset_exact,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_shell_script,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _absolute_record(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _relative_record(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def test_retained_scene_render_authority_environment_restores_retry_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "caller-api")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "caller-launch")
    monkeypatch.setenv("BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS", "caller-retry")

    with _authority_environment():
        assert os.environ["BLUEPRINT_ALLOW_VAST_API_CALLS"] == "1"
        assert os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "1"
        assert os.environ["BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"] == "0"

    assert os.environ["BLUEPRINT_ALLOW_VAST_API_CALLS"] == "caller-api"
    assert os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "caller-launch"
    assert os.environ["BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"] == "caller-retry"


def _task_freeze(task_id: str, slot: int) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "dual_task_task_freeze.v1",
        "task_id": task_id,
        "prompt": f"relocate observed object {slot}",
        "task_kind": "rigid_object_manipulation",
        "scene_freeze_digest": _digest("a"),
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "frozen_before_learned_policy_execution": True,
        "learned_policy_outcomes_accessed": False,
        "source_object": {
            "instance_id": f"source_{slot}",
            "semantic_label": "fixture_object",
            "observed_bounds_world_m": {
                "minimum": [0.0, 0.0, 0.0],
                "maximum": [0.1, 0.1, 0.1],
            },
            "observed_pose_world": {
                "position_world_m": [0.05, 0.05, 0.05],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "support_or_attachment_id": f"support_{slot}",
            "collision_identity_receipt_digest": _digest("b"),
            "support_receipt_digest": _digest("c"),
            "franka_placement_packet_digest": _digest("d"),
            "visibility_receipt_digest": _digest("e"),
        },
        "removal_plan": {
            "removal_id": f"removal_{slot}",
            "mask_set_id": f"mask_set_{slot}",
            "source_collider_prim_path": f"/Root/source_{slot}",
            "collider_deletion_id": f"collider_{slot}",
            "replacement_asset_id": f"replacement_{slot}",
            "replacement_qualification_id": f"qualification_{slot}",
        },
        "cameras": {
            "external": f"external_{slot}",
            "wrist": f"wrist_{slot}",
            "overview": f"overview_{slot}",
        },
        "overview_camera_policy_input": False,
        "overview_camera_deterministic_scoring_input": False,
        "execution_contract": {
            "control_frequency_hz": 20,
            "maximum_steps": 200,
            "settle_window_steps": 10,
            "seeds": [slot],
            "canonical_scenario_cell_id": f"canonical_{slot}",
            "reset_state": {"robot": "home", "object": "source_start"},
        },
        "deterministic_success_predicates": ["released", "settled"],
        "failure_rungs": ["never_moved", "collision_failure"],
        "target_configuration": {
            "kind": "pose_volume",
            "position_bounds_world_m": {"minimum": [0.2, 0.2, 0.0], "maximum": [0.3, 0.3, 0.1]},
            "orientation_reference_xyzw": [0.0, 0.0, 0.0, 1.0],
            "maximum_orientation_error_rad": 0.1,
            "support_id": f"destination_{slot}",
            "release_required": True,
        },
        "articulation_graph": None,
        "task_freeze_digest": "",
    }
    value["task_freeze_digest"] = canonical_digest(value, digest_field="task_freeze_digest")
    return value


def _source_ply(path: Path) -> Path:
    values = np.arange(10, dtype=np.float32)
    return write_standard_3dgs_ply(
        SplatData(
            count=10,
            xyz=np.stack((values, values + 10, values + 20), axis=1),
            opacity=values + 30,
            f_dc=np.stack((values + 40, values + 50, values + 60), axis=1),
            scales=np.stack((values + 70, values + 80, values + 90), axis=1),
            quats=np.stack((values + 100, values + 110, values + 120, values + 130), axis=1),
            properties=(),
            sh_rest=None,
        ),
        path,
    )


def _camera_contract(path: Path, camera_id: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "camera_id": camera_id,
            "T_world_camera_provider_frame": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "width": 2, "height": 2},
        }
    ]
    path.write_text(canonical_json(rows) + "\n", encoding="utf-8")
    return path


def _authority(path: Path) -> Path:
    value: dict[str, object] = {
        "schema_version": "third_scene_dual_task_execution_authority.v1",
        "program_id": "arm-decision-proof-v1",
        "publisher_scene_id": "840920",
        "private_rights_admitted_scene_derived_uploads_authorized": True,
        "raw_interiorgs_upload_authorized": False,
        "training_authorized": False,
        "public_dataset_bytes_publication_authorized": False,
        "retention": "bounded_to_goal_then_provider_zero",
        "paid_compute": {
            "provider": "vast",
            "hard_total_spend_cap_usd": 12.0,
            "zero_retry": True,
            "provider_zero_required_for_lane": True,
        },
        "authority_digest": "",
    }
    value["authority_digest"] = canonical_digest(value, digest_field="authority_digest")
    _write_json(path, value)
    return path


def _repo(root: Path) -> tuple[Path, Path]:
    repo = root / "repo"
    renderer = repo / "tools" / "splat_render"
    (renderer / "src").mkdir(parents=True)
    for relative in (
        "render_splat.mjs",
        "harness.html",
        "package.json",
        "package-lock.json",
        "src/render_entry.mjs",
    ):
        target = renderer / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "{}\n" if target.suffix == ".json" else "fixture\n",
            encoding="utf-8",
        )
    scripts = repo / "scripts"
    scripts.mkdir()
    checkout = Path(__file__).resolve().parents[1]
    for name in (
        "run_adp_retained_scene_render_provider_runtime.sh",
        "adp_retained_scene_render_provider_runner.mjs",
    ):
        shutil.copy2(checkout / "scripts" / name, scripts / name)
    vendor = root / "vendor"
    for package in ("@sparkjsdev/spark", "fflate", "playwright", "playwright-core", "three"):
        target = vendor / package
        target.mkdir(parents=True)
        (target / "index.js").write_text("fixture\n", encoding="utf-8")
    for command in (
        ("init",),
        ("add", "."),
        (
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            "fixture",
        ),
    ):
        subprocess.run(["git", "-C", str(repo), *command], check=True, capture_output=True)
    return repo, vendor


def _inputs(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "direct_set"
    source = _source_ply(root / "source.ply")
    shared = root / "shared_scene_union"
    shared.mkdir(parents=True)
    deleted = np.array([1, 6], dtype=np.int64)
    retained_indices = np.array([0, 2, 3, 4, 5, 7, 8, 9], dtype=np.int64)
    np.save(shared / "deleted_source_indices.npy", deleted, allow_pickle=False)
    np.save(shared / "retained_source_indices.npy", retained_indices, allow_pickle=False)
    retained = write_standard_3dgs_ply_subset_exact(
        source, shared / "retained_scene_gaussians.ply", retained_indices
    )
    tasks: list[dict[str, object]] = []
    lanes: list[dict[str, object]] = []
    for slot in (1, 2):
        task = _task_freeze(f"task_{slot}", slot)
        freeze = tmp_path / "freezes" / f"task_{slot}.json"
        _write_json(freeze, task)
        removal = task["removal_plan"]
        assert isinstance(removal, dict)
        tasks.append(
            {
                "task_id": task["task_id"],
                "task_freeze_digest": task["task_freeze_digest"],
                "removal_id": removal["removal_id"],
                "mask_set_id": removal["mask_set_id"],
                "task_freeze": _absolute_record(freeze),
            }
        )
        camera = _camera_contract(tmp_path / "cameras" / f"task_{slot}.json", f"camera_{slot}")
        lanes.append({"task_id": task["task_id"], "camera_contract_path": str(camera)})
    candidate: dict[str, object] = {
        "schema_version": "adp009b_direct_evidence_expansion_set.v1",
        "task_candidates": tasks,
        "shared_scene_union": {
            "counts": {"source": 10, "deleted_total": 2, "retained_total": 8},
            "outputs": {
                "deleted_source_indices": _relative_record(
                    root, shared / "deleted_source_indices.npy"
                ),
                "retained_source_indices": _relative_record(
                    root, shared / "retained_source_indices.npy"
                ),
                "retained_scene_gaussians": _relative_record(root, retained),
            },
        },
        "source_standard_splat": _absolute_record(source),
        "claim_boundary": {"candidate_derived_layers_only": True},
        "receipt_digest": "",
    }
    candidate["receipt_digest"] = canonical_digest(candidate, digest_field="receipt_digest")
    candidate_path = root / "candidate.json"
    _write_json(candidate_path, candidate)
    return candidate_path, {"lanes": lanes, "candidate": candidate}


def test_seals_two_task_bundle_and_rehearses_exact_uploaded_entrypoint(tmp_path: Path) -> None:
    candidate, inputs = _inputs(tmp_path)
    repo, vendor = _repo(tmp_path)
    authority = _authority(tmp_path / "authority.json")
    request: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_render_execution": True,
        "learned_policy_outcomes_accessed": False,
        "candidate_set_path": str(candidate),
        "execution_authority_path": str(authority),
        "renderer_vendor_root": str(vendor),
        "task_lanes": inputs["lanes"],
        "private_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "provider_training": False,
            "publication": False,
            "retention": "bounded_to_goal_then_provider_zero",
        },
    }
    request = build_retained_scene_gpu_render_request(request)
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)

    receipt = build_retained_scene_gpu_render_bundle(
        request_path=request_path, repo_root=repo, job_dir=tmp_path / "job"
    )

    assert receipt["status"] == "ready"
    assert receipt["source_pair_per_task"] is True
    assert receipt["retained_frame_per_task"] is True
    assert receipt["exact_bundle_entrypoint_rehearsal"]["status"] == "passed"
    assert receipt["exact_bundle_entrypoint_rehearsal"]["gpu_runtime_started"] is False
    assert receipt["exact_bundle_entrypoint_rehearsal"]["provider_mutations_performed"] == 0
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
    assert "provider_runtime/input/source_standard.ply" in names
    assert "provider_runtime/input/shared_retained_scene.ply" in names
    assert "provider_runtime/input/direct_evidence_successor_set.json" in names
    assert (tmp_path / "job/provider_runtime/input/source_standard.ply").stat().st_ino == (
        tmp_path / "direct_set/source.ply"
    ).stat().st_ino
    source_shell = repo / "scripts/run_adp_retained_scene_render_provider_runtime.sh"
    bundled_shell = (
        tmp_path / "job/provider_runtime/run_adp_retained_scene_render_provider_runtime.sh"
    )
    assert source_shell.stat().st_ino != bundled_shell.stat().st_ino
    assert source_shell.stat().st_mode & 0o111 == 0
    assert bundled_shell.stat().st_mode & 0o111
    assert (
        validate_retained_scene_render_bundle(receipt)["bundle_sha256"] == receipt["bundle_sha256"]
    )
    dry_run = run_retained_scene_render_vast(
        job_dir=tmp_path / "vast_dry_run",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=receipt,
    )
    assert dry_run["status"] == "dry_run_ready"
    assert dry_run["provider_mutations_performed"] == 0
    attempt_authority: dict[str, object] = {
        "schema_version": "adp009d_retained_scene_gpu_render_paid_attempt_authority.v1",
        "authority_kind": "explicit_user_direction_in_current_goal",
        "purpose": "exact_retained_scene_gpu_render",
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": receipt["execution_authority"]["authority_digest"],
        "bundle_sha256": receipt["bundle_sha256"],
        "blueprint_commit": receipt["blueprint_commit"],
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": 12.0,
        "maximum_single_resource_ttl_seconds": 10_800,
        "maximum_hourly_rate_usd": 2.0,
        "external_active_instance_allowlist": [47373597],
        "authorization_digest": "",
    }
    attempt_authority["authorization_digest"] = canonical_digest(
        attempt_authority, digest_field="authorization_digest"
    )
    assert (
        validate_retained_scene_render_paid_attempt_authority(
            attempt_authority,
            prepared_bundle=receipt,
            max_hourly_rate_usd=2.0,
            hard_ttl_seconds=10_800,
            allowed_active_instance_ids=[47373597],
        )["authorization_digest"]
        == attempt_authority["authorization_digest"]
    )
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "vast_preflight",
        generated_at="2026-08-11T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_retained_scene_render",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.test/bundle",
        provider_output_put_url="https://example.test/output",
    )
    assert preflight["status"] == "passed", preflight
    probe = _probe_shell_script(
        "https://example.test/heartbeat",
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_retained_scene_render",
    )
    assert "adp_retained_scene_render_provider_bundle" in probe
    assert "apt-get" not in probe


def test_rejects_more_than_five_task_lanes() -> None:
    request = {
        "schema_version": "adp009d_retained_scene_gpu_render_request.v1",
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009D",
        "frozen_before_render_execution": True,
        "learned_policy_outcomes_accessed": False,
        "candidate_set_path": "/candidate.json",
        "execution_authority_path": "/authority.json",
        "renderer_vendor_root": "/vendor",
        "task_lanes": [
            {"task_id": f"task_{slot}", "camera_contract_path": "/camera.json"} for slot in range(6)
        ],
        "private_upload_policy": {
            "raw_dataset_bytes_upload": False,
            "private_derived_upload": True,
            "provider_training": False,
            "publication": False,
            "retention": "bounded_to_goal_then_provider_zero",
        },
    }
    with pytest.raises(RetainedSceneRenderPacketError, match="task_lane_count_invalid"):
        build_retained_scene_gpu_render_request(request)
