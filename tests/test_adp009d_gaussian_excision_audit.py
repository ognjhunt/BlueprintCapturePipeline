from __future__ import annotations

import json
import hashlib
import importlib.util
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline import adp_gaussian_excision_vast as excision_vast
from blueprint_pipeline.paid_attempt_authority import bind_lane_prior_spend
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import vast_provider_adapter as vast_adapter
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip
from blueprint_pipeline.public_scene_gaussian_excision_audit import (
    CONTRIBUTION_CLASS_ORDER,
    CONTRIBUTION_EVIDENCE_SCHEMA,
    FREEZE_SCHEMA,
    OWNERSHIP_AGGREGATION_POLICY_SCHEMA,
    OWNERSHIP_RECEIPT_SCHEMA,
    OWNERSHIP_REPLAY_SCHEMA,
    classify_excision_ownership,
    materialize_excision_audit_freeze,
    materialize_excision_ownership_aggregation_policy,
    materialize_excision_ownership,
    materialize_excision_ownership_replay,
    select_maximally_diverse_holdout_pair,
    _normalized_camera_row,
    _verified_render_input_packet,
)


POLICY = {
    "minimum_per_view_contribution": 1.0 / 255.0,
    "owned_min_core_fraction": 0.98,
    "retained_max_core_fraction": 0.20,
    "minimum_core_camera_count": 2,
    "maximum_protected_camera_count_for_owned": 0,
    "minimum_geometry_score_owned": 0.5,
    "geometry_sigma_extent": 3.0,
    "geometry_margin_m": 0.02,
    "neighbor_count": 2,
    "neighbor_iterations": 2,
    "neighbor_radius_m": 0.01,
    "neighbor_blend": 0.25,
    "graph_owned_min_score": 0.95,
    "graph_retained_max_score": 0.20,
    "deterministic_repetitions": 2,
    "contribution_quantization_decimals": 6,
}


def _camera(camera_id: str, x: float, angle_deg: float) -> dict[str, object]:
    angle = np.deg2rad(angle_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[0, 3] = x
    return {
        "camera_id": camera_id,
        "T_world_camera_opencv": transform.tolist(),
        "intrinsics": {
            "model": "PINHOLE",
            "fx": 40.0,
            "fy": 40.0,
            "cx": 32.0,
            "cy": 24.0,
            "width": 64,
            "height": 48,
        },
    }


def test_diverse_holdout_split_is_outcome_blind_and_deterministic() -> None:
    cameras = [
        _camera("front", 0.0, 0.0),
        _camera("near_left", -0.4, 5.0),
        _camera("near_right", 0.4, -5.0),
        _camera("far_left", -1.5, 35.0),
        _camera("far_right", 1.5, -35.0),
        _camera("raised", 0.1, 10.0),
        _camera("low", -0.1, -10.0),
        _camera("working", 0.2, 2.0),
    ]
    fractions = {str(row["camera_id"]): 0.2 for row in cameras}

    first = select_maximally_diverse_holdout_pair(cameras, projected_target_fraction=fractions)
    second = select_maximally_diverse_holdout_pair(
        list(reversed(cameras)), projected_target_fraction=fractions
    )

    assert first == second
    assert first["heldout_camera_ids"] == ["far_left", "far_right"]
    assert len(first["calibration_camera_ids"]) == 6
    assert first["outcome_fields_accessed"] is False


def test_provider_frame_camera_normalizes_at_shared_excision_seam() -> None:
    camera = _camera("front", 0.0, 0.0)
    provider_row = dict(camera)
    provider_row["T_world_camera_provider_frame"] = provider_row.pop(
        "T_world_camera_opencv"
    )

    normalized = _normalized_camera_row(provider_row)

    assert "T_world_camera_provider_frame" not in normalized
    assert normalized["T_world_camera_opencv"] == camera["T_world_camera_opencv"]


def test_excision_freeze_opens_render_packet_and_rejects_mask_tamper(
    tmp_path: Path,
) -> None:
    cameras = [_camera("front", 0.0, 0.0)]
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(json.dumps(cameras), encoding="utf-8")
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    image = np.full((48, 64, 3), 64, dtype=np.uint8)
    mask = np.full((48, 64), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_root / "front.png"), image)
    assert cv2.imwrite(str(mask_root / "front.png"), mask)
    scene = {
        "publisher_scene_id": "840920",
        "task_id": "task_a",
        "target_instance_id": "165",
        "mask_set_id": "mask_a",
        "removal_id": "removal_a",
    }
    receipt = {
        "schema_version": "public_scene_interiorgs_edit_input_receipt.v2",
        "status": "render_derived_input_packet_materialized",
        "request_digest": "sha256:" + "1" * 64,
        "scene": scene,
        "derived_artifacts": {
            "cameras": {
                "sha256": "sha256:"
                + hashlib.sha256(camera_path.read_bytes()).hexdigest(),
                "size_bytes": camera_path.stat().st_size,
            },
            "images": [_record(image_root / "front.png", image_root)],
            "masks": [_record(mask_root / "front.png", mask_root)],
        },
        "renderer": {
            "authorization_class": "method_input",
            "purpose_bound": True,
            "render_manifest_digests": {"images": "sha256:" + "2" * 64},
        },
        "proof_boundaries": {"gaussian_ownership_qualified": False},
    }
    receipt["derived_artifacts"]["images"][0]["camera_id"] = "front"
    receipt["derived_artifacts"]["masks"][0]["camera_id"] = "front"
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    binding = _verified_render_input_packet(
        receipt_path=receipt_path,
        camera_path=camera_path,
        image_root=image_root,
        outer_mask_root=mask_root,
        scene=scene,
    )
    assert binding["authorization_class"] == "method_input"
    assert binding["gaussian_ownership_qualified_upstream"] is False

    (mask_root / "front.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="masks_join_mismatch"):
        _verified_render_input_packet(
            receipt_path=receipt_path,
            camera_path=camera_path,
            image_root=image_root,
            outer_mask_root=mask_root,
            scene=scene,
        )


def test_contribution_geometry_and_neighborhood_create_exhaustive_three_way_labels() -> None:
    # Class order is protected, target_core, uncertain.  Gaussian 0 is clean
    # target evidence, 1 is protected, 2 is mixed, 3 is unseen/far, and 4 has
    # strong target evidence but a protected-view veto.
    evidence = np.zeros((3, 3, 5), dtype=np.float64)
    evidence[:, 1, 0] = 2.0
    evidence[:, 0, 1] = 2.0
    evidence[:, 0, 2] = 1.0
    evidence[:, 1, 2] = 1.0
    evidence[:, 1, 4] = 2.0
    evidence[0, 0, 4] = 0.1
    xyz = np.asarray(
        [[0.5, 0.5, 0.5], [2.0, 2.0, 2.0], [0.6, 0.6, 0.6], [5, 5, 5], [0.4, 0.4, 0.4]],
        dtype=np.float64,
    )
    log_scales = np.full_like(xyz, -4.0)

    result = classify_excision_ownership(
        evidence,
        xyz=xyz,
        log_scales=log_scales,
        target_aabb_min_m=[0.0, 0.0, 0.0],
        target_aabb_max_m=[1.0, 1.0, 1.0],
        policy=POLICY,
    )

    assert np.flatnonzero(result["owned"]).tolist() == [0]
    assert np.flatnonzero(result["retained"]).tolist() == [1, 3]
    assert np.flatnonzero(result["ambiguous"]).tolist() == [2, 4]
    assert np.all(
        result["owned"].astype(np.uint8)
        + result["retained"].astype(np.uint8)
        + result["ambiguous"].astype(np.uint8)
        == 1
    )
    assert result["protected_camera_count"][4] == 1


def _splat(path: Path) -> Path:
    xyz = np.asarray(
        [[-0.5, -0.5, 5.0], [0.5, -0.5, 5.0], [0.5, 0.5, 5.0], [-0.5, 0.5, 5.0]],
        dtype=np.float32,
    )
    count = len(xyz)
    splat = SplatData(
        count=count,
        xyz=xyz,
        opacity=np.ones(count, dtype=np.float32),
        f_dc=np.zeros((count, 3), dtype=np.float32),
        scales=np.full((count, 3), -4.0, dtype=np.float32),
        quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        properties=(),
    )
    return write_standard_3dgs_ply(splat, path)


def _record(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _provider_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/adp_gaussian_excision_provider_runner.py"
    spec = importlib.util.spec_from_file_location(
        "adp_gaussian_excision_provider_runner_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provider_runner_camera_and_zone_conversion_are_exact(tmp_path: Path) -> None:
    runner = _provider_runner_module()
    camera = _camera("front", 0.0, 0.0)
    parameters = runner.camera_parameters(camera)
    expected = np.linalg.inv(np.asarray(camera["T_world_camera_opencv"]))
    assert np.allclose(parameters["R"], expected[:3, :3].T)
    assert np.allclose(parameters["T"], expected[:3, 3])
    mask_root = tmp_path / "masks"
    mask_root.mkdir()
    zones = {
        "protected": np.asarray([[255, 0], [0, 0]], dtype=np.uint8),
        "target_core": np.asarray([[0, 255], [0, 0]], dtype=np.uint8),
        "uncertain": np.asarray([[0, 0], [255, 255]], dtype=np.uint8),
    }
    for name, values in zones.items():
        assert cv2.imwrite(str(mask_root / f"front.{name}.png"), values)
    assert runner.load_class_labels(mask_root, "front").tolist() == [
        [0.0, 1.0],
        [2.0, 2.0],
    ]


def test_provider_runner_accepts_provider_frame_camera_alias() -> None:
    runner = _provider_runner_module()
    camera = _camera("front", 0.4, 12.0)
    provider_camera = dict(camera)
    provider_camera["T_world_camera_provider_frame"] = provider_camera.pop(
        "T_world_camera_opencv"
    )

    expected = runner.camera_parameters(camera)
    actual = runner.camera_parameters(provider_camera)

    assert np.array_equal(actual["R"], expected["R"])
    assert np.array_equal(actual["T"], expected["T"])
    assert actual["FoVx"] == expected["FoVx"]
    assert actual["FoVy"] == expected["FoVy"]


def test_provider_runner_rejects_conflicting_camera_aliases() -> None:
    runner = _provider_runner_module()
    camera = _camera("front", 0.0, 0.0)
    conflicting = dict(camera)
    conflicting_transform = np.asarray(camera["T_world_camera_opencv"]).copy()
    conflicting_transform[0, 3] = 1.0
    conflicting["T_world_camera_provider_frame"] = conflicting_transform.tolist()

    with pytest.raises(
        ValueError, match="gaussian_excision_camera_transform_alias_conflict"
    ):
        runner.camera_parameters(conflicting)


def test_provider_runner_failure_diagnostics_are_stable_and_sanitized() -> None:
    runner = _provider_runner_module()

    stable = runner._failure_diagnostics(
        ValueError("gaussian_excision_camera_transform_missing")
    )
    unsafe = runner._failure_diagnostics(ValueError("token=secret value /tmp/input"))

    assert stable["failure_type"] == "ValueError"
    assert stable["failure_code"] == "gaussian_excision_camera_transform_missing"
    assert stable["failure_message_sha256"].startswith("sha256:")
    assert unsafe["failure_code"] is None
    assert "secret" not in json.dumps(unsafe)


def test_provider_runner_accepts_exact_legacy_camera_partition_without_counts() -> None:
    runner = _provider_runner_module()
    camera_ids = [
        "front_medium",
        "front_working",
        "left_translate",
        "right_translate",
        "far_left",
        "far_right",
        "raised_left",
        "low_right",
    ]
    camera_split = {
        "calibration_camera_ids": [
            "front_medium",
            "front_working",
            "left_translate",
            "low_right",
            "raised_left",
            "right_translate",
        ],
        "heldout_camera_ids": ["far_left", "far_right"],
    }

    calibration = runner.validated_camera_split(
        camera_split, {camera_id: {} for camera_id in camera_ids}
    )

    assert calibration == camera_split["calibration_camera_ids"]


def test_provider_runner_rejects_present_camera_count_mismatch() -> None:
    runner = _provider_runner_module()
    camera_split = {
        "calibration_camera_ids": ["front", "side"],
        "heldout_camera_ids": ["heldout"],
        "camera_count": 0,
    }

    with pytest.raises(ValueError, match="gaussian_excision_camera_split_invalid"):
        runner.validated_camera_split(
            camera_split, {camera_id: {} for camera_id in ("front", "side", "heldout")}
        )


def test_provider_runner_import_preflight_reports_full_missing_set(
    tmp_path: Path,
) -> None:
    runner = _provider_runner_module()
    attempted = []

    def importer(module_name: str) -> object:
        attempted.append(module_name)
        if module_name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'", name="cv2")
        if module_name == "simple_knn._C":
            raise ModuleNotFoundError("No module named 'simple_knn'", name="simple_knn")
        if module_name == "scene.gaussian_model":
            raise RuntimeError("fixture compiled extension incompatibility")
        return object()

    result = runner.runtime_import_preflight(
        source_dir=tmp_path,
        importer=importer,
    )

    assert attempted == list(runner.RUNTIME_IMPORT_MODULES)
    assert result["all_imports_attempted"] is True
    assert result["failed_import_count"] == 3
    assert result["failed_modules"] == [
        "cv2",
        "simple_knn._C",
        "scene.gaussian_model",
    ]
    assert result["missing_module_names"] == ["cv2", "simple_knn"]
    assert result["blockers"] == ["gaussian_excision_runtime_import_closure_incomplete"]


def test_gaussian_excision_runtime_closure_pins_all_released_dependencies() -> None:
    assert excision_vast.EXPECTED_SUBMODULES[excision_vast.SIMPLE_KNN_PATH] == (
        excision_vast.SIMPLE_KNN_COMMIT
    )
    entrypoint = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp_gaussian_excision_provider_runtime.sh"
    ).read_text(encoding="utf-8")
    assert "submodules/simple-knn" in entrypoint
    assert "opencv-python-headless==4.11.0.86" in entrypoint
    assert "download.pytorch.org" not in entrypoint
    assert "PIP_NO_INDEX=1" in entrypoint
    assert "--find-links" in entrypoint
    assert excision_vast.DEFAULT_IMAGE.startswith("docker.io/pytorch/pytorch@sha256:")


def test_gaussian_excision_provider_identity_is_distinct_per_frozen_target() -> None:
    first = excision_vast.gaussian_excision_lane_identity(
        {
            "scene": {"publisher_scene_id": "840920", "target_instance_id": "165"},
            "freeze_digest": "sha256:" + "1" * 64,
        }
    )
    second = excision_vast.gaussian_excision_lane_identity(
        {
            "scene": {"publisher_scene_id": "840920", "target_instance_id": "385"},
            "freeze_digest": "sha256:" + "2" * 64,
        }
    )

    assert first["object_store_key_prefix"] != second["object_store_key_prefix"]
    assert first["instance_label_prefix"] != second["instance_label_prefix"]
    assert first["lane_id"].startswith("840920-165-")


def test_provider_output_recognizes_gaussian_excision_result(tmp_path: Path) -> None:
    output_zip = tmp_path / "gaussian-excision-output.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "adp009b_gaussian_excision_result.json",
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["gaussian_excision_runtime_import_closure_incomplete"],
                }
            ),
        )

    inspection = inspect_provider_runtime_output_zip(output_zip)
    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_status"] == "blocked"
    assert inspection["runtime_result"]["blockers"] == [
        "gaussian_excision_runtime_import_closure_incomplete"
    ]


def _prepared_excision_bundle(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("provider_runtime/fixture", "fixture")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "status": "ready",
        "provider_bundle_kind": excision_vast.PROVIDER_BUNDLE_KIND,
        "bundle_path": str(path),
        "bundle_sha256": digest,
        "blueprint_commit": "a" * 40,
        "execution_authority_digest": "sha256:" + "2" * 64,
        "freeze_digest": "sha256:" + "1" * 64,
        "hard_cap_usd": 1.5,
        "hard_ttl_seconds": 3600,
        "dependency_wheelhouse_manifest_digest": "sha256:" + "3" * 64,
        "provider_network_dependency_install_required": False,
        "exact_bundle_entrypoint_rehearsal": {
            "status": "passed",
            "bundle_sha256": digest,
            "entrypoint_relative_path": (
                "run_adp_gaussian_excision_provider_runtime.sh"
            ),
            "returncode": 0,
            "gpu_runtime_started": False,
            "paid_inference_performed": False,
            "provider_mutations_performed": 0,
        },
    }


def _paid_attempt_authority(
    bundle: Mapping[str, object],
    *,
    ordinal: int = 1,
    previous_attempt_receipt_digest: str | None = None,
    external_instance_allowlist: list[int] | None = None,
) -> dict[str, object]:
    authority: dict[str, object] = {
        "schema_version": excision_vast.PAID_ATTEMPT_AUTHORITY_SCHEMA,
        "authority_kind": "explicit_user_direction_in_current_goal",
        "authority_reference": "fixture-explicit-user-authority",
        "authorized_by": "fixture-user",
        "authorized_on": "2026-08-10",
        "purpose": str(
            bundle.get("execution_purpose")
            or "released_code_gaussian_ownership_audit"
        ),
        "provider": "vast",
        "paid_compute_authorized": True,
        "parent_execution_authority_digest": bundle["execution_authority_digest"],
        "freeze_digest": bundle["freeze_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "corrective_blueprint_commit": bundle["blueprint_commit"],
        "paid_attempt_ordinal": ordinal,
        "previous_attempt_receipt_digest": previous_attempt_receipt_digest,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
        "automatic_paid_retry_authorized": False,
        "hard_attempt_spend_cap_usd": bundle["hard_cap_usd"],
        "maximum_single_resource_ttl_seconds": bundle["hard_ttl_seconds"],
        "external_instance_allowlist": external_instance_allowlist or [],
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    return authority


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _bound_file_record(
    path: Path, *, receipt_digest: object | None = None
) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if receipt_digest is not None:
        record["receipt_digest"] = receipt_digest
    return record


def _prior_gaussian_spend_reconciliation(
    tmp_path: Path, previous_path: Path
) -> Path:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    instance_id = 123
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "PASS",
        "continuing_spend_from_this_run": False,
        "vast_instance_ids": [instance_id],
    }
    teardown_path = tmp_path / "prior-teardown.json"
    _write_json(teardown_path, teardown)
    provider_zero: dict[str, object] = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
    }
    provider_zero["receipt_digest"] = canonical_digest(
        provider_zero, digest_field="receipt_digest"
    )
    provider_zero_path = tmp_path / "prior-provider-zero.json"
    _write_json(provider_zero_path, provider_zero)
    billing = {
        "schema_version": "vast.official.charges.v1",
        "results": [{"source": f"instance-{instance_id}", "amount": 0.0}],
    }
    billing_path = tmp_path / "prior-official-billing.json"
    _write_json(billing_path, billing)
    billing_source: dict[str, object] = {
        "schema_version": "blueprint.provider_billing_source_receipt.v1",
        "status": "reconciled",
        "sources": [
            {
                "provider": "vast",
                "retained_path": str(billing_path.resolve()),
                "response_digest": _bound_file_record(billing_path)["sha256"],
                "response_size_bytes": billing_path.stat().st_size,
            }
        ],
    }
    billing_source["receipt_digest"] = canonical_digest(
        billing_source, digest_field="receipt_digest"
    )
    billing_source_path = tmp_path / "prior-billing-source.json"
    _write_json(billing_source_path, billing_source)
    sources = [
        {
            "role": "terminal_result",
            "schema_version": previous["schema_version"],
            "digest_field": "receipt_digest",
            "record": _bound_file_record(
                previous_path, receipt_digest=previous["receipt_digest"]
            ),
        },
        {
            "role": "teardown_manifest",
            "schema_version": teardown["schema_version"],
            "digest_field": None,
            "legacy_digest_gap": (
                "exact_source_bytes_sha256_bound_no_canonical_digest"
            ),
            "record": _bound_file_record(teardown_path),
        },
        {
            "role": "provider_zero",
            "schema_version": provider_zero["schema_version"],
            "digest_field": "receipt_digest",
            "record": _bound_file_record(
                provider_zero_path,
                receipt_digest=provider_zero["receipt_digest"],
            ),
        },
        {
            "role": "official_billing_response",
            "schema_version": billing["schema_version"],
            "digest_field": None,
            "legacy_digest_gap": (
                "exact_source_bytes_sha256_bound_no_canonical_digest"
            ),
            "record": _bound_file_record(billing_path),
        },
        {
            "role": "provider_billing_source_receipt",
            "schema_version": billing_source["schema_version"],
            "digest_field": "receipt_digest",
            "record": _bound_file_record(
                billing_source_path,
                receipt_digest=billing_source["receipt_digest"],
            ),
        },
    ]
    bindings = [
        {
            "kind": "cost_usd",
            "source_role": "official_billing_response",
            "json_path": ["results", 0, "amount"],
            "expected_value": 0.0,
        },
        {
            "kind": "continuing_spend",
            "source_role": "terminal_result",
            "json_path": ["continuing_spend_from_this_run"],
            "expected_value": False,
        },
        {
            "kind": "instance_id",
            "source_role": "official_billing_response",
            "json_path": ["results", 0, "source"],
            "expected_value": f"instance-{instance_id}",
        },
        {
            "kind": "authority_digest",
            "source_role": "terminal_result",
            "json_path": ["authorization_consumption", "authorization_digest"],
            "expected_value": previous["authorization_consumption"][
                "authorization_digest"
            ],
        },
        {
            "kind": "provider_zero",
            "source_role": "provider_zero",
            "json_path": ["provider_zero_verified"],
            "expected_value": True,
        },
        {
            "kind": "bundle_sha256",
            "source_role": "terminal_result",
            "json_path": ["bundle_sha256"],
            "expected_value": previous["bundle_sha256"],
        },
    ]
    entry: dict[str, object] = {
        "schema_version": "adp_same_goal_spend_entry.v1",
        "goal_id": "arm-decision-proof-v1",
        "attempt_id": "gaussian-excision-prior-1",
        "lane": "gaussian_excision",
        "evidence_kind": "fully_bound_official_billing",
        "provider_instance_id": instance_id,
        "cost_usd": 0.0,
        "authority_digest": previous["authorization_consumption"][
            "authorization_digest"
        ],
        "bundle_sha256": previous["bundle_sha256"],
        "continuing_spend_from_this_run": False,
        "provider_zero_confirmed": True,
        "source_receipts": sources,
        "bindings": bindings,
    }
    entry["entry_digest"] = canonical_digest(entry, digest_field="entry_digest")
    reconciliation: dict[str, object] = {
        "schema_version": "adp_same_goal_spend_reconciliation.v1",
        "status": "all_same_goal_paid_attempts_terminal_and_provider_zero",
        "goal_id": "arm-decision-proof-v1",
        "entries": [entry],
        "entry_count": 1,
        "total_cost_usd": 0.0,
    }
    reconciliation["receipt_digest"] = canonical_digest(
        reconciliation, digest_field="receipt_digest"
    )
    reconciliation_path = tmp_path / "prior-spend-reconciliation.json"
    _write_json(reconciliation_path, reconciliation)
    return reconciliation_path


def test_paid_attempt_authority_binds_the_exact_external_instance_allowlist(
    tmp_path: Path,
) -> None:
    bundle = _prepared_excision_bundle(tmp_path)
    authority = _paid_attempt_authority(
        bundle, external_instance_allowlist=[17, 23]
    )

    validated = excision_vast.validate_gaussian_excision_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=None,
        allowed_active_instance_ids=[23, 17],
    )
    assert validated["external_instance_allowlist"] == [17, 23]

    with pytest.raises(ValueError, match="external_instance_allowlist_mismatch"):
        excision_vast.validate_gaussian_excision_paid_attempt_authority(
            authority,
            prepared_bundle=bundle,
            previous_attempt_receipt=None,
            allowed_active_instance_ids=[17],
        )


def test_paid_attempt_authority_binds_same_goal_concurrent_instances(
    tmp_path: Path,
) -> None:
    bundle = _prepared_excision_bundle(tmp_path)
    authority = _paid_attempt_authority(bundle)
    authority.pop("external_instance_allowlist")
    authority.update(
        {
            "active_instance_allowlist": {
                "external_provider_owned": [17],
                "same_goal_concurrent": [23],
            },
            "concurrent_goal_id": "fixture-bounded-objects",
            "same_goal_concurrent_members": [
                {
                    "instance_id": 23,
                    "paid_attempt_authority_digest": "sha256:" + "a" * 64,
                }
            ],
        }
    )
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    excision_vast.validate_gaussian_excision_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=None,
        allowed_active_instance_ids=[23, 17],
    )

    authority["same_goal_concurrent_members"] = []
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    with pytest.raises(
        ValueError, match="same_goal_concurrent_allowlist_metadata_invalid"
    ):
        excision_vast.validate_gaussian_excision_paid_attempt_authority(
            authority,
            prepared_bundle=bundle,
            previous_attempt_receipt=None,
            allowed_active_instance_ids=[23, 17],
        )


def test_paid_attempt_authority_accepts_segment_contribution_sweep_purpose(
    tmp_path: Path,
) -> None:
    bundle = _prepared_excision_bundle(tmp_path)
    bundle["execution_purpose"] = "released_code_segment_contribution_sweep"
    authority = _paid_attempt_authority(bundle)

    validated = excision_vast.validate_gaussian_excision_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=None,
    )

    assert validated["purpose"] == "released_code_segment_contribution_sweep"


def test_first_attempt_accepts_reconciled_same_lane_spend_from_other_freeze(
    tmp_path: Path,
) -> None:
    bundle = _prepared_excision_bundle(tmp_path)
    prior_result: dict[str, object] = {
        "schema_version": "adp009b_gaussian_excision_vast_run.v1",
        "status": "completed",
        "continuing_spend_from_this_run": False,
        "estimated_cost_usd": 0.025,
        "authorization_consumption": {
            "authorization_digest": "sha256:" + "9" * 64,
        },
        "bundle_sha256": "sha256:" + "8" * 64,
        "freeze_digest": "sha256:" + "7" * 64,
    }
    prior_result["receipt_digest"] = canonical_digest(
        prior_result, digest_field="receipt_digest"
    )
    prior_result_path = tmp_path / "task-a-terminal-result.json"
    _write_json(prior_result_path, prior_result)
    reconciliation_path = _prior_gaussian_spend_reconciliation(
        tmp_path, prior_result_path
    )
    binding = bind_lane_prior_spend(
        prior_result_paths=[prior_result_path],
        reconciliation_path=reconciliation_path,
        lane="gaussian_excision",
    )
    authority = _paid_attempt_authority(bundle)
    authority.update(
        {
            "prior_terminal_attempts": binding["prior_terminal_attempts"],
            "prior_spend_reconciliation": binding["reconciliation"],
            "prior_actual_provider_spend_usd": binding["actual_total_usd"],
        }
    )
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    validated = excision_vast.validate_gaussian_excision_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=None,
    )

    assert validated["paid_attempt_ordinal"] == 1
    assert validated["prior_actual_provider_spend_usd"] == 0.0


def test_gaussian_excision_vast_dry_run_is_zero_mutation(tmp_path: Path) -> None:
    result = excision_vast.run_gaussian_excision_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=_prepared_excision_bundle(tmp_path),
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0


def test_corrected_paid_attempt_authority_binds_legacy_terminal_receipt(
    tmp_path: Path,
) -> None:
    bundle = _prepared_excision_bundle(tmp_path)
    previous: dict[str, object] = {
        "schema_version": "adp_gaussian_excision_attempt_receipt.v1",
        "status": "sealed_blocked_attempt",
        "freeze_digest": bundle["freeze_digest"],
        "bundle_sha256": bundle["bundle_sha256"],
        "estimated_cost_usd": 0.5,
        "retry_cap": 0,
        "continuing_spend": False,
        "continuing_spend_from_this_run": False,
        "provider_absence_confirmed": True,
        "authorization_consumption": {
            "authorization_digest": "sha256:" + "9" * 64,
        },
    }
    previous["receipt_digest"] = canonical_digest(
        previous, digest_field="receipt_digest"
    )
    authority = _paid_attempt_authority(
        bundle,
        ordinal=2,
        previous_attempt_receipt_digest=str(previous["receipt_digest"]),
    )
    previous_path = tmp_path / "previous-attempt.json"
    _write_json(previous_path, previous)
    reconciliation_path = _prior_gaussian_spend_reconciliation(
        tmp_path, previous_path
    )
    binding = bind_lane_prior_spend(
        prior_result_paths=[previous_path],
        reconciliation_path=reconciliation_path,
        lane="gaussian_excision",
    )
    authority.update(
        {
            "prior_terminal_attempts": binding["prior_terminal_attempts"],
            "prior_spend_reconciliation": binding["reconciliation"],
            "prior_actual_provider_spend_usd": binding["actual_total_usd"],
        }
    )
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )

    validated = excision_vast.validate_gaussian_excision_paid_attempt_authority(
        authority,
        prepared_bundle=bundle,
        previous_attempt_receipt=previous,
    )

    assert validated["paid_attempt_ordinal"] == 2
    with pytest.raises(ValueError, match="previous_attempt_receipt_missing"):
        excision_vast.validate_gaussian_excision_paid_attempt_authority(
            authority,
            prepared_bundle=bundle,
            previous_attempt_receipt=None,
        )

    invalid_schema = dict(previous)
    invalid_schema["schema_version"] = "unrelated_terminal_receipt.v1"
    invalid_schema["receipt_digest"] = canonical_digest(
        invalid_schema, digest_field="receipt_digest"
    )
    invalid_authority = _paid_attempt_authority(
        bundle,
        ordinal=2,
        previous_attempt_receipt_digest=str(invalid_schema["receipt_digest"]),
    )
    with pytest.raises(ValueError, match="previous_attempt_receipt_schema_invalid"):
        excision_vast.validate_gaussian_excision_paid_attempt_authority(
            invalid_authority,
            prepared_bundle=bundle,
            previous_attempt_receipt=invalid_schema,
        )


def test_paid_attempt_authority_consumption_is_single_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _prepared_excision_bundle(tmp_path)
    authority = _paid_attempt_authority(bundle)
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str((tmp_path / "consumed").parent))

    first = excision_vast.consume_gaussian_excision_paid_attempt_authority_once(
        authority, blueprint_commit=str(bundle["blueprint_commit"])
    )
    second = excision_vast.consume_gaussian_excision_paid_attempt_authority_once(
        authority, blueprint_commit=str(bundle["blueprint_commit"])
    )

    assert first["status"] == "consumed"
    assert second == {
        "status": "blocked",
        "blockers": ["gaussian_excision_paid_attempt_authority_consumed"],
    }


@pytest.mark.parametrize("current_watchdog_handoff", [False, True])
def test_attempt_and_recovery_receipts_join_files_without_upgrading_claims(
    tmp_path: Path, current_watchdog_handoff: bool
) -> None:
    bundle = {
        "provider_bundle_kind": excision_vast.PROVIDER_BUNDLE_KIND,
        "blueprint_commit": "a" * 40,
        "bundle_sha256": "sha256:" + "1" * 64,
        "freeze_digest": "sha256:" + "2" * 64,
    }
    run = {
        "status": "blocked",
        "bundle_sha256": bundle["bundle_sha256"],
        "estimated_cost_usd": 0.01,
        "retry_cap": 0,
        "paid_attempt_ordinal": 1,
        "attempt_authority_digest": "sha256:" + "4" * 64,
        "previous_attempt_receipt_digest": None,
        "authorization_consumption": {
            "status": "consumed",
            "authorization_digest": "sha256:" + "4" * 64,
        },
    }
    execution = {
        "status": "blocked",
        "blockers": ["fixture_runtime_blocked"],
        "released_code_executed": False,
        "heldout_cameras_accessed_for_classification": False,
    }
    teardown = {
        "status": "completed",
        "vast_instance_ids": [7],
        "continuing_spend_from_this_run": False,
    }
    watchdog: dict[str, object] = {
        "status": "provider_terminal",
        "provider_absence_confirmed": True,
    }
    if current_watchdog_handoff:
        watchdog["instance_ids"] = [7]
    else:
        watchdog["recorded_vast_instance_teardown"] = {
            "instance_id": "7",
            "provider_absence_confirmed": True,
        }
    payloads = {
        "bundle.json": bundle,
        "run.json": run,
        "execution.json": execution,
        "teardown.json": teardown,
        "watchdog.json": watchdog,
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    attempt = excision_vast.materialize_gaussian_excision_attempt_receipt(
        evidence_root=tmp_path,
        bundle_receipt_path=tmp_path / "bundle.json",
        run_result_path=tmp_path / "run.json",
        execution_result_path=tmp_path / "execution.json",
        teardown_manifest_path=tmp_path / "teardown.json",
        watchdog_evidence_path=tmp_path / "watchdog.json",
        output_path=tmp_path / "attempt.json",
    )

    assert attempt["status"] == "sealed_blocked_attempt"
    assert attempt["paid_attempt_ordinal"] == 1
    assert attempt["authorization_consumption"]["status"] == "consumed"
    assert attempt["provider_absence_confirmed"] is True
    assert attempt["proof_boundaries"]["gaussian_ownership_qualified"] is False
    assert attempt["proof_boundaries"]["policy_outcome_available"] is False

    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[1]
            / "scripts/materialize_gaussian_excision_attempt_receipt.py"
        ),
        "--evidence-root",
        str(tmp_path),
        "--bundle-receipt",
        str(tmp_path / "bundle.json"),
        "--run-result",
        str(tmp_path / "run.json"),
        "--execution-result",
        str(tmp_path / "execution.json"),
        "--teardown-manifest",
        str(tmp_path / "teardown.json"),
        "--watchdog-evidence",
        str(tmp_path / "watchdog.json"),
        "--output",
        str(tmp_path / "attempt-cli.json"),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json.loads(completed.stdout)["status"] == "sealed"
    cli_attempt = json.loads((tmp_path / "attempt-cli.json").read_text())
    assert cli_attempt["receipt_digest"] == attempt["receipt_digest"]

    dependency = {
        "schema_version": excision_vast.DEPENDENCY_WHEELHOUSE_SCHEMA,
        "status": "ready",
        "provider_network_install_required": False,
        "manifest_digest": "sha256:" + "3" * 64,
    }
    bundle.update(
        {
            "status": "ready",
            "container_image": excision_vast.DEFAULT_IMAGE,
            "dependency_wheelhouse_manifest_digest": dependency["manifest_digest"],
            "provider_network_dependency_install_required": False,
            "exact_bundle_entrypoint_rehearsal": {"status": "passed"},
        }
    )
    admission = {
        "status": "admitted",
        "allocation_binding": {
            "bundle_sha256": bundle["bundle_sha256"],
            "orchestrator_source_commit": bundle["blueprint_commit"],
        },
    }
    dry_run = {
        "status": "dry_run_ready",
        "provider_mutations_performed": 0,
        "retry_cap": 0,
    }
    for name, payload in {
        "dependency.json": dependency,
        "repaired-bundle.json": bundle,
        "admission.json": admission,
        "dry-run.json": dry_run,
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    readiness = excision_vast.materialize_gaussian_excision_recovery_readiness(
        evidence_root=tmp_path,
        dependency_manifest_path=tmp_path / "dependency.json",
        bundle_receipt_path=tmp_path / "repaired-bundle.json",
        admission_path=tmp_path / "admission.json",
        dry_run_result_path=tmp_path / "dry-run.json",
        output_path=tmp_path / "readiness.json",
    )

    assert readiness["status"] == "ready_for_new_authority_not_executed"
    assert readiness["proof_boundaries"]["gpu_runtime_executed"] is False
    assert readiness["proof_boundaries"]["new_paid_authority_required"] is True

    scene_freeze = {"selected_scene_id": "fixture"}
    scene_freeze["scene_freeze_digest"] = canonical_digest(
        scene_freeze, digest_field="scene_freeze_digest"
    )
    task_freeze = {
        "scene_freeze_digest": scene_freeze["scene_freeze_digest"],
        "task_id": "fixture_task",
    }
    task_freeze["task_freeze_digest"] = canonical_digest(
        task_freeze, digest_field="task_freeze_digest"
    )
    excision_freeze = {
        "scene": {"publisher_scene_id": "fixture", "task_id": "fixture_task"}
    }
    excision_freeze["freeze_digest"] = canonical_digest(
        excision_freeze, digest_field="freeze_digest"
    )
    # The fixture attempt/readiness pair is rebound to this exact excision
    # freeze before exercising the task-level join.
    for path in (tmp_path / "attempt.json", tmp_path / "readiness.json"):
        payload = json.loads(path.read_text())
        payload["freeze_digest"] = excision_freeze["freeze_digest"]
        payload["receipt_digest"] = canonical_digest(
            payload, digest_field="receipt_digest"
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
    for name, payload in {
        "scene-freeze.json": scene_freeze,
        "task-freeze.json": task_freeze,
        "excision-freeze.json": excision_freeze,
    }.items():
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    abstention = excision_vast.materialize_gaussian_excision_task_abstention(
        scene_freeze_path=tmp_path / "scene-freeze.json",
        task_freeze_path=tmp_path / "task-freeze.json",
        excision_freeze_path=tmp_path / "excision-freeze.json",
        attempt_receipt_path=tmp_path / "attempt.json",
        recovery_readiness_path=tmp_path / "readiness.json",
        output_path=tmp_path / "abstention.json",
    )
    assert abstention["status"] == "typed_evidence_backed_abstention"
    assert abstention["controls_executed"] is False
    assert abstention["automatic_paid_retry_executed"] is False

    inventory = {
        "instances": [
            {
                "provider": "vast",
                "id": "99",
                "name": "external-lane",
                "state": "running",
                "live": True,
                "cost_per_hr_usd": 0.5,
            }
        ]
    }
    (tmp_path / "inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    closeout = excision_vast.materialize_gaussian_excision_provider_closeout(
        evidence_root=tmp_path,
        attempt_receipt_paths=[tmp_path / "attempt.json"],
        provider_inventory_path=tmp_path / "inventory.json",
        output_path=tmp_path / "closeout.json",
    )
    assert closeout["status"] == "lane_owned_provider_zero"
    assert closeout["global_provider_zero_claimed"] is False
    assert closeout["external_live_instances"][0]["charged_to_this_lane"] is False


def test_canonical_allocator_binds_gaussian_excision_bundle(monkeypatch, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"immutable-gaussian-excision-runtime")
    bundle_digest = "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    receipt = {
        "status": "ready",
        "provider_bundle_kind": excision_vast.PROVIDER_BUNDLE_KIND,
        "container_image": excision_vast.DEFAULT_IMAGE,
        "blueprint_commit": "a" * 40,
        "released_code": {
            "tree": excision_vast.SOURCE_TREE,
            "source_modified": False,
        },
        "hard_cap_usd": 1.5,
        "hard_ttl_seconds": 3600,
        "maximum_paid_attempts": 1,
        "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True,
        "dependency_wheelhouse_manifest_digest": "sha256:" + "3" * 64,
        "provider_network_dependency_install_required": False,
        "raw_interiorgs_downloaded_bytes_included": False,
        "private_scene_derived_standard_splat_included": True,
        "freeze_digest": "sha256:" + "1" * 64,
        "execution_authority_digest": "sha256:" + "2" * 64,
        "blockers": [],
        "bundle_path": str(bundle),
        "bundle_sha256": bundle_digest,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    avoidlist_path = tmp_path / "machine-avoidlist.json"
    avoidlist_path.write_text(
        json.dumps(
            {
                "schema_version": "vast_machine_avoidlist.v1",
                "machine_ids": [8207],
            }
        ),
        encoding="utf-8",
    )
    avoidlist_digest = "sha256:" + hashlib.sha256(avoidlist_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            [],
            {"orchestrator_source_commit": "a" * 40, "checkout_clean": True},
        ),
    )
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_gaussian_excision_vast", fake_run)
    arguments = [
        "gpu-canary",
        "--probe-kind",
        excision_vast.PROBE_KIND,
        "--provider",
        "vast",
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--expected-source-commit",
        "a" * 40,
        "--adp-gaussian-excision-bundle-receipt",
        str(receipt_path),
        "--adp-job-dir",
        str(tmp_path / "run"),
        "--adp-max-hourly-rate-usd",
        "0.60",
        "--adp-max-spend-usd",
        "1.50",
        "--adp-hard-ttl-seconds",
        "3600",
        "--adp-machine-avoidlist",
        str(avoidlist_path),
    ]

    assert allocator.main(arguments) == 0
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["status"] == "admitted"
    assert admission["allocation_binding"]["bundle_sha256"] == bundle_digest
    assert admission["allocation_binding"]["machine_avoidlist_sha256"] == avoidlist_digest
    assert admission["heldout_cameras_accessed_for_classification"] is False
    assert observed["execute"] is False
    assert observed["machine_avoidlist_path"] == avoidlist_path

    assert allocator.main([*arguments, "--execute"]) == 2
    blocked = json.loads((tmp_path / "adapter.json").read_text())
    blocked_admission = json.loads((tmp_path / "admission.json").read_text())
    assert "gaussian_excision_paid_attempt_authority_missing" in blocked_admission[
        "blockers"
    ]
    assert blocked["provider_mutations_performed"] == 0

    attempt_authority = _paid_attempt_authority(receipt)
    attempt_authority_path = tmp_path / "attempt-authority.json"
    attempt_authority_path.write_text(
        json.dumps(attempt_authority), encoding="utf-8"
    )
    assert (
        allocator.main(
            [
                *arguments,
                "--execute",
                "--adp-gaussian-excision-attempt-authority",
                str(attempt_authority_path),
            ]
        )
        == 0
    )
    assert observed["execute"] is True
    assert observed["paid_attempt_authority"] == attempt_authority
    assert observed["previous_attempt_receipt"] is None


def test_live_gaussian_excision_run_arms_watchdog_and_closes_resources(
    monkeypatch, tmp_path: Path
) -> None:
    events = []
    started_path = tmp_path / "started_instance.txt"
    staging = tmp_path / "job/object_store_staging"

    def fake_stage(**kwargs):
        assert list((tmp_path / "consumed").glob("gaussian-excision-*.json"))
        events.append("stage")
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.com/private", encoding="utf-8")
        return {"status": "completed"}

    def fake_arm(**kwargs):
        events.append("watchdog")
        return {"status": "armed"}, SimpleNamespace(
            started_instance_id_path=started_path,
            # A real handle carries the prefix it armed on.
            pod_name_prefix=kwargs["pod_name_prefix"],
        )

    def fake_adapter(**kwargs):
        events.append("adapter")
        assert kwargs["provider_bundle_kind"] == "adp_gaussian_excision"
        assert kwargs["machine_avoidlist_path"] == tmp_path / "avoidlist.json"
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                "adp009b_gaussian_excision_result.json",
                json.dumps(
                    {
                        "status": "completed",
                        "blockers": [],
                        "released_code_executed": True,
                        "heldout_cameras_accessed_for_classification": False,
                        "provider_zero_required_after_return": True,
                        "depth_anything_3_used": False,
                        "contribution_manifest": {"relative_path": "manifest.json"},
                    }
                ),
            )
        (output_zip.parent / "vast_teardown_manifest.json").write_text(
            json.dumps({"vast_instance_ids": [7], "continuing_spend_from_this_run": False}),
            encoding="utf-8",
        )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.2}

    monkeypatch.setattr(excision_vast, "_remaining_minutes", lambda **kwargs: 60)
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str((tmp_path / "consumed").parent))
    monkeypatch.setattr(excision_vast, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(excision_vast, "arm_independent_vast_watchdog", fake_arm)
    monkeypatch.setattr(excision_vast, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        excision_vast,
        "cleanup_staged_wam_provider_objects",
        lambda value: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        excision_vast,
        "close_independent_vast_watchdog",
        lambda **kwargs: {"status": "provider_terminal"},
    )

    bundle = _prepared_excision_bundle(tmp_path)
    authority = _paid_attempt_authority(bundle)
    result = excision_vast.run_gaussian_excision_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=bundle,
        paid_attempt_authority=authority,
        machine_avoidlist_path=tmp_path / "avoidlist.json",
    )

    assert events == ["stage", "watchdog", "adapter"]
    assert result["status"] == "completed"
    assert result["continuing_spend_from_this_run"] is False
    assert result["authorization_consumption"]["status"] == "consumed"

    repeated = excision_vast.run_gaussian_excision_vast(
        job_dir=tmp_path / "second-job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=bundle,
        paid_attempt_authority=authority,
    )
    assert repeated["status"] == "blocked"
    assert repeated["provider_mutations_performed"] == 0
    assert repeated["blockers"] == [
        "gaussian_excision_paid_attempt_authority_consumed"
    ]
    assert events == ["stage", "watchdog", "adapter"]


def test_vast_adapter_preflights_and_routes_gaussian_excision_root_bundle(
    tmp_path: Path,
) -> None:
    camera_ids = ("calibration_left", "calibration_right", "heldout_front")
    bundle = tmp_path / "adp_gaussian_excision_provider_runtime_bundle.zip"
    source_zip = tmp_path / "flashsplat_source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("README.md", "fixture released source")
    repo = Path(__file__).resolve().parents[1]
    members: dict[str, bytes | str] = {
        "run_adp_gaussian_excision_provider_runtime.sh": (
            repo / "scripts/run_adp_gaussian_excision_provider_runtime.sh"
        ).read_text(encoding="utf-8"),
        "adp_gaussian_excision_provider_runner.py": (
            repo / "scripts/adp_gaussian_excision_provider_runner.py"
        ).read_text(encoding="utf-8"),
        "adp_gaussian_excision_provider_manifest.json": "{}",
        "execution_authority.json": "{}",
        "flashsplat_source.zip": source_zip.read_bytes(),
        "input/scene_standard.ply": ("ply\nformat binary_little_endian 1.0\nend_header\n"),
        "input/cameras.v1.json": "{}",
        "freeze/adp009b_gaussian_excision_audit_freeze.v1.json": json.dumps(
            {
                "camera_split": {
                    "calibration_camera_ids": list(camera_ids[:2]),
                    "heldout_camera_ids": list(camera_ids[2:]),
                }
            }
        ),
    }
    dependency_wheel = b"fixture-wheel"
    dependency_manifest = {
        "schema_version": "adp_gaussian_excision_dependency_wheelhouse.v1",
        "status": "ready",
        "provider_network_install_required": False,
        "wheels": [
            {
                "filename": "fixture-1.0-py3-none-any.whl",
                "size_bytes": len(dependency_wheel),
                "sha256": "sha256:"
                + hashlib.sha256(dependency_wheel).hexdigest(),
            }
        ],
    }
    members["dependency_wheelhouse_manifest.json"] = json.dumps(
        dependency_manifest
    )
    members["dependency_wheelhouse/fixture-1.0-py3-none-any.whl"] = (
        dependency_wheel
    )
    members.update(
        {
            f"freeze/masks/{camera_id}.{zone}.png": b"fixture"
            for camera_id in camera_ids
            for zone in ("target_core", "protected", "uncertain")
        }
    )
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)

    preflight = vast_adapter._blueprint_bundle_preflight(
        job_dir=tmp_path,
        generated_at="2026-08-09T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_gaussian_excision",
        bundle_path=bundle,
        provider_bundle_url="https://example.com/private-bundle",
        provider_output_put_url="https://example.com/private-output",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []
    assert preflight["missing_zip_entries"] == []

    probe = vast_adapter._probe_shell_script(
        "https://example.com/heartbeat",
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_gaussian_excision",
    )
    assert "adp_gaussian_excision_provider_bundle" in probe
    assert "run_adp_gaussian_excision_provider_runtime.sh" in probe
    assert "BLUEPRINT_ADP_GAUSSIAN_EXCISION_OUTPUT_DIR" in probe
    assert "run_wam_provider_runtime.sh" not in probe

    incomplete = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(incomplete, "w") as archive:
        for name, payload in sorted(members.items()):
            if name != "freeze/masks/heldout_front.protected.png":
                archive.writestr(name, payload)
    failed = vast_adapter._blueprint_bundle_preflight(
        job_dir=tmp_path / "incomplete-job",
        generated_at="2026-08-09T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_gaussian_excision",
        bundle_path=incomplete,
        provider_bundle_url="https://example.com/private-bundle",
        provider_output_put_url="https://example.com/private-output",
    )
    assert "provider_runtime_bundle_required_entries_missing" in failed["blockers"]
    assert failed["missing_zip_entries"] == ["freeze/masks/heldout_front.protected.png"]


def test_freeze_builds_independent_core_uncertain_and_protected_masks(
    monkeypatch, tmp_path: Path
) -> None:
    source = _splat(tmp_path / "scene.ply")
    collision = tmp_path / "collision.usda"
    collision.write_text(
        """#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Root"
{
    def Mesh "Target"
    {
        point3f[] points = [(-0.5, -0.5, 4.9), (0.5, -0.5, 4.9), (0.5, 0.5, 4.9), (-0.5, 0.5, 4.9), (-0.5, -0.5, 5.1), (0.5, -0.5, 5.1), (0.5, 0.5, 5.1), (-0.5, 0.5, 5.1)]
        int[] faceVertexCounts = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0]
    }
}
""",
        encoding="utf-8",
    )
    cameras = [
        _camera("front", 0.0, 0.0),
        _camera("near_left", -0.2, 1.0),
        _camera("near_right", 0.2, -1.0),
        _camera("far_left", -0.5, 3.0),
        _camera("far_right", 0.5, -3.0),
    ]
    camera_path = tmp_path / "cameras.json"
    camera_path.write_text(json.dumps(cameras), encoding="utf-8")
    image_root = tmp_path / "images"
    outer_root = tmp_path / "outer"
    image_root.mkdir()
    outer_root.mkdir()
    outer = np.zeros((48, 64), dtype=np.uint8)
    outer[8:40, 12:52] = 255
    for camera in cameras:
        camera_id = str(camera["camera_id"])
        assert cv2.imwrite(
            str(image_root / f"{camera_id}.png"),
            np.zeros((48, 64, 3), dtype=np.uint8),
        )
        assert cv2.imwrite(str(outer_root / f"{camera_id}.png"), outer)
    registered_frame = {
        "schema_version": "interiorgs_sage_shared_frame_candidate.v1",
        "status": "multi_object_identity_alignment_candidate",
        "shared_frame_status": "provider_declared_not_independently_validated",
        "metric_scale_status": "provider_declared_not_independently_validated",
        "provider_transform": {
            "source_to_collision": "identity",
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "handedness": "not_independently_proven",
        },
        "claim_boundary": {
            "independent_metric_metrology_completed": False,
            "handedness_independently_proven": False,
        },
    }
    registered_frame["receipt_digest"] = canonical_digest(
        registered_frame, digest_field="receipt_digest"
    )
    registered_frame_path = tmp_path / "registered-frame.json"
    registered_frame_path.write_text(
        canonical_json(registered_frame) + "\n", encoding="utf-8"
    )

    freeze = materialize_excision_audit_freeze(
        source_standard_splat_path=source,
        source_collision_path=collision,
        target_collision_prim_path="/Root/Target",
        registered_frame_receipt_path=registered_frame_path,
        camera_contract_path=camera_path,
        source_image_root=image_root,
        historical_outer_mask_root=outer_root,
        scene={
            "publisher_scene_id": "fixture",
            "target_instance_id": "target",
            "target_semantic_label": "refrigerator",
        },
        policy=POLICY,
        historical_baseline={
            "method": "center_inside_registered_target_aabb",
            "center_aabb_min_m": [-0.6, -0.6, 4.9],
            "center_aabb_max_m": [0.6, 0.6, 5.1],
            "selected_gaussian_count": 4,
        },
        output_root=tmp_path / "freeze",
    )

    assert freeze["schema_version"] == FREEZE_SCHEMA
    assert freeze["camera_split"]["outcome_fields_accessed"] is False
    assert len(freeze["camera_split"]["heldout_camera_ids"]) == 2
    assert len(freeze["camera_split"]["calibration_camera_ids"]) == 3
    assert freeze["camera_split"]["camera_count"] == 5
    assert freeze["camera_split"]["calibration_camera_count"] == 3
    assert freeze["camera_split"]["heldout_camera_count"] == 2
    assert freeze["camera_split"]["camera_split_digest"] == canonical_digest(
        freeze["camera_split"], digest_field="camera_split_digest"
    )
    assert freeze["registered_frame"]["shared_frame_status"] == (
        "provider_declared_not_independently_validated"
    )
    assert freeze["registered_frame"]["provider_transform"]["handedness"] == (
        "not_independently_proven"
    )
    assert freeze["scale_and_bounds"]["collision_stage_meters_per_unit"] == 1.0
    assert freeze["scale_and_bounds"]["source_gaussian_count"] == 4
    assert freeze["contribution_method"]["depth_anything_3_used"] is False
    assert freeze["historical_baseline"]["selected_gaussian_count"] == 4
    assert all(row["target_core_is_subset_of_historical_outer_mask"] for row in freeze["masks"])
    assert (tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json").is_file()

    calibration = freeze["camera_split"]["calibration_camera_ids"]
    evidence = np.zeros((len(calibration), len(CONTRIBUTION_CLASS_ORDER), 4))
    evidence[:, 1, 0] = 2.0
    evidence[:, 0, 1:] = 2.0
    gpu_root = tmp_path / "gpu"
    gpu_root.mkdir()
    repetitions = []
    for index in range(2):
        path = gpu_root / f"contribution_{index}.npz"
        np.savez_compressed(path, per_view_class_contribution=evidence)
        repetitions.append(_record(path, gpu_root))
    manifest = {
        "schema_version": CONTRIBUTION_EVIDENCE_SCHEMA,
        "freeze_digest": freeze["freeze_digest"],
        "class_order": list(CONTRIBUTION_CLASS_ORDER),
        "camera_ids": calibration,
        "method": {
            **freeze["contribution_method"],
            "released_code_executed": True,
        },
        "repetitions": repetitions,
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_path = gpu_root / f"{CONTRIBUTION_EVIDENCE_SCHEMA}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    receipt = materialize_excision_ownership(
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        contribution_manifest_path=manifest_path,
        source_standard_splat_path=source,
        output_root=tmp_path / "ownership",
    )
    replayed_receipt = materialize_excision_ownership(
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        contribution_manifest_path=manifest_path,
        source_standard_splat_path=source,
        output_root=tmp_path / "ownership-replay",
    )
    replay = materialize_excision_ownership_replay(
        ownership_receipt_paths=[
            tmp_path / "ownership" / f"{OWNERSHIP_RECEIPT_SCHEMA}.json",
            tmp_path / "ownership-replay" / f"{OWNERSHIP_RECEIPT_SCHEMA}.json",
        ],
        output_root=tmp_path / "replay-verification",
    )

    assert receipt["schema_version"] == OWNERSHIP_RECEIPT_SCHEMA
    assert receipt["ownership"] == {
        "source_gaussian_count": 4,
        "owned_count": 1,
        "retained_count": 3,
        "ambiguous_count": 0,
        "historical_obb_count": 4,
        "exhaustive": True,
        "pairwise_disjoint": True,
    }
    assert all(row["retained_rows_byte_exact"] is True for row in receipt["preservation"].values())
    assert replayed_receipt == receipt
    assert replay["schema_version"] == OWNERSHIP_REPLAY_SCHEMA
    assert replay["gate_passed"] is True
    assert replay["canonical_manifests_identical"] is True
    assert replay["receipt_files_byte_identical"] is True
    assert replay["output_digests_identical"] is True
    assert replay["index_sets_identical"] is True
    assert replay["protected_source_records_byte_identical"] is True

    jittered = evidence.copy()
    jittered[0, 1, 0] += 0.000002
    jittered_path = gpu_root / "contribution_1.npz"
    np.savez_compressed(jittered_path, per_view_class_contribution=jittered)
    manifest["repetitions"][1] = _record(jittered_path, gpu_root)
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    aggregation_policy_path = tmp_path / "aggregation-policy.json"
    aggregation_policy = materialize_excision_ownership_aggregation_policy(
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        contribution_manifest_path=manifest_path,
        output_path=aggregation_policy_path,
    )
    assert aggregation_policy["schema_version"] == OWNERSHIP_AGGREGATION_POLICY_SCHEMA
    assert aggregation_policy["heldout_cameras_accessed"] is False
    assert aggregation_policy["calibration_evidence_only"] is True

    aggregated = materialize_excision_ownership(
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        contribution_manifest_path=manifest_path,
        source_standard_splat_path=source,
        output_root=tmp_path / "ownership-aggregated",
        aggregation_policy_path=aggregation_policy_path,
    )

    assert aggregated["ownership"] == receipt["ownership"]
    assert aggregated["determinism"] == {
        "repetition_count": 2,
        "quantization_decimals": 6,
        "quantized_contribution_arrays_identical": False,
        "label_disagreement_count": 0,
        "aggregation_rule": "unanimous_owned_and_retained_else_ambiguous",
        "aggregation_policy_digest": aggregation_policy["aggregation_policy_digest"],
        "disputed_gaussians_forced_ambiguous": True,
    }

    authority = {
        "schema_version": excision_vast.AUTHORITY_SCHEMA,
        "purpose": "released_code_gaussian_ownership_audit",
        "publisher_scene_id": "fixture",
        "target_instance_id": "target",
        "freeze_digest": freeze["freeze_digest"],
        "private_scene_derived_standard_splat_upload_authorized": True,
        "paid_compute_authorized": True,
        "provider_zero_required_before_and_after": True,
        "teardown_required": True,
        "raw_interiorgs_downloaded_bytes_upload_authorized": False,
        "public_disclosure_authorized": False,
        "model_training_authorized": False,
        "automatic_paid_retry_authorized": False,
        "retention_policy": "bounded_to_goal_then_provider_zero",
        "hard_attempt_spend_cap_usd": 1.5,
        "maximum_single_resource_ttl_seconds": 3600,
        "maximum_paid_attempts": 1,
        "maximum_automatic_retries": 0,
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(canonical_json(authority) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        excision_vast, "_git", lambda *args: "" if args[-2:] == ("status", "--short") else "fixture"
    )
    monkeypatch.setattr(
        excision_vast,
        "_source_identity",
        lambda source: {
            "repository": excision_vast.SOURCE_REPOSITORY,
            "commit": excision_vast.SOURCE_COMMIT,
            "tree": excision_vast.SOURCE_TREE,
            "submodules": dict(excision_vast.EXPECTED_SUBMODULES),
            "source_modified": False,
        },
    )
    def write_fixture_source_archive(source: Path, destination: Path) -> None:
        del source
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("README.md", "fixture-source")

    monkeypatch.setattr(
        excision_vast,
        "_write_source_archive",
        write_fixture_source_archive,
    )
    wheelhouse = tmp_path / "dependency-wheelhouse"
    wheelhouse.mkdir()
    for distribution, version in excision_vast.DEPENDENCY_REQUIREMENTS.items():
        filename_distribution = distribution.replace("-", "_")
        wheel_path = wheelhouse / f"{filename_distribution}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(wheel_path, "w") as archive:
            archive.writestr(
                f"{filename_distribution}-{version}.dist-info/METADATA",
                f"Name: {distribution}\nVersion: {version}\n",
            )
    dependency_manifest_path = tmp_path / "dependency-wheelhouse.json"
    excision_vast.materialize_gaussian_excision_dependency_wheelhouse(
        wheelhouse_path=wheelhouse,
        manifest_path=dependency_manifest_path,
    )
    bundle = excision_vast.build_gaussian_excision_vast_bundle(
        repo_root=Path(__file__).resolve().parents[1],
        flashsplat_root=tmp_path,
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        source_standard_splat_path=source,
        camera_contract_path=camera_path,
        execution_authority_path=authority_path,
        dependency_wheelhouse_path=wheelhouse,
        dependency_manifest_path=dependency_manifest_path,
        job_dir=tmp_path / "bundle",
        generated_at="2026-08-09T00:00:00Z",
    )
    assert bundle["status"] == "ready"
    assert bundle["raw_interiorgs_downloaded_bytes_included"] is False
    assert bundle["private_scene_derived_standard_splat_included"] is True
    assert bundle["provider_network_dependency_install_required"] is False
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        assert "input/scene_standard.ply" in archive.namelist()
        assert "freeze/masks/front.target_core.png" in archive.namelist()
        assert "run_adp_gaussian_excision_provider_runtime.sh" in archive.namelist()
        assert "dependency_wheelhouse_manifest.json" in archive.namelist()
