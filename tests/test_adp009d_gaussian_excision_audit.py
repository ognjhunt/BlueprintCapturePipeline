from __future__ import annotations

import json
import hashlib
import importlib.util
from pathlib import Path
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline import adp_gaussian_excision_vast as excision_vast
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import vast_provider_adapter as vast_adapter
from blueprint_pipeline.public_scene_gaussian_excision_audit import (
    CONTRIBUTION_CLASS_ORDER,
    CONTRIBUTION_EVIDENCE_SCHEMA,
    FREEZE_SCHEMA,
    OWNERSHIP_RECEIPT_SCHEMA,
    classify_excision_ownership,
    materialize_excision_audit_freeze,
    materialize_excision_ownership,
    select_maximally_diverse_holdout_pair,
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

    first = select_maximally_diverse_holdout_pair(
        cameras, projected_target_fraction=fractions
    )
    second = select_maximally_diverse_holdout_pair(
        list(reversed(cameras)), projected_target_fraction=fractions
    )

    assert first == second
    assert first["heldout_camera_ids"] == ["far_left", "far_right"]
    assert len(first["calibration_camera_ids"]) == 6
    assert first["outcome_fields_accessed"] is False


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
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/adp_gaussian_excision_provider_runner.py"
    )
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


def _prepared_excision_bundle(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("provider_runtime/fixture", "fixture")
    return {
        "status": "ready",
        "provider_bundle_kind": excision_vast.PROVIDER_BUNDLE_KIND,
        "bundle_path": str(path),
        "bundle_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


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


def test_canonical_allocator_binds_gaussian_excision_bundle(
    monkeypatch, tmp_path: Path
) -> None:
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
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-gaussian-excision",
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
    ]

    assert allocator.main(arguments) == 0
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["status"] == "admitted"
    assert admission["allocation_binding"]["bundle_sha256"] == bundle_digest
    assert admission["heldout_cameras_accessed_for_classification"] is False
    assert observed["execute"] is False


def test_live_gaussian_excision_run_arms_watchdog_and_closes_resources(
    monkeypatch, tmp_path: Path
) -> None:
    events = []
    started_path = tmp_path / "started_instance.txt"
    staging = tmp_path / "job/object_store_staging"

    def fake_stage(**kwargs):
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text(
                "https://example.com/private", encoding="utf-8"
            )
        return {"status": "completed"}

    def fake_arm(**kwargs):
        events.append("watchdog")
        return {"status": "armed"}, SimpleNamespace(
            started_instance_id_path=started_path
        )

    def fake_adapter(**kwargs):
        events.append("adapter")
        assert kwargs["provider_bundle_kind"] == "adp_gaussian_excision"
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
            json.dumps(
                {"vast_instance_ids": [7], "continuing_spend_from_this_run": False}
            ),
            encoding="utf-8",
        )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.2}

    monkeypatch.setattr(excision_vast, "_remaining_minutes", lambda **kwargs: 60)
    monkeypatch.setattr(
        excision_vast, "stage_wam_provider_bundle_object_store", fake_stage
    )
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

    result = excision_vast.run_gaussian_excision_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=object(),  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=_prepared_excision_bundle(tmp_path),
    )

    assert events == ["watchdog", "adapter"]
    assert result["status"] == "completed"
    assert result["continuing_spend_from_this_run"] is False


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
        "input/scene_standard.ply": (
            "ply\nformat binary_little_endian 1.0\nend_header\n"
        ),
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
    assert failed["missing_zip_entries"] == [
        "freeze/masks/heldout_front.protected.png"
    ]


def test_freeze_builds_independent_core_uncertain_and_protected_masks(
    monkeypatch, tmp_path: Path
) -> None:
    source = _splat(tmp_path / "scene.ply")
    collision = tmp_path / "collision.usda"
    collision.write_text(
        '''#usda 1.0
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
''',
        encoding="utf-8",
    )
    cameras = [
        _camera("front", 0.0, 0.0),
        _camera("near_left", -0.2, 1.0),
        _camera("near_right", 0.2, -1.0),
        _camera("far_left", -0.5, 3.0),
        _camera("far_right", 0.5, -3.0),
        _camera("raised", 0.1, 2.0),
        _camera("low", -0.1, -2.0),
        _camera("working", 0.05, 0.5),
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

    freeze = materialize_excision_audit_freeze(
        source_standard_splat_path=source,
        source_collision_path=collision,
        target_collision_prim_path="/Root/Target",
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
    assert len(freeze["camera_split"]["calibration_camera_ids"]) == 6
    assert freeze["scale_and_bounds"]["meters_per_unit"] == 1.0
    assert freeze["scale_and_bounds"]["source_gaussian_count"] == 4
    assert freeze["contribution_method"]["depth_anything_3_used"] is False
    assert freeze["historical_baseline"]["selected_gaussian_count"] == 4
    assert all(
        row["target_core_is_subset_of_historical_outer_mask"]
        for row in freeze["masks"]
    )
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
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = gpu_root / f"{CONTRIBUTION_EVIDENCE_SCHEMA}.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    receipt = materialize_excision_ownership(
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        contribution_manifest_path=manifest_path,
        source_standard_splat_path=source,
        output_root=tmp_path / "ownership",
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
    assert all(
        row["retained_rows_byte_exact"] is True
        for row in receipt["preservation"].values()
    )

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
    monkeypatch.setattr(excision_vast, "_git", lambda *args: "" if args[-2:] == ("status", "--short") else "fixture")
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
    monkeypatch.setattr(
        excision_vast,
        "_write_source_archive",
        lambda source, destination: destination.write_bytes(b"fixture-source"),
    )
    bundle = excision_vast.build_gaussian_excision_vast_bundle(
        repo_root=Path(__file__).resolve().parents[1],
        flashsplat_root=tmp_path,
        freeze_path=tmp_path / "freeze" / f"{FREEZE_SCHEMA}.json",
        source_standard_splat_path=source,
        camera_contract_path=camera_path,
        execution_authority_path=authority_path,
        job_dir=tmp_path / "bundle",
        generated_at="2026-08-09T00:00:00Z",
    )
    assert bundle["status"] == "ready"
    assert bundle["raw_interiorgs_downloaded_bytes_included"] is False
    assert bundle["private_scene_derived_standard_splat_included"] is True
    with zipfile.ZipFile(bundle["bundle_path"]) as archive:
        assert "input/scene_standard.ply" in archive.namelist()
        assert "freeze/masks/front.target_core.png" in archive.namelist()
        assert "run_adp_gaussian_excision_provider_runtime.sh" in archive.namelist()
