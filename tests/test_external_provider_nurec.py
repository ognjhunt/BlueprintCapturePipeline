from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import zipfile

import jsonschema
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.external_provider_nurec import (
    ExternalProviderNuRecError,
    build_acquisition_receipt,
    build_external_source_import_request,
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_request_from_checkout,
    build_provider_nurec_isaac_runtime_result,
    import_external_source,
    normalize_provider_nurec_isaac_verification,
    qualify_provider_nurec_usdz,
    sha256_file,
)
from blueprint_pipeline.reconstruction_isaac_output_bundle import (
    compile_isaac_verification_output_bundle,
    validate_and_extract_isaac_verification_output_bundle,
)
from blueprint_pipeline.reconstruction_isaac_vast_operation import (
    _validate_bindings as validate_isaac_vast_bindings,
)
from blueprint_pipeline.reconstruction_isaac_worker_bundle import (
    IsaacWorkerBundleError,
    compile_isaac_verification_worker_bundle,
    extract_isaac_verification_worker_bundle,
    validate_isaac_verification_worker_bundle_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
D = ["sha256:" + character * 64 for character in "abcdef"]


def _rights(profile: str) -> dict:
    public = profile == "public_provider_sample"
    return {
        "terms_version": "niantic-business-terms-2026-02-20",
        "rights_terms_source": "https://www.nianticspatial.com/legal/business-terms",
        "ownership_or_license_status": (
            "official_public_sample_local_inspection"
            if public
            else "operator_attested_private_export"
        ),
        "commercial_use_status": "not_requested",
        "consent_privacy_status": "not_required_public_sample"
        if public
        else "operator_attested_private",
        "confidential": not public,
        "public_reporting_allowed": public,
        "retention_status": "bounded_local_cache",
        "deletion_status": "delete_on_request",
        "model_training_status": "not_authorized",
        "benchmarking_status": "restricted_pending_review",
        "allowed_uses": ["local_engineering_inspection"],
        "remote_upload_authorized": False,
    }


def _request(asset: Path, profile: str = "public_provider_sample", **updates) -> dict:
    digest = sha256_file(asset)
    value = {
        "stable_run_identity": "provider-nurec-test",
        "source_commit_sha": "1" * 40,
        "source_profile": profile,
        "acquisition_or_export_receipt_digest": D[0],
        "external_source_identity": {
            "provider": "scaniverse",
            "provider_asset_identifiers": {"public_sample": "ethel_sim"}
            if profile.startswith("public")
            else {},
            "local_asset_digest": digest,
            "acquisition_or_export_time": "2026-08-01T19:29:32-05:00",
            "capture_modality": {"status": "unknown", "value": "unknown"},
            "operator_reference": "public-supplier"
            if profile.startswith("public")
            else "private_operator_001",
            "terms_version": "niantic-business-terms-2026-02-20",
            "source_relationship_to_blueprint_raw_capture": "none",
        },
        "asset_binding": {
            "relative_path": asset.name,
            "digest": digest,
        },
        "rights_scope": _rights(profile),
        "remote_calls_authorized": False,
        "remote_calls_performed": False,
        "external_derived_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "proof_effect": "external_import_request_only",
        "claim_ceiling": "none",
    }
    value.update(updates)
    return build_external_source_import_request(value)


def test_valid_public_and_private_external_source_profiles_are_not_raw_capture(
    tmp_path: Path,
) -> None:
    for profile in ("public_provider_sample", "user_managed_provider_export"):
        source = tmp_path / profile
        source.mkdir()
        asset = source / "asset.usdz"
        asset.write_bytes(profile.encode())
        request = _request(asset, profile)
        assert (
            request["external_source_identity"]["source_relationship_to_blueprint_raw_capture"]
            == "none"
        )
        assert request["blueprint_raw_capture_truth"] is False
        schema = json.loads(
            (
                ROOT / "docs/schemas/external_reconstruction_import_request.v2.schema.json"
            ).read_text()
        )
        jsonschema.Draft202012Validator(schema).validate(request)


def test_external_source_contract_rejects_fabricated_raw_identity_missing_digest_and_rights_drift(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "asset.usdz"
    asset.write_bytes(b"asset")
    with pytest.raises(
        ExternalProviderNuRecError, match="fabricated_raw_capture_identity_forbidden"
    ):
        _request(asset, source_capture_identity="fake-blueprint-capture")

    request = dict(_request(asset))
    request.pop("external_import_request_digest")
    request["asset_binding"]["digest"] = None
    with pytest.raises(ExternalProviderNuRecError, match="external_source_asset_digest_invalid"):
        build_external_source_import_request(request)

    rights = _rights("public_provider_sample")
    rights["terms_version"] = "different-terms"
    with pytest.raises(ExternalProviderNuRecError, match="external_source_rights_terms_mismatch"):
        _request(asset, rights_scope=rights)


def test_external_source_contract_rejects_unsafe_symlink_digest_mismatch_and_unsupported_profile(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "asset.usdz"
    asset.write_bytes(b"asset")
    request = dict(_request(asset))
    request.pop("external_import_request_digest")
    request["asset_binding"]["relative_path"] = "../asset.usdz"
    with pytest.raises(ExternalProviderNuRecError, match="external_source_asset_path_unsafe"):
        build_external_source_import_request(request)

    link = source / "link.usdz"
    link.symlink_to(asset)
    link_request = dict(_request(asset))
    link_request.pop("external_import_request_digest")
    link_request["asset_binding"]["relative_path"] = link.name
    link_request = build_external_source_import_request(link_request)
    with pytest.raises(ExternalProviderNuRecError, match="external_source_asset_symlink_forbidden"):
        import_external_source(
            source_artifact=link_request,
            artifact_root=source,
            output_root=tmp_path / "link-out",
        )

    mismatched = _request(asset)
    asset.write_bytes(b"changed")
    with pytest.raises(ExternalProviderNuRecError, match="external_source_asset_digest_mismatch"):
        import_external_source(
            source_artifact=mismatched,
            artifact_root=source,
            output_root=tmp_path / "mismatch-out",
        )

    with pytest.raises(ExternalProviderNuRecError, match="external_source_profile_unsupported"):
        _request(asset, "unsupported_source")


def test_public_confidentiality_mismatch_and_provider_receipt_replay_fail_closed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "asset.usdz"
    asset.write_bytes(b"asset")
    wrong = _rights("public_provider_sample")
    wrong["confidential"] = True
    wrong["public_reporting_allowed"] = False
    with pytest.raises(
        ExternalProviderNuRecError, match="public_provider_sample_confidentiality_invalid"
    ):
        _request(asset, rights_scope=wrong)

    request = _request(asset)
    receipt = import_external_source(
        source_artifact=request,
        artifact_root=source,
        output_root=tmp_path / "out",
    )
    assert (
        import_external_source(
            source_artifact=request,
            artifact_root=source,
            output_root=tmp_path / "out",
        )
        == receipt
    )
    final = tmp_path / "out" / request["external_import_request_digest"][7:]
    rights_path = final / "external_provider_provenance_rights_receipt.v2.json"
    rights = json.loads(rights_path.read_text())
    rights["rights_scope"]["benchmarking_status"] = "silently-expanded"
    rights_path.write_text(json.dumps(rights))
    with pytest.raises(ExternalProviderNuRecError, match="external_source_import_replay_tampered"):
        import_external_source(
            source_artifact=request,
            artifact_root=source,
            output_root=tmp_path / "out",
        )


def test_acquisition_receipt_and_import_receipts_validate_against_versioned_schemas(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    asset = source / "ethel_sim.usdz"
    asset.write_bytes(b"asset")
    acquisition = build_acquisition_receipt(
        {
            "source_page_url": "https://example.test/samples",
            "final_download_url": "https://example.test/ethel_sim.usdz",
            "original_filename": "ethel_sim.usdz",
            "acquisition_timestamp": "2026-08-01T19:29:32-05:00",
            "byte_size": asset.stat().st_size,
            "asset_digest": sha256_file(asset),
            "http_metadata": {"content-length": str(asset.stat().st_size)},
            "supplier_identity": "Niantic Spatial",
            "declared_sample_scene": "ethel_sim",
            "rights_terms_source": "https://example.test/terms",
            "rights_review_status": "local_inspection_only_reviewed",
            "source_class": "public_provider_sample",
            "confidential": False,
            "blueprint_raw_capture": False,
            "provider_reconstruction": True,
            "capture_hardware": "unknown",
            "capture_modality": "unknown",
            "source_commit_sha": "1" * 40,
            "tool_version": "fixture",
            "remote_provider_login_performed": False,
            "proof_effect": "immutable_external_asset_acquisition_only",
            "claim_ceiling": "file_identity_and_declared_source_only",
        }
    )
    request = _request(
        asset, acquisition_or_export_receipt_digest=acquisition["acquisition_receipt_digest"]
    )
    receipt = import_external_source(
        source_artifact=request,
        artifact_root=source,
        output_root=tmp_path / "out",
    )
    final = tmp_path / "out" / request["external_import_request_digest"][7:]
    rights = json.loads((final / "external_provider_provenance_rights_receipt.v2.json").read_text())
    for filename, artifact in (
        ("external_provider_acquisition_receipt.v1.schema.json", acquisition),
        ("external_reconstruction_import_request.v2.schema.json", request),
        ("external_provider_provenance_rights_receipt.v2.schema.json", rights),
        ("external_reconstruction_import_receipt.v2.schema.json", receipt),
    ):
        schema = json.loads((ROOT / "docs/schemas" / filename).read_text())
        jsonschema.Draft202012Validator(schema).validate(artifact)


def test_usdz_qualification_rejects_duplicate_members_before_openusd(tmp_path: Path) -> None:
    package = tmp_path / "duplicate.usdz"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("default.usda", "#usda 1.0")
        archive.writestr("default.usda", "#usda 1.0")
    with pytest.raises(ExternalProviderNuRecError, match="provider_nurec_usdz_duplicate_member"):
        qualify_provider_nurec_usdz(
            package_path=package,
            expected_digest=sha256_file(package),
            external_import_receipt_digest=D[1],
        )


def test_provider_isaac_request_is_paid_authority_bounded() -> None:
    value = {
        "stable_run_identity": "provider-nurec-test",
        "source_commit_sha": "1" * 40,
        "package_digest": D[0],
        "package_artifact_reference": "source/ethel_sim.usdz",
        "external_import_receipt_digest": D[1],
        "qualification_report_digest": D[2],
        "fixed_camera_spec_digest": D[3],
        "fixed_camera_ids": ["probe-near", "probe-wide"],
        "runtime_implementation_digest": D[4],
        "runtime_container_image_digest": "registry.test/isaac@" + D[5],
        "expected_prim_paths": {
            "appearance": "/World/gauss/gauss",
            "collision": "/World/gauss/mesh",
        },
        "physics_probe_request": {
            "ground_collider_prim": "/World/gauss/mesh",
            "ground_height_m": -1.85,
            "probe_xy_m": [-21.9, 2.0],
            "selection_status": "cpu_geometry_candidate_unverified_in_isaac",
            "manufacture_ground_plane": False,
            "require_contact_event": True,
            "steps": 240,
        },
        "timeout_seconds": 3600,
        "spend_controls": {
            "authorized": False,
            "estimated_max_spend_usd": 2.0,
            "hard_ttl_seconds": 3600,
            "teardown_required": True,
            "provider_zero_required_before_and_after": True,
        },
        "provider_authored_package": True,
        "exact_package_required": True,
        "headless": True,
        "display_attached": False,
        "execution_status": "awaiting_explicit_paid_runtime_authorization",
        "provider_allocation_performed": False,
        "expected_runtime_schema": "provider_nurec_isaac_runtime_result.v1",
        "proof_effect": "none",
        "claim_ceiling": "request_only",
    }
    request = build_provider_nurec_isaac_request(value)
    assert request["spend_controls"]["authorized"] is False
    schema = json.loads(
        (ROOT / "docs/schemas/provider_nurec_isaac_verification_request.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(request)
    overgrant = dict(value)
    overgrant["spend_controls"] = dict(value["spend_controls"], authorized=True)
    with pytest.raises(ExternalProviderNuRecError, match="paid_authority_must_be_false"):
        build_provider_nurec_isaac_request(overgrant)

    runtime = build_provider_nurec_isaac_runtime_result(
        {
            "schema_version": "provider_nurec_isaac_runtime_result.v1",
            "status": "completed",
            "isaac_verification_request_digest": request["isaac_verification_request_digest"],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_identity": {
                "runtime": "isaac_sim",
                "renderer": "RayTracedLighting",
                "python_version": "3.11",
                "headless": True,
            },
            "raw_secret_values_recorded": False,
            "cost_usd": 0.25,
            "duration_seconds": 120.0,
            "stage": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "transforms_valid": True,
                "dependency_inspection_available": True,
                "missing_asset_count": 0,
                "particlefield_prim_count": 2,
                "active_collision_prim_count": 1,
                "obvious_scale_mismatch_detected": False,
                "expected_prim_paths": request["expected_prim_paths"],
            },
            "physics_probe": {
                "ground_contact_surface_present": True,
                "steps_executed": 240,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 1,
                "probe_configuration": {},
            },
            "cameras": [
                {"id": camera_id, "digest": D[0], "pixel_std": 10.0, "nonblank": True}
                for camera_id in request["fixed_camera_ids"]
            ],
            "proof_boundary": {
                "isaac_load_render_physics_presence_compatibility": True,
                "simulator_task_success_proven": False,
                "physics_navigation_control_proven": False,
                "physical_success_proven": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
            },
        },
        verification_request=request,
    )
    schema = json.loads(
        (ROOT / "docs/schemas/provider_nurec_isaac_runtime_result.v1.schema.json").read_text()
    )
    # Complete runner output includes these image facts; add them to the minimal
    # validator fixture before the JSON-schema replay.
    for row in runtime["cameras"]:
        row.update(
            artifact_reference=f"frames/{row['id']}.png",
            width=1280,
            height=960,
            pixel_mean=100.0,
        )
    runtime["isaac_runtime_result_digest"] = canonical_digest(
        runtime, digest_field="isaac_runtime_result_digest"
    )
    jsonschema.Draft202012Validator(schema).validate(runtime)


def test_provider_isaac_request_materialization_binds_real_clean_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test Operator"],
        check=True,
    )
    tracked = checkout / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "-m", "freeze"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    request_value = {
        "stable_run_identity": "provider-nurec-checkout-bound",
        "package_digest": D[0],
        "package_artifact_reference": "source/ethel_sim.usdz",
        "external_import_receipt_digest": D[1],
        "qualification_report_digest": D[2],
        "fixed_camera_spec_digest": D[3],
        "fixed_camera_ids": ["probe-near", "probe-wide"],
        "runtime_implementation_digest": D[4],
        "runtime_container_image_digest": "registry.test/isaac@" + D[5],
        "expected_prim_paths": {
            "appearance": "/World/gauss/gauss",
            "collision": "/World/gauss/mesh",
        },
        "physics_probe_request": {
            "ground_collider_prim": "/World/gauss/mesh",
            "ground_height_m": -1.85,
            "probe_xy_m": [-21.9, 2.0],
            "selection_status": "cpu_geometry_candidate_unverified_in_isaac",
            "manufacture_ground_plane": False,
            "require_contact_event": True,
            "steps": 240,
        },
        "timeout_seconds": 3600,
        "spend_controls": {
            "authorized": False,
            "estimated_max_spend_usd": 2.0,
            "hard_ttl_seconds": 3600,
            "teardown_required": True,
            "provider_zero_required_before_and_after": True,
        },
        "provider_authored_package": True,
        "exact_package_required": True,
        "headless": True,
        "display_attached": False,
        "execution_status": "awaiting_explicit_paid_runtime_authorization",
        "provider_allocation_performed": False,
        "expected_runtime_schema": "provider_nurec_isaac_runtime_result.v1",
        "proof_effect": "none",
        "claim_ceiling": "request_only",
    }
    request = build_provider_nurec_isaac_request_from_checkout(
        request_value,
        source_checkout=checkout,
    )
    assert request["source_commit_sha"] == commit

    with pytest.raises(
        ExternalProviderNuRecError,
        match="provider_isaac_source_checkout_commit_mismatch",
    ):
        build_provider_nurec_isaac_request_from_checkout(
            {**request_value, "source_commit_sha": "f" * 40},
            source_checkout=checkout,
        )

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(
        ExternalProviderNuRecError,
        match="provider_isaac_source_checkout_not_clean",
    ):
        build_provider_nurec_isaac_request_from_checkout(
            request_value,
            source_checkout=checkout,
        )


def test_provider_isaac_worker_bundle_preserves_exact_package_and_dynamic_prims(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package-root"
    package = package_root / "source/ethel_sim.usdz"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"exact-provider-package")
    cameras = tmp_path / "fixed_cameras.json"
    cameras.write_text(
        json.dumps(
            [
                {
                    "id": "probe-near",
                    "spec": {"pos": [0, -2, 1], "target": [0, 0, 0], "up": [0, 0, 1], "fov": 60},
                },
                {
                    "id": "probe-wide",
                    "spec": {"pos": [0, -5, 3], "target": [0, 0, 0], "up": [0, 0, 1], "fov": 70},
                },
            ]
        )
    )
    runner = tmp_path / "runner.py"
    runner.write_text("print('provider fixture')\n")
    render_options = tmp_path / "render_options.json"
    render_options.write_text(
        json.dumps(
            {
                "robot_id": "franka_panda",
                "robot_usd": "Robots/FrankaRobotics/FrankaPanda/franka.usd",
                "robot_prim_path": "/World/Franka",
                "robot_pose": [-21.9, 2.0, -1.85, 0.0],
                "robot_ground_z": -1.85,
                "robot_only_pass": True,
                "robot_placement_digest": D[3],
                "placement_proposal_digest": D[4],
                "articulated_policy_trace_request": {
                    "schema_version": "franka_articulated_policy_trace_request.v1",
                    "robot_id": "franka_panda",
                    "robot_prim_path": "/World/Franka",
                    "controller_id": "deterministic_franka_joint_position_pair.v1",
                    "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                    "start_joint_positions_rad": [0.0, -0.55, 0.0, -2.6, 0.0, 2.05, 0.75],
                    "physics_dt_seconds": 1.0 / 60.0,
                    "reset_settle_steps": 30,
                    "sample_interval_steps": 10,
                    "distinctness_threshold_rad": 0.1,
                    "identical_start_tolerance_rad": 0.02,
                    "candidates": [
                        {
                            "policy_id": "franka-fixed-hold-v1",
                            "duration_steps": 120,
                            "final_joint_positions_rad": [0.0, -0.55, 0.0, -2.6, 0.0, 2.05, 0.75],
                        },
                        {
                            "policy_id": "franka-inspection-sweep-v1",
                            "duration_steps": 120,
                            "final_joint_positions_rad": [0.35, -0.55, 0.0, -2.6, 0.0, 2.05, 0.75],
                        },
                    ],
                    "egocentric_camera": {
                        "parent_link_name": "panda_hand",
                        "local_position_m": [0.05, 0.0, 0.04],
                        "local_target_m": [0.3, 0.0, 0.04],
                        "local_up": [0.0, 0.0, 1.0],
                        "fov_degrees": 70.0,
                        "width": 320,
                        "height": 240,
                    },
                    "physical_success_claimed": False,
                },
            }
        )
    )
    request = build_provider_nurec_isaac_request(
        {
            "stable_run_identity": "provider-nurec-test",
            "source_commit_sha": "1" * 40,
            "package_digest": sha256_file(package),
            "package_artifact_reference": "source/ethel_sim.usdz",
            "external_import_receipt_digest": D[1],
            "qualification_report_digest": D[2],
            "fixed_camera_spec_digest": sha256_file(cameras),
            "fixed_camera_ids": ["probe-near", "probe-wide"],
            "runtime_implementation_digest": sha256_file(runner),
            "render_options_digest": sha256_file(render_options),
            "runtime_container_image_digest": "registry.test/isaac@" + D[5],
            "expected_prim_paths": {
                "appearance": "/World/gauss/gauss",
                "collision": "/World/gauss/mesh",
            },
            "physics_probe_request": {
                "ground_collider_prim": "/World/gauss/mesh",
                "ground_height_m": -1.85,
                "probe_xy_m": [-21.9, 2.0],
                "selection_status": "cpu_geometry_candidate_unverified_in_isaac",
                "manufacture_ground_plane": False,
                "require_contact_event": True,
                "steps": 240,
            },
            "timeout_seconds": 3600,
            "spend_controls": {
                "authorized": False,
                "estimated_max_spend_usd": 2.0,
                "hard_ttl_seconds": 3600,
                "teardown_required": True,
                "provider_zero_required_before_and_after": True,
            },
            "provider_authored_package": True,
            "exact_package_required": True,
            "headless": True,
            "display_attached": False,
            "execution_status": "awaiting_explicit_paid_runtime_authorization",
            "provider_allocation_performed": False,
            "expected_runtime_schema": "provider_nurec_isaac_runtime_result.v1",
            "proof_effect": "none",
            "claim_ceiling": "request_only",
        }
    )
    receipt = compile_isaac_verification_worker_bundle(
        verification_request=request,
        package_artifact_root=package_root,
        fixed_camera_spec_path=cameras,
        runner_path=runner,
        output_root=tmp_path / "bundles",
        render_options_path=render_options,
    )
    assert receipt["schema_version"] == "provider_nurec_isaac_worker_bundle.v1"
    assert "--provider-package-mode" in receipt["command"]
    assert receipt["bundle_member_count"] == 6
    assert receipt["render_options_digest"] == sha256_file(render_options)
    assert validate_isaac_verification_worker_bundle_receipt(receipt) == receipt
    schema = json.loads(
        (ROOT / "docs/schemas/provider_nurec_isaac_worker_bundle.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(receipt)
    bound = {
        "schema_version": "reconstruction_gpu_canary_request.v1",
        "operation": "provider_nurec_isaac_canary",
        "request_digest": D[0],
        "operation_request_digest": request["isaac_verification_request_digest"],
        "operation_input_bundle_digest": receipt["bundle_digest"],
        "expected_runtime_result_schema": "provider_nurec_isaac_runtime_result.v1",
        "worker_image_digest": request["runtime_container_image_digest"],
        "source_commit_sha": request["source_commit_sha"],
        "bound_provider": "vast",
        "provider_mutation_authorized": True,
        "bound_checkout_clean": True,
        "bound_checkout_source_commit": request["source_commit_sha"],
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
        "proof_effect": "none",
        "isaac_image_release_digest": D[1],
    }
    bound["bound_request_digest"] = canonical_digest(bound, digest_field="bound_request_digest")
    validated_bound, validated_receipt = validate_isaac_vast_bindings(
        bound_request=bound, bundle_receipt=receipt
    )
    assert validated_bound["operation"] == "provider_nurec_isaac_canary"
    assert validated_receipt == receipt
    bundle = (
        tmp_path
        / "bundles"
        / request["isaac_verification_request_digest"][7:]
        / "isaac_verification_worker_bundle.zip"
    )
    extraction = extract_isaac_verification_worker_bundle(
        bundle_path=bundle,
        bundle_receipt=receipt,
        output_root=tmp_path / "extracted",
    )
    assert (
        extraction["isaac_verification_request_digest"]
        == request["isaac_verification_request_digest"]
    )
    extracted_root = tmp_path / "extracted" / receipt["bundle_digest"][7:]
    assert (extracted_root / "render_options.json").read_bytes() == render_options.read_bytes()

    with pytest.raises(IsaacWorkerBundleError, match="render_options_binding_incomplete"):
        compile_isaac_verification_worker_bundle(
            verification_request=request,
            package_artifact_root=package_root,
            fixed_camera_spec_path=cameras,
            runner_path=runner,
            output_root=tmp_path / "missing-options-bundle",
        )

    bad_options = tmp_path / "bad_render_options.json"
    bad_value = json.loads(render_options.read_text())
    bad_value["api_key"] = "must-not-enter-bundle"
    bad_options.write_text(json.dumps(bad_value))
    bad_request_value = dict(request)
    bad_request_value.pop("isaac_verification_request_digest")
    bad_request_value["render_options_digest"] = sha256_file(bad_options)
    bad_request = build_provider_nurec_isaac_request(bad_request_value)
    with pytest.raises(IsaacWorkerBundleError, match="render_options_secret_value_forbidden"):
        compile_isaac_verification_worker_bundle(
            verification_request=bad_request,
            package_artifact_root=package_root,
            fixed_camera_spec_path=cameras,
            runner_path=runner,
            output_root=tmp_path / "secret-options-bundle",
            render_options_path=bad_options,
        )


def test_provider_isaac_worker_bundle_rejects_unbound_or_secret_render_options(
    tmp_path: Path,
) -> None:
    with pytest.raises(ExternalProviderNuRecError, match="render_options_digest_invalid"):
        build_provider_nurec_isaac_request(
            {
                "schema_version": "provider_nurec_isaac_verification_request.v1",
                "render_options_digest": "not-a-digest",
            }
        )


def test_provider_runtime_output_is_independently_rehashed_and_allocator_transportable(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "package-root"
    package = package_root / "source/ethel_sim.usdz"
    package.parent.mkdir(parents=True)
    package.write_bytes(b"exact-provider-package")
    cameras = tmp_path / "fixed_cameras.json"
    camera_ids = ["probe-near", "probe-wide"]
    cameras.write_text(
        json.dumps(
            [
                {
                    "id": camera_id,
                    "spec": {
                        "pos": [0, -2 - index, 1 + index],
                        "target": [0, 0, 0],
                        "up": [0, 0, 1],
                        "fov": 60,
                    },
                }
                for index, camera_id in enumerate(camera_ids)
            ]
        ),
        encoding="utf-8",
    )
    runner = tmp_path / "runner.py"
    runner.write_text("print('provider fixture')\n", encoding="utf-8")
    request = build_provider_nurec_isaac_request(
        {
            "stable_run_identity": "provider-nurec-output-test",
            "source_commit_sha": "1" * 40,
            "package_digest": sha256_file(package),
            "package_artifact_reference": "source/ethel_sim.usdz",
            "external_import_receipt_digest": D[1],
            "qualification_report_digest": D[2],
            "fixed_camera_spec_digest": sha256_file(cameras),
            "fixed_camera_ids": camera_ids,
            "runtime_implementation_digest": sha256_file(runner),
            "runtime_container_image_digest": "registry.test/isaac@" + D[5],
            "expected_prim_paths": {
                "appearance": "/World/gauss/gauss",
                "collision": "/World/gauss/mesh",
            },
            "physics_probe_request": {
                "ground_collider_prim": "/World/gauss/mesh",
                "ground_height_m": -1.85,
                "probe_xy_m": [-21.9, 2.0],
                "selection_status": "cpu_geometry_candidate_unverified_in_isaac",
                "manufacture_ground_plane": False,
                "require_contact_event": True,
                "steps": 240,
            },
            "timeout_seconds": 3600,
            "spend_controls": {
                "authorized": False,
                "estimated_max_spend_usd": 2.0,
                "hard_ttl_seconds": 3600,
                "teardown_required": True,
                "provider_zero_required_before_and_after": True,
            },
            "provider_authored_package": True,
            "exact_package_required": True,
            "headless": True,
            "display_attached": False,
            "execution_status": "awaiting_explicit_paid_runtime_authorization",
            "provider_allocation_performed": False,
            "expected_runtime_schema": "provider_nurec_isaac_runtime_result.v1",
            "proof_effect": "none",
            "claim_ceiling": "request_only",
        }
    )
    receipt = compile_isaac_verification_worker_bundle(
        verification_request=request,
        package_artifact_root=package_root,
        fixed_camera_spec_path=cameras,
        runner_path=runner,
        output_root=tmp_path / "bundles",
    )
    bundle = (
        tmp_path
        / "bundles"
        / request["isaac_verification_request_digest"][7:]
        / "isaac_verification_worker_bundle.zip"
    )
    extraction = extract_isaac_verification_worker_bundle(
        bundle_path=bundle,
        bundle_receipt=receipt,
        output_root=tmp_path / "materialized",
    )
    work_root = tmp_path / "worker"
    work_root.mkdir()
    os.replace(
        tmp_path / "materialized" / receipt["bundle_digest"][7:],
        work_root / "bundle",
    )
    runtime_root = work_root / "out"
    frames = runtime_root / "frames"
    frames.mkdir(parents=True)
    camera_rows = []
    for index, camera_id in enumerate(camera_ids):
        frame = frames / f"{camera_id}.png"
        frame.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index + 1]) * 64)
        camera_rows.append(
            {
                "id": camera_id,
                "artifact_reference": f"frames/{camera_id}.png",
                "digest": sha256_file(frame),
                "width": 1280,
                "height": 960,
                "pixel_mean": 100.0,
                "pixel_std": 10.0,
                "nonblank": True,
            }
        )
    robot_frames = runtime_root / "frames_robot_only"
    robot_frames.mkdir()
    robot_rgb = robot_frames / f"{camera_ids[0]}.png"
    robot_rgb.write_bytes(b"\x89PNG\r\n\x1a\nrobot-evidence")
    robot_distance = robot_frames / f"{camera_ids[0]}_distance.npy"
    robot_distance.write_bytes(b"NUMPY-DISTANCE-EVIDENCE")
    runtime = build_provider_nurec_isaac_runtime_result(
        {
            "schema_version": "provider_nurec_isaac_runtime_result.v1",
            "status": "completed",
            "isaac_verification_request_digest": request["isaac_verification_request_digest"],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_identity": {
                "runtime": "isaac_sim",
                "renderer": "RayTracedLighting",
                "python_version": "3.11",
                "headless": True,
            },
            "raw_secret_values_recorded": False,
            "cost_usd": 0.25,
            "duration_seconds": 120.0,
            "stage": {
                "meters_per_unit": 1.0,
                "up_axis": "Z",
                "transforms_valid": True,
                "dependency_inspection_available": True,
                "missing_asset_count": 0,
                "particlefield_prim_count": 2,
                "active_collision_prim_count": 1,
                "obvious_scale_mismatch_detected": False,
                "expected_prim_paths": request["expected_prim_paths"],
            },
            "physics_probe": {
                "ground_contact_surface_present": True,
                "steps_executed": 240,
                "live_rigid_body_pose_observed": True,
                "test_body_fell_through_floor": False,
                "contact_event_count": 1,
                "probe_configuration": {},
            },
            "cameras": camera_rows,
            "robot": {
                "requested": True,
                "composited": True,
                "geometry_streamed": True,
                "robot_only_pass": [
                    {
                        "id": camera_ids[0],
                        "pixel_std": 12.0,
                        "nonblank": True,
                        "depth_npy": True,
                        "rgb_artifact_reference": (f"frames_robot_only/{camera_ids[0]}.png"),
                        "rgb_digest": sha256_file(robot_rgb),
                        "distance_artifact_reference": (
                            f"frames_robot_only/{camera_ids[0]}_distance.npy"
                        ),
                        "distance_digest": sha256_file(robot_distance),
                    }
                ],
            },
            "proof_boundary": {
                "isaac_load_render_physics_presence_compatibility": True,
                "simulator_task_success_proven": False,
                "physics_navigation_control_proven": False,
                "physical_success_proven": False,
                "physical_robot_readiness_proven": False,
                "deployment_readiness_proven": False,
            },
        },
        verification_request=request,
    )
    (runtime_root / "isaac_runtime_result.json").write_text(json.dumps(runtime), encoding="utf-8")
    output_bundle = tmp_path / "provider-output.zip"
    compiled = compile_isaac_verification_output_bundle(
        bundle_receipt=receipt,
        runtime_output_root=runtime_root,
        output_path=output_bundle,
    )
    assert compiled["runtime_result_schema"] == "provider_nurec_isaac_runtime_result.v1"
    validated, validated_runtime, validated_root = (
        validate_and_extract_isaac_verification_output_bundle(
            bundle_path=output_bundle,
            expected_input_receipt=receipt,
            expected_source_commit_sha=request["source_commit_sha"],
            output_root=tmp_path / "validated-output",
        )
    )
    assert validated["runtime_result_digest"] == runtime["isaac_runtime_result_digest"]
    assert validated_runtime == runtime
    assert (validated_root / f"frames_robot_only/{camera_ids[0]}.png").is_file()
    assert (validated_root / f"frames_robot_only/{camera_ids[0]}_distance.npy").is_file()
    qualification = normalize_provider_nurec_isaac_verification(
        verification_request=request,
        runtime_result=validated_runtime,
        package_artifact_root=package_root,
        runtime_artifact_root=validated_root,
    )
    assert qualification["status"] == "verified_compatibility_only"
    schema = json.loads(
        (ROOT / "docs/schemas/provider_nurec_isaac_verification_result.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator(schema).validate(qualification)
    assert extraction["proof_effect"] == "none"
