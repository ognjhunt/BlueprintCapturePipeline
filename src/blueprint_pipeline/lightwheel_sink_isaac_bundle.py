"""Build an immutable derivative-USD input bundle for the Lightwheel sink canary."""

from __future__ import annotations

import hashlib
import json
import base64
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .measurement_isaac_runtime_release import ISAAC_VERSION, RUNTIME_IMAGE, build_measurement_isaac_runtime_release


BUNDLE_SCHEMA_VERSION = "lightwheel_sink_isaac_input_bundle.v1"
RECEIPT_SCHEMA_VERSION = "lightwheel_sink_isaac_input_bundle_receipt.v1"
RUNTIME_RESULT_SCHEMA_VERSION = "lightwheel_sink_isaac_runtime_result.v1"
RUNNER_PATH = Path("scripts/run_lightwheel_sink_isaac_bundle.py")
WORKER_PATH = Path("src/blueprint_pipeline/lightwheel_sink_isaac_worker.py")
MODEL_ARCHIVE_PATH = "asset/model.usd"
TEXTURE_ARCHIVE_ROOT = "asset/textures"
WRAPPER_ARCHIVE_PATH = "scene/lightwheel_sink_test.usda"


class LightwheelSinkIsaacBundleError(ValueError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_identity(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if head.returncode != 0 or status.returncode != 0 or status.stdout.strip():
        raise LightwheelSinkIsaacBundleError("lightwheel_sink_source_not_clean_commit")
    commit = head.stdout.strip().lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise LightwheelSinkIsaacBundleError("lightwheel_sink_source_commit_invalid")
    return commit


def derivative_wrapper_usda() -> str:
    """Return the test-only USD overlay; the referenced Lightwheel layer stays immutable."""

    return '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def PhysicsScene "physicsScene"
    {
        uniform token physics:broadphaseType = "MBP"
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }

    def Xform "Sink" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
        prepend references = @../asset/model.usd@
    )
    {
        custom string blueprint:sourceAssetPolicy = "immutable-reference"
        custom string blueprint:testPurpose = "articulation-and-franka-contact-development-canary"

        def PhysicsFixedJoint "BlueprintFixedRoot"
        {
            rel physics:body1 = </World/Sink/RootCom>
        }
    }
}
'''


def _zip_bytes(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, payload)


def _asset_records(model: Path, textures: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    if not model.is_file() or model.is_symlink():
        raise LightwheelSinkIsaacBundleError("lightwheel_sink_model_invalid")
    if not textures.is_dir() or textures.is_symlink():
        raise LightwheelSinkIsaacBundleError("lightwheel_sink_textures_invalid")
    files = sorted(path for path in textures.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise LightwheelSinkIsaacBundleError("lightwheel_sink_texture_files_invalid")
    records = [
        {
            "path": MODEL_ARCHIVE_PATH,
            "digest": _sha256(model),
            "size_bytes": model.stat().st_size,
            "role": "source_usd",
        }
    ]
    for path in files:
        relative = path.relative_to(textures).as_posix()
        records.append(
            {
                "path": f"{TEXTURE_ARCHIVE_ROOT}/{relative}",
                "digest": _sha256(path),
                "size_bytes": path.stat().st_size,
                "role": "source_texture",
            }
        )
    return records, files


def compile_lightwheel_sink_isaac_input_bundle(
    *,
    repo_root: str | Path,
    model_path: str | Path,
    textures_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = _git_identity(root)
    model = Path(model_path).expanduser().resolve()
    textures = Path(textures_path).expanduser().resolve()
    asset_records, texture_files = _asset_records(model, textures)
    wrapper = derivative_wrapper_usda().encode("utf-8")
    runtime_release = build_measurement_isaac_runtime_release()
    source_files = sorted((root / "src/blueprint_pipeline").rglob("*.py"))
    source_files.append(root / RUNNER_PATH)
    if any(not path.is_file() or path.is_symlink() for path in source_files):
        raise LightwheelSinkIsaacBundleError("lightwheel_sink_source_files_invalid")
    source_records = [
        {"path": path.relative_to(root).as_posix(), "digest": _sha256(path)}
        for path in source_files
    ]
    texture_manifest = {
        "schema_version": "lightwheel_sink_source_texture_manifest.v1",
        "files": [row for row in asset_records if row["role"] == "source_texture"],
    }
    texture_manifest["texture_manifest_digest"] = canonical_digest(
        texture_manifest, digest_field="texture_manifest_digest"
    )
    test_configuration = {
        "schema_version": "lightwheel_sink_isaac_test_configuration.v1",
        "sink_prim_path": "/World/Sink",
        "base_prim_path": "/World/Sink/RootCom",
        "handle_prim_path": "/World/Sink/Com_001",
        "handle_joint_path": "/World/Sink/Com_001/RootCom_to_Com_001_0",
        "handle_joint_targets_degrees": [0.0, 30.0, 60.0, 90.0, 120.0],
        "handle_joint_limits_degrees": [0.0, 120.0],
        "handle_pivot_source_world_m": [0.0615, 0.2643, 0.3615],
        "handle_center_source_world_m": [0.119967, 0.256083, 0.401194],
        "sink_translation_world_m": [0.4885, -0.2643, 0.0885],
        "franka_prim_path": "/World/Franka",
        "franka_asset_relative_path": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "physics_dt_seconds": 1.0 / 60.0,
        "settle_steps": 45,
        "target_steps": 45,
        "pusher_steps": 90,
        "ik_steps_per_waypoint": 90,
        "renderer": "RayTracedLighting",
        "render_resolution": [640, 480],
        "deterministic_seed": 47,
        "development_only": True,
    }
    test_configuration["test_configuration_digest"] = canonical_digest(
        test_configuration, digest_field="test_configuration_digest"
    )
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_commit_sha": commit,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": runtime_release["runtime_release_digest"],
        "isaac_sim_version": ISAAC_VERSION,
        "asset_files": asset_records,
        "source_model_digest": asset_records[0]["digest"],
        "texture_manifest": texture_manifest,
        "wrapper_path": WRAPPER_ARCHIVE_PATH,
        "wrapper_digest": _sha256_bytes(wrapper),
        "runner_path": RUNNER_PATH.as_posix(),
        "worker_path": WORKER_PATH.as_posix(),
        "source_files": source_records,
        "test_configuration": test_configuration,
        "source_asset_modified": False,
        "remote_upload_authorized_by_bundle": False,
        "paid_execution_authorized_by_bundle": False,
        "development_only": True,
        "physical_success_established": False,
        "production_route_eligible": False,
    }
    manifest["bundle_manifest_digest"] = canonical_digest(
        manifest, digest_field="bundle_manifest_digest"
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        _zip_bytes(archive, "bundle_manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
        _zip_bytes(archive, WRAPPER_ARCHIVE_PATH, wrapper)
        _zip_bytes(archive, MODEL_ARCHIVE_PATH, model.read_bytes())
        for path in texture_files:
            _zip_bytes(archive, f"{TEXTURE_ARCHIVE_ROOT}/{path.relative_to(textures).as_posix()}", path.read_bytes())
        for path in source_files:
            _zip_bytes(archive, path.relative_to(root).as_posix(), path.read_bytes())
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source_commit_sha": commit,
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": runtime_release["runtime_release_digest"],
        "bundle_manifest_digest": manifest["bundle_manifest_digest"],
        "input_bundle_digest": _sha256(output),
        "input_bundle_size_bytes": output.stat().st_size,
        "source_model_digest": manifest["source_model_digest"],
        "texture_manifest_digest": texture_manifest["texture_manifest_digest"],
        "wrapper_digest": manifest["wrapper_digest"],
        "test_configuration_digest": test_configuration["test_configuration_digest"],
        "asset_file_count": len(asset_records),
        "raw_secret_values_recorded": False,
        "provider_allocation_performed": False,
        "paid_execution_authorized_by_bundle": False,
        "proof_effect": "none",
        "claim_ceiling": "immutable_external_asset_development_input_only",
    }
    receipt["bundle_receipt_digest"] = canonical_digest(receipt, digest_field="bundle_receipt_digest")
    return validate_lightwheel_sink_isaac_input_bundle_receipt(receipt)


def validate_lightwheel_sink_isaac_input_bundle_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        errors.append("lightwheel_sink_bundle_receipt_schema_invalid")
    if receipt.get("runtime_image_digest") != RUNTIME_IMAGE:
        errors.append("lightwheel_sink_bundle_receipt_image_invalid")
    for key in (
        "bundle_manifest_digest",
        "input_bundle_digest",
        "source_model_digest",
        "texture_manifest_digest",
        "wrapper_digest",
        "test_configuration_digest",
    ):
        value = str(receipt.get(key) or "")
        if len(value) != 71 or not value.startswith("sha256:"):
            errors.append(f"lightwheel_sink_bundle_receipt_{key}_invalid")
    for key, expected in (
        ("raw_secret_values_recorded", False),
        ("provider_allocation_performed", False),
        ("paid_execution_authorized_by_bundle", False),
        ("proof_effect", "none"),
        ("claim_ceiling", "immutable_external_asset_development_input_only"),
    ):
        if receipt.get(key) != expected:
            errors.append(f"lightwheel_sink_bundle_receipt_{key}_invalid")
    if receipt.get("bundle_receipt_digest") != canonical_digest(receipt, digest_field="bundle_receipt_digest"):
        errors.append("lightwheel_sink_bundle_receipt_digest_mismatch")
    if errors:
        raise LightwheelSinkIsaacBundleError(";".join(sorted(set(errors))))
    return receipt


def validate_lightwheel_sink_isaac_runtime_result(
    value: Mapping[str, Any],
    *,
    bound_request: Mapping[str, Any],
    bundle_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if result.get("runtime_result_digest") != canonical_digest(
        result, digest_field="runtime_result_digest"
    ):
        errors.append("lightwheel_sink_runtime_result_digest_mismatch")
    expected = {
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        "status": "passed",
        "source_commit_sha": bound_request.get("source_commit_sha"),
        "runtime_image_digest": RUNTIME_IMAGE,
        "runtime_release_digest": bound_request.get("measurement_isaac_runtime_release_digest"),
        "input_bundle_digest": bundle_receipt.get("input_bundle_digest"),
        "bundle_manifest_digest": bundle_receipt.get("bundle_manifest_digest"),
        "source_model_digest": bundle_receipt.get("source_model_digest"),
        "texture_manifest_digest": bundle_receipt.get("texture_manifest_digest"),
        "wrapper_digest": bundle_receipt.get("wrapper_digest"),
        "test_configuration_digest": bundle_receipt.get("test_configuration_digest"),
        "development_only": True,
        "external_generated_support_asset": True,
        "blueprint_raw_capture_truth": False,
        "physical_success_established": False,
        "production_route_eligible": False,
        "qualification_created": False,
        "raw_secret_values_recorded": False,
        "proof_effect": "development_execution_only",
        "claim_ceiling": "isaac_articulation_and_scripted_franka_contact_development",
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            errors.append(f"lightwheel_sink_runtime_{key}_mismatch")
    gates = result.get("gates")
    required_gates = {
        "source_digests_preserved",
        "wrapper_composed",
        "asset_sweep_completed",
        "asset_sweep_targets_reached",
        "root_anchor_stable",
        "limit_behavior_stable",
        "capsule_contact_observed",
        "capsule_handle_motion_observed",
        "franka_contact_observed",
        "franka_handle_motion_observed",
        "franka_fixed_base_verified",
        "joint_effort_readback_available",
        "numerical_state_finite",
        "omnipbr_materials_present",
        "texture_bindings_present",
        "rgb_frames_valid",
    }
    if not isinstance(gates, Mapping) or any(gates.get(key) is not True for key in required_gates):
        errors.append("lightwheel_sink_runtime_gates_invalid")
    frames = result.get("runtime", {}).get("rendering", {}).get("frames", [])
    if not isinstance(frames, list) or len(frames) < 7:
        errors.append("lightwheel_sink_runtime_frames_missing")
    else:
        for row in frames:
            if not isinstance(row, Mapping):
                errors.append("lightwheel_sink_runtime_frame_invalid")
                continue
            try:
                payload = base64.b64decode(str(row.get("png_base64") or ""), validate=True)
            except (ValueError, TypeError):
                errors.append("lightwheel_sink_runtime_frame_base64_invalid")
                continue
            if _sha256_bytes(payload) != row.get("png_digest"):
                errors.append("lightwheel_sink_runtime_frame_digest_mismatch")
    if result.get("blockers") != []:
        errors.append("lightwheel_sink_runtime_reported_blockers")
    if errors:
        raise LightwheelSinkIsaacBundleError(";".join(sorted(set(errors))))
    return result


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "LightwheelSinkIsaacBundleError",
    "RECEIPT_SCHEMA_VERSION",
    "RUNTIME_RESULT_SCHEMA_VERSION",
    "compile_lightwheel_sink_isaac_input_bundle",
    "derivative_wrapper_usda",
    "validate_lightwheel_sink_isaac_runtime_result",
    "validate_lightwheel_sink_isaac_input_bundle_receipt",
]
