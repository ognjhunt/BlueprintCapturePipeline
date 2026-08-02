"""Build the non-spending, digest-bound input bundle for Isaac verification.

This compiler never allocates a provider.  Its output is the immutable input to
Blueprint's canonical paid-resource allocator after separate budget/TTL/retry
admission succeeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any, Mapping
import zipfile

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .isaac_reconstruction_verification import (
    IsaacReconstructionVerificationError,
    build_isaac_asset_verification_request,
)
from .external_provider_nurec import (
    ExternalProviderNuRecError,
    ISAAC_REQUEST_SCHEMA as PROVIDER_ISAAC_REQUEST_SCHEMA,
    build_provider_nurec_isaac_request,
)
from .external_scene_isaac_verification import (
    ExternalSceneIsaacVerificationError,
    REQUEST_SCHEMA as EXTERNAL_SCENE_ISAAC_REQUEST_SCHEMA,
    build_external_scene_isaac_verification_request,
)


ISAAC_WORKER_BUNDLE_SCHEMA = "isaac_verification_worker_bundle.v1"
PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA = "provider_nurec_isaac_worker_bundle.v1"
ISAAC_WORKER_EXTRACTION_SCHEMA = "isaac_verification_worker_bundle_extraction.v1"
MAX_BUNDLE_MEMBER_BYTES = 4_000_000_000
MAX_BUNDLE_TOTAL_BYTES = 5_000_000_000
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class IsaacWorkerBundleError(ValueError):
    def __init__(self, codes: list[str] | tuple[str, ...]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _safe_source(root: Path, reference: Any, digest: str, suffix: str, code: str) -> Path:
    text = str(reference or "").replace("\\", "/")
    relative = PurePosixPath(text)
    if (
        not text
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
        or relative.suffix.lower() != suffix
    ):
        raise IsaacWorkerBundleError([f"{code}_reference_unsafe"])
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise IsaacWorkerBundleError([f"{code}_symlink_forbidden"])
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacWorkerBundleError([f"{code}_missing"]) from exc
    if root != resolved and root not in resolved.parents:
        raise IsaacWorkerBundleError([f"{code}_path_escape"])
    if not resolved.is_file() or _sha256(resolved) != digest:
        raise IsaacWorkerBundleError([f"{code}_digest_mismatch"])
    if resolved.stat().st_size > MAX_BUNDLE_MEMBER_BYTES:
        raise IsaacWorkerBundleError([f"{code}_oversized"])
    return resolved


def _explicit_file(path_value: str | Path, *, digest: str, suffix: str, code: str) -> Path:
    path = Path(path_value)
    if path.is_symlink():
        raise IsaacWorkerBundleError([f"{code}_symlink_forbidden"])
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacWorkerBundleError([f"{code}_missing"]) from exc
    if not path.is_file() or path.suffix.lower() != suffix or _sha256(path) != digest:
        raise IsaacWorkerBundleError([f"{code}_digest_or_format_mismatch"])
    if path.stat().st_size > MAX_BUNDLE_MEMBER_BYTES:
        raise IsaacWorkerBundleError([f"{code}_oversized"])
    return path


def _validate_render_options(path: Path) -> dict[str, Any]:
    """Validate the optional robot/placement payload before bundling it.

    The runner intentionally accepts an optional ``render_options.json`` beside
    the fixed cameras.  This bundle compiler is the authority boundary: it
    requires the file to be an object, rejects credential-shaped keys, and
    validates the exact fields that can affect robot placement.  The complete
    bytes remain bound by ``render_options_digest`` in the verification request.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacWorkerBundleError(["isaac_render_options_json_invalid"]) from exc
    if not isinstance(value, Mapping):
        raise IsaacWorkerBundleError(["isaac_render_options_json_invalid"])

    def secret_paths(nested: Any, prefix: str = "") -> list[str]:
        found: list[str] = []
        if isinstance(nested, Mapping):
            for raw_key, child in nested.items():
                key = str(raw_key)
                path_text = f"{prefix}.{key}" if prefix else key
                lowered = key.lower()
                if any(
                    token in lowered for token in ("password", "secret", "credential", "api_key")
                ):
                    if child not in (None, "", [], {}):
                        found.append(path_text)
                found.extend(secret_paths(child, path_text))
        elif isinstance(nested, list):
            for index, child in enumerate(nested):
                found.extend(secret_paths(child, f"{prefix}[{index}]"))
        return found

    errors: list[str] = []
    if secret_paths(value):
        errors.append("isaac_render_options_secret_value_forbidden")
    robot_usd = value.get("robot_usd")
    pose = value.get("robot_pose")
    if not isinstance(robot_usd, str) or not robot_usd.strip():
        errors.append("isaac_render_options_robot_usd_invalid")
    if (
        not isinstance(pose, list)
        or len(pose) != 4
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in (pose or [])
        )
    ):
        errors.append("isaac_render_options_robot_pose_invalid")
    if not isinstance(value.get("robot_id"), str) or not value.get("robot_id", "").strip():
        errors.append("isaac_render_options_robot_id_invalid")
    for key in ("robot_placement_digest", "placement_proposal_digest"):
        if _DIGEST.fullmatch(str(value.get(key) or "")) is None:
            errors.append(f"isaac_render_options_{key}_invalid")
    prim_path = value.get("robot_prim_path", "/World/RobotVisual")
    if not isinstance(prim_path, str) or not prim_path.startswith("/") or ".." in prim_path:
        errors.append("isaac_render_options_robot_prim_path_invalid")
    ground_z = value.get("robot_ground_z")
    if (
        isinstance(ground_z, bool)
        or not isinstance(ground_z, (int, float))
        or not math.isfinite(float(ground_z))
    ):
        errors.append("isaac_render_options_robot_ground_z_invalid")
    for key in ("robot_only_pass",):
        if value.get(key) is not True:
            errors.append(f"isaac_render_options_{key}_must_be_true")
    trace = value.get("articulated_policy_trace_request")
    if trace is not None:
        if not isinstance(trace, Mapping):
            errors.append("isaac_render_options_policy_trace_request_invalid")
        else:
            expected_joint_names = [f"panda_joint{index}" for index in range(1, 8)]
            joint_names = trace.get("joint_names")
            start = trace.get("start_joint_positions_rad")
            candidates = trace.get("candidates")
            if trace.get("schema_version") != "franka_articulated_policy_trace_request.v1":
                errors.append("isaac_render_options_policy_trace_schema_invalid")
            if trace.get("robot_id") != "franka_panda" or trace.get("robot_prim_path") != prim_path:
                errors.append("isaac_render_options_policy_trace_robot_binding_invalid")
            if trace.get("controller_id") != "deterministic_franka_joint_position_pair.v1":
                errors.append("isaac_render_options_policy_trace_controller_invalid")
            if joint_names != expected_joint_names:
                errors.append("isaac_render_options_policy_trace_joint_names_invalid")

            def finite_vector(vector: Any) -> bool:
                return (
                    isinstance(vector, list)
                    and len(vector) == len(expected_joint_names)
                    and all(
                        not isinstance(item, bool)
                        and isinstance(item, (int, float))
                        and math.isfinite(float(item))
                        for item in vector
                    )
                )

            if not finite_vector(start):
                errors.append("isaac_render_options_policy_trace_start_invalid")
            if (
                not isinstance(trace.get("physics_dt_seconds"), (int, float))
                or isinstance(trace.get("physics_dt_seconds"), bool)
                or abs(float(trace.get("physics_dt_seconds", 0.0)) - (1.0 / 60.0)) > 1e-12
            ):
                errors.append("isaac_render_options_policy_trace_physics_dt_invalid")
            for key, low, high in (
                ("reset_settle_steps", 2, 600),
                ("sample_interval_steps", 1, 60),
            ):
                item = trace.get(key)
                if not isinstance(item, int) or isinstance(item, bool) or not low <= item <= high:
                    errors.append(f"isaac_render_options_policy_trace_{key}_invalid")
            threshold = trace.get("distinctness_threshold_rad")
            if (
                not isinstance(threshold, (int, float))
                or isinstance(threshold, bool)
                or not 0.01 <= float(threshold) <= 1.0
            ):
                errors.append("isaac_render_options_policy_trace_distinctness_threshold_invalid")
            start_tolerance = trace.get("identical_start_tolerance_rad")
            if (
                not isinstance(start_tolerance, (int, float))
                or isinstance(start_tolerance, bool)
                or not 0.0 <= float(start_tolerance) <= 0.05
            ):
                errors.append("isaac_render_options_policy_trace_start_tolerance_invalid")
            expected_policy_ids = ["franka-fixed-hold-v1", "franka-inspection-sweep-v1"]
            if (
                not isinstance(candidates, list)
                or len(candidates) != 2
                or [row.get("policy_id") for row in candidates if isinstance(row, Mapping)]
                != expected_policy_ids
            ):
                errors.append("isaac_render_options_policy_trace_candidates_invalid")
            elif finite_vector(start):
                for index, candidate in enumerate(candidates):
                    final = candidate.get("final_joint_positions_rad")
                    steps = candidate.get("duration_steps")
                    if not finite_vector(final):
                        errors.append(
                            f"isaac_render_options_policy_trace_candidate_{index}_target_invalid"
                        )
                    if (
                        not isinstance(steps, int)
                        or isinstance(steps, bool)
                        or not 30 <= steps <= 1_800
                    ):
                        errors.append(
                            f"isaac_render_options_policy_trace_candidate_{index}_steps_invalid"
                        )
                if all(finite_vector(row.get("final_joint_positions_rad")) for row in candidates):
                    hold = candidates[0]["final_joint_positions_rad"]
                    sweep = candidates[1]["final_joint_positions_rad"]
                    if max(abs(float(a) - float(b)) for a, b in zip(start, hold)) > 1e-9:
                        errors.append("isaac_render_options_policy_trace_hold_policy_invalid")
                    if max(abs(float(a) - float(b)) for a, b in zip(start, sweep)) < float(
                        threshold or 0.0
                    ):
                        errors.append("isaac_render_options_policy_trace_sweep_policy_not_distinct")
            camera = trace.get("egocentric_camera")
            if not isinstance(camera, Mapping):
                errors.append("isaac_render_options_policy_trace_egocentric_camera_missing")
            else:
                if camera.get("parent_link_name") != "panda_hand":
                    errors.append("isaac_render_options_policy_trace_camera_parent_invalid")
                for key in ("local_position_m", "local_target_m", "local_up"):
                    vector = camera.get(key)
                    if (
                        not isinstance(vector, list)
                        or len(vector) != 3
                        or any(
                            isinstance(item, bool)
                            or not isinstance(item, (int, float))
                            or not math.isfinite(float(item))
                            for item in (vector or [])
                        )
                    ):
                        errors.append(f"isaac_render_options_policy_trace_camera_{key}_invalid")
                for key, low, high in (("width", 64, 1280), ("height", 64, 1280)):
                    item = camera.get(key)
                    if (
                        not isinstance(item, int)
                        or isinstance(item, bool)
                        or not low <= item <= high
                    ):
                        errors.append(f"isaac_render_options_policy_trace_camera_{key}_invalid")
                fov = camera.get("fov_degrees")
                if (
                    not isinstance(fov, (int, float))
                    or isinstance(fov, bool)
                    or not 20.0 <= float(fov) <= 140.0
                ):
                    errors.append("isaac_render_options_policy_trace_camera_fov_invalid")
            if trace.get("physical_success_claimed") is not False:
                errors.append("isaac_render_options_policy_trace_physical_claim_forbidden")
    if errors:
        raise IsaacWorkerBundleError(errors)
    return dict(value)


def _write_zip_member(archive: zipfile.ZipFile, name: str, source: Path) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    with source.open("rb") as input_stream, archive.open(info, "w", force_zip64=True) as output:
        shutil.copyfileobj(input_stream, output, length=1024 * 1024)


def _command_for_request(request: Mapping[str, Any]) -> list[str]:
    command = [
        "/isaac-sim/python.sh",
        "/workspace/bundle/run_isaac_splat_nurec_render.py",
        "--usdz",
        "/workspace/bundle/reconstruction.usdz",
        "--cameras",
        "/workspace/bundle/fixed_cameras.json",
        "--out-dir",
        "/workspace/out",
        "--qualification-mode",
        "--package-digest",
        str(request["package_digest"]),
        "--verification-request-digest",
        str(request["isaac_verification_request_digest"]),
        "--camera-spec-digest",
        str(request["fixed_camera_spec_digest"]),
        "--runtime-container-image-digest",
        str(request["runtime_container_image_digest"]),
        "--runtime-implementation-digest",
        str(request["runtime_implementation_digest"]),
        "--physics-probe-steps",
        str(request["physics_probe_request"]["steps"]),
    ]
    probe = request["physics_probe_request"]
    if probe.get("ground_collider_prim"):
        command.extend(["--ground-collider-prim", str(probe["ground_collider_prim"])])
    if probe.get("ground_height_m") is not None:
        command.extend(["--ground-height", str(probe["ground_height_m"])])
    if probe.get("probe_xy_m") is not None:
        command.extend(["--probe-xy", *[str(value) for value in probe["probe_xy_m"]]])
    if request.get("schema_version") == PROVIDER_ISAAC_REQUEST_SCHEMA:
        command.extend(
            [
                "--provider-package-mode",
                "--expected-appearance-prim",
                str(request["expected_prim_paths"]["appearance"]),
                "--expected-collision-prim",
                str(request["expected_prim_paths"]["collision"]),
            ]
        )
    return command


def _validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") == PROVIDER_ISAAC_REQUEST_SCHEMA:
        try:
            return build_provider_nurec_isaac_request(value)
        except ExternalProviderNuRecError as exc:
            raise IsaacWorkerBundleError(
                [f"provider_isaac_verification_request_invalid:{code}" for code in exc.codes]
            ) from exc
    if value.get("schema_version") == EXTERNAL_SCENE_ISAAC_REQUEST_SCHEMA:
        try:
            return build_external_scene_isaac_verification_request(value)
        except ExternalSceneIsaacVerificationError as exc:
            raise IsaacWorkerBundleError(
                [f"external_scene_isaac_verification_request_invalid:{code}" for code in exc.codes]
            ) from exc
    try:
        return build_isaac_asset_verification_request(value)
    except IsaacReconstructionVerificationError as exc:
        raise IsaacWorkerBundleError(
            [f"isaac_verification_request_invalid:{code}" for code in exc.codes]
        ) from exc


def validate_isaac_verification_worker_bundle_receipt(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a non-authorizing Isaac bundle receipt before transport."""

    receipt = json.loads(json.dumps(dict(value)))
    errors: list[str] = []
    if receipt.get("schema_version") not in {
        ISAAC_WORKER_BUNDLE_SCHEMA,
        PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA,
    }:
        errors.append("isaac_bundle_receipt_schema_invalid")
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        errors.append("isaac_bundle_receipt_digest_mismatch")
    for key in (
        "isaac_verification_request_digest",
        "package_digest",
        "fixed_camera_spec_digest",
        "runtime_implementation_digest",
        "bundle_manifest_digest",
        "bundle_digest",
    ):
        text = str(receipt.get(key) or "")
        if _DIGEST.fullmatch(text) is None:
            errors.append(f"isaac_bundle_receipt_{key}_invalid")
    if _COMMIT.fullmatch(str(receipt.get("source_commit_sha") or "")) is None:
        errors.append("isaac_bundle_receipt_source_commit_invalid")
    expected_runtime = (
        "provider_nurec_isaac_runtime_result.v1"
        if receipt.get("schema_version") == PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA
        else "isaac_splat_nurec_render_result.v3"
    )
    render_options_digest = receipt.get("render_options_digest")
    expected_member_count = 6 if render_options_digest is not None else 5
    if render_options_digest is not None and _DIGEST.fullmatch(str(render_options_digest)) is None:
        errors.append("isaac_bundle_receipt_render_options_digest_invalid")
    if (
        receipt.get("expected_runtime_schema") != expected_runtime
        or receipt.get("raw_secret_values_included") is not False
        or receipt.get("provider_allocation_performed") is not False
        or receipt.get("paid_execution_authorized_by_bundle") is not False
        or receipt.get("proof_effect") != "none"
        or receipt.get("claim_ceiling") != "request_only"
        or receipt.get("bundle_member_count") != expected_member_count
    ):
        errors.append("isaac_bundle_receipt_boundary_invalid")
    command = receipt.get("command")
    if (
        not isinstance(command, list)
        or len(command) < 21
        or command[:2]
        != [
            "/isaac-sim/python.sh",
            "/workspace/bundle/run_isaac_splat_nurec_render.py",
        ]
        or any(not isinstance(item, str) or not item for item in command)
    ):
        errors.append("isaac_bundle_receipt_command_invalid")
    if errors:
        raise IsaacWorkerBundleError(errors)
    return receipt


def compile_isaac_verification_worker_bundle(
    *,
    verification_request: Mapping[str, Any],
    package_artifact_root: str | Path,
    fixed_camera_spec_path: str | Path,
    runner_path: str | Path,
    output_root: str | Path,
    render_options_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile a deterministic exact-package Isaac input bundle without spending."""

    request = _validate_request(verification_request)
    provider_request = request["schema_version"] == PROVIDER_ISAAC_REQUEST_SCHEMA
    external_scene_request = request["schema_version"] == EXTERNAL_SCENE_ISAAC_REQUEST_SCHEMA
    bundle_schema = (
        PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA if provider_request else ISAAC_WORKER_BUNDLE_SCHEMA
    )
    request_member = (
        "provider_nurec_isaac_verification_request.v1.json"
        if provider_request
        else (
            "external_scene_isaac_verification_request.v1.json"
            if external_scene_request
            else "isaac_asset_verification_request.v1.json"
        )
    )
    package_root = Path(package_artifact_root)
    if package_root.is_symlink() or not package_root.is_dir():
        raise IsaacWorkerBundleError(["isaac_package_root_invalid"])
    package_root = package_root.resolve()
    package = _safe_source(
        package_root,
        request["package_artifact_reference"],
        request["package_digest"],
        ".usdz",
        "isaac_package",
    )
    cameras = _explicit_file(
        fixed_camera_spec_path,
        digest=request["fixed_camera_spec_digest"],
        suffix=".json",
        code="isaac_camera_spec",
    )
    try:
        camera_rows = json.loads(cameras.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsaacWorkerBundleError(["isaac_camera_spec_json_invalid"]) from exc
    if (
        not isinstance(camera_rows, list)
        or [row.get("id") for row in camera_rows if isinstance(row, Mapping)]
        != request["fixed_camera_ids"]
    ):
        raise IsaacWorkerBundleError(["isaac_camera_spec_ids_mismatch"])
    runner = _explicit_file(
        runner_path,
        digest=request["runtime_implementation_digest"],
        suffix=".py",
        code="isaac_runner",
    )
    declared_render_options_digest = request.get("render_options_digest")
    if (declared_render_options_digest is None) != (render_options_path is None):
        raise IsaacWorkerBundleError(["isaac_render_options_binding_incomplete"])
    render_options: Path | None = None
    if render_options_path is not None:
        render_options = _explicit_file(
            render_options_path,
            digest=str(declared_render_options_digest),
            suffix=".json",
            code="isaac_render_options",
        )
        _validate_render_options(render_options)
    total = sum(
        path.stat().st_size
        for path in (package, cameras, runner, render_options)
        if path is not None
    )
    if total > MAX_BUNDLE_TOTAL_BYTES:
        raise IsaacWorkerBundleError(["isaac_worker_bundle_oversized"])

    destination = Path(output_root)
    if destination.is_symlink():
        raise IsaacWorkerBundleError(["isaac_bundle_output_root_symlink_forbidden"])
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    content_id = request["isaac_verification_request_digest"][7:]
    final = destination / content_id
    receipt_filename = (
        "provider_nurec_isaac_worker_bundle.v1.json"
        if provider_request
        else "isaac_verification_worker_bundle.v1.json"
    )
    receipt_path = final / receipt_filename
    bundle_path = final / "isaac_verification_worker_bundle.zip"
    if final.exists():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IsaacWorkerBundleError(["isaac_bundle_existing_output_tampered"]) from exc
        try:
            observed_bundle_digest = _sha256(bundle_path)
        except OSError as exc:
            raise IsaacWorkerBundleError(["isaac_bundle_existing_output_tampered"]) from exc
        try:
            receipt = validate_isaac_verification_worker_bundle_receipt(receipt)
        except IsaacWorkerBundleError as exc:
            raise IsaacWorkerBundleError(["isaac_bundle_existing_output_tampered"]) from exc
        if receipt.get("bundle_digest") != observed_bundle_digest:
            raise IsaacWorkerBundleError(["isaac_bundle_replay_digest_mismatch"])
        return receipt

    temporary = Path(tempfile.mkdtemp(prefix=".isaac-bundle-", dir=destination))
    try:
        request_path = temporary / request_member
        write_json(request_path, request)
        command = _command_for_request(request)
        manifest = {
            "schema_version": bundle_schema,
            "isaac_verification_request_digest": request["isaac_verification_request_digest"],
            "package_digest": request["package_digest"],
            "fixed_camera_spec_digest": request["fixed_camera_spec_digest"],
            "runtime_implementation_digest": request["runtime_implementation_digest"],
            "runtime_container_image_digest": request["runtime_container_image_digest"],
            "source_commit_sha": request["source_commit_sha"],
            "fixed_camera_ids": request["fixed_camera_ids"],
            "command": command,
            "expected_runtime_schema": request.get(
                "expected_runtime_schema", "isaac_splat_nurec_render_result.v3"
            ),
            "verification_request_member": request_member,
            "raw_secret_values_included": False,
            "provider_allocation_performed": False,
            "paid_execution_authorized_by_bundle": False,
            "canonical_allocator_command": (
                "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
            ),
            "proof_effect": "none",
            "claim_ceiling": "request_only",
        }
        if declared_render_options_digest is not None:
            manifest["render_options_digest"] = declared_render_options_digest
        manifest["bundle_manifest_digest"] = canonical_digest(
            manifest, digest_field="bundle_manifest_digest"
        )
        manifest_path = temporary / "bundle_manifest.json"
        write_json(manifest_path, manifest)
        archive_path = temporary / "isaac_verification_worker_bundle.zip"
        with zipfile.ZipFile(archive_path, "w", allowZip64=True) as archive:
            members: list[tuple[str, Path]] = [
                ("bundle_manifest.json", manifest_path),
                ("fixed_cameras.json", cameras),
                (request_member, request_path),
                ("reconstruction.usdz", package),
                ("run_isaac_splat_nurec_render.py", runner),
            ]
            if render_options is not None:
                members.append(("render_options.json", render_options))
            for name, source in members:
                _write_zip_member(archive, name, source)
        bundle_digest = _sha256(archive_path)
        receipt = {
            **manifest,
            "bundle_digest": bundle_digest,
            "bundle_artifact_reference": f"{content_id}/isaac_verification_worker_bundle.zip",
            "bundle_member_count": len(members),
            "bundle_bytes": archive_path.stat().st_size,
            "cost_usd": 0.0,
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        write_json(temporary / receipt_filename, receipt)
        os.replace(temporary, final)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def extract_isaac_verification_worker_bundle(
    *,
    bundle_path: str | Path,
    bundle_receipt: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    """Materialize exact Isaac inputs without trusting archive paths or metadata."""

    receipt = validate_isaac_verification_worker_bundle_receipt(bundle_receipt)
    source = Path(bundle_path)
    if source.is_symlink():
        raise IsaacWorkerBundleError(["isaac_bundle_archive_symlink_forbidden"])
    try:
        source = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IsaacWorkerBundleError(["isaac_bundle_archive_missing"]) from exc
    if (
        not source.is_file()
        or source.stat().st_size > MAX_BUNDLE_TOTAL_BYTES
        or _sha256(source) != receipt["bundle_digest"]
    ):
        raise IsaacWorkerBundleError(["isaac_bundle_archive_binding_invalid"])
    destination = Path(output_root)
    if destination.is_symlink():
        raise IsaacWorkerBundleError(["isaac_bundle_extraction_root_symlink_forbidden"])
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    final = destination / receipt["bundle_digest"][7:]
    extraction_path = final / "isaac_verification_worker_bundle_extraction.v1.json"
    if final.exists():
        try:
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IsaacWorkerBundleError(["isaac_bundle_extraction_replay_invalid"]) from exc
        if extraction.get("bundle_digest") != receipt["bundle_digest"] or extraction.get(
            "extraction_receipt_digest"
        ) != canonical_digest(extraction, digest_field="extraction_receipt_digest"):
            raise IsaacWorkerBundleError(["isaac_bundle_extraction_replay_tampered"])
        for row in extraction.get("extracted_members") or []:
            target = final.joinpath(*PurePosixPath(str(row.get("archive_path") or "")).parts)
            if target.is_symlink() or not target.is_file() or _sha256(target) != row.get("digest"):
                raise IsaacWorkerBundleError(["isaac_bundle_extraction_replay_tampered"])
        return extraction

    request_member = str(
        receipt.get("verification_request_member") or "isaac_asset_verification_request.v1.json"
    )
    expected_names = {
        "bundle_manifest.json",
        "fixed_cameras.json",
        request_member,
        "reconstruction.usdz",
        "run_isaac_splat_nurec_render.py",
    }
    if receipt.get("render_options_digest") is not None:
        expected_names.add("render_options.json")
    temporary = Path(tempfile.mkdtemp(prefix=".isaac-extract-", dir=destination))
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            errors: list[str] = []
            total = 0
            if len(names) != len(set(names)) or set(names) != expected_names:
                errors.append("isaac_bundle_archive_inventory_invalid")
            for member in members:
                relative = PurePosixPath(member.filename.replace("\\", "/"))
                if (
                    relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or member.compress_type != zipfile.ZIP_STORED
                    or member.file_size > MAX_BUNDLE_MEMBER_BYTES
                ):
                    errors.append("isaac_bundle_archive_member_unsafe")
                total += member.file_size
            if total > MAX_BUNDLE_TOTAL_BYTES:
                errors.append("isaac_bundle_archive_uncompressed_oversized")
            if errors:
                raise IsaacWorkerBundleError(errors)
            extracted_rows: list[dict[str, Any]] = []
            for name in sorted(expected_names):
                target = temporary.joinpath(*PurePosixPath(name).parts)
                digest = hashlib.sha256()
                written = 0
                with archive.open(name, "r") as input_stream, target.open("wb") as output:
                    while True:
                        chunk = input_stream.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_BUNDLE_MEMBER_BYTES:
                            raise IsaacWorkerBundleError(
                                ["isaac_bundle_extraction_member_oversized"]
                            )
                        digest.update(chunk)
                        output.write(chunk)
                extracted_rows.append(
                    {
                        "archive_path": name,
                        "digest": "sha256:" + digest.hexdigest(),
                        "bytes": written,
                    }
                )
        manifest = json.loads((temporary / "bundle_manifest.json").read_text())
        request = _validate_request(json.loads((temporary / request_member).read_text()))
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("bundle_manifest_digest")
            != canonical_digest(manifest, digest_field="bundle_manifest_digest")
            or manifest.get("bundle_manifest_digest") != receipt["bundle_manifest_digest"]
            or request["isaac_verification_request_digest"]
            != receipt["isaac_verification_request_digest"]
            or _command_for_request(request) != receipt["command"]
            or _sha256(temporary / "reconstruction.usdz") != receipt["package_digest"]
            or _sha256(temporary / "fixed_cameras.json") != receipt["fixed_camera_spec_digest"]
            or _sha256(temporary / "run_isaac_splat_nurec_render.py")
            != receipt["runtime_implementation_digest"]
            or (
                receipt.get("render_options_digest") is not None
                and _sha256(temporary / "render_options.json") != receipt["render_options_digest"]
            )
        ):
            raise IsaacWorkerBundleError(["isaac_bundle_extraction_binding_invalid"])
        if receipt.get("render_options_digest") is not None:
            _validate_render_options(temporary / "render_options.json")
        extraction = {
            "schema_version": ISAAC_WORKER_EXTRACTION_SCHEMA,
            "status": "extracted",
            "bundle_digest": receipt["bundle_digest"],
            "bundle_manifest_digest": receipt["bundle_manifest_digest"],
            "isaac_verification_request_digest": receipt["isaac_verification_request_digest"],
            "extracted_members": extracted_rows,
            "raw_secret_values_extracted": False,
            "provider_allocation_inferred": False,
            "proof_effect": "none",
            "claim_ceiling": "isaac_candidate_input_materialization_only",
        }
        extraction["extraction_receipt_digest"] = canonical_digest(
            extraction, digest_field="extraction_receipt_digest"
        )
        write_json(
            temporary / "isaac_verification_worker_bundle_extraction.v1.json",
            extraction,
        )
        os.replace(temporary, final)
        return extraction
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "ISAAC_WORKER_BUNDLE_SCHEMA",
    "PROVIDER_ISAAC_WORKER_BUNDLE_SCHEMA",
    "ISAAC_WORKER_EXTRACTION_SCHEMA",
    "IsaacWorkerBundleError",
    "compile_isaac_verification_worker_bundle",
    "extract_isaac_verification_worker_bundle",
    "validate_isaac_verification_worker_bundle_receipt",
]
