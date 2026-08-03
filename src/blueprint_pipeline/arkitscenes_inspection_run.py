"""Prepare a retained ARKitScenes scripted-inspection Task Evaluation Run.

The producer is deliberately development-only.  It verifies the public-dataset
source files, turns the retained depth surface into separate visualization and
collision candidates, regenerates a visible-object target with the current
rendered-scene orchestration contract, proposes a Franka placement, and authors
an Isaac package plus an exact five-controller run packet.  It never treats the
public dataset as Blueprint Raw Contract evidence and it never manufactures an
Isaac result when a compatible runtime is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh

from .decision_evidence_contracts import canonical_digest, canonical_json
from .external_scene_collision_candidate import compile_external_scene_collision_candidate
from .external_scene_isaac_package import compile_external_scene_isaac_package
from .external_scene_robot_placement import propose_external_scene_robot_placement
from .gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from .rendered_scene_task_target_orchestrator import (
    RenderedSceneAnalyzerBackend,
    compile_rendered_scene_task_target_with_analyzer,
)
from .torchvision_rendered_scene_analyzer import (
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_WEIGHT_SHA256,
    analyze_payload,
    build_analyzer_contract,
)


SOURCE_PROFILE_SCHEMA = "arkitscenes_public_dataset_source_profile.v1"
SOURCE_VERIFICATION_SCHEMA = "arkitscenes_source_digest_verification.v1"
GEOMETRY_ADAPTER_SCHEMA = "arkitscenes_depth_geometry_support_adapter.v1"
FRAME_BINDING_SCHEMA = "arkitscenes_same_source_frame_binding.v1"
ISAAC_PACKET_SCHEMA = "arkitscenes_scripted_inspection_isaac_packet.v1"
TERMINAL_REPORT_SCHEMA = "arkitscenes_scripted_inspection_terminal_report.v1"
RUN_MANIFEST_SCHEMA = "arkitscenes_scripted_inspection_run_manifest.v1"
DEFAULT_MAX_SPEND_USD = 1.0
DEFAULT_TTL_SECONDS = 3600
EXPECTED_CONTROLLER_COUNT = 5
EXPECTED_CONTROLLER_IDS = (
    "franka-inspection-center-hold-v1",
    "franka-inspection-left-narrow-v1",
    "franka-inspection-right-narrow-v1",
    "franka-inspection-left-wide-v1",
    "franka-inspection-right-wide-v1",
)


class ArkitScenesInspectionRunError(ValueError):
    """Stable failure for malformed, drifted, or unsafe run preparation."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArkitScenesInspectionRunError([f"json_invalid:{path.name}"]) from exc
    if not isinstance(value, dict):
        raise ArkitScenesInspectionRunError([f"json_object_required:{path.name}"])
    return value


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise ArkitScenesInspectionRunError([f"immutable_output_conflict:{path.name}"])


def _canonical_receipt(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    field = "arkitscenes_proxy_compilation_digest"
    if value.get("schema_version") != "arkitscenes_raw_proxy_compilation.v1":
        raise ArkitScenesInspectionRunError(["arkitscenes_proxy_schema_invalid"])
    if value.get(field) != canonical_digest(value, digest_field=field):
        raise ArkitScenesInspectionRunError(["arkitscenes_proxy_digest_mismatch"])
    return value


def verify_retained_sources(
    *, scene_root: Path, selected_compilation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every retained source against every recorded compilation receipt."""

    root = scene_root.resolve(strict=True)
    selected_path = selected_compilation_path.resolve(strict=True)
    try:
        selected_path.relative_to(root)
    except ValueError as exc:
        raise ArkitScenesInspectionRunError(["selected_compilation_outside_scene_root"]) from exc
    receipts = sorted(root.glob("compiled/*/arkitscenes_raw_proxy_compilation.json"))
    if selected_path not in [path.resolve() for path in receipts]:
        raise ArkitScenesInspectionRunError(["selected_compilation_not_retained"])
    receipt_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    reference_sets: set[str] = set()
    for receipt_path in receipts:
        receipt = _canonical_receipt(receipt_path)
        references = receipt.get("original_file_references")
        if not isinstance(references, list) or not references:
            raise ArkitScenesInspectionRunError(["arkitscenes_source_references_missing"])
        verified: list[dict[str, Any]] = []
        for raw in references:
            if not isinstance(raw, Mapping):
                raise ArkitScenesInspectionRunError(["arkitscenes_source_reference_invalid"])
            relative = Path(str(raw.get("relative_path") or ""))
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ArkitScenesInspectionRunError(["arkitscenes_source_reference_unsafe"])
            source = (root / relative).resolve(strict=True)
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise ArkitScenesInspectionRunError(["arkitscenes_source_outside_root"]) from exc
            observed_digest = _sha256(source)
            observed_size = source.stat().st_size
            if observed_digest != raw.get("digest") or observed_size != raw.get("size_bytes"):
                raise ArkitScenesInspectionRunError(
                    [f"arkitscenes_source_digest_or_size_mismatch:{relative.as_posix()}"]
                )
            verified.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": observed_size,
                    "digest": observed_digest,
                    "digest_matches_recorded": True,
                    "size_matches_recorded": True,
                }
            )
        normalized = canonical_digest(
            {"files": sorted(verified, key=lambda row: row["relative_path"])}
        )
        reference_sets.add(normalized)
        receipt_rows.append(
            {
                "relative_path": receipt_path.resolve().relative_to(root).as_posix(),
                "compilation_digest": receipt["arkitscenes_proxy_compilation_digest"],
                "source_capture_digest": receipt["source_capture_digest"],
                "file_count": len(verified),
                "all_files_verified": True,
            }
        )
        if receipt_path.resolve() == selected_path:
            selected = receipt
    if selected is None or len(reference_sets) != 1:
        raise ArkitScenesInspectionRunError(["arkitscenes_recorded_source_sets_disagree"])
    report = {
        "schema_version": SOURCE_VERIFICATION_SCHEMA,
        "status": "verified",
        "scene_id": selected["source_capture_identity"],
        "source_capture_digest": selected["source_capture_digest"],
        "selected_compilation_digest": selected["arkitscenes_proxy_compilation_digest"],
        "retained_compilation_receipts_checked": receipt_rows,
        "retained_compilation_receipt_count": len(receipt_rows),
        "recorded_source_sets_agree": True,
        "all_recorded_source_files_match": True,
        "verification_effect": "local_file_integrity_only",
    }
    report["verification_digest"] = canonical_digest(report, digest_field="verification_digest")
    return selected, report


def build_public_dataset_source_profile(
    *, compilation: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    if verification.get("status") != "verified":
        raise ArkitScenesInspectionRunError(["arkitscenes_source_not_verified"])
    profile = {
        "schema_version": SOURCE_PROFILE_SCHEMA,
        "status": "admitted_provider_derived_support",
        "source_capture_identity": compilation["source_capture_identity"],
        "source_capture_digest": compilation["source_capture_digest"],
        "source_compilation_digest": compilation["arkitscenes_proxy_compilation_digest"],
        "source_verification_digest": verification["verification_digest"],
        "source_class": "public-dataset proxy",
        "provider": "ARKitScenes public dataset",
        "capture_device_class": "public_dataset_iPad",
        "rights_and_authority": dict(compilation.get("authority_used") or {}),
        "metric_scale_status": "dataset_declared_not_independently_validated",
        "coordinate_frame_declaration": dict(compilation.get("coordinate_frame_declaration") or {}),
        "claim_boundary": {
            "provider_derived_support": True,
            "blueprint_raw_contract_truth": False,
            "blueprint_raw_contract_v3_2_proven": False,
            "iphone_route_proven": False,
            "public_dataset_proxy": True,
        },
        "proof_effect": "public_dataset_source_admission_only",
        "claim_ceiling": "public_dataset_calibrated_observation_proxy",
    }
    profile["source_profile_digest"] = canonical_digest(
        profile, digest_field="source_profile_digest"
    )
    return profile


def _surface_arrays(surface: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vertex_rows = surface.get("vertices")
    face_rows = surface.get("faces")
    if not isinstance(vertex_rows, list) or not isinstance(face_rows, list):
        raise ArkitScenesInspectionRunError(["arkitscenes_surface_arrays_missing"])
    identifiers: dict[str, int] = {}
    vertices = np.empty((len(vertex_rows), 3), dtype=np.float32)
    for index, row in enumerate(vertex_rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("position_m"), list):
            raise ArkitScenesInspectionRunError(["arkitscenes_surface_vertex_invalid"])
        identifiers[str(row.get("vertex_id") or "")] = index
        vertices[index] = np.asarray(row["position_m"], dtype=np.float32)
    faces = np.empty((len(face_rows), 3), dtype=np.int64)
    try:
        for index, row in enumerate(face_rows):
            faces[index] = [identifiers[str(item)] for item in row["vertex_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArkitScenesInspectionRunError(["arkitscenes_surface_face_invalid"]) from exc
    if not np.isfinite(vertices).all():
        raise ArkitScenesInspectionRunError(["arkitscenes_surface_nonfinite"])
    return vertices, faces


def adapt_depth_surface(
    *, surface_path: Path, output_root: Path, source_capture_digest: str
) -> tuple[dict[str, Any], Path, Path]:
    surface = _load_json(surface_path.resolve(strict=True))
    if surface.get("source_capture_digest") != source_capture_digest:
        raise ArkitScenesInspectionRunError(["arkitscenes_surface_source_mismatch"])
    vertices, faces = _surface_arrays(surface)
    count = len(vertices)
    splat_path = output_root / "derived_geometry" / "arkit_depth_analysis_splat.ply"
    splat_path.parent.mkdir(parents=True, exist_ok=True)
    write_standard_3dgs_ply(
        SplatData(
            count=count,
            xyz=vertices,
            opacity=np.full(count, math.log(0.99 / 0.01), dtype=np.float32),
            f_dc=np.zeros((count, 3), dtype=np.float32),
            scales=np.full((count, 3), math.log(0.0125), dtype=np.float32),
            quats=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
            properties=(),
        ),
        splat_path,
    )
    glb_path = output_root / "derived_geometry" / "arkit_depth_collision_source.glb"
    glb_path.write_bytes(
        trimesh.Scene(trimesh.Trimesh(vertices=vertices, faces=faces, process=False)).export(
            file_type="glb"
        )
    )
    adapter = {
        "schema_version": GEOMETRY_ADAPTER_SCHEMA,
        "status": "derived_support_candidates_compiled",
        "source_surface_digest": _sha256(surface_path),
        "source_capture_digest": source_capture_digest,
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "analysis_splat_digest": _sha256(splat_path),
        "collision_source_digest": _sha256(glb_path),
        "coordinate_frame_declaration": dict(surface.get("coordinate_frame_declaration") or {}),
        "metric_scale_status": str(surface.get("metric_scale_status") or "unverified"),
        "generated_fill_used": bool(surface.get("generated_fill_used")),
        "claim_boundary": {
            "depth_geometry_is_derived_support_only": True,
            "appearance_reconstruction_proven": False,
            "metric_scale_independently_validated": False,
            "collision_qualified": False,
            "coordinate_convention_independently_validated": False,
        },
        "proof_effect": "derived_depth_geometry_transport_only",
    }
    adapter["adapter_digest"] = canonical_digest(adapter, digest_field="adapter_digest")
    return adapter, splat_path, glb_path


def _camera_from_observation(
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    camera = observation.get("camera")
    if not isinstance(camera, Mapping):
        raise ArkitScenesInspectionRunError(["arkitscenes_camera_missing"])
    matrix = np.asarray(camera.get("T_world_camera"), dtype=np.float64)
    intrinsics = camera.get("rgb_intrinsics")
    if matrix.shape != (4, 4) or not isinstance(intrinsics, Mapping):
        raise ArkitScenesInspectionRunError(["arkitscenes_camera_invalid"])
    position = matrix[:3, 3]
    forward = matrix[:3, 2]
    up = -matrix[:3, 1]
    height = int(intrinsics["height"])
    width = int(intrinsics["width"])
    vertical_fov = math.degrees(2.0 * math.atan(height / (2.0 * float(intrinsics["fy"]))))
    portable = {
        "pos": [round(float(item), 12) for item in position],
        "target": [round(float(item), 12) for item in position + forward],
        "up": [round(float(item), 12) for item in up],
        "fov": round(vertical_fov, 12),
        "source_convention": "arkitscenes_camera_to_world_opencv_x_right_y_down_z_forward",
    }
    return portable, {"width": width, "height": height}


def _rendered_views(
    *, compilation_root: Path, camera_manifest: Mapping[str, Any], view_ids: Sequence[str]
) -> list[dict[str, Any]]:
    observations = camera_manifest.get("observations")
    if not isinstance(observations, list):
        raise ArkitScenesInspectionRunError(["arkitscenes_camera_observations_missing"])
    by_id = {
        str(row.get("observation_id") or ""): row
        for row in observations
        if isinstance(row, Mapping)
    }
    views: list[dict[str, Any]] = []
    for view_id in view_ids:
        observation = by_id.get(view_id)
        if observation is None:
            raise ArkitScenesInspectionRunError([f"arkitscenes_view_missing:{view_id}"])
        relative = Path(str(observation.get("image_relative_path") or ""))
        image = (compilation_root / relative).resolve(strict=True)
        try:
            image.relative_to(compilation_root.resolve())
        except ValueError as exc:
            raise ArkitScenesInspectionRunError(["arkitscenes_view_outside_compilation"]) from exc
        if _sha256(image) != observation.get("image_digest"):
            raise ArkitScenesInspectionRunError([f"arkitscenes_view_digest_mismatch:{view_id}"])
        camera, image_size = _camera_from_observation(observation)
        views.append(
            {
                "view_id": view_id,
                "rgb_path": str(image),
                "rgb_digest": observation["image_digest"],
                "observation_source": "raw_capture",
                "camera_spec_digest": canonical_digest(camera),
                "image_size": image_size,
                "camera": camera,
            }
        )
    return views


def _default_analyzer(
    weights_path: Path, *, minimum_visual_confidence: float
) -> RenderedSceneAnalyzerBackend:
    contract = build_analyzer_contract()

    def backend(request: Mapping[str, Any], runtime_inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return analyze_payload(
            {"analyzer_request": dict(request), "runtime_inputs": dict(runtime_inputs)},
            weights_path=weights_path,
            score_threshold=minimum_visual_confidence,
            maximum_proposals=4,
        )

    setattr(backend, "contract", contract)
    return backend


def _frame_binding(*, geometry: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": FRAME_BINDING_SCHEMA,
        "status": "same_source_lineage_bound_coordinate_convention_unqualified",
        "source_surface_digest": geometry["source_surface_digest"],
        "analysis_splat_digest": geometry["analysis_splat_digest"],
        "collision_source_digest": geometry["collision_source_digest"],
        "source_to_collision_stage_matrix": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "transform_convention": "right_handed_source_Y_up_to_Isaac_Z_up_rx_plus_90",
        "lineage_match": True,
        "independent_coordinate_qualification": False,
        "metric_scale_qualification": False,
        "proof_effect": "same_derived_surface_coordinate_transport_only",
    }
    value["scene_frame_binding_digest"] = canonical_digest(
        value, digest_field="scene_frame_binding_digest"
    )
    return value


def _to_collision_stage(position: Sequence[float]) -> list[float]:
    x, y, z = [float(item) for item in position]
    return [x, -z, y]


def _runtime_available() -> dict[str, Any]:
    isaac_binary = shutil.which("isaac-sim")
    nvidia_smi = shutil.which("nvidia-smi")
    compatible = bool(isaac_binary and nvidia_smi and platform.machine() in {"x86_64", "amd64"})
    return {
        "host_platform": platform.system(),
        "host_machine": platform.machine(),
        "isaac_sim_binary": isaac_binary,
        "nvidia_smi_binary": nvidia_smi,
        "compatible_local_nvidia_isaac_runtime_available": compatible,
    }


def compile_arkitscenes_inspection_run(
    *,
    scene_root: str | Path,
    selected_compilation_path: str | Path,
    output_root: str | Path,
    implementation_source_commit_sha: str,
    view_ids: Sequence[str],
    allowed_object_labels: Sequence[str],
    weights_path: str | Path | None = None,
    analyzer_backend: RenderedSceneAnalyzerBackend | None = None,
    analyzer_contract: Mapping[str, Any] | None = None,
    minimum_visual_confidence: float = 0.5,
    maximum_spend_usd: float = DEFAULT_MAX_SPEND_USD,
    hard_ttl_seconds: int = DEFAULT_TTL_SECONDS,
    paid_runtime_authority: str | None = None,
) -> dict[str, Any]:
    """Compile all locally available evidence and stop at a truthful runtime gate."""

    root = Path(scene_root).resolve(strict=True)
    selected_path = Path(selected_compilation_path).resolve(strict=True)
    destination = Path(output_root).resolve()
    if destination.exists() and destination.is_symlink():
        raise ArkitScenesInspectionRunError(["output_root_symlink_forbidden"])
    if destination.exists() and any(destination.iterdir()):
        raise ArkitScenesInspectionRunError(["output_root_must_be_empty"])
    if not view_ids or not allowed_object_labels:
        raise ArkitScenesInspectionRunError(["inspection_views_and_labels_required"])
    if re.fullmatch(r"[0-9a-f]{40}", implementation_source_commit_sha) is None:
        raise ArkitScenesInspectionRunError(["implementation_source_commit_sha_invalid"])
    if maximum_spend_usd <= 0 or hard_ttl_seconds < 300:
        raise ArkitScenesInspectionRunError(["paid_runtime_request_limits_invalid"])
    if paid_runtime_authority is not None and not paid_runtime_authority.strip():
        raise ArkitScenesInspectionRunError(["paid_runtime_authority_invalid"])
    if not 0.0 <= minimum_visual_confidence <= 1.0:
        raise ArkitScenesInspectionRunError(["minimum_visual_confidence_invalid"])
    destination.mkdir(parents=True, exist_ok=True)

    compilation, verification = verify_retained_sources(
        scene_root=root, selected_compilation_path=selected_path
    )
    source_profile = build_public_dataset_source_profile(
        compilation=compilation, verification=verification
    )
    compilation_root = selected_path.parent
    surface_result_path = (
        compilation_root / "observed_surface_proxy_v1" / ("arkit_depth_surface_proxy_result.json")
    )
    surface_result = _load_json(surface_result_path)
    surface_relative = Path(
        str(dict(surface_result.get("surface_asset") or {}).get("relative_path"))
    )
    surface_path = (root / surface_relative).resolve(strict=True)
    if _sha256(surface_path) != dict(surface_result["surface_asset"])["digest"]:
        raise ArkitScenesInspectionRunError(["arkitscenes_surface_digest_mismatch"])
    geometry, analysis_splat, collision_glb = adapt_depth_surface(
        surface_path=surface_path,
        output_root=destination,
        source_capture_digest=compilation["source_capture_digest"],
    )
    frame_binding = _frame_binding(geometry=geometry)
    camera_manifest_path = compilation_root / "camera_observations_proxy.json"
    camera_manifest = _load_json(camera_manifest_path)
    if camera_manifest.get("camera_observation_digest") != canonical_digest(
        camera_manifest, digest_field="camera_observation_digest"
    ):
        raise ArkitScenesInspectionRunError(["arkitscenes_camera_manifest_digest_mismatch"])
    views = _rendered_views(
        compilation_root=compilation_root,
        camera_manifest=camera_manifest,
        view_ids=view_ids,
    )

    if analyzer_backend is None:
        if weights_path is None:
            raise ArkitScenesInspectionRunError(["torchvision_weights_required"])
        weights = Path(weights_path).resolve(strict=True)
        if _sha256(weights) != "sha256:" + MODEL_WEIGHT_SHA256:
            raise ArkitScenesInspectionRunError(["torchvision_weights_digest_mismatch"])
        analyzer_backend = _default_analyzer(
            weights, minimum_visual_confidence=minimum_visual_confidence
        )
        analyzer_contract = build_analyzer_contract()
    if analyzer_contract is None:
        raise ArkitScenesInspectionRunError(["analyzer_contract_required"])
    target = compile_rendered_scene_task_target_with_analyzer(
        analyzer_backend=analyzer_backend,
        analyzer_id=str(analyzer_contract.get("analyzer_id") or ""),
        analyzer_implementation_version=str(
            analyzer_contract.get("implementation_version") or MODEL_IMPLEMENTATION_VERSION
        ),
        analyzer_contract_digest=str(analyzer_contract.get("analyzer_contract_digest") or ""),
        analysis_splat_path=analysis_splat,
        scene_id=str(compilation["source_capture_identity"]),
        source_scene_digest=geometry["source_surface_digest"],
        rendered_views=views,
        source_video_available=True,
        robot_id="franka_panda",
        metric_scale_status="sensor_metric_unvalidated",
        collision_support={
            "status": "candidate_compiled",
            "collision_digest": geometry["collision_source_digest"],
            "source_scene_digest": geometry["source_surface_digest"],
        },
        reach_support={"status": "not_checked"},
        task_context={
            "site_task_intent": "inspect the selected visible object without contact",
            "requested_interaction_mode": "inspection_only",
            "allowed_object_labels": list(allowed_object_labels),
        },
        minimum_visual_confidence=minimum_visual_confidence,
        minimum_projected_splats=16,
    )
    if target.get("status") != "target_ready_for_bounded_sim":
        raise ArkitScenesInspectionRunError(
            list(target.get("blockers") or ["no_qualified_3d_task_target"])
        )
    selected_target = dict(target["target_analysis"]["selected_target"])
    binding_row = next(
        row
        for row in target["binding_results"]
        if row["proposal_id"] == selected_target["proposal_id"]
    )
    binding = dict(binding_row["binding"])

    collision_usd = destination / "scene" / "arkit_depth_collision.usda"
    collision_result = compile_external_scene_collision_candidate(
        source_path=collision_glb,
        request={
            "schema_version": "external_scene_collision_compilation_request.v1",
            "source_asset_digest": geometry["collision_source_digest"],
            "source_format": "glb",
            "source_coordinate_frame": {"up_axis": "Y", "handedness": "right"},
            "metric_scale_status": "sensor_metric_unvalidated",
            "source_video_available": True,
            "generated_fill_allowed": False,
            "collision_validated": False,
        },
        output_path=collision_usd,
    )
    placement_packet = propose_external_scene_robot_placement(
        collision_glb_path=collision_glb,
        request={
            "schema_version": "external_scene_robot_placement_request.v1",
            "robot_id": "franka_panda",
            "source_scene_digest": geometry["source_surface_digest"],
            "target_analysis_digest": target["target_analysis"]["target_analysis_digest"],
            "target_binding_digest": binding["binding_evidence_digest"],
            "scene_frame_binding_digest": frame_binding["scene_frame_binding_digest"],
            "collision_candidate_digest": collision_result["collision_candidate_digest"],
            "collision_source_digest": geometry["collision_source_digest"],
            "target_label": selected_target["object_label"],
            "visual_confidence": selected_target["visual_confidence"],
            "target_position_collision_stage": _to_collision_stage(
                selected_target["target_position_scene"]
            ),
            "target_spatial_uncertainty_stage_units": selected_target[
                "spatial_uncertainty_scene_units"
            ],
            "metric_scale_status": "provider_declared_not_independently_validated",
            "collision_status": "candidate_compiled",
            "candidate_may_self_authorize": False,
        },
        target_analysis=target["target_analysis"],
    )
    placement = placement_packet["placement"]
    placement_ready = bool(
        placement.get("status") == "runtime_visualization_candidate_only"
        and placement.get("mesh_triangle_aabb_overlap_probe_clear") is True
        and placement.get("analytic_reach_candidate") is True
    )
    controller_request = placement_packet["render_options"]["articulated_policy_trace_request"]
    candidates = controller_request.get("candidates")
    controller_ids = (
        [str(row.get("policy_id")) for row in candidates] if isinstance(candidates, list) else []
    )
    if controller_ids != list(EXPECTED_CONTROLLER_IDS):
        raise ArkitScenesInspectionRunError(["exactly_five_scripted_controllers_not_prepared"])

    package_path = destination / "scene" / "arkitscenes_40958756_isaac_candidate.usdz"
    package_result = compile_external_scene_isaac_package(
        analysis_splat_path=analysis_splat,
        collision_usd_path=collision_usd,
        output_path=package_path,
        request={
            "schema_version": "external_scene_isaac_package_request.v1",
            "appearance_scene_digest": geometry["source_surface_digest"],
            "analysis_splat_digest": geometry["analysis_splat_digest"],
            "collision_candidate_digest": collision_result["collision_candidate_digest"],
            "collision_asset_digest": _sha256(collision_usd),
            "scene_frame_binding_digest": frame_binding["scene_frame_binding_digest"],
            "source_to_collision_stage_matrix": frame_binding["source_to_collision_stage_matrix"],
            "metric_scale_status": "provider_declared_not_independently_validated",
            "collision_validated": False,
            "source_video_available": True,
            "generated_fill_allowed": False,
            "maximum_nonfinite_splat_fraction": 0.001,
        },
    )
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(package_path.resolve()))
        appearance_prim = (
            stage.GetPrimAtPath("/World/BlueprintReconstruction/Appearance/Gaussians")
            if stage
            else None
        )
        collision_prim = (
            stage.GetPrimAtPath("/World/BlueprintReconstruction/Collision/ExternalSceneMesh")
            if stage
            else None
        )
        package_local_inspection = {
            "stage_opened": bool(stage),
            "appearance_prim_present": bool(appearance_prim and appearance_prim.IsValid()),
            "collision_prim_present": bool(collision_prim and collision_prim.IsValid()),
            "collision_api_present": bool(
                collision_prim
                and collision_prim.IsValid()
                and collision_prim.HasAPI(UsdPhysics.CollisionAPI)
            ),
        }
    except ImportError as exc:
        raise ArkitScenesInspectionRunError(["openusd_local_inspection_unavailable"]) from exc
    if not all(package_local_inspection.values()):
        raise ArkitScenesInspectionRunError(["isaac_package_local_inspection_failed"])
    local_runtime = _runtime_available()
    paid_authorized = paid_runtime_authority is not None
    paid_request = {
        "schema_version": "paid_nvidia_isaac_runtime_authorization_request.v1",
        "status": (
            "authorized_not_launched_placement_gate"
            if paid_authorized and not placement_ready
            else "authorized_ready_for_canonical_allocator"
            if paid_authorized
            else "authorization_required_not_granted"
        ),
        "provider_selection": "canonical_allocator_best_available_qualified_nvidia_runtime",
        "allocator_entrypoint": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
        "maximum_spend_usd": round(float(maximum_spend_usd), 2),
        "hard_ttl_seconds": int(hard_ttl_seconds),
        "retry_cap": 0,
        "independent_watchdog_required": True,
        "provider_zero_required_before_and_after": True,
        "exact_package_digest": package_result["package_digest"],
        "exact_source_commit_sha": implementation_source_commit_sha,
        "input_compiler_source_commit_sha": compilation.get("source_commit_sha"),
        "paid_compute_authorized": paid_authorized,
        "provider_upload_authorized": paid_authorized,
        "authorization_source": paid_runtime_authority,
        "launch_performed": False,
    }
    paid_request["authorization_request_digest"] = canonical_digest(
        paid_request, digest_field="authorization_request_digest"
    )
    isaac_packet = {
        "schema_version": ISAAC_PACKET_SCHEMA,
        "status": "immutable_packet_ready_runtime_not_executed",
        "scene_id": compilation["source_capture_identity"],
        "implementation_source_commit_sha": implementation_source_commit_sha,
        "source_profile_digest": source_profile["source_profile_digest"],
        "package_digest": package_result["package_digest"],
        "package_result_digest": package_result["package_result_digest"],
        "target_orchestration_digest": target["orchestration_digest"],
        "target_binding_digest": binding["binding_evidence_digest"],
        "placement_proposal_digest": placement["placement_proposal_digest"],
        "official_isaac_franka_asset": placement["official_isaac_asset"],
        "robot_prim_path": placement["robot_prim_path"],
        "render_options": placement_packet["render_options"],
        "package_local_inspection": package_local_inspection,
        "controller_count": EXPECTED_CONTROLLER_COUNT,
        "controller_identities": [
            {"controller_id": row["policy_id"], "scripted_controller": True} for row in candidates
        ],
        "matched_reset_requirement": {
            "one_frozen_reset_digest_required": True,
            "one_frozen_observation_outcome_contract_required": True,
            "reset_stability_must_be_observed_in_isaac": True,
        },
        "required_retained_outputs": [
            "five_controller_traces",
            "matched_reset_evidence",
            "camera_specs_and_renders",
            "controller_identities",
            "ranking_inputs",
            "ranking_exclusions",
            "ranking_result",
        ],
        "local_runtime_probe": local_runtime,
        "paid_runtime_authorization_request_digest": paid_request["authorization_request_digest"],
        "execution_authorized": bool(paid_authorized and placement_ready),
        "execution_performed": False,
        "claim_boundary": {
            "public_dataset_proxy": True,
            "scripted_controller_evidence_not_learned_policy_evidence": True,
            "single_scenario": True,
            "simulation_only": True,
            "physical_success_proven": False,
            "deployment_proven": False,
            "safety_proven": False,
            "transfer_proven": False,
        },
    }
    isaac_packet["isaac_packet_digest"] = canonical_digest(
        isaac_packet, digest_field="isaac_packet_digest"
    )
    runtime_missing = not local_runtime["compatible_local_nvidia_isaac_runtime_available"]
    terminal_stage = "isaac_runtime_execution" if placement_ready else "robot_placement"
    smallest_code = (
        "collision_clear_supported_franka_placement_missing"
        if not placement_ready
        else "paid_nvidia_isaac_runtime_launch_result_missing"
        if runtime_missing and paid_authorized
        else "explicit_paid_nvidia_isaac_runtime_authorization_missing"
        if runtime_missing
        else "local_isaac_runtime_execution_result_missing"
    )
    smallest_instruction = (
        "Obtain a coordinate-validated floor/support and collision surface that yields all five "
        "Franka footprint support samples, zero overlapping obstacle triangles, and retained "
        "analytic reach; then replay this exact target placement."
        if not placement_ready
        else (
            (
                "Launch the exact authorized request once with the canonical allocator and "
                "retain the five matched-reset controller traces."
                if paid_authorized
                else "Authorize the exact digest-bound paid runtime request, then execute once "
                "with the canonical allocator and retain the five matched-reset controller "
                "traces."
            )
            if runtime_missing
            else "Execute the prepared packet once in the available local Isaac runtime."
        )
    )
    terminal = {
        "schema_version": TERMINAL_REPORT_SCHEMA,
        "run_id": "arkitscenes-40958756-visible-sink-inspection-scripted-v1",
        "implementation_source_commit_sha": implementation_source_commit_sha,
        "status": (
            "abstained"
            if (not placement_ready or runtime_missing)
            else "awaiting_local_isaac_execution"
        ),
        "terminal_stage": terminal_stage,
        "smallest_missing_measurement": {
            "code": smallest_code,
            "instruction": smallest_instruction,
            "authorization_request_digest": paid_request["authorization_request_digest"],
        },
        "subsequent_runtime_requirement": {
            "compatible_local_runtime_available": not runtime_missing,
            "paid_runtime_authorization_granted": paid_authorized,
            "authorization_request_digest": paid_request["authorization_request_digest"],
        },
        "source_class": "public-dataset proxy",
        "evidence_class": "scripted-controller evidence, not learned-policy evidence",
        "scenario_scope": "single scenario",
        "execution_scope": "simulation-only",
        "no_claims": [
            "physical success",
            "deployment",
            "safety",
            "transfer",
        ],
        "exact_status": {
            "source_digests": "verified_against_all_retained_recorded_receipts",
            "metric_scale": "dataset_declared_sensor_meters_not_independently_validated",
            "coordinate_conventions": (
                "camera_and_depth_conventions_preserved; handedness up-axis and gravity not "
                "independently validated"
            ),
            "collision": "depth_derived_static_triangle_candidate_not_contact_qualified",
            "placement": placement.get("status"),
            "footprint_clearance": bool(placement.get("mesh_triangle_aabb_overlap_probe_clear")),
            "triangle_overlap_probe_hits": placement.get("mesh_triangle_aabb_overlap_probe_hits"),
            "analytic_target_reach": bool(placement.get("analytic_reach_candidate")),
            "metric_target_reach_qualified": bool(placement.get("metric_reach_qualified")),
            "reset_stability": "not_measured_requires_isaac_execution",
            "isaac_scene_package": "candidate_packaged_and_locally_opened",
            "official_isaac_franka": "requested_not_runtime_observed",
            "isaac_runtime": (
                "not_reached_due_to_unqualified_placement_and_unavailable_locally"
                if not placement_ready and runtime_missing
                else "unavailable_locally_not_executed"
                if runtime_missing
                else "available_not_executed"
            ),
            "controller_count_prepared": EXPECTED_CONTROLLER_COUNT,
            "controller_trace_count": 0,
            "controller_ranking": "not_executed_no_result",
        },
        "artifact_digests": {
            "source_profile": source_profile["source_profile_digest"],
            "target_orchestration": target["orchestration_digest"],
            "placement": placement["placement_proposal_digest"],
            "isaac_package": package_result["package_digest"],
            "isaac_packet": isaac_packet["isaac_packet_digest"],
        },
        "task_evaluation_run_completed": False,
        "decision_envelope_emitted": False,
        "explicit_terminal_abstention": True,
    }
    terminal["terminal_report_digest"] = canonical_digest(
        terminal, digest_field="terminal_report_digest"
    )

    artifacts: dict[str, Mapping[str, Any]] = {
        "source_digest_verification.json": verification,
        "public_dataset_source_profile.json": source_profile,
        "derived_geometry_adapter.json": geometry,
        "same_source_frame_binding.json": frame_binding,
        "target_orchestration.json": target,
        "collision_candidate.json": collision_result,
        "robot_placement.json": placement,
        "render_options.json": placement_packet["render_options"],
        "isaac_package_result.json": package_result,
        "paid_runtime_authorization_request.json": paid_request,
        "isaac_run_packet.json": isaac_packet,
        "terminal_report.json": terminal,
    }
    for name, value in artifacts.items():
        _write_immutable(destination / name, value)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": terminal["status"],
        "terminal_report_digest": terminal["terminal_report_digest"],
        "artifacts": [
            {"relative_path": name, "digest": _sha256(destination / name)}
            for name in sorted(artifacts)
        ]
        + [
            {
                "relative_path": path.resolve().relative_to(destination).as_posix(),
                "digest": _sha256(path),
            }
            for path in (analysis_splat, collision_glb, collision_usd, package_path)
        ],
    }
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    _write_immutable(destination / "run_manifest.json", manifest)
    return terminal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--selected-compilation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--implementation-source-commit-sha", required=True)
    parser.add_argument("--view-id", action="append", required=True)
    parser.add_argument("--allowed-object-label", action="append", required=True)
    parser.add_argument("--torchvision-weights", required=True, type=Path)
    parser.add_argument("--minimum-visual-confidence", type=float, default=0.5)
    parser.add_argument("--maximum-spend-usd", type=float, default=DEFAULT_MAX_SPEND_USD)
    parser.add_argument("--hard-ttl-seconds", type=int, default=DEFAULT_TTL_SECONDS)
    parser.add_argument("--paid-runtime-authority")
    args = parser.parse_args(argv)
    report = compile_arkitscenes_inspection_run(
        scene_root=args.scene_root,
        selected_compilation_path=args.selected_compilation,
        output_root=args.output_root,
        implementation_source_commit_sha=args.implementation_source_commit_sha,
        view_ids=args.view_id,
        allowed_object_labels=args.allowed_object_label,
        weights_path=args.torchvision_weights,
        minimum_visual_confidence=args.minimum_visual_confidence,
        maximum_spend_usd=args.maximum_spend_usd,
        hard_ttl_seconds=args.hard_ttl_seconds,
        paid_runtime_authority=args.paid_runtime_authority,
    )
    print(canonical_json(report))
    return 0 if report["status"] != "abstained" else 2


__all__ = [
    "ArkitScenesInspectionRunError",
    "build_public_dataset_source_profile",
    "compile_arkitscenes_inspection_run",
    "verify_retained_sources",
]


if __name__ == "__main__":
    raise SystemExit(main())
