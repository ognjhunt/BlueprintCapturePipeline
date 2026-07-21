"""Build model-derived visual augmentation packets for post-training support.

The packet reuses simulator-valid robot motion and camera/skeleton provenance to
prepare visual variation requests for OSCAR/Cosmos/future video backends. It is
support data for Post-Training Data Packages and visual distribution-shift
evaluation; it is not contact-physics, real-robot, or deployment proof.
"""

from __future__ import annotations

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


PACKET_SCHEMA_VERSION = "oscar_visual_augmentation_packet.v1"
VARIANT_REQUESTS_SCHEMA_VERSION = "oscar_visual_augmentation_variant_requests.v1"
BACKEND_REGISTRY_SCHEMA_VERSION = "visual_augmentation_backend_registry.v1"
DISTRIBUTION_SHIFT_PROTOCOL_SCHEMA_VERSION = "visual_distribution_shift_eval_protocol.v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "visual_augmentation_claim_boundary.v1"

PACKET_DIR_NAME = "oscar_visual_augmentation_packet"
PACKET_MANIFEST_NAME = "oscar_visual_augmentation_packet_manifest.json"

DEFAULT_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant_id": "same_geometry_real_site_context",
        "environment_tags": ["real_site", "same_camera_geometry"],
        "lighting_profile": "source_or_neutral",
        "object_appearance_profile": "source_aligned",
        "background_profile": "preserve_source_layout",
        "prompt": (
            "Preserve the source camera geometry and robot pose while rendering a realistic "
            "real-site version of the same task context."
        ),
    },
    {
        "variant_id": "cluttered_kitchen_counter",
        "environment_tags": ["kitchen", "countertop", "clutter"],
        "lighting_profile": "warm_indoor_mixed_light",
        "object_appearance_profile": "household_packaging_variation",
        "background_profile": "visible_appliances_and_counter_texture",
        "prompt": (
            "Render the same robot motion in a realistic kitchen counter scene with natural "
            "surface texture, visible background objects, and moderate clutter."
        ),
    },
    {
        "variant_id": "warehouse_conveyor",
        "environment_tags": ["warehouse", "conveyor", "industrial"],
        "lighting_profile": "cool_overhead_lighting",
        "object_appearance_profile": "warehouse_package_variation",
        "background_profile": "industrial_floor_and_shelving",
        "prompt": (
            "Render the same robot motion near a warehouse conveyor with realistic floor "
            "texture, shelves, and varied package appearance."
        ),
    },
    {
        "variant_id": "low_light_backroom",
        "environment_tags": ["backroom", "low_light", "occlusion_risk"],
        "lighting_profile": "dim_indoor_light",
        "object_appearance_profile": "partially_shadowed_target",
        "background_profile": "storage_room_with_occluding_objects",
        "prompt": (
            "Render the same robot motion in a dim backroom with partial object occlusion "
            "and realistic shadowing."
        ),
    },
    {
        "variant_id": "reflective_floor_high_glare",
        "environment_tags": ["facility", "reflective_surface", "glare"],
        "lighting_profile": "bright_overhead_glare",
        "object_appearance_profile": "high_contrast_target",
        "background_profile": "polished_floor_with_reflections",
        "prompt": (
            "Render the same robot motion in a facility scene with reflective floor glare, "
            "bright overhead lighting, and realistic texture variation."
        ),
    },
)

MODEL_BACKEND_CONTRACTS: tuple[dict[str, Any], ...] = (
    {
        "backend_id": "oscar_wam",
        "backend_family": "skeleton_conditioned_robot_video_generator",
        "role": "first_frame_plus_skeleton_video_visual_realism_backend",
        "command_envs": [
            "BLUEPRINT_OSCAR_VISUAL_AUGMENTATION_COMMAND",
            "BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND",
        ],
        "related_wam_rollout_command_envs_require_visual_augmentation_wrapper": [
            "BLUEPRINT_OSCAR_WAM_COMMAND",
            "BLUEPRINT_OSCAR_WAM_PROVIDER_COMMAND",
        ],
        "checkpoint_envs": ["BLUEPRINT_OSCAR_WAM_CHECKPOINT"],
        "api_key_envs": ["BLUEPRINT_OSCAR_WAM_API_KEY", "OSCAR_WAM_API_KEY"],
        "input_contract": ["first_frame", "skeleton_conditioning_video", "variant_prompt"],
        "output_contract": ["model_derived_generated_video", "backend_execution_manifest"],
    },
    {
        "backend_id": "cosmos_wam",
        "backend_family": "world_video_rollout_or_refinement_backend",
        "role": "replaceable_video_world_model_backend",
        "command_envs": [
            "BLUEPRINT_COSMOS_VISUAL_AUGMENTATION_COMMAND",
            "BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND",
        ],
        "related_wam_rollout_command_envs_require_visual_augmentation_wrapper": [
            "BLUEPRINT_COSMOS_WAM_COMMAND",
            "BLUEPRINT_COSMOS_WAM_PROVIDER_COMMAND",
        ],
        "checkpoint_envs": ["BLUEPRINT_COSMOS_WAM_CHECKPOINT"],
        "api_key_envs": ["BLUEPRINT_COSMOS_WAM_API_KEY", "COSMOS_WAM_API_KEY"],
        "input_contract": ["first_frame", "motion_conditioning", "variant_prompt"],
        "output_contract": ["model_derived_generated_video", "backend_execution_manifest"],
    },
    {
        "backend_id": "future_video_wam",
        "backend_family": "future_replaceable_video_or_world_model_backend",
        "role": "placeholder_contract_for_future_backend_swap",
        "command_envs": ["BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND"],
        "checkpoint_envs": ["BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_CHECKPOINT"],
        "api_key_envs": ["BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_API_KEY"],
        "input_contract": ["first_frame", "motion_or_skeleton_conditioning", "variant_prompt"],
        "output_contract": ["model_derived_generated_video", "backend_execution_manifest"],
    },
)

CLAIM_BOUNDARY: Dict[str, Any] = {
    "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
    "artifact_purpose": "visual_augmentation_support_for_post_training_and_distribution_shift_eval",
    "model_derived_visual_augmentation": True,
    "generated_videos_are_model_derived_support_assets": True,
    "generated_videos_are_raw_capture_evidence": False,
    "generated_videos_are_physical_robot_episode_evidence": False,
    "camera_provenance_required": True,
    "skeleton_provenance_required": True,
    "simulator_or_owner_motion_source_required": True,
    "real_robot_episodes_required_for_packet": False,
    "real_site_capture_required_for_generic_packet": False,
    "real_site_capture_preferred_for_blueprint_post_training_package": True,
    "resimulation_required_for_visual_variants": False,
    "backend_swappable": True,
    "contact_physics_proven": False,
    "object_drop_physics_proven": False,
    "robot_policy_execution_proven": False,
    "real_robot_readiness_proven": False,
    "deployment_safety_proven": False,
    "public_claim_upgrade_allowed": False,
    "rank_fidelity_result_proven": False,
    "sim_to_real_calibration_proven": False,
    "external_accuracy_claim_requires_real_world_anchors": True,
    "disallowed_claims": [
        "contact_physics_validated",
        "object_drop_physics_validated",
        "physical_robot_ready",
        "deployment_approved",
        "safety_validated",
        "real_world_policy_success",
        "sim_to_real_calibration_measured",
    ],
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_optional_mapping(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _path_candidate(value: Any, *, base_dir: Path | None = None) -> Path | None:
    text = _string(value)
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _artifact(path: Path | None, *, output_dir: Path) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": 0, "sha256": None}
    return {
        "path": _relative_to(output_dir, path),
        "absolute_path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha_file(path),
    }


def _manifest_path_value(
    manifest: Mapping[str, Any],
    section: str,
    *,
    key: str = "path",
    base_dir: Path | None = None,
) -> Path | None:
    value = _mapping(manifest.get(section)).get(key)
    return _path_candidate(value, base_dir=base_dir)


def _discover_first_existing(job_dir: Path | None, names: Sequence[str]) -> Path | None:
    if job_dir is None:
        return None
    for name in names:
        path = job_dir / name
        if path.is_file():
            return path.resolve()
    return None


def _load_variant_specs(path: str | Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    payload = read_json_any(Path(path).expanduser())
    raw: Any
    if isinstance(payload, Mapping):
        raw = payload.get("variants") or payload.get("variant_requests")
    else:
        raw = payload
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _normalize_variants(variants: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    source = list(variants) if variants else list(DEFAULT_VARIANTS)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, variant in enumerate(source, start=1):
        variant_id = _string(variant.get("variant_id") or variant.get("id"))
        if not variant_id:
            variant_id = f"visual_variant_{index:03d}"
        if variant_id in seen:
            variant_id = f"{variant_id}_{index:03d}"
        seen.add(variant_id)
        rows.append(
            {
                "schema_version": VARIANT_REQUESTS_SCHEMA_VERSION,
                "variant_id": variant_id,
                "prompt": _string(variant.get("prompt"))
                or "Render a realistic visual variant while preserving camera geometry.",
                "environment_tags": list(variant.get("environment_tags") or []),
                "lighting_profile": variant.get("lighting_profile") or "unspecified",
                "object_appearance_profile": (
                    variant.get("object_appearance_profile") or "unspecified"
                ),
                "background_profile": variant.get("background_profile") or "unspecified",
                "camera_geometry_policy": "preserve_source_camera_geometry",
                "motion_conditioning_policy": "reuse_source_skeleton_conditioning",
                "resimulation_required": False,
                "model_backend_output_required_for_completion": False,
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        )
    return rows


def _parse_generated_video_specs(specs: Sequence[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs or [], start=1):
        text = _string(spec)
        if not text:
            continue
        backend_id = "unknown_backend"
        variant_and_path = text
        if "::" in text:
            backend_id, variant_and_path = text.split("::", 1)
        if "=" in variant_and_path:
            variant_id, path_text = variant_and_path.split("=", 1)
        else:
            path_text = variant_and_path
            variant_id = f"generated_variant_{index:03d}"
        rows.append(
            {
                "variant_id": _string(variant_id) or f"generated_variant_{index:03d}",
                "path": _string(path_text),
                "model_backend_id": _string(backend_id) or "unknown_backend",
            }
        )
    return rows


def _backend_registry(*, selected_backend_id: str, generated_at: str) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for contract in MODEL_BACKEND_CONTRACTS:
        command_envs = list(contract.get("command_envs") or [])
        checkpoint_envs = list(contract.get("checkpoint_envs") or [])
        api_key_envs = list(contract.get("api_key_envs") or [])
        rows.append(
            {
                **dict(contract),
                "selected_for_packet": contract.get("backend_id") == selected_backend_id,
                "command_configured": any(bool(os.getenv(name)) for name in command_envs),
                "checkpoint_configured": any(bool(os.getenv(name)) for name in checkpoint_envs),
                "api_key_present": any(bool(os.getenv(name)) for name in api_key_envs),
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            }
        )
    return {
        "schema_version": BACKEND_REGISTRY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "selected_backend_id": selected_backend_id,
        "backend_count": len(rows),
        "backends": rows,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "one_backend_is_not_permanent_platform_dependency": True,
        },
    }


def _distribution_shift_protocol(
    *,
    generated_at: str,
    variant_count: int,
) -> Dict[str, Any]:
    return {
        "schema_version": DISTRIBUTION_SHIFT_PROTOCOL_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "ready_for_visual_distribution_shift_review"
        if variant_count
        else "blocked_no_variants",
        "evaluation_purpose": "visual_robustness_stress_test_and_training_support",
        "held_constant": [
            "source_robot_motion_or_skeleton_conditioning",
            "source_camera_geometry",
            "task_id_and_scenario_when_available",
        ],
        "varied_axes": [
            "environment_appearance",
            "lighting",
            "background_clutter",
            "surface_texture",
            "target_object_appearance",
            "occlusion_risk",
        ],
        "recommended_review_checks": [
            "target_visible",
            "robot_or_end_effector_not_visually_corrupted",
            "camera_geometry_consistent_with_source",
            "motion_follows_source_skeleton",
            "policy_input_frame_decodable",
            "generated_video_marked_model_derived",
        ],
        "blocked_claims": list(CLAIM_BOUNDARY["disallowed_claims"]),
        "real_world_anchor_requirements_for_calibration": [
            "accepted_real_world_anchor.v1 rows",
            "exact scenario_eval_run_id/policy_id/task_id/variation joins",
            "owner evidence or operator attestation",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def build_oscar_visual_augmentation_packet(
    *,
    capture_root: str | Path,
    job_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    source_input_package: str | Path | None = None,
    first_frame: str | Path | None = None,
    skeleton_video: str | Path | None = None,
    camera_provenance: str | Path | None = None,
    skeleton_provenance: str | Path | None = None,
    variant_specs: str | Path | None = None,
    generated_videos: Sequence[Mapping[str, Any]] | None = None,
    selected_backend_id: str = "oscar_wam",
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    resolved_job_dir = Path(job_dir).resolve() if job_dir else None
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else (resolved_job_dir / PACKET_DIR_NAME if resolved_job_dir else pipeline_dir / PACKET_DIR_NAME)
    )
    ensure_dir(resolved_output_dir)
    generated_at = utc_now_iso()

    source_package_path = Path(source_input_package).expanduser().resolve() if source_input_package else None
    source_package = _read_optional_mapping(source_package_path)
    source_package_base = source_package_path.parent if source_package_path else None

    first_frame_path = _path_candidate(first_frame) or _manifest_path_value(
        source_package,
        "first_frame",
        base_dir=source_package_base,
    )
    skeleton_video_path = _path_candidate(skeleton_video) or _manifest_path_value(
        source_package,
        "skeleton_video",
        base_dir=source_package_base,
    )
    camera_provenance_path = _path_candidate(camera_provenance) or _discover_first_existing(
        resolved_job_dir,
        (
            "camera_calibration_quality_gate.json",
            "policy_visual_observation_manifest.json",
            "robot_camera_profile_launch_readiness.json",
            "selected_initial_policy_observation.json",
        ),
    )
    skeleton_provenance_path = (
        _path_candidate(skeleton_provenance)
        or _manifest_path_value(source_package, "projected_skeleton_trace", base_dir=source_package_base)
        or _discover_first_existing(
            resolved_job_dir,
            (
                "g1_projected_skeleton_trace.jsonl",
                "g1_projected_skeleton_manifest.json",
                "robot_fk_projected_skeleton_trace.jsonl",
            ),
        )
    )

    variants = _normalize_variants(_load_variant_specs(variant_specs))
    backend_registry = _backend_registry(
        selected_backend_id=selected_backend_id,
        generated_at=generated_at,
    )
    distribution_shift_protocol = _distribution_shift_protocol(
        generated_at=generated_at,
        variant_count=len(variants),
    )

    blockers: list[str] = []
    if first_frame_path is None or not first_frame_path.is_file():
        blockers.append("missing_first_frame_visual_context")
    if skeleton_video_path is None or not skeleton_video_path.is_file():
        blockers.append("missing_skeleton_conditioning_video")
    if camera_provenance_path is None or not camera_provenance_path.is_file():
        blockers.append("missing_camera_provenance")
    if skeleton_provenance_path is None or not skeleton_provenance_path.is_file():
        blockers.append("missing_skeleton_provenance")
    if not variants:
        blockers.append("missing_visual_variant_requests")

    generated_rows: list[dict[str, Any]] = []
    for row in generated_videos or []:
        variant_id = _string(row.get("variant_id") or row.get("id"))
        video_path = _path_candidate(row.get("path") or row.get("generated_video_path"))
        backend_id = _string(row.get("model_backend_id") or selected_backend_id)
        model_derived = bool(row.get("model_derived", True))
        generated_artifact_kind = _string(row.get("generated_artifact_kind")) or (
            "model_derived_visual_augmentation"
            if model_derived
            else "fixture_or_backend_generated_support_video"
        )
        if video_path is None or not video_path.is_file():
            blockers.append(f"missing_generated_video:{variant_id or 'unknown_variant'}")
        generated_rows.append(
            {
                "schema_version": (
                    "model_derived_visual_augmentation_generated_video.v1"
                    if model_derived
                    else "visual_augmentation_generated_support_video.v1"
                ),
                "variant_id": variant_id or "unknown_variant",
                "model_backend_id": backend_id,
                "generated_artifact_kind": generated_artifact_kind,
                "generated_video": _artifact(video_path, output_dir=resolved_output_dir),
                "model_derived": model_derived,
                "raw_capture_evidence": False,
                "physical_robot_episode_evidence": False,
                "contact_physics_proven": False,
                "deployment_safety_proven": False,
                "review_status": "pending_visual_review"
                if video_path is not None and video_path.is_file()
                else "missing",
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        )

    input_assets = {
        "first_frame": _artifact(first_frame_path, output_dir=resolved_output_dir),
        "skeleton_conditioning_video": _artifact(
            skeleton_video_path,
            output_dir=resolved_output_dir,
        ),
        "camera_provenance": _artifact(camera_provenance_path, output_dir=resolved_output_dir),
        "skeleton_provenance": _artifact(
            skeleton_provenance_path,
            output_dir=resolved_output_dir,
        ),
        "source_input_package": _artifact(source_package_path, output_dir=resolved_output_dir),
    }

    for variant in variants:
        variant["source_assets"] = {
            "first_frame": input_assets["first_frame"]["path"],
            "skeleton_conditioning_video": input_assets["skeleton_conditioning_video"]["path"],
            "camera_provenance": input_assets["camera_provenance"]["path"],
            "skeleton_provenance": input_assets["skeleton_provenance"]["path"],
        }
        variant["candidate_model_backends"] = [
            row["backend_id"] for row in backend_registry["backends"]
        ]

    if blockers:
        status = "blocked_missing_provenance_or_generation_inputs"
    elif generated_rows and all(row.get("model_derived") is True for row in generated_rows):
        status = "completed_with_model_derived_generated_videos"
    elif generated_rows:
        status = "completed_with_generated_support_videos_pending_model_truth"
    else:
        status = "ready_for_model_backend_generation"

    variant_requests_path = resolved_output_dir / "visual_augmentation_variant_requests.jsonl"
    _write_jsonl(variant_requests_path, variants)
    backend_registry_path = (
        resolved_output_dir / "visual_augmentation_backend_registry.json"
    )
    write_json(backend_registry_path, backend_registry)
    legacy_backend_registry_path = resolved_output_dir / "model_backend_registry.json"
    write_json(legacy_backend_registry_path, backend_registry)
    protocol_path = resolved_output_dir / "visual_distribution_shift_eval_protocol.json"
    write_json(protocol_path, distribution_shift_protocol)
    claim_boundary_path = resolved_output_dir / "claim_boundary.json"
    write_json(claim_boundary_path, CLAIM_BOUNDARY)

    manifest = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "packet_type": "oscar_visual_augmentation_packet",
        "packet_purpose": "post_training_data_package_visual_augmentation_and_distribution_shift_eval",
        "status": status,
        "blockers": sorted(set(blockers)),
        "selected_backend_id": selected_backend_id,
        "backend_registry_path": _relative_to(resolved_output_dir, backend_registry_path),
        "legacy_backend_registry_path": _relative_to(
            resolved_output_dir,
            legacy_backend_registry_path,
        ),
        "variant_requests_path": _relative_to(resolved_output_dir, variant_requests_path),
        "visual_distribution_shift_eval_protocol_path": _relative_to(
            resolved_output_dir,
            protocol_path,
        ),
        "claim_boundary_path": _relative_to(resolved_output_dir, claim_boundary_path),
        "input_assets": input_assets,
        "source_context": {
            "capture_root": str(Path(capture_root).expanduser().resolve()),
            "job_dir": str(resolved_job_dir) if resolved_job_dir else None,
            "source_input_package": str(source_package_path) if source_package_path else None,
            "source_review_video_path": source_package.get("source_review_video_path"),
            "scenario_eval_run_id": source_package.get("scenario_eval_run_id"),
            "task_id": source_package.get("task_id"),
            "spawn_id": source_package.get("spawn_id"),
        },
        "variant_count": len(variants),
        "generated_video_count": len(generated_rows),
        "variant_requests": variants,
        "generated_videos": generated_rows,
        "distribution_shift_axes": distribution_shift_protocol["varied_axes"],
        "post_training_data_package_contract": {
            "include_as_model_derived_support_asset": True,
            "eligible_for_visual_distribution_shift_evaluation": status != (
                "blocked_missing_provenance_or_generation_inputs"
            ),
            "requires_human_or_vlm_review_before_training_use": True,
            "generated_videos_must_remain_model_derived": bool(
                not generated_rows or all(row.get("model_derived") is True for row in generated_rows)
            ),
            "non_model_fixture_outputs_are_plumbing_tests_only": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_output_dir / PACKET_MANIFEST_NAME, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an OSCAR/Cosmos-swappable visual augmentation packet"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--source-input-package")
    parser.add_argument("--first-frame")
    parser.add_argument("--skeleton-video")
    parser.add_argument("--camera-provenance")
    parser.add_argument("--skeleton-provenance")
    parser.add_argument("--variant-specs")
    parser.add_argument("--selected-backend-id", default="oscar_wam")
    parser.add_argument(
        "--generated-video",
        action="append",
        default=[],
        help="Optional backend::variant_id=/path/to/video.mp4 generated output reference",
    )
    args = parser.parse_args(argv)
    result = build_oscar_visual_augmentation_packet(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
        source_input_package=args.source_input_package,
        first_frame=args.first_frame,
        skeleton_video=args.skeleton_video,
        camera_provenance=args.camera_provenance,
        skeleton_provenance=args.skeleton_provenance,
        variant_specs=args.variant_specs,
        generated_videos=_parse_generated_video_specs(args.generated_video),
        selected_backend_id=args.selected_backend_id,
    )
    default_output_dir = Path(args.capture_root) / "pipeline" / PACKET_DIR_NAME
    manifest_dir = Path(args.output_dir or (Path(args.job_dir) / PACKET_DIR_NAME if args.job_dir else default_output_dir))
    print(f"[oscar-visual-augmentation-packet] manifest={manifest_dir / PACKET_MANIFEST_NAME}")
    print(f"[oscar-visual-augmentation-packet] status={result['status']}")
    return 0 if not str(result["status"]).startswith("blocked") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
