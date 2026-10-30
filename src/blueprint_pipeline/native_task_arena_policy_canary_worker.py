"""Provider worker for one warm, paired internal-policy canary session."""

from __future__ import annotations

from blueprint_pipeline.policy_canary_media_integrity import (
    bound_media_artifact as _bound_media_artifact,
    require_completed_episode_media as _require_completed_episode_media,
)

from copy import deepcopy
from dataclasses import dataclass
from blueprint_pipeline.native_policy_canary_control_gate import controls_required, execute_native_controls, validate_controls_receipt
from blueprint_pipeline.native_policy_canary_matrix_gate import (
    execute_strict_matrix, _episode_embodiment_parity_diagnostic,
    ISOLATED_CELL_PROCESS_TIMEOUT_SECONDS as ISOLATED_CELL_PROCESS_TIMEOUT_SECONDS,
    _HARD_DEADLINE_ENV, _isolated_cell_timeout_seconds,
)
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_execution_evidence import (
    candidate_execution_kinds, evidence_kind, retained_rollout_counts, SCRIPTED, scripted_fields,
)
from blueprint_pipeline.team_evaluation_worker_retention import (
    _compact_team_child_result, _prefix_episode_evidence_paths, _validate_isolated_cell_result,
    _validate_provider_manifest, seal_scripted_interruption,
    _team_checkpoint as _retention_checkpoint,
)
from blueprint_pipeline.team_embodiment import apply_team_embodiment, is_team_plan
from blueprint_pipeline.policy_canary_scene_resolution import resolved_scene_plan as _resolved_scene_plan
from blueprint_pipeline.task_family_policy_canary_execution import (
    _run_family_canary_episode, _construction_lineage_mode, family_render_backend, qualify_family_policy_scene, beta_result_binding,
)
from blueprint_pipeline.task_family_policy_worker_execution import (
    is_task_family_scene_plan, verify_family_operator_inputs,
)
from blueprint_pipeline.policy_canary_worker_evidence import (
    _prepolicy_refusal_result,
    _write_episode_failure_gap, _write_episode_json_artifact,
    _write_indexed_telemetry as _write_policy_canary_telemetry,
)
from blueprint_pipeline.appearance_render_backend import (
    BACKEND_ISAAC_NATIVE_NUREC, BACKEND_ISAAC_USD_GEOMETRY,
    BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE,
    BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE,
    AppearanceRenderBackendError,
    build_appearance_render_backend,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    BETA_RUN_KIND, PolicyCanaryEpisodeFailure, PROVIDER_RESULT_FILENAME, TEAM_RUN_KIND,
    PROVIDER_STATUS_FILENAME, PROVIDER_STATUS_SCHEMA,
    execute_paired_session,
    validate_runtime_input_manifest,
    validate_session_authority,
)


DROID_PARITY_MINIMUM_APPROACH_M = 0.05
_CELL_PROGRESS_FD_ENV = "BLUEPRINT_POLICY_CANARY_PROGRESS_FD"
_CELL_PROGRESS_STAGES = frozenset({
    "static_preflight_passed", "isaac_launch_started", "isaac_launch_completed",
    "observation_gate_passed", "policy_loaded", "episode_completed",
})
# Preserve the existing helper import surface used by retained tooling.
_write_indexed_telemetry = _write_policy_canary_telemetry


def _policy_camera_visibility_contract(
    snapshot: Mapping[str, Any],
    *,
    preserve_official_droid_calibration: bool,
    require_task_support: bool = False,
    wrist_robot_occlusion_qualified: bool = False,
) -> dict[str, Any]:
    """Qualify task visibility according to each camera's policy role.

    A fixed external camera must frame the task at reset. The official DROID
    wrist camera is attached to the gripper, so its reset observation may put
    the task near an image edge before the policy approaches it. For that
    camera, visible native-semantic task pixels are the reset requirement;
    centering remains recorded as a typed notice and is measured during the
    rollout.
    """

    rows = {
        str(row.get("role")): row
        for row in snapshot.get("cameras") or []
        if isinstance(row, Mapping) and str(row.get("role") or "")
    }
    expected_roles = {"external", "wrist", "overview"}
    raw_visibility = {
        role: bool((row.get("observability") or {}).get("passed"))
        for role, row in rows.items()
    }
    qualifications: dict[str, Any] = {}
    blockers: list[str] = []
    notices: list[str] = []
    if set(rows) != expected_roles:
        blockers.append("policy_canary_camera_role_inventory_invalid")
    for role in sorted(expected_roles):
        row = rows.get(role)
        observability = (
            row.get("observability")
            if isinstance(row, Mapping)
            and isinstance(row.get("observability"), Mapping)
            else {}
        )
        raw_passed = bool(observability.get("passed"))
        pixel_count = int(observability.get("pixel_count") or 0)
        thresholds = observability.get("thresholds")
        minimum_pixels = (
            int(thresholds.get("effective_minimum_pixels") or 0)
            if isinstance(thresholds, Mapping)
            else 0
        )
        raw_render_passed = observability.get("render_passed") is True
        render_passed = raw_render_passed or (
            role == "wrist"
            and preserve_official_droid_calibration
            and wrist_robot_occlusion_qualified
            and (observability.get("render_evidence") or {}).get("target_rendered") is True
        )
        centroid_within_margin = (
            observability.get("centroid_within_margin") is True
        )
        if preserve_official_droid_calibration and role == "wrist":
            passed = (
                render_passed
                and minimum_pixels > 0
                and pixel_count >= minimum_pixels
                and bool(observability.get("target_semantic_ids"))
            )
            status = (
                "centered"
                if passed and centroid_within_margin
                else "initial_edge_visible"
                if passed
                else "insufficient_task_pixels"
            )
            if passed and not centroid_within_margin:
                notices.append("droid_wrist_task_initially_near_frame_edge")
            if not passed:
                blockers.append("droid_wrist_task_pixels_below_threshold")
        else:
            passed = raw_passed
            status = "centered" if passed else "not_qualified"
            if not passed:
                blockers.append(f"policy_canary_{role}_task_visibility_failed")
        qualifications[role] = {
            "status": status,
            "passed": passed,
            "raw_observability_passed": raw_passed,
            "pixel_count": pixel_count,
            "minimum_pixels": minimum_pixels,
            "render_passed": render_passed,
            "raw_render_passed": raw_render_passed,
            "wrist_robot_occlusion_qualified": (
                role == "wrist" and wrist_robot_occlusion_qualified
            ),
            "centroid_within_margin": centroid_within_margin,
        }
        if require_task_support and role in {"external", "overview"}:
            labels = row.get("semantic_label_pixels") if isinstance(row, Mapping) else None
            support_measurement = labels.get("task_support") if isinstance(labels, Mapping) else None
            support_pixels = (
                support_measurement.get("pixel_count")
                if isinstance(support_measurement, Mapping) else None
            )
            support_visible = (
                type(support_pixels) is int
                and support_measurement.get("target_label") == "task_support"
                and support_measurement.get("measurement_authority") == "native_semantic_segmentation_aov"
                and bool(support_measurement.get("target_semantic_ids"))
                and minimum_pixels > 0
                and support_pixels >= minimum_pixels
                and render_passed
            )
            qualifications[role].update(
                destination_pixel_count=support_pixels,
                minimum_destination_pixels=minimum_pixels,
                destination_visible=support_visible,
                passed=bool(passed and support_visible),
            )
            if not support_visible:
                qualifications[role]["status"] = "destination_not_visible"
                blockers.append(f"policy_canary_{role}_destination_visibility_failed")
    return {
        "passed": not blockers and set(rows) == expected_roles,
        "camera_visibility": {
            role: bool(qualifications.get(role, {}).get("passed"))
            for role in sorted(expected_roles)
        },
        "raw_camera_visibility": raw_visibility,
        "role_qualifications": qualifications,
        "blockers": sorted(set(blockers)),
        "notices": sorted(set(notices)),
    }


def _qualify_official_droid_wrist_body_occlusion(
    *, snapshot: Mapping[str, Any], visual: Mapping[str, Any], output_root: Path,
) -> tuple[dict[str, Any], bool]:
    """Keep a visible target when exact black pixels are the robot's own body."""
    import numpy as np
    from PIL import Image
    from blueprint_pipeline.native_task_camera_observability import (
        BLOCKER_FRAME_STRUCTURE_FAILED, BLOCKER_SITE_DOMINANT_COLOR,
        BLOCKER_SITE_VOID, NEAR_BLACK_LUMINANCE_MAX,
        REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK,
    )

    unchanged = deepcopy(dict(visual))
    rows = [row for row in snapshot.get("cameras") or []
            if isinstance(row, Mapping) and row.get("role") == "wrist"]
    if len(rows) != 1:
        return unchanged, False
    wrist = rows[0]
    observation = wrist.get("observability") or {}
    render = observation.get("render_evidence") or {}
    target = (wrist.get("semantic_label_pixels") or {}).get("task_object") or {}
    expected_blocker = f"{REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK}:wrist"
    wrist_view = (unchanged.get("views") or {}).get("wrist") or {}
    if (
        wrist_view.get("blockers") != [REFUSAL_PREPOLICY_VISUAL_FRAME_NEAR_BLACK]
        or expected_blocker not in unchanged.get("blockers", [])
        or wrist_view.get("render_presence", {}).get("passed") is not True
        or wrist_view.get("saturation", {}).get("passed") is not True
        or observation.get("semantic_passed") is not True
        or render.get("target_rendered") is not True
        or not set(render.get("blockers") or []).issubset(
            {BLOCKER_SITE_VOID, BLOCKER_SITE_DOMINANT_COLOR}
        )
        or float(target.get("pixel_fraction") or 0.0) < 0.15
    ):
        return unchanged, False
    root = output_root.resolve()
    paths = []
    for record in (wrist.get("rgb_png"), wrist.get("robot_semantic_mask")):
        if not isinstance(record, Mapping):
            return unchanged, False
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            return unchanged, False
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            return unchanged, False
        if _sha256(path) != record.get("sha256"):
            return unchanged, False
        paths.append(path)
    with Image.open(paths[0]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(paths[1]) as image:
        robot = np.asarray(image.convert("L")) > 0
    if robot.shape != rgb.shape[:2]:
        return unchanged, False
    black = rgb.astype(np.float64).mean(axis=-1) <= NEAR_BLACK_LUMINANCE_MAX
    black_count = int(black.sum())
    robot_count = int(robot.sum())
    robot_record = wrist["robot_semantic_mask"]
    if robot_count != robot_record.get("pixel_count") or not black_count:
        return unchanged, False
    overlap = int((black & robot).sum()) / black_count
    outside_black = int((black & ~robot).sum()) / max(1, int((~robot).sum()))
    robot_fraction = robot_count / robot.size
    if robot_fraction < 0.5 or overlap < 0.98 or outside_black > 0.05:
        return unchanged, False
    wrist_view["blockers"] = []
    wrist_view["passed"] = True
    wrist_view["robot_body_occlusion"] = {
        "status": "qualified_for_development_policy_input",
        "robot_mask": dict(robot_record),
        "robot_pixel_fraction": robot_fraction,
        "near_black_inside_robot_fraction": overlap,
        "near_black_outside_robot_fraction": outside_black,
        "target_pixel_fraction": target["pixel_fraction"],
    }
    unchanged["blockers"] = [b for b in unchanged["blockers"] if b != expected_blocker]
    unchanged["frame_structure_passed"] = not unchanged["blockers"]
    unchanged["passed"] = unchanged["frame_structure_passed"]
    if unchanged["frame_structure_passed"]:
        unchanged["policy_observation_integrity_blockers"] = [
            b for b in unchanged.get("policy_observation_integrity_blockers", [])
            if b != BLOCKER_FRAME_STRUCTURE_FAILED
        ]
    return unchanged, True


def _sha256_prefixed(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith("sha256:") else f"sha256:{text}"


PACKET_REQUEST_FILENAME = "native_task_arena_packet_request.v1.json"
OFFICIAL_TRANSCODE_IMPLEMENTATION = "nvidia_3dgrut_direct_nurec_transcode"


def appearance_render_backend_from_plan(
    plan: Mapping[str, Any], *, packet_request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Seal the appearance backend the sealed scene plan actually composes.

    The packet request's ``appearance_variant`` (source Gaussian digest and
    authoring implementation) is the preferred identity source; a plan whose
    ``appearance_frame_alignment`` carries ``source_asset_sha256`` and
    ``conversion_identity`` is accepted when no packet request is present.
    """

    if is_task_family_scene_plan(plan):
        return family_render_backend(plan)

    from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
    from blueprint_pipeline.native_task_nurec_render_setup import (
        AppearanceRenderPathError,
        appearance_render_path_from_plan,
    )

    try:
        render_path = appearance_render_path_from_plan(plan)
    except AppearanceRenderPathError as exc:
        raise RuntimeError("policy_canary_appearance_render_path_unresolved") from exc
    rows = [
        row
        for row in plan.get("objects") or []
        if isinstance(row, Mapping) and row.get("semantic_role") == "scene_appearance"
    ]
    if len(rows) != 1 or not str(rows[0].get("sha256") or "").strip():
        raise RuntimeError("policy_canary_scene_appearance_asset_not_exact")
    composed_digest = _sha256_prefixed(rows[0]["sha256"])
    alignment = dict(plan.get("appearance_frame_alignment") or {})
    variant = (packet_request or {}).get("appearance_variant")
    variant = dict(variant) if isinstance(variant, Mapping) else {}
    # Configured-scene compilation seals the same provenance in a nested
    # backend record. Join that producer's identity before adapting its fields.
    if "render_backend" in variant:
        configured = variant["render_backend"]
        if (not isinstance(configured, Mapping)
                or configured.get("schema_version") != "task_evaluation_appearance_render_backend.v1"
                or configured.get("backend_digest") != canonical_digest(configured, digest_field="backend_digest")
                or configured.get("particlefield_digest") != composed_digest
                or configured.get("source_configured_appearance_digest") != variant.get("source_configured_appearance_digest")
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(configured.get("source_configured_appearance_digest") or ""))
                or not str(configured.get("kind") or "").strip()):
            raise RuntimeError("policy_canary_configured_appearance_backend_binding_invalid")
        converter = configured.get("upstream_converter")
        if configured["kind"] == OFFICIAL_TRANSCODE_IMPLEMENTATION and (
            not isinstance(converter, Mapping)
            or converter.get("module") != "threedgrut.export.scripts.transcode"
            or converter.get("source_identity_verified") is not True
            or not re.fullmatch(r"[0-9a-f]{40}", str(converter.get("source_revision") or ""))
        ):
            raise RuntimeError("policy_canary_configured_appearance_converter_invalid")
        variant = {**variant, "source_gaussian_sha256": configured["source_configured_appearance_digest"],
                   "particlefield_authoring_implementation": configured["kind"],
                   "upstream_converter": converter}
    try:
        if render_path == "particlefield_3d_gaussian_splat":
            source_digest = str(
                variant.get("source_gaussian_sha256")
                or alignment.get("source_asset_sha256")
                or ""
            ).strip()
            if not source_digest:
                raise RuntimeError("policy_canary_appearance_source_digest_missing")
            implementation = str(variant.get("particlefield_authoring_implementation") or "")
            upstream = variant.get("upstream_converter")
            upstream = dict(upstream) if isinstance(upstream, Mapping) else {}
            if implementation == OFFICIAL_TRANSCODE_IMPLEMENTATION:
                conversion_identity = (
                    "threedgrut.export.scripts.transcode@"
                    f"{upstream.get('source_revision') or 'unpinned'}"
                )
            elif implementation:
                conversion_identity = (
                    f"{implementation}@{upstream.get('version') or upstream.get('source_revision') or 'unpinned'}"
                )
            else:
                conversion_identity = str(alignment.get("conversion_identity") or "").strip()
            official = conversion_identity.startswith("threedgrut.export.scripts.transcode")
            backend = build_appearance_render_backend(
                kind=(
                    BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE
                    if official
                    else BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE
                ),
                source_asset_digest=_sha256_prefixed(source_digest),
                derived_asset_digest=composed_digest,
                renderer_identity=NATIVE_TASK_ARENA_IMAGE,
                conversion_identity=(
                    conversion_identity
                    or "blueprint_pipeline.particlefield_usd.write_particlefield_usd_from_nurec"
                    "+usd-convert-gsplat:0.1.15"
                ),
                camera_frame_contract="registered_world",
                development_only=not official,
            )
        else:
            backend = build_appearance_render_backend(
                kind=BACKEND_ISAAC_USD_GEOMETRY if render_path == "usd_geometry" else BACKEND_ISAAC_NATIVE_NUREC,
                development_only=render_path == "usd_geometry",
                source_asset_digest=composed_digest,
                derived_asset_digest=None,
                renderer_identity=NATIVE_TASK_ARENA_IMAGE,
                conversion_identity=None,
                camera_frame_contract="registered_world",
            )
    except AppearanceRenderBackendError as exc:
        raise RuntimeError("policy_canary_appearance_render_backend_invalid") from exc
    if backend["launch_render_path"] != render_path:
        raise RuntimeError("policy_canary_appearance_render_backend_mismatch")
    return backend


OBSERVATION_INTEGRITY_AUTHORITY_FILENAME = "policy_observation_integrity_authority.v1.json"


def policy_observation_gate_mode(
    packet_request: Mapping[str, Any], resolved_plan: Mapping[str, Any]
) -> str:
    """Choose the camera gate from the resolved embodiment, before Isaac starts."""
    profile = resolved_plan.get("policy_canary_embodiment_profile")
    if is_team_plan(resolved_plan):
        return "native_team_embodiment_camera"
    if profile is not None:
        from blueprint_pipeline.droid_policy_canary_embodiment import apply_droid_policy_canary_profile
        expected = apply_droid_policy_canary_profile(resolved_plan)["policy_canary_embodiment_profile"]
        if profile != expected:
            raise RuntimeError("policy_canary_official_camera_profile_invalid")
        return "native_official_droid_camera"
    registry = packet_request.get("wrist_camera_mount_registry")
    if registry is not None:
        from blueprint_pipeline.native_task_wrist_camera_mount_sweep import validate_wrist_camera_mount_registry
        validate_wrist_camera_mount_registry(registry)
        return "native_registered_camera_sweep"
    return "sealed_reference_and_human_review"


def _team_embodiment_binding(specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The sealed team-robot binding a team candidate's spec carries, else None (Blueprint's robot)."""

    bindings = [spec["team_embodiment"] for spec in specs.values() if spec.get("team_embodiment") is not None]
    if len(bindings) > 1 or (bindings and len(specs) != 1):
        raise RuntimeError("policy_canary_team_embodiment_binding_ambiguous")
    return dict(bindings[0]) if bindings else None


def preflight_policy_canary_static_inputs(
    *, inputs: Mapping[str, Any], authority: Mapping[str, Any],
    base_scene_plan: Mapping[str, Any], construction: Mapping[str, Any],
    packet_request: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]],
    observation_integrity_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise every cell's static startup path and report all independent refusals."""
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    def check(name: str, callback: Callable[[], Any]) -> Any:
        try:
            value = callback()
            checks.append({"check": name, "status": "passed"})
            return value
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            blockers.append(f"{name}:{reason}")
            checks.append({"check": name, "status": "blocked", "reason": reason})
            return None

    check("runtime_input_manifest", lambda: validate_runtime_input_manifest(inputs))
    check("session_authority", lambda: validate_session_authority(authority))
    check("construction_lineage", lambda: _construction_lineage_mode(
        inputs=inputs, base_scene_plan=base_scene_plan, construction=construction))
    backend = check("appearance_backend", lambda: appearance_render_backend_from_plan(
        base_scene_plan, packet_request=packet_request))
    if authority.get("runtime_inputs_digest") != inputs.get("runtime_inputs_digest"):
        blockers.append("policy_canary_bundle_authority_input_mismatch")
    if set(specs) != set(inputs.get("candidate_ids") or []):
        blockers.append("policy_canary_execution_spec_bindings_missing")
    for candidate, spec in specs.items():
        def validate_spec(candidate: str = candidate, spec: Mapping[str, Any] = spec) -> None:
            if (spec.get("candidate_id") != candidate
                or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")
                or spec.get("task_success_contract_digest") != inputs.get("task_success_contract_digest")
                or evidence_kind(spec) != candidate_execution_kinds(inputs).get(candidate)):
                raise RuntimeError("policy_canary_frozen_execution_spec_mismatch")
            if inputs.get("run_kind") == BETA_RUN_KIND:
                if is_task_family_scene_plan(base_scene_plan):
                    from blueprint_pipeline.task_family_policy_execution import validate_task_family_policy_execution_spec
                    validate_task_family_policy_execution_spec(spec, candidate_id=candidate, scene_plan=base_scene_plan)
                else:
                    from blueprint_pipeline.beta_robolab_policy_execution import validate_robolab_policy_execution_spec
                    validate_robolab_policy_execution_spec(spec, candidate_id=candidate)
        check(f"execution_spec.{candidate}", validate_spec)
    cells: list[dict[str, Any]] = []
    team_binding = check("team_embodiment_binding", lambda: _team_embodiment_binding(specs))

    def resolve(cell: Mapping[str, Any]) -> dict[str, Any]:
        plan = _resolved_scene_plan(base_scene_plan, cell, task_success_contract=inputs.get("task_success_contract"))
        return apply_team_embodiment(plan, team_binding) if team_binding is not None else plan

    for index, cell in enumerate(inputs.get("cells") or []):
        plan = check(f"cell.{index}.resolved_plan", lambda cell=cell: resolve(cell))
        if plan is None:
            continue
        if is_task_family_scene_plan(plan):
            from blueprint_pipeline.task_family_policy_execution import family_policy_execution_binding
            check(f"cell.{index}.native_family_binding", lambda plan=plan: family_policy_execution_binding(plan))
            cells.append({"cell_id": cell.get("cell_id"), "observation_gate_mode": "qualified_native_family"})
            continue
        def validate_native_camera_parameters(plan=plan):
            from blueprint_pipeline.native_task_arena_runtime import camera_runtime_parameters
            for camera in plan["cameras"]:
                camera_runtime_parameters(camera, robot_id=str(plan["robot"]["robot_id"]))
        check(f"cell.{index}.native_camera_parameters", validate_native_camera_parameters)
        if (cell.get("control_diagnostic") or {}).get("mode") == "nonblocking_omitted_by_user":
            def validate_final_camera_start(plan=plan):
                from blueprint_pipeline.native_task_camera_start_configuration import validate_camera_start_configuration
                binding = plan.get("policy_canary_camera_start_configuration")
                if not isinstance(binding, Mapping):
                    raise RuntimeError("policy_camera_final_start_configuration_missing")
                validate_camera_start_configuration(plan, binding)
            check(f"cell.{index}.final_camera_start", validate_final_camera_start)
        def validate_cadence(plan: Mapping[str, Any] = plan) -> None:
            control = float(plan["robot"]["control_hz"]) if is_team_plan(plan) else 15.0
            if not (float(plan["task_spec"]["control_frequency_hz"]) == 15.0
                    and float(plan["cadence"]["control_frequency_hz"]) == control):
                raise RuntimeError("policy_canary_control_frequency_invalid")
        check(f"cell.{index}.cadence", validate_cadence)
        for candidate, spec in specs.items() if not is_team_plan(plan) else ():
            def validate_episode_budget(plan=plan, spec=spec):
                from blueprint_pipeline.adp009d_policy_episode import _resolved_task_spec
                _resolved_task_spec(
                    task_spec=plan["task_spec"], destination_position_world_m=None,
                    settle_window_samples=int(plan["task_spec"]["settle_window_samples"]),
                    max_policy_queries=int(spec["max_policy_queries"]),
                    open_loop_horizon=int(spec["open_loop_horizon"]),
                )
            check(f"cell.{index}.episode_budget.{candidate}", validate_episode_budget)
        mode = check(f"cell.{index}.observation_route", lambda plan=plan: policy_observation_gate_mode(packet_request, plan))
        cells.append({"cell_id": cell.get("cell_id"), "observation_gate_mode": mode})
        if mode == "sealed_reference_and_human_review" and backend is not None:
            gate = preload_observation_integrity_gate(observation_integrity_authority, appearance_render_backend=backend)
            blockers.extend(gate["blockers"])
    return {
        "schema_version": "policy_canary_static_startup_preflight.v1",
        "status": "blocked" if blockers else "passed",
        "run_id": authority.get("run_id"), "runtime_inputs_digest": inputs.get("runtime_inputs_digest"),
        "checks": checks, "cells": cells, "blockers": sorted(set(blockers)),
        "provider_mutation_performed": False, "candidate_policy_queried": False,
        "native_checks_remaining": ["simulator_dependencies_and_device", "appearance_renderer",
            "camera_frames_and_task_visibility", "robot_and_gripper_readback", "embodiment_motion_parity"],
    }


def preflight_policy_canary_runtime_directory(runtime_root: Path) -> dict[str, Any]:
    missing: list[str] = []
    def read(relative: str) -> dict[str, Any]:
        try:
            return _read(runtime_root / relative)
        except (OSError, ValueError) as exc:
            missing.append(f"{relative}:{type(exc).__name__}")
            return {}
    inputs = read("runtime_inputs/policy_canary_runtime_inputs.json")
    authority = read("runtime_inputs/policy_canary_session_authority.json")
    manifest = read("adp_arena_provider_manifest.json")
    observation_path = runtime_root / "runtime_inputs" / OBSERVATION_INTEGRITY_AUTHORITY_FILENAME
    base_scene = read("native_task_packet/native_task_arena_scene_plan.v1.json")
    construction_path = runtime_root / "runtime_inputs/native_task_arena_construction_result.v1.json"
    receipt = preflight_policy_canary_static_inputs(
        inputs=inputs, authority=authority,
        base_scene_plan=base_scene,
        packet_request=read("native_task_packet/native_task_arena_packet_request.v1.json"),
        construction=(read("runtime_inputs/native_task_arena_construction_result.v1.json")
                      if construction_path.is_file() or not is_task_family_scene_plan(base_scene) else {}),
        specs={candidate: read(f"runtime_inputs/policy_execution_spec.{candidate}.json")
               for candidate in inputs.get("candidate_ids") or ()},
        observation_integrity_authority=read(str(observation_path.relative_to(runtime_root))) if observation_path.is_file() else None,
    )
    try:
        _validate_provider_manifest(manifest, runtime_inputs=inputs, authority=authority)
    except Exception as exc:
        missing.append("provider_manifest:" + str(exc))
    receipt["blockers"] = sorted(set([*receipt["blockers"], *missing]))
    receipt["status"] = "blocked" if receipt["blockers"] else "passed"
    return receipt


def preload_observation_integrity_gate(
    authority: Mapping[str, Any] | None,
    *,
    appearance_render_backend: Mapping[str, Any],
    authority_path: Path | None = None,
) -> dict[str, Any]:
    """Decide, before any candidate client is loaded, whether observations may be used.

    No frames exist yet at this point in the session, so this gate checks the
    sealed authority alone: it must be present, valid, bound by digest to the
    exact backend this session launched with, carry a passed same-pose
    reference parity, and an approved human review.  The per-episode gate
    re-checks the same authority against the live frames' structure.
    """

    from blueprint_pipeline.native_task_camera_observability import (
        BLOCKER_APPEARANCE_REFERENCE_PARITY_BACKEND_MISMATCH,
        BLOCKER_APPEARANCE_REFERENCE_PARITY_FAILED,
        BLOCKER_APPEARANCE_REFERENCE_PARITY_MISSING,
        BLOCKER_HUMAN_VISUAL_REVIEW_NOT_APPROVED,
        NativeTaskCameraObservabilityError,
        validate_policy_observation_integrity_authority,
    )

    backend_digest = str(appearance_render_backend.get("receipt_digest") or "")
    receipt: dict[str, Any] = {
        "schema_version": "policy_canary_preload_observation_integrity_gate.v1",
        "authority_path": str(authority_path) if authority_path is not None else None,
        "authority_present": authority is not None,
        "session_backend_receipt_digest": backend_digest,
        "candidate_policy_loaded": False,
        "candidate_policy_queried": False,
        "blockers": [],
        "policy_observation_integrity_passed": False,
    }
    if authority is None:
        receipt["blockers"] = [
            BLOCKER_APPEARANCE_REFERENCE_PARITY_MISSING,
            BLOCKER_HUMAN_VISUAL_REVIEW_NOT_APPROVED,
        ]
        return receipt
    try:
        sealed = validate_policy_observation_integrity_authority(authority)
    except NativeTaskCameraObservabilityError as exc:
        receipt["blockers"] = list(exc.errors)
        return receipt
    blockers: list[str] = []
    receipt["authority_digest"] = sealed.get("authority_digest")
    receipt["authority_backend_receipt_digest"] = sealed[
        "appearance_render_backend_receipt_digest"
    ]
    if sealed["appearance_render_backend_receipt_digest"] != backend_digest:
        blockers.append(BLOCKER_APPEARANCE_REFERENCE_PARITY_BACKEND_MISMATCH)
    elif sealed["appearance_reference_parity"].get("passed") is not True:
        blockers.append(BLOCKER_APPEARANCE_REFERENCE_PARITY_FAILED)
    review_status = str(sealed["human_visual_review"].get("status"))
    receipt["human_visual_review_status"] = review_status
    if review_status != "approved":
        blockers.append(BLOCKER_HUMAN_VISUAL_REVIEW_NOT_APPROVED)
    receipt["blockers"] = sorted(set(blockers))
    receipt["policy_observation_integrity_passed"] = not blockers
    return receipt


def _digest(value: Any) -> str:
    return canonical_digest({"value": value})


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"policy_canary_input_not_object:{path.name}")
    return value


@dataclass(frozen=True)
class CellRuntime:
    """Every simulator, robot, and policy seam one isolated cell process touches.

    Production binds these to Isaac Lab, the native arena runtime, and the two
    frozen policy clients through :func:`isaac_cell_runtime`.  The hermetic
    lifecycle rehearsal binds the same names to fakes that keep Isaac's real
    semantics (``SimulationApp.close`` ends the interpreter, a closed
    environment cannot be rebuilt in-process) while the policy clients stay
    real and the episode runner stays real.  The orchestration code between
    those seams is therefore exercised unchanged on every fast-lane run.
    """

    device: str
    launch_isaac: Callable[..., tuple[Any, Mapping[str, Any]]]
    preflight_dependency_matrix: Callable[..., Mapping[str, Any]]
    prepare_preconstruction: Callable[..., Mapping[str, Any]]
    build_environment: Callable[..., Any]
    prepare_appearance_renderer: Callable[..., Mapping[str, Any]]
    read_device_binding: Callable[..., Mapping[str, Any]]
    gripper_probe: Callable[..., Mapping[str, Any]]
    make_servo: Callable[..., Any]
    make_task_readback: Callable[..., Any]
    build_episode_environment: Callable[..., tuple[Any, Mapping[str, Any]]]
    to_tensor: Callable[[Any], Any]
    policy_client: Callable[..., Any]
    groot_worker_identity: Callable[..., tuple[Mapping[str, Any], Mapping[str, Any]]]
    run_policy_episode: Callable[..., Mapping[str, Any]]
    prepolicy_camera_gate: Callable[..., Mapping[str, Any]] | None = None
    configure_post_gate_renderer: Callable[..., Mapping[str, Any]] | None = None
    make_rigid_task_readback: Callable[..., Any] | None = None
    wrap_rigid_scoring_environment: Callable[..., Any] | None = None
    begin_native_asset_monitor: Callable[..., Any] | None = None
    native_asset_feasibility_gate: Callable[..., Mapping[str, Any]] | None = None


def isaac_cell_runtime() -> CellRuntime:
    """Bind the production Isaac Lab, native arena, and policy client seams.

    Imports stay lazy so the control plane, the bundle builder, and the
    hermetic rehearsal never load Isaac or torch; the provider worker resolves
    them once per isolated cell process.
    """

    from blueprint_pipeline.adp009d_policy_episode import run_policy_episode
    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaDifferentialIkServo,
    )
    from blueprint_pipeline.native_task_arena_construction_worker import (
        _gripper_convention_probe,
        preflight_native_dependency_matrix,
    )
    from blueprint_pipeline.native_task_arena_device_readback import (
        read_native_task_arena_device_binding,
    )
    from blueprint_pipeline.native_task_arena_policy_worker import (
        _runtime_groot_worker_identity,
        _to_tensor,
    )
    from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
    from blueprint_pipeline.native_task_arena_preconstruction import (
        prepare_native_task_arena_preconstruction,
    )
    from blueprint_pipeline.native_task_arena_readback import (
        NativeArticulatedTaskArenaReadback,
        NativeRigidTaskArenaReadback,
    )
    from blueprint_pipeline.native_task_arena_runtime import (
        build_native_task_arena_environment,
    )
    from blueprint_pipeline.native_task_nurec_render_setup import (
        camera_site_appearance_required,
        prepare_site_appearance_renderer,
    )
    from blueprint_pipeline.native_task_rtx_streaming_guard import (
        configure_post_gate_rtx_streaming_wait,
    )
    from blueprint_pipeline.native_task_episode_environment import (
        NativeRigidScoringEnvironment,
        build_native_task_episode_environment,
    )
    from blueprint_pipeline.native_task_isaaclab_launch import (
        NATIVE_TASK_ARENA_DEVICE,
        launch_native_task_isaaclab,
    )
    from blueprint_pipeline.task_object_native_settle_runtime import (
        NativeAssetPhysicsMonitor, run_isaac_asset_feasibility,
    )

    def gripper_probe(*, env: Any, robot: Any, seed: int) -> Mapping[str, Any]:
        # torch is resolved only once Isaac has launched and an environment
        # exists, exactly where the pre-seam worker first imported it.
        import torch

        return _gripper_convention_probe(env=env, robot=robot, seed=seed, torch=torch)

    def prepolicy_camera_gate(
        *,
        simulation_app: Any,
        built: Any,
        packet_request: Mapping[str, Any],
        plan: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]:
        """Prove the task-facing mount and exact reset frames before policy load."""

        if is_team_plan(plan):
            from blueprint_pipeline.native_task_arena_construction_worker import _camera_snapshot
            from blueprint_pipeline.team_embodiment_runtime import team_observation_gate

            receipt = team_observation_gate(
                built=built, plan=plan, output_root=output_root, camera_snapshot=_camera_snapshot,
                site_appearance_render_expected=camera_site_appearance_required(plan))
            (output_root / "prepolicy_observation_gate" / "team_embodiment_observation_gate.v1.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return receipt

        import torch

        from blueprint_pipeline.native_task_arena_construction_worker import (
            _body_pose_world,
            _camera_snapshot,
            _jsonable,
        )
        from blueprint_pipeline.native_task_arena_runtime_preflight_worker import (
            _prepolicy_visual_gate_from_snapshot,
            _run_wrist_camera_mount_sweep,
        )

        env = built.env
        seed = int(plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        positions = _jsonable(getattr(robot.data.joint_pos, "torch", robot.data.joint_pos))[0]
        measured_joints = dict(zip(robot.joint_names, positions, strict=True))
        reset_errors = [
            abs(float(measured_joints[name]) - float(expected))
            for name, expected in plan["robot"]["joint_reset_positions_rad"].items()
        ]
        if any(not (error <= 1e-4) for error in reset_errors):
            raise RuntimeError("policy_canary_native_joint_reset_readback_mismatch")
        reset_observation_readback = {
            "camera_rerenders_on_reset": built.cfg.num_rerenders_on_reset,
            "camera_pose_backends": {
                role: {
                    "view_class": type(env.unwrapped.scene[name]._view).__name__,
                    "use_fabric": getattr(env.unwrapped.scene[name]._view, "_use_fabric", None),
                    "update_latest_camera_pose": env.unwrapped.scene[name].cfg.update_latest_camera_pose,
                }
                for role, name in built.camera_scene_names.items()
            },
            "requested_joint_positions_rad": dict(plan["robot"]["joint_reset_positions_rad"]),
            "observed_joint_positions_rad": {
                name: float(measured_joints[name])
                for name in plan["robot"]["joint_reset_positions_rad"]
            },
            "maximum_joint_reset_error_rad": max(reset_errors, default=0.),
            "native_wrist_camera_attachment": _jsonable(
                getattr(built, "native_configuration_readback", {}).get("native_wrist_camera_attachment")
            ),
        }
        root = output_root / "prepolicy_observation_gate"
        root.mkdir(parents=True, exist_ok=True)
        droid_profile = plan.get("policy_canary_embodiment_profile")
        if plan.get("operator_wrist_camera_aim") is not None:
            selected = dict(built.native_configuration_readback["direct_wrist_camera_aim"])
            selected["admitted"] = True
            selected["blockers"] = []
            selection = {"schema_version": "policy_canary_wrist_camera_mount_selection.v1",
                "status": "selected", "selected_candidate": selected,
                "contact_sheet": None, "blockers": [],
                "selection_digest": canonical_digest(selected)}
        elif (
            isinstance(droid_profile, Mapping)
            and droid_profile.get("preserve_official_policy_camera_calibration")
            is True
        ):
            selected = {
                "candidate_id": "official_arena_droid_wrist_camera",
                "source": "official_arena_droid_embodiment",
                "arena_source": dict(droid_profile["arena_source"]),
                "robot_preset_id": droid_profile["robot_preset_id"],
                "admitted": True,
                "blockers": [],
            }
            selection = {
                "schema_version": "policy_canary_wrist_camera_mount_selection.v1",
                "status": "selected",
                "selected_candidate": selected,
                "contact_sheet": None,
                "blockers": [],
                "selection_digest": canonical_digest(selected),
            }
        else:
            selection = _run_wrist_camera_mount_sweep(
                simulation_app=simulation_app,
                env=env,
                built=built,
                packet_request=dict(packet_request),
                plan=dict(plan),
                output_root=root,
                torch=torch,
                body_pose_reader=_body_pose_world,
                camera_snapshot=_camera_snapshot,
            )
        snapshot = _camera_snapshot(
            env=env,
            camera_scene_names=built.camera_scene_names,
            output_root=root,
            snapshot_id="selected_mount_reset",
            framing_expectations=(plan.get("task_object_observability") or {}).get(
                "cameras"
            ),
            site_appearance_render_expected=camera_site_appearance_required(plan),
        )
        visual = _prepolicy_visual_gate_from_snapshot(
            snapshot=snapshot,
            output_root=root,
        )
        official_droid = (
            isinstance(droid_profile, Mapping)
            and droid_profile.get("policy_camera_roles") == ["external", "wrist"]
            and droid_profile.get("preserve_official_policy_camera_calibration") is True
        )
        wrist_body_qualified = False
        if official_droid:
            visual, wrist_body_qualified = _qualify_official_droid_wrist_body_occlusion(
                snapshot=snapshot, visual=visual, output_root=root
            )
        visibility_contract = _policy_camera_visibility_contract(
            snapshot,
            preserve_official_droid_calibration=official_droid,
            wrist_robot_occlusion_qualified=wrist_body_qualified,
            require_task_support=(
                bool(plan["task_spec"].get("destination_support_asset_id"))
                or any(row.get("semantic_role") == "task_support" for row in plan["objects"])
            ),
        )
        visibility = dict(visibility_contract["camera_visibility"])
        blockers = list((selection or {}).get("blockers") or [])
        blockers.extend(visual.get("blockers") or [])
        blockers.extend(visibility_contract["blockers"])
        if visibility_contract["passed"] is not True:
            blockers.append("policy_canary_task_semantic_visibility_failed")
        from blueprint_pipeline.native_task_asset_composition_gate import run_native_asset_composition_gate

        composition = run_native_asset_composition_gate(
            built=built, plan=plan, output_root=root)
        blockers.extend(composition["blockers"])
        passed = (
            isinstance(selection, Mapping)
            and selection.get("status") == "selected"
            and visual.get("frame_structure_passed") is True
            and not blockers
        )
        receipt: dict[str, Any] = {
            "schema_version": "policy_canary_runtime_observation_integrity_gate.v1",
            "status": "passed" if passed else "blocked",
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "appearance_render_backend_receipt_digest": (
                appearance_render_backend_from_plan(
                    plan, packet_request=packet_request
                )["receipt_digest"]
            ),
            "wrist_camera_mount_selection_digest": (
                selection.get("selection_digest")
                if isinstance(selection, Mapping)
                else None
            ),
            "frame_structure_passed": visual.get("frame_structure_passed") is True,
            "target_semantic_visibility_passed": bool(visibility) and all(
                visibility.values()
            ),
            "camera_visibility": visibility,
            "raw_camera_visibility": visibility_contract["raw_camera_visibility"],
            "camera_role_qualifications": visibility_contract[
                "role_qualifications"
            ],
            "notices": visibility_contract["notices"],
            "selected_wrist_camera_mount": (
                selection.get("selected_candidate")
                if isinstance(selection, Mapping)
                else None
            ),
            "wrist_camera_contact_sheet": (
                selection.get("contact_sheet")
                if isinstance(selection, Mapping)
                else None
            ),
            "snapshot": snapshot,
            "reset_observation_readback": reset_observation_readback,
            "visual_gate": visual,
            "asset_composition_gate": composition,
            "human_visual_review_status": (
                "not_required_for_internal_diagnostic_policy_execution"
            ),
            "candidate_policy_loaded": False,
            "candidate_policy_queried": False,
            "official_ranking_permitted": False,
            "scene_promotion_permitted": False,
            "blockers": sorted(set(str(item) for item in blockers if str(item))),
            "policy_observation_integrity_passed": passed,
            "gate_digest": "",
        }
        receipt["gate_digest"] = canonical_digest(
            receipt, digest_field="gate_digest"
        )
        (root / "policy_canary_runtime_observation_integrity_gate.v1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt

    return CellRuntime(
        device=NATIVE_TASK_ARENA_DEVICE,
        launch_isaac=launch_native_task_isaaclab,
        preflight_dependency_matrix=preflight_native_dependency_matrix,
        prepare_preconstruction=prepare_native_task_arena_preconstruction,
        build_environment=build_native_task_arena_environment,
        prepare_appearance_renderer=prepare_site_appearance_renderer,
        read_device_binding=read_native_task_arena_device_binding,
        gripper_probe=gripper_probe,
        make_servo=NativeFrankaDifferentialIkServo,
        make_task_readback=NativeArticulatedTaskArenaReadback,
        build_episode_environment=build_native_task_episode_environment,
        to_tensor=_to_tensor,
        policy_client=canary_policy_client,
        groot_worker_identity=_runtime_groot_worker_identity,
        run_policy_episode=run_policy_episode,
        prepolicy_camera_gate=prepolicy_camera_gate,
        configure_post_gate_renderer=configure_post_gate_rtx_streaming_wait,
        make_rigid_task_readback=NativeRigidTaskArenaReadback,
        wrap_rigid_scoring_environment=NativeRigidScoringEnvironment,
        begin_native_asset_monitor=NativeAssetPhysicsMonitor,
        native_asset_feasibility_gate=run_isaac_asset_feasibility,
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _seal_result(*, result_path: Path, result: Mapping[str, Any]) -> None:
    """Durably seal one final provider result before process teardown."""

    value = dict(result)
    if value.get("result_digest") != canonical_digest(
        value, digest_field="result_digest"
    ):
        raise RuntimeError("policy_canary_result_digest_invalid_before_close")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.sealing")
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_CLOEXEC", 0),
            0o440,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, result_path)
        directory = os.open(
            result_path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    if result_path.name == PROVIDER_RESULT_FILENAME and value.get("run_kind") == TEAM_RUN_KIND:
        _write_team_status_sidecar(result_path=result_path, result=value)


def _write_team_status_sidecar(*, result_path: Path, result: Mapping[str, Any]) -> None:
    """Bind small diagnostic status to the exact sealed terminal bytes, without grading them."""
    summary = {"schema_version": PROVIDER_STATUS_SCHEMA, "run_kind": TEAM_RUN_KIND,
               "claim_ceiling": "diagnostic_policy_execution", "status": result.get("status"),
               "blockers": result.get("blockers") or [],
               "source_result": {"file_name": result_path.name, "size_bytes": result_path.stat().st_size,
                                 "sha256": _sha256(result_path), "result_digest": result["result_digest"]},
               "inspection_scope": "hash_bound_status_metadata_only", "result_digest": ""}
    summary["result_digest"] = canonical_digest(summary, digest_field="result_digest")
    if len(json.dumps(summary).encode()) > 64 * 1024:
        raise RuntimeError("team_status_summary_budget_exceeded")
    _seal_result(result_path=result_path.with_name(PROVIDER_STATUS_FILENAME), result=summary)


def _seal_result_before_simulation_close(
    *, result_path: Path, result: Mapping[str, Any], simulation_app: Any
) -> None:
    """Seal the provider result before Isaac can terminate the interpreter."""

    close = getattr(simulation_app, "close", None)
    if not callable(close):
        raise RuntimeError("policy_canary_simulation_close_unavailable")
    _seal_result(result_path=result_path, result=result)
    close()


def _aggregate_isolated_cell_results(
    *,
    authority: Mapping[str, Any],
    inputs: Mapping[str, Any],
    child_results: list[Mapping[str, Any]],
    output_root: Path,
    construction_lineage_mode: str,
    archive_members: Mapping[str, Mapping[str, Any]] | None = None,  # control-plane adoption only
) -> dict[str, Any]:
    candidate_ids = tuple(str(value) for value in inputs["candidate_ids"])
    if len(child_results) != len(inputs["cells"]):
        raise RuntimeError("policy_canary_isolated_cell_result_count_invalid")
    indices = [child.get("selected_cell_index") for child in child_results]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices) or set(indices) != set(range(len(inputs["cells"]))):
        raise RuntimeError("policy_canary_isolated_cell_indices_invalid")
    child_results = sorted(child_results, key=lambda child: child["selected_cell_index"])
    episodes: list[dict[str, Any]] = []
    failed_cells: list[dict[str, Any]] = []
    for index, child in enumerate(child_results):
        if not _validate_isolated_cell_result(child=child, inputs=inputs, index=index):
            failed_cells.append({"cell": inputs["cells"][index], "raw_child_result": dict(child)})
            continue
        prefix = f"cell_runs/{index:02d}"
        episodes.extend(
            _prefix_episode_evidence_paths(row, prefix=prefix)
            for row in sorted(child["episodes"], key=lambda row: str(row.get("candidate_id")))
        )
    expected = {
        (candidate, str(cell["cell_id"]), int(cell["seed"]))
        for candidate in candidate_ids
        for cell in inputs["cells"]
        if cell not in [failure["cell"] for failure in failed_cells]
    }
    observed = {
        (str(row.get("candidate_id")), str(row.get("cell_id")), int(row.get("seed")))
        for row in episodes
    }
    prepolicy_only = (inputs.get("run_kind") in (TEAM_RUN_KIND, BETA_RUN_KIND) and bool(failed_cells)
                      and len(failed_cells) == len(child_results))
    if observed != expected or (not episodes and not prepolicy_only):
        raise RuntimeError("policy_canary_isolated_cell_pairing_invalid")
    result: dict[str, Any] = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "runtime_partial_pending_closeout" if failed_cells else "runtime_completed_unqualified_pending_closeout",
        "run_kind": inputs.get("run_kind") or "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(candidate_ids),
        "task_success_contract": inputs["task_success_contract"],
        "task_success_contract_digest": inputs["task_success_contract_digest"],
        "episodes_per_policy": len(inputs["cells"]) - len(failed_cells),
        **retained_rollout_counts(candidate_execution_kinds(inputs), episodes),
        "provider_allocations_observed": None,
        "retry_cap": 0,
        "warm_session_open_count": 1,
        "isolated_simulation_process_count": len(child_results),
        "policy_loads": [
            {"candidate_id": candidate, "loaded_once": True}
            for candidate in candidate_ids
        ] if episodes else [],
        "episodes": episodes,
        "session_closeout": {
            "status": "runtime_closed_pending_provider_teardown",
            "runtime_closed": True,
            "provider_closeout_pending": True,
            "isolated_simulation_process_count": len(child_results),
        },
        "session_failure_type": None,
        "scene_promotion_performed": False,
        "official_ranking_performed": False,
        "candidate_policy_queried": any(
            row.get("candidate_policy_queried") is True for row in episodes
        ),
        "provider_zero_required_after_return": True,
        "construction_lineage_mode": construction_lineage_mode,
        "matrix_digest": inputs.get("matrix_digest") or _digest(inputs["cells"]),
        "result_digest": "",
    }
    if inputs.get("run_kind") == BETA_RUN_KIND:
        result.update(beta_result_binding(inputs))
    if failed_cells:
        result["planned_episodes_per_policy"] = len(inputs["cells"])
        result["isolated_cell_failures"] = failed_cells
    telemetry_index, telemetry_artifacts = _write_policy_canary_telemetry(output_root, result["episodes"], **(
        {"archive_members": archive_members} if archive_members is not None else {}))
    result["telemetry"] = telemetry_index
    from blueprint_pipeline.policy_paired_summary import paired_summary
    result["frozen_cells"] = [{key: cell[key] for key in ("cell_id", "seed", "family", "partition", "cell_spec_digest", "resolved_scenario_digest") if key in cell} for cell in inputs["cells"]]
    result["paired_diagnostic_summary"] = paired_summary(result["episodes"], candidate_ids=candidate_ids,
        planned_cells=result["frozen_cells"]) if len(candidate_ids) == 2 else None  # one team candidate: no pair
    if authority.get("execution_release") is not None:
        result["execution_release"] = authority["execution_release"]
    if child_results[0].get("robot_free_probe_trace") is not None:
        result["robot_free_probe_trace"] = child_results[0]["robot_free_probe_trace"]
    result["artifact_inventory"] = telemetry_artifacts
    result["artifact_inventory_digest"] = _digest(telemetry_artifacts)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _spawn_isolated_cell_process(
    *, index: int, runtime_root: Path, output_root: Path, child_root: Path, controls_only: bool = False
) -> int:
    """Run exactly one matrix cell in a fresh interpreter (and a fresh Isaac)."""

    timeout_seconds = _isolated_cell_timeout_seconds()
    if timeout_seconds <= 0:
        raise subprocess.TimeoutExpired("policy_canary_export_deadline", timeout=0)
    child_log = child_root / "worker_console.log"
    environment = dict(os.environ)
    environment["BLUEPRINT_POLICY_CANARY_CELL_INDEX"] = str(index)
    environment["BLUEPRINT_POLICY_CANARY_CONTROLS_ONLY"] = "1" if controls_only else "0"
    environment["BLUEPRINT_ADP_ARENA_PARENT_OUTPUT_DIR"] = str(output_root)
    environment["BLUEPRINT_ADP_ARENA_OUTPUT_DIR"] = str(child_root)
    try:
        progress_fd = os.dup(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        progress_fd = None
    if progress_fd is not None:
        environment[_CELL_PROGRESS_FD_ENV] = str(progress_fd)
    else:
        environment.pop(_CELL_PROGRESS_FD_ENV, None)
    try:
        with child_log.open("xb") as stream:
            completed = subprocess.run(
                [sys.executable, str(runtime_root / Path(__file__).name)],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                pass_fds=(progress_fd,) if progress_fd is not None else (),
            )
    finally:
        if progress_fd is not None:
            os.close(progress_fd)
    return int(completed.returncode)


def _emit_cell_progress(stage: str, index: int, *, progress_path: Path | None = None) -> None:
    """Expose completed worker milestones to the outer paid-run watchdog."""
    if stage not in _CELL_PROGRESS_STAGES or index < 0:
        raise ValueError("policy_canary_cell_progress_invalid")
    marker = f"BLUEPRINT_POLICY_CANARY_PROGRESS:cell={index}:stage={stage}\n"
    if progress_path is not None:
        with progress_path.open("a", encoding="ascii") as stream:
            stream.write(marker)
    raw_fd = os.environ.get(_CELL_PROGRESS_FD_ENV)
    if raw_fd is None:
        return
    try:
        fd = int(raw_fd)
        if fd < 0:
            return
        os.write(fd, marker.encode("ascii"))
    except (OSError, ValueError):
        # A lost diagnostics channel must not change scientific execution.
        return


def _team_checkpoint(*, output_root: Path, inputs: Mapping[str, Any],
                     authority: Mapping[str, Any], children: list[Mapping[str, Any]]) -> None:
    _retention_checkpoint(output_root=output_root, inputs=inputs, authority=authority,
                          children=children, seal_result=_seal_result)


def _seal_team_prefix(*, output_root: Path, inputs: Mapping[str, Any], authority: Mapping[str, Any],
                      children: list[Mapping[str, Any]], construction_lineage_mode: str,
                      reason: str, stopped_before_next_cell: bool = False,
                      interrupted_cell_index: int | None = None) -> None:
    if inputs.get("run_kind") != TEAM_RUN_KIND or not children:
        return
    result = _aggregate_isolated_cell_results(
        inputs={**inputs, "cells": inputs["cells"][:len(children)]}, authority=authority,
        child_results=children, output_root=output_root, construction_lineage_mode=construction_lineage_mode)
    result.update(status="blocked", blockers=[reason], planned_episodes_per_policy=len(inputs["cells"]),
                  execution_stopped_after_cell=len(children),
                  stopped_before_next_cell=stopped_before_next_cell)
    if interrupted_cell_index is not None:
        result["interrupted_cell_index"] = interrupted_cell_index
        if result.get("candidate_policy_queried") is not True:
            result["candidate_policy_queried"] = None
        if not result.get("policy_loads"):
            result["policy_loads"] = None
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=result)


def _run_isolated_cell_processes(
    *,
    runtime_root: Path | None = None,
    output_root: Path | None = None,
    run_cell_process: Callable[..., int] | None = None,
) -> int:
    runtime = (
        Path(runtime_root).resolve()
        if runtime_root is not None
        else Path(__file__).resolve().parent
    )
    output_root = Path(
        output_root
        or os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    spawn = run_cell_process or _spawn_isolated_cell_process
    output_root.mkdir(parents=True, exist_ok=True)
    inputs = validate_runtime_input_manifest(
        _read(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json")
    )
    authority = validate_session_authority(
        _read(runtime / "runtime_inputs" / "policy_canary_session_authority.json")
    )
    base_scene_plan = _read(
        runtime / "native_task_packet" / "native_task_arena_scene_plan.v1.json"
    )
    construction_path = runtime / "runtime_inputs" / "native_task_arena_construction_result.v1.json"
    construction = _read(construction_path) if construction_path.is_file() or not is_task_family_scene_plan(base_scene_plan) else {}
    construction_lineage_mode = _construction_lineage_mode(
        inputs=inputs,
        base_scene_plan=base_scene_plan,
        construction=construction,
    )
    if controls_required(inputs["task_success_contract"]):
        from blueprint_pipeline.native_policy_canary_witness_delivery import transfer_paired_witness
        return execute_strict_matrix(
            runtime=runtime, output_root=output_root, inputs=inputs, authority=authority,
            spawn=spawn, aggregate=_aggregate_isolated_cell_results, seal=_seal_result,
            construction_lineage_mode=construction_lineage_mode, transfer_pair=transfer_paired_witness,
            stage=os.environ.get("BLUEPRINT_POLICY_CANARY_MATRIX_STAGE", "all"))
    child_results: list[Mapping[str, Any]] = []
    diagnostic_continuation_gate = None
    for index in range(len(inputs["cells"])):
        child_root = output_root / "cell_runs" / f"{index:02d}"
        child_root.mkdir(parents=True, exist_ok=False)
        try:
            exit_code = int(
                spawn(
                    index=index,
                    runtime_root=runtime,
                    output_root=output_root,
                    child_root=child_root,
                )
            )
        except subprocess.TimeoutExpired as exc:
            from blueprint_pipeline.policy_canary_interrupted_cell_recovery import recover_interrupted_cell_result

            reason = ("policy_canary_export_deadline_reached"
                      if _HARD_DEADLINE_ENV in os.environ and _isolated_cell_timeout_seconds() <= 0
                      else "policy_canary_isolated_cell_process_timeout")
            if seal_scripted_interruption(inputs=inputs, authority=authority, children=child_results,
                    child_root=child_root, output_root=output_root, index=index, reason=reason, seal_result=_seal_result):
                return 1
            try:
                child_result_path = child_root / PROVIDER_RESULT_FILENAME
                if child_was_sealed := child_result_path.is_file():
                    # Isaac may hang while closing after sealing its final result.
                    # Keep that verified receipt instead of trying to overwrite it.
                    child = _read(child_result_path)
                    _validate_isolated_cell_result(child=child, inputs=inputs, index=index)
                else:
                    child = recover_interrupted_cell_result(
                        runtime_root=runtime, child_root=child_root,
                        selected_cell_index=index,
                        reason=reason,
                        timeout_seconds=exc.timeout if exc.timeout > 0 else None,
                    )
            except Exception:  # noqa: BLE001 - earlier completed children must survive recovery failure
                _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                    children=child_results, construction_lineage_mode=construction_lineage_mode,
                    reason="policy_canary_isolated_cell_timeout_recovery_failed", interrupted_cell_index=index)
                raise
            if inputs.get("run_kind") == TEAM_RUN_KIND and (child_results or child_was_sealed):
                _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                    children=[*child_results, child] if child_was_sealed else child_results, construction_lineage_mode=construction_lineage_mode,
                    reason=reason, interrupted_cell_index=index)
                return 1
            from blueprint_pipeline.policy_canary_interrupted_cell_recovery import seal_interrupted_session_result
            seal_interrupted_session_result(inputs=inputs, children=[*child_results, child],
                output_root=output_root, interrupted_cell_index=index, reason=reason, seal=_seal_result)
            return 1
        except Exception:
            _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                children=child_results, construction_lineage_mode=construction_lineage_mode,
                reason="policy_canary_isolated_cell_process_failed", interrupted_cell_index=index)
            raise
        child_result_path = child_root / PROVIDER_RESULT_FILENAME
        if not child_result_path.is_file():
            _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                children=child_results, construction_lineage_mode=construction_lineage_mode,
                reason="policy_canary_isolated_cell_result_missing", interrupted_cell_index=index)
            raise RuntimeError(
                f"policy_canary_isolated_cell_result_missing:{index}:"
                f"exit_{exit_code}"
            )
        try:
            child_result = _read(child_result_path)
        except Exception:  # noqa: BLE001 - retain measured prefix even if a child file is unreadable
            _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                children=child_results, construction_lineage_mode=construction_lineage_mode,
                reason="policy_canary_isolated_cell_result_unreadable", interrupted_cell_index=index)
            raise
        child_results.append(child_result)
        if index == 0 and child_results[0].get("robot_free_probe_status") not in (None, "passed"):
            # The scene failed its robot-free check in cell 0: no further team episode runs on it.
            blocked = {"schema_version": "native_task_arena_policy_canary_session_result.v1", "status": "blocked",
                       "run_kind": inputs["run_kind"], "claim_ceiling": "diagnostic_policy_execution",
                       "task_success_contract": inputs["task_success_contract"],
                       "task_success_contract_digest": inputs["task_success_contract_digest"],
                       "episodes": [], "artifact_inventory": [], "candidate_policy_queried": False,
                       "blockers": ["team_scene_robot_free_probe_not_passed"],
                       **{key: child_results[0][key] for key in ("robot_free_probe_trace", "robot_free_probe_status")}}
            if inputs.get("run_kind") == BETA_RUN_KIND:
                blocked.update(beta_result_binding(inputs))
            blocked["result_digest"] = canonical_digest(blocked, digest_field="result_digest")
            _seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=blocked)
            return 1
        if inputs.get("run_kind") == TEAM_RUN_KIND:
            try:
                _team_checkpoint(output_root=output_root, inputs=inputs, authority=authority, children=child_results)
                if len(inputs["cells"]) > 25:
                    child_results[-1] = _compact_team_child_result(child_results[-1], child_root=child_root, output_root=output_root)
                child_result = None  # only the current raw child may occupy memory
            except Exception as exc:  # noqa: BLE001 - preserve validated prefix and stop paid work
                _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                    children=child_results[:-1], construction_lineage_mode=construction_lineage_mode,
                    reason=str(exc), interrupted_cell_index=index)
                raise
            from blueprint_pipeline.team_policy_episode_outcome import FIRST_BATCH_CELLS, first_batch_failure
            if len(child_results) == FIRST_BATCH_CELLS:
                first_rows = [row for child in child_results for row in child.get("episodes", [])]
                reason = first_batch_failure(first_rows) if first_rows else "team_evaluation_shard_blocked"
                if reason:
                    _seal_team_prefix(output_root=output_root, inputs=inputs, authority=authority,
                        children=child_results, construction_lineage_mode=construction_lineage_mode,
                        reason=reason, stopped_before_next_cell=True)
                    return 1
        if index == 0 and inputs.get("diagnostic_continuation_protocol") is not None:
            from blueprint_pipeline.native_policy_canary_diagnostic_continuation import assess_diagnostic_first_cell, GATE_FILENAME
            diagnostic_continuation_gate = assess_diagnostic_first_cell(runtime_root=runtime, child_root=child_root)
            _seal_result(result_path=output_root / GATE_FILENAME, result=diagnostic_continuation_gate)
            if diagnostic_continuation_gate["status"] != "passed":
                if diagnostic_continuation_gate.get("failure_stage") == "prepolicy_observation":
                    result = _prepolicy_refusal_result(child=child_result, gate=diagnostic_continuation_gate,
                        output_root=output_root, cell_index=index, planned_cells=len(inputs["cells"]), gate_filename=GATE_FILENAME)
                    _seal_result(result_path=output_root / PROVIDER_RESULT_FILENAME, result=result)
                    return 1
                raise RuntimeError("diagnostic_continuation_first_cell_blocked:" + ",".join(diagnostic_continuation_gate["blockers"]))
        elif index == 0 and inputs.get("run_kind") not in (TEAM_RUN_KIND, BETA_RUN_KIND):
            # The DROID harness-parity probe qualifies Blueprint's frozen canary; a team's
            # policy that does not approach the object in cell 0 is scored, not blocked.
            diagnostics = [
                row.get("embodiment_parity_diagnostic")
                for row in child_results[-1].get("episodes") or []
                if isinstance(row, Mapping)
            ]
            if (
                len(diagnostics) != len(inputs["candidate_ids"])
                or any(
                    not isinstance(row, Mapping) or row.get("status") != "passed"
                    for row in diagnostics
                )
            ):
                raise RuntimeError("droid_policy_canary_embodiment_parity_probe_failed")
    result = _aggregate_isolated_cell_results(
        authority=authority,
        inputs=inputs,
        child_results=child_results,
        output_root=output_root,
        construction_lineage_mode=construction_lineage_mode,
    )
    if diagnostic_continuation_gate is not None:
        result["diagnostic_continuation_gate"] = diagnostic_continuation_gate
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _seal_result(
        result_path=output_root / PROVIDER_RESULT_FILENAME,
        result=result,
    )
    return 0


def _run_selected_cell(
    selected_cell_index: int,
    *,
    runtime_root: Path | None = None,
    output_root: Path | None = None,
    provider_output_root: Path | None = None,
    controls_only: bool = False,
    cell_runtime: CellRuntime | None = None,
) -> int:
    runtime = (
        Path(runtime_root).resolve()
        if runtime_root is not None
        else Path(__file__).resolve().parent
    )
    output_root = Path(
        output_root
        or os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    provider_output_root = Path(
        provider_output_root
        or os.environ.get("BLUEPRINT_ADP_ARENA_PARENT_OUTPUT_DIR")
        or output_root
    ).resolve()
    bound_runtime = cell_runtime
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / PROVIDER_RESULT_FILENAME
    def report_progress(stage: str) -> None:
        _emit_cell_progress(stage, selected_cell_index, progress_path=output_root / "cell_progress.log")
    inputs = validate_runtime_input_manifest(
        _read(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json")
    )
    authority = validate_session_authority(
        _read(runtime / "runtime_inputs" / "policy_canary_session_authority.json")
    )
    provider_manifest = _validate_provider_manifest(
        _read(runtime / "adp_arena_provider_manifest.json"),
        runtime_inputs=inputs,
        authority=authority,
    )
    base_scene_plan = _read(
        runtime / "native_task_packet" / "native_task_arena_scene_plan.v1.json"
    )
    family_scene = is_task_family_scene_plan(base_scene_plan)
    if family_scene:
        os.environ["BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH"] = str(verify_family_operator_inputs(runtime, provider_manifest))
    if bound_runtime is None and not family_scene:
        bound_runtime = isaac_cell_runtime()
    construction_path = runtime / "runtime_inputs" / "native_task_arena_construction_result.v1.json"
    construction = _read(construction_path) if construction_path.is_file() or not family_scene else {}
    construction_lineage_mode = _construction_lineage_mode(
        inputs=inputs,
        base_scene_plan=base_scene_plan,
        construction=construction,
    )
    specs = {
        candidate: _read(
            runtime
            / "runtime_inputs"
            / f"policy_execution_spec.{candidate}.json"
        )
        for candidate in inputs["candidate_ids"]
    }
    if set(provider_manifest.get("execution_spec_digests") or {}) != set(specs):
        raise RuntimeError("policy_canary_execution_spec_bindings_missing")
    for candidate, spec in specs.items():
        if (spec.get("candidate_id") != candidate
                or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")
                or spec.get("execution_spec_digest") != provider_manifest["execution_spec_digests"][candidate]
                or spec.get("task_success_contract_digest") != inputs["task_success_contract_digest"]):
            raise RuntimeError("policy_canary_frozen_execution_spec_mismatch:" + candidate)
    current_env: dict[str, Any] = {}
    current_session: dict[str, Any] = {}
    team_binding = _team_embodiment_binding(specs)
    from .native_task_runtime_capabilities import policy_controller
    controller_backend = policy_controller(team_binding=team_binding, robot_id=str(base_scene_plan["robot"]["robot_id"]))
    probe_request = next((spec["robot_free_probe"] for spec in specs.values()
                          if spec.get("robot_free_probe") is not None), None)

    def team_plan(plan: dict[str, Any]) -> dict[str, Any]:
        return apply_team_embodiment(plan, team_binding) if team_binding is not None else plan

    packet_request_path = runtime / "native_task_packet" / PACKET_REQUEST_FILENAME
    packet_request = _read(packet_request_path) if packet_request_path.is_file() else {}
    appearance_render_backend = appearance_render_backend_from_plan(
        base_scene_plan,
        packet_request=packet_request or None,
    )
    authority_path = (
        runtime / "runtime_inputs" / OBSERVATION_INTEGRITY_AUTHORITY_FILENAME
    )
    observation_integrity_authority = (
        _read(authority_path) if authority_path.is_file() else None
    )
    static_preflight = preflight_policy_canary_static_inputs(
        inputs=inputs, authority=authority, base_scene_plan=base_scene_plan,
        construction=construction, packet_request=packet_request, specs=specs,
        observation_integrity_authority=observation_integrity_authority,
    )
    static_preflight["result_digest"] = canonical_digest(static_preflight, digest_field="result_digest")
    _seal_result(result_path=output_root / "policy_canary_static_startup_preflight.v1.json", result=static_preflight)
    if static_preflight["status"] != "passed":
        raise RuntimeError("policy_canary_static_startup_preflight_failed:" + ",".join(static_preflight["blockers"]))
    report_progress("static_preflight_passed")

    def open_session(_inputs: Mapping[str, Any]) -> dict[str, Any]:
        if family_scene:
            current_session["native_family"] = True
            return {"native_family": True, "provider_session_identity": authority["authority_digest"]}
        report_progress("isaac_launch_started")
        simulation_app, launch = bound_runtime.launch_isaac(
            provider_output_root / "native_task_runtime_source_provisioning.v1.json",
            device=bound_runtime.device, controller_backend=controller_backend,
            appearance_render_path=appearance_render_backend["launch_render_path"],
        )
        report_progress("isaac_launch_completed")
        current_session["simulation_app"] = simulation_app
        if inputs.get("run_kind") == BETA_RUN_KIND:
            from blueprint_pipeline.beta_robolab_runtime import preflight_robolab_runtime
            current_session["robolab_runtime_preflight"] = preflight_robolab_runtime()
        if base_scene_plan.get("task_spec", {}).get("astra_asset_adoption") is not None:
            if bound_runtime.begin_native_asset_monitor is None:
                raise RuntimeError("policy_canary_native_asset_monitor_unavailable")
            current_session["native_asset_monitor"] = bound_runtime.begin_native_asset_monitor()
        return {
            "simulation_app": simulation_app,
            "launch": launch,
            "appearance_render_backend": dict(appearance_render_backend),
            "provider_session_identity": _digest(
                {"launch": launch, "appearance_render_backend": appearance_render_backend}
            ),
        }

    def load_policy(_session: Mapping[str, Any], candidate: str) -> dict[str, Any]:
        spec = specs[candidate]
        groot_identity = None
        runtime_identity: Mapping[str, Any] = spec.get("runtime_identity") or {}
        if spec.get("worker_identity_requirement") == "groot_droid_runtime_measurement":
            groot_identity, runtime_identity = bound_runtime.groot_worker_identity(
                output_root=provider_output_root, spec=spec
            )
        if family_scene:
            from blueprint_pipeline.team_candidate_policy_client import canary_policy_client
            client = canary_policy_client(spec)
        else:
            client = bound_runtime.policy_client(
                spec, groot_worker_identity_receipt=groot_identity
            )
        report_progress("policy_loaded")
        return {
            "candidate_id": candidate,
            "client": client,
            "spec": spec,
            "checkpoint_digest": spec["checkpoint_digest"],
            "runtime_identity_digest": spec.get("runtime_identity_digest")
            or _digest(runtime_identity),
        }

    def _run_episode_impl(
        _session: Mapping[str, Any],
        policy: Mapping[str, Any],
        context: Mapping[str, Any],
        episode_progress: dict[str, Any],
    ) -> dict[str, Any]:
        started_ns = time.time_ns()
        from blueprint_pipeline.adp009d_droid_action_execution import GripperConvention
        from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker
        from blueprint_pipeline.policy_interface_binding import observation_interface

        scene_plan = team_plan(_resolved_scene_plan(base_scene_plan, context))
        if family_scene:
            return _run_family_canary_episode(
                scene_plan=scene_plan, policy=policy, context=context,
                authority=authority, inputs=inputs, packet_root=runtime / "native_task_packet",
                output_root=output_root, progress=episode_progress,
                site_qualification=current_session["native_family_gate"]["site_qualification"],
            )
        dependencies = bound_runtime.preflight_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"]), controller_backend=controller_backend
        )
        if not dependencies["all_required_available"]:
            raise RuntimeError("policy_canary_dependency_preflight_failed")
        preconstruction = bound_runtime.prepare_preconstruction(
            expected_device=bound_runtime.device
        )
        if not preconstruction["passed"]:
            raise RuntimeError("policy_canary_preconstruction_failed")
        built = current_env.get("built")
        if built is None:
            built = bound_runtime.build_environment(
                scene_plan,
                device=bound_runtime.device,
                bundle_root=runtime / "native_task_packet",
                preconstruction_receipt=preconstruction,
            )
            appearance_renderer = bound_runtime.prepare_appearance_renderer(
                simulation_app=current_session["simulation_app"],
                plan=scene_plan,
            )
            if appearance_renderer.get("passed") is not True:
                raise RuntimeError("policy_canary_appearance_renderer_unqualified")
            current_env["built"] = built
            current_env["cell_id"] = str(context["cell_id"])
            current_env["appearance_renderer"] = dict(appearance_renderer)
        elif current_env.get("cell_id") != str(context["cell_id"]):
            raise RuntimeError("policy_canary_isolated_cell_environment_mismatch")
        device = bound_runtime.read_device_binding(
            built, expected_device=bound_runtime.device
        )
        if not device["passed"]:
            raise RuntimeError("policy_canary_device_binding_failed")
        if (probe_request is not None and selected_cell_index == 0
                and "robot_free_probe_trace" not in current_session
                and bound_runtime.make_rigid_task_readback is not None):
            # Beta L5: record the scene's undecided robot-free probe in the built scene, before
            # any episode, with the robot idle; closeout scores and retains it.
            from blueprint_pipeline.team_embodiment_runtime import record_arena_robot_free_probe

            current_session["robot_free_probe_trace"] = record_arena_robot_free_probe(
                built=built, task_readback=bound_runtime.make_rigid_task_readback(built), plan=scene_plan,
                request=probe_request)
            from blueprint_pipeline.team_embodiment_runtime import probe_status

            current_session["robot_free_probe_status"] = probe_status(
                request=probe_request, trace=current_session["robot_free_probe_trace"])
        if current_session.get("robot_free_probe_status") not in (None, "passed"):
            # Owner decision 2026-10-12: no team episode on a scene whose success check is unproven.
            raise RuntimeError("team_scene_robot_free_probe_not_passed")
        env = built.env
        seed = int(context["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        if is_team_plan(scene_plan):
            # A team's own robot: its own loop, the same scorer, media and failure evidence.
            from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker
            from blueprint_pipeline.team_embodiment_runtime import run_team_episode

            tracker = _PolicyQueryTracker(policy["client"])
            spec = policy["spec"]
            episode_id = f"{authority['run_id']}--{context['cell_id']}--{context['candidate_id']}"
            environment_receipt = {
                "adapter": "team_embodiment_runtime", "embodiment_digest": scene_plan["robot"]["embodiment_digest"],
                "task_state_source": "native_rigid_scoring_frame_and_filtered_contact_readback",
                "appearance_renderer": dict(current_env["appearance_renderer"]),
                "post_gate_rtx_streaming_guard": dict(current_session.get("post_gate_rtx_streaming_guard") or {}),
            }
            try:
                episode = run_team_episode(
                    built=built, task_readback=bound_runtime.make_rigid_task_readback(built), policy=tracker,
                    plan=scene_plan, prompt=str(spec["prompt"]), max_policy_queries=int(spec["max_policy_queries"]),
                    open_loop_horizon=int(spec["open_loop_horizon"]), media_root=output_root / "episodes",
                    episode_id=episode_id, progress=episode_progress,
                    evaluation_backend=spec.get("evaluation_backend"))
                _require_completed_episode_media(
                    output_root, episode, camera_ids=[*(row["role"] for row in scene_plan["cameras"]
                                                        if row.get("policy_input") is True), "overview"])
            except Exception as exc:
                if evidence_kind(spec) == SCRIPTED:
                    episode_progress.update(scripted_fields(invoked=tracker.scripted_controller_invoked))
                failure_path = _write_episode_failure_gap(
                    output_root=output_root, run_id=str(authority["run_id"]), context=context, failure=exc,
                    progress=episode_progress)
                raise PolicyCanaryEpisodeFailure(cause=exc, evidence=_read(failure_path)) from exc
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
                current_env.clear()
        else:
            gripper = bound_runtime.gripper_probe(env=env, robot=robot, seed=seed)
            if gripper["status"] != "measured":
                raise RuntimeError("policy_canary_gripper_unresolved")
            env.reset(seed=seed)
            servo = bound_runtime.make_servo(
                env=env, robot=robot, gripper_convention=gripper
            )
            task_kind = str(scene_plan["task_kind"])
            if task_kind == "articulated_open_close":
                task_readback = bound_runtime.make_task_readback(
                    built,
                    grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
                )
            elif task_kind == "rigid_pick_place":
                if bound_runtime.make_rigid_task_readback is None:
                    raise RuntimeError("policy_canary_rigid_task_readback_missing")
                task_readback = bound_runtime.make_rigid_task_readback(built)
            else:
                task_readback = None
            episode_environment, environment_receipt = bound_runtime.build_episode_environment(
                built=built,
                gripper_convention=gripper,
                servo=servo,
                task_readback=task_readback,
                to_tensor=bound_runtime.to_tensor,
            )
            if task_kind == "rigid_pick_place":
                if bound_runtime.wrap_rigid_scoring_environment is None:
                    raise RuntimeError("policy_canary_rigid_scoring_environment_missing")
                episode_environment = bound_runtime.wrap_rigid_scoring_environment(
                    environment=episode_environment,
                    task_readback=task_readback,
                    task_spec=scene_plan["task_spec"],
                )
                environment_receipt = {
                    **dict(environment_receipt),
                    "task_state_source": (
                        "native_rigid_scoring_frame_and_filtered_contact_readback"
                    ),
                }
            environment_receipt = {
                **dict(environment_receipt),
                "appearance_renderer": dict(current_env["appearance_renderer"]),
                "post_gate_rtx_streaming_guard": dict(
                    current_session.get("post_gate_rtx_streaming_guard") or {}
                ),
            }
            tracker = _PolicyQueryTracker(policy["client"])
            spec = policy["spec"]
            episode_id = f"{authority['run_id']}--{context['cell_id']}--{context['candidate_id']}"
            def scientific_reset_reader():
                from blueprint_pipeline.policy_scientific_reset import ScientificResetScenarioMismatch, compare_reset_readbacks, read_native_reset_channels, seal_reset_readback
                binding = {
                    "candidate_id": context["candidate_id"], "cell_id": context["cell_id"],
                    "seed": context["seed"], "resolved_scenario_digest": context["resolved_scenario_digest"],
                    "task_spec_digest": canonical_digest(scene_plan["task_spec"]),
                }
                try:
                    channels = read_native_reset_channels(built, episode_environment)
                except ScientificResetScenarioMismatch as exc:
                    episode_progress["scientific_reset"] = seal_reset_readback(binding=binding, **exc.channels)
                    raise
                receipt = seal_reset_readback(binding=binding, **channels)
                reference = current_env.setdefault("scientific_reset_reference", receipt)
                parity = compare_reset_readbacks(reference, receipt)
                if parity["status"] == "mismatch":
                    raise RuntimeError("policy_canary_scientific_reset_pair_mismatch:" + ",".join(parity["mismatches"]))
                return receipt
            try:
                episode = bound_runtime.run_policy_episode(
                    environment=episode_environment,
                    policy=tracker,
                    candidate_id=str(context["candidate_id"]),
                    interface_candidate_id=observation_interface(spec),
                    prompt=str(spec["prompt"]),
                    task_spec=scene_plan["task_spec"],
                    max_policy_queries=int(spec["max_policy_queries"]),
                    settle_window_samples=int(
                        scene_plan["task_spec"]["settle_window_samples"]
                    ),
                    open_loop_horizon=int(spec["open_loop_horizon"]),
                    gripper=GripperConvention(
                        closed_command=float(gripper["closed_command"]),
                        open_command=float(gripper["open_command"]),
                        measured_by_probe=True,
                    ),
                    media_output_dir=output_root / "episodes",
                    episode_id=episode_id,
                    scoring_authorized=True,
                    require_complete_multicamera_media=True,
                    require_prestart_readiness=True,
                    observation_integrity={
                        "authority": observation_integrity_authority,
                        "appearance_render_backend_receipt_digest": (
                            appearance_render_backend["receipt_digest"]
                        ),
                        "runtime_gate": current_session.get(
                            "policy_observation_runtime_gate"
                        ),
                    },
                    progress=episode_progress,
                    scientific_reset_reader=scientific_reset_reader,
                    **({"evaluation_backend": spec["evaluation_backend"]}
                       if spec.get("evaluation_backend") else {}),
                )
                try:
                    _require_completed_episode_media(output_root, episode)
                except (OSError, ValueError) as media_error:
                    episode_progress["media_integrity_failure"] = str(media_error)
                    raise
            except Exception as exc:
                failure_path = _write_episode_failure_gap(
                    output_root=output_root,
                    run_id=str(authority["run_id"]),
                    context=context,
                    failure=exc,
                    progress=episode_progress,
                )
                raise PolicyCanaryEpisodeFailure(
                    cause=exc,
                    evidence=_read(failure_path),
                ) from exc
            finally:
                if str(context["candidate_id"]) == str(inputs["candidate_ids"][-1]):
                    close = getattr(env, "close", None)
                    if callable(close):
                        close()
                    current_env.clear()
        visual = episode.get("visual_evidence") or {}
        media = episode.get("media_artifacts") or {}
        motion = episode.get("motion_evidence") or {}
        telemetry = {
            "schema_version": "policy_canary_episode_telemetry.v1",
            "timebase": "unix_ns",
            "started_at_unix_ns": started_ns,
            "completed_at_unix_ns": time.time_ns(),
            "camera_calibration": episode.get("camera_calibration"),
            "policy_query_latency": episode.get("policy_query_latency"),
            "resource_telemetry": episode.get("resource_telemetry"),
            "channels": episode.get("telemetry_channels"),
            "wall_time_ns": time.time_ns() - started_ns,
            "evidence_gaps": [
                name
                for name, value in (
                    ("camera_calibration_unavailable", episode.get("camera_calibration")),
                    ("policy_query_latency_unavailable", episode.get("policy_query_latency")),
                    ("resource_telemetry_unavailable", episode.get("resource_telemetry")),
                    ("telemetry_channels_unavailable", episode.get("telemetry_channels")),
                )
                if value is None
            ],
            "mcap_gap": (
                None
                if episode.get("mcap_artifact")
                else "mcap_library_or_runtime_capture_unavailable"
            ),
        }
        media_artifacts = episode.get("media_artifacts") or []
        media_root = output_root / "episodes"
        observation_support_rows = [
            ((row.get("policy_inference_evidence") or {}).get(
                "eef_position_observed_support"
            ) or {})
            for row in episode.get("queries") or []
            if isinstance(row, Mapping)
        ]
        observation_support_qualified = bool(
            spec.get("require_observed_eef_support") is not True
            or (
                observation_support_rows
                and all(
                    row.get("inside_checkpoint_observed_extrema") is True
                    for row in observation_support_rows
                )
            )
        )
        parity_diagnostic = _episode_embodiment_parity_diagnostic(
            episode,
            observation_support_qualified=observation_support_qualified,
        )
        episode["embodiment_parity_diagnostic"] = parity_diagnostic
        evidence_artifacts = {
            "episode_receipt": {**_write_episode_json_artifact(output_root,
                episode_id=episode_id, role="episode_receipt", value=episode), "media_root_relative": "episodes"},
            "frame_manifest": _bound_media_artifact(
                output_root,
                media_root=media_root,
                artifacts=media_artifacts,
                role="lossless_frame_manifest",
                role_match=lambda name: "frame_manifest" in name,
            ),
            "review_video": _bound_media_artifact(
                output_root,
                media_root=media_root,
                artifacts=media_artifacts,
                role="review_video",
                role_match=lambda name: "video" in name,
            ),
            "reset_state": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="reset_state",
                value={"adapter_binding": environment_receipt, "scientific_reset": episode.get("scientific_reset")},
            ),
            "policy_query_receipt": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="policy_query_receipt",
                value={
                    "candidate_policy_queried": tracker.candidate_policy_queried,
                    **(scripted_fields(invoked=tracker.scripted_controller_invoked) if evidence_kind(spec) == SCRIPTED else {}),
                    "exact_policy_request_artifacts": episode.get("policy_request_artifacts") or [],
                    "policy_request_evidence_complete": episode.get("policy_request_evidence_complete") is True,
                    "sensor_freshness": episode.get("sensor_freshness") or [],
                    "policy_queries": episode.get("policy_queries"),
                    "policy_query_latency": episode.get("policy_query_latency"),
                    "queries": episode.get("queries"),
                    "candidate_policy_action_queries": episode.get(
                        "candidate_policy_action_queries"
                    ),
                    "observation_support_qualified": (
                        observation_support_qualified
                    ),
                },
            ),
            "action_sequence": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="action_sequence",
                value=episode.get("commanded_actions"),
            ),
            "action_delivery_readback": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="action_delivery_readback",
                value=motion,
            ),
            "state_trace": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="state_trace",
                value=episode.get("state_trace"),
            ),
            "contact_force_trace": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="contact_force_trace",
                value=episode.get("contact_force_evidence"),
            ),
            "task_object_trajectory": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="task_object_trajectory",
                value=episode.get("task_object_trajectory"),
            ),
            "score_receipt": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="score_receipt",
                value=episode.get("score"),
            ),
            "embodiment_parity_diagnostic": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="embodiment_parity_diagnostic",
                value=parity_diagnostic,
            ),
        }
        return {
            "status": "completed",
            "candidate_policy_queried": tracker.candidate_policy_queried,
            **(scripted_fields(invoked=tracker.scripted_controller_invoked) if evidence_kind(spec) == SCRIPTED else {}),
            "actions_reached_robot": bool(motion.get("actions_reached_robot")),
            "arm_moved": bool(motion.get("arm_moved")),
            "observation_support_qualified": observation_support_qualified,
            "embodiment_parity_diagnostic": parity_diagnostic,
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
            "scientific_reset": episode.get("scientific_reset"),
            "lossless_frame_manifest_digest": _digest(visual),
            "review_video_digest": _digest(media),
            "returned_action_sequence_digest": _digest(
                episode.get("commanded_actions")
            ),
            "action_delivery_readback_digest": _digest(motion),
            "state_trace_digest": _digest(episode.get("state_trace")),
            "contact_force_digest": _digest(episode.get("contact_force_evidence")),
            "task_object_trajectory_digest": _digest(
                episode.get("task_object_trajectory")
            ),
            "deterministic_score_digest": _digest(episode.get("score")),
            "scoring_authority": "deterministic_simulator_state",
            "episode": episode,
            "episode_environment": environment_receipt,
            "telemetry": telemetry,
            "telemetry_digest": _digest(telemetry),
            "code_identity_digest": authority.get("source_commit_digest")
            or authority["authority_digest"],
            "container_identity_digest": policy["runtime_identity_digest"],
            "scene_revision_digest": inputs.get("scene_revision_digest")
            or inputs["configuration_digest"],
            "scoring_version_digest": _digest("deterministic_simulator_state"),
            "evidence_artifacts": evidence_artifacts,
        }

    def run_episode(
        session: Mapping[str, Any],
        policy: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        episode_progress: dict[str, Any] = {}
        try:
            result = _run_episode_impl(session, policy, context, episode_progress)
            report_progress("episode_completed")
            return result
        except PolicyCanaryEpisodeFailure:
            raise
        except Exception as exc:
            failure_path = _write_episode_failure_gap(
                output_root=output_root,
                run_id=str(authority["run_id"]),
                context=context,
                failure=exc,
                progress=episode_progress,
            )
            raise PolicyCanaryEpisodeFailure(
                cause=exc,
                evidence=_read(failure_path),
            ) from exc

    def close_policy(policy: Mapping[str, Any]) -> None:
        close = getattr(policy.get("client"), "close", None)
        if callable(close):
            close()

    def close_session(session: Mapping[str, Any]) -> dict[str, Any]:
        if family_scene:
            return {"status": "native_family_runtime_closed", "runtime_closed": True,
                    "provider_closeout_pending": True}
        close = getattr(session.get("simulation_app"), "close", None)
        if not callable(close) or session.get("simulation_app") is not current_session.get(
            "simulation_app"
        ):
            raise RuntimeError("policy_canary_simulation_close_unavailable")
        return {
            "status": "runtime_close_committed_after_result_seal",
            "runtime_closed": True,
            "runtime_close_deferred_until_result_sealed": True,
            "provider_closeout_pending": True,
        }

    def prepolicy_observation_gate(session: Mapping[str, Any]) -> dict[str, Any]:
        if family_scene:
            plan = _resolved_scene_plan(base_scene_plan, inputs["cells"][selected_cell_index],
                task_success_contract=inputs["task_success_contract"])
            current_session["native_family_gate"] = qualify_family_policy_scene(plan=plan, specs=specs,
                packet_root=runtime / "native_task_packet", output_root=output_root)
            return current_session["native_family_gate"]
        cell = inputs["cells"][selected_cell_index]
        scene_plan = team_plan(_resolved_scene_plan(
            base_scene_plan, cell, task_success_contract=inputs["task_success_contract"],
        ))
        if policy_observation_gate_mode(packet_request, scene_plan) != "sealed_reference_and_human_review":
            if bound_runtime.prepolicy_camera_gate is None:
                raise RuntimeError("policy_canary_runtime_camera_gate_unavailable")
            dependencies = bound_runtime.preflight_dependency_matrix(
                robot_id=str(scene_plan["robot"]["robot_id"]), controller_backend=controller_backend
            )
            (output_root / "prepolicy_dependency_matrix.v1.json").write_text(
                json.dumps(dependencies, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if not dependencies["all_required_available"]:
                raise RuntimeError("policy_canary_dependency_preflight_failed")
            preconstruction = bound_runtime.prepare_preconstruction(
                expected_device=bound_runtime.device
            )
            if not preconstruction["passed"]:
                raise RuntimeError("policy_canary_preconstruction_failed")
            built = bound_runtime.build_environment(
                scene_plan,
                device=bound_runtime.device,
                bundle_root=runtime / "native_task_packet",
                preconstruction_receipt=preconstruction,
            )
            appearance_renderer = bound_runtime.prepare_appearance_renderer(
                simulation_app=current_session["simulation_app"],
                plan=scene_plan,
            )
            if appearance_renderer.get("passed") is not True:
                raise RuntimeError("policy_canary_appearance_renderer_unqualified")
            current_env.update(
                built=built,
                cell_id=str(cell["cell_id"]),
                appearance_renderer=dict(appearance_renderer),
            )
            feasibility = None
            if scene_plan.get("task_spec", {}).get("astra_asset_adoption") is not None:
                if bound_runtime.native_asset_feasibility_gate is None:
                    raise RuntimeError("policy_canary_native_asset_feasibility_gate_unavailable")
                feasibility = dict(bound_runtime.native_asset_feasibility_gate(
                    built=built, plan=scene_plan, monitor=current_session["native_asset_monitor"],
                    output_root=output_root / "prepolicy_asset_feasibility"))
                current_session["native_asset_feasibility_gate"] = feasibility
                if feasibility.get("status") != "passed":
                    return {**feasibility, "policy_observation_integrity_passed": False}
            gate = dict(
                bound_runtime.prepolicy_camera_gate(
                    simulation_app=current_session["simulation_app"],
                    built=built,
                    packet_request=packet_request,
                    plan=scene_plan,
                    output_root=output_root,
                )
            )
            if feasibility is not None:
                gate["native_asset_feasibility"] = feasibility
                gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")
                (output_root / "prepolicy_asset_feasibility/combined_observation_gate.json").write_text(
                    json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if gate.get("status") != "passed" or gate.get("policy_observation_integrity_passed") is not True:
                current_session["policy_observation_runtime_gate"] = gate
                return gate
            if bound_runtime.configure_post_gate_renderer is None:
                raise RuntimeError(
                    "policy_canary_post_gate_renderer_guard_unavailable"
                )
            renderer_guard = dict(
                bound_runtime.configure_post_gate_renderer(
                    observation_gate=gate,
                    output_path=(
                        output_root
                        / "prepolicy_observation_gate"
                        / "post_gate_rtx_streaming_guard.v1.json"
                    ),
                )
            )
            if renderer_guard.get("status") != "configured":
                raise RuntimeError("policy_canary_post_gate_renderer_guard_failed")
            current_session["policy_observation_runtime_gate"] = gate
            current_session["post_gate_rtx_streaming_guard"] = renderer_guard
            report_progress("observation_gate_passed")
            return gate
        gate = preload_observation_integrity_gate(
            observation_integrity_authority,
            appearance_render_backend=dict(session["appearance_render_backend"]),
            authority_path=authority_path,
        )
        if gate.get("status") == "passed":
            report_progress("observation_gate_passed")
        return gate

    execution_binding = {"run_id": authority.get("run_id"), "authority_digest": authority["authority_digest"],
                         "runtime_inputs_digest": inputs["runtime_inputs_digest"],
                         "implementation_commit": provider_manifest.get("implementation_commit"),
                         "execution_release": authority.get("execution_release")}

    def prepolicy_control_gate(_session: Mapping[str, Any]) -> dict[str, Any]:
        cell = inputs["cells"][selected_cell_index]
        plan = _resolved_scene_plan(base_scene_plan, cell, task_success_contract=inputs["task_success_contract"])
        controls_root = provider_output_root / "control_runs" / f"{selected_cell_index:02d}"
        receipt = _read(controls_root / "policy_canary_cell_controls.v1.json")
        validate_controls_receipt(receipt, scene_plan=plan,
                                  gate=current_session.get("policy_observation_runtime_gate") or {}, root=controls_root,
                                  execution_binding=execution_binding)
        return {"status": "passed", "control_receipt_digest": receipt["receipt_digest"]}

    if controls_only:
        try:
            if not controls_required(inputs["task_success_contract"]):
                raise RuntimeError("strict_controls_mode_requires_explicit_contract")
            session = open_session(inputs)
            gate = prepolicy_observation_gate(session)
            if current_env.get("built") is None:
                raise RuntimeError("strict_controls_native_camera_gate_required")
            plan = _resolved_scene_plan(base_scene_plan, inputs["cells"][selected_cell_index],
                                        task_success_contract=inputs["task_success_contract"])
            controls = execute_native_controls(cell_runtime=bound_runtime, built=current_env["built"],
                                               scene_plan=plan, gate=gate, output_root=output_root,
                                               execution_binding=execution_binding)
            result = {"schema_version": "policy_canary_control_cell_result.v1", "controls": controls,
                      "status": controls["status"], "episodes": [], "candidate_policy_queried": False}
        except Exception as exc:
            result = {"schema_version": "policy_canary_control_cell_result.v1", "status": "blocked",
                      "blockers": [str(exc)], "episodes": [], "candidate_policy_queried": False}
        _seal_result_before_simulation_close(result_path=result_path, result=result,
                                            simulation_app=current_session.get("simulation_app"))
        return 0 if result["status"] == "passed" else 1

    result = execute_paired_session(
        authority=authority,
        runtime_inputs=inputs,
        open_session=open_session,
        load_policy=load_policy,
        run_episode=run_episode,
        close_policy=close_policy,
        close_session=close_session,
        output_path=result_path,
        provider_closeout_pending=True,
        selected_cell_index=selected_cell_index,
        prepolicy_observation_gate=prepolicy_observation_gate,
        prepolicy_control_gate=prepolicy_control_gate,
    )
    result["appearance_render_backend"] = dict(appearance_render_backend)
    if current_session.get("robot_free_probe_trace") is not None:
        result["robot_free_probe_trace"] = current_session["robot_free_probe_trace"]
        result["robot_free_probe_status"] = current_session.get("robot_free_probe_status")
    telemetry_index, telemetry_artifacts = _write_policy_canary_telemetry(
        output_root, result["episodes"]
    )
    result["telemetry"] = telemetry_index
    result["construction_lineage_mode"] = construction_lineage_mode
    if authority.get("execution_release") is not None:
        result["execution_release"] = authority["execution_release"]
    result["artifact_inventory"] = telemetry_artifacts
    result["artifact_inventory_digest"] = _digest(telemetry_artifacts)
    result["matrix_digest"] = inputs.get("matrix_digest") or _digest(inputs["cells"])
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    if family_scene:
        _seal_result(result_path=result_path, result=result)
    else:
        _seal_result_before_simulation_close(
            result_path=result_path,
            result=result,
            simulation_app=current_session.get("simulation_app"),
        )
    return (
        0
        if result["status"]
        == "runtime_selected_cell_completed_pending_aggregation"
        else 1
    )


def main() -> int:
    raw_index = os.environ.get("BLUEPRINT_POLICY_CANARY_CELL_INDEX")
    if raw_index is None:
        return _run_isolated_cell_processes()
    try:
        selected_cell_index = int(raw_index)
    except ValueError as exc:
        raise RuntimeError("policy_canary_cell_index_invalid") from exc
    return _run_selected_cell(selected_cell_index, controls_only=os.environ.get("BLUEPRINT_POLICY_CANARY_CONTROLS_ONLY") == "1")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
