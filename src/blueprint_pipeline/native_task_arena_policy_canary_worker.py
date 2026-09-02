"""Provider worker for one warm, paired internal-policy canary session."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.appearance_render_backend import (
    BACKEND_ISAAC_NATIVE_NUREC,
    BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE,
    BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE,
    AppearanceRenderBackendError,
    build_appearance_render_backend,
)
from blueprint_pipeline.native_task_arena_policy_canary_session import (
    CANDIDATE_IDS,
    PolicyCanaryEpisodeFailure,
    PROVIDER_RESULT_FILENAME,
    execute_paired_session,
    validate_runtime_input_manifest,
    validate_session_authority,
)


ISOLATED_CELL_PROCESS_TIMEOUT_SECONDS = 900


def _sha256_prefixed(value: Any) -> str:
    text = str(value or "")
    return text if text.startswith("sha256:") else f"sha256:{text}"


PACKET_REQUEST_FILENAME = "native_task_arena_packet_request.v1.json"
OFFICIAL_TRANSCODE_IMPLEMENTATION = "nvidia_3dgrut_direct_nurec_transcode"


def appearance_render_backend_from_plan(
    plan: Mapping[str, Any], *, packet_request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Seal the appearance backend the sealed scene plan actually composes.

    Scene 839873's render-only probe named the ParticleField path while this
    worker launched Isaac with no path and inherited a legacy default.  The
    backend is now derived from the plan here, passed explicitly to the
    launcher, and carried by digest through the session receipt so a same-pose
    parity authority can be bound to exactly this renderer and conversion.

    The packet request's ``appearance_variant`` (source Gaussian digest and
    authoring implementation) is the preferred identity source; a plan whose
    ``appearance_frame_alignment`` carries ``source_asset_sha256`` and
    ``conversion_identity`` is accepted when no packet request is present.
    """

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
                kind=BACKEND_ISAAC_NATIVE_NUREC,
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


def isaac_cell_runtime() -> CellRuntime:
    """Bind the production Isaac Lab, native arena, and policy client seams.

    Imports stay lazy so the control plane, the bundle builder, and the
    hermetic rehearsal never load Isaac or torch; the provider worker resolves
    them once per isolated cell process.
    """

    from blueprint_pipeline import adp009d_policy_episode as policy_episode_module
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
        _policy_client,
        _runtime_groot_worker_identity,
        _to_tensor,
    )
    from blueprint_pipeline.native_task_arena_preconstruction import (
        prepare_native_task_arena_preconstruction,
    )
    from blueprint_pipeline.native_task_arena_readback import (
        NativeArticulatedTaskArenaReadback,
    )
    from blueprint_pipeline.native_task_arena_runtime import (
        build_native_task_arena_environment,
    )
    from blueprint_pipeline.native_task_nurec_render_setup import (
        prepare_site_appearance_renderer,
    )
    from blueprint_pipeline.native_task_episode_environment import (
        build_native_task_episode_environment,
    )
    from blueprint_pipeline.native_task_isaaclab_launch import (
        NATIVE_TASK_ARENA_DEVICE,
        launch_native_task_isaaclab,
    )

    def gripper_probe(*, env: Any, robot: Any, seed: int) -> Mapping[str, Any]:
        # torch is resolved only once Isaac has launched and an environment
        # exists, exactly where the pre-seam worker first imported it.
        import torch

        return _gripper_convention_probe(env=env, robot=robot, seed=seed, torch=torch)

    def run_policy_episode(**kwargs: Any) -> Mapping[str, Any]:
        """Use a native diagnostic camera receipt only for this signed canary."""

        original = policy_episode_module.prepolicy_visual_readiness_evidence

        def evidence(**evidence_kwargs: Any) -> Mapping[str, Any]:
            binding = dict(evidence_kwargs.get("observation_integrity") or {})
            gate = binding.get("runtime_gate")
            gate = dict(gate) if isinstance(gate, Mapping) else None
            valid = bool(
                gate
                and gate.get("schema_version")
                == "policy_canary_runtime_observation_integrity_gate.v1"
                and gate.get("status") == "passed"
                and gate.get("run_kind") == "internal_policy_canary"
                and gate.get("claim_ceiling") == "diagnostic_policy_execution"
                and gate.get("frame_structure_passed") is True
                and gate.get("target_semantic_visibility_passed") is True
                and gate.get("candidate_policy_loaded") is False
                and gate.get("candidate_policy_queried") is False
                and gate.get("official_ranking_permitted") is False
                and gate.get("scene_promotion_permitted") is False
                and gate.get("blockers") == []
                and gate.get("policy_observation_integrity_passed") is True
                and gate.get("appearance_render_backend_receipt_digest")
                == binding.get("appearance_render_backend_receipt_digest")
                and gate.get("gate_digest")
                == canonical_digest(gate, digest_field="gate_digest")
            )
            if not valid:
                return original(**evidence_kwargs)
            from blueprint_pipeline.native_task_camera_observability import (
                measure_native_task_prepolicy_visual_frames,
            )

            receipt = measure_native_task_prepolicy_visual_frames(
                evidence_kwargs["camera_rgb"],
                candidate_policy_loaded=evidence_kwargs["candidate_policy_loaded"],
                candidate_policy_queried=evidence_kwargs.get(
                    "candidate_policy_queried", False
                ),
            )
            if receipt.get("frame_structure_passed") is not True:
                return original(**evidence_kwargs)
            receipt.update(
                policy_observation_integrity_passed=True,
                policy_observation_integrity_blockers=[],
                target_semantic_visibility_passed=True,
                appearance_reference_parity_passed=False,
                human_visual_review_status=(
                    "not_required_for_internal_diagnostic_policy_execution"
                ),
                runtime_observation_gate={
                    "gate_digest": gate["gate_digest"],
                    "wrist_camera_mount_selection_digest": gate.get(
                        "wrist_camera_mount_selection_digest"
                    ),
                },
                quality_boundary=(
                    "signed development-only internal canary: native camera sweep, "
                    "semantic task visibility, and exact reset-frame structure; "
                    "no ranking, qualification, parity, or scene promotion"
                ),
            )
            return receipt

        policy_episode_module.prepolicy_visual_readiness_evidence = evidence
        try:
            return policy_episode_module.run_policy_episode(**kwargs)
        finally:
            policy_episode_module.prepolicy_visual_readiness_evidence = original

    def prepolicy_camera_gate(
        *,
        simulation_app: Any,
        built: Any,
        packet_request: Mapping[str, Any],
        plan: Mapping[str, Any],
        output_root: Path,
    ) -> Mapping[str, Any]:
        """Aim bounded robot-preset mounts through Isaac and retain every frame."""

        import hashlib

        from PIL import Image
        import torch

        from blueprint_pipeline.native_task_arena_construction_worker import (
            _body_pose_world,
            _camera_snapshot,
        )
        from blueprint_pipeline.native_task_camera_observability import (
            measure_native_task_prepolicy_visual_frames,
        )

        registry = dict(packet_request.get("wrist_camera_mount_registry") or {})
        candidates = registry.get("candidates")
        if (
            registry.get("schema_version")
            != "policy_canary_wrist_camera_mount_registry.v1"
            or registry.get("selection_authority") != "native_render_measurements"
            or not isinstance(candidates, list)
            or not 2 <= len(candidates) <= 12
            or registry.get("registry_digest")
            != canonical_digest(registry, digest_field="registry_digest")
        ):
            raise RuntimeError("policy_canary_wrist_camera_mount_registry_invalid")
        wrist = [
            row
            for row in packet_request.get("cameras") or []
            if isinstance(row, Mapping) and row.get("role") == "wrist"
        ]
        if len(wrist) != 1:
            raise RuntimeError("policy_canary_wrist_camera_mount_role_invalid")
        env = built.env
        seed = int(plan["scenario"]["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        body_name = str(wrist[0].get("parent_prim_path") or "").rsplit("/", 1)[-1]
        body = [
            float(value)
            for value in _body_pose_world(robot, body_name=body_name, torch=torch)[:3]
        ]
        target = [float(value) for value in plan["task_spec"]["start_pose_world"][:3]]
        direction = [target[index] - body[index] for index in range(3)]
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1.0e-9:
            raise RuntimeError("policy_canary_wrist_camera_task_direction_invalid")
        forward = [value / norm for value in direction]
        right_raw = [forward[1], -forward[0], 0.0]
        right_norm = math.sqrt(sum(value * value for value in right_raw))
        if right_norm <= 1.0e-9:
            raise RuntimeError("policy_canary_wrist_camera_task_direction_invalid")
        right = [value / right_norm for value in right_raw]
        camera = env.unwrapped.scene[built.camera_scene_names["wrist"]]
        root = output_root / "prepolicy_observation_gate"
        root.mkdir(parents=True, exist_ok=True)
        observations: list[dict[str, Any]] = []
        frames: list[Path] = []

        def apply(row: Mapping[str, Any]) -> tuple[list[float], list[float]]:
            eye = [
                body[index]
                + float(row["forward_offset_m"]) * forward[index]
                + float(row["lateral_offset_m"]) * right[index]
                + float(row["vertical_offset_m"]) * (1.0 if index == 2 else 0.0)
                for index in range(3)
            ]
            camera.set_world_poses_from_view(eyes=[eye], targets=[target])
            for _ in range(6):
                simulation_app.update()
            camera.update(0.0, force_recompute=True)
            return eye, target

        for row in candidates:
            if row.get("candidate_digest") != canonical_digest(
                row, digest_field="candidate_digest"
            ):
                raise RuntimeError("policy_canary_wrist_camera_candidate_invalid")
            eye, aimed_target = apply(row)
            candidate_root = root / "wrist_mount_sweep" / str(row["candidate_id"])
            snapshot = _camera_snapshot(
                env=env,
                camera_scene_names={"wrist": built.camera_scene_names["wrist"]},
                output_root=candidate_root,
                snapshot_id="candidate",
            )
            measured = snapshot["cameras"][0]
            frame = candidate_root / measured["rgb_png"]["path"]
            frames.append(frame)
            task = measured["semantic_label_pixels"]["task_object"]
            robot_pixels = measured["semantic_label_pixels"]["robot"]
            admitted = bool(
                int(task["pixel_count"]) >= 120
                and float(task["pixel_fraction"]) >= 0.002
                and float(robot_pixels["pixel_fraction"]) <= 0.30
                and measured["observability"]["render_passed"] is True
            )
            observations.append(
                {
                    "candidate_id": row["candidate_id"],
                    "candidate_digest": row["candidate_digest"],
                    "eye_position_world_m": eye,
                    "target_position_world_m": aimed_target,
                    "task_object": task,
                    "robot": robot_pixels,
                    "frame_png": measured["rgb_png"],
                    "admitted": admitted,
                }
            )
        admitted = [row for row in observations if row["admitted"]]
        selected = (
            sorted(
                admitted,
                key=lambda row: (
                    -int(row["task_object"]["pixel_count"]),
                    float(row["robot"]["pixel_fraction"]),
                    str(row["candidate_id"]),
                ),
            )[0]
            if admitted
            else None
        )
        opened = [Image.open(path).convert("RGB") for path in frames]
        width = max(image.width for image in opened)
        height = max(image.height for image in opened)
        sheet = Image.new(
            "RGB", (3 * width, math.ceil(len(opened) / 3) * height), color=(0, 0, 0)
        )
        for index, image in enumerate(opened):
            sheet.paste(image, ((index % 3) * width, (index // 3) * height))
            image.close()
        contact = root / "wrist_mount_sweep" / "contact_sheet.png"
        contact.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(contact, format="PNG", compress_level=9)
        sheet.close()
        if selected is not None:
            camera.set_world_poses_from_view(
                eyes=[selected["eye_position_world_m"]],
                targets=[selected["target_position_world_m"]],
            )
            for _ in range(6):
                simulation_app.update()
            camera.update(0.0, force_recompute=True)
        snapshot = _camera_snapshot(
            env=env,
            camera_scene_names=built.camera_scene_names,
            output_root=root,
            snapshot_id="selected_mount_reset",
            framing_expectations=(plan.get("task_object_observability") or {}).get(
                "cameras"
            ),
        )
        frame_arrays = {}
        for row in snapshot["cameras"]:
            with Image.open(root / row["rgb_png"]["path"]) as image:
                import numpy as np

                frame_arrays[str(row["role"])] = np.asarray(image.convert("RGB"))
        visual = measure_native_task_prepolicy_visual_frames(
            frame_arrays, candidate_policy_loaded=False
        )
        visibility = {
            str(row["role"]): bool(row["observability"]["passed"])
            for row in snapshot["cameras"]
        }
        blockers = []
        if selected is None:
            blockers.append("policy_canary_wrist_camera_no_admissible_candidate")
        blockers.extend(visual.get("blockers") or [])
        if set(visibility) != {"external", "wrist", "overview"} or not all(
            visibility.values()
        ):
            blockers.append("policy_canary_task_semantic_visibility_failed")
        receipt: dict[str, Any] = {
            "schema_version": "policy_canary_runtime_observation_integrity_gate.v1",
            "status": "passed" if not blockers else "blocked",
            "run_kind": "internal_policy_canary",
            "claim_ceiling": "diagnostic_policy_execution",
            "appearance_render_backend_receipt_digest": (
                appearance_render_backend_from_plan(
                    plan, packet_request=packet_request
                )["receipt_digest"]
            ),
            "wrist_camera_mount_selection_digest": canonical_digest(
                {"registry_digest": registry["registry_digest"], "observations": observations}
            ),
            "frame_structure_passed": visual["frame_structure_passed"],
            "target_semantic_visibility_passed": bool(visibility)
            and all(visibility.values()),
            "candidate_policy_loaded": False,
            "candidate_policy_queried": False,
            "official_ranking_permitted": False,
            "scene_promotion_permitted": False,
            "selected_wrist_camera_mount": selected,
            "camera_visibility": visibility,
            "contact_sheet": {
                "path": str(contact.relative_to(root)),
                "sha256": "sha256:" + hashlib.sha256(contact.read_bytes()).hexdigest(),
            },
            "snapshot": snapshot,
            "human_visual_review_status": (
                "not_required_for_internal_diagnostic_policy_execution"
            ),
            "blockers": sorted(set(blockers)),
            "policy_observation_integrity_passed": not blockers,
            "gate_digest": "",
        }
        receipt["gate_digest"] = canonical_digest(receipt, digest_field="gate_digest")
        (root / "policy_canary_runtime_observation_integrity_gate.v1.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        policy_client=_policy_client,
        groot_worker_identity=_runtime_groot_worker_identity,
        run_policy_episode=run_policy_episode,
        prepolicy_camera_gate=prepolicy_camera_gate,
    )


def _construction_lineage_mode(
    *,
    inputs: Mapping[str, Any],
    base_scene_plan: Mapping[str, Any],
    construction: Mapping[str, Any],
) -> str:
    """Accept strict construction or typed compiled-scene diagnostic lineage."""

    if (
        base_scene_plan.get("plan_digest")
        != canonical_digest(base_scene_plan, digest_field="plan_digest")
    ):
        raise RuntimeError("policy_canary_scene_plan_invalid")
    if construction.get("schema_version") == "native_task_arena_construction_result.v1":
        if (
            construction.get("status") != "completed"
            or construction.get("construction_gate_qualified") is not True
            or construction.get("scene_plan_digest")
            != base_scene_plan.get("plan_digest")
            or construction.get("result_digest")
            != canonical_digest(construction, digest_field="result_digest")
        ):
            raise RuntimeError("policy_canary_construction_result_invalid")
        return "qualified_native_construction_result"
    if construction.get("schema_version") == "task_evaluation_episode_compilation_result.v1":
        if (
            construction.get("status") != "compiled_for_production_launch"
            or construction.get("blockers") != []
            or construction.get("configured_scene_revision_digest")
            != inputs.get("scene_revision_digest")
            or construction.get("provider_mutation_performed") is not False
            or construction.get("paid_execution_requested") is not False
            or construction.get("result_digest")
            != canonical_digest(construction, digest_field="result_digest")
        ):
            raise RuntimeError("policy_canary_compiled_scene_lineage_invalid")
        return "compiled_configured_scene_diagnostic"
    raise RuntimeError("policy_canary_construction_lineage_schema_invalid")


def _yaw_quaternion_xyzw(degrees: float) -> list[float]:
    half = math.radians(degrees) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _quaternion_product_xyzw(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _validate_provider_manifest(
    value: Mapping[str, Any],
    *,
    runtime_inputs: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = json.loads(json.dumps(dict(value), allow_nan=False))
    if (
        manifest.get("schema_version")
        != "native_task_arena_policy_canary_provider_bundle.v1"
        or manifest.get("execution_mode")
        != "internal_policy_canary_paired_session"
        or manifest.get("run_kind") != "internal_policy_canary"
        or manifest.get("claim_ceiling") != "diagnostic_policy_execution"
        or manifest.get("runtime_inputs_digest")
        != runtime_inputs.get("runtime_inputs_digest")
        or manifest.get("authority_digest") != authority.get("authority_digest")
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        raise RuntimeError("policy_canary_provider_manifest_invalid")
    return manifest


def _resolved_scene_plan(base: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    plan = deepcopy(dict(base))
    scenario = deepcopy(dict(cell["resolved_scenario"]))
    scenario["cell_id"] = cell["cell_id"]
    scenario["seed"] = cell["seed"]
    parameters = dict(scenario.get("parameters") or {})
    applications: list[dict[str, Any]] = []
    coverage_gaps: list[dict[str, Any]] = []
    subject = next(row for row in plan["objects"] if row.get("task_subject") is True)
    if "object_start_y_delta_m" in parameters:
        delta = float(parameters["object_start_y_delta_m"])
        subject["pose_world"]["position_world_m"][1] += delta
        subject["reset_state"]["root_pose_world"]["position_world_m"][1] += delta
        applications.append(
            {
                "parameter_id": "object_start_y_delta_m",
                "readback_kind": "task_subject_root_position_y_m",
                "expected_native_value": subject["pose_world"]["position_world_m"][1],
                "delta_from_nominal": delta,
            }
        )
    if "object_yaw_delta_degrees" in parameters:
        delta = float(parameters["object_yaw_delta_degrees"])
        orientation = _quaternion_product_xyzw(
            _yaw_quaternion_xyzw(delta),
            list(subject["pose_world"]["orientation_xyzw"]),
        )
        subject["pose_world"]["orientation_xyzw"] = orientation
        subject["reset_state"]["root_pose_world"]["orientation_xyzw"] = list(
            orientation
        )
        applications.append(
            {
                "parameter_id": "object_yaw_delta_degrees",
                "readback_kind": "task_subject_root_orientation_xyzw",
                "expected_native_value": orientation,
                "delta_from_nominal": delta,
            }
        )
    if "external_camera_x_delta_m" in parameters:
        delta = float(parameters["external_camera_x_delta_m"])
        camera = next(row for row in plan["cameras"] if row["role"] == "external")
        camera["frame_from_camera_matrix"][3] += delta
        applications.append(
            {
                "parameter_id": "external_camera_x_delta_m",
                "readback_kind": "camera_offset_position_x_m",
                "camera_role": "external",
                "expected_native_value": camera["frame_from_camera_matrix"][3],
                "delta_from_nominal": delta,
            }
        )
    if "task_light_intensity_scale" in parameters:
        scale = float(parameters["task_light_intensity_scale"])
        applications.append(
            {
                "parameter_id": "task_light_intensity_scale",
                "readback_kind": "task_light_intensity_scale",
                "expected_native_value": scale,
                "nominal_native_intensity": 1500.0,
                "application_tolerance": 1.0e-6,
            }
        )
    if "dynamic_friction" in parameters:
        coverage_gaps.append(
            {
                "family": "bounded_physics",
                "reason": "runtime_material_link_binding_unavailable",
                "fallback": "canonical_task_material",
            }
        )
    if "material_cousin" in parameters:
        coverage_gaps.append(
            {
                "family": "admitted_object_material_cousin",
                "reason": "admitted_runtime_material_asset_unavailable",
                "fallback": "canonical_task_material",
            }
        )
    scenario["parameter_applications"] = applications
    scenario["runtime_coverage_gaps"] = coverage_gaps
    plan["scenario"] = scenario
    # Both frozen DROID adapters emit actions at 15 Hz. Keep PhysX at 120 Hz
    # and change the exact canary cadence to an integral decimation of eight.
    plan["cadence"]["control_frequency_hz"] = 15.0
    plan["cadence"]["control_decimation"] = 8
    action_steps = int(plan["cadence"]["maximum_action_steps"])
    settle_samples = int(plan["cadence"]["settle_window_samples"])
    plan["task_spec"]["control_frequency_hz"] = 15.0
    plan["task_spec"]["maximum_episode_seconds"] = action_steps / 15.0
    plan["cadence"]["episode_length_seconds"] = (
        action_steps / 15.0
        + settle_samples / 15.0
        + 6.0 * float(plan["cadence"]["physics_dt_seconds"])
    )
    plan["canary_cadence_adjustment"] = {
        "source_control_frequency_hz": float(base["cadence"]["control_frequency_hz"]),
        "resolved_control_frequency_hz": 15.0,
        "physics_frequency_hz": float(plan["cadence"]["physics_frequency_hz"]),
        "control_decimation": 8,
        "reason": "frozen_droid_policy_action_cadence",
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


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


def _seal_result_before_simulation_close(
    *, result_path: Path, result: Mapping[str, Any], simulation_app: Any
) -> None:
    """Seal the provider result before Isaac can terminate the interpreter."""

    close = getattr(simulation_app, "close", None)
    if not callable(close):
        raise RuntimeError("policy_canary_simulation_close_unavailable")
    _seal_result(result_path=result_path, result=result)
    close()


def _write_episode_failure_gap(
    *,
    output_root: Path,
    run_id: str,
    context: Mapping[str, Any],
    failure: Exception,
    progress: Mapping[str, Any] | None = None,
) -> Path:
    """Retain the strongest episode evidence reached before a typed failure."""

    raw_message = str(failure).strip().replace("\n", " ").replace("\r", " ")
    safe_message = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s/:]+/)*[^\s:]+", "<path>", raw_message)
    safe_message = safe_message[:512]
    episode_id = (
        f"{run_id}--{context.get('cell_id')}--{context.get('candidate_id')}"
    )
    progress = progress if isinstance(progress, Mapping) else {}
    first_observation_retained = progress.get("first_observation_retained") is True
    candidate_policy_queried = progress.get("candidate_policy_queried") is True
    candidate_action_returned = progress.get("candidate_action_returned") is True
    action_applied = progress.get("candidate_action_applied") is True
    violations = [
        str(item)
        for item in getattr(failure, "errors", ())
        if str(item).startswith("candidate_action_joint_position_bounds_invalid")
    ]
    action_rejected = bool(
        candidate_policy_queried
        and candidate_action_returned
        and not action_applied
        and progress.get("phase") == "policy_action_bounds_refused"
        and violations
    )
    failure_stage = (
        "action_delivery_rejected"
        if action_rejected
        else "after_first_observation"
        if first_observation_retained
        else "before_first_observation"
    )
    suffix = "failure_evidence" if first_observation_retained else "failure_gap"
    path = output_root / "episodes" / f"{episode_id}.{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    visual_evidence: dict[str, Any]
    media_artifacts: list[dict[str, Any]] = []
    if first_observation_retained:
        finalizer = progress.get("_failure_media_finalizer")
        if callable(finalizer):
            try:
                visual, artifacts = finalizer(
                    failure_reason=f"{type(failure).__name__}:{safe_message}"
                )
                visual_evidence = dict(visual)
                media_artifacts = [
                    dict(item) for item in artifacts if isinstance(item, Mapping)
                ]
            except Exception as exc:  # noqa: BLE001 - preserve the primary failure
                visual_evidence = {
                    "status": "incomplete_after_first_observation",
                    "media_gap": {
                        "type": "after_first_observation_media_seal_failed",
                        "reason": type(exc).__name__,
                    },
                }
        else:
            visual_evidence = {
                "status": "incomplete_after_first_observation",
                "media_gap": {
                    "type": "after_first_observation_media_finalizer_missing",
                    "reason": "policy_canary_episode_runner_failed",
                },
            }
    else:
        visual_evidence = {
            "status": "unavailable_before_first_observation",
            "media_gap": {
                "type": "before_first_observation",
                "reason": "policy_canary_episode_runner_failed",
            },
        }
    raw_queries = [
        dict(item)
        for item in progress.get("candidate_policy_action_queries") or []
        if isinstance(item, Mapping)
    ]
    commanded_actions = [
        dict(item)
        for item in progress.get("commanded_actions") or []
        if isinstance(item, Mapping)
    ]
    action_rejection = None
    if action_rejected:
        action_rejection = {
            "schema_version": "policy_canary_action_delivery_rejection.v1",
            "status": "rejected_before_robot",
            "reason": "hard_joint_limit_violation",
            "violations": violations,
            "clamping_performed": False,
            "delivery_attempted": False,
            "actions_reached_robot": False,
            "rejection_digest": "",
        }
        action_rejection["rejection_digest"] = canonical_digest(
            action_rejection, digest_field="rejection_digest"
        )
    evidence_artifacts: dict[str, Any] = {}
    media_root = output_root / "episodes"
    if media_artifacts:
        evidence_artifacts["frame_manifest"] = _bound_media_artifact(
            output_root,
            media_root=media_root,
            artifacts=media_artifacts,
            role="lossless_frame_manifest",
            role_match=lambda name: "frame_manifest" in name,
        )
        evidence_artifacts["review_video"] = _bound_media_artifact(
            output_root,
            media_root=media_root,
            artifacts=media_artifacts,
            role="review_video",
            role_match=lambda name: "video" in name,
        )
    if candidate_policy_queried:
        evidence_artifacts["policy_query_receipt"] = _write_episode_json_artifact(
            output_root,
            episode_id=episode_id,
            role="policy_query_receipt",
            value={
                "candidate_policy_queried": True,
                "candidate_action_returned": candidate_action_returned,
                "policy_queries": raw_queries,
            },
        )
    if candidate_action_returned:
        evidence_artifacts["action_sequence"] = _write_episode_json_artifact(
            output_root,
            episode_id=episode_id,
            role="action_sequence",
            value=raw_queries,
        )
    if action_rejection is not None:
        evidence_artifacts["action_delivery_readback"] = _write_episode_json_artifact(
            output_root,
            episode_id=episode_id,
            role="action_delivery_readback",
            value=action_rejection,
        )
    value = {
        "schema_version": "policy_canary_episode_failure_evidence.v2",
        "status": "blocked",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_id": context.get("candidate_id"),
        "cell_id": context.get("cell_id"),
        "seed": context.get("seed"),
        "episode_failure_stage": failure_stage,
        "first_observation_retained": first_observation_retained,
        "reset_state_digest": canonical_digest(
            {
                "resolved_scenario": context.get("resolved_scenario"),
                "seed": context.get("seed"),
                "execution_performed": first_observation_retained,
            }
        ),
        "candidate_policy_queried": candidate_policy_queried,
        "candidate_action_returned": candidate_action_returned,
        "candidate_action_shape_validated": (
            progress.get("candidate_action_shape_validated") is True
        ),
        "candidate_action_finite_validated": (
            progress.get("candidate_action_finite_validated") is True
        ),
        "candidate_action_bounds_validated": (
            progress.get("candidate_action_bounds_validated") is True
        ),
        "actions_reached_robot": action_applied,
        "arm_moved": False,
        "policy_outcome_interpretable": False,
        "failure_type": type(failure).__name__,
        "typed_harness_failure": type(failure).__name__,
        "failure_message": safe_message or None,
        "failure_message_digest": _digest(raw_message),
        "candidate_policy_action_queries": raw_queries,
        "commanded_actions": commanded_actions,
        "action_delivery_rejection": action_rejection,
        "visual_evidence": visual_evidence,
        "lossless_frame_manifest_digest": (
            _digest(visual_evidence) if first_observation_retained else None
        ),
        "review_video_digest": (
            _digest(media_artifacts) if media_artifacts else None
        ),
        "returned_action_sequence_digest": (
            _digest(raw_queries) if candidate_action_returned else None
        ),
        "action_delivery_readback_digest": (
            action_rejection["rejection_digest"]
            if action_rejection is not None
            else None
        ),
        "evidence_artifacts": evidence_artifacts,
        "episode": {
            "episode_id": episode_id,
            "candidate_policy_action_queries": raw_queries,
            "commanded_actions": commanded_actions,
            "visual_evidence": visual_evidence,
            "media_artifacts": media_artifacts,
            "motion_evidence": {
                "actions_reached_robot": action_applied,
                "arm_moved": False,
                "policy_outcome_interpretable": False,
                "action_delivery_rejection": action_rejection,
            },
            "score": {
                "status": "not_scored",
                "blockers": ["policy_outcome_uninterpretable"],
            },
        },
        "gap_digest": "",
    }
    value["gap_digest"] = canonical_digest(value, digest_field="gap_digest")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_episode_json_artifact(
    output_root: Path, *, episode_id: str, role: str, value: Any
) -> dict[str, Any]:
    path = output_root / "episodes" / f"{episode_id}.{role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _bound_media_artifact(
    output_root: Path,
    *,
    media_root: Path,
    artifacts: Any,
    role: str,
    role_match: Callable[[str], bool],
) -> dict[str, Any] | None:
    """Bind one episode media artifact into run-root-relative evidence.

    The episode runner records media rows relative to its ``media_output_dir``
    (the run's ``episodes`` directory), not to the run root.  Resolving them
    against the run root silently returned ``None`` for every frame manifest
    and review video, so paid runs shipped episode evidence without either.
    The hermetic lifecycle rehearsal pins the corrected binding.
    """

    matches = [
        row
        for row in artifacts or []
        if isinstance(row, Mapping) and role_match(str(row.get("role") or ""))
    ]
    if not matches:
        return None
    row = matches[0]
    path = (media_root / str(row.get("relative_path") or "")).resolve()
    try:
        path.relative_to(output_root)
    except ValueError:
        return None
    if path.is_symlink() or not path.is_file():
        return None
    return {
        "role": role,
        "relative_path": path.relative_to(output_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write_indexed_telemetry(
    output_root: Path, episodes: list[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [
        {
            "run_kind": episode.get("run_kind"),
            "candidate_id": episode.get("candidate_id"),
            "cell_id": episode.get("cell_id"),
            "seed": episode.get("seed"),
            "telemetry": episode.get("telemetry"),
        }
        for episode in episodes
    ]
    telemetry_path = output_root / "policy_canary_telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    channels = {
        "observations": sum(bool((row.get("telemetry") or {}).get("channels")) for row in rows),
        "episode_envelopes": len(rows),
    }
    schema = {
        "schema_version": "policy_canary_telemetry_schema.v1",
        "timebase": "unix_ns",
        "channels": {
            "episode_envelopes": "policy_canary_episode_telemetry.v1",
            "observations": "native_policy_observation_manifest_reference.v1",
        },
    }
    schema_path = output_root / "policy_canary_telemetry_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    primary_path = telemetry_path
    primary_format = "typed_jsonl"
    mcap_gap: str | None = None
    try:
        from mcap.writer import Writer

        mcap_path = output_root / "policy_canary_telemetry.mcap"
        with mcap_path.open("wb") as stream:
            writer = Writer(stream)
            writer.start(profile="blueprint-policy-canary", library="blueprint_pipeline")
            schema_id = writer.register_schema(
                name="policy_canary_episode_telemetry.v1",
                encoding="jsonschema",
                data=json.dumps(schema, sort_keys=True).encode("utf-8"),
            )
            channel_id = writer.register_channel(
                topic="/blueprint/policy_canary/episode",
                message_encoding="json",
                schema_id=schema_id,
            )
            for row in rows:
                telemetry = row.get("telemetry") or {}
                timestamp = int(telemetry.get("completed_at_unix_ns") or time.time_ns())
                writer.add_message(
                    channel_id=channel_id,
                    log_time=timestamp,
                    publish_time=timestamp,
                    data=json.dumps(row, sort_keys=True).encode("utf-8"),
                )
            writer.finish()
        primary_path = mcap_path
        primary_format = "mcap"
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        mcap_gap = f"mcap_unavailable:{type(exc).__name__}"
    index = {
        "schema_version": "policy_canary_telemetry_index.v1",
        "format": primary_format,
        "artifact": {
            "path": primary_path.name,
            "size_bytes": primary_path.stat().st_size,
            "sha256": _sha256(primary_path),
        },
        "schema": {
            "path": schema_path.name,
            "size_bytes": schema_path.stat().st_size,
            "sha256": _sha256(schema_path),
        },
        "channel_message_counts": channels,
        "message_count": len(rows),
        "attachments": [],
        "calibration_references": [
            (row.get("telemetry") or {}).get("camera_calibration") for row in rows
        ],
        "mcap_gap": mcap_gap,
        "evidence_gaps": sorted(
            {
                gap
                for row in rows
                for gap in (row.get("telemetry") or {}).get("evidence_gaps", [])
            }
        ),
        "index_digest": "",
    }
    index["index_digest"] = canonical_digest(index, digest_field="index_digest")
    index_path = output_root / "policy_canary_telemetry_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    import mimetypes

    artifacts = []
    seen: set[str] = set()
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == PROVIDER_RESULT_FILENAME:
            continue
        relative = path.relative_to(output_root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        lowered = relative.lower()
        typed_evidence_role = next(
            (
                evidence_role
                for evidence_role in (
                    "reset_state",
                    "policy_query_receipt",
                    "action_sequence",
                    "action_delivery_readback",
                    "state_trace",
                    "contact_force_trace",
                    "task_object_trajectory",
                    "score_receipt",
                )
                if f".{evidence_role}.json" in lowered
            ),
            None,
        )
        role = (
            "indexed_episode_telemetry"
            if path in {primary_path, telemetry_path}
            else "telemetry_schema"
            if path == schema_path
            else "telemetry_index"
            if path == index_path
            else "review_video"
            if path.suffix.lower() in {".mp4", ".mov", ".webm"}
            else "lossless_frame_manifest"
            if "frame" in lowered and "manifest" in lowered
            else typed_evidence_role
            if typed_evidence_role is not None
            else "episode_evidence"
            if "episode" in lowered
            else "runtime_supporting_evidence"
        )
        artifacts.append(
            {
                "role": role,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return index, artifacts


def _prefix_episode_evidence_paths(
    episode: Mapping[str, Any], *, prefix: str
) -> dict[str, Any]:
    value = deepcopy(dict(episode))
    evidence = value.get("evidence_artifacts")
    if isinstance(evidence, Mapping):
        rewritten: dict[str, Any] = {}
        for role, record in evidence.items():
            if isinstance(record, Mapping) and isinstance(
                record.get("relative_path"), str
            ):
                rewritten[role] = {
                    **record,
                    "relative_path": f"{prefix}/{record['relative_path']}",
                }
            else:
                rewritten[role] = record
        value["evidence_artifacts"] = rewritten
    return value


def _aggregate_isolated_cell_results(
    *,
    authority: Mapping[str, Any],
    inputs: Mapping[str, Any],
    child_results: list[Mapping[str, Any]],
    output_root: Path,
    construction_lineage_mode: str,
) -> dict[str, Any]:
    if len(child_results) != len(inputs["cells"]):
        raise RuntimeError("policy_canary_isolated_cell_result_count_invalid")
    episodes: list[dict[str, Any]] = []
    for index, child in enumerate(child_results):
        if (
            child.get("selected_cell_index") != index
            or child.get("status")
            != "runtime_selected_cell_completed_pending_aggregation"
            or not isinstance(child.get("episodes"), list)
            or len(child["episodes"]) != len(CANDIDATE_IDS)
        ):
            raise RuntimeError("policy_canary_isolated_cell_result_invalid")
        prefix = f"cell_runs/{index:02d}"
        episodes.extend(
            _prefix_episode_evidence_paths(row, prefix=prefix)
            for row in child["episodes"]
        )
    expected = {
        (candidate, str(cell["cell_id"]), int(cell["seed"]))
        for candidate in CANDIDATE_IDS
        for cell in inputs["cells"]
    }
    observed = {
        (str(row.get("candidate_id")), str(row.get("cell_id")), int(row.get("seed")))
        for row in episodes
    }
    if observed != expected:
        raise RuntimeError("policy_canary_isolated_cell_pairing_invalid")
    result: dict[str, Any] = {
        "schema_version": "native_task_arena_policy_canary_session_result.v1",
        "status": "runtime_completed_unqualified_pending_closeout",
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "candidate_ids": list(CANDIDATE_IDS),
        "episodes_per_policy": 10,
        "learned_policy_rollout_count": 20,
        "provider_allocations_observed": None,
        "retry_cap": 0,
        "warm_session_open_count": 1,
        "isolated_simulation_process_count": len(child_results),
        "policy_loads": [
            {"candidate_id": candidate, "loaded_once": True}
            for candidate in CANDIDATE_IDS
        ],
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
    telemetry_index, telemetry_artifacts = _write_indexed_telemetry(
        output_root, result["episodes"]
    )
    result["telemetry"] = telemetry_index
    if authority.get("execution_release") is not None:
        result["execution_release"] = authority["execution_release"]
    result["artifact_inventory"] = telemetry_artifacts
    result["artifact_inventory_digest"] = _digest(telemetry_artifacts)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


def _spawn_isolated_cell_process(
    *, index: int, runtime_root: Path, output_root: Path, child_root: Path
) -> int:
    """Run exactly one matrix cell in a fresh interpreter (and a fresh Isaac)."""

    child_log = child_root / "worker_console.log"
    environment = dict(os.environ)
    environment["BLUEPRINT_POLICY_CANARY_CELL_INDEX"] = str(index)
    environment["BLUEPRINT_ADP_ARENA_PARENT_OUTPUT_DIR"] = str(output_root)
    environment["BLUEPRINT_ADP_ARENA_OUTPUT_DIR"] = str(child_root)
    with child_log.open("xb") as stream:
        completed = subprocess.run(
            [sys.executable, str(runtime_root / Path(__file__).name)],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=ISOLATED_CELL_PROCESS_TIMEOUT_SECONDS,
            check=False,
        )
    return int(completed.returncode)


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
    construction = _read(
        runtime / "runtime_inputs" / "native_task_arena_construction_result.v1.json"
    )
    construction_lineage_mode = _construction_lineage_mode(
        inputs=inputs,
        base_scene_plan=base_scene_plan,
        construction=construction,
    )
    child_results: list[Mapping[str, Any]] = []
    for index in range(len(inputs["cells"])):
        child_root = output_root / "cell_runs" / f"{index:02d}"
        child_root.mkdir(parents=True, exist_ok=False)
        exit_code = int(
            spawn(
                index=index,
                runtime_root=runtime,
                output_root=output_root,
                child_root=child_root,
            )
        )
        child_result_path = child_root / PROVIDER_RESULT_FILENAME
        if not child_result_path.is_file():
            raise RuntimeError(
                f"policy_canary_isolated_cell_result_missing:{index}:"
                f"exit_{exit_code}"
            )
        child_results.append(_read(child_result_path))
    result = _aggregate_isolated_cell_results(
        authority=authority,
        inputs=inputs,
        child_results=child_results,
        output_root=output_root,
        construction_lineage_mode=construction_lineage_mode,
    )
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
    bound_runtime = cell_runtime if cell_runtime is not None else isaac_cell_runtime()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / PROVIDER_RESULT_FILENAME
    inputs = validate_runtime_input_manifest(
        _read(runtime / "runtime_inputs" / "policy_canary_runtime_inputs.json")
    )
    authority = validate_session_authority(
        _read(runtime / "runtime_inputs" / "policy_canary_session_authority.json")
    )
    _validate_provider_manifest(
        _read(runtime / "adp_arena_provider_manifest.json"),
        runtime_inputs=inputs,
        authority=authority,
    )
    base_scene_plan = _read(
        runtime / "native_task_packet" / "native_task_arena_scene_plan.v1.json"
    )
    construction = _read(
        runtime
        / "runtime_inputs"
        / "native_task_arena_construction_result.v1.json"
    )
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
        for candidate in CANDIDATE_IDS
    }
    current_env: dict[str, Any] = {}
    current_session: dict[str, Any] = {}

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

    def open_session(_inputs: Mapping[str, Any]) -> dict[str, Any]:
        simulation_app, launch = bound_runtime.launch_isaac(
            provider_output_root / "native_task_runtime_source_provisioning.v1.json",
            device=bound_runtime.device,
            appearance_render_path=appearance_render_backend["launch_render_path"],
        )
        current_session["simulation_app"] = simulation_app
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
        if candidate == "groot_n17_droid":
            groot_identity, runtime_identity = bound_runtime.groot_worker_identity(
                output_root=provider_output_root, spec=spec
            )
        client = bound_runtime.policy_client(
            spec, groot_worker_identity_receipt=groot_identity
        )
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
        from blueprint_pipeline.native_task_arena_policy_worker import (
            _PolicyQueryTracker,
        )

        scene_plan = _resolved_scene_plan(base_scene_plan, context)
        dependencies = bound_runtime.preflight_dependency_matrix(
            robot_id=str(scene_plan["robot"]["robot_id"])
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
        env = built.env
        seed = int(context["seed"])
        env.reset(seed=seed)
        robot = env.unwrapped.scene["robot"]
        gripper = bound_runtime.gripper_probe(env=env, robot=robot, seed=seed)
        if gripper["status"] != "measured":
            raise RuntimeError("policy_canary_gripper_unresolved")
        env.reset(seed=seed)
        servo = bound_runtime.make_servo(
            env=env, robot=robot, gripper_convention=gripper
        )
        task_readback = (
            bound_runtime.make_task_readback(
                built,
                grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
            )
            if scene_plan["task_kind"] == "articulated_open_close"
            else None
        )
        episode_environment, environment_receipt = bound_runtime.build_episode_environment(
            built=built,
            gripper_convention=gripper,
            servo=servo,
            task_readback=task_readback,
            to_tensor=bound_runtime.to_tensor,
        )
        environment_receipt = {
            **dict(environment_receipt),
            "appearance_renderer": dict(current_env["appearance_renderer"]),
        }
        tracker = _PolicyQueryTracker(policy["client"])
        spec = policy["spec"]
        episode_id = f"{authority['run_id']}--{context['cell_id']}--{context['candidate_id']}"
        try:
            episode = bound_runtime.run_policy_episode(
                environment=episode_environment,
                policy=tracker,
                candidate_id=str(context["candidate_id"]),
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
            )
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
            if str(context["candidate_id"]) == CANDIDATE_IDS[-1]:
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
            str(context["candidate_id"]) != "groot_n17_droid"
            or (
                observation_support_rows
                and all(
                    row.get("inside_checkpoint_observed_extrema") is True
                    for row in observation_support_rows
                )
            )
        )
        evidence_artifacts = {
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
                value=environment_receipt,
            ),
            "policy_query_receipt": _write_episode_json_artifact(
                output_root,
                episode_id=episode_id,
                role="policy_query_receipt",
                value={
                    "candidate_policy_queried": tracker.candidate_policy_queried,
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
        }
        return {
            "status": "completed",
            "candidate_policy_queried": tracker.candidate_policy_queried,
            "actions_reached_robot": bool(motion.get("actions_reached_robot")),
            "arm_moved": bool(motion.get("arm_moved")),
            "observation_support_qualified": observation_support_qualified,
            "checkpoint_digest": policy["checkpoint_digest"],
            "runtime_identity_digest": policy["runtime_identity_digest"],
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
            return _run_episode_impl(session, policy, context, episode_progress)
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
        if packet_request.get("wrist_camera_mount_registry") is not None:
            if bound_runtime.prepolicy_camera_gate is None:
                raise RuntimeError("policy_canary_runtime_camera_gate_unavailable")
            cell = inputs["cells"][selected_cell_index]
            scene_plan = _resolved_scene_plan(base_scene_plan, cell)
            dependencies = bound_runtime.preflight_dependency_matrix(
                robot_id=str(scene_plan["robot"]["robot_id"])
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
            gate = dict(
                bound_runtime.prepolicy_camera_gate(
                    simulation_app=current_session["simulation_app"],
                    built=built,
                    packet_request=packet_request,
                    plan=scene_plan,
                    output_root=output_root,
                )
            )
            current_session["policy_observation_runtime_gate"] = gate
            return gate
        return preload_observation_integrity_gate(
            observation_integrity_authority,
            appearance_render_backend=dict(session["appearance_render_backend"]),
            authority_path=authority_path,
        )

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
    )
    result["appearance_render_backend"] = dict(appearance_render_backend)
    telemetry_index, telemetry_artifacts = _write_indexed_telemetry(
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
    return _run_selected_cell(selected_cell_index)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
