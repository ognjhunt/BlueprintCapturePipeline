"""One-policy, one-arm GPU canary for the frozen new-site diagnostic lane."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .common import write_json
from .droid_oscar_closed_loop_adapter import (
    EXTERIOR_VIEW,
    WRIST_VIEW,
    DroidOscarSkeletonTransitionAdapter,
    SkeletonOnlyIntendedMotionArm,
)
from .droid_policy_bridge import DROID_EXTERIOR_VIEW_2, validate_droid_observation
from .franka_can_tray_feasibility import _CAN_INITIAL
from .franka_droid_closed_loop import (
    _EXTERNAL_CAMERA,
    _render_hybrid_external_observation,
    _render_link_mounted_observation,
    prepare_franka_droid_runtime,
)
from .franka_droid_skeleton_conditioning import FrankaDroidSkeletonConditioningBuilder
from .new_site_diagnostic_smoke import assess_canary
from .openpi_droid_policy_runtime import load_policy_spec, verify_local_checkpoint
from .openpi_policy_ranking_gpu_job import (
    LocalOpenPIDroidPolicyClient,
    _default_checkpoint_downloader,
    _default_openpi_loader,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256
from .policy_wam_closed_loop import (
    ClosedLoopConfig,
    policy_observation_sha256,
    run_policy_wam_closed_loop,
)
from .policy_wam_reliability_gate import FrozenMaximumHorizonTerminalCriterion
from .scene_placement.stance_cameras import link_mounted_camera_spec
from .wam_rollout_reliability import (
    FLAG_MOTION_WITHOUT_COMMAND,
    FLAG_STATIC_UNDER_COMMAND,
    FLAG_TIMING_DISAGREEMENT,
    TIMING_SCOPE_SESSION,
    ReliabilityThresholds,
    RolloutReliabilityReport,
    action_energy_series,
    assess_rollout_reliability,
)


SCHEMA_VERSION = "new_site_diagnostic_canary_gpu.v1"
INPUT_SCHEMA_VERSION = "new_site_diagnostic_canary_input.v2"
INPUT_RECEIPT_SCHEMA_VERSION = "new_site_diagnostic_canary_input_receipt.v2"
PROTOCOL_NAME = "protocol.json"
BACKGROUND_NAME = "captured_site_background.png"
MANIFEST_NAME = "bundle_manifest.json"
NATIVE_CAMERA_RESULT_NAME = "native_camera_canary_result.json"
NATIVE_EXTERNAL_NAME = "native_external_initial.png"
NATIVE_EXTERNAL_2_NAME = "native_external_2_initial.png"
NATIVE_WRIST_NAME = "native_wrist_initial.png"
MAX_INPUT_BYTES = 8 * 1024 * 1024
BUNDLE_SUPPORTED_ARMS = frozenset({"skeleton_only", "ctrl_world"})
NATIVE_CAMERA_REQUIREMENTS = {
    "skeleton_only": (
        (EXTERIOR_VIEW, NATIVE_EXTERNAL_NAME, "external"),
        (WRIST_VIEW, NATIVE_WRIST_NAME, "wrist"),
    ),
    "ctrl_world": (
        (EXTERIOR_VIEW, NATIVE_EXTERNAL_NAME, "external"),
        (DROID_EXTERIOR_VIEW_2, NATIVE_EXTERNAL_2_NAME, "external_2"),
        (WRIST_VIEW, NATIVE_WRIST_NAME, "wrist"),
    ),
}
SKELETON_WRIST_REFERENCE_DISPLACEMENT_MIN_M = 0.001
_SKELETON_WRIST_PIXEL_MOTION_FLAGS = frozenset(
    {FLAG_STATIC_UNDER_COMMAND, FLAG_MOTION_WITHOUT_COMMAND, FLAG_TIMING_DISAGREEMENT}
)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"new_site_canary_expected_json_object:{path}")
    return dict(value)


def _validate_protocol_identity(protocol: Mapping[str, Any]) -> None:
    declared = protocol.get("protocol_sha256")
    payload = dict(protocol)
    payload.pop("protocol_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("new_site_canary_protocol_sha256_invalid")


def materialize_canary_background(
    *, source_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Deterministically resize one frozen scene-reference frame for OpenPI."""

    from PIL import Image

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError("new_site_canary_background_source_missing_or_unsafe")
    if output.exists():
        raise FileExistsError("new_site_canary_background_overwrite_forbidden")
    with Image.open(source) as image:
        source_size = image.size
        rgb = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        output.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(output, format="PNG", optimize=False, compress_level=9)
    result: dict[str, Any] = {
        "schema_version": "new_site_diagnostic_canary_background.v1",
        "status": "completed",
        "source_path": str(source),
        "source_sha256": file_sha256(source),
        "source_size_px": list(source_size),
        "output_path": str(output),
        "output_sha256": file_sha256(output),
        "output_size_px": [224, 224],
        "transform": "rgb_lanczos_resize_224x224",
        "crop_applied": False,
        "camera_selection_changed": False,
    }
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def build_canary_input_bundle(
    *,
    protocol_path: str | Path,
    background_path: str | Path,
    output_zip: str | Path,
    arm_id: str,
    native_camera_canary_result_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a private, protocol-bound single-arm input bundle."""

    protocol_file = Path(protocol_path).expanduser().resolve()
    background = Path(background_path).expanduser().resolve()
    destination = Path(output_zip).expanduser().resolve()
    protocol = _read_object(protocol_file)
    if not background.is_file() or background.is_symlink():
        raise FileNotFoundError("new_site_canary_background_missing_or_unsafe")
    from PIL import Image

    with Image.open(background) as image:
        if image.size != (224, 224):
            raise ValueError("new_site_canary_background_must_be_224_square")
    if arm_id not in BUNDLE_SUPPORTED_ARMS:
        raise ValueError("new_site_canary_arm_not_supported_by_input_bundle")
    if arm_id == "ctrl_world" and native_camera_canary_result_path is None:
        raise ValueError("new_site_ctrl_world_native_three_view_result_required")
    if protocol.get("schema_version") != "policy_ranking_new_site_diagnostic_smoke_protocol.v1":
        raise ValueError("new_site_canary_protocol_schema_invalid")
    _validate_protocol_identity(protocol)
    if protocol.get("diagnostic_namespace") != protocol.get("experiment_id"):
        raise ValueError("new_site_canary_diagnostic_namespace_invalid")
    if protocol.get("paid_execution_admitted") is not False:
        raise ValueError("new_site_canary_protocol_must_be_prospective")
    canary_rule = protocol.get("canary_rule")
    canary_rule = canary_rule if isinstance(canary_rule, Mapping) else {}
    scene = protocol.get("scene")
    scene = scene if isinstance(scene, Mapping) else {}
    manifest: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "experiment_id": protocol.get("experiment_id"),
        "protocol_filename": PROTOCOL_NAME,
        "protocol_file_sha256": file_sha256(protocol_file),
        "protocol_sha256": protocol.get("protocol_sha256"),
        "arm_id": arm_id,
        "scene_id": scene.get("scene_id"),
        "task_instruction": scene.get("task_instruction"),
        "policy_id": canary_rule.get("frozen_policy_id"),
        "variant": canary_rule.get("frozen_variant"),
        "background_filename": BACKGROUND_NAME,
        "background_sha256": file_sha256(background),
        "background_size_bytes": background.stat().st_size,
        "raw_3dgs_included": False,
        "redistribution_authorized": False,
        "label_free": True,
        "purpose": "private_internal_noncommercial_new_site_diagnostic_canary",
        "initial_observation_source": "mujoco_hybrid_camera_render",
    }
    native_camera_material: dict[str, Any] | None = None
    if native_camera_canary_result_path is not None:
        native_result_path = Path(native_camera_canary_result_path).expanduser().resolve()
        native_result = _read_object(native_result_path)
        declared_result_sha256 = native_result.pop("result_sha256", None)
        if declared_result_sha256 != canonical_sha256(native_result):
            raise ValueError("new_site_canary_native_camera_result_sha256_invalid")
        native_result["result_sha256"] = declared_result_sha256
        if (
            native_result.get("status") != "passed"
            or native_result.get("label_free") is not True
            or native_result.get("rankings_or_policy_outcomes_accessed") is not False
        ):
            raise ValueError("new_site_canary_native_camera_result_not_admissible")
        assessment = native_result.get("assessment")
        assessment = assessment if isinstance(assessment, Mapping) else {}
        views = assessment.get("views")
        views = views if isinstance(views, Mapping) else {}
        native_camera_material = {
            "result_path": native_result_path,
            "result": native_result,
            "frames": {},
        }
        camera_requirements = NATIVE_CAMERA_REQUIREMENTS[arm_id]
        if arm_id == "ctrl_world" and assessment.get("required_views") != [
            "external",
            "external_2",
            "wrist",
        ]:
            raise ValueError("new_site_ctrl_world_native_three_view_assessment_required")
        for view_id, filename, view_key in camera_requirements:
            view = views.get(view_key)
            view = view if isinstance(view, Mapping) else {}
            frames = view.get("frames")
            frames = frames if isinstance(frames, Mapping) else {}
            frame = frames.get("initial")
            frame = frame if isinstance(frame, Mapping) else {}
            path = Path(str(frame.get("path") or "")).expanduser().resolve()
            if not path.is_file():
                relative_value = str(frame.get("relative_path") or "")
                relative = PurePosixPath(relative_value)
                if relative_value and not relative.is_absolute() and ".." not in relative.parts:
                    path = (native_result_path.parent / relative.as_posix()).resolve()
            if (
                not path.is_file()
                or path.is_symlink()
                or frame.get("sha256") != file_sha256(path)
                or frame.get("resolution") != [640, 480]
                or frame.get("nonblank") is not True
            ):
                raise ValueError(f"new_site_canary_native_initial_frame_invalid:{view_key}")
            native_camera_material["frames"][view_id] = (filename, path)
        manifest.update(
            {
                "initial_observation_source": "native_isaac_simready_warehouse_camera_canary",
                "native_camera_canary_result_filename": NATIVE_CAMERA_RESULT_NAME,
                "native_camera_canary_result_sha256": file_sha256(native_result_path),
                "native_camera_canary_result_identity_sha256": declared_result_sha256,
                "native_initial_cameras": {
                    view_id: {
                        "filename": filename,
                        "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size,
                        "resolution": [640, 480],
                    }
                    for view_id, (filename, path) in native_camera_material["frames"].items()
                },
            }
        )
    if manifest["variant"] != "center":
        raise ValueError("new_site_canary_variant_not_center")
    if not str(manifest["policy_id"] or ""):
        raise ValueError("new_site_canary_policy_missing")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.write(protocol_file, PROTOCOL_NAME)
        archive.write(background, BACKGROUND_NAME)
        if native_camera_material is not None:
            archive.write(native_camera_material["result_path"], NATIVE_CAMERA_RESULT_NAME)
            for filename, path in native_camera_material["frames"].values():
                archive.write(path, filename)
    if destination.stat().st_size > MAX_INPUT_BYTES:
        destination.unlink(missing_ok=True)
        raise ValueError("new_site_canary_input_bundle_too_large")
    return {
        "schema_version": INPUT_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "bundle_path": str(destination),
        "bundle_sha256": file_sha256(destination),
        "bundle_size_bytes": destination.stat().st_size,
        "manifest": manifest,
    }


def extract_canary_input_bundle(
    *, bundle_path: str | Path, expected_bundle_sha256: str, output_dir: str | Path
) -> dict[str, Any]:
    bundle = Path(bundle_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not bundle.is_file() or bundle.is_symlink() or bundle.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("new_site_canary_input_missing_unsafe_or_too_large")
    if file_sha256(bundle) != expected_bundle_sha256:
        raise ValueError("new_site_canary_input_sha256_mismatch")
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = {info.filename for info in infos}
        base_names = {MANIFEST_NAME, PROTOCOL_NAME, BACKGROUND_NAME}
        native_two_view_names = base_names | {
            NATIVE_CAMERA_RESULT_NAME,
            NATIVE_EXTERNAL_NAME,
            NATIVE_WRIST_NAME,
        }
        native_three_view_names = native_two_view_names | {NATIVE_EXTERNAL_2_NAME}
        if frozenset(names) not in {
            frozenset(base_names),
            frozenset(native_two_view_names),
            frozenset(native_three_view_names),
        }:
            raise ValueError("new_site_canary_input_member_allowlist_mismatch")
        if len(infos) not in {3, 6, 7}:
            raise ValueError("new_site_canary_input_member_count_invalid")
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or info.file_size > MAX_INPUT_BYTES:
                raise ValueError("new_site_canary_input_member_unsafe")
        manifest_value = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        protocol_bytes = archive.read(PROTOCOL_NAME)
        background_bytes = archive.read(BACKGROUND_NAME)
        native_result_bytes = (
            archive.read(NATIVE_CAMERA_RESULT_NAME) if NATIVE_CAMERA_RESULT_NAME in names else None
        )
        native_frame_bytes = {
            filename: archive.read(filename)
            for filename in (NATIVE_EXTERNAL_NAME, NATIVE_EXTERNAL_2_NAME, NATIVE_WRIST_NAME)
            if filename in names
        }
    if not isinstance(manifest_value, Mapping):
        raise ValueError("new_site_canary_input_manifest_not_object")
    manifest = dict(manifest_value)
    declared = manifest.pop("manifest_sha256", None)
    if declared != canonical_sha256(manifest):
        raise ValueError("new_site_canary_input_manifest_sha256_mismatch")
    manifest["manifest_sha256"] = declared
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("new_site_canary_input_schema_invalid")
    arm_id = manifest.get("arm_id")
    if arm_id not in BUNDLE_SUPPORTED_ARMS:
        raise ValueError("new_site_canary_input_arm_unsupported")
    camera_requirements = NATIVE_CAMERA_REQUIREMENTS[str(arm_id)]
    expected_native_names = (
        base_names
        | {NATIVE_CAMERA_RESULT_NAME}
        | {filename for _view_id, filename, _view_key in camera_requirements}
    )
    if native_result_bytes is None:
        if arm_id == "ctrl_world":
            raise ValueError("new_site_ctrl_world_native_three_view_result_required")
        expected_names = base_names
    else:
        expected_names = expected_native_names
    if names != expected_names:
        raise ValueError("new_site_canary_input_arm_camera_members_mismatch")
    if hashlib.sha256(protocol_bytes).hexdigest() != manifest.get("protocol_file_sha256"):
        raise ValueError("new_site_canary_protocol_file_sha256_mismatch")
    if hashlib.sha256(background_bytes).hexdigest() != manifest.get("background_sha256"):
        raise ValueError("new_site_canary_background_sha256_mismatch")
    if len(background_bytes) != manifest.get("background_size_bytes"):
        raise ValueError("new_site_canary_background_size_mismatch")
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / PROTOCOL_NAME
    background_path = output / BACKGROUND_NAME
    protocol_path.write_bytes(protocol_bytes)
    background_path.write_bytes(background_bytes)
    initial_camera_paths: dict[str, str] = {}
    if native_result_bytes is not None:
        if hashlib.sha256(native_result_bytes).hexdigest() != manifest.get(
            "native_camera_canary_result_sha256"
        ):
            raise ValueError("new_site_canary_native_result_file_sha256_mismatch")
        native_manifest = manifest.get("native_initial_cameras")
        if not isinstance(native_manifest, Mapping):
            raise ValueError("new_site_canary_native_camera_manifest_missing")
        for view_id, filename, _view_key in camera_requirements:
            entry = native_manifest.get(view_id)
            entry = entry if isinstance(entry, Mapping) else {}
            data = native_frame_bytes[filename]
            if (
                entry.get("filename") != filename
                or hashlib.sha256(data).hexdigest() != entry.get("sha256")
                or len(data) != entry.get("size_bytes")
            ):
                raise ValueError(f"new_site_canary_native_frame_manifest_mismatch:{view_id}")
            path = output / filename
            path.write_bytes(data)
            initial_camera_paths[view_id] = str(path)
    protocol = _read_object(protocol_path)
    _validate_protocol_identity(protocol)
    if protocol.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise ValueError("new_site_canary_protocol_identity_mismatch")
    return {
        "manifest": manifest,
        "protocol_path": str(protocol_path),
        "background_path": str(background_path),
        "initial_camera_paths": initial_camera_paths,
    }


def _initial_observation(
    *,
    runtime: Mapping[str, Any],
    background_path: str | Path,
    prompt: str,
    initial_camera_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Render the exact frozen initial DROID observation without stepping a task rollout."""

    from PIL import Image

    mujoco = runtime["mujoco"]
    np_module = runtime["np"]
    model = runtime["model"]
    data = runtime["data"]
    ids = runtime["ids"]
    targets = runtime["targets"]
    background_file = Path(background_path).expanduser().resolve()
    with Image.open(background_file) as image:
        background = np_module.asarray(image.convert("RGB"), dtype=np_module.uint8)
    if background.shape != (224, 224, 3):
        raise ValueError("new_site_canary_background_must_be_224_square_rgb")
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = targets["pregrasp"]
    data.qpos[7:9] = 0.04
    data.qpos[9:16] = (*_CAN_INITIAL, 1.0, 0.0, 0.0, 0.0)
    data.ctrl[:7] = targets["pregrasp"]
    data.ctrl[7] = 0.04
    mujoco.mj_forward(model, data)
    for _ in range(int(0.8 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    wrist_spec = link_mounted_camera_spec(
        parent_translation=data.xpos[ids["hand"]],
        parent_rotation_row_major=data.xmat[ids["hand"]],
        mount_translation=(0.0, 0.10, 0.03),
        mount_forward=(0.0, 0.0, 1.0),
        mount_up=(0.0, 1.0, 0.0),
        look_distance_m=0.5,
        fov_deg=82.0,
    )
    if initial_camera_paths:
        if set(initial_camera_paths) != {EXTERIOR_VIEW, WRIST_VIEW}:
            raise ValueError("new_site_canary_native_initial_camera_set_invalid")
        loaded = {}
        for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
            with Image.open(Path(initial_camera_paths[view_id]).expanduser().resolve()) as image:
                loaded[view_id] = np_module.asarray(
                    image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS),
                    dtype=np_module.uint8,
                )
        external, wrist = loaded[EXTERIOR_VIEW], loaded[WRIST_VIEW]
        interaction_pixels = int(np_module.count_nonzero(np_module.std(external, axis=2) > 1.0))
    else:
        renderer = mujoco.Renderer(model, height=224, width=224)
        try:
            external, interaction_pixels = _render_hybrid_external_observation(
                renderer,
                model,
                data,
                _EXTERNAL_CAMERA,
                background,
                mujoco,
                np_module,
            )
            wrist = _render_link_mounted_observation(
                renderer,
                model,
                data,
                wrist_spec,
                camera_id=int(ids["wrist_camera"]),
                camera_mocap_id=int(ids["wrist_camera_mocap"]),
                mujoco=mujoco,
                np=np_module,
            )
        finally:
            renderer.close()
    observation = {
        EXTERIOR_VIEW: external,
        WRIST_VIEW: wrist,
        "observation/joint_position": np_module.asarray(data.qpos[:7], dtype=float).copy(),
        "observation/gripper_position": np_module.asarray(
            [float(1.0 - np_module.clip(data.qpos[7] / 0.04, 0.0, 1.0))], dtype=float
        ),
        "prompt": prompt,
    }
    blockers = validate_droid_observation(observation)
    if blockers:
        raise ValueError(f"new_site_canary_initial_observation_invalid:{blockers[0]}")
    observation["_diagnostic_interaction_pixel_count"] = int(interaction_pixels)
    return observation


@dataclass(frozen=True)
class MultiViewCanaryReliabilityGate:
    """Apply camera-aware reliability checks to both required videos.

    A skeleton rigidly expressed in its own link-mounted wrist camera can be
    image-static while the wrist moves substantially in the robot/world frame.
    For that one outcome-blind control arm, media integrity still comes from
    the wrist video while motion presence comes from the hash-bound FK trace.
    Learned WAM arms continue to require visible wrist-video response.
    """

    thresholds: ReliabilityThresholds
    required_views: tuple[str, ...] = (EXTERIOR_VIEW, WRIST_VIEW)
    assessor: Callable[..., RolloutReliabilityReport] = assess_rollout_reliability
    gate_id: str = "new_site_multiview_tier1_reliability_v1"

    def assess(
        self,
        *,
        previous_observation: Mapping[str, Any],
        prepared_transition: Mapping[str, Any],
        wam_prediction: Mapping[str, Any],
        query_index: int,
        output_dir: Path,
    ) -> dict[str, Any]:
        del previous_observation, query_index, output_dir
        videos = wam_prediction.get("generated_videos_by_view")
        if not isinstance(videos, Mapping) or set(videos) != set(self.required_views):
            raise ValueError("new_site_canary_required_view_videos_mismatch")
        actions = np.asarray(prepared_transition.get("reliability_actions_10d"), dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
            raise ValueError("new_site_canary_reliability_actions_invalid")
        reports: dict[str, Any] = {}
        flags: list[str] = []
        skeleton_only = (
            wam_prediction.get("skeleton_only") is True
            and wam_prediction.get("intended_motion_only") is True
        )
        for view_id in self.required_views:
            video = Path(str(videos[view_id])).expanduser().resolve()
            if not video.is_file() or video.is_symlink():
                raise ValueError(f"new_site_canary_video_missing:{view_id}")
            report = self.assessor(
                video,
                actions,
                self.thresholds,
                timing_flag_scope=TIMING_SCOPE_SESSION,
            )
            report_payload = report.as_dict()
            effective_flags = list(report.flags)
            if skeleton_only and view_id == WRIST_VIEW:
                motion = _assess_skeleton_wrist_reference_motion(
                    prepared_transition=prepared_transition,
                    actions=actions,
                    thresholds=self.thresholds,
                )
                effective_flags = [
                    flag
                    for flag in effective_flags
                    if flag not in _SKELETON_WRIST_PIXEL_MOTION_FLAGS
                ]
                effective_flags.extend(motion["flags"])
                report_payload.update(
                    {
                        "raw_pixel_flags": list(report.flags),
                        "flags": effective_flags,
                        "reliable": not effective_flags,
                        "motion_basis": "hash_verified_reference_frame_fk_trace",
                        "pixel_motion_used_for_media_integrity_only": True,
                        "kinematic_motion": motion,
                    }
                )
            reports[view_id] = report_payload
            flags.extend(f"{view_id}:{flag}" for flag in effective_flags)
        return {
            "status": "passed" if not flags else "failed",
            "abstain": bool(flags),
            "reasons": flags,
            "reports_by_view": reports,
            "required_views": list(self.required_views),
            "thresholds": asdict(self.thresholds),
            "thresholds_sha256": canonical_sha256(asdict(self.thresholds)),
            "timing_flag_scope": TIMING_SCOPE_SESSION,
            "label_free": True,
            "claim_boundary": "multiview technical reliability only; not task success",
        }


def _assess_skeleton_wrist_reference_motion(
    *,
    prepared_transition: Mapping[str, Any],
    actions: np.ndarray,
    thresholds: ReliabilityThresholds,
) -> dict[str, Any]:
    evidence_value = prepared_transition.get("conditioning_builder_evidence")
    if not isinstance(evidence_value, Mapping):
        raise ValueError("new_site_canary_wrist_conditioning_evidence_missing")
    evidence = dict(evidence_value)
    declared_evidence_sha256 = evidence.pop("evidence_sha256", None)
    if not isinstance(
        declared_evidence_sha256, str
    ) or declared_evidence_sha256 != canonical_sha256(evidence):
        raise ValueError("new_site_canary_wrist_conditioning_evidence_sha256_invalid")
    trace_evidence = evidence.get("trace_evidence")
    if not isinstance(trace_evidence, Mapping):
        raise ValueError("new_site_canary_wrist_trace_evidence_missing")
    trace_value = trace_evidence.get(WRIST_VIEW)
    if not isinstance(trace_value, Mapping):
        raise ValueError("new_site_canary_wrist_trace_manifest_missing")
    trace_path = Path(str(trace_value.get("trace_path") or "")).expanduser().resolve()
    if not trace_path.is_file() or trace_path.is_symlink():
        raise ValueError("new_site_canary_wrist_trace_missing_or_unsafe")
    if file_sha256(trace_path) != trace_value.get("trace_sha256"):
        raise ValueError("new_site_canary_wrist_trace_sha256_mismatch")

    rows: list[dict[str, Any]] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError("new_site_canary_wrist_trace_row_invalid")
        row = dict(value)
        declared_frame_sha256 = row.pop("frame_sha256", None)
        if declared_frame_sha256 != canonical_sha256(row):
            raise ValueError("new_site_canary_wrist_trace_frame_sha256_invalid")
        row["frame_sha256"] = declared_frame_sha256
        rows.append(row)
    if not rows or len(rows) != trace_value.get("frame_count"):
        raise ValueError("new_site_canary_wrist_trace_frame_count_mismatch")

    positions: list[np.ndarray] = []
    for frame_index, row in enumerate(rows):
        if (
            row.get("schema_version") != "franka_droid_skeleton_projection_frame.v1"
            or row.get("view_id") != WRIST_VIEW
            or row.get("frame_index") != frame_index
            or row.get("physical_future_observation_used") is not False
        ):
            raise ValueError("new_site_canary_wrist_trace_contract_invalid")
        landmarks = row.get("landmarks")
        if not isinstance(landmarks, list):
            raise ValueError("new_site_canary_wrist_trace_landmarks_invalid")
        centers = [
            landmark
            for landmark in landmarks
            if isinstance(landmark, Mapping)
            and landmark.get("landmark_id") == "wrist_action_center"
        ]
        if len(centers) != 1:
            raise ValueError("new_site_canary_wrist_action_center_invalid")
        position = np.asarray(centers[0].get("reference_position_m"), dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("new_site_canary_wrist_reference_position_invalid")
        positions.append(position)

    displacement = np.linalg.norm(np.asarray(positions) - positions[0], axis=1)
    maximum_displacement_m = float(np.max(displacement))
    energy = action_energy_series(actions)
    commanded_active = float(np.mean(energy)) >= thresholds.command_active_energy_min
    commanded_null = float(np.max(energy)) <= thresholds.command_null_energy_max
    motion_present = maximum_displacement_m > SKELETON_WRIST_REFERENCE_DISPLACEMENT_MIN_M
    motion_flags: list[str] = []
    if commanded_active and not motion_present:
        motion_flags.append("kinematic_static_under_command")
    if commanded_null and motion_present:
        motion_flags.append("kinematic_motion_without_command")
    return {
        "flags": motion_flags,
        "reliable": not motion_flags,
        "maximum_reference_displacement_m": maximum_displacement_m,
        "minimum_reference_displacement_m": SKELETON_WRIST_REFERENCE_DISPLACEMENT_MIN_M,
        "commanded_active": commanded_active,
        "commanded_null": commanded_null,
        "landmark_id": "wrist_action_center",
        "reference_frame": "pinned_mujoco_franka_world",
        "trace_path": str(trace_path),
        "trace_sha256": trace_value["trace_sha256"],
        "frame_count": len(rows),
        "physical_future_observation_used": False,
        "claim_boundary": "commanded FK motion only; not scene response or task success",
    }


def _thresholds(protocol: Mapping[str, Any]) -> ReliabilityThresholds:
    reliability = protocol.get("reliability_freeze")
    reliability = reliability if isinstance(reliability, Mapping) else {}
    values = reliability.get("thresholds")
    values = values if isinstance(values, Mapping) else {}
    expected = set(asdict(ReliabilityThresholds()))
    if set(values) != expected:
        raise ValueError("new_site_canary_frozen_threshold_fields_invalid")
    return ReliabilityThresholds(**{key: values[key] for key in expected})


def run_skeleton_only_canary(
    *,
    protocol_path: str | Path,
    background_path: str | Path,
    cohort_path: str | Path,
    checkpoint_inventory_path: str | Path,
    menagerie_root: str | Path,
    output_dir: str | Path,
    initial_camera_paths: Mapping[str, str | Path] | None = None,
    checkpoint_downloader: Callable[[str], Path] = _default_checkpoint_downloader,
    policy_loader: Callable[[Any, Path], Any] = _default_openpi_loader,
) -> dict[str, Any]:
    """Run one learned-policy query through the skeleton-only WAM plumbing."""

    protocol = _read_object(protocol_path)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rule = protocol.get("canary_rule")
    rule = rule if isinstance(rule, Mapping) else {}
    policy_id = str(rule.get("frozen_policy_id") or "")
    spec = load_policy_spec(cohort_path, policy_id=policy_id)
    checkpoint = checkpoint_downloader(spec.checkpoint_uri)
    local_verification = verify_local_checkpoint(
        spec=spec,
        checkpoint_dir=checkpoint,
        checkpoint_inventory_path=checkpoint_inventory_path,
    )
    policy = policy_loader(spec, checkpoint)
    client = LocalOpenPIDroidPolicyClient(
        spec=spec,
        policy=policy,
        local_verification=local_verification,
    )
    runtime = prepare_franka_droid_runtime(
        menagerie_root=menagerie_root,
        output_dir=output / "runtime",
    )
    prompt = str(protocol["scene"]["task_instruction"])
    observation = _initial_observation(
        runtime=runtime,
        background_path=background_path,
        prompt=prompt,
        initial_camera_paths=initial_camera_paths,
    )
    interaction_pixels = int(observation.pop("_diagnostic_interaction_pixel_count"))
    exact_initial_observation_sha256 = policy_observation_sha256(observation)
    initial_camera_sha256_by_view = (
        {
            view_id: file_sha256(Path(path).expanduser().resolve())
            for view_id, path in initial_camera_paths.items()
        }
        if initial_camera_paths
        else {}
    )
    loop = run_policy_wam_closed_loop(
        initial_observation=observation,
        policy_client=client,
        wam_arm=SkeletonOnlyIntendedMotionArm(),
        transition_adapter=DroidOscarSkeletonTransitionAdapter(
            conditioning_builder=FrankaDroidSkeletonConditioningBuilder(
                runtime=runtime,
                camera_contract=protocol["scene"],
            ),
            action_chunk_rows=spec.action_chunk_rows,
        ),
        reliability_gate=MultiViewCanaryReliabilityGate(_thresholds(protocol)),
        terminal_criterion=FrozenMaximumHorizonTerminalCriterion(),
        config=ClosedLoopConfig(
            task_prompt=prompt,
            executed_prefix_steps=8,
            max_policy_queries=2,
            execution_mode="engineering_smoke",
        ),
        output_dir=output / "loop",
    )
    trace_path = Path(loop["trace_path"])
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    row_mappings = [row if isinstance(row, Mapping) else {} for row in rows]
    reliabilities = []
    reasons: list[str] = []
    if any(not isinstance(row, Mapping) for row in rows):
        reasons.append("policy_wam_trace_row_invalid")
    for row in row_mappings:
        reliability_value = row.get("reliability")
        reliability = reliability_value if isinstance(reliability_value, Mapping) else {}
        reliabilities.append(reliability)
        reasons.extend(str(reason) for reason in reliability.get("reasons") or [])
    first_provenance_value = (
        row_mappings[0].get("next_observation_provenance") if row_mappings else None
    )
    first_provenance = first_provenance_value if isinstance(first_provenance_value, Mapping) else {}
    first_policy_observation_sha256 = (
        row_mappings[0].get("policy_observation_sha256") if row_mappings else None
    )
    first_wam_observation_sha256 = (
        row_mappings[0].get("next_observation_sha256") if row_mappings else None
    )
    second_policy_observation_sha256 = (
        row_mappings[1].get("policy_observation_sha256") if len(row_mappings) >= 2 else None
    )
    round_trip_passed = (
        len(rows) == 2
        and len(row_mappings) == 2
        and all(isinstance(row, Mapping) for row in rows)
        and row_mappings[0].get("query_index") == 0
        and row_mappings[1].get("query_index") == 1
        and all(
            row.get("schema_version") == "policy_wam_closed_loop_trace.v1" for row in row_mappings
        )
        and loop.get("trace_sha256") == file_sha256(trace_path)
        and loop.get("policy_call_count") == 2
        and loop.get("wam_call_count") == 2
        and loop.get("initial_observation_sha256") == exact_initial_observation_sha256
        and first_policy_observation_sha256 == exact_initial_observation_sha256
        and first_provenance.get("visual_source") == "wam_prediction"
        and _is_sha256(first_wam_observation_sha256)
        and first_wam_observation_sha256 == second_policy_observation_sha256
    )
    all_transitions_completed = bool(row_mappings) and all(
        row.get("status") == "completed" for row in row_mappings
    )
    all_reliability_passed = bool(reliabilities) and all(
        reliability.get("status") == "passed" for reliability in reliabilities
    )
    if not round_trip_passed:
        reasons.append("policy_wam_policy_round_trip_incomplete")
    evidence = {
        "arm_id": "skeleton_only",
        "protocol_sha256": protocol["protocol_sha256"],
        "label_free": True,
        "ranking_outputs_accessed": False,
        "attempt_stage": "rollout",
        "model_invoked": int(loop.get("policy_call_count") or 0) >= 1,
        "scene_id": protocol["scene"]["scene_id"],
        "task_instruction": prompt,
        "policy_id": policy_id,
        "variant": rule.get("frozen_variant"),
        "observation_validity_passed": all_transitions_completed and round_trip_passed,
        "motion_passed": all_reliability_passed,
        "collapse_checks_passed": all_reliability_passed,
        "policy_wam_policy_round_trip_passed": round_trip_passed,
        "transition_count": len(rows),
        "exact_initial_observation_sha256": exact_initial_observation_sha256,
        "initial_camera_sha256_by_view": initial_camera_sha256_by_view,
        "first_wam_observation_sha256": first_wam_observation_sha256,
        "second_policy_observation_sha256": second_policy_observation_sha256,
        "blockers": [*loop.get("blockers", []), *reasons],
    }
    canary = assess_canary(protocol, evidence)
    write_json(output / "canary_attempt_evidence.json", evidence)
    write_json(output / "canary.json", canary)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if canary["status"] in {"passed", "failed"} else "blocked",
        "arm_id": "skeleton_only",
        "protocol_sha256": protocol["protocol_sha256"],
        "policy_id": policy_id,
        "variant": rule.get("frozen_variant"),
        "policy_checkpoint_verification": local_verification,
        "initial_observation_interaction_pixel_count": interaction_pixels,
        "exact_initial_observation_sha256": exact_initial_observation_sha256,
        "initial_camera_sha256_by_view": initial_camera_sha256_by_view,
        "first_wam_observation_sha256": first_wam_observation_sha256,
        "second_policy_observation_sha256": second_policy_observation_sha256,
        "initial_observation_source": (
            "native_isaac_simready_warehouse_camera_canary"
            if initial_camera_paths
            else "mujoco_hybrid_camera_render"
        ),
        "loop_manifest": loop,
        "policy_wam_policy_round_trip_passed": round_trip_passed,
        "canary": canary,
        "provider_execution_required_for_result": True,
        "physical_robot_operated": False,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    write_json(output / "new_site_diagnostic_canary_gpu.json", result)
    return result


__all__ = [
    "INPUT_RECEIPT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "MultiViewCanaryReliabilityGate",
    "build_canary_input_bundle",
    "extract_canary_input_bundle",
    "materialize_canary_background",
    "run_skeleton_only_canary",
]
