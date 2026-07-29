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
from .droid_policy_bridge import validate_droid_observation
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
from .policy_wam_closed_loop import ClosedLoopConfig, run_policy_wam_closed_loop
from .policy_wam_reliability_gate import FrozenMaximumHorizonTerminalCriterion
from .scene_placement.stance_cameras import link_mounted_camera_spec
from .wam_rollout_reliability import (
    TIMING_SCOPE_SESSION,
    ReliabilityThresholds,
    RolloutReliabilityReport,
    assess_rollout_reliability,
)


SCHEMA_VERSION = "new_site_diagnostic_canary_gpu.v1"
INPUT_SCHEMA_VERSION = "new_site_diagnostic_canary_input.v1"
INPUT_RECEIPT_SCHEMA_VERSION = "new_site_diagnostic_canary_input_receipt.v1"
PROTOCOL_NAME = "protocol.json"
BACKGROUND_NAME = "captured_site_background.png"
MANIFEST_NAME = "bundle_manifest.json"
MAX_INPUT_BYTES = 8 * 1024 * 1024
SUPPORTED_ARMS = frozenset({"skeleton_only"})


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
    if arm_id not in SUPPORTED_ARMS:
        raise ValueError("new_site_canary_arm_not_supported_by_openpi_image")
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
    }
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
        if {info.filename for info in infos} != {MANIFEST_NAME, PROTOCOL_NAME, BACKGROUND_NAME}:
            raise ValueError("new_site_canary_input_member_allowlist_mismatch")
        if len(infos) != 3:
            raise ValueError("new_site_canary_input_member_count_invalid")
        for info in infos:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts or info.file_size > MAX_INPUT_BYTES:
                raise ValueError("new_site_canary_input_member_unsafe")
        manifest_value = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        protocol_bytes = archive.read(PROTOCOL_NAME)
        background_bytes = archive.read(BACKGROUND_NAME)
    if not isinstance(manifest_value, Mapping):
        raise ValueError("new_site_canary_input_manifest_not_object")
    manifest = dict(manifest_value)
    declared = manifest.pop("manifest_sha256", None)
    if declared != canonical_sha256(manifest):
        raise ValueError("new_site_canary_input_manifest_sha256_mismatch")
    manifest["manifest_sha256"] = declared
    if manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("new_site_canary_input_schema_invalid")
    if manifest.get("arm_id") not in SUPPORTED_ARMS:
        raise ValueError("new_site_canary_input_arm_unsupported")
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
    protocol = _read_object(protocol_path)
    _validate_protocol_identity(protocol)
    if protocol.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise ValueError("new_site_canary_protocol_identity_mismatch")
    return {
        "manifest": manifest,
        "protocol_path": str(protocol_path),
        "background_path": str(background_path),
    }


def _initial_observation(
    *, runtime: Mapping[str, Any], background_path: str | Path, prompt: str
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
    """Apply the frozen reliability thresholds to both required camera videos."""

    thresholds: ReliabilityThresholds
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
        if not isinstance(videos, Mapping) or set(videos) != {EXTERIOR_VIEW, WRIST_VIEW}:
            raise ValueError("new_site_canary_requires_external_and_wrist_videos")
        actions = np.asarray(prepared_transition.get("reliability_actions_10d"), dtype=float)
        if actions.ndim != 2 or actions.shape[1] != 10 or not np.isfinite(actions).all():
            raise ValueError("new_site_canary_reliability_actions_invalid")
        reports: dict[str, Any] = {}
        flags: list[str] = []
        for view_id in (EXTERIOR_VIEW, WRIST_VIEW):
            video = Path(str(videos[view_id])).expanduser().resolve()
            if not video.is_file() or video.is_symlink():
                raise ValueError(f"new_site_canary_video_missing:{view_id}")
            report = self.assessor(
                video,
                actions,
                self.thresholds,
                timing_flag_scope=TIMING_SCOPE_SESSION,
            )
            reports[view_id] = report.as_dict()
            flags.extend(f"{view_id}:{flag}" for flag in report.flags)
        return {
            "status": "passed" if not flags else "failed",
            "abstain": bool(flags),
            "reasons": flags,
            "reports_by_view": reports,
            "thresholds": asdict(self.thresholds),
            "thresholds_sha256": canonical_sha256(asdict(self.thresholds)),
            "timing_flag_scope": TIMING_SCOPE_SESSION,
            "label_free": True,
            "claim_boundary": "multiview technical reliability only; not task success",
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
    )
    interaction_pixels = int(observation.pop("_diagnostic_interaction_pixel_count"))
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
            max_policy_queries=1,
            execution_mode="engineering_smoke",
        ),
        output_dir=output / "loop",
    )
    trace_path = Path(loop["trace_path"])
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    row = rows[0] if rows else {}
    reliability = row.get("reliability")
    reliability = reliability if isinstance(reliability, Mapping) else {}
    reasons = [str(reason) for reason in reliability.get("reasons") or []]
    evidence = {
        "arm_id": "skeleton_only",
        "protocol_sha256": protocol["protocol_sha256"],
        "label_free": True,
        "ranking_outputs_accessed": False,
        "attempt_stage": "rollout",
        "model_invoked": loop.get("policy_call_count") == 1,
        "scene_id": protocol["scene"]["scene_id"],
        "task_instruction": prompt,
        "policy_id": policy_id,
        "variant": rule.get("frozen_variant"),
        "observation_validity_passed": row.get("status") in {"completed", "abstained"},
        "motion_passed": reliability.get("status") == "passed",
        "collapse_checks_passed": reliability.get("status") == "passed",
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
        "loop_manifest": loop,
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
