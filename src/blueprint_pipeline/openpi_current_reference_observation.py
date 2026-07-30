"""Observation contracts for current-reference OpenPI policy queries."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import write_json
from .ctrl_world_current_reference_wam import (
    ARM_ID,
    validate_ctrl_world_current_reference_result,
)
from .droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_CURRENT_REFERENCE_POLICIES,
    CTRL_WORLD_RELEASED_VIEW_ORDER,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


PUBLIC_INITIAL_OBSERVATION_SCHEMA = "ctrl_world_public_initial_observation.v1"
GENERATED_OBSERVATION_SCHEMA = "openpi_current_reference_generated_observation.v1"
SUPPORTED_OBSERVATION_SCHEMAS = frozenset(
    {PUBLIC_INITIAL_OBSERVATION_SCHEMA, GENERATED_OBSERVATION_SCHEMA}
)


def _read_object(path: str | Path, *, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(reason) from exc
    if not isinstance(value, Mapping):
        raise ValueError(reason)
    return dict(value)


def validate_current_reference_policy_observation_manifest(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate either additive observation variant at the policy boundary."""

    normalized = dict(payload)
    recorded = str(normalized.get("manifest_sha256") or "")
    digest_payload = dict(normalized)
    digest_payload.pop("manifest_sha256", None)
    schema = normalized.get("schema_version")
    if schema not in SUPPORTED_OBSERVATION_SCHEMAS or recorded != canonical_sha256(
        digest_payload
    ):
        raise ValueError("current_reference_policy_observation_manifest_invalid")
    if normalized.get("engineering_canary_eligible") is not True or normalized.get(
        "confirmation_eligible"
    ) is not False:
        raise ValueError("current_reference_policy_observation_claim_boundary_invalid")
    for key in ("physical_future_rgb_used", "future_recorded_state_used"):
        if key in normalized and normalized.get(key) is not False:
            raise ValueError("current_reference_policy_observation_claim_boundary_invalid")
    views = normalized.get("views")
    state = normalized.get("state")
    if not isinstance(views, Mapping) or set(views) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("current_reference_policy_observation_views_invalid")
    if not isinstance(state, Mapping) or set(state) != {
        "joint_position",
        "gripper_position",
        "cartesian_pose_7d",
    }:
        raise ValueError("current_reference_policy_observation_state_invalid")
    if schema == GENERATED_OBSERVATION_SCHEMA:
        source = normalized.get("source")
        query_index = normalized.get("query_index")
        if (
            not isinstance(source, Mapping)
            or source.get("type") != "ctrl_world_wam_generated"
            or source.get("arm_id") != ARM_ID
            or source.get("same_candidate_policy_required") is not True
            or source.get("physical_outcome_accessed") is not False
            or normalized.get("visual_source") != "wam_prediction"
            or normalized.get("state_source") != "commanded_prefix_kinematics"
            or normalized.get("physical_future_rgb_used") is not False
            or normalized.get("future_recorded_state_used") is not False
            or isinstance(query_index, bool)
            or not isinstance(query_index, int)
            or query_index < 1
        ):
            raise ValueError("current_reference_generated_observation_provenance_invalid")
    return normalized


def _finite_array(value: Any, *, shape: tuple[int, ...], reason: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(reason)
    return array


def _validated_policy_receipt(path: str | Path) -> tuple[dict[str, Any], str, str, Path]:
    resolved = Path(path).expanduser().resolve()
    receipt = _read_object(
        resolved,
        reason="current_reference_generated_observation_policy_receipt_invalid",
    )
    digest_payload = dict(receipt)
    manifest_sha256 = str(digest_payload.pop("manifest_sha256", ""))
    policy_id = str(receipt.get("policy_id") or "")
    if (
        receipt.get("schema_version") != "openpi_current_reference_policy_query_receipt.v1"
        or manifest_sha256 != canonical_sha256(digest_payload)
        or policy_id not in CTRL_WORLD_CURRENT_REFERENCE_POLICIES
        or receipt.get("physical_outcome_accessed") is not False
        or receipt.get("wam_called") is not False
    ):
        raise ValueError("current_reference_generated_observation_policy_receipt_invalid")
    return receipt, manifest_sha256, policy_id, resolved


def write_current_reference_transition_evidence(
    *,
    prepared_transition: Mapping[str, Any],
    prior_policy_receipt_path: str | Path,
    wam_request_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Bind commanded state to the native policy action and staged WAM request."""

    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"current_reference_transition_evidence_exists:{destination}")
    prior_receipt, prior_manifest_sha256, policy_id, prior_file = _validated_policy_receipt(
        prior_policy_receipt_path
    )
    request_file = Path(wam_request_receipt_path).expanduser().resolve()
    request_receipt = _read_object(
        request_file,
        reason="current_reference_generated_observation_wam_request_receipt_invalid",
    )
    request_sha256 = str(request_receipt.get("request_sha256") or "")
    seed = request_receipt.get("seed")
    if len(request_sha256) != 64 or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("current_reference_generated_observation_wam_request_receipt_invalid")
    native_action_path = Path(
        str(prepared_transition.get("native_policy_action_path") or "")
    ).expanduser().resolve()
    native_action_sha256 = str(prepared_transition.get("native_policy_action_sha256") or "")
    if (
        not native_action_path.is_file()
        or native_action_path.is_symlink()
        or file_sha256(native_action_path) != native_action_sha256
        or prior_receipt.get("native_action_file_sha256") != native_action_sha256
        or prepared_transition.get("physical_future_observation_used") is not False
    ):
        raise ValueError("current_reference_transition_native_action_binding_invalid")
    wam_request = prepared_transition.get("wam_request")
    adapter_evidence = prepared_transition.get("action_adapter_evidence")
    if not isinstance(wam_request, Mapping) or not isinstance(adapter_evidence, Mapping):
        raise ValueError("current_reference_transition_adapter_evidence_invalid")
    if (
        wam_request.get("physical_future_observation_used") is not False
        or adapter_evidence.get("physical_future_observation_used") is not False
        or adapter_evidence.get("task_outcome_accessed") is not False
    ):
        raise ValueError("current_reference_transition_claim_boundary_invalid")
    state = {
        "joint_position": _finite_array(
            prepared_transition.get("next_joint_position"),
            shape=(7,),
            reason="current_reference_generated_observation_joint_position_invalid",
        ).tolist(),
        "gripper_position": _finite_array(
            prepared_transition.get("next_gripper_position"),
            shape=(1,),
            reason="current_reference_generated_observation_gripper_position_invalid",
        ).tolist(),
        "cartesian_pose_7d": _finite_array(
            prepared_transition.get("next_cartesian_pose_7d"),
            shape=(7,),
            reason="current_reference_generated_observation_cartesian_pose_invalid",
        ).tolist(),
    }
    evidence: dict[str, Any] = {
        "schema_version": "ctrl_world_current_reference_transition_evidence.v1",
        "policy_id": policy_id,
        "prior_policy_receipt_manifest_sha256": prior_manifest_sha256,
        "prior_policy_receipt_file_sha256": file_sha256(prior_file),
        "native_action_file_sha256": native_action_sha256,
        "native_action_shape": list(prepared_transition.get("native_policy_action_shape") or []),
        "action_conditioning_sha256": adapter_evidence.get("conditioning_sha256"),
        "wam_request_sha256": request_sha256,
        "wam_request_receipt_file_sha256": file_sha256(request_file),
        "seed": seed,
        "next_state": state,
        "state_source": "commanded_prefix_kinematics",
        "visual_source": "wam_prediction_pending",
        "physical_future_rgb_used": False,
        "future_recorded_state_used": False,
        "physical_outcome_accessed": False,
    }
    evidence["manifest_sha256"] = canonical_sha256(evidence)
    write_json(destination, evidence)
    return evidence


def build_generated_current_reference_policy_observation(
    *,
    wam_result_path: str | Path,
    wam_result_root: str | Path,
    wam_request_receipt_path: str | Path,
    prior_policy_receipt_path: str | Path,
    transition_evidence_path: str | Path,
    task_prompt: str,
    query_index: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Freeze generated camera views and commanded state for a policy re-query."""

    if isinstance(query_index, bool) or not isinstance(query_index, int) or query_index < 1:
        raise ValueError("current_reference_generated_observation_query_index_invalid")
    prompt = task_prompt.strip()
    if not prompt:
        raise ValueError("current_reference_generated_observation_task_prompt_missing")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"current_reference_generated_observation_output_exists:{output}")
    output.mkdir(parents=True, exist_ok=True)

    request_receipt = _read_object(
        wam_request_receipt_path,
        reason="current_reference_generated_observation_wam_request_receipt_invalid",
    )
    request_sha256 = str(request_receipt.get("request_sha256") or "")
    seed = request_receipt.get("seed")
    if len(request_sha256) != 64 or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("current_reference_generated_observation_wam_request_receipt_invalid")
    raw_wam_result = _read_object(
        wam_result_path, reason="current_reference_generated_observation_wam_result_invalid"
    )
    provider_result_sha256 = str(raw_wam_result.get("result_sha256") or "")
    validated_wam = validate_ctrl_world_current_reference_result(
        raw_wam_result,
        request_receipt=request_receipt,
        seed=seed,
        result_root=wam_result_root,
    )
    _, prior_manifest_sha256, policy_id, prior_policy_receipt_file = (
        _validated_policy_receipt(prior_policy_receipt_path)
    )
    transition_file = Path(transition_evidence_path).expanduser().resolve()
    transition = _read_object(
        transition_file,
        reason="current_reference_generated_observation_transition_evidence_invalid",
    )
    transition_payload = dict(transition)
    transition_manifest_sha256 = str(transition_payload.pop("manifest_sha256", ""))
    if (
        transition.get("schema_version")
        != "ctrl_world_current_reference_transition_evidence.v1"
        or transition_manifest_sha256 != canonical_sha256(transition_payload)
        or transition.get("policy_id") != policy_id
        or transition.get("prior_policy_receipt_manifest_sha256")
        != prior_manifest_sha256
        or transition.get("prior_policy_receipt_file_sha256")
        != file_sha256(prior_policy_receipt_file)
        or transition.get("wam_request_sha256") != request_sha256
        or transition.get("wam_request_receipt_file_sha256")
        != file_sha256(Path(wam_request_receipt_path).expanduser().resolve())
        or transition.get("state_source") != "commanded_prefix_kinematics"
        or transition.get("physical_future_rgb_used") is not False
        or transition.get("future_recorded_state_used") is not False
        or transition.get("physical_outcome_accessed") is not False
    ):
        raise ValueError("current_reference_generated_observation_transition_evidence_invalid")
    transition_state = transition.get("next_state")
    if not isinstance(transition_state, Mapping):
        raise ValueError("current_reference_generated_observation_transition_evidence_invalid")

    state_arrays = {
        "joint_position": _finite_array(
            transition_state.get("joint_position"),
            shape=(7,),
            reason="current_reference_generated_observation_joint_position_invalid",
        ),
        "gripper_position": _finite_array(
            transition_state.get("gripper_position"),
            shape=(1,),
            reason="current_reference_generated_observation_gripper_position_invalid",
        ),
        "cartesian_pose_7d": _finite_array(
            transition_state.get("cartesian_pose_7d"),
            shape=(7,),
            reason="current_reference_generated_observation_cartesian_pose_invalid",
        ),
    }
    state_rows: dict[str, Any] = {}
    for state_id, array in state_arrays.items():
        path = output / f"{state_id}.npy"
        np.save(path, array, allow_pickle=False)
        state_rows[state_id] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    sequences = validated_wam["generated_view_frame_sequences"]
    views: dict[str, Any] = {}
    for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
        source = Path(sequences[view_id][-1]).resolve()
        with Image.open(source) as image:
            if image.size != (320, 192):
                raise ValueError(
                    f"current_reference_generated_observation_frame_geometry_invalid:{view_id}"
                )
        destination = output / f"view_{view_index}_generated_final.png"
        shutil.copyfile(source, destination)
        source_digest = file_sha256(source)
        if source_digest != validated_wam["generated_view_frame_sha256"][view_id][-1]:
            raise ValueError(
                f"current_reference_generated_observation_source_hash_mismatch:{view_id}"
            )
        views[view_id] = {
            "frame_path": str(destination),
            "frame_sha256": file_sha256(destination),
            "native_shape": [192, 320, 3],
            "generated_frame_index": 4,
            "wam_source_frame_sha256": source_digest,
        }

    wam_result_file = Path(wam_result_path).expanduser().resolve()
    request_receipt_file = Path(wam_request_receipt_path).expanduser().resolve()
    manifest: dict[str, Any] = {
        "schema_version": GENERATED_OBSERVATION_SCHEMA,
        "query_index": query_index,
        "task_prompt": prompt,
        "source": {
            "type": "ctrl_world_wam_generated",
            "arm_id": ARM_ID,
            "policy_id": policy_id,
            "same_candidate_policy_required": True,
            "wam_request_sha256": request_sha256,
            "wam_request_receipt_file_sha256": file_sha256(request_receipt_file),
            "wam_provider_result_sha256": provider_result_sha256,
            "wam_result_file_sha256": file_sha256(wam_result_file),
            "prior_policy_receipt_manifest_sha256": prior_manifest_sha256,
            "prior_policy_receipt_file_sha256": file_sha256(prior_policy_receipt_file),
            "transition_evidence_manifest_sha256": transition_manifest_sha256,
            "transition_evidence_file_sha256": file_sha256(transition_file),
            "physical_outcome_accessed": False,
        },
        "views": views,
        "state": state_rows,
        "visual_source": "wam_prediction",
        "state_source": "commanded_prefix_kinematics",
        "observation_history_seed_rule": (
            "repeat_current_generated_frame_and_commanded_state_24_times_for_policy_query_only"
        ),
        "closed_loop_history_continuity_owned_by_transition_adapter": True,
        "physical_future_rgb_used": False,
        "future_recorded_state_used": False,
        "confirmation_eligible": False,
        "engineering_canary_eligible": True,
        "claim_boundary": (
            "generated-observation engineering policy re-query only; not causal qualification, "
            "ranking fidelity, physical success, or confirmation"
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    validate_current_reference_policy_observation_manifest(manifest)
    manifest_path = output / "policy_observation_manifest.json"
    write_json(manifest_path, manifest)
    result = {
        "schema_version": "openpi_current_reference_generated_observation_build.v1",
        "status": "completed",
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "policy_id": policy_id,
        "query_index": query_index,
        "visual_source": "wam_prediction",
        "state_source": "commanded_prefix_kinematics",
        "physical_future_rgb_used": False,
        "future_recorded_state_used": False,
    }
    write_json(output / "build_receipt.json", result)
    return result


__all__ = [
    "GENERATED_OBSERVATION_SCHEMA",
    "PUBLIC_INITIAL_OBSERVATION_SCHEMA",
    "SUPPORTED_OBSERVATION_SCHEMAS",
    "build_generated_current_reference_policy_observation",
    "validate_current_reference_policy_observation_manifest",
    "write_current_reference_transition_evidence",
]
