"""Finite one-query-each canary for current official OpenPI DROID policies."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .common import ensure_dir, write_json
from .ctrl_world_openpi_preprocessing import (
    preprocess_ctrl_world_frame_for_openpi_policy,
)
from .droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_INITIAL_HISTORY_LENGTH,
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_STATE_HISTORY,
    CTRL_WORLD_VIEW_HISTORY_PATHS,
)
from .openpi_current_reference_droid_policy_runtime import (
    CURRENT_REFERENCE_INVENTORY_FILES,
    OpenPICurrentReferenceDroidPolicyClient,
    OpenPICurrentReferenceDroidPolicySpec,
    load_current_reference_policy_specs,
    verify_local_current_reference_checkpoint,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "openpi_current_reference_policy_canary.v1"
FROZEN_POLICY_ORDER = ("pi0_droid", "pi0_fast_droid", "pi05_droid")


def _resolve_packet_path(value: Any, *, manifest_dir: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    return (path if path.is_absolute() else manifest_dir / path).resolve()


def _load_verified_array(
    row: Mapping[str, Any], *, expected_shape: tuple[int, ...], manifest_dir: Path
) -> np.ndarray:
    path = _resolve_packet_path(row.get("path"), manifest_dir=manifest_dir)
    if not path.is_file() or path.is_symlink() or file_sha256(path) != row.get("sha256"):
        raise ValueError("current_reference_initial_state_file_invalid")
    array = np.load(path, allow_pickle=False)
    if array.shape != expected_shape or not np.isfinite(array).all():
        raise ValueError("current_reference_initial_state_array_invalid")
    return np.asarray(array, dtype=np.float64)


def load_current_reference_initial_observation(
    manifest_path: str | Path,
    *,
    image_preprocessor: Callable[[str | Path], np.ndarray] = (
        preprocess_ctrl_world_frame_for_openpi_policy
    ),
) -> dict[str, Any]:
    """Load the exposed engineering packet while verifying every referenced byte."""

    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_digest = str(payload.get("manifest_sha256") or "")
    digest_payload = dict(payload)
    digest_payload.pop("manifest_sha256", None)
    if (
        payload.get("schema_version") != "ctrl_world_public_initial_observation.v1"
        or recorded_digest != canonical_sha256(digest_payload)
        or payload.get("engineering_canary_eligible") is not True
        or payload.get("confirmation_eligible") is not False
    ):
        raise ValueError("current_reference_initial_observation_manifest_invalid")
    views = payload.get("views")
    state = payload.get("state")
    if not isinstance(views, Mapping) or set(views) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("current_reference_initial_observation_views_invalid")
    if not isinstance(state, Mapping):
        raise ValueError("current_reference_initial_observation_state_invalid")
    observation: dict[str, Any] = {
        "prompt": str(payload.get("task_prompt") or ""),
        "observation/joint_position": _load_verified_array(
            state["joint_position"], expected_shape=(7,), manifest_dir=path.parent
        ),
        "observation/gripper_position": _load_verified_array(
            state["gripper_position"], expected_shape=(1,), manifest_dir=path.parent
        ),
        "blueprint/ctrl_world_cartesian_pose_7d": _load_verified_array(
            state["cartesian_pose_7d"], expected_shape=(7,), manifest_dir=path.parent
        ),
    }
    history_paths: dict[str, list[str]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        row = views[view_id]
        if not isinstance(row, Mapping):
            raise ValueError("current_reference_initial_observation_view_row_invalid")
        frame = _resolve_packet_path(row.get("frame_path"), manifest_dir=path.parent)
        if (
            not frame.is_file()
            or frame.is_symlink()
            or file_sha256(frame) != row.get("frame_sha256")
        ):
            raise ValueError("current_reference_initial_observation_frame_invalid")
        observation[view_id] = image_preprocessor(frame)
        history_paths[view_id] = [str(frame)] * CTRL_WORLD_INITIAL_HISTORY_LENGTH
    observation[CTRL_WORLD_VIEW_HISTORY_PATHS] = history_paths
    observation[CTRL_WORLD_STATE_HISTORY] = np.repeat(
        observation["blueprint/ctrl_world_cartesian_pose_7d"][None, :],
        CTRL_WORLD_INITIAL_HISTORY_LENGTH,
        axis=0,
    )
    observation["blueprint/initial_observation_manifest_sha256"] = recorded_digest
    return observation


def _default_checkpoint_downloader(uri: str) -> Path:
    try:
        from openpi.shared import download
    except ImportError as exc:  # pragma: no cover - exercised in pinned GPU runtime
        raise RuntimeError("openpi_current_reference_runtime_not_installed") from exc
    return Path(download.maybe_download(uri)).expanduser().resolve()


def _default_policy_loader(spec: OpenPICurrentReferenceDroidPolicySpec, checkpoint: Path) -> Any:
    try:
        from openpi.policies import policy_config
        from openpi.training import config as training_config
    except ImportError as exc:  # pragma: no cover - exercised in pinned GPU runtime
        raise RuntimeError("openpi_current_reference_runtime_not_installed") from exc
    config = training_config.get_config(spec.config_name)
    if int(config.model.action_horizon) != spec.action_chunk_rows:
        raise ValueError("openpi_current_reference_config_action_horizon_mismatch")
    return policy_config.create_trained_policy(config, checkpoint)


def _gpu_runtime_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {"jax_imported": False, "gpu_device_present": False}
    try:
        import jax

        devices = [
            {
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "id": int(device.id),
            }
            for device in jax.devices()
        ]
        evidence.update(
            {
                "jax_imported": True,
                "jax_version": str(jax.__version__),
                "devices": devices,
                "gpu_device_present": any(row["platform"] == "gpu" for row in devices),
            }
        )
    except Exception as exc:  # noqa: BLE001 - becomes a preserved admission blocker
        evidence["error_type"] = type(exc).__name__
    return evidence


def run_current_reference_policy_canary(
    *,
    source_freeze_path: str | Path,
    checkpoint_inventory_dir: str | Path,
    initial_observation_manifest_path: str | Path,
    output_dir: str | Path,
    checkpoint_downloader: Callable[[str], Path] = _default_checkpoint_downloader,
    policy_loader: Callable[[OpenPICurrentReferenceDroidPolicySpec, Path], Any] = (
        _default_policy_loader
    ),
    initial_observation_loader: Callable[[str | Path], Mapping[str, Any]] = (
        load_current_reference_initial_observation
    ),
    gpu_evidence_collector: Callable[[], Mapping[str, Any]] = _gpu_runtime_evidence,
) -> dict[str, Any]:
    """Run exactly one identity-bound request from each registered policy."""

    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    inventory_dir = Path(checkpoint_inventory_dir).expanduser().resolve()
    specs = load_current_reference_policy_specs(
        source_freeze_path=source_freeze_path,
        checkpoint_inventory_dir=inventory_dir,
    )
    observation = dict(initial_observation_loader(initial_observation_manifest_path))
    gpu = dict(gpu_evidence_collector())
    blockers: list[str] = []
    if gpu.get("gpu_device_present") is not True:
        blockers.append("openpi_current_reference_jax_gpu_not_present")
    policy_results: list[dict[str, Any]] = []
    if not blockers:
        for policy_id in FROZEN_POLICY_ORDER:
            spec = specs[policy_id]
            policy = None
            stage_started = time.monotonic()
            try:
                download_started = time.monotonic()
                checkpoint = checkpoint_downloader(spec.checkpoint_uri)
                download_seconds = time.monotonic() - download_started
                inventory_path = inventory_dir / CURRENT_REFERENCE_INVENTORY_FILES[policy_id]
                verification_started = time.monotonic()
                verification = verify_local_current_reference_checkpoint(
                    spec=spec,
                    checkpoint_dir=checkpoint,
                    checkpoint_inventory_path=inventory_path,
                )
                verification_seconds = time.monotonic() - verification_started
                load_started = time.monotonic()
                policy = policy_loader(spec, checkpoint)
                policy_load_seconds = time.monotonic() - load_started
                client = OpenPICurrentReferenceDroidPolicyClient(
                    spec=spec,
                    policy=policy,
                    local_verification=verification,
                )
                inference_started = time.monotonic()
                response = client.infer(observation)
                policy_inference_seconds = time.monotonic() - inference_started
                action = np.asarray(response["actions"], dtype=np.float64)
                action_path = output / f"{policy_id}_native_action.npy"
                np.save(action_path, action, allow_pickle=False)
                receipt_path = output / f"{policy_id}_policy_receipt.json"
                receipt = {
                    "schema_version": "openpi_current_reference_policy_query_receipt.v1",
                    "policy_id": policy_id,
                    "policy_identity": spec.server_metadata(),
                    "local_checkpoint_verification": verification,
                    "query": response["policy_request_receipt"],
                    "artifact_path_mode": "result_root_relative",
                    "native_action_path": action_path.name,
                    "native_action_file_sha256": file_sha256(action_path),
                    "physical_outcome_accessed": False,
                    "wam_called": False,
                }
                receipt["manifest_sha256"] = canonical_sha256(receipt)
                write_json(receipt_path, receipt)
                policy_results.append(
                    {
                        "policy_id": policy_id,
                        "status": "completed",
                        "artifact_path_mode": "result_root_relative",
                        "receipt_path": receipt_path.name,
                        "receipt_file_sha256": file_sha256(receipt_path),
                        "receipt_manifest_sha256": receipt["manifest_sha256"],
                        "native_action_shape": list(action.shape),
                        "native_action_file_sha256": file_sha256(action_path),
                        "checkpoint_download_seconds": download_seconds,
                        "checkpoint_verification_seconds": verification_seconds,
                        "policy_load_seconds": policy_load_seconds,
                        "policy_inference_seconds": policy_inference_seconds,
                        "stage_elapsed_seconds": time.monotonic() - stage_started,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - failure is experimental evidence
                blocker = (
                    f"openpi_current_reference_policy_failed:{policy_id}:{type(exc).__name__}:{exc}"
                )
                blockers.append(blocker)
                policy_results.append(
                    {
                        "policy_id": policy_id,
                        "status": "blocked",
                        "blocker": blocker,
                        "stage_elapsed_seconds": time.monotonic() - stage_started,
                    }
                )
            finally:
                policy = None
                gc.collect()
                try:
                    import jax

                    jax.clear_caches()
                except Exception:  # noqa: BLE001 - cleanup is best effort
                    pass
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers and len(policy_results) == 3 else "blocked",
        "gpu_runtime": gpu,
        "source_freeze_file_sha256": file_sha256(Path(source_freeze_path).expanduser().resolve()),
        "initial_observation_manifest_file_sha256": file_sha256(
            Path(initial_observation_manifest_path).expanduser().resolve()
        ),
        "frozen_policy_order": list(FROZEN_POLICY_ORDER),
        "requests_per_policy": 1,
        "artifact_path_mode": "result_root_relative",
        "policy_results": policy_results,
        "blockers": blockers,
        "wam_called": False,
        "judge_called": False,
        "physical_outcome_accessed": False,
        "claim_boundary": (
            "real learned-policy identity and inference canary only; not a WAM loop, "
            "policy ranking, physical success, or thesis evidence"
        ),
    }
    result["manifest_sha256"] = canonical_sha256(result)
    write_json(output / "openpi_current_reference_policy_canary.json", result)
    return result


__all__ = [
    "FROZEN_POLICY_ORDER",
    "load_current_reference_initial_observation",
    "run_current_reference_policy_canary",
]
