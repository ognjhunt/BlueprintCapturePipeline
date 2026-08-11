"""Digest-bound native episode traces for task-neutral deformable transfers.

This module is simulator-runtime independent.  It joins the reusable
multi-entity contract to the deterministic deformable-transfer scorer, but it
does not import Isaac, a policy runtime, a renderer, or a learned evaluator.
Unsigned caller mappings remain structural candidates.  Native admission also
requires exact raw event bytes bound to an externally pinned frozen-run seal
and the configured trusted runner key.

The resulting receipt keeps three questions separate:

* did the native task state satisfy the frozen deterministic scorer;
* did action delivery, robot response, gripper response, and contact evidence
  make a learned-policy outcome interpretable; and
* did episode-integrity and media gates admit that result as evaluation
  evidence.

Images never grade the task, an overview image never becomes policy input, and
caller-authored rewards or success labels are rejected.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError

from .deformable_transfer_scoring import score_deformable_transfer
from .native_task_entity_contract import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    materialize_native_task_entity_contract,
)
from .trusted_execution_envelope import verify_trusted_execution_envelope


SCHEMA_VERSION = "native_deformable_episode_trace.v4"
FROZEN_RUN_SCHEMA_VERSION = "native_deformable_frozen_run.v3"
FROZEN_RUN_SEAL_SCHEMA_VERSION = "native_deformable_frozen_run_seal.v1"
CELL_EVALUATION_SCHEMA_VERSION = "native_deformable_cell_evaluation.v1"
MEDIA_MANIFEST_SCHEMA_VERSION = "native_deformable_media_manifest.v2"
NATIVE_TRACE_MANIFEST_SCHEMA_VERSION = "native_deformable_trace_manifest.v1"
RESET_STATE_PROJECTION_SCHEMA_VERSION = "native_reset_state_projection.v1"

EPISODE_KINDS = (
    "zero_action_control",
    "scripted_positive_control",
    "learned_policy_evaluation",
)

CAMERA_IDS = ("external", "wrist", "overview")
POLICY_CAMERA_IDS = frozenset({"external", "wrist"})
OVERVIEW_CAMERA_ID = "overview"
FREE_KINEMATIC_FLAG = 1.0

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,1023}$")

MAX_STEPS = 10_000
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_FRAME_PIXELS = 16_777_216
MAX_NATIVE_TRACE_BYTES = 64 * 1024 * 1024
MAX_REVIEW_VIDEO_BYTES = 256 * 1024 * 1024
ACTION_REPLAY_SCHEMA_VERSION = "native_action_adapter_replay.v1"
ACTION_REPLAY_CONTRACT_ID = "affine_index_map.v1"
H264_DECODE_VERIFIER_CONTRACT_ID = "opencv_videoio_h264_exact_rgb_sequence.v2"
MINIMUM_DEFORMABLE_DISPLACEMENT_FLOOR_M = 1.0e-3
MINIMUM_ARM_MOTION_EPSILON_RAD = 1.0e-3
MINIMUM_GRIPPER_MOTION_EPSILON_M = 1.0e-3
_H264DecodeCacheValue = tuple[int, int, int, tuple[str, ...]]
_H264_DECODE_CACHE: dict[str, _H264DecodeCacheValue | None] = {}

_CALLER_GRADE_FIELDS = frozenset(
    {
        "reward",
        "success",
        "succeeded",
        "score",
        "outcome",
        "policy_grade",
        "learned_grade",
        "human_grade",
        "never_moved",
    }
)


class NativeDeformableEpisodeTraceError(ValueError):
    """Fail-closed structural error with stable, sorted identifiers."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _raise(error: str) -> None:
    raise NativeDeformableEpisodeTraceError([error])


def _clone_mapping(value: Any, *, error: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _raise(error)
    try:
        cloned = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise NativeDeformableEpisodeTraceError([error]) from exc
    if not isinstance(cloned, dict):
        _raise(error)
    return cloned


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    normalized = dict(value)
    if digest_field is not None:
        normalized.pop(digest_field, None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NativeDeformableEpisodeTraceError(
            ["deformable_episode_canonical_json_invalid"]
        ) from exc


def _bytes_digest(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _string(value: Any, *, error: str, identifier: bool = False) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result or (identifier and not _IDENTIFIER.fullmatch(result)):
        _raise(error)
    return result


def _digest(value: Any, *, error: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not _SHA256.fullmatch(result):
        _raise(error)
    return result


def _bool(value: Any, *, error: str) -> bool:
    if not isinstance(value, bool):
        _raise(error)
    return value


def _integer(value: Any, *, error: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _raise(error)
    return int(value)


def _number(
    value: Any,
    *,
    error: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool):
        _raise(error)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NativeDeformableEpisodeTraceError([error]) from exc
    if not math.isfinite(result):
        _raise(error)
    if minimum is not None and result < minimum:
        _raise(error)
    if strictly_positive and result <= 0.0:
        _raise(error)
    return result


def _vector(
    value: Any,
    *,
    size: int | None,
    error: str,
    nonempty: bool = True,
) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        _raise(error)
    try:
        result = [_number(item, error=error) for item in value]
    except TypeError as exc:
        raise NativeDeformableEpisodeTraceError([error]) from exc
    if (nonempty and not result) or (size is not None and len(result) != size):
        _raise(error)
    return result


def _matrix(value: Any, *, columns: int, error: str) -> list[list[float]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        _raise(error)
    try:
        result = [_vector(row, size=columns, error=error) for row in value]
    except TypeError as exc:
        raise NativeDeformableEpisodeTraceError([error]) from exc
    if not result:
        _raise(error)
    return result


def _tensor_3x3(value: Any, *, error: str) -> list[list[list[float]]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        _raise(error)
    try:
        result = [_matrix(row, columns=3, error=error) for row in value]
    except TypeError as exc:
        raise NativeDeformableEpisodeTraceError([error]) from exc
    if not result or any(len(row) != 3 for row in result):
        _raise(error)
    return result


def _quaternion(value: Any, *, error: str) -> list[float]:
    result = _vector(value, size=4, error=error)
    norm = math.sqrt(sum(component * component for component in result))
    if abs(norm - 1.0) > 1.0e-6:
        _raise(error)
    return [component / norm for component in result]


def _canonical_record(value: Any, *, digest_field: str, error: str) -> dict[str, Any]:
    record = _clone_mapping(value, error=error)
    observed = _digest(record.get(digest_field), error=error)
    if observed != _canonical_digest(record, digest_field=digest_field):
        _raise(error)
    return record


def _relative_artifact_path(value: Any, *, error: str) -> str:
    relative_path = _string(value, error=error)
    if (
        not _RELATIVE_PATH.fullmatch(relative_path)
        or "\\" in relative_path
        or relative_path.startswith("/")
    ):
        _raise(error)
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _raise(error)
    return pure.as_posix()


def _resolve_evidence_root(value: str | Path | None) -> Path:
    if value is None:
        _raise("deformable_episode_evidence_root_required")
    root = Path(value)
    try:
        if root.is_symlink():
            _raise("deformable_episode_evidence_root_symlink_forbidden")
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NativeDeformableEpisodeTraceError(
            ["deformable_episode_evidence_root_invalid"]
        ) from exc
    if not resolved.is_dir():
        _raise("deformable_episode_evidence_root_invalid")
    return resolved


def _artifact_bytes(
    *,
    evidence_root: Path,
    relative_path: str,
    maximum_bytes: int,
    error: str,
) -> bytes:
    relative_path = _relative_artifact_path(relative_path, error=error)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nonblocking_flag = getattr(os, "O_NONBLOCK", None)
    if no_follow is None or directory_flag is None:
        _raise(f"{error}_no_follow_unavailable")
    if nonblocking_flag is None:
        _raise(f"{error}_nonblocking_open_unavailable")
    descriptors: list[int] = []
    try:
        flags = os.O_RDONLY | directory_flag | no_follow
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        current = os.open(evidence_root, flags)
        descriptors.append(current)
        parts = PurePosixPath(relative_path).parts
        for part in parts[:-1]:
            current = os.open(part, flags, dir_fd=current)
            descriptors.append(current)
        file_flags = os.O_RDONLY | no_follow | nonblocking_flag
        if hasattr(os, "O_CLOEXEC"):
            file_flags |= os.O_CLOEXEC
        descriptor = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size < 1 or before.st_size > maximum_bytes:
            _raise(error)
        payload_parts: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            payload_parts.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                _raise(error)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or total != before.st_size:
            _raise(f"{error}_changed_during_read")
        payload = b"".join(payload_parts)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            _raise(f"{error}_symlink_forbidden")
        raise NativeDeformableEpisodeTraceError([error]) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    return payload


def _strict_json_object_bytes(payload: bytes, *, error: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_key")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise NativeDeformableEpisodeTraceError([error]) from exc
    if not isinstance(parsed, dict):
        _raise(error)
    return parsed


def _native_authority(
    *,
    source_trace: Mapping[str, Any],
    frozen_run: Mapping[str, Any],
    evidence_root: Path,
    frozen_run_seal_relative_path: str | None,
    expected_frozen_run_seal_sha256: str | None,
    trusted_execution_envelope_relative_path: str | None,
    native_event_relative_path: str | None,
) -> dict[str, Any]:
    """Verify external frozen authority without trusting the structural trace."""

    supplied = (
        frozen_run_seal_relative_path,
        expected_frozen_run_seal_sha256,
        trusted_execution_envelope_relative_path,
        native_event_relative_path,
    )
    if not any(item is not None for item in supplied):
        return {
            "status": "untrusted_structural_candidate",
            "native_event_authority_verified": False,
            "frozen_run_seal_sha256": None,
            "trusted_runner_public_key_sha256": None,
            "native_event_sha256": None,
            "native_event_size_bytes": None,
            "blockers": ["trusted_native_event_authority_missing"],
            "claim_scope": "untrusted_structural_candidate_only",
            "does_not_establish": ["native_simulator_execution", "provider_zero", "physical_truth"],
        }
    if any(item is None for item in supplied):
        return {
            "status": "blocked",
            "native_event_authority_verified": False,
            "frozen_run_seal_sha256": None,
            "trusted_runner_public_key_sha256": None,
            "native_event_sha256": None,
            "native_event_size_bytes": None,
            "blockers": ["trusted_native_event_authority_incomplete"],
            "claim_scope": "blocked_trusted_runner_event_join",
            "does_not_establish": ["native_simulator_execution", "provider_zero", "physical_truth"],
        }

    seal_path = _relative_artifact_path(
        frozen_run_seal_relative_path,
        error="deformable_episode_frozen_run_seal_path_invalid",
    )
    envelope_path = _relative_artifact_path(
        trusted_execution_envelope_relative_path,
        error="deformable_episode_trusted_envelope_path_invalid",
    )
    event_path = _relative_artifact_path(
        native_event_relative_path,
        error="deformable_episode_native_event_path_invalid",
    )
    blockers: list[str] = []
    try:
        expected_seal_digest = _digest(
            expected_frozen_run_seal_sha256,
            error="deformable_episode_expected_frozen_run_seal_digest_invalid",
        )
        seal_bytes = _artifact_bytes(
            evidence_root=evidence_root,
            relative_path=seal_path,
            maximum_bytes=MAX_NATIVE_TRACE_BYTES,
            error="deformable_episode_frozen_run_seal_file_invalid",
        )
        actual_seal_digest = _bytes_digest(seal_bytes)
        if actual_seal_digest != expected_seal_digest:
            blockers.append("frozen_run_seal_external_digest_mismatch")
        seal = _strict_json_object_bytes(
            seal_bytes, error="deformable_episode_frozen_run_seal_json_invalid"
        )
        if seal_bytes != _canonical_json_bytes(seal) + b"\n":
            blockers.append("frozen_run_seal_encoding_not_canonical")
        seal = _canonical_record(
            seal,
            digest_field="seal_digest",
            error="deformable_episode_frozen_run_seal_invalid",
        )
        if set(seal) != {
            "schema_version",
            "frozen_run_contract_digest",
            "native_event_sha256",
            "native_event_size_bytes",
            "trusted_runner_public_key_sha256",
            "trusted_execution",
            "seal_digest",
        }:
            blockers.append("frozen_run_seal_fields_invalid")
        if seal.get("schema_version") != FROZEN_RUN_SEAL_SCHEMA_VERSION:
            blockers.append("frozen_run_seal_schema_version_invalid")
        if seal.get("frozen_run_contract_digest") != frozen_run["contract_digest"]:
            blockers.append("frozen_run_seal_contract_digest_mismatch")
        event_bytes = _artifact_bytes(
            evidence_root=evidence_root,
            relative_path=event_path,
            maximum_bytes=MAX_NATIVE_TRACE_BYTES,
            error="deformable_episode_native_event_file_invalid",
        )
        event_digest = _bytes_digest(event_bytes)
        sealed_event_digest = _digest(
            seal.get("native_event_sha256"),
            error="deformable_episode_frozen_run_seal_native_event_digest_invalid",
        )
        sealed_event_size = _integer(
            seal.get("native_event_size_bytes"),
            error="deformable_episode_frozen_run_seal_native_event_size_invalid",
            minimum=1,
        )
        if sealed_event_digest != event_digest or sealed_event_size != len(event_bytes):
            blockers.append("frozen_run_seal_native_event_identity_mismatch")
        event_document = _strict_json_object_bytes(
            event_bytes, error="deformable_episode_native_event_json_invalid"
        )
        if event_bytes != _canonical_json_bytes(event_document) + b"\n":
            blockers.append("native_event_encoding_not_canonical")
        if event_bytes != _canonical_json_bytes(source_trace) + b"\n":
            blockers.append("native_event_exact_trace_bytes_mismatch")

        trusted = seal.get("trusted_execution")
        if not isinstance(trusted, Mapping):
            blockers.append("frozen_run_seal_trusted_execution_invalid")
            trusted = {}
        elif set(trusted) != {
            "nonce",
            "package_digest",
            "execution_request_digest",
            "worker_entrypoint",
            "worker_source_tree_digest",
            "worker_container_digest",
            "instance_id",
            "allocator_lifecycle_artifact_digests",
        }:
            blockers.append("frozen_run_seal_trusted_execution_fields_invalid")
        lifecycle = trusted.get("allocator_lifecycle_artifact_digests")
        if not isinstance(lifecycle, Mapping):
            lifecycle = {}
            blockers.append("frozen_run_seal_lifecycle_digests_invalid")
        envelope_bytes = _artifact_bytes(
            evidence_root=evidence_root,
            relative_path=envelope_path,
            maximum_bytes=MAX_NATIVE_TRACE_BYTES,
            error="deformable_episode_trusted_envelope_file_invalid",
        )
        with tempfile.TemporaryDirectory(prefix="blueprint-native-authority-") as temporary:
            snapshot_root = Path(temporary)
            envelope_snapshot = snapshot_root / "trusted-envelope.json"
            event_snapshot = snapshot_root / "native-event.json"
            envelope_snapshot.write_bytes(envelope_bytes)
            event_snapshot.write_bytes(event_bytes)
            verification = verify_trusted_execution_envelope(
                envelope_snapshot,
                return_zip_path=event_snapshot,
                expected_nonce=_string(
                    trusted.get("nonce"),
                    error="deformable_episode_frozen_run_seal_nonce_invalid",
                    identifier=True,
                ),
                expected_run_digest=frozen_run["contract_digest"],
                expected_package_digest=_digest(
                    trusted.get("package_digest"),
                    error="deformable_episode_frozen_run_seal_package_digest_invalid",
                ),
                expected_execution_request_digest=_digest(
                    trusted.get("execution_request_digest"),
                    error="deformable_episode_frozen_run_seal_request_digest_invalid",
                ),
                expected_worker_entrypoint=_string(
                    trusted.get("worker_entrypoint"),
                    error="deformable_episode_frozen_run_seal_worker_entrypoint_invalid",
                ),
                expected_worker_source_tree_digest=_digest(
                    trusted.get("worker_source_tree_digest"),
                    error="deformable_episode_frozen_run_seal_source_tree_digest_invalid",
                ),
                expected_worker_container_digest=_digest(
                    trusted.get("worker_container_digest"),
                    error="deformable_episode_frozen_run_seal_container_digest_invalid",
                ),
                expected_instance_id=_string(
                    trusted.get("instance_id"),
                    error="deformable_episode_frozen_run_seal_instance_id_invalid",
                    identifier=True,
                ),
                expected_allocator_lifecycle_artifact_digests=lifecycle,
            )
        blockers.extend(verification["blockers"])
        runner_key = _digest(
            seal.get("trusted_runner_public_key_sha256"),
            error="deformable_episode_frozen_run_seal_runner_key_invalid",
        )
        if verification.get("presented_public_key_sha256") != runner_key:
            blockers.append("frozen_run_seal_runner_identity_mismatch")
    except NativeDeformableEpisodeTraceError as exc:
        blockers.extend(exc.errors)
        actual_seal_digest = None
        runner_key = None
        event_digest = None
        event_bytes = b""
        verification = {"structural_trust_verified": False}

    blockers = sorted(set(blockers))
    verified = bool(not blockers and verification.get("structural_trust_verified"))
    return {
        "status": "verified" if verified else "blocked",
        "native_event_authority_verified": verified,
        "frozen_run_seal_sha256": actual_seal_digest,
        "trusted_runner_public_key_sha256": runner_key,
        "native_event_sha256": event_digest,
        "native_event_size_bytes": len(event_bytes) if event_bytes else None,
        "blockers": blockers,
        "claim_scope": "trusted_runner_attested_event_bytes_only",
        "does_not_establish": [
            "allocator_lifecycle_semantics",
            "provider_zero",
            "physical_material_equivalence",
            "real_robot_performance",
        ],
    }


def _nonempty_digest_map(value: Any, *, error: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _raise(error)
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _string(raw_key, error=error, identifier=True)
        if key in result:
            _raise(error)
        result[key] = _digest(raw_value, error=error)
    return dict(sorted(result.items()))


def _exact_identifier_map(
    value: Any, *, expected_keys: set[str] | None, error: str
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _raise(error)
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = _string(raw_key, error=error, identifier=True)
        item = _string(raw_value, error=error, identifier=True)
        if key in result:
            _raise(error)
        result[key] = item
    if expected_keys is not None and set(result) != expected_keys:
        _raise(error)
    return dict(sorted(result.items()))


def _normalize_action_replay_contract(value: Any, *, error: str) -> dict[str, Any]:
    source = _canonical_record(value, digest_field="contract_digest", error=error)
    indices_source = source.get("arm_source_indices")
    if isinstance(indices_source, (str, bytes, bytearray, Mapping)) or not isinstance(
        indices_source, Sequence
    ):
        _raise(error)
    indices = [_integer(index, error=error) for index in indices_source]
    if not indices or len(indices) != len(set(indices)):
        _raise(error)
    scales = _vector(source.get("arm_scale"), size=len(indices), error=error)
    offsets = _vector(source.get("arm_offset"), size=len(indices), error=error)
    source_size = _integer(source.get("source_output_size"), error=error, minimum=1)
    gripper_index = _integer(source.get("gripper_source_index"), error=error)
    if max([*indices, gripper_index]) >= source_size:
        _raise(error)
    normalized = {
        "schema_version": ACTION_REPLAY_SCHEMA_VERSION,
        "contract_id": ACTION_REPLAY_CONTRACT_ID,
        "command_space": _string(source.get("command_space"), error=error, identifier=True),
        "source_output_size": source_size,
        "arm_source_indices": indices,
        "arm_scale": scales,
        "arm_offset": offsets,
        "gripper_source_index": gripper_index,
        "gripper_scale": _number(source.get("gripper_scale"), error=error),
        "gripper_offset": _number(source.get("gripper_offset"), error=error),
        "native_action_layout": _string(
            source.get("native_action_layout"), error=error, identifier=True
        ),
    }
    if (
        source.get("schema_version") != ACTION_REPLAY_SCHEMA_VERSION
        or source.get("contract_id") != ACTION_REPLAY_CONTRACT_ID
        or normalized["native_action_layout"] != "arm_then_gripper"
    ):
        _raise("deformable_episode_frozen_action_replay_contract_unsupported")
    normalized["contract_digest"] = _canonical_digest(normalized)
    if normalized["contract_digest"] != source["contract_digest"]:
        _raise(error)
    return normalized


def _normalize_action_replay_contract_map(
    value: Any, *, expected_keys: set[str], error: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        _raise(error)
    return {
        actor_digest: _normalize_action_replay_contract(value[actor_digest], error=error)
        for actor_digest in sorted(expected_keys)
    }


def _normalize_frozen_run_contract(
    value: Any,
    *,
    entity_contract_digest: str,
    task_spec_sha256: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    error = "deformable_episode_frozen_run_contract_invalid"
    source = _canonical_record(value, digest_field="contract_digest", error=error)
    if source.get("schema_version") != FROZEN_RUN_SCHEMA_VERSION:
        _raise(error)
    cells = _nonempty_digest_map(source.get("cell_identity_digest_by_id"), error=error)
    candidates = _nonempty_digest_map(source.get("candidate_identity_digest_by_id"), error=error)
    if len(candidates) != 2 or len(set(candidates.values())) != 2:
        _raise(error)
    controls = _nonempty_digest_map(
        source.get("control_identity_digest_by_episode_kind"), error=error
    )
    if set(controls) != {"zero_action_control", "scripted_positive_control"}:
        _raise(error)
    camera_calibrations = _nonempty_digest_map(
        source.get("camera_calibration_digest_by_camera_id"), error=error
    )
    renderers = _nonempty_digest_map(
        source.get("renderer_identity_sha256_by_camera_id"), error=error
    )
    if set(camera_calibrations) != set(CAMERA_IDS) or set(renderers) != set(CAMERA_IDS):
        _raise(error)
    actor_identity_digests = set(candidates.values()) | set(controls.values())
    replay_contracts = _normalize_action_replay_contract_map(
        source.get("action_replay_contract_by_actor_identity_digest"),
        expected_keys=actor_identity_digests,
        error=error,
    )
    review_video_codecs = _exact_identifier_map(
        source.get("review_video_codec_by_camera_id"),
        expected_keys=set(CAMERA_IDS),
        error=error,
    )
    review_video_containers = _exact_identifier_map(
        source.get("review_video_container_by_camera_id"),
        expected_keys=set(CAMERA_IDS),
        error=error,
    )
    if set(review_video_codecs.values()) != {"h264"} or set(review_video_containers.values()) != {
        "mp4"
    }:
        _raise("deformable_episode_frozen_review_video_contract_unsupported")
    normalized = {
        "schema_version": FROZEN_RUN_SCHEMA_VERSION,
        "suite_id": _string(source.get("suite_id"), error=error, identifier=True),
        "entity_contract_digest": _digest(source.get("entity_contract_digest"), error=error),
        "task_spec_sha256": _digest(source.get("task_spec_sha256"), error=error),
        "prompt_sha256": _digest(source.get("prompt_sha256"), error=error),
        "cell_identity_digest_by_id": cells,
        "candidate_identity_digest_by_id": candidates,
        "control_identity_digest_by_episode_kind": controls,
        "trace_thresholds_sha256": _digest(source.get("trace_thresholds_sha256"), error=error),
        "camera_calibration_digest_by_camera_id": camera_calibrations,
        "renderer_identity_sha256_by_camera_id": renderers,
        "action_replay_contract_by_actor_identity_digest": replay_contracts,
        "review_video_codec_by_camera_id": review_video_codecs,
        "review_video_container_by_camera_id": review_video_containers,
        "frozen_reset_state_id": _string(
            source.get("frozen_reset_state_id"), error=error, identifier=True
        ),
        "frozen_reset_state_sha256": _digest(source.get("frozen_reset_state_sha256"), error=error),
        "frozen_deformable_start_state_sha256": _digest(
            source.get("frozen_deformable_start_state_sha256"), error=error
        ),
    }
    normalized["contract_digest"] = _canonical_digest(normalized)
    if normalized["contract_digest"] != source["contract_digest"]:
        _raise("deformable_episode_frozen_run_contract_projection_mismatch")
    if normalized["entity_contract_digest"] != entity_contract_digest:
        _raise("deformable_episode_frozen_entity_contract_mismatch")
    if normalized["task_spec_sha256"] != task_spec_sha256:
        _raise("deformable_episode_frozen_task_spec_mismatch")
    if normalized["prompt_sha256"] != prompt_sha256:
        _raise("deformable_episode_frozen_prompt_mismatch")
    return normalized


def _contains_caller_grade(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _CALLER_GRADE_FIELDS:
                return True
            if _contains_caller_grade(item):
                return True
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_caller_grade(item) for item in value)
    return False


def _normalize_cell(value: Any) -> dict[str, Any]:
    source = _clone_mapping(value, error="deformable_episode_cell_invalid")
    seed = _integer(source.get("seed"), error="deformable_episode_cell_seed_invalid")
    return {
        "cell_id": _string(
            source.get("cell_id"), error="deformable_episode_cell_id_invalid", identifier=True
        ),
        "family": _string(
            source.get("family"), error="deformable_episode_cell_family_invalid", identifier=True
        ),
        "seed": seed,
        "scene_sha256": _digest(
            source.get("scene_sha256"), error="deformable_episode_cell_scene_digest_invalid"
        ),
        "asset_bundle_sha256": _digest(
            source.get("asset_bundle_sha256"),
            error="deformable_episode_cell_asset_bundle_digest_invalid",
        ),
        "resolved_parameters_sha256": _digest(
            source.get("resolved_parameters_sha256"),
            error="deformable_episode_cell_parameters_digest_invalid",
        ),
        "camera_contract_sha256": _digest(
            source.get("camera_contract_sha256"),
            error="deformable_episode_cell_camera_digest_invalid",
        ),
        "native_applied_parameters_receipt_sha256": _digest(
            source.get("native_applied_parameters_receipt_sha256"),
            error="deformable_episode_cell_applied_parameters_digest_invalid",
        ),
    }


def _normalize_thresholds(value: Any) -> dict[str, Any]:
    source = _clone_mapping(value, error="deformable_episode_trace_thresholds_invalid")
    return {
        "maximum_frame_sync_skew_ns": _integer(
            source.get("maximum_frame_sync_skew_ns"),
            error="deformable_episode_frame_sync_threshold_invalid",
            minimum=1,
        ),
        "maximum_frame_simulation_time_skew_s": _number(
            source.get("maximum_frame_simulation_time_skew_s"),
            error="deformable_episode_simulation_sync_threshold_invalid",
            minimum=0.0,
        ),
        "arm_motion_epsilon_rad": _number(
            source.get("arm_motion_epsilon_rad"),
            error="deformable_episode_arm_motion_epsilon_invalid",
            minimum=MINIMUM_ARM_MOTION_EPSILON_RAD,
        ),
        "gripper_motion_epsilon_m": _number(
            source.get("gripper_motion_epsilon_m"),
            error="deformable_episode_gripper_motion_epsilon_invalid",
            minimum=MINIMUM_GRIPPER_MOTION_EPSILON_M,
        ),
        "action_epsilon": _number(
            source.get("action_epsilon"),
            error="deformable_episode_action_epsilon_invalid",
            strictly_positive=True,
        ),
        "contact_force_epsilon_n": _number(
            source.get("contact_force_epsilon_n"),
            error="deformable_episode_contact_force_epsilon_invalid",
            minimum=0.0,
        ),
        "minimum_deformable_displacement_m": _number(
            source.get("minimum_deformable_displacement_m"),
            error="deformable_episode_minimum_deformable_displacement_invalid",
            minimum=MINIMUM_DEFORMABLE_DISPLACEMENT_FLOOR_M,
        ),
    }


def _normalize_actor(value: Any, *, episode_kind: str) -> dict[str, Any]:
    source = _canonical_record(
        value,
        digest_field="identity_digest",
        error="deformable_episode_actor_identity_invalid",
    )
    if episode_kind == "learned_policy_evaluation":
        if source.get("kind") != "learned_policy":
            _raise("deformable_episode_actor_kind_invalid")
        normalized: dict[str, Any] = {
            "kind": "learned_policy",
            "candidate_id": _string(
                source.get("candidate_id"),
                error="deformable_episode_candidate_id_invalid",
                identifier=True,
            ),
            "source_reference": _string(
                source.get("source_reference"),
                error="deformable_episode_policy_source_invalid",
            ),
            "source_revision": _string(
                source.get("source_revision"),
                error="deformable_episode_policy_revision_invalid",
            ),
            "checkpoint_reference": _string(
                source.get("checkpoint_reference"),
                error="deformable_episode_checkpoint_reference_invalid",
            ),
            "checkpoint_sha256": _digest(
                source.get("checkpoint_sha256"),
                error="deformable_episode_checkpoint_digest_invalid",
            ),
            "runtime_sha256": _digest(
                source.get("runtime_sha256"),
                error="deformable_episode_policy_runtime_digest_invalid",
            ),
            "preprocessing_sha256": _digest(
                source.get("preprocessing_sha256"),
                error="deformable_episode_preprocessing_digest_invalid",
            ),
            "action_adapter_sha256": _digest(
                source.get("action_adapter_sha256"),
                error="deformable_episode_action_adapter_digest_invalid",
            ),
            "model_seed": _integer(
                source.get("model_seed"),
                error="deformable_episode_model_seed_invalid",
            ),
            "policy_self_grading_allowed": _bool(
                source.get("policy_self_grading_allowed"),
                error="deformable_episode_policy_self_grading_invalid",
            ),
        }
        if normalized["policy_self_grading_allowed"]:
            _raise("deformable_episode_policy_self_grading_forbidden")
    else:
        if source.get("kind") != "deterministic_control":
            _raise("deformable_episode_actor_kind_invalid")
        expected_control = (
            "zero_action" if episode_kind == "zero_action_control" else "scripted_positive"
        )
        normalized = {
            "kind": "deterministic_control",
            "control_id": _string(
                source.get("control_id"),
                error="deformable_episode_control_id_invalid",
                identifier=True,
            ),
            "source_revision": _string(
                source.get("source_revision"),
                error="deformable_episode_control_revision_invalid",
            ),
            "controller_sha256": _digest(
                source.get("controller_sha256"),
                error="deformable_episode_controller_digest_invalid",
            ),
            "control_seed": _integer(
                source.get("control_seed"),
                error="deformable_episode_control_seed_invalid",
            ),
            "policy_self_grading_allowed": _bool(
                source.get("policy_self_grading_allowed"),
                error="deformable_episode_control_self_grading_invalid",
            ),
        }
        if normalized["control_id"] != expected_control:
            _raise("deformable_episode_control_identity_mismatch")
        if normalized["policy_self_grading_allowed"]:
            _raise("deformable_episode_policy_self_grading_forbidden")
    normalized["identity_digest"] = _canonical_digest(normalized)
    if normalized["identity_digest"] != source["identity_digest"]:
        _raise("deformable_episode_actor_identity_projection_mismatch")
    return normalized


def _entity_ids(
    *, entity_contract: Mapping[str, Any], task_spec: Mapping[str, Any]
) -> dict[str, str]:
    roles = entity_contract["semantic_role_index"]
    expected = {
        "deformable": ("deformable_entity_id", "movable_deformable"),
        "destination": ("destination_entity_id", "destination_receptacle"),
        "robot": ("robot_entity_id", "robot"),
    }
    result: dict[str, str] = {}
    for label, (spec_key, role) in expected.items():
        entity_id = _string(
            task_spec.get(spec_key), error=f"deformable_episode_{label}_entity_id_invalid"
        )
        if entity_id not in roles.get(role, []):
            _raise(f"deformable_episode_{label}_entity_role_mismatch:{entity_id}")
        result[label] = entity_id
    if len(set(result.values())) != len(result):
        _raise("deformable_episode_scoring_entity_ids_not_distinct")
    return result


def _zero_vector(value: Sequence[float]) -> bool:
    return all(item == 0.0 for item in value)


def _normalize_reset_state_projection(
    value: Any,
    *,
    entity_id: str,
    physics_type: str,
) -> dict[str, Any]:
    error = f"deformable_episode_reset_state_projection_invalid:{entity_id}"
    source = _canonical_record(value, digest_field="projection_digest", error=error)
    if (
        source.get("schema_version") != RESET_STATE_PROJECTION_SCHEMA_VERSION
        or source.get("entity_id") != entity_id
        or source.get("physics_type") != physics_type
    ):
        _raise(error)
    normalized: dict[str, Any] = {
        "schema_version": RESET_STATE_PROJECTION_SCHEMA_VERSION,
        "entity_id": entity_id,
        "physics_type": physics_type,
    }
    if physics_type == "deformable_volume":
        positions = _matrix(source.get("nodal_positions_world_m"), columns=3, error=error)
        velocities = _matrix(source.get("nodal_velocities_world_mps"), columns=3, error=error)
        targets = _matrix(source.get("nodal_kinematic_targets"), columns=4, error=error)
        if (
            len(velocities) != len(positions)
            or len(targets) != len(positions)
            or any(
                target[:3] != position for target, position in zip(targets, positions, strict=True)
            )
            or any(not math.isclose(target[3], FREE_KINEMATIC_FLAG) for target in targets)
            or any(not _zero_vector(velocity) for velocity in velocities)
        ):
            _raise(error)
        normalized.update(
            {
                "nodal_positions_world_m": positions,
                "nodal_velocities_world_mps": velocities,
                "nodal_kinematic_targets": targets,
            }
        )
    elif physics_type == "robot_articulation":
        joint_positions = _vector(source.get("joint_positions_rad"), size=None, error=error)
        joint_velocities = _vector(
            source.get("joint_velocities_rad_s"),
            size=len(joint_positions),
            error=error,
        )
        if not _zero_vector(joint_velocities):
            _raise(error)
        normalized.update(
            {
                "joint_positions_rad": joint_positions,
                "joint_velocities_rad_s": joint_velocities,
                "gripper_width_m": _number(source.get("gripper_width_m"), error=error, minimum=0.0),
            }
        )
    elif physics_type in {"rigid_body", "static_collider", "articulation"}:
        pose = _clone_mapping(source.get("pose_world"), error=error)
        linear_velocity = _vector(source.get("linear_velocity_world_mps"), size=3, error=error)
        angular_velocity = _vector(source.get("angular_velocity_world_radps"), size=3, error=error)
        if not _zero_vector(linear_velocity) or not _zero_vector(angular_velocity):
            _raise(error)
        normalized.update(
            {
                "pose_world": {
                    "position_m": _vector(pose.get("position_m"), size=3, error=error),
                    "orientation_xyzw": _quaternion(pose.get("orientation_xyzw"), error=error),
                },
                "linear_velocity_world_mps": linear_velocity,
                "angular_velocity_world_radps": angular_velocity,
            }
        )
        if physics_type == "articulation":
            joint_positions = _vector(source.get("joint_positions_rad"), size=None, error=error)
            joint_velocities = _vector(
                source.get("joint_velocities_rad_s"),
                size=len(joint_positions),
                error=error,
            )
            if not _zero_vector(joint_velocities):
                _raise(error)
            normalized.update(
                {
                    "joint_positions_rad": joint_positions,
                    "joint_velocities_rad_s": joint_velocities,
                }
            )
    else:
        _raise(f"deformable_episode_reset_projection_physics_type_unsupported:{entity_id}")
    normalized["projection_digest"] = _canonical_digest(normalized)
    if normalized["projection_digest"] != source["projection_digest"]:
        _raise(error)
    return normalized


def _normalize_reset_state_projections(
    value: Any,
    *,
    entity_physics_type_by_id: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != set(entity_physics_type_by_id):
        _raise("deformable_episode_reset_state_projection_entity_set_mismatch")
    return {
        entity_id: _normalize_reset_state_projection(
            value[entity_id],
            entity_id=entity_id,
            physics_type=entity_physics_type_by_id[entity_id],
        )
        for entity_id in sorted(entity_physics_type_by_id)
    }


def _reset_state_projection_set_digest(
    projections: Mapping[str, Mapping[str, Any]],
) -> str:
    return _canonical_digest({"native_reset_state_projection_by_entity_id": dict(projections)})


def _normalize_reset(
    value: Any,
    *,
    all_entity_ids: Sequence[str],
    entity_physics_type_by_id: Mapping[str, str],
    deformable_entity_id: str,
    robot_entity_id: str,
    actor: Mapping[str, Any],
    frozen_run: Mapping[str, Any],
    episode_start_timestamp_ns: int,
) -> dict[str, Any]:
    source = _canonical_record(
        value,
        digest_field="receipt_digest",
        error="deformable_episode_reset_receipt_invalid",
    )
    reset_timestamp = _integer(
        source.get("reset_timestamp_ns"), error="deformable_episode_reset_timestamp_invalid"
    )
    readback_timestamp = _integer(
        source.get("native_readback_timestamp_ns"),
        error="deformable_episode_reset_readback_timestamp_invalid",
    )
    actor_reset_timestamp = _integer(
        source.get("actor_reset_timestamp_ns"),
        error="deformable_episode_actor_reset_timestamp_invalid",
    )
    if not (
        reset_timestamp <= actor_reset_timestamp <= readback_timestamp <= episode_start_timestamp_ns
    ):
        _raise("deformable_episode_reset_timestamp_order_invalid")

    count_source = source.get("native_reset_write_count_by_entity_id")
    readback_source = source.get("native_state_readback_sha256_by_entity_id")
    if not isinstance(count_source, Mapping) or not isinstance(readback_source, Mapping):
        _raise("deformable_episode_reset_entity_readback_invalid")
    expected_entity_ids = set(all_entity_ids)
    if set(count_source) != expected_entity_ids or set(readback_source) != expected_entity_ids:
        _raise("deformable_episode_reset_entity_set_mismatch")
    counts: dict[str, int] = {}
    readbacks: dict[str, str] = {}
    for entity_id in sorted(expected_entity_ids):
        counts[entity_id] = _integer(
            count_source.get(entity_id),
            error=f"deformable_episode_reset_count_invalid:{entity_id}",
        )
        readbacks[entity_id] = _digest(
            readback_source.get(entity_id),
            error=f"deformable_episode_reset_readback_digest_invalid:{entity_id}",
        )

    projections = _normalize_reset_state_projections(
        source.get("native_reset_state_projection_by_entity_id"),
        entity_physics_type_by_id=entity_physics_type_by_id,
    )
    for entity_id, projection in projections.items():
        if readbacks[entity_id] != projection["projection_digest"]:
            _raise(f"deformable_episode_reset_readback_projection_mismatch:{entity_id}")
    deformable_projection = projections[deformable_entity_id]
    robot_projection = projections[robot_entity_id]
    deformable_start_positions = deformable_projection["nodal_positions_world_m"]
    deformable_start_velocities = deformable_projection["nodal_velocities_world_mps"]
    deformable_start_targets = deformable_projection["nodal_kinematic_targets"]
    robot_joints = robot_projection["joint_positions_rad"]
    robot_joint_velocities = robot_projection["joint_velocities_rad_s"]
    gripper_width = robot_projection["gripper_width_m"]
    if (
        _matrix(
            source.get("deformable_nodal_positions_world_m"),
            columns=3,
            error="deformable_episode_reset_deformable_nodal_positions_invalid",
        )
        != deformable_start_positions
        or _matrix(
            source.get("deformable_nodal_velocities_world_mps"),
            columns=3,
            error="deformable_episode_reset_deformable_nodal_velocities_invalid",
        )
        != deformable_start_velocities
        or _matrix(
            source.get("deformable_nodal_kinematic_targets"),
            columns=4,
            error="deformable_episode_reset_deformable_kinematic_targets_invalid",
        )
        != deformable_start_targets
    ):
        _raise("deformable_episode_reset_deformable_projection_mismatch")
    if (
        _vector(
            source.get("robot_joint_positions_rad"),
            size=len(robot_joints),
            error="deformable_episode_reset_robot_joints_invalid",
        )
        != robot_joints
        or _vector(
            source.get("robot_joint_velocities_rad_s"),
            size=len(robot_joints),
            error="deformable_episode_reset_robot_joint_velocities_invalid",
        )
        != robot_joint_velocities
        or _number(
            source.get("gripper_width_m"),
            error="deformable_episode_reset_gripper_width_invalid",
            minimum=0.0,
        )
        != gripper_width
    ):
        _raise("deformable_episode_reset_robot_projection_mismatch")
    deformable_start_state_sha256 = deformable_projection["projection_digest"]
    reset_state_projection_sha256 = _reset_state_projection_set_digest(projections)
    if (
        _digest(
            source.get("deformable_start_state_sha256"),
            error="deformable_episode_reset_deformable_start_state_digest_invalid",
        )
        != deformable_start_state_sha256
        or frozen_run["frozen_deformable_start_state_sha256"] != deformable_start_state_sha256
    ):
        _raise("deformable_episode_reset_deformable_start_state_mismatch")
    expected_reset_method = (
        "policy.reset" if actor["kind"] == "learned_policy" else "controller.reset"
    )
    actor_seed = actor["model_seed"] if actor["kind"] == "learned_policy" else actor["control_seed"]
    normalized = {
        "reset_id": _string(
            source.get("reset_id"), error="deformable_episode_reset_id_invalid", identifier=True
        ),
        "frozen_reset_state_id": _string(
            source.get("frozen_reset_state_id"),
            error="deformable_episode_frozen_reset_state_id_invalid",
            identifier=True,
        ),
        "frozen_reset_state_sha256": reset_state_projection_sha256,
        "reset_timestamp_ns": reset_timestamp,
        "actor_reset_timestamp_ns": actor_reset_timestamp,
        "native_readback_timestamp_ns": readback_timestamp,
        "actor_identity_digest": _digest(
            source.get("actor_identity_digest"),
            error="deformable_episode_reset_actor_identity_invalid",
        ),
        "actor_seed": _integer(
            source.get("actor_seed"), error="deformable_episode_reset_actor_seed_invalid"
        ),
        "actor_reset_method": _string(
            source.get("actor_reset_method"),
            error="deformable_episode_actor_reset_method_invalid",
            identifier=True,
        ),
        "actor_reset_invoked": _bool(
            source.get("actor_reset_invoked"),
            error="deformable_episode_actor_reset_invocation_invalid",
        ),
        "native_reset_write_count_by_entity_id": counts,
        "native_state_readback_sha256_by_entity_id": readbacks,
        "native_reset_state_projection_by_entity_id": projections,
        "deformable_nodal_positions_world_m": deformable_start_positions,
        "deformable_nodal_velocities_world_mps": deformable_start_velocities,
        "deformable_nodal_kinematic_targets": deformable_start_targets,
        "deformable_start_state_sha256": deformable_start_state_sha256,
        "robot_joint_positions_rad": robot_joints,
        "robot_joint_velocities_rad_s": robot_joint_velocities,
        "gripper_width_m": gripper_width,
        "native_readback_matches_frozen_state": _bool(
            source.get("native_readback_matches_frozen_state"),
            error="deformable_episode_reset_match_invalid",
        ),
        "initial_penetration_observed": _bool(
            source.get("initial_penetration_observed"),
            error="deformable_episode_reset_penetration_invalid",
        ),
    }
    if (
        not normalized["native_readback_matches_frozen_state"]
        or normalized["initial_penetration_observed"]
    ):
        _raise("deformable_episode_reset_gate_failed")
    if (
        _digest(
            source.get("frozen_reset_state_sha256"),
            error="deformable_episode_frozen_reset_state_digest_invalid",
        )
        != reset_state_projection_sha256
        or normalized["frozen_reset_state_id"] != frozen_run["frozen_reset_state_id"]
        or normalized["frozen_reset_state_sha256"] != frozen_run["frozen_reset_state_sha256"]
    ):
        _raise("deformable_episode_frozen_reset_state_mismatch")
    if normalized["actor_identity_digest"] != actor["identity_digest"]:
        _raise("deformable_episode_reset_actor_identity_mismatch")
    if normalized["actor_seed"] != actor_seed:
        _raise("deformable_episode_reset_actor_seed_mismatch")
    if (
        normalized["actor_reset_method"] != expected_reset_method
        or not normalized["actor_reset_invoked"]
    ):
        _raise("deformable_episode_actor_reset_not_proven")
    normalized["receipt_digest"] = _canonical_digest(normalized)
    if normalized["receipt_digest"] != source["receipt_digest"]:
        _raise("deformable_episode_reset_receipt_projection_mismatch")
    return normalized


def _normalize_pose(value: Any, *, error: str) -> dict[str, Any]:
    source = _clone_mapping(value, error=error)
    return {
        "position_m": _vector(source.get("position_m"), size=3, error=error),
        "orientation_xyzw": _quaternion(source.get("orientation_xyzw"), error=error),
    }


def _numeric_map(value: Any, *, error: str, integer: bool = False) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or not value:
        _raise(error)
    normalized: dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = _string(raw_key, error=error, identifier=True)
        normalized[key] = (
            _integer(raw_value, error=error)
            if integer
            else _number(raw_value, error=error, minimum=0.0)
        )
    return dict(sorted(normalized.items()))


def _normalize_deformable(
    value: Any, *, entity_ids: Mapping[str, str], sample_index: int
) -> dict[str, Any]:
    error = f"deformable_episode_deformable_readback_invalid:{sample_index}"
    source = _clone_mapping(value, error=error)
    positions = _matrix(source.get("nodal_positions_world_m"), columns=3, error=error)
    velocities = _matrix(source.get("nodal_velocities_world_mps"), columns=3, error=error)
    flags = _vector(source.get("nodal_kinematic_flags"), size=len(positions), error=error)
    gradients = _tensor_3x3(source.get("deformation_gradients"), error=error)
    if len(velocities) != len(positions):
        _raise(f"deformable_episode_nodal_state_shape_mismatch:{sample_index}")
    pair_counts = _numeric_map(
        source.get("contact_pair_count_by_entity_id"), error=error, integer=True
    )
    forces = _numeric_map(source.get("contact_normal_force_n_by_entity_id"), error=error)
    for required in (entity_ids["robot"], entity_ids["destination"]):
        if required not in pair_counts or required not in forces:
            _raise(
                f"deformable_episode_deformable_contact_entity_missing:{sample_index}:{required}"
            )
    return {
        "nodal_positions_world_m": positions,
        "nodal_velocities_world_mps": velocities,
        "deformation_gradients": gradients,
        "nodal_kinematic_flags": flags,
        "state_write_count_after_episode_start": _integer(
            source.get("state_write_count_after_episode_start"), error=error
        ),
        "solver_divergence_count": _integer(source.get("solver_divergence_count"), error=error),
        "contact_pair_count_by_entity_id": pair_counts,
        "contact_normal_force_n_by_entity_id": forces,
        "hidden_attachment_active": _bool(source.get("hidden_attachment_active"), error=error),
        "grasp_representation": _string(
            source.get("grasp_representation"), error=error, identifier=True
        ),
    }


def _normalize_destination(value: Any, *, sample_index: int) -> dict[str, Any]:
    error = f"deformable_episode_destination_readback_invalid:{sample_index}"
    source = _clone_mapping(value, error=error)
    return {
        "pose_world": _normalize_pose(source.get("pose_world"), error=error),
        "linear_velocity_world_mps": _vector(
            source.get("linear_velocity_world_mps"), size=3, error=error
        ),
        "angular_velocity_world_radps": _vector(
            source.get("angular_velocity_world_radps"), size=3, error=error
        ),
        "state_write_count_after_episode_start": _integer(
            source.get("state_write_count_after_episode_start"), error=error
        ),
    }


def _normalize_robot(
    value: Any,
    *,
    entity_ids: Mapping[str, str],
    sample_index: int,
    joint_count: int,
) -> dict[str, Any]:
    error = f"deformable_episode_robot_readback_invalid:{sample_index}"
    source = _clone_mapping(value, error=error)
    contact_counts = _numeric_map(
        source.get("gripper_contact_pair_count_by_entity_id"), error=error, integer=True
    )
    contact_forces = _numeric_map(
        source.get("gripper_contact_normal_force_n_by_entity_id"), error=error
    )
    attachment_counts = _numeric_map(
        source.get("gripper_attachment_constraint_count_by_entity_id"),
        error=error,
        integer=True,
    )
    deformable_id = entity_ids["deformable"]
    if any(
        deformable_id not in values
        for values in (contact_counts, contact_forces, attachment_counts)
    ):
        _raise(f"deformable_episode_robot_deformable_contact_missing:{sample_index}")
    return {
        "arm_joint_positions_rad": _vector(
            source.get("arm_joint_positions_rad"), size=joint_count, error=error
        ),
        "arm_joint_velocities_rad_s": _vector(
            source.get("arm_joint_velocities_rad_s"), size=joint_count, error=error
        ),
        "gripper_width_m": _number(source.get("gripper_width_m"), error=error, minimum=0.0),
        "gripper_clearance_points_world_m": _matrix(
            source.get("gripper_clearance_points_world_m"), columns=3, error=error
        ),
        "gripper_contact_pair_count_by_entity_id": contact_counts,
        "gripper_contact_normal_force_n_by_entity_id": contact_forces,
        "gripper_attachment_constraint_count_by_entity_id": attachment_counts,
        "state_write_count_after_episode_start": _integer(
            source.get("state_write_count_after_episode_start"), error=error
        ),
    }


def _normalize_calibration(value: Any, *, camera_id: str, sample_index: int) -> dict[str, Any]:
    error = f"deformable_episode_camera_calibration_invalid:{sample_index}:{camera_id}"
    source = _canonical_record(value, digest_field="calibration_digest", error=error)
    intrinsics = _clone_mapping(source.get("intrinsics"), error=error)
    transform = _clone_mapping(source.get("transform_world_from_camera"), error=error)
    normalized = {
        "camera_id": camera_id,
        "transform_world_from_camera": {
            "position_m": _vector(transform.get("position_m"), size=3, error=error),
            "orientation_xyzw": _quaternion(transform.get("orientation_xyzw"), error=error),
        },
        "intrinsics": {
            "fx_px": _number(intrinsics.get("fx_px"), error=error, strictly_positive=True),
            "fy_px": _number(intrinsics.get("fy_px"), error=error, strictly_positive=True),
            "cx_px": _number(intrinsics.get("cx_px"), error=error, minimum=0.0),
            "cy_px": _number(intrinsics.get("cy_px"), error=error, minimum=0.0),
            "width_px": _integer(intrinsics.get("width_px"), error=error, minimum=1),
            "height_px": _integer(intrinsics.get("height_px"), error=error, minimum=1),
        },
    }
    normalized["calibration_digest"] = _canonical_digest(normalized)
    if normalized["calibration_digest"] != source["calibration_digest"]:
        _raise(error)
    return normalized


def _normalize_frame(
    value: Any,
    *,
    camera_id: str,
    sample_index: int,
    sample_timestamp_ns: int,
    simulation_time_s: float,
    actor_observation: bool,
    thresholds: Mapping[str, Any],
    evidence_root: Path,
) -> dict[str, Any]:
    error = f"deformable_episode_frame_invalid:{sample_index}:{camera_id}"
    source = _canonical_record(value, digest_field="frame_digest", error=error)
    if source.get("camera_id") != camera_id:
        _raise(error)
    timestamp_ns = _integer(source.get("timestamp_ns"), error=error)
    frame_simulation_time = _number(source.get("simulation_time_s"), error=error, minimum=0.0)
    if abs(timestamp_ns - sample_timestamp_ns) > thresholds["maximum_frame_sync_skew_ns"]:
        _raise(f"deformable_episode_frame_timestamp_unsynchronized:{sample_index}:{camera_id}")
    if (
        abs(frame_simulation_time - simulation_time_s)
        > thresholds["maximum_frame_simulation_time_skew_s"]
    ):
        _raise(
            f"deformable_episode_frame_simulation_time_unsynchronized:{sample_index}:{camera_id}"
        )

    calibration = _normalize_calibration(
        source.get("calibration"), camera_id=camera_id, sample_index=sample_index
    )
    width = _integer(source.get("width_px"), error=error, minimum=1)
    height = _integer(source.get("height_px"), error=error, minimum=1)
    if (
        calibration["intrinsics"]["width_px"] != width
        or calibration["intrinsics"]["height_px"] != height
    ):
        _raise(error)
    relative_path = _relative_artifact_path(source.get("relative_path"), error=error)

    expected_policy_eligible = camera_id in POLICY_CAMERA_IDS
    expected_presented = bool(actor_observation and expected_policy_eligible)
    expected_review_only = camera_id == OVERVIEW_CAMERA_ID
    if (
        _bool(source.get("policy_input_eligible"), error=error) is not expected_policy_eligible
        or _bool(source.get("presented_to_actor"), error=error) is not expected_presented
        or _bool(source.get("review_only"), error=error) is not expected_review_only
        or _bool(source.get("used_for_deterministic_scoring"), error=error)
    ):
        _raise(f"deformable_episode_camera_role_invalid:{sample_index}:{camera_id}")

    if source.get("encoding") != "png" or source.get("channels") != 3:
        _raise(error)
    payload = _artifact_bytes(
        evidence_root=evidence_root,
        relative_path=relative_path,
        maximum_bytes=MAX_FRAME_BYTES,
        error=f"deformable_episode_frame_file_invalid:{sample_index}:{camera_id}",
    )
    actual_file_digest = _bytes_digest(payload)
    if (
        _integer(source.get("size_bytes"), error=error) != len(payload)
        or _digest(source.get("lossless_file_sha256"), error=error) != actual_file_digest
    ):
        _raise(f"deformable_episode_frame_file_mismatch:{sample_index}:{camera_id}")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or image.size != (width, height)
                or width * height > MAX_FRAME_PIXELS
            ):
                _raise(f"deformable_episode_frame_png_invalid:{sample_index}:{camera_id}")
            image.load()
            raw_rgb = image.tobytes()
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise NativeDeformableEpisodeTraceError(
            [f"deformable_episode_frame_png_invalid:{sample_index}:{camera_id}"]
        ) from exc
    actual_raw_digest = _bytes_digest(raw_rgb)
    if (
        len(raw_rgb) != width * height * 3
        or _digest(source.get("raw_rgb_sha256"), error=error) != actual_raw_digest
    ):
        _raise(f"deformable_episode_frame_rgb_mismatch:{sample_index}:{camera_id}")

    normalized = {
        "camera_id": camera_id,
        "frame_sequence_index": _integer(source.get("frame_sequence_index"), error=error),
        "timestamp_ns": timestamp_ns,
        "simulation_time_s": frame_simulation_time,
        "policy_input_eligible": expected_policy_eligible,
        "presented_to_actor": expected_presented,
        "review_only": expected_review_only,
        "used_for_deterministic_scoring": False,
        "encoding": "png",
        "width_px": width,
        "height_px": height,
        "channels": 3,
        "relative_path": relative_path,
        "size_bytes": len(payload),
        "lossless_file_sha256": actual_file_digest,
        "raw_rgb_sha256": actual_raw_digest,
        "calibration": calibration,
        "renderer_identity_sha256": _digest(source.get("renderer_identity_sha256"), error=error),
    }
    normalized["frame_digest"] = _canonical_digest(normalized)
    if normalized["frame_digest"] != source["frame_digest"]:
        _raise(error)
    return normalized


def _normalize_policy_inference(
    value: Any,
    *,
    actor_identity_digest: str,
    source_output_sha256: str,
    episode_start_timestamp_ns: int,
    command_timestamp_ns: int,
    sample_index: int,
) -> dict[str, Any]:
    error = f"deformable_episode_policy_inference_invalid:{sample_index}"
    source = _canonical_record(value, digest_field="receipt_digest", error=error)
    frame_digests = _nonempty_digest_map(
        source.get("policy_input_frame_digest_by_camera_id"), error=error
    )
    if set(frame_digests) != set(POLICY_CAMERA_IDS):
        _raise(error)
    started = _integer(source.get("inference_started_timestamp_ns"), error=error)
    completed = _integer(source.get("inference_completed_timestamp_ns"), error=error)
    if not episode_start_timestamp_ns <= started <= completed <= command_timestamp_ns:
        _raise(f"deformable_episode_policy_inference_timestamp_order_invalid:{sample_index}")
    normalized = {
        "actor_identity_digest": _digest(source.get("actor_identity_digest"), error=error),
        "policy_input_frame_digest_by_camera_id": frame_digests,
        "inference_started_timestamp_ns": started,
        "inference_completed_timestamp_ns": completed,
        "source_output_sha256": _digest(source.get("source_output_sha256"), error=error),
    }
    if (
        normalized["actor_identity_digest"] != actor_identity_digest
        or normalized["source_output_sha256"] != source_output_sha256
    ):
        _raise(f"deformable_episode_policy_inference_identity_mismatch:{sample_index}")
    normalized["receipt_digest"] = _canonical_digest(normalized)
    if normalized["receipt_digest"] != source["receipt_digest"]:
        _raise(error)
    return normalized


def _normalize_action(
    value: Any,
    *,
    episode_kind: str,
    actor: Mapping[str, Any],
    sample_index: int,
    joint_count: int,
    episode_start_timestamp_ns: int,
    sample_timestamp_ns: int,
    frozen_run: Mapping[str, Any],
) -> dict[str, Any]:
    error = f"deformable_episode_action_invalid:{sample_index}"
    source = _canonical_record(value, digest_field="action_digest", error=error)
    delivery = _canonical_record(source.get("delivery"), digest_field="receipt_digest", error=error)
    command_timestamp = _integer(source.get("command_timestamp_ns"), error=error)
    delivery_timestamp = _integer(delivery.get("delivery_timestamp_ns"), error=error)
    if not (
        episode_start_timestamp_ns <= command_timestamp <= delivery_timestamp <= sample_timestamp_ns
    ):
        _raise(f"deformable_episode_action_timestamp_order_invalid:{sample_index}")

    origin_kind = _string(source.get("origin_kind"), error=error, identifier=True)
    allowed_origins = {
        "zero_action_control": {"zero_action_control"},
        "scripted_positive_control": {"scripted_control"},
        "learned_policy_evaluation": {"learned_policy", "harness_settle"},
    }[episode_kind]
    if origin_kind not in allowed_origins:
        _raise(f"deformable_episode_action_origin_invalid:{sample_index}")
    source_output = _vector(source.get("source_output"), size=None, error=error)
    source_output_sha256 = _digest(source.get("source_output_sha256"), error=error)
    if source_output_sha256 != _bytes_digest(_canonical_json_bytes(source_output)):
        _raise(f"deformable_episode_source_output_digest_mismatch:{sample_index}")
    if episode_kind == "learned_policy_evaluation":
        adapter_sha256 = actor["action_adapter_sha256"]
    else:
        adapter_sha256 = actor["controller_sha256"]
    if delivery.get("adapter_sha256") != adapter_sha256:
        _raise(f"deformable_episode_action_adapter_mismatch:{sample_index}")

    arm_command = _vector(source.get("arm_command"), size=joint_count, error=error)
    gripper_delta = _number(source.get("gripper_delta_command_m"), error=error)
    native_action = _vector(source.get("native_action"), size=None, error=error)
    actor_identity_digest = actor["identity_digest"]
    replay_contract = frozen_run["action_replay_contract_by_actor_identity_digest"].get(
        actor_identity_digest
    )
    if not isinstance(replay_contract, Mapping):
        _raise(f"deformable_episode_action_replay_contract_mismatch:{sample_index}")
    command_space = _string(source.get("command_space"), error=error, identifier=True)
    if replay_contract["command_space"] != command_space:
        _raise(f"deformable_episode_action_command_space_mismatch:{sample_index}")
    if (
        len(source_output) != replay_contract["source_output_size"]
        or len(replay_contract["arm_source_indices"]) != joint_count
    ):
        _raise(f"deformable_episode_action_adapter_replay_shape_mismatch:{sample_index}")
    if origin_kind == "harness_settle":
        if any(item != 0.0 for item in source_output):
            _raise(f"deformable_episode_harness_settle_action_nonzero:{sample_index}")
        replayed_arm = [0.0] * joint_count
        replayed_gripper = 0.0
    else:
        replayed_arm = [
            source_output[source_index] * scale + offset
            for source_index, scale, offset in zip(
                replay_contract["arm_source_indices"],
                replay_contract["arm_scale"],
                replay_contract["arm_offset"],
                strict=True,
            )
        ]
        replayed_gripper = (
            source_output[replay_contract["gripper_source_index"]]
            * replay_contract["gripper_scale"]
            + replay_contract["gripper_offset"]
        )
    replayed_action = [*replayed_arm, replayed_gripper]
    if (
        arm_command != replayed_arm
        or gripper_delta != replayed_gripper
        or native_action != replayed_action
    ):
        _raise(f"deformable_episode_action_adapter_replay_mismatch:{sample_index}")
    if episode_kind == "learned_policy_evaluation" and origin_kind == "learned_policy":
        if source.get("policy_inference") is None:
            _raise(f"deformable_episode_policy_inference_missing:{sample_index}")
        policy_inference = _normalize_policy_inference(
            source.get("policy_inference"),
            actor_identity_digest=actor_identity_digest,
            source_output_sha256=source_output_sha256,
            episode_start_timestamp_ns=episode_start_timestamp_ns,
            command_timestamp_ns=command_timestamp,
            sample_index=sample_index,
        )
    else:
        if source.get("policy_inference") is not None:
            _raise(f"deformable_episode_policy_inference_forbidden:{sample_index}")
        policy_inference = None
    native_action_digest = _bytes_digest(_canonical_json_bytes(native_action))
    normalized_delivery = {
        "attempted": _bool(delivery.get("attempted"), error=error),
        "delivered_to_robot": _bool(delivery.get("delivered_to_robot"), error=error),
        "delivery_timestamp_ns": delivery_timestamp,
        "native_action_sha256": _digest(delivery.get("native_action_sha256"), error=error),
        "adapter_sha256": adapter_sha256,
    }
    if normalized_delivery["native_action_sha256"] != native_action_digest:
        _raise(f"deformable_episode_native_action_digest_mismatch:{sample_index}")
    if normalized_delivery["delivered_to_robot"] and not normalized_delivery["attempted"]:
        _raise(f"deformable_episode_action_delivery_state_invalid:{sample_index}")
    normalized_delivery["receipt_digest"] = _canonical_digest(normalized_delivery)
    if normalized_delivery["receipt_digest"] != delivery["receipt_digest"]:
        _raise(error)

    normalized = {
        "action_index": _integer(source.get("action_index"), error=error),
        "origin_kind": origin_kind,
        "command_space": command_space,
        "command_timestamp_ns": command_timestamp,
        "arm_command": arm_command,
        "gripper_delta_command_m": gripper_delta,
        "native_action": native_action,
        "source_output": source_output,
        "source_output_sha256": source_output_sha256,
        "policy_inference": policy_inference,
        "delivery": normalized_delivery,
    }
    normalized["action_digest"] = _canonical_digest(normalized)
    if normalized["action_digest"] != source["action_digest"]:
        _raise(error)
    return normalized


def _normalize_steps(
    value: Any,
    *,
    episode_kind: str,
    actor: Mapping[str, Any],
    entity_ids: Mapping[str, str],
    all_entity_ids: Sequence[str],
    reset: Mapping[str, Any],
    episode_start_timestamp_ns: int,
    thresholds: Mapping[str, Any],
    evidence_root: Path,
    frozen_run: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        _raise("deformable_episode_steps_invalid")
    if len(value) > MAX_STEPS:
        _raise("deformable_episode_steps_limit_exceeded")
    joint_count = len(reset["robot_joint_positions_rad"])
    normalized: list[dict[str, Any]] = []
    previous_timestamp = episode_start_timestamp_ns - 1
    previous_simulation_time = -1.0
    calibration_digest_by_camera: dict[str, str] = {}
    frame_sequence_by_camera: dict[str, int] = {camera_id: -1 for camera_id in CAMERA_IDS}
    observed_frame_paths: set[str] = set()
    nodal_count: int | None = None
    gradient_count: int | None = None

    for sequence_index, raw_step in enumerate(value):
        source = _clone_mapping(raw_step, error=f"deformable_episode_step_invalid:{sequence_index}")
        sample_index = _integer(
            source.get("sample_index"),
            error=f"deformable_episode_sample_index_invalid:{sequence_index}",
        )
        if sample_index != sequence_index:
            _raise("deformable_episode_sample_indices_not_contiguous")
        timestamp_ns = _integer(
            source.get("timestamp_ns"),
            error=f"deformable_episode_sample_timestamp_invalid:{sample_index}",
        )
        simulation_time = _number(
            source.get("simulation_time_s"),
            error=f"deformable_episode_simulation_time_invalid:{sample_index}",
            minimum=0.0,
        )
        if timestamp_ns <= previous_timestamp:
            _raise("deformable_episode_sample_timestamps_not_increasing")
        if simulation_time <= previous_simulation_time:
            _raise("deformable_episode_simulation_times_not_increasing")
        previous_timestamp = timestamp_ns
        previous_simulation_time = simulation_time

        observation_kind = _string(
            source.get("observation_kind"),
            error=f"deformable_episode_observation_kind_invalid:{sample_index}",
            identifier=True,
        )
        actor_observation = _bool(
            source.get("actor_observation"),
            error=f"deformable_episode_actor_observation_invalid:{sample_index}",
        )
        if observation_kind not in {"actor_input", "control_sample", "review_sample", "terminal"}:
            _raise(f"deformable_episode_observation_kind_invalid:{sample_index}")
        if actor_observation != (
            episode_kind == "learned_policy_evaluation" and observation_kind == "actor_input"
        ):
            _raise(f"deformable_episode_actor_observation_role_invalid:{sample_index}")

        entities = source.get("entities")
        if not isinstance(entities, Mapping):
            _raise(f"deformable_episode_entities_invalid:{sample_index}")
        if set(entities) != set(entity_ids.values()):
            _raise(f"deformable_episode_scored_entity_set_mismatch:{sample_index}")
        state_digests_source = source.get("native_state_sha256_by_entity_id")
        if not isinstance(state_digests_source, Mapping) or set(state_digests_source) != set(
            all_entity_ids
        ):
            _raise(f"deformable_episode_native_state_entity_set_mismatch:{sample_index}")
        state_digests = {
            entity_id: _digest(
                state_digests_source[entity_id],
                error=(
                    f"deformable_episode_native_state_digest_invalid:{sample_index}:{entity_id}"
                ),
            )
            for entity_id in sorted(all_entity_ids)
        }
        write_counts_source = source.get("state_write_count_after_episode_start_by_entity_id")
        if not isinstance(write_counts_source, Mapping) or set(write_counts_source) != set(
            all_entity_ids
        ):
            _raise(f"deformable_episode_state_write_entity_set_mismatch:{sample_index}")
        write_counts = {
            entity_id: _integer(
                write_counts_source[entity_id],
                error=(f"deformable_episode_state_write_count_invalid:{sample_index}:{entity_id}"),
            )
            for entity_id in sorted(all_entity_ids)
        }
        deformable = _normalize_deformable(
            entities.get(entity_ids["deformable"]),
            entity_ids=entity_ids,
            sample_index=sample_index,
        )
        destination = _normalize_destination(
            entities.get(entity_ids["destination"]), sample_index=sample_index
        )
        robot = _normalize_robot(
            entities.get(entity_ids["robot"]),
            entity_ids=entity_ids,
            sample_index=sample_index,
            joint_count=joint_count,
        )
        normalized_scored_states = {
            entity_ids["deformable"]: deformable,
            entity_ids["destination"]: destination,
            entity_ids["robot"]: robot,
        }
        for entity_id, normalized_state in normalized_scored_states.items():
            if state_digests[entity_id] != _canonical_digest(normalized_state):
                _raise(
                    f"deformable_episode_native_state_digest_mismatch:{sample_index}:{entity_id}"
                )
            if write_counts[entity_id] != normalized_state["state_write_count_after_episode_start"]:
                _raise(f"deformable_episode_state_write_count_mismatch:{sample_index}:{entity_id}")
        robot_pairs = robot["gripper_contact_pair_count_by_entity_id"][entity_ids["deformable"]]
        deformable_pairs = deformable["contact_pair_count_by_entity_id"][entity_ids["robot"]]
        robot_force = robot["gripper_contact_normal_force_n_by_entity_id"][entity_ids["deformable"]]
        deformable_force = deformable["contact_normal_force_n_by_entity_id"][entity_ids["robot"]]
        if robot_pairs != deformable_pairs or not math.isclose(
            robot_force, deformable_force, rel_tol=1.0e-6, abs_tol=1.0e-9
        ):
            _raise(f"deformable_episode_bilateral_contact_mismatch:{sample_index}")
        if len(deformable["nodal_positions_world_m"]) != len(
            reset["deformable_nodal_positions_world_m"]
        ):
            _raise("deformable_episode_reset_native_tensor_shape_mismatch")
        if nodal_count is None:
            nodal_count = len(deformable["nodal_positions_world_m"])
            gradient_count = len(deformable["deformation_gradients"])
        elif (
            len(deformable["nodal_positions_world_m"]) != nodal_count
            or len(deformable["deformation_gradients"]) != gradient_count
        ):
            _raise("deformable_episode_native_tensor_shape_changed")

        action = _normalize_action(
            source.get("action"),
            episode_kind=episode_kind,
            actor=actor,
            sample_index=sample_index,
            joint_count=joint_count,
            episode_start_timestamp_ns=episode_start_timestamp_ns,
            sample_timestamp_ns=timestamp_ns,
            frozen_run=frozen_run,
        )
        if action["action_index"] != sample_index:
            _raise("deformable_episode_action_indices_not_contiguous")

        frames_source = source.get("frames")
        if not isinstance(frames_source, Mapping) or set(frames_source) != set(CAMERA_IDS):
            _raise(f"deformable_episode_required_frames_missing:{sample_index}")
        frames: dict[str, Any] = {}
        for camera_id in CAMERA_IDS:
            frame = _normalize_frame(
                frames_source[camera_id],
                camera_id=camera_id,
                sample_index=sample_index,
                sample_timestamp_ns=timestamp_ns,
                simulation_time_s=simulation_time,
                actor_observation=actor_observation,
                thresholds=thresholds,
                evidence_root=evidence_root,
            )
            expected_sequence = frame_sequence_by_camera[camera_id] + 1
            if frame["frame_sequence_index"] != expected_sequence:
                _raise(f"deformable_episode_frame_sequence_invalid:{sample_index}:{camera_id}")
            frame_sequence_by_camera[camera_id] = expected_sequence
            calibration_digest = frame["calibration"]["calibration_digest"]
            prior_calibration = calibration_digest_by_camera.setdefault(
                camera_id, calibration_digest
            )
            if calibration_digest != prior_calibration:
                _raise(f"deformable_episode_camera_calibration_changed:{camera_id}")
            if (
                frozen_run["camera_calibration_digest_by_camera_id"].get(camera_id)
                != calibration_digest
                or frozen_run["renderer_identity_sha256_by_camera_id"].get(camera_id)
                != frame["renderer_identity_sha256"]
            ):
                _raise(f"deformable_episode_frozen_camera_contract_mismatch:{camera_id}")
            if frame["relative_path"] in observed_frame_paths:
                _raise(f"deformable_episode_frame_path_reused:{frame['relative_path']}")
            observed_frame_paths.add(frame["relative_path"])
            frames[camera_id] = frame
        frame_timestamps = [frame["timestamp_ns"] for frame in frames.values()]
        if max(frame_timestamps) - min(frame_timestamps) > thresholds["maximum_frame_sync_skew_ns"]:
            _raise(f"deformable_episode_camera_group_unsynchronized:{sample_index}")
        if episode_kind == "learned_policy_evaluation":
            expected_origin = "learned_policy" if actor_observation else "harness_settle"
            if action["origin_kind"] != expected_origin:
                _raise(f"deformable_episode_policy_action_observation_join_invalid:{sample_index}")
            inference = action["policy_inference"]
            if actor_observation:
                if not isinstance(inference, Mapping):
                    _raise(f"deformable_episode_policy_inference_missing:{sample_index}")
                expected_frame_digests = {
                    camera_id: frames[camera_id]["frame_digest"]
                    for camera_id in sorted(POLICY_CAMERA_IDS)
                }
                if inference["policy_input_frame_digest_by_camera_id"] != expected_frame_digests:
                    _raise(f"deformable_episode_policy_input_frame_join_mismatch:{sample_index}")
                latest_policy_frame_timestamp = max(
                    frames[camera_id]["timestamp_ns"] for camera_id in POLICY_CAMERA_IDS
                )
                if latest_policy_frame_timestamp > inference["inference_started_timestamp_ns"]:
                    _raise(f"deformable_episode_policy_inference_precedes_input:{sample_index}")
            elif inference is not None:
                _raise(f"deformable_episode_harness_settle_inference_forbidden:{sample_index}")

        normalized.append(
            {
                "sample_index": sample_index,
                "timestamp_ns": timestamp_ns,
                "simulation_time_s": simulation_time,
                "observation_kind": observation_kind,
                "actor_observation": actor_observation,
                "action": action,
                "native_state_sha256_by_entity_id": state_digests,
                "state_write_count_after_episode_start_by_entity_id": write_counts,
                "entities": {
                    entity_ids["deformable"]: deformable,
                    entity_ids["destination"]: destination,
                    entity_ids["robot"]: robot,
                },
                "frames": frames,
                "observation_digest": "",
            }
        )
        normalized[-1]["observation_digest"] = _canonical_digest(
            normalized[-1], digest_field="observation_digest"
        )
        if source.get("observation_digest") != normalized[-1]["observation_digest"]:
            _raise(f"deformable_episode_observation_digest_mismatch:{sample_index}")
    return normalized


def _normalize_terminal_gap(
    value: Any,
    *,
    episode_start_timestamp_ns: int,
    evidence_root: Path,
) -> dict[str, Any]:
    source = _clone_mapping(value, error="deformable_episode_terminal_invalid")
    if source.get("status") != "failed_before_first_observation":
        _raise("deformable_episode_terminal_status_invalid")
    timestamp_ns = _integer(
        source.get("terminal_timestamp_ns"),
        error="deformable_episode_terminal_timestamp_invalid",
    )
    if timestamp_ns < episode_start_timestamp_ns:
        _raise("deformable_episode_terminal_timestamp_invalid")
    gap = _canonical_record(
        source.get("media_gap"),
        digest_field="gap_receipt_digest",
        error="deformable_episode_terminal_media_gap_invalid",
    )
    failure_log_relative_path = _relative_artifact_path(
        gap.get("failure_log_relative_path"),
        error="deformable_episode_terminal_media_gap_invalid",
    )
    failure_log = _artifact_bytes(
        evidence_root=evidence_root,
        relative_path=failure_log_relative_path,
        maximum_bytes=MAX_NATIVE_TRACE_BYTES,
        error="deformable_episode_terminal_failure_log_invalid",
    )
    normalized_gap = {
        "gap_type": _string(
            gap.get("gap_type"), error="deformable_episode_terminal_media_gap_invalid"
        ),
        "required_camera_ids": sorted(
            _string(item, error="deformable_episode_terminal_media_gap_invalid")
            for item in gap.get("required_camera_ids", [])
        ),
        "observation_count": _integer(
            gap.get("observation_count"),
            error="deformable_episode_terminal_media_gap_invalid",
        ),
        "first_observation_attempted": _bool(
            gap.get("first_observation_attempted"),
            error="deformable_episode_terminal_media_gap_invalid",
        ),
        "failure_log_relative_path": failure_log_relative_path,
        "failure_log_size_bytes": _integer(
            gap.get("failure_log_size_bytes"),
            error="deformable_episode_terminal_media_gap_invalid",
            minimum=1,
        ),
        "failure_log_sha256": _digest(
            gap.get("failure_log_sha256"),
            error="deformable_episode_terminal_media_gap_invalid",
        ),
    }
    if (
        normalized_gap["gap_type"] != "no_frames_before_first_observation"
        or normalized_gap["required_camera_ids"] != sorted(CAMERA_IDS)
        or normalized_gap["observation_count"] != 0
        or normalized_gap["failure_log_size_bytes"] != len(failure_log)
        or normalized_gap["failure_log_sha256"] != _bytes_digest(failure_log)
    ):
        _raise("deformable_episode_terminal_media_gap_invalid")
    normalized_gap["gap_receipt_digest"] = _canonical_digest(normalized_gap)
    if normalized_gap["gap_receipt_digest"] != gap["gap_receipt_digest"]:
        _raise("deformable_episode_terminal_media_gap_invalid")
    return {
        "status": "failed_before_first_observation",
        "terminal_timestamp_ns": timestamp_ns,
        "failure_type": _string(
            source.get("failure_type"),
            error="deformable_episode_terminal_failure_type_invalid",
            identifier=True,
        ),
        "failure_stage": _string(
            source.get("failure_stage"),
            error="deformable_episode_terminal_failure_stage_invalid",
            identifier=True,
        ),
        "media_gap": normalized_gap,
    }


def _normalize_complete_terminal(
    value: Any, *, steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    source = _clone_mapping(value, error="deformable_episode_terminal_invalid")
    if source.get("status") != "complete" or not steps:
        _raise("deformable_episode_terminal_status_invalid")
    terminal_index = _integer(
        source.get("terminal_step_index"),
        error="deformable_episode_terminal_step_invalid",
    )
    if terminal_index != steps[-1]["sample_index"] or steps[-1]["observation_kind"] != "terminal":
        _raise("deformable_episode_terminal_step_invalid")
    if steps[-1]["actor_observation"]:
        _raise("deformable_episode_terminal_as_actor_input_forbidden")
    timestamp_ns = _integer(
        source.get("terminal_timestamp_ns"),
        error="deformable_episode_terminal_timestamp_invalid",
    )
    if timestamp_ns < steps[-1]["timestamp_ns"]:
        _raise("deformable_episode_terminal_timestamp_invalid")
    if source.get("media_gap") is not None:
        _raise("deformable_episode_completed_media_gap_forbidden")
    terminal_observation_digest = _digest(
        source.get("terminal_observation_digest"),
        error="deformable_episode_terminal_observation_digest_invalid",
    )
    if terminal_observation_digest != steps[-1]["observation_digest"]:
        _raise("deformable_episode_terminal_observation_digest_mismatch")
    return {
        "status": "complete",
        "terminal_step_index": terminal_index,
        "terminal_timestamp_ns": timestamp_ns,
        "terminal_observation_digest": terminal_observation_digest,
        "media_gap": None,
    }


def _normalize_manifest_file(
    value: Any,
    *,
    schema_version: str,
    expected_payload: bytes,
    row_count: int,
    rows_sha256: str,
    evidence_root: Path,
    error: str,
) -> dict[str, Any]:
    source = _canonical_record(value, digest_field="manifest_digest", error=error)
    if source.get("schema_version") != schema_version:
        _raise(error)
    relative_path = _relative_artifact_path(source.get("relative_path"), error=error)
    payload = _artifact_bytes(
        evidence_root=evidence_root,
        relative_path=relative_path,
        maximum_bytes=MAX_NATIVE_TRACE_BYTES,
        error=f"{error}_file_invalid",
    )
    actual_digest = _bytes_digest(payload)
    if payload != expected_payload:
        _raise(f"{error}_content_mismatch")
    normalized = {
        "schema_version": schema_version,
        "relative_path": relative_path,
        "file_size_bytes": _integer(source.get("file_size_bytes"), error=error, minimum=1),
        "file_sha256": _digest(source.get("file_sha256"), error=error),
        "row_count": _integer(source.get("row_count"), error=error),
        "rows_sha256": _digest(source.get("rows_sha256"), error=error),
    }
    if (
        normalized["file_size_bytes"] != len(payload)
        or normalized["file_sha256"] != actual_digest
        or normalized["row_count"] != row_count
        or normalized["rows_sha256"] != rows_sha256
    ):
        _raise(f"{error}_receipt_mismatch")
    normalized["manifest_digest"] = _canonical_digest(normalized)
    if normalized["manifest_digest"] != source["manifest_digest"]:
        _raise(f"{error}_projection_mismatch")
    return normalized


def _media_manifest_document(
    *, episode_id: str, steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    frames_by_camera: dict[str, list[dict[str, Any]]] = {}
    for camera_id in CAMERA_IDS:
        rows: list[dict[str, Any]] = []
        for step in steps:
            frame = step["frames"][camera_id]
            rows.append(
                {
                    "sample_index": step["sample_index"],
                    "frame_sequence_index": frame["frame_sequence_index"],
                    "timestamp_ns": frame["timestamp_ns"],
                    "simulation_time_s": frame["simulation_time_s"],
                    "relative_path": frame["relative_path"],
                    "size_bytes": frame["size_bytes"],
                    "lossless_file_sha256": frame["lossless_file_sha256"],
                    "raw_rgb_sha256": frame["raw_rgb_sha256"],
                    "calibration_digest": frame["calibration"]["calibration_digest"],
                    "renderer_identity_sha256": frame["renderer_identity_sha256"],
                    "presented_to_actor": frame["presented_to_actor"],
                    "review_only": frame["review_only"],
                    "used_for_deterministic_scoring": False,
                    "frame_digest": frame["frame_digest"],
                }
            )
        frames_by_camera[camera_id] = rows
    return {
        "schema_version": MEDIA_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "frames_by_camera": frames_by_camera,
    }


def _normalize_media_manifest(
    value: Any,
    *,
    episode_id: str,
    steps: Sequence[Mapping[str, Any]],
    evidence_root: Path,
) -> dict[str, Any]:
    document = _media_manifest_document(episode_id=episode_id, steps=steps)
    frames_by_camera = document["frames_by_camera"]
    expected_payload = _canonical_json_bytes(document) + b"\n"
    return _normalize_manifest_file(
        value,
        schema_version=MEDIA_MANIFEST_SCHEMA_VERSION,
        expected_payload=expected_payload,
        row_count=sum(len(rows) for rows in frames_by_camera.values()),
        rows_sha256=_bytes_digest(_canonical_json_bytes(frames_by_camera)),
        evidence_root=evidence_root,
        error="deformable_episode_media_manifest_invalid",
    )


def _mp4_boxes(payload: bytes, *, offset: int = 0) -> list[tuple[bytes, bytes]] | None:
    boxes: list[tuple[bytes, bytes]] = []
    while offset < len(payload):
        if len(payload) - offset < 8:
            return None
        size = int.from_bytes(payload[offset : offset + 4], "big")
        box_type = payload[offset + 4 : offset + 8]
        header_size = 8
        if size == 1:
            if len(payload) - offset < 16:
                return None
            size = int.from_bytes(payload[offset + 8 : offset + 16], "big")
            header_size = 16
        if size < header_size or offset + size > len(payload):
            return None
        boxes.append((box_type, payload[offset + header_size : offset + size]))
        offset += size
    return boxes


def _sole_box(payload: bytes, box_type: bytes, *, offset: int = 0) -> bytes | None:
    boxes = _mp4_boxes(payload, offset=offset)
    if boxes is None:
        return None
    matches = [box_payload for observed_type, box_payload in boxes if observed_type == box_type]
    return matches[0] if len(matches) == 1 else None


def _avcc_has_parameter_sets(payload: bytes) -> bool:
    if len(payload) < 7 or payload[0] != 1 or payload[1] == 0:
        return False
    sps_count = payload[5] & 0x1F
    if sps_count < 1:
        return False
    offset = 6
    for _ in range(sps_count):
        if len(payload) - offset < 2:
            return False
        length = int.from_bytes(payload[offset : offset + 2], "big")
        offset += 2
        if length < 1 or len(payload) - offset < length:
            return False
        offset += length
    if len(payload) - offset < 1:
        return False
    pps_count = payload[offset]
    offset += 1
    if pps_count < 1:
        return False
    for _ in range(pps_count):
        if len(payload) - offset < 2:
            return False
        length = int.from_bytes(payload[offset : offset + 2], "big")
        offset += 2
        if length < 1 or len(payload) - offset < length:
            return False
        offset += length
    return offset <= len(payload)


def _mp4_h264_sample_count(payload: bytes) -> int | None:
    """Validate the AVC sample entry, parameter sets, and non-empty sample table."""

    top = _mp4_boxes(payload)
    if top is None:
        return None

    def unique_top(box_type: bytes) -> bytes | None:
        matches = [box_payload for observed, box_payload in top if observed == box_type]
        return matches[0] if len(matches) == 1 else None

    ftyp = unique_top(b"ftyp")
    moov = unique_top(b"moov")
    mdat = unique_top(b"mdat")
    if not ftyp or not moov or not mdat or len(ftyp) < 8:
        return None
    trak = _sole_box(moov, b"trak")
    mdia = _sole_box(trak, b"mdia") if trak is not None else None
    minf = _sole_box(mdia, b"minf") if mdia is not None else None
    stbl = _sole_box(minf, b"stbl") if minf is not None else None
    stsd = _sole_box(stbl, b"stsd") if stbl is not None else None
    stsz = _sole_box(stbl, b"stsz") if stbl is not None else None
    if stsd is None or stsz is None or len(stsd) < 8 or len(stsz) < 12:
        return None
    if int.from_bytes(stsd[4:8], "big") != 1:
        return None
    avc1 = _sole_box(stsd, b"avc1", offset=8)
    if avc1 is None or len(avc1) < 78:
        return None
    avcc = _sole_box(avc1, b"avcC", offset=78)
    if avcc is None or not _avcc_has_parameter_sets(avcc):
        return None
    sample_count = int.from_bytes(stsz[8:12], "big")
    chunk_offsets_present = bool(
        _sole_box(stbl, b"stco") is not None or _sole_box(stbl, b"co64") is not None
    )
    return sample_count if sample_count > 0 and chunk_offsets_present else None


def _verified_h264_decode(payload: bytes) -> dict[str, Any] | None:
    """Decode verifier-owned bytes and bind the exact RGB sample sequence."""

    def fresh(value: _H264DecodeCacheValue | None) -> dict[str, Any] | None:
        if value is None:
            return None
        decoded_sample_count, width_px, height_px, raw_rgb_digests = value
        return {
            "decoded_sample_count": decoded_sample_count,
            "width_px": width_px,
            "height_px": height_px,
            "raw_rgb_sha256_by_sample": list(raw_rgb_digests),
        }

    structural_count = _mp4_h264_sample_count(payload)
    if structural_count is None:
        return None
    payload_digest = _bytes_digest(payload)
    if payload_digest in _H264_DECODE_CACHE:
        return fresh(_H264_DECODE_CACHE[payload_digest])
    decoded_evidence: dict[str, Any] | None = None
    try:
        import cv2  # Lazy import keeps non-media callers simulator-runtime independent.
    except ImportError:
        cv2 = None
    if cv2 is not None:
        try:
            with tempfile.TemporaryDirectory(prefix="blueprint-h264-verify-") as temporary:
                snapshot = Path(temporary) / "review.mp4"
                snapshot.write_bytes(payload)
                capture = cv2.VideoCapture(
                    str(snapshot),
                    cv2.CAP_FFMPEG,
                    [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                        5_000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                        5_000,
                    ],
                )
                try:
                    if not capture.isOpened():
                        return None
                    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
                    fourcc = "".join(
                        chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
                    ).lower()
                    if fourcc not in {"avc1", "avc3", "h264"}:
                        return None
                    observed = 0
                    dimensions: tuple[int, int] | None = None
                    raw_rgb_sha256_by_sample: list[str] = []
                    while observed <= structural_count:
                        decoded, frame = capture.read()
                        if not decoded:
                            break
                        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                            return None
                        current_dimensions = (int(frame.shape[1]), int(frame.shape[0]))
                        if min(current_dimensions) < 1:
                            return None
                        if dimensions is None:
                            dimensions = current_dimensions
                        elif dimensions != current_dimensions:
                            return None
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        raw_rgb_sha256_by_sample.append(_bytes_digest(rgb.tobytes()))
                        observed += 1
                    if observed == structural_count and dimensions is not None:
                        decoded_evidence = {
                            "decoded_sample_count": observed,
                            "width_px": dimensions[0],
                            "height_px": dimensions[1],
                            "raw_rgb_sha256_by_sample": raw_rgb_sha256_by_sample,
                        }
                finally:
                    capture.release()
        except (cv2.error, OSError, ValueError):
            decoded_evidence = None
    if len(_H264_DECODE_CACHE) >= 64:
        _H264_DECODE_CACHE.clear()
    immutable_evidence = (
        None
        if decoded_evidence is None
        else (
            decoded_evidence["decoded_sample_count"],
            decoded_evidence["width_px"],
            decoded_evidence["height_px"],
            tuple(decoded_evidence["raw_rgb_sha256_by_sample"]),
        )
    )
    _H264_DECODE_CACHE[payload_digest] = immutable_evidence
    return fresh(immutable_evidence)


def _normalize_review_videos(
    value: Any,
    *,
    steps: Sequence[Mapping[str, Any]],
    media_manifest: Mapping[str, Any],
    frozen_run: Mapping[str, Any],
    evidence_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(CAMERA_IDS):
        _raise("deformable_episode_review_videos_invalid")
    normalized: dict[str, Any] = {}
    for camera_id in CAMERA_IDS:
        error = f"deformable_episode_review_video_invalid:{camera_id}"
        source = _canonical_record(
            value[camera_id], digest_field="video_receipt_digest", error=error
        )
        if source.get("camera_id") != camera_id:
            _raise(error)
        relative_path = _relative_artifact_path(source.get("relative_path"), error=error)
        payload = _artifact_bytes(
            evidence_root=evidence_root,
            relative_path=relative_path,
            maximum_bytes=MAX_REVIEW_VIDEO_BYTES,
            error=f"{error}_file",
        )
        decoded_video = _verified_h264_decode(payload)
        if decoded_video is None:
            _raise(f"{error}_h264_structure")
        frame_digests = [step["frames"][camera_id]["frame_digest"] for step in steps]
        raw_rgb_digests = [step["frames"][camera_id]["raw_rgb_sha256"] for step in steps]
        frame_dimensions = {
            (
                step["frames"][camera_id]["width_px"],
                step["frames"][camera_id]["height_px"],
            )
            for step in steps
        }
        if len(frame_dimensions) != 1:
            _raise(f"{error}_lossless_frame_dimensions_changed")
        expected_width, expected_height = next(iter(frame_dimensions))
        expected_sequence_digest = _bytes_digest(_canonical_json_bytes(frame_digests))
        codec = _string(source.get("codec"), error=error, identifier=True)
        container = _string(source.get("container"), error=error, identifier=True)
        if (
            codec != frozen_run["review_video_codec_by_camera_id"][camera_id]
            or container != frozen_run["review_video_container_by_camera_id"][camera_id]
        ):
            _raise(f"{error}_frozen_contract_mismatch")
        normalized_video = {
            "camera_id": camera_id,
            "relative_path": relative_path,
            "size_bytes": _integer(source.get("size_bytes"), error=error, minimum=1),
            "file_sha256": _digest(source.get("file_sha256"), error=error),
            "container": container,
            "codec": codec,
            "source_media_manifest_digest": _digest(
                source.get("source_media_manifest_digest"), error=error
            ),
            "source_frame_digest_sequence_sha256": _digest(
                source.get("source_frame_digest_sequence_sha256"), error=error
            ),
            "source_frame_count": _integer(
                source.get("source_frame_count"), error=error, minimum=1
            ),
            "derivation_tool_id": _string(
                source.get("derivation_tool_id"), error=error, identifier=True
            ),
            "derivation_tool_sha256": _digest(source.get("derivation_tool_sha256"), error=error),
            "derivation_command_sha256": _digest(
                source.get("derivation_command_sha256"), error=error
            ),
        }
        if (
            normalized_video["size_bytes"] != len(payload)
            or normalized_video["file_sha256"] != _bytes_digest(payload)
            or normalized_video["source_media_manifest_digest"] != media_manifest["manifest_digest"]
            or normalized_video["source_frame_digest_sequence_sha256"] != expected_sequence_digest
            or normalized_video["source_frame_count"] != len(frame_digests)
            or decoded_video["decoded_sample_count"] != len(frame_digests)
            or decoded_video["width_px"] != expected_width
            or decoded_video["height_px"] != expected_height
            or decoded_video["raw_rgb_sha256_by_sample"] != raw_rgb_digests
        ):
            _raise(f"{error}_derivation_join_mismatch")
        normalized_video["video_receipt_digest"] = _canonical_digest(normalized_video)
        if normalized_video["video_receipt_digest"] != source["video_receipt_digest"]:
            _raise(f"{error}_projection_mismatch")
        normalized_video["verifier_owned_decode"] = {
            "contract_id": H264_DECODE_VERIFIER_CONTRACT_ID,
            **decoded_video,
            "decoded_raw_rgb_sequence_sha256": _bytes_digest(
                _canonical_json_bytes(raw_rgb_digests)
            ),
            "exact_lossless_rgb_correspondence": True,
        }
        normalized_video["verified_video_receipt_digest"] = _canonical_digest(normalized_video)
        normalized[camera_id] = normalized_video
    return normalized


def _trace_manifest_rows(
    *, episode_id: str, steps: Sequence[Mapping[str, Any]], entity_ids: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    deformable_id = entity_ids["deformable"]
    robot_id = entity_ids["robot"]
    for step in steps:
        deformable = step["entities"][deformable_id]
        robot = step["entities"][robot_id]
        contact_projection = {
            "deformable_contact_pair_count_by_entity_id": deformable[
                "contact_pair_count_by_entity_id"
            ],
            "deformable_contact_normal_force_n_by_entity_id": deformable[
                "contact_normal_force_n_by_entity_id"
            ],
            "robot_gripper_contact_pair_count_by_entity_id": robot[
                "gripper_contact_pair_count_by_entity_id"
            ],
            "robot_gripper_contact_normal_force_n_by_entity_id": robot[
                "gripper_contact_normal_force_n_by_entity_id"
            ],
            "robot_gripper_attachment_constraint_count_by_entity_id": robot[
                "gripper_attachment_constraint_count_by_entity_id"
            ],
            "hidden_attachment_active": deformable["hidden_attachment_active"],
        }
        rows.append(
            {
                "episode_id": episode_id,
                "sample_index": step["sample_index"],
                "timestamp_ns": step["timestamp_ns"],
                "simulation_time_s": step["simulation_time_s"],
                "action_digest": step["action"]["action_digest"],
                "native_action_sha256": step["action"]["delivery"]["native_action_sha256"],
                "delivery_receipt_digest": step["action"]["delivery"]["receipt_digest"],
                "observation_digest": step["observation_digest"],
                "native_state_sha256_by_entity_id": step["native_state_sha256_by_entity_id"],
                "state_write_count_after_episode_start_by_entity_id": step[
                    "state_write_count_after_episode_start_by_entity_id"
                ],
                "contact_readback_sha256": _canonical_digest(contact_projection),
                "frame_digest_by_camera": {
                    camera_id: step["frames"][camera_id]["frame_digest"] for camera_id in CAMERA_IDS
                },
            }
        )
    return rows


def _normalize_trace_manifest(
    value: Any,
    *,
    episode_id: str,
    steps: Sequence[Mapping[str, Any]],
    entity_ids: Mapping[str, str],
    evidence_root: Path,
) -> dict[str, Any]:
    rows = _trace_manifest_rows(episode_id=episode_id, steps=steps, entity_ids=entity_ids)
    expected_payload = b"".join(_canonical_json_bytes(row) + b"\n" for row in rows)
    return _normalize_manifest_file(
        value,
        schema_version=NATIVE_TRACE_MANIFEST_SCHEMA_VERSION,
        expected_payload=expected_payload,
        row_count=len(rows),
        rows_sha256=_bytes_digest(_canonical_json_bytes(rows)),
        evidence_root=evidence_root,
        error="deformable_episode_native_trace_manifest_invalid",
    )


def _scorer_samples(
    steps: Sequence[Mapping[str, Any]], *, entity_ids: Mapping[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in steps:
        deformable = step["entities"][entity_ids["deformable"]]
        destination = step["entities"][entity_ids["destination"]]
        robot = step["entities"][entity_ids["robot"]]
        rows.append(
            {
                "sample_index": step["sample_index"],
                "time_seconds": step["simulation_time_s"],
                "entities": {
                    entity_ids["deformable"]: {
                        key: deformable[key]
                        for key in (
                            "nodal_positions_world_m",
                            "nodal_velocities_world_mps",
                            "deformation_gradients",
                            "nodal_kinematic_flags",
                            "state_write_count_after_episode_start",
                            "solver_divergence_count",
                        )
                    },
                    entity_ids["destination"]: {
                        key: destination[key]
                        for key in (
                            "pose_world",
                            "linear_velocity_world_mps",
                            "angular_velocity_world_radps",
                        )
                    },
                    entity_ids["robot"]: {
                        key: robot[key]
                        for key in (
                            "gripper_clearance_points_world_m",
                            "gripper_contact_pair_count_by_entity_id",
                            "gripper_contact_normal_force_n_by_entity_id",
                        )
                    },
                },
            }
        )
    return rows


def _centroid(points: Sequence[Sequence[float]]) -> list[float]:
    count = len(points)
    return [sum(point[axis] for point in points) / count for axis in range(3)]


def _point_displacement_m(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((second[axis] - first[axis]) ** 2 for axis in range(3)))


def _maximum_corresponding_node_displacement_m(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    return max(
        _point_displacement_m(first_point, second_point)
        for first_point, second_point in zip(first, second, strict=True)
    )


def _inside_destination_obb(point_world_m: Sequence[float], obb: Mapping[str, Any]) -> bool:
    center = obb["center_world_m"]
    half_extents = obb["half_extents_m"]
    x, y, z, w = obb["orientation_xyzw"]
    world_from_local = (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )
    offset = [point_world_m[axis] - center[axis] for axis in range(3)]
    local = [
        sum(
            world_from_local[world_axis][local_axis] * offset[world_axis] for world_axis in range(3)
        )
        for local_axis in range(3)
    ]
    return all(abs(local[axis]) <= half_extents[axis] for axis in range(3))


def _evidence(
    *,
    steps: Sequence[Mapping[str, Any]],
    reset: Mapping[str, Any],
    entity_ids: Mapping[str, str],
    thresholds: Mapping[str, Any],
    episode_kind: str,
    deterministic_score: Mapping[str, Any],
    media_complete: bool,
) -> dict[str, Any]:
    deformable_id = entity_ids["deformable"]
    robot_id = entity_ids["robot"]
    joint_trace = [reset["robot_joint_positions_rad"]] + [
        step["entities"][robot_id]["arm_joint_positions_rad"] for step in steps
    ]
    gripper_trace = [reset["gripper_width_m"]] + [
        step["entities"][robot_id]["gripper_width_m"] for step in steps
    ]
    reset_joints = joint_trace[0]
    max_joint_delta = [
        max(abs(sample[index] - reset_joints[index]) for sample in joint_trace)
        for index in range(len(reset_joints))
    ]
    max_gripper_delta = max(abs(value - gripper_trace[0]) for value in gripper_trace)
    arm_moved = max(max_joint_delta, default=0.0) > thresholds["arm_motion_epsilon_rad"]
    gripper_responded = max_gripper_delta > thresholds["gripper_motion_epsilon_m"]

    nontrivial_arm_actions = 0
    nontrivial_gripper_actions = 0
    arm_response_rows = 0
    gripper_response_rows = 0
    arm_response_sample_indices: list[int] = []
    gripper_response_sample_indices: list[int] = []
    commanded_gripper_close_response_sample_indices: list[int] = []
    commanded_gripper_open_response_sample_indices: list[int] = []
    delivered_rows = 0
    attempted_rows = 0
    previous_joints = reset_joints
    previous_gripper = gripper_trace[0]
    for step in steps:
        action = step["action"]
        delivery = action["delivery"]
        attempted_rows += int(delivery["attempted"])
        delivered_rows += int(delivery["delivered_to_robot"])
        arm_nontrivial = (
            max(abs(value) for value in action["arm_command"]) > thresholds["action_epsilon"]
        )
        gripper_nontrivial = abs(action["gripper_delta_command_m"]) > thresholds["action_epsilon"]
        nontrivial_arm_actions += int(arm_nontrivial)
        nontrivial_gripper_actions += int(gripper_nontrivial)
        current_robot = step["entities"][robot_id]
        observed_arm_delta = max(
            abs(current - previous)
            for current, previous in zip(
                current_robot["arm_joint_positions_rad"], previous_joints, strict=True
            )
        )
        observed_gripper_signed_delta = current_robot["gripper_width_m"] - previous_gripper
        observed_gripper_delta = abs(observed_gripper_signed_delta)
        if (
            arm_nontrivial
            and delivery["delivered_to_robot"]
            and observed_arm_delta > thresholds["arm_motion_epsilon_rad"]
        ):
            arm_response_rows += 1
            arm_response_sample_indices.append(step["sample_index"])
        if (
            gripper_nontrivial
            and delivery["delivered_to_robot"]
            and observed_gripper_delta > thresholds["gripper_motion_epsilon_m"]
        ):
            gripper_response_rows += 1
            gripper_response_sample_indices.append(step["sample_index"])
        if (
            action["gripper_delta_command_m"] < -thresholds["action_epsilon"]
            and delivery["delivered_to_robot"]
            and observed_gripper_signed_delta < -thresholds["gripper_motion_epsilon_m"]
        ):
            commanded_gripper_close_response_sample_indices.append(step["sample_index"])
        if (
            action["gripper_delta_command_m"] > thresholds["action_epsilon"]
            and delivery["delivered_to_robot"]
            and observed_gripper_signed_delta > thresholds["gripper_motion_epsilon_m"]
        ):
            commanded_gripper_open_response_sample_indices.append(step["sample_index"])
        previous_joints = current_robot["arm_joint_positions_rad"]
        previous_gripper = current_robot["gripper_width_m"]

    all_delivery_receipts_valid = bool(steps) and attempted_rows == delivered_rows == len(steps)
    arm_channel_response_proven = bool(nontrivial_arm_actions > 0 and arm_response_rows > 0)
    gripper_channel_response_proven = bool(
        nontrivial_gripper_actions == 0 or gripper_response_rows > 0
    )
    actions_reached_robot = bool(
        all_delivery_receipts_valid
        and arm_channel_response_proven
        and gripper_channel_response_proven
    )

    robot_contact_pairs = max(
        step["entities"][robot_id]["gripper_contact_pair_count_by_entity_id"][deformable_id]
        for step in steps
    )
    robot_contact_force = max(
        step["entities"][robot_id]["gripper_contact_normal_force_n_by_entity_id"][deformable_id]
        for step in steps
    )
    deformable_contact_pairs = max(
        step["entities"][deformable_id]["contact_pair_count_by_entity_id"][robot_id]
        for step in steps
    )
    deformable_contact_force = max(
        step["entities"][deformable_id]["contact_normal_force_n_by_entity_id"][robot_id]
        for step in steps
    )
    contact_rows = [
        step["sample_index"]
        for step in steps
        if (
            step["entities"][robot_id]["gripper_contact_pair_count_by_entity_id"][deformable_id] > 0
            and step["entities"][deformable_id]["contact_pair_count_by_entity_id"][robot_id] > 0
            and step["entities"][robot_id]["gripper_contact_normal_force_n_by_entity_id"][
                deformable_id
            ]
            > thresholds["contact_force_epsilon_n"]
            and step["entities"][deformable_id]["contact_normal_force_n_by_entity_id"][robot_id]
            > thresholds["contact_force_epsilon_n"]
        )
    ]
    contact_observed = bool(contact_rows)
    last_contact_index = max(contact_rows) if contact_rows else None
    released_contact_rows = [
        step["sample_index"]
        for step in steps
        if last_contact_index is not None
        and step["sample_index"] > last_contact_index
        and step["entities"][robot_id]["gripper_contact_pair_count_by_entity_id"][deformable_id]
        == 0
        and step["entities"][deformable_id]["contact_pair_count_by_entity_id"][robot_id] == 0
    ]
    contact_then_release_observed = bool(contact_rows and released_contact_rows)

    reset_positions = reset["deformable_nodal_positions_world_m"]
    reset_centroid = _centroid(reset_positions)
    destination_obb = deterministic_score["frozen_destination_interior_obb"]
    minimum_particle_fraction_inside = deterministic_score["thresholds"][
        "minimum_particle_fraction_inside"
    ]
    reset_nodes_inside = sum(
        _inside_destination_obb(point, destination_obb) for point in reset_positions
    )
    initial_deformable_outside_destination = reset_nodes_inside == 0
    node_count_inside_destination_by_sample = [
        sum(
            _inside_destination_obb(point, destination_obb)
            for point in step["entities"][deformable_id]["nodal_positions_world_m"]
        )
        for step in steps
    ]
    contained_by_sample = [
        bool(
            node_count / len(step["entities"][deformable_id]["nodal_positions_world_m"])
            >= minimum_particle_fraction_inside
            and _inside_destination_obb(
                _centroid(step["entities"][deformable_id]["nodal_positions_world_m"]),
                destination_obb,
            )
        )
        for step, node_count in zip(steps, node_count_inside_destination_by_sample, strict=True)
    ]
    qualifying_contact_rows = [
        sample_index
        for sample_index in deterministic_score["measurements"][
            "qualifying_grasp_contact_sample_indices"
        ]
        if sample_index in contact_rows
    ]
    first_qualifying_contact_index = (
        min(qualifying_contact_rows) if qualifying_contact_rows else None
    )
    all_samples_through_qualifying_contact_outside_destination = bool(
        initial_deformable_outside_destination
        and first_qualifying_contact_index is not None
        and all(
            node_count == 0
            for step, node_count in zip(steps, node_count_inside_destination_by_sample, strict=True)
            if step["sample_index"] <= first_qualifying_contact_index
        )
    )
    contact_centroid = (
        _centroid(
            steps[first_qualifying_contact_index]["entities"][deformable_id][
                "nodal_positions_world_m"
            ]
        )
        if first_qualifying_contact_index is not None
        else None
    )
    contact_positions = (
        steps[first_qualifying_contact_index]["entities"][deformable_id]["nodal_positions_world_m"]
        if first_qualifying_contact_index is not None
        else None
    )
    centroid_displacement_from_reset_by_sample_m = [
        _point_displacement_m(
            reset_centroid,
            _centroid(step["entities"][deformable_id]["nodal_positions_world_m"]),
        )
        for step in steps
    ]
    post_contact_centroid_displacement_by_sample_m = [
        (
            _point_displacement_m(
                contact_centroid,
                _centroid(step["entities"][deformable_id]["nodal_positions_world_m"]),
            )
            if contact_centroid is not None
            and step["sample_index"] > first_qualifying_contact_index
            else 0.0
        )
        for step in steps
    ]
    maximum_nodal_displacement_from_reset_by_sample_m = [
        _maximum_corresponding_node_displacement_m(
            reset_positions,
            step["entities"][deformable_id]["nodal_positions_world_m"],
        )
        for step in steps
    ]
    maximum_post_contact_nodal_displacement_by_sample_m = [
        (
            _maximum_corresponding_node_displacement_m(
                contact_positions,
                step["entities"][deformable_id]["nodal_positions_world_m"],
            )
            if contact_positions is not None
            and step["sample_index"] > first_qualifying_contact_index
            else 0.0
        )
        for step in steps
    ]
    task_relevant_displacement_sample_indices = [
        step["sample_index"]
        for step, displacement in zip(
            steps, post_contact_centroid_displacement_by_sample_m, strict=True
        )
        if first_qualifying_contact_index is not None
        and step["sample_index"] > first_qualifying_contact_index
        and displacement >= thresholds["minimum_deformable_displacement_m"]
    ]
    first_task_relevant_displacement_index = (
        min(task_relevant_displacement_sample_indices)
        if task_relevant_displacement_sample_indices
        else None
    )
    post_contact_containment_sample_indices = [
        step["sample_index"]
        for step, contained in zip(steps, contained_by_sample, strict=True)
        if contained
        and first_qualifying_contact_index is not None
        and step["sample_index"] > first_qualifying_contact_index
    ]
    first_post_contact_containment_index = (
        min(post_contact_containment_sample_indices)
        if post_contact_containment_sample_indices
        else None
    )
    post_contact_displacement_then_containment_transition_observed = bool(
        all_samples_through_qualifying_contact_outside_destination
        and first_task_relevant_displacement_index is not None
        and first_post_contact_containment_index is not None
        and first_task_relevant_displacement_index <= first_post_contact_containment_index
    )
    first_post_contact_release_index = min(released_contact_rows) if released_contact_rows else None
    settle_window_samples_used = deterministic_score["measurements"]["settle_window_samples_used"]
    settle_window_start_index = (
        steps[-settle_window_samples_used]["sample_index"]
        if deterministic_score["measurements"]["settle_window_available"]
        and settle_window_samples_used > 0
        else None
    )
    final_index = steps[-1]["sample_index"]
    final_displacement_from_reset = centroid_displacement_from_reset_by_sample_m[-1]
    ordered_contact_displacement_release_final_settle = bool(
        all_samples_through_qualifying_contact_outside_destination
        and first_qualifying_contact_index is not None
        and first_task_relevant_displacement_index is not None
        and first_post_contact_containment_index is not None
        and first_post_contact_release_index is not None
        and settle_window_start_index is not None
        and first_qualifying_contact_index
        < first_task_relevant_displacement_index
        <= first_post_contact_containment_index
        < first_post_contact_release_index
        < final_index
        and post_contact_displacement_then_containment_transition_observed
        and first_post_contact_release_index <= settle_window_start_index
        and final_displacement_from_reset >= thresholds["minimum_deformable_displacement_m"]
        and deterministic_score["predicates"]["contained"]
        and deterministic_score["predicates"]["released"]
        and deterministic_score["predicates"]["settled"]
        and contact_then_release_observed
    )

    minimum_gripper_width = min(gripper_trace)
    minimum_width_trace_index = gripper_trace.index(minimum_gripper_width)
    gripper_close_observed = bool(
        gripper_trace[0] - minimum_gripper_width > thresholds["gripper_motion_epsilon_m"]
    )
    release_width_samples = (
        [steps[index]["entities"][robot_id]["gripper_width_m"] for index in released_contact_rows]
        if released_contact_rows
        else []
    )
    gripper_release_observed = bool(
        release_width_samples
        and max(release_width_samples) - minimum_gripper_width
        > thresholds["gripper_motion_epsilon_m"]
        and minimum_width_trace_index <= (last_contact_index or 0) + 1
    )
    commanded_gripper_close_response_before_or_at_grasp_indices = [
        sample_index
        for sample_index in commanded_gripper_close_response_sample_indices
        if first_qualifying_contact_index is not None
        and sample_index <= first_qualifying_contact_index
    ]
    first_commanded_gripper_close_response_index = (
        min(commanded_gripper_close_response_before_or_at_grasp_indices)
        if commanded_gripper_close_response_before_or_at_grasp_indices
        else None
    )
    commanded_gripper_release_response_at_release_indices = [
        sample_index
        for sample_index in commanded_gripper_open_response_sample_indices
        if first_post_contact_release_index is not None
        and sample_index == first_post_contact_release_index
    ]
    first_commanded_gripper_release_response_index = (
        min(commanded_gripper_release_response_at_release_indices)
        if commanded_gripper_release_response_at_release_indices
        else None
    )
    transport_arm_response_sample_indices = [
        sample_index
        for sample_index in arm_response_sample_indices
        if first_qualifying_contact_index is not None
        and first_task_relevant_displacement_index is not None
        and first_qualifying_contact_index < sample_index <= first_task_relevant_displacement_index
    ]
    delivered_arm_transport_response_observed = bool(transport_arm_response_sample_indices)
    commanded_close_then_release_response_ordered = bool(
        first_commanded_gripper_close_response_index is not None
        and first_qualifying_contact_index is not None
        and first_task_relevant_displacement_index is not None
        and first_post_contact_containment_index is not None
        and first_commanded_gripper_release_response_index is not None
        and first_commanded_gripper_close_response_index
        <= first_qualifying_contact_index
        < first_task_relevant_displacement_index
        <= first_post_contact_containment_index
        < first_commanded_gripper_release_response_index
        < final_index
    )
    retreat_response_observed = bool(
        last_contact_index is not None
        and any(index > last_contact_index for index in arm_response_sample_indices)
        and deterministic_score["predicates"]["robot_retreated"]
    )

    post_start_write_count_by_entity = {
        entity_id: max(
            step["state_write_count_after_episode_start_by_entity_id"][entity_id] for step in steps
        )
        for entity_id in steps[0]["state_write_count_after_episode_start_by_entity_id"]
    }
    maximum_nonfree_nodes = max(
        sum(
            not math.isclose(value, FREE_KINEMATIC_FLAG, abs_tol=1.0e-9)
            for value in step["entities"][deformable_id]["nodal_kinematic_flags"]
        )
        for step in steps
    )
    hidden_attachment_active = any(
        step["entities"][deformable_id]["hidden_attachment_active"]
        or step["entities"][robot_id]["gripper_attachment_constraint_count_by_entity_id"][
            deformable_id
        ]
        > 0
        for step in steps
    )
    native_contact_only = all(
        step["entities"][deformable_id]["grasp_representation"] == "native_contact_only"
        for step in steps
    )
    no_post_start_writes = not any(post_start_write_count_by_entity.values())
    no_hidden_attachment = bool(
        maximum_nonfree_nodes == 0 and not hidden_attachment_active and native_contact_only
    )
    manipulation_integrity_valid = no_post_start_writes and no_hidden_attachment

    reset_proven = bool(
        reset["actor_reset_invoked"]
        and reset["native_readback_matches_frozen_state"]
        and not reset["initial_penetration_observed"]
    )
    if not media_complete:
        interpretation = "lossless_media_or_manifest_not_proven_harness_fault"
    elif not reset_proven:
        interpretation = "reset_not_proven_harness_fault"
    elif not all_delivery_receipts_valid:
        interpretation = "action_delivery_not_proven_harness_fault"
    elif not manipulation_integrity_valid:
        interpretation = "manipulation_integrity_violation"
    elif episode_kind == "zero_action_control":
        interpretation = "zero_action_control_outcome_interpretable"
    elif not arm_moved:
        interpretation = "arm_motion_not_observed_policy_outcome_uninterpretable"
    elif nontrivial_gripper_actions > 0 and not gripper_responded:
        interpretation = "gripper_response_not_observed_policy_outcome_uninterpretable"
    elif not actions_reached_robot:
        interpretation = "action_response_not_observed_policy_outcome_uninterpretable"
    elif episode_kind != "learned_policy_evaluation" and not contact_observed:
        interpretation = "native_manipulation_contact_not_observed_outcome_uninterpretable"
    elif episode_kind == "learned_policy_evaluation":
        interpretation = "policy_task_outcome_interpretable"
    else:
        interpretation = "scripted_positive_control_outcome_interpretable"

    policy_outcome_interpretable = bool(
        episode_kind == "learned_policy_evaluation"
        and media_complete
        and reset_proven
        and all_delivery_receipts_valid
        and actions_reached_robot
        and arm_moved
        and manipulation_integrity_valid
    )
    zero_control_outcome_interpretable = bool(
        episode_kind == "zero_action_control"
        and media_complete
        and reset_proven
        and all_delivery_receipts_valid
        and manipulation_integrity_valid
    )
    scripted_control_outcome_interpretable = bool(
        episode_kind == "scripted_positive_control"
        and media_complete
        and reset_proven
        and all_delivery_receipts_valid
        and actions_reached_robot
        and arm_moved
        and gripper_responded
        and contact_observed
        and manipulation_integrity_valid
    )
    control_outcome_interpretable = bool(
        zero_control_outcome_interpretable or scripted_control_outcome_interpretable
    )
    manipulation_sequence_complete = bool(
        ordered_contact_displacement_release_final_settle
        and gripper_close_observed
        and gripper_release_observed
        and commanded_close_then_release_response_ordered
        and delivered_arm_transport_response_observed
        and retreat_response_observed
    )
    return {
        "action_delivery": {
            "action_rows": len(steps),
            "delivery_attempted_rows": attempted_rows,
            "delivered_to_robot_rows": delivered_rows,
            "all_action_delivery_receipts_valid": all_delivery_receipts_valid,
            "nontrivial_arm_action_rows": nontrivial_arm_actions,
            "nontrivial_gripper_action_rows": nontrivial_gripper_actions,
            "arm_response_rows": arm_response_rows,
            "gripper_response_rows": gripper_response_rows,
            "arm_response_sample_indices": arm_response_sample_indices,
            "gripper_response_sample_indices": gripper_response_sample_indices,
            "commanded_gripper_close_response_sample_indices": (
                commanded_gripper_close_response_sample_indices
            ),
            "commanded_gripper_open_response_sample_indices": (
                commanded_gripper_open_response_sample_indices
            ),
            "actions_reached_robot": actions_reached_robot,
            "arm_channel_response_proven": arm_channel_response_proven,
            "gripper_channel_response_proven": gripper_channel_response_proven,
        },
        "robot_motion": {
            "joint_position_reset_rad": reset_joints,
            "joint_position_end_rad": joint_trace[-1],
            "max_abs_joint_delta_from_reset_rad": max_joint_delta,
            "arm_motion_epsilon_rad": thresholds["arm_motion_epsilon_rad"],
            "minimum_arm_motion_epsilon_floor_rad": (MINIMUM_ARM_MOTION_EPSILON_RAD),
            "arm_moved": arm_moved,
            "gripper_width_reset_m": gripper_trace[0],
            "gripper_width_end_m": gripper_trace[-1],
            "maximum_gripper_width_delta_from_reset_m": max_gripper_delta,
            "gripper_motion_epsilon_m": thresholds["gripper_motion_epsilon_m"],
            "minimum_gripper_motion_epsilon_floor_m": (MINIMUM_GRIPPER_MOTION_EPSILON_M),
            "gripper_responded": gripper_responded,
            "gripper_close_observed": gripper_close_observed,
            "gripper_release_observed": gripper_release_observed,
            "first_commanded_gripper_close_response_sample_index": (
                first_commanded_gripper_close_response_index
            ),
            "first_commanded_gripper_release_response_sample_index": (
                first_commanded_gripper_release_response_index
            ),
            "commanded_gripper_close_response_observed_before_or_at_grasp": bool(
                commanded_gripper_close_response_before_or_at_grasp_indices
            ),
            "commanded_gripper_release_response_observed_at_release": bool(
                commanded_gripper_release_response_at_release_indices
            ),
            "commanded_close_then_release_response_ordered": (
                commanded_close_then_release_response_ordered
            ),
            "transport_arm_response_sample_indices": transport_arm_response_sample_indices,
            "delivered_arm_transport_response_observed": (
                delivered_arm_transport_response_observed
            ),
            "retreat_response_observed_after_contact": retreat_response_observed,
        },
        "contact_evidence": {
            "maximum_robot_gripper_contact_pair_count": robot_contact_pairs,
            "maximum_robot_gripper_contact_normal_force_n": robot_contact_force,
            "maximum_deformable_robot_contact_pair_count": deformable_contact_pairs,
            "maximum_deformable_robot_contact_normal_force_n": deformable_contact_force,
            "bilateral_contact_sample_indices": contact_rows,
            "last_bilateral_contact_sample_index": last_contact_index,
            "post_contact_release_sample_indices": released_contact_rows,
            "genuine_native_manipulation_contact_observed": contact_observed,
            "contact_then_release_observed": contact_then_release_observed,
        },
        "deformable_motion": {
            "frozen_start_state_sha256": reset["deformable_start_state_sha256"],
            "reset_nodal_position_count": len(reset_positions),
            "reset_node_count_inside_destination": reset_nodes_inside,
            "initial_deformable_outside_destination": (initial_deformable_outside_destination),
            "node_count_inside_destination_by_sample": (node_count_inside_destination_by_sample),
            "contained_by_sample": contained_by_sample,
            "qualifying_grasp_contact_sample_indices": qualifying_contact_rows,
            "first_qualifying_grasp_contact_sample_index": (first_qualifying_contact_index),
            "all_samples_through_qualifying_contact_outside_destination": (
                all_samples_through_qualifying_contact_outside_destination
            ),
            "centroid_displacement_from_reset_by_sample_m": (
                centroid_displacement_from_reset_by_sample_m
            ),
            "post_contact_centroid_displacement_by_sample_m": (
                post_contact_centroid_displacement_by_sample_m
            ),
            "maximum_nodal_displacement_from_reset_by_sample_m": (
                maximum_nodal_displacement_from_reset_by_sample_m
            ),
            "maximum_post_contact_nodal_displacement_by_sample_m": (
                maximum_post_contact_nodal_displacement_by_sample_m
            ),
            "maximum_centroid_displacement_from_reset_m": max(
                centroid_displacement_from_reset_by_sample_m
            ),
            "maximum_post_contact_centroid_displacement_m": max(
                post_contact_centroid_displacement_by_sample_m
            ),
            "maximum_nodal_displacement_from_reset_m": max(
                maximum_nodal_displacement_from_reset_by_sample_m
            ),
            "maximum_post_contact_nodal_displacement_m": max(
                maximum_post_contact_nodal_displacement_by_sample_m
            ),
            "minimum_deformable_displacement_m": thresholds["minimum_deformable_displacement_m"],
            "physical_minimum_deformable_displacement_floor_m": (
                MINIMUM_DEFORMABLE_DISPLACEMENT_FLOOR_M
            ),
            "task_relevant_displacement_sample_indices": (
                task_relevant_displacement_sample_indices
            ),
            "first_task_relevant_displacement_sample_index": (
                first_task_relevant_displacement_index
            ),
            "post_contact_containment_sample_indices": (post_contact_containment_sample_indices),
            "first_post_contact_containment_sample_index": (first_post_contact_containment_index),
            "post_contact_displacement_then_containment_transition_observed": (
                post_contact_displacement_then_containment_transition_observed
            ),
            "first_post_contact_release_sample_index": (first_post_contact_release_index),
            "settle_window_start_sample_index": settle_window_start_index,
            "final_centroid_displacement_from_reset_m": (final_displacement_from_reset),
            "ordered_contact_displacement_release_final_settle": (
                ordered_contact_displacement_release_final_settle
            ),
        },
        "integrity": {
            "post_start_state_write_count_by_entity_id": post_start_write_count_by_entity,
            "no_direct_state_writes_after_episode_start": no_post_start_writes,
            "maximum_nonfree_deformable_node_count": maximum_nonfree_nodes,
            "hidden_attachment_readback_active": hidden_attachment_active,
            "native_contact_only_grasp_representation": native_contact_only,
            "no_hidden_kinematic_attachment": no_hidden_attachment,
            "manipulation_integrity_valid": manipulation_integrity_valid,
        },
        "interpretability": {
            "policy_outcome_interpretable": policy_outcome_interpretable,
            "control_outcome_interpretable": control_outcome_interpretable,
            "zero_control_outcome_interpretable": zero_control_outcome_interpretable,
            "scripted_control_outcome_interpretable": (scripted_control_outcome_interpretable),
            "reset_proven": reset_proven,
            "lossless_media_and_manifests_proven": media_complete,
            "manipulation_sequence_complete": manipulation_sequence_complete,
            "interpretation": interpretation,
        },
    }


def materialize_native_deformable_episode_trace(
    *,
    task_entities: Sequence[Mapping[str, Any]],
    task_spec: Mapping[str, Any],
    frozen_run_contract: Mapping[str, Any],
    episode_trace: Mapping[str, Any],
    evidence_root: str | Path,
    frozen_run_seal_relative_path: str | None = None,
    expected_frozen_run_seal_sha256: str | None = None,
    trusted_execution_envelope_relative_path: str | None = None,
    native_event_relative_path: str | None = None,
) -> dict[str, Any]:
    """Validate one native episode trace and emit deterministic evidence.

    A complete trace is scored exactly once with
    :func:`score_deformable_transfer`.  A native failure before the first
    observation may instead emit a typed terminal-media gap; such a receipt has
    no scorer input and can never be interpreted as a policy outcome.  Omitting
    the four authority arguments deliberately produces an untrusted structural
    candidate.  ``expected_frozen_run_seal_sha256`` is a trust-root input from
    the frozen aggregator configuration; deriving it from the returned bundle
    would not establish admission authority.
    """

    source = _clone_mapping(episode_trace, error="deformable_episode_trace_invalid")
    if _contains_caller_grade(source):
        _raise("deformable_episode_caller_authored_grade_forbidden")
    resolved_evidence_root = _resolve_evidence_root(evidence_root)

    episode_kind = _string(source.get("episode_kind"), error="deformable_episode_kind_invalid")
    if episode_kind not in EPISODE_KINDS:
        _raise("deformable_episode_kind_invalid")
    episode_id = _string(
        source.get("episode_id"), error="deformable_episode_id_invalid", identifier=True
    )
    episode_start_timestamp_ns = _integer(
        source.get("episode_start_timestamp_ns"),
        error="deformable_episode_start_timestamp_invalid",
    )
    prompt = _string(source.get("prompt"), error="deformable_episode_prompt_invalid")
    expected_prompt_digest = _bytes_digest(prompt.encode("utf-8"))
    if (
        _digest(
            source.get("prompt_sha256"),
            error="deformable_episode_prompt_digest_invalid",
        )
        != expected_prompt_digest
    ):
        _raise("deformable_episode_prompt_digest_mismatch")

    entity_contract = materialize_native_task_entity_contract(
        task_kind=TASK_KIND_DEFORMABLE_TRANSFER, task_entities=task_entities
    )
    normalized_task_spec = _clone_mapping(task_spec, error="deformable_episode_task_spec_invalid")
    if _contains_caller_grade(normalized_task_spec):
        _raise("deformable_episode_task_spec_caller_grade_forbidden")
    task_spec_sha256 = _canonical_digest(normalized_task_spec)
    entity_ids = _entity_ids(entity_contract=entity_contract, task_spec=normalized_task_spec)
    all_entity_ids = sorted(entity["entity_id"] for entity in entity_contract["task_entities"])
    entity_physics_type_by_id = {
        entity["entity_id"]: entity["physics_type"] for entity in entity_contract["task_entities"]
    }
    frozen_run = _normalize_frozen_run_contract(
        frozen_run_contract,
        entity_contract_digest=entity_contract["contract_digest"],
        task_spec_sha256=task_spec_sha256,
        prompt_sha256=expected_prompt_digest,
    )
    native_authority = _native_authority(
        source_trace=source,
        frozen_run=frozen_run,
        evidence_root=resolved_evidence_root,
        frozen_run_seal_relative_path=frozen_run_seal_relative_path,
        expected_frozen_run_seal_sha256=expected_frozen_run_seal_sha256,
        trusted_execution_envelope_relative_path=trusted_execution_envelope_relative_path,
        native_event_relative_path=native_event_relative_path,
    )
    cell = _normalize_cell(source.get("cell"))
    thresholds = _normalize_thresholds(source.get("trace_thresholds"))
    if _canonical_digest(thresholds) != frozen_run["trace_thresholds_sha256"]:
        _raise("deformable_episode_frozen_trace_thresholds_mismatch")
    frozen_camera_contract = {
        "camera_calibration_digest_by_camera_id": frozen_run[
            "camera_calibration_digest_by_camera_id"
        ],
        "renderer_identity_sha256_by_camera_id": frozen_run[
            "renderer_identity_sha256_by_camera_id"
        ],
    }
    if cell["camera_contract_sha256"] != _canonical_digest(frozen_camera_contract):
        _raise("deformable_episode_frozen_cell_camera_contract_mismatch")
    actor = _normalize_actor(source.get("actor"), episode_kind=episode_kind)
    expected_cell_digest = frozen_run["cell_identity_digest_by_id"].get(cell["cell_id"])
    if expected_cell_digest is None or expected_cell_digest != _canonical_digest(cell):
        _raise("deformable_episode_frozen_cell_identity_mismatch")
    if actor["kind"] == "learned_policy":
        expected_actor_digest = frozen_run["candidate_identity_digest_by_id"].get(
            actor["candidate_id"]
        )
    else:
        expected_actor_digest = frozen_run["control_identity_digest_by_episode_kind"].get(
            episode_kind
        )
    if expected_actor_digest != actor["identity_digest"]:
        _raise("deformable_episode_frozen_actor_identity_mismatch")
    reset = _normalize_reset(
        source.get("reset_receipt"),
        all_entity_ids=all_entity_ids,
        entity_physics_type_by_id=entity_physics_type_by_id,
        deformable_entity_id=entity_ids["deformable"],
        robot_entity_id=entity_ids["robot"],
        actor=actor,
        frozen_run=frozen_run,
        episode_start_timestamp_ns=episode_start_timestamp_ns,
    )
    steps = _normalize_steps(
        source.get("steps"),
        episode_kind=episode_kind,
        actor=actor,
        entity_ids=entity_ids,
        all_entity_ids=all_entity_ids,
        reset=reset,
        episode_start_timestamp_ns=episode_start_timestamp_ns,
        thresholds=thresholds,
        evidence_root=resolved_evidence_root,
        frozen_run=frozen_run,
    )

    if not steps:
        terminal = _normalize_terminal_gap(
            source.get("terminal"),
            episode_start_timestamp_ns=episode_start_timestamp_ns,
            evidence_root=resolved_evidence_root,
        )
        if (
            source.get("media_manifest") is not None
            or source.get("native_trace_manifest") is not None
        ):
            _raise("deformable_episode_pre_observation_manifests_forbidden")
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_kind": episode_kind,
            "episode_start_timestamp_ns": episode_start_timestamp_ns,
            "cell": cell,
            "prompt": prompt,
            "prompt_sha256": expected_prompt_digest,
            "actor": actor,
            "frozen_run_contract_digest": frozen_run["contract_digest"],
            "task_spec_sha256": task_spec_sha256,
            "entity_contract_digest": entity_contract["contract_digest"],
            "task_entities": entity_contract["task_entities"],
            "semantic_role_index": entity_contract["semantic_role_index"],
            "entity_ids": entity_ids,
            "reset_receipt": reset,
            "trace_thresholds": thresholds,
            "steps": [],
            "terminal": terminal,
            "media_manifest": None,
            "native_trace_manifest": None,
            "review_videos": None,
            "media_validation_blockers": ["typed_gap_before_first_observation"],
            "native_event_authority": native_authority,
            "scorer_ready_samples": [],
            "deterministic_task_state_score": None,
            "evidence": {
                "action_delivery": None,
                "robot_motion": None,
                "contact_evidence": None,
                "deformable_motion": None,
                "integrity": None,
                "interpretability": {
                    "policy_outcome_interpretable": False,
                    "control_outcome_interpretable": False,
                    "zero_control_outcome_interpretable": False,
                    "scripted_control_outcome_interpretable": False,
                    "reset_proven": True,
                    "lossless_media_and_manifests_proven": False,
                    "manipulation_sequence_complete": False,
                    "interpretation": "failed_before_first_observation_typed_media_gap",
                },
            },
            "policy_outcome": None,
            "control_outcome": None,
            "native_episode_evidence_admitted": False,
            "native_episode_admitted_deterministic_success": False,
            "evaluation_admitted_deterministic_success": False,
            "evaluation_admission_blockers": ["failed_before_first_observation"],
            "media_complete": False,
            "media_status": "typed_gap_before_first_observation",
            "scoring_authority": "no_scorer_input",
            "deterministic_score_claim_status": "not_scored",
            "overview_used_by_policy": False,
            "overview_used_by_deterministic_scorer": False,
            "cell_family_changes_trace_contract": False,
            "claim_boundary": (
                "trusted_runner_typed_native_pre_observation_failure_not_a_policy_"
                "outcome_or_physical_deformable_or_robot_claim"
                if native_authority["native_event_authority_verified"]
                else "untrusted_structural_pre_observation_failure_candidate_only"
            ),
        }
        receipt["receipt_digest"] = _canonical_digest(receipt)
        return receipt

    terminal = _normalize_complete_terminal(source.get("terminal"), steps=steps)
    if episode_kind == "learned_policy_evaluation" and not any(
        step["actor_observation"] for step in steps
    ):
        _raise("deformable_episode_policy_inputs_missing")
    if episode_kind == "zero_action_control":
        if any(
            any(value != 0.0 for value in step["action"]["arm_command"])
            or step["action"]["gripper_delta_command_m"] != 0.0
            or any(value != 0.0 for value in step["action"]["source_output"])
            or any(value != 0.0 for value in step["action"]["native_action"])
            or not step["action"]["delivery"]["attempted"]
            or not step["action"]["delivery"]["delivered_to_robot"]
            for step in steps
        ):
            _raise("deformable_episode_zero_action_seam_not_all_zero_and_delivered")
    if episode_kind == "scripted_positive_control":
        action_epsilon = thresholds["action_epsilon"]
        if not any(
            max(abs(value) for value in step["action"]["arm_command"]) > action_epsilon
            or abs(step["action"]["gripper_delta_command_m"]) > action_epsilon
            for step in steps
        ):
            _raise("deformable_episode_scripted_positive_actions_trivial")

    media_manifest = _normalize_media_manifest(
        source.get("media_manifest"),
        episode_id=episode_id,
        steps=steps,
        evidence_root=resolved_evidence_root,
    )
    media_validation_blockers: list[str] = []
    try:
        review_videos: dict[str, Any] | None = _normalize_review_videos(
            source.get("review_videos"),
            steps=steps,
            media_manifest=media_manifest,
            frozen_run=frozen_run,
            evidence_root=resolved_evidence_root,
        )
    except NativeDeformableEpisodeTraceError as exc:
        review_videos = None
        media_validation_blockers.extend(exc.errors)
    native_trace_manifest = _normalize_trace_manifest(
        source.get("native_trace_manifest"),
        episode_id=episode_id,
        steps=steps,
        entity_ids=entity_ids,
        evidence_root=resolved_evidence_root,
    )
    frame_paths = {frame["relative_path"] for step in steps for frame in step["frames"].values()}
    manifest_paths = {
        media_manifest["relative_path"],
        native_trace_manifest["relative_path"],
    }
    video_paths = (
        {video["relative_path"] for video in review_videos.values()}
        if review_videos is not None
        else set()
    )
    if (
        len(manifest_paths) != 2
        or (review_videos is not None and len(video_paths) != len(CAMERA_IDS))
        or frame_paths & manifest_paths
        or frame_paths & video_paths
        or manifest_paths & video_paths
    ):
        _raise("deformable_episode_artifact_path_collision")

    scorer_ready_samples = _scorer_samples(steps, entity_ids=entity_ids)
    deterministic_score = score_deformable_transfer(
        task_spec=normalized_task_spec, samples=scorer_ready_samples
    )
    media_complete = bool(
        native_authority["native_event_authority_verified"]
        and review_videos is not None
        and not media_validation_blockers
    )
    evidence = _evidence(
        steps=steps,
        reset=reset,
        entity_ids=entity_ids,
        thresholds=thresholds,
        episode_kind=episode_kind,
        deterministic_score=deterministic_score,
        media_complete=media_complete,
    )
    policy_outcome = None
    if not native_authority["native_event_authority_verified"]:
        control_outcome = (
            "untrusted_structural_control_candidate"
            if episode_kind != "learned_policy_evaluation"
            else None
        )
    elif episode_kind == "zero_action_control":
        if not evidence["interpretability"]["zero_control_outcome_interpretable"]:
            control_outcome = "zero_action_control_harness_fault"
        elif deterministic_score["deterministic_success"]:
            control_outcome = "unexpected_zero_action_success_harness_task_blocker"
        else:
            control_outcome = "required_zero_action_failure_observed"
    elif episode_kind == "scripted_positive_control":
        if not evidence["interpretability"]["scripted_control_outcome_interpretable"]:
            control_outcome = "scripted_positive_control_harness_fault"
        elif (
            deterministic_score["deterministic_success"]
            and evidence["interpretability"]["manipulation_sequence_complete"]
        ):
            control_outcome = "required_scripted_positive_success_observed"
        else:
            control_outcome = "scripted_positive_harness_task_construction_blocker"
    else:
        control_outcome = None
    kind_interpretable = bool(
        evidence["interpretability"]["policy_outcome_interpretable"]
        if episode_kind == "learned_policy_evaluation"
        else evidence["interpretability"]["scripted_control_outcome_interpretable"]
        if episode_kind == "scripted_positive_control"
        else False
    )
    native_episode_evidence_admitted = bool(
        native_authority["native_event_authority_verified"]
        and kind_interpretable
        and (
            episode_kind == "learned_policy_evaluation"
            or evidence["interpretability"]["manipulation_sequence_complete"]
        )
    )
    native_episode_admitted_success = bool(
        episode_kind != "zero_action_control"
        and native_episode_evidence_admitted
        and deterministic_score["deterministic_success"]
        and evidence["interpretability"]["manipulation_sequence_complete"]
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "episode_kind": episode_kind,
        "episode_start_timestamp_ns": episode_start_timestamp_ns,
        "cell": cell,
        "prompt": prompt,
        "prompt_sha256": expected_prompt_digest,
        "actor": actor,
        "frozen_run_contract_digest": frozen_run["contract_digest"],
        "task_spec_sha256": task_spec_sha256,
        "entity_contract_digest": entity_contract["contract_digest"],
        "task_entities": entity_contract["task_entities"],
        "semantic_role_index": entity_contract["semantic_role_index"],
        "entity_ids": entity_ids,
        "reset_receipt": reset,
        "trace_thresholds": thresholds,
        "steps": steps,
        "terminal": terminal,
        "media_manifest": media_manifest,
        "native_trace_manifest": native_trace_manifest,
        "review_videos": review_videos,
        "media_validation_blockers": sorted(set(media_validation_blockers)),
        "native_event_authority": native_authority,
        "scorer_ready_samples": scorer_ready_samples,
        "deterministic_task_state_score": deterministic_score,
        "evidence": evidence,
        "policy_outcome": policy_outcome,
        "control_outcome": control_outcome,
        "native_episode_evidence_admitted": native_episode_evidence_admitted,
        "native_episode_admitted_deterministic_success": native_episode_admitted_success,
        "evaluation_admitted_deterministic_success": False,
        "evaluation_admission_blockers": (
            sorted(
                set(native_authority["blockers"])
                | {"identical_cell_controls_not_joined_by_cell_aggregator"}
            )
            if episode_kind == "learned_policy_evaluation"
            else list(native_authority["blockers"])
        ),
        "media_complete": media_complete,
        "media_status": (
            "complete"
            if media_complete
            else "incomplete_review_video_evidence"
            if media_validation_blockers
            else "byte_verified_media_untrusted_native_trace"
        ),
        "scoring_authority": (
            "trusted_runner_attested_native_state_deterministic_scorer"
            if native_authority["native_event_authority_verified"]
            else "untrusted_structural_candidate_state_projection_only"
        ),
        "deterministic_score_claim_status": (
            "trusted_runner_attested_native_state_score"
            if native_authority["native_event_authority_verified"]
            else "untrusted_structural_projection"
        ),
        "overview_used_by_policy": False,
        "overview_used_by_deterministic_scorer": False,
        "cell_family_changes_trace_contract": False,
        "claim_boundary": (
            "trusted_runner_native_simulator_trace_only_not_physical_material_"
            "equivalence_real_robot_performance_or_deployment_truth"
            if native_authority["native_event_authority_verified"]
            else "untrusted_structural_candidate_no_native_simulator_or_evaluation_claim"
        ),
    }
    receipt["receipt_digest"] = _canonical_digest(receipt)
    return receipt


def aggregate_native_deformable_cell_evaluation(
    *,
    zero_action_receipt: Mapping[str, Any],
    scripted_positive_receipt: Mapping[str, Any],
    learned_policy_receipt: Mapping[str, Any],
    expected_episode_receipt_digest_by_kind: Mapping[str, str],
    episode_replay_inputs_by_kind: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Join identical-cell controls to one learned episode, fail closed.

    The expected receipt digests are an external aggregator input, not fields
    copied from any episode.  Each admitted episode must additionally carry a
    cryptographically verified runner/native-event authority result.  Replay
    inputs cause the aggregator to re-materialize every episode from the sealed
    event artifacts, so a self-rehashed receipt cannot borrow authority from a
    different trusted trace.
    """

    expected_kinds = set(EPISODE_KINDS)
    expected_digests = _nonempty_digest_map(
        expected_episode_receipt_digest_by_kind,
        error="deformable_cell_evaluation_expected_receipt_digests_invalid",
    )
    if set(expected_digests) != expected_kinds:
        _raise("deformable_cell_evaluation_expected_receipt_digests_invalid")
    if (
        not isinstance(episode_replay_inputs_by_kind, Mapping)
        or set(episode_replay_inputs_by_kind) != expected_kinds
    ):
        _raise("deformable_cell_evaluation_replay_inputs_invalid")
    receipts = {
        "zero_action_control": _canonical_record(
            zero_action_receipt,
            digest_field="receipt_digest",
            error="deformable_cell_evaluation_zero_receipt_invalid",
        ),
        "scripted_positive_control": _canonical_record(
            scripted_positive_receipt,
            digest_field="receipt_digest",
            error="deformable_cell_evaluation_scripted_receipt_invalid",
        ),
        "learned_policy_evaluation": _canonical_record(
            learned_policy_receipt,
            digest_field="receipt_digest",
            error="deformable_cell_evaluation_policy_receipt_invalid",
        ),
    }
    blockers: list[str] = []
    for kind, receipt in receipts.items():
        if receipt.get("schema_version") != SCHEMA_VERSION or receipt.get("episode_kind") != kind:
            blockers.append(f"episode_kind_or_schema_mismatch:{kind}")
        if receipt.get("receipt_digest") != expected_digests[kind]:
            blockers.append(f"external_episode_receipt_digest_mismatch:{kind}")
        replay_inputs = episode_replay_inputs_by_kind[kind]
        if not isinstance(replay_inputs, Mapping):
            blockers.append(f"episode_replay_inputs_invalid:{kind}")
        else:
            try:
                replayed = materialize_native_deformable_episode_trace(**replay_inputs)
            except (NativeDeformableEpisodeTraceError, TypeError):
                blockers.append(f"episode_cryptographic_replay_failed:{kind}")
            else:
                if replayed != receipt:
                    blockers.append(f"episode_cryptographic_replay_mismatch:{kind}")
        authority = receipt.get("native_event_authority")
        if not isinstance(authority, Mapping) or not authority.get(
            "native_event_authority_verified"
        ):
            blockers.append(f"trusted_native_event_authority_missing:{kind}")

    join_fields = (
        "frozen_run_contract_digest",
        "task_spec_sha256",
        "entity_contract_digest",
        "prompt_sha256",
        "trace_thresholds",
        "cell",
    )
    policy_receipt = receipts["learned_policy_evaluation"]
    for kind, receipt in receipts.items():
        for field in join_fields:
            if receipt.get(field) != policy_receipt.get(field):
                blockers.append(f"identical_cell_join_mismatch:{kind}:{field}")

    reset_readbacks_by_kind: dict[str, dict[str, str]] = {}
    for kind, receipt in receipts.items():
        reset_receipt = receipt.get("reset_receipt")
        reset_readbacks = (
            reset_receipt.get("native_state_readback_sha256_by_entity_id")
            if isinstance(reset_receipt, Mapping)
            else None
        )
        if not isinstance(reset_readbacks, Mapping) or not reset_readbacks:
            blockers.append(f"identical_cell_native_reset_readbacks_invalid:{kind}")
            reset_readbacks_by_kind[kind] = {}
            continue
        normalized_readbacks: dict[str, str] = {}
        try:
            for raw_entity_id, raw_digest in reset_readbacks.items():
                entity_id = _string(
                    raw_entity_id,
                    error=f"identical_cell_native_reset_readbacks_invalid:{kind}",
                    identifier=True,
                )
                if entity_id in normalized_readbacks:
                    _raise(f"identical_cell_native_reset_readbacks_invalid:{kind}")
                normalized_readbacks[entity_id] = _digest(
                    raw_digest,
                    error=f"identical_cell_native_reset_readbacks_invalid:{kind}",
                )
        except NativeDeformableEpisodeTraceError:
            blockers.append(f"identical_cell_native_reset_readbacks_invalid:{kind}")
            reset_readbacks_by_kind[kind] = {}
            continue
        reset_readbacks_by_kind[kind] = dict(sorted(normalized_readbacks.items()))

    reset_reference = reset_readbacks_by_kind.get("zero_action_control", {})
    reset_reference_entities = set(reset_reference)
    if reset_reference:
        for kind in EPISODE_KINDS[1:]:
            observed = reset_readbacks_by_kind.get(kind, {})
            if not observed:
                continue
            if set(observed) != reset_reference_entities:
                blockers.append(f"identical_cell_native_reset_entity_set_mismatch:{kind}")
            for entity_id in sorted(reset_reference_entities & set(observed)):
                if observed[entity_id] != reset_reference[entity_id]:
                    blockers.append(
                        f"identical_cell_native_reset_readback_mismatch:{kind}:{entity_id}"
                    )

    zero = receipts["zero_action_control"]
    zero_score = zero.get("deterministic_task_state_score")
    if (
        zero.get("control_outcome") != "required_zero_action_failure_observed"
        or not isinstance(zero_score, Mapping)
        or zero_score.get("deterministic_success") is not False
    ):
        blockers.append("identical_cell_zero_action_control_not_passing")
    scripted = receipts["scripted_positive_control"]
    scripted_score = scripted.get("deterministic_task_state_score")
    if (
        scripted.get("control_outcome") != "required_scripted_positive_success_observed"
        or not isinstance(scripted_score, Mapping)
        or scripted_score.get("deterministic_success") is not True
        or scripted.get("native_episode_admitted_deterministic_success") is not True
    ):
        blockers.append("identical_cell_scripted_positive_control_not_passing")
    policy_score = policy_receipt.get("deterministic_task_state_score")
    if (
        not isinstance(policy_score, Mapping)
        or policy_receipt.get("native_episode_evidence_admitted") is not True
    ):
        blockers.append("learned_episode_native_admission_not_passing")
    policy_evidence = policy_receipt.get("evidence")
    policy_interpretability = (
        policy_evidence.get("interpretability") if isinstance(policy_evidence, Mapping) else None
    )
    policy_manipulation_sequence_complete = bool(
        isinstance(policy_interpretability, Mapping)
        and policy_interpretability.get("manipulation_sequence_complete") is True
    )
    policy_score_claims_success = bool(
        isinstance(policy_score, Mapping)
        and (
            policy_score.get("deterministic_success") is True
            or policy_score.get("outcome") == "succeeded"
        )
    )
    if policy_score_claims_success and (
        policy_receipt.get("native_episode_admitted_deterministic_success") is not True
        or not policy_manipulation_sequence_complete
    ):
        blockers.append("learned_episode_succeeded_without_admitted_manipulation_sequence")

    blockers = sorted(set(blockers))
    admitted = not blockers
    control_integrity_blocker_prefixes = (
        "episode_kind_or_schema_mismatch:",
        "external_episode_receipt_digest_mismatch:",
        "episode_replay_inputs_invalid:",
        "episode_cryptographic_replay_",
        "trusted_native_event_authority_missing:",
        "identical_cell_join_mismatch:",
        "identical_cell_native_reset_",
        "identical_cell_zero_action_control",
        "identical_cell_scripted_positive_control",
    )
    controls_passed = not any(
        blocker.startswith(control_integrity_blocker_prefixes) for blocker in blockers
    )
    admitted_success = bool(
        admitted
        and policy_score.get("deterministic_success") is True
        and policy_receipt.get("native_episode_admitted_deterministic_success") is True
        and policy_manipulation_sequence_complete
    )
    result = {
        "schema_version": CELL_EVALUATION_SCHEMA_VERSION,
        "cell": policy_receipt.get("cell"),
        "candidate_id": (
            policy_receipt.get("actor", {}).get("candidate_id")
            if isinstance(policy_receipt.get("actor"), Mapping)
            else None
        ),
        "episode_receipt_digest_by_kind": {
            kind: receipt.get("receipt_digest") for kind, receipt in receipts.items()
        },
        "native_reset_readback_sha256_by_entity_id_by_episode_kind": {
            kind: reset_readbacks_by_kind.get(kind, {}) for kind in EPISODE_KINDS
        },
        "identical_cell_native_reset_readback_sha256_by_entity_id": (
            reset_reference
            if reset_reference
            and not any(blocker.startswith("identical_cell_native_reset_") for blocker in blockers)
            else None
        ),
        "identical_cell_controls_passed": controls_passed,
        "evaluation_admitted": admitted,
        "evaluation_admitted_deterministic_success": admitted_success,
        "policy_outcome": policy_score.get("outcome") if admitted else None,
        "blockers": blockers,
        "claim_boundary": (
            "trusted_native_simulator_identical_cell_evaluation_only_not_physical_truth"
        ),
    }
    result["receipt_digest"] = _canonical_digest(result)
    return result


__all__ = [
    "ACTION_REPLAY_CONTRACT_ID",
    "ACTION_REPLAY_SCHEMA_VERSION",
    "CAMERA_IDS",
    "CELL_EVALUATION_SCHEMA_VERSION",
    "EPISODE_KINDS",
    "FROZEN_RUN_SCHEMA_VERSION",
    "FROZEN_RUN_SEAL_SCHEMA_VERSION",
    "H264_DECODE_VERIFIER_CONTRACT_ID",
    "MEDIA_MANIFEST_SCHEMA_VERSION",
    "MINIMUM_ARM_MOTION_EPSILON_RAD",
    "MINIMUM_GRIPPER_MOTION_EPSILON_M",
    "NATIVE_TRACE_MANIFEST_SCHEMA_VERSION",
    "NativeDeformableEpisodeTraceError",
    "RESET_STATE_PROJECTION_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "aggregate_native_deformable_cell_evaluation",
    "materialize_native_deformable_episode_trace",
]
