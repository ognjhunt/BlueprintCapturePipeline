"""Collect immutable prerequisites before inserting one movable task asset.

This boundary is intentionally narrower than native readiness.  It verifies a
path-bound evidence graph for the scene, frozen task, rights, task entities,
placements, deterministic scorer, cameras, scenario cells, static runtime
inventory, and execution-trust configuration.  Only after every non-movable
asset gate passes does it expose one typed asset-insertion slot.

The slot is not a native qualification.  Dynamic composition, contact, reset,
camera application, controls, policy execution, and provider lifecycle proof
remain dependent native gates.  The collector accepts no in-memory evidence
mapping, follows no symlinks, and never derives a native-success claim from a
caller-authored boolean.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from PIL import Image, UnidentifiedImageError

from .adp_task_scoring import (
    TASK_KIND_ARTICULATED_OPEN_CLOSE,
    TASK_KIND_DEFORMABLE_TRANSFER,
    TASK_KIND_RIGID_PICK_PLACE,
    TASK_SPEC_SCHEMA_VERSION,
    TaskNeutralScoringError,
    validate_articulated_task_spec,
    validate_deformable_task_spec,
)
from .composed_paired_entity_placement import (
    RECEIPT_SCHEMA_VERSION as COMPOSED_PLACEMENT_SCHEMA_VERSION,
)
from .composed_paired_entity_placement import plan_composed_paired_entity_placement
from .decision_evidence_contracts import canonical_digest
from .deformable_native_capability_preflight import (
    DYNAMIC_NATIVE_CANARY_GATES,
    MATRIX_SCHEMA_VERSION as DEFORMABLE_PREFLIGHT_MATRIX_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION as DEFORMABLE_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    build_deformable_native_capability_preflight,
)
from .task_entity_asset_candidate import (
    SCHEMA_VERSION as TASK_ENTITY_ASSET_CANDIDATE_SCHEMA_VERSION,
)
from .task_entity_asset_candidate import (
    TaskEntityAssetCandidateError,
    materialize_task_entity_asset_candidate,
)
from .semantic_review_attestation import (
    SemanticReviewAttestationError,
    TRUSTED_PUBLIC_KEY_SHA256_ENV as SEMANTIC_AUTHORITY_PUBLIC_KEY_SHA256_ENV,
    semantic_frame_evidence_digest,
    verify_semantic_review_attestation,
)
from .registered_static_receptacle_asset import (
    CANDIDATE_FILENAME as REGISTERED_RECEPTACLE_CANDIDATE_FILENAME,
    RECEIPT_FILENAME as REGISTERED_RECEPTACLE_RECEIPT_FILENAME,
    VISUAL_BASIS_FILENAME as REGISTERED_RECEPTACLE_VISUAL_BASIS_FILENAME,
    RegisteredStaticReceptacleAssetError,
    build_registered_static_receptacle_asset,
)
from .trusted_execution_envelope import SCHEMA_VERSION as TRUSTED_ENVELOPE_SCHEMA_VERSION


INPUT_SCHEMA_VERSION = "task_preinsertion_readiness_input.v1"
RECEIPT_SCHEMA_VERSION = "task_preinsertion_readiness_receipt.v1"
SCENE_SCHEMA_VERSION = "task_preinsertion_scene_freeze.v1"
TASK_SCHEMA_VERSION = "task_preinsertion_task_freeze.v1"
RIGHTS_SCHEMA_VERSION = "task_preinsertion_rights.v1"
ENTITY_SCHEMA_VERSION = "task_preinsertion_entity_inventory.v1"
PLACEMENT_SCHEMA_VERSION = "task_preinsertion_placement_suite.v1"
SCORER_SCHEMA_VERSION = "task_preinsertion_scorer_freeze.v1"
CAMERA_SCHEMA_VERSION = "task_preinsertion_camera_freeze.v1"
SCENARIO_SCHEMA_VERSION = "task_preinsertion_scenario_suite.v1"
RUNTIME_SCHEMA_VERSION = "task_preinsertion_runtime_static_preflight.v1"
TRUST_SCHEMA_VERSION = "task_preinsertion_trust_policy.v1"
SOURCE_EVIDENCE_SCHEMA_VERSION = "task_preinsertion_source_observation_evidence.v1"
REGISTRATION_EVIDENCE_SCHEMA_VERSION = "task_preinsertion_registration_evidence.v1"
TOPOLOGY_EVIDENCE_SCHEMA_VERSION = "task_preinsertion_topology_evidence.v1"
PREFLIGHT_OBSERVATIONS_SCHEMA_VERSION = "task_preinsertion_deformable_preflight_observations.v1"
ENGINEERED_ASSET_EVIDENCE_SCHEMA_VERSION = "task_preinsertion_engineered_asset_evidence.v1"
RIGHTS_EVIDENCE_SCHEMA_VERSION = "task_preinsertion_rights_evidence.v1"
REGISTRATION_TRANSFORM_SCHEMA_VERSION = "task_preinsertion_registration_transform.v1"
TOPOLOGY_SURVEY_SCHEMA_VERSION = "task_preinsertion_topology_survey.v1"
CAMERA_EXTRINSICS_SCHEMA_VERSION = "task_preinsertion_camera_extrinsics.v1"
RESOLVED_SCENARIO_CELL_SCHEMA_VERSION = "task_preinsertion_resolved_scenario_cell.v1"
RIGHTS_INTERPRETATION_VERSION = "task_preinsertion_rights_interpretation.v1"
RIGHTS_AUTHORITY_PUBLIC_KEY_SHA256_ENV = "BLUEPRINT_RIGHTS_AUTHORITY_PUBLIC_KEY_SHA256"
TASK_FREEZE_AUTHORITY_PUBLIC_KEY_SHA256_ENV = "BLUEPRINT_TASK_FREEZE_AUTHORITY_PUBLIC_KEY_SHA256"
RIGHTS_SIGNATURE_DOMAIN = b"blueprint.task_preinsertion_rights_evidence.v1\x00"
TASK_FREEZE_SIGNATURE_DOMAIN = b"blueprint.task_preinsertion_task_freeze.v1\x00"
SOURCE_OBSERVATION_SIGNATURE_DOMAIN = (
    b"blueprint.task_preinsertion_source_observation_evidence.v1\x00"
)
REGISTERED_RECEPTACLE_RECEIPT_SCHEMA_VERSION = "registered_static_receptacle_asset.v1"
ENGINEERED_RECEPTACLE_VISUAL_BASIS_SCHEMA_VERSION = "engineered_receptacle_visual_design_basis.v1"
REGISTERED_COLLISION_TOPOLOGY_SCHEMA_VERSION = "interiorgs_sage_collision_component_topology.v2"
REGISTERED_RECEPTACLE_REPLAY_REQUEST_SCHEMA_VERSION = (
    "registered_static_receptacle_replay_request.v1"
)
VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION = "task_preinsertion_visual_observation_provenance.v1"

_TASK_KINDS = frozenset(
    {
        TASK_KIND_RIGID_PICK_PLACE,
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TASK_KIND_DEFORMABLE_TRANSFER,
    }
)
_TARGET_BY_TASK_KIND = {
    TASK_KIND_RIGID_PICK_PLACE: ("movable_rigid", "rigid_body"),
    TASK_KIND_ARTICULATED_OPEN_CLOSE: ("articulated_fixture", "articulation"),
    TASK_KIND_DEFORMABLE_TRANSFER: ("movable_deformable", "deformable_volume"),
}
_REQUIRED_ROLES_BY_TASK_KIND = {
    TASK_KIND_RIGID_PICK_PLACE: frozenset({"movable_rigid", "robot"}),
    TASK_KIND_ARTICULATED_OPEN_CLOSE: frozenset({"articulated_fixture", "robot"}),
    TASK_KIND_DEFORMABLE_TRANSFER: frozenset(
        {
            "movable_deformable",
            "destination_receptacle",
            "support_surface",
            "obstacle",
            "robot",
        }
    ),
}
_PHYSICS_BY_ROLE = {
    "movable_rigid": frozenset({"rigid_body"}),
    "articulated_fixture": frozenset({"articulation"}),
    "movable_deformable": frozenset({"deformable_volume"}),
    "destination_receptacle": frozenset({"rigid_body", "static_collider"}),
    "support_surface": frozenset({"rigid_body", "static_collider"}),
    "obstacle": frozenset({"rigid_body", "articulation", "static_collider"}),
    "robot": frozenset({"robot_articulation"}),
}
_CORE_PURPOSES = (
    "scene",
    "task",
    "rights",
    "entities",
    "placement",
    "scorer",
    "cameras",
    "scenario",
    "runtime",
    "trust",
)
_CORE_SCHEMAS = {
    "scene": SCENE_SCHEMA_VERSION,
    "task": TASK_SCHEMA_VERSION,
    "rights": RIGHTS_SCHEMA_VERSION,
    "entities": ENTITY_SCHEMA_VERSION,
    "placement": PLACEMENT_SCHEMA_VERSION,
    "scorer": SCORER_SCHEMA_VERSION,
    "cameras": CAMERA_SCHEMA_VERSION,
    "scenario": SCENARIO_SCHEMA_VERSION,
    "runtime": RUNTIME_SCHEMA_VERSION,
    "trust": TRUST_SCHEMA_VERSION,
}
_EXPECTED_PREFLIGHT_DYNAMIC_GATES = frozenset(
    check_id for check_id, _required_proof in DYNAMIC_NATIVE_CANARY_GATES
)
_DEPENDENT_NATIVE_GATES_BY_TASK_KIND = {
    TASK_KIND_RIGID_PICK_PLACE: (
        "native_rigid_asset_composition_and_schema_readback",
        "native_rigid_grasp_contact_lift_release",
        "native_rigid_reset_and_state_readback",
        "native_renderer_camera_capture",
        "native_policy_adapter_action_delivery",
        "native_applied_parameter_readback",
    ),
    TASK_KIND_ARTICULATED_OPEN_CLOSE: (
        "native_articulation_composition_and_joint_schema_readback",
        "native_articulation_contact_and_joint_motion",
        "native_articulation_reset_and_state_readback",
        "native_renderer_camera_capture",
        "native_policy_adapter_action_delivery",
        "native_applied_parameter_readback",
    ),
    TASK_KIND_DEFORMABLE_TRANSFER: (
        "native_deformable_composition_and_cooking",
        "native_cuda_warp_execution",
        "native_genuine_gripper_deformable_contact_lift_release",
        "native_nodal_reset_repeatability",
        "native_deformable_settling_and_strain_readback",
        "native_renderer_camera_capture",
        "native_policy_adapter_action_delivery",
        "native_applied_parameter_readback",
    ),
}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_ARTIFACT_BYTES = 256 * 1024 * 1024
_MAX_BINDINGS = 128
_READ_CHUNK_BYTES = 1024 * 1024


class TaskPreinsertionReadinessError(ValueError):
    """Stable fail-closed structural errors at the file boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _is_identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER_RE.fullmatch(value))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and bool(_DIGEST_RE.fullmatch(value))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def rights_evidence_signature_message(value: Mapping[str, Any]) -> bytes:
    """Return the domain-separated message signed by a rights authority.

    The retained document identity, verifier-owned interpretation identity, and
    every interpreted permission are signed.  The transport receipt digest and
    signature bytes are excluded to avoid a circular encoding.
    """

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (RecursionError, TypeError, ValueError) as exc:
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_rights_evidence_signature_payload_invalid"]
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("authority"), dict):
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_rights_evidence_signature_payload_invalid"]
        )
    payload.pop("receipt_digest", None)
    payload["authority"].pop("signature_base64", None)
    return RIGHTS_SIGNATURE_DOMAIN + _canonical_json_bytes(payload)


def task_freeze_signature_message(value: Mapping[str, Any]) -> bytes:
    """Return the exact domain-separated task freeze signed by its authority.

    The signature covers the prompt, canonical task-spec digest, complete
    prompt/spec-suite digest, candidate set, and every freeze assertion.  Only
    the circular transport digest and signature bytes are excluded.
    """

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (RecursionError, TypeError, ValueError) as exc:
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_task_freeze_signature_payload_invalid"]
        ) from exc
    authority = payload.get("freeze_authority") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(authority, dict):
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_task_freeze_signature_payload_invalid"]
        )
    payload.pop("receipt_digest", None)
    authority.pop("signature_base64", None)
    return TASK_FREEZE_SIGNATURE_DOMAIN + _canonical_json_bytes(payload)


def prompt_task_spec_freeze_digest(
    *, task_kind: str, prompt: str, cell_task_spec_digests: Mapping[str, str]
) -> str:
    """Bind one exact prompt to every frozen cell-specific task specification."""

    if (
        task_kind not in _TASK_KINDS
        or not isinstance(prompt, str)
        or not prompt.strip()
        or not isinstance(cell_task_spec_digests, Mapping)
        or not cell_task_spec_digests
        or any(
            not _is_identifier(cell_id) or not _is_digest(spec_digest)
            for cell_id, spec_digest in cell_task_spec_digests.items()
        )
    ):
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_prompt_task_spec_freeze_payload_invalid"]
        )
    return canonical_digest(
        {
            "task_kind": task_kind,
            "prompt": prompt,
            "cell_task_spec_digests": {
                str(cell_id): str(spec_digest)
                for cell_id, spec_digest in sorted(cell_task_spec_digests.items())
            },
        }
    )


def _task_freeze_authority_signature_valid(value: Mapping[str, Any]) -> bool:
    authority = value.get("freeze_authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "authority_id",
        "key_id",
        "public_key_base64",
        "public_key_sha256",
        "signature_base64",
    }:
        return False
    if not _is_identifier(authority.get("authority_id")) or not _is_identifier(
        authority.get("key_id")
    ):
        return False
    try:
        public_key = base64.b64decode(authority.get("public_key_base64", ""), validate=True)
        signature = base64.b64decode(authority.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error):
        return False
    if (
        len(public_key) != 32
        or len(signature) != 64
        or base64.b64encode(public_key).decode("ascii") != authority.get("public_key_base64")
        or base64.b64encode(signature).decode("ascii") != authority.get("signature_base64")
    ):
        return False
    fingerprint = _sha256_bytes(public_key)
    configured = os.getenv(TASK_FREEZE_AUTHORITY_PUBLIC_KEY_SHA256_ENV)
    if (
        authority.get("public_key_sha256") != fingerprint
        or not _is_digest(configured)
        or configured != fingerprint
    ):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            task_freeze_signature_message(value),
        )
    except (InvalidSignature, TaskPreinsertionReadinessError, ValueError):
        return False
    return True


def source_observation_signature_message(value: Mapping[str, Any]) -> bytes:
    """Return the exact source-observation payload signed by semantic authority.

    Bounds, rest state, support relation, and every cited decoded frame identity
    are all inside the signed payload.  The transport receipt and signature bytes
    are excluded to avoid circular encodings; the selected authority identity and
    public key remain signed.
    """

    try:
        payload = json.loads(json.dumps(value, allow_nan=False))
    except (RecursionError, TypeError, ValueError) as exc:
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_source_observation_signature_payload_invalid"]
        ) from exc
    authority = payload.get("semantic_authority") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(authority, dict):
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_source_observation_signature_payload_invalid"]
        )
    payload.pop("receipt_digest", None)
    authority.pop("signature_base64", None)
    return SOURCE_OBSERVATION_SIGNATURE_DOMAIN + _canonical_json_bytes(payload)


def _source_observation_authority_signature_valid(value: Mapping[str, Any]) -> bool:
    authority = value.get("semantic_authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "authority_id",
        "key_id",
        "public_key_base64",
        "public_key_sha256",
        "signature_base64",
    }:
        return False
    if not _is_identifier(authority.get("authority_id")) or not _is_identifier(
        authority.get("key_id")
    ):
        return False
    try:
        public_key = base64.b64decode(authority.get("public_key_base64", ""), validate=True)
        signature = base64.b64decode(authority.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error):
        return False
    if (
        len(public_key) != 32
        or len(signature) != 64
        or base64.b64encode(public_key).decode("ascii") != authority.get("public_key_base64")
        or base64.b64encode(signature).decode("ascii") != authority.get("signature_base64")
    ):
        return False
    fingerprint = _sha256_bytes(public_key)
    configured = os.getenv(SEMANTIC_AUTHORITY_PUBLIC_KEY_SHA256_ENV)
    if (
        authority.get("public_key_sha256") != fingerprint
        or not _is_digest(configured)
        or configured != fingerprint
    ):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            source_observation_signature_message(value),
        )
    except (InvalidSignature, TaskPreinsertionReadinessError, ValueError):
        return False
    return True


def _rights_authority_signature_valid(value: Mapping[str, Any]) -> bool:
    authority = value.get("authority")
    if not isinstance(authority, Mapping) or set(authority) != {
        "authority_id",
        "key_id",
        "public_key_base64",
        "public_key_sha256",
        "signature_base64",
    }:
        return False
    if not _is_identifier(authority.get("authority_id")) or not _is_identifier(
        authority.get("key_id")
    ):
        return False
    try:
        public_key = base64.b64decode(authority.get("public_key_base64", ""), validate=True)
        signature = base64.b64decode(authority.get("signature_base64", ""), validate=True)
    except (TypeError, ValueError, binascii.Error):
        return False
    if (
        len(public_key) != 32
        or len(signature) != 64
        or base64.b64encode(public_key).decode("ascii") != authority.get("public_key_base64")
        or base64.b64encode(signature).decode("ascii") != authority.get("signature_base64")
    ):
        return False
    fingerprint = _sha256_bytes(public_key)
    configured = os.getenv(RIGHTS_AUTHORITY_PUBLIC_KEY_SHA256_ENV)
    if (
        authority.get("public_key_sha256") != fingerprint
        or not _is_digest(configured)
        or configured != fingerprint
    ):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            rights_evidence_signature_message(value),
        )
    except (InvalidSignature, TaskPreinsertionReadinessError, ValueError):
        return False
    return True


def _finite_vector(value: Any, *, length: int) -> bool:
    return bool(
        isinstance(value, list)
        and len(value) == length
        and all(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in value
        )
    )


def _finite_vectors_close(
    left: Any,
    right: Any,
    *,
    length: int,
    absolute_tolerance: float = 1.0e-7,
) -> bool:
    return bool(
        _finite_vector(left, length=length)
        and _finite_vector(right, length=length)
        and all(
            abs(float(left[index]) - float(right[index])) <= absolute_tolerance
            for index in range(length)
        )
    )


def _decoded_visual_identity(content: bytes) -> dict[str, Any] | None:
    """Decode one bounded lossless observation and return its exact RGB identity."""

    try:
        with Image.open(BytesIO(content)) as image:
            if image.format != "PNG":
                return None
            width, height = image.size
            if width < 2 or height < 2 or width * height > 16 * 1024 * 1024:
                return None
            image.load()
            rgb = image.convert("RGB").tobytes()
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError):
        return None
    unique_pixels = {rgb[index : index + 3] for index in range(0, len(rgb), 3)}
    if len(unique_pixels) < 2:
        return None
    return {
        "width": width,
        "height": height,
        "decoded_rgb_sha256": _sha256_bytes(rgb),
    }


def _nonempty_unique_strings(value: Any) -> bool:
    return bool(
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item for item in value)
        and len(set(value)) == len(value)
    )


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        return []
    if any(not isinstance(row, Mapping) for row in value):
        return []
    return list(value)


def _strict_json_object(content: bytes, *, error: str) -> dict[str, Any]:
    if not content or content.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        raise TaskPreinsertionReadinessError([error])

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise TaskPreinsertionReadinessError([error]) from exc
    if not isinstance(parsed, dict):
        raise TaskPreinsertionReadinessError([error])
    return parsed


def _read_absolute_once(
    path_value: str | Path, *, label: str, maximum_size: int
) -> tuple[bytes, Path]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise TaskPreinsertionReadinessError([f"task_preinsertion_{label}_no_follow_unavailable"])
    path = Path(os.path.abspath(Path(path_value).expanduser()))
    descriptors: list[int] = []
    try:
        anchor = path.anchor
        if not anchor or not path.name:
            raise OSError("absolute_path_invalid")
        parent = os.open(
            anchor,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
        descriptors.append(parent)
        relative_parts = path.parts[1:]
        for component in relative_parts[:-1]:
            parent = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            descriptors.append(parent)
        descriptor = os.open(relative_parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum_size:
            raise TaskPreinsertionReadinessError([f"task_preinsertion_{label}_file_invalid"])
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if len(content) != before.st_size or identity_before != identity_after:
            raise TaskPreinsertionReadinessError(
                [f"task_preinsertion_{label}_changed_while_reading"]
            )
        return content, path
    except TaskPreinsertionReadinessError:
        raise
    except OSError as exc:
        raise TaskPreinsertionReadinessError([f"task_preinsertion_{label}_file_invalid"]) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _binding_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise TaskPreinsertionReadinessError(["task_preinsertion_binding_relative_path_invalid"])
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise TaskPreinsertionReadinessError(["task_preinsertion_binding_relative_path_invalid"])
    return path


def _read_relative_once(root: Path, relative: PurePosixPath, *, label: str) -> tuple[bytes, Path]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise TaskPreinsertionReadinessError([f"task_preinsertion_{label}_no_follow_unavailable"])
    descriptors: list[int] = []
    try:
        absolute_root = Path(os.path.abspath(root))
        anchor = absolute_root.anchor
        if not anchor or not absolute_root.name:
            raise OSError("artifact_root_invalid")
        root_descriptor = os.open(
            anchor,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
        )
        descriptors.append(root_descriptor)
        for component in absolute_root.parts[1:]:
            root_descriptor = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
                dir_fd=root_descriptor,
            )
            descriptors.append(root_descriptor)
        parent = root_descriptor
        for component in relative.parts[:-1]:
            parent = os.open(
                component,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            descriptors.append(parent)
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_ARTIFACT_BYTES
        ):
            raise OSError("not_regular_or_size_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        )
        if len(content) != before.st_size or identity_before != identity_after:
            raise TaskPreinsertionReadinessError(
                [f"task_preinsertion_{label}_changed_while_reading"]
            )
        return content, root.joinpath(*relative.parts)
    except TaskPreinsertionReadinessError:
        raise
    except OSError as exc:
        raise TaskPreinsertionReadinessError([f"task_preinsertion_{label}_file_invalid"]) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _materialize_verifier_snapshot(
    *, root: Path, filename: str, artifact: Mapping[str, Any]
) -> Path:
    """Write one already-verified artifact snapshot into a private temp root.

    Replay helpers accept paths, so passing the original evidence path would
    reopen mutable bytes after ``_load_bindings`` completed its descriptor-
    scoped read.  A fresh exclusive file preserves the exact bytes whose digest
    was admitted while still satisfying those path-only helper interfaces.
    """

    content = artifact.get("content")
    if (
        not isinstance(content, bytes)
        or not content
        or not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise TaskPreinsertionReadinessError(["task_preinsertion_verifier_snapshot_invalid"])
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / filename
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("snapshot_write_incomplete")
            remaining = remaining[written:]
    except OSError as exc:
        raise TaskPreinsertionReadinessError(
            ["task_preinsertion_verifier_snapshot_invalid"]
        ) from exc
    finally:
        os.close(descriptor)
    return path


def _normalize_verifier_path_metadata(value: Any, expected: Any) -> Any:
    """Return ``value`` with verifier-local path metadata projected to ``expected``.

    Some released verifiers retain the absolute path used to read an artifact
    alongside its digest and size.  Replaying from verifier-owned snapshots is
    intentionally path-distinct from the original authoring run, so absolute
    transport paths are not deterministic evidence.  All non-path fields,
    including the loaded-byte identities, remain exact.
    """

    if isinstance(value, Mapping) and isinstance(expected, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            expected_item = expected.get(key)
            if key == "path" and isinstance(item, str) and isinstance(expected_item, str):
                normalized[key] = expected_item
            else:
                normalized[key] = _normalize_verifier_path_metadata(item, expected_item)
        return normalized
    if isinstance(value, list) and isinstance(expected, list) and len(value) == len(expected):
        return [
            _normalize_verifier_path_metadata(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        ]
    return value


def _manifest_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "run_id",
        "scene_id",
        "task_id",
        "task_kind",
        "candidate_ids",
        "asset_slot",
        "bindings",
        "manifest_digest",
    }
    errors: list[str] = []
    if set(manifest) != expected_fields or manifest.get("schema_version") != INPUT_SCHEMA_VERSION:
        errors.append("task_preinsertion_manifest_contract_invalid")
    for field in ("run_id", "scene_id", "task_id"):
        if not _is_identifier(manifest.get(field)):
            errors.append(f"task_preinsertion_manifest_{field}_invalid")
    task_kind = manifest.get("task_kind")
    if not isinstance(task_kind, str) or task_kind not in _TASK_KINDS:
        errors.append("task_preinsertion_manifest_task_kind_invalid")
    candidates = manifest.get("candidate_ids")
    candidate_ids_valid = bool(
        isinstance(candidates, list)
        and len(candidates) == 2
        and all(_is_identifier(value) for value in candidates)
    )
    if not candidate_ids_valid or (
        isinstance(candidates, list)
        and (len(set(candidates)) != 2 or candidates != sorted(candidates))
    ):
        errors.append("task_preinsertion_manifest_candidate_ids_invalid")

    slot = manifest.get("asset_slot")
    slot_fields = {"entity_id", "semantic_role", "physics_type", "status", "blocker_code"}
    if not isinstance(slot, Mapping) or set(slot) != slot_fields:
        errors.append("task_preinsertion_manifest_asset_slot_invalid")
        slot = {}
    expected_role, expected_physics = (
        _TARGET_BY_TASK_KIND.get(task_kind, (None, None))
        if isinstance(task_kind, str)
        else (None, None)
    )
    expected_blocker = (
        f"simready_{expected_role}_asset_and_native_insertion_required" if expected_role else None
    )
    if (
        not _is_identifier(slot.get("entity_id"))
        or slot.get("semantic_role") != expected_role
        or slot.get("physics_type") != expected_physics
        or slot.get("status") != "unresolved"
        or slot.get("blocker_code") != expected_blocker
    ):
        errors.append("task_preinsertion_manifest_asset_slot_invalid")

    bindings = manifest.get("bindings")
    if (
        not isinstance(bindings, list)
        or not bindings
        or len(bindings) > _MAX_BINDINGS
        or any(not isinstance(row, Mapping) for row in bindings)
    ):
        errors.append("task_preinsertion_manifest_bindings_invalid")
        bindings = []
    normalized_bindings: list[dict[str, Any]] = []
    ids: set[str] = set()
    purposes: dict[str, int] = {}
    for row in bindings:
        fields = {
            "binding_id",
            "purpose",
            "relative_path",
            "sha256",
            "content_type",
            "schema_version",
        }
        if set(row) != fields:
            errors.append("task_preinsertion_manifest_binding_fields_invalid")
            continue
        binding_id = row.get("binding_id")
        purpose = row.get("purpose")
        content_type = row.get("content_type")
        schema_version = row.get("schema_version")
        try:
            relative = _binding_path(row.get("relative_path"))
        except TaskPreinsertionReadinessError as exc:
            errors.extend(exc.errors)
            continue
        if not _is_identifier(binding_id) or binding_id in ids:
            errors.append("task_preinsertion_manifest_binding_id_invalid")
        ids.add(str(binding_id))
        if not isinstance(purpose, str) or purpose not in {
            *_CORE_PURPOSES,
            "supporting_evidence",
        }:
            errors.append(f"task_preinsertion_manifest_binding_purpose_invalid:{binding_id}")
        else:
            purposes[str(purpose)] = purposes.get(str(purpose), 0) + 1
        if not _is_digest(row.get("sha256")):
            errors.append(f"task_preinsertion_manifest_binding_digest_invalid:{binding_id}")
        if not isinstance(content_type, str) or content_type not in {"json", "opaque"}:
            errors.append(f"task_preinsertion_manifest_binding_content_type_invalid:{binding_id}")
        if content_type == "json" and not _is_identifier(schema_version):
            errors.append(f"task_preinsertion_manifest_binding_schema_invalid:{binding_id}")
        if content_type == "opaque" and schema_version is not None:
            errors.append(f"task_preinsertion_manifest_binding_schema_invalid:{binding_id}")
        if (
            isinstance(purpose, str)
            and purpose in _CORE_SCHEMAS
            and (content_type != "json" or schema_version != _CORE_SCHEMAS[purpose])
        ):
            errors.append(f"task_preinsertion_manifest_core_schema_invalid:{purpose}")
        normalized_bindings.append(
            {
                **dict(row),
                "relative_path": relative.as_posix(),
            }
        )
    for purpose in _CORE_PURPOSES:
        if purposes.get(purpose) != 1:
            errors.append(f"task_preinsertion_manifest_core_binding_invalid:{purpose}")
    try:
        expected_manifest_digest = canonical_digest(manifest, digest_field="manifest_digest")
    except (RecursionError, TypeError, ValueError):
        expected_manifest_digest = None
    if manifest.get("manifest_digest") != expected_manifest_digest:
        errors.append("task_preinsertion_manifest_digest_invalid")
    if errors:
        raise TaskPreinsertionReadinessError(errors)
    normalized = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    normalized["bindings"] = sorted(normalized_bindings, key=lambda row: row["binding_id"])
    return normalized


def _load_bindings(
    *, manifest: Mapping[str, Any], root: Path
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    blockers: list[str] = []
    retained_size = 0
    budget_exhausted = False
    for binding in manifest["bindings"]:
        binding_id = binding["binding_id"]
        relative = PurePosixPath(binding["relative_path"])
        if budget_exhausted or retained_size >= _MAX_TOTAL_ARTIFACT_BYTES:
            blockers.append("task_preinsertion_artifact_set_size_invalid")
            continue
        try:
            content, path = _read_relative_once(root, relative, label=f"binding_{binding_id}")
        except TaskPreinsertionReadinessError as exc:
            blockers.extend(exc.errors)
            continue
        if retained_size + len(content) > _MAX_TOTAL_ARTIFACT_BYTES:
            blockers.append("task_preinsertion_artifact_set_size_invalid")
            budget_exhausted = True
            continue
        retained_size += len(content)
        digest = _sha256_bytes(content)
        if digest != binding["sha256"]:
            blockers.append(f"task_preinsertion_binding_digest_mismatch:{binding_id}")
            continue
        payload: dict[str, Any] | None = None
        if binding["content_type"] == "json":
            try:
                payload = _strict_json_object(
                    content,
                    error=f"task_preinsertion_binding_json_invalid:{binding_id}",
                )
            except TaskPreinsertionReadinessError as exc:
                blockers.extend(exc.errors)
                continue
            if payload.get("schema_version") != binding["schema_version"]:
                blockers.append(f"task_preinsertion_binding_schema_mismatch:{binding_id}")
                continue
        artifacts[binding_id] = {
            "binding": binding,
            "payload": payload,
            "content": content,
            "path": path,
        }
        receipts.append(
            {
                "binding_id": binding_id,
                "purpose": binding["purpose"],
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": len(content),
                "content_type": binding["content_type"],
                "schema_version": binding["schema_version"],
            }
        )
    return artifacts, sorted(receipts, key=lambda row: row["binding_id"]), sorted(set(blockers))


def _core_payload(
    artifacts: Mapping[str, Mapping[str, Any]], purpose: str
) -> tuple[Mapping[str, Any], str | None]:
    rows = [
        (binding_id, artifact)
        for binding_id, artifact in artifacts.items()
        if artifact["binding"]["purpose"] == purpose
    ]
    if len(rows) != 1 or not isinstance(rows[0][1].get("payload"), Mapping):
        return {}, None
    return rows[0][1]["payload"], rows[0][0]


def _receipt_digest_valid(value: Mapping[str, Any], field: str) -> bool:
    try:
        expected = canonical_digest(value, digest_field=field)
    except (RecursionError, TypeError, ValueError):
        return False
    return _is_digest(value.get(field)) and value.get(field) == expected


def _canonical_digest_or_none(value: Mapping[str, Any]) -> str | None:
    try:
        return canonical_digest(value)
    except (RecursionError, TypeError, ValueError):
        return None


def _typed_payload(
    artifacts: Mapping[str, Mapping[str, Any]],
    binding_id: Any,
    *,
    schema_version: str,
) -> Mapping[str, Any]:
    if not _is_identifier(binding_id):
        return {}
    artifact = artifacts.get(str(binding_id), {})
    payload = artifact.get("payload")
    if (
        artifact.get("binding", {}).get("content_type") != "json"
        or artifact.get("binding", {}).get("schema_version") != schema_version
        or not isinstance(payload, Mapping)
        or payload.get("schema_version") != schema_version
    ):
        return {}
    return payload


def _binding_matches(
    artifacts: Mapping[str, Mapping[str, Any]],
    binding_id: Any,
    *,
    sha256: Any,
) -> bool:
    return bool(
        _is_identifier(binding_id)
        and _is_digest(sha256)
        and binding_id in artifacts
        and artifacts[str(binding_id)].get("binding", {}).get("sha256") == sha256
    )


def _scene_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    fields = {
        "schema_version",
        "scene_id",
        "status",
        "appearance",
        "collision",
        "registration",
        "topology",
        "receipt_digest",
    }
    if set(value) != fields or value.get("scene_id") != manifest["scene_id"]:
        errors.append("task_preinsertion_scene_contract_invalid")
    if value.get("status") != "frozen" or not _receipt_digest_valid(value, "receipt_digest"):
        errors.append("task_preinsertion_scene_not_frozen")
    sources: dict[str, Mapping[str, Any]] = {}
    source_fields = {
        "source_id",
        "source_kind",
        "revision",
        "source_path",
        "size_bytes",
        "sha256",
        "coordinate_frame_id",
        "rights_source_id",
    }
    for role in ("appearance", "collision"):
        row = value.get(role)
        if not isinstance(row, Mapping) or set(row) != source_fields:
            errors.append(f"task_preinsertion_scene_{role}_invalid")
            continue
        if (
            not _is_identifier(row.get("source_id"))
            or not _is_identifier(row.get("source_kind"))
            or not isinstance(row.get("revision"), str)
            or not row.get("revision")
            or not isinstance(row.get("source_path"), str)
            or not row.get("source_path")
            or type(row.get("size_bytes")) is not int
            or row.get("size_bytes", 0) <= 0
            or not _is_digest(row.get("sha256"))
            or not _is_identifier(row.get("coordinate_frame_id"))
            or not _is_identifier(row.get("rights_source_id"))
        ):
            errors.append(f"task_preinsertion_scene_{role}_invalid")
        sources[role] = row
    registration = value.get("registration")
    if (
        not isinstance(registration, Mapping)
        or set(registration)
        != {
            "status",
            "shared_coordinates_proved",
            "scale_axes_transform_proved",
            "transform_sha256",
            "evidence_binding_id",
        }
        or registration.get("status") != "passed"
        or registration.get("shared_coordinates_proved") is not True
        or registration.get("scale_axes_transform_proved") is not True
        or not _is_digest(registration.get("transform_sha256"))
        or sources.get("appearance", {}).get("coordinate_frame_id")
        != sources.get("collision", {}).get("coordinate_frame_id")
    ):
        errors.append("task_preinsertion_scene_registration_invalid")
    registration_evidence = _typed_payload(
        artifacts,
        registration.get("evidence_binding_id") if isinstance(registration, Mapping) else None,
        schema_version=REGISTRATION_EVIDENCE_SCHEMA_VERSION,
    )
    registration_evidence_fields = {
        "schema_version",
        "evidence_id",
        "scene_id",
        "status",
        "appearance_source_sha256",
        "collision_source_sha256",
        "coordinate_frame_id",
        "transform_binding_id",
        "transform_sha256",
        "receipt_digest",
    }
    transform_binding_id = registration_evidence.get("transform_binding_id")
    transform_artifact = artifacts.get(str(transform_binding_id), {})
    transform_payload = _typed_payload(
        artifacts,
        transform_binding_id,
        schema_version=REGISTRATION_TRANSFORM_SCHEMA_VERSION,
    )
    if (
        set(registration_evidence) != registration_evidence_fields
        or not _is_identifier(registration_evidence.get("evidence_id"))
        or registration_evidence.get("scene_id") != manifest["scene_id"]
        or registration_evidence.get("status") != "verified_registration"
        or registration_evidence.get("appearance_source_sha256")
        != sources.get("appearance", {}).get("sha256")
        or registration_evidence.get("collision_source_sha256")
        != sources.get("collision", {}).get("sha256")
        or registration_evidence.get("coordinate_frame_id")
        != sources.get("appearance", {}).get("coordinate_frame_id")
        or registration_evidence.get("transform_sha256") != registration.get("transform_sha256")
        or not _is_identifier(transform_binding_id)
        or transform_binding_id not in artifacts
        or transform_artifact.get("binding", {}).get("sha256")
        != registration_evidence.get("transform_sha256")
        or set(transform_payload)
        != {
            "schema_version",
            "evidence_id",
            "scene_id",
            "appearance_source_id",
            "collision_source_id",
            "coordinate_frame_id",
            "meters_per_unit",
            "up_axis",
            "appearance_to_collision_matrix_row_major",
            "receipt_digest",
        }
        or not _is_identifier(transform_payload.get("evidence_id"))
        or transform_payload.get("scene_id") != manifest["scene_id"]
        or transform_payload.get("appearance_source_id")
        != sources.get("appearance", {}).get("source_id")
        or transform_payload.get("collision_source_id")
        != sources.get("collision", {}).get("source_id")
        or transform_payload.get("coordinate_frame_id")
        != sources.get("appearance", {}).get("coordinate_frame_id")
        or isinstance(transform_payload.get("meters_per_unit"), bool)
        or not isinstance(transform_payload.get("meters_per_unit"), (int, float))
        or not math.isfinite(float(transform_payload.get("meters_per_unit")))
        or float(transform_payload.get("meters_per_unit")) != 1.0
        or transform_payload.get("up_axis") != "Z"
        or not _finite_vector(
            transform_payload.get("appearance_to_collision_matrix_row_major"),
            length=16,
        )
        or not _receipt_digest_valid(transform_payload, "receipt_digest")
        or not _receipt_digest_valid(registration_evidence, "receipt_digest")
    ):
        errors.append("task_preinsertion_scene_registration_evidence_invalid")
    topology = value.get("topology")
    if (
        not isinstance(topology, Mapping)
        or set(topology)
        != {
            "complete_known_topology_surveyed",
            "source_observation_limits_recorded",
            "unseen_or_occluded_regions",
            "evidence_binding_id",
        }
        or topology.get("complete_known_topology_surveyed") is not True
        or topology.get("source_observation_limits_recorded") is not True
        or not isinstance(topology.get("unseen_or_occluded_regions"), list)
        or any(
            not isinstance(item, str) or not item
            for item in topology.get("unseen_or_occluded_regions", [])
        )
    ):
        errors.append("task_preinsertion_scene_topology_invalid")
    topology_evidence = _typed_payload(
        artifacts,
        topology.get("evidence_binding_id") if isinstance(topology, Mapping) else None,
        schema_version=TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
    )
    topology_evidence_fields = {
        "schema_version",
        "evidence_id",
        "scene_id",
        "survey_binding_id",
        "survey_sha256",
        "complete_known_topology_surveyed",
        "source_observation_limits_recorded",
        "unseen_or_occluded_regions",
        "receipt_digest",
    }
    survey_binding_id = topology_evidence.get("survey_binding_id")
    survey_artifact = artifacts.get(str(survey_binding_id), {})
    survey_payload = _typed_payload(
        artifacts,
        survey_binding_id,
        schema_version=TOPOLOGY_SURVEY_SCHEMA_VERSION,
    )
    if (
        set(topology_evidence) != topology_evidence_fields
        or not _is_identifier(topology_evidence.get("evidence_id"))
        or topology_evidence.get("scene_id") != manifest["scene_id"]
        or topology_evidence.get("complete_known_topology_surveyed")
        != topology.get("complete_known_topology_surveyed")
        or topology_evidence.get("source_observation_limits_recorded")
        != topology.get("source_observation_limits_recorded")
        or topology_evidence.get("unseen_or_occluded_regions")
        != topology.get("unseen_or_occluded_regions")
        or not _is_identifier(survey_binding_id)
        or survey_binding_id not in artifacts
        or survey_artifact.get("binding", {}).get("sha256")
        != topology_evidence.get("survey_sha256")
        or set(survey_payload)
        != {
            "schema_version",
            "evidence_id",
            "scene_id",
            "appearance_source_sha256",
            "collision_source_sha256",
            "surveyed_region_ids",
            "unseen_or_occluded_regions",
            "completed_at",
            "receipt_digest",
        }
        or not _is_identifier(survey_payload.get("evidence_id"))
        or survey_payload.get("scene_id") != manifest["scene_id"]
        or survey_payload.get("appearance_source_sha256")
        != sources.get("appearance", {}).get("sha256")
        or survey_payload.get("collision_source_sha256")
        != sources.get("collision", {}).get("sha256")
        or not _nonempty_unique_strings(survey_payload.get("surveyed_region_ids"))
        or survey_payload.get("unseen_or_occluded_regions")
        != topology.get("unseen_or_occluded_regions")
        or not isinstance(survey_payload.get("completed_at"), str)
        or not survey_payload.get("completed_at")
        or not _receipt_digest_valid(survey_payload, "receipt_digest")
        or not _receipt_digest_valid(topology_evidence, "receipt_digest")
    ):
        errors.append("task_preinsertion_scene_topology_evidence_invalid")
    return sorted(set(errors)), {
        "scene_id": value.get("scene_id"),
        "appearance_source_id": sources.get("appearance", {}).get("source_id"),
        "collision_source_id": sources.get("collision", {}).get("source_id"),
        "coordinate_frame_id": sources.get("appearance", {}).get("coordinate_frame_id"),
        "unseen_or_occluded_regions": (
            topology.get("unseen_or_occluded_regions", []) if isinstance(topology, Mapping) else []
        ),
    }


def _task_gate(
    value: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    fields = {
        "schema_version",
        "scene_id",
        "task_id",
        "task_kind",
        "status",
        "prompt",
        "candidate_ids",
        "outcome_blind",
        "entities_frozen",
        "start_state_frozen",
        "destination_frozen",
        "controls_frozen",
        "seeds_frozen",
        "matrix_subset_frozen",
        "task_spec_digest",
        "prompt_task_spec_digest",
        "freeze_authority",
        "receipt_digest",
    }
    errors: list[str] = []
    freeze_authority_valid = _task_freeze_authority_signature_valid(value)
    if (
        set(value) != fields
        or value.get("scene_id") != manifest["scene_id"]
        or value.get("task_id") != manifest["task_id"]
        or value.get("task_kind") != manifest["task_kind"]
        or value.get("candidate_ids") != manifest["candidate_ids"]
    ):
        errors.append("task_preinsertion_task_contract_invalid")
    if (
        value.get("status") != "frozen"
        or not isinstance(value.get("prompt"), str)
        or not value.get("prompt", "").strip()
        or not _is_digest(value.get("task_spec_digest"))
        or not _is_digest(value.get("prompt_task_spec_digest"))
        or not freeze_authority_valid
        or any(
            value.get(field) is not True
            for field in (
                "outcome_blind",
                "entities_frozen",
                "start_state_frozen",
                "destination_frozen",
                "controls_frozen",
                "seeds_frozen",
                "matrix_subset_frozen",
            )
        )
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_task_not_frozen")
    if not freeze_authority_valid:
        errors.append("task_preinsertion_task_freeze_authority_invalid")
    return sorted(set(errors)), {
        "task_id": value.get("task_id"),
        "task_kind": value.get("task_kind"),
        "candidate_ids": value.get("candidate_ids"),
        "prompt": value.get("prompt"),
        "task_spec_digest": value.get("task_spec_digest"),
        "prompt_task_spec_digest": value.get("prompt_task_spec_digest"),
    }


def _rights_evidence_valid(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    binding_id: Any,
    sha256: Any,
    evidence_kind: str,
    subject_id: str,
    document_id: str,
    source_revision: str | None,
    private_derived_processing_permitted: bool,
    raw_upload_permitted: bool,
    provider_retention_permitted: bool,
    provider_training_permitted: bool,
    output_rights_bound: bool,
) -> bool:
    payload = _typed_payload(
        artifacts,
        binding_id,
        schema_version=RIGHTS_EVIDENCE_SCHEMA_VERSION,
    )
    document_binding_id = payload.get("document_binding_id")
    document = artifacts.get(str(document_binding_id), {})
    verifier_binding_id = payload.get("verifier_source_binding_id")
    verifier = artifacts.get(str(verifier_binding_id), {})
    expected_verifier_source = Path(__file__)
    return bool(
        _binding_matches(artifacts, binding_id, sha256=sha256)
        and set(payload)
        == {
            "schema_version",
            "evidence_id",
            "evidence_kind",
            "subject_id",
            "document_id",
            "document_binding_id",
            "document_sha256",
            "document_size_bytes",
            "source_revision",
            "private_derived_processing_permitted",
            "raw_upload_permitted",
            "provider_retention_permitted",
            "provider_training_permitted",
            "output_rights_bound",
            "interpretation_version",
            "verifier_source_binding_id",
            "verifier_source_sha256",
            "authority",
            "receipt_digest",
        }
        and _is_identifier(payload.get("evidence_id"))
        and payload.get("evidence_kind") == evidence_kind
        and payload.get("subject_id") == subject_id
        and payload.get("document_id") == document_id
        and _is_identifier(document_binding_id)
        and document_binding_id in artifacts
        and document.get("binding", {}).get("content_type") == "opaque"
        and _is_digest(payload.get("document_sha256"))
        and document.get("binding", {}).get("sha256") == payload.get("document_sha256")
        and type(payload.get("document_size_bytes")) is int
        and payload.get("document_size_bytes", 0) > 0
        and len(document.get("content", b"")) == payload.get("document_size_bytes")
        and payload.get("source_revision") == source_revision
        and payload.get("private_derived_processing_permitted")
        is private_derived_processing_permitted
        and payload.get("raw_upload_permitted") is raw_upload_permitted
        and payload.get("provider_retention_permitted") is provider_retention_permitted
        and payload.get("provider_training_permitted") is provider_training_permitted
        and payload.get("output_rights_bound") is output_rights_bound
        and payload.get("interpretation_version") == RIGHTS_INTERPRETATION_VERSION
        and _is_identifier(verifier_binding_id)
        and verifier_binding_id in artifacts
        and verifier.get("binding", {}).get("content_type") == "opaque"
        and _is_digest(payload.get("verifier_source_sha256"))
        and verifier.get("binding", {}).get("sha256") == payload.get("verifier_source_sha256")
        and verifier.get("content") == expected_verifier_source.read_bytes()
        and _rights_authority_signature_valid(payload)
        and _receipt_digest_valid(payload, "receipt_digest")
    )


def _rights_gate(
    value: Mapping[str, Any],
    *,
    required_sources: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any], dict[str, Mapping[str, Any]]]:
    errors: list[str] = []
    if set(value) != {
        "schema_version",
        "status",
        "sources",
        "provider_processing",
        "receipt_digest",
    }:
        errors.append("task_preinsertion_rights_contract_invalid")
    sources: dict[str, Mapping[str, Any]] = {}
    source_fields = {
        "source_id",
        "revision",
        "license_id",
        "license_binding_id",
        "license_sha256",
        "source_path",
        "size_bytes",
        "sha256",
        "attribution",
        "disclosure_class",
        "raw_upload_permitted",
        "private_derived_processing_permitted",
        "provider_retention_permitted",
        "provider_training_permitted",
        "output_rights_id",
        "output_rights_binding_id",
        "output_rights_sha256",
    }
    source_rows = _rows(value.get("sources"))
    if not source_rows or len(source_rows) > 64:
        errors.append("task_preinsertion_rights_source_inventory_invalid")
    for row in source_rows:
        source_id = row.get("source_id")
        if set(row) != source_fields or not _is_identifier(source_id) or source_id in sources:
            errors.append("task_preinsertion_rights_source_invalid")
            continue
        if (
            not isinstance(row.get("revision"), str)
            or not row.get("revision")
            or not _is_identifier(row.get("license_id"))
            or not _rights_evidence_valid(
                artifacts,
                binding_id=row.get("license_binding_id"),
                sha256=row.get("license_sha256"),
                evidence_kind="source_license",
                subject_id=str(source_id),
                document_id=str(row.get("license_id")),
                source_revision=str(row.get("revision")),
                private_derived_processing_permitted=bool(
                    row.get("private_derived_processing_permitted")
                ),
                raw_upload_permitted=bool(row.get("raw_upload_permitted")),
                provider_retention_permitted=bool(row.get("provider_retention_permitted")),
                provider_training_permitted=bool(row.get("provider_training_permitted")),
                output_rights_bound=False,
            )
            or not isinstance(row.get("source_path"), str)
            or not row.get("source_path")
            or type(row.get("size_bytes")) is not int
            or row.get("size_bytes", 0) <= 0
            or not _is_digest(row.get("sha256"))
            or not isinstance(row.get("attribution"), str)
            or not row.get("attribution")
            or not isinstance(row.get("disclosure_class"), str)
            or row.get("disclosure_class")
            not in {
                "public_redistributable",
                "restricted_nonredistributable",
                "runtime_bundled",
                "generated_derivative",
            }
            or not isinstance(row.get("raw_upload_permitted"), bool)
            or not isinstance(row.get("private_derived_processing_permitted"), bool)
            or row.get("private_derived_processing_permitted") is not True
            or not isinstance(row.get("provider_retention_permitted"), bool)
            or not isinstance(row.get("provider_training_permitted"), bool)
            or not _is_identifier(row.get("output_rights_id"))
            or not _rights_evidence_valid(
                artifacts,
                binding_id=row.get("output_rights_binding_id"),
                sha256=row.get("output_rights_sha256"),
                evidence_kind="source_output_rights",
                subject_id=str(source_id),
                document_id=str(row.get("output_rights_id")),
                source_revision=str(row.get("revision")),
                private_derived_processing_permitted=bool(
                    row.get("private_derived_processing_permitted")
                ),
                raw_upload_permitted=bool(row.get("raw_upload_permitted")),
                provider_retention_permitted=bool(row.get("provider_retention_permitted")),
                provider_training_permitted=bool(row.get("provider_training_permitted")),
                output_rights_bound=True,
            )
            or (
                row.get("disclosure_class") == "restricted_nonredistributable"
                and row.get("raw_upload_permitted") is not False
            )
        ):
            errors.append(f"task_preinsertion_rights_source_invalid:{source_id}")
        sources[str(source_id)] = row
    if not set(required_sources).issubset(sources):
        errors.append("task_preinsertion_rights_source_join_invalid")
    for source_id, scene_source in required_sources.items():
        rights_source = sources.get(source_id, {})
        if any(
            rights_source.get(field) != scene_source.get(field)
            for field in ("revision", "source_path", "size_bytes", "sha256")
        ):
            errors.append(f"task_preinsertion_rights_source_join_invalid:{source_id}")
    processing = value.get("provider_processing")
    if (
        not isinstance(processing, Mapping)
        or set(processing)
        != {
            "provider_id",
            "private_derived_upload_authority_id",
            "private_derived_upload_authority_binding_id",
            "private_derived_upload_authority_sha256",
            "provider_terms_id",
            "provider_terms_binding_id",
            "provider_terms_sha256",
        }
        or not _is_identifier(processing.get("provider_id"))
        or not _is_identifier(processing.get("private_derived_upload_authority_id"))
        or not _is_identifier(processing.get("provider_terms_id"))
        or not _rights_evidence_valid(
            artifacts,
            binding_id=processing.get("private_derived_upload_authority_binding_id"),
            sha256=processing.get("private_derived_upload_authority_sha256"),
            evidence_kind="private_derived_processing_authority",
            subject_id="run-private-derived-processing",
            document_id=str(processing.get("private_derived_upload_authority_id")),
            source_revision=None,
            private_derived_processing_permitted=True,
            raw_upload_permitted=False,
            provider_retention_permitted=False,
            provider_training_permitted=False,
            output_rights_bound=True,
        )
        or not _rights_evidence_valid(
            artifacts,
            binding_id=processing.get("provider_terms_binding_id"),
            sha256=processing.get("provider_terms_sha256"),
            evidence_kind="provider_terms",
            subject_id=str(processing.get("provider_id")),
            document_id=str(processing.get("provider_terms_id")),
            source_revision=None,
            private_derived_processing_permitted=True,
            raw_upload_permitted=False,
            provider_retention_permitted=False,
            provider_training_permitted=False,
            output_rights_bound=True,
        )
        or value.get("status") != "admitted"
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_rights_not_admitted")
    return (
        sorted(set(errors)),
        {
            "source_ids": sorted(sources),
            "restricted_source_ids": sorted(
                source_id
                for source_id, row in sources.items()
                if row.get("disclosure_class") == "restricted_nonredistributable"
            ),
        },
        sources,
    )


def _asset_candidate_input(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "entity_id",
        "asset_id",
        "asset_class",
        "source_observation",
        "rights",
        "authoring",
        "files",
        "transform",
        "simulator_import",
        "retained_diagnostic_requirements",
    }
    asset_class = value.get("asset_class")
    if asset_class == "rigid_receptacle":
        fields.add("receptacle_configuration")
    elif asset_class == "deformable_volume":
        fields.add("deformable_configuration")
    return {field: value.get(field) for field in fields}


def _rigid_receptacle_structure_errors(
    *,
    candidate: Mapping[str, Any],
    files_by_role: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    error = "task_preinsertion_entity_engineered_static_structure_invalid"
    try:
        collision_obj_points: list[tuple[float, float, float]] = []
        for role in ("visual_geometry", "collision_geometry"):
            artifact = files_by_role[role]
            path = str(artifact["binding"]["relative_path"])
            text = artifact["content"].decode("utf-8")
            if not path.lower().endswith(".obj"):
                return [error]
            vertices = [line for line in text.splitlines() if line.startswith("v ")]
            faces = [line for line in text.splitlines() if line.startswith("f ")]
            if len(vertices) < 8 or len(faces) < 5:
                return [error]
            parsed_vertices = [
                tuple(float(item) for item in line.split()[1:4]) for line in vertices
            ]
            if any(
                len(point) != 3 or any(not math.isfinite(axis) for axis in point)
                for point in parsed_vertices
            ):
                return [error]
            for face in faces:
                face_indices = [item.split("/", 1)[0] for item in face.split()[1:]]
                if len(face_indices) < 3 or any(
                    not item or int(item) == 0 or abs(int(item)) > len(parsed_vertices)
                    for item in face_indices
                ):
                    return [error]
            if role == "collision_geometry":
                collision_obj_points = parsed_vertices

        runtime = files_by_role["runtime_usd"]
        runtime_path = str(runtime["binding"]["relative_path"])
        if not runtime_path.lower().endswith((".usd", ".usda", ".usdc")):
            return [error]
        runtime_text = runtime["content"].decode("utf-8")
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        layer = Sdf.Layer.CreateAnonymous("task-preinsertion-runtime.usda")
        if not layer.ImportFromString(runtime_text):
            return [error]
        if (
            layer.subLayerPaths
            or layer.GetExternalAssetDependencies()
            or layer.GetExternalReferences()
        ):
            return [error]
        stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadAll)
        if (
            stage is None
            or abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-12
            or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z
        ):
            return [error]
        collision_prims = [
            prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(collision_prims) != 1:
            return [error]
        collision_prim = collision_prims[0]
        collision_api = UsdPhysics.CollisionAPI(collision_prim)
        if (
            not collision_prim.IsA(UsdGeom.Mesh)
            or collision_api.GetCollisionEnabledAttr().Get() is not True
        ):
            return [error]
        collision_meshes: list[tuple[list[tuple[float, float, float]], list[int], list[int]]] = []
        mesh = UsdGeom.Mesh(collision_prim)
        local_points = mesh.GetPointsAttr().Get() or []
        counts = [int(count) for count in (mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(index) for index in (mesh.GetFaceVertexIndicesAttr().Get() or [])]
        if (
            len(local_points) < 3
            or not counts
            or any(count < 3 for count in counts)
            or sum(counts) != len(indices)
            or any(index < 0 or index >= len(local_points) for index in indices)
        ):
            return [error]
        local_to_world = UsdGeom.Xformable(collision_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        transformed_points = []
        for point in local_points:
            transformed = local_to_world.Transform(
                Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
            )
            transformed_points.append(tuple(float(axis) for axis in transformed))
        collision_meshes.append((transformed_points, counts, indices))
        all_points = [point for points, _counts, _indices in collision_meshes for point in points]
        x_min = min(point[0] for point in all_points)
        x_max = max(point[0] for point in all_points)
        y_min = min(point[1] for point in all_points)
        y_max = max(point[1] for point in all_points)
        z_min = min(point[2] for point in all_points)
        z_max = max(point[2] for point in all_points)
        footprint_area = (x_max - x_min) * (y_max - y_min)
        if footprint_area <= 0.0 or z_max <= z_min:
            return [error]

        def bounds(points: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
            return (
                [min(point[axis] for point in points) for axis in range(3)],
                [max(point[axis] for point in points) for axis in range(3)],
            )

        obj_minimum, obj_maximum = bounds(collision_obj_points)
        usd_minimum, usd_maximum = bounds(all_points)
        if any(
            abs(obj_minimum[axis] - usd_minimum[axis]) > 1.0e-7
            or abs(obj_maximum[axis] - usd_maximum[axis]) > 1.0e-7
            for axis in range(3)
        ):
            return [error]
        source_observation = candidate.get("source_observation")
        transform = candidate.get("transform")
        dimensions = (
            source_observation.get("metric_dimensions_m")
            if isinstance(source_observation, Mapping)
            else None
        )
        scale = transform.get("scale_xyz") if isinstance(transform, Mapping) else None
        if (
            not _finite_vector(dimensions, length=3)
            or not _finite_vector(scale, length=3)
            or any(float(item) <= 0.0 for item in scale)
        ):
            return [error]
        runtime_dimensions = [
            (usd_maximum[axis] - usd_minimum[axis]) * float(scale[axis]) for axis in range(3)
        ]
        if any(
            abs(runtime_dimensions[axis] - float(dimensions[axis])) > 1.0e-6 for axis in range(3)
        ):
            return [error]

        configuration = candidate.get("receptacle_configuration")
        geometry = configuration.get("geometry") if isinstance(configuration, Mapping) else None
        wall_clearances = (
            geometry.get("wall_clearances_m") if isinstance(geometry, Mapping) else None
        )
        interior_dimensions = (
            geometry.get("interior_dimensions_m") if isinstance(geometry, Mapping) else None
        )
        floor_thickness = (
            geometry.get("floor_thickness_m") if isinstance(geometry, Mapping) else None
        )
        if (
            not isinstance(geometry, Mapping)
            or geometry.get("open_interior") is not True
            or geometry.get("top_cap_present") is not False
            or not isinstance(wall_clearances, Mapping)
            or set(wall_clearances) != {"x_min", "x_max", "y_min", "y_max"}
            or not _finite_vector(interior_dimensions, length=3)
            or isinstance(floor_thickness, bool)
            or not isinstance(floor_thickness, (int, float))
            or not math.isfinite(float(floor_thickness))
            or float(floor_thickness) <= 0.0
            or any(
                isinstance(wall_clearances.get(side), bool)
                or not isinstance(wall_clearances.get(side), (int, float))
                or not math.isfinite(float(wall_clearances[side]))
                or float(wall_clearances[side]) <= 0.0
                for side in ("x_min", "x_max", "y_min", "y_max")
            )
        ):
            return [error]
        opening_rectangle = (
            x_min + float(wall_clearances["x_min"]) / float(scale[0]),
            x_max - float(wall_clearances["x_max"]) / float(scale[0]),
            y_min + float(wall_clearances["y_min"]) / float(scale[1]),
            y_max - float(wall_clearances["y_max"]) / float(scale[1]),
        )
        opening_x_min, opening_x_max, opening_y_min, opening_y_max = opening_rectangle
        if (
            opening_x_max <= opening_x_min
            or opening_y_max <= opening_y_min
            or abs(
                (opening_x_max - opening_x_min) * float(scale[0]) - float(interior_dimensions[0])
            )
            > 1.0e-6
            or abs(
                (opening_y_max - opening_y_min) * float(scale[1]) - float(interior_dimensions[1])
            )
            > 1.0e-6
        ):
            return [error]

        def clipped_triangle_area(
            triangle: Sequence[tuple[float, float]],
        ) -> float:
            polygon = list(triangle)
            boundaries = (
                (0, opening_x_min, True),
                (0, opening_x_max, False),
                (1, opening_y_min, True),
                (1, opening_y_max, False),
            )
            for axis, boundary, keep_greater in boundaries:
                if not polygon:
                    return 0.0
                clipped: list[tuple[float, float]] = []
                previous = polygon[-1]
                previous_inside = (
                    previous[axis] >= boundary if keep_greater else previous[axis] <= boundary
                )
                for current in polygon:
                    current_inside = (
                        current[axis] >= boundary if keep_greater else current[axis] <= boundary
                    )
                    if current_inside != previous_inside:
                        delta = current[axis] - previous[axis]
                        if abs(delta) > 1.0e-15:
                            fraction = (boundary - previous[axis]) / delta
                            intersection = (
                                previous[0] + fraction * (current[0] - previous[0]),
                                previous[1] + fraction * (current[1] - previous[1]),
                            )
                            clipped.append(intersection)
                    if current_inside:
                        clipped.append(current)
                    previous = current
                    previous_inside = current_inside
                polygon = clipped
            if len(polygon) < 3:
                return 0.0
            return (
                abs(
                    sum(
                        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                        for index in range(len(polygon))
                    )
                )
                / 2.0
            )

        tolerance = max(z_max - z_min, 1.0) * 1.0e-7
        # Mesh points are authored as ``point3f``.  Treat sub-nanometre-square
        # overlap caused solely by float32 boundary quantization as zero while
        # retaining a many-orders-of-magnitude separation from even a thin
        # physical cap over the admitted opening.
        open_area_tolerance = max(footprint_area, 1.0) * 1.0e-9
        admitted_floor_top = z_min + float(floor_thickness) / float(scale[2])
        for points, counts, indices in collision_meshes:
            cursor = 0
            for count in counts:
                face_indices = indices[cursor : cursor + count]
                cursor += count
                if len(face_indices) < 3 or any(
                    index < 0 or index >= len(points) for index in face_indices
                ):
                    return [error]
                face = [points[index] for index in face_indices]
                if max(point[2] for point in face) > admitted_floor_top + tolerance:
                    projected = [(point[0], point[1]) for point in face]
                    for index in range(1, len(projected) - 1):
                        if (
                            clipped_triangle_area(
                                (projected[0], projected[index], projected[index + 1])
                            )
                            > open_area_tolerance
                        ):
                            return [error]
            if cursor != len(indices):
                return [error]
    except (
        ImportError,
        KeyError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ):
        return [error]
    return []


def _registered_receptacle_evidence_errors(
    *,
    entity_id: str,
    candidate: Mapping[str, Any],
    candidate_binding_id: Any,
    source: Mapping[str, Any],
    authoring_receipt: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    error = f"task_preinsertion_entity_engineered_registered_receipt_invalid:{entity_id}"
    try:
        registered_binding_id = authoring_receipt.get("registered_asset_receipt_binding_id")
        topology_binding_id = authoring_receipt.get("topology_receipt_binding_id")
        visual_binding_id = authoring_receipt.get("visual_basis_binding_id")
        attestation_binding_id = authoring_receipt.get("semantic_attestation_binding_id")
        selection_binding_id = authoring_receipt.get("semantic_selection_binding_id")
        builder_binding_id = authoring_receipt.get("builder_source_binding_id")
        joins = (
            (
                registered_binding_id,
                authoring_receipt.get("registered_asset_receipt_sha256"),
            ),
            (topology_binding_id, authoring_receipt.get("topology_receipt_sha256")),
            (visual_binding_id, authoring_receipt.get("visual_basis_sha256")),
            (
                attestation_binding_id,
                authoring_receipt.get("semantic_attestation_sha256"),
            ),
            (
                selection_binding_id,
                authoring_receipt.get("semantic_selection_sha256"),
            ),
            (builder_binding_id, authoring_receipt.get("builder_source_sha256")),
        )
        if any(
            not _binding_matches(artifacts, binding_id, sha256=sha256)
            for binding_id, sha256 in joins
        ):
            return [error]
        registered = _typed_payload(
            artifacts,
            registered_binding_id,
            schema_version=REGISTERED_RECEPTACLE_RECEIPT_SCHEMA_VERSION,
        )
        topology = _typed_payload(
            artifacts,
            topology_binding_id,
            schema_version=REGISTERED_COLLISION_TOPOLOGY_SCHEMA_VERSION,
        )
        visual_basis = _typed_payload(
            artifacts,
            visual_binding_id,
            schema_version=ENGINEERED_RECEPTACLE_VISUAL_BASIS_SCHEMA_VERSION,
        )
        builder = artifacts.get(str(builder_binding_id), {})
        expected_builder_source = Path(__file__).with_name("registered_static_receptacle_asset.py")
        if (
            builder.get("binding", {}).get("content_type") != "opaque"
            or builder.get("content") != expected_builder_source.read_bytes()
            or not _receipt_digest_valid(registered, "receipt_digest")
            or not _receipt_digest_valid(topology, "receipt_digest")
            or not _receipt_digest_valid(visual_basis, "basis_digest")
        ):
            return [error]
        with tempfile.TemporaryDirectory(
            prefix="task-preinsertion-semantic-snapshot-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as semantic_raw:
            semantic_root = Path(semantic_raw)
            attestation_snapshot = _materialize_verifier_snapshot(
                root=semantic_root,
                filename="semantic_attestation.json",
                artifact=artifacts[str(attestation_binding_id)],
            )
            selection_snapshot = _materialize_verifier_snapshot(
                root=semantic_root,
                filename="semantic_selection.json",
                artifact=artifacts[str(selection_binding_id)],
            )
            semantic = verify_semantic_review_attestation(
                attestation_path=attestation_snapshot,
                selection_contract_path=selection_snapshot,
            )
        source_evidence = _typed_payload(
            artifacts,
            source.get("evidence_binding_id"),
            schema_version=SOURCE_EVIDENCE_SCHEMA_VERSION,
        )
        source_instance_id = source_evidence.get("source_instance_id")
        source_citations = _rows(source_evidence.get("cited_visual_evidence"))
        candidate_source = candidate.get("source_observation")
        topology_targets = _rows(topology.get("targets"))
        matching_topology_targets = [
            row
            for row in topology_targets
            if row.get("interiorgs_instance_id") == source_instance_id
        ]
        if len(matching_topology_targets) != 1:
            return [error]
        topology_target = matching_topology_targets[0]
        candidate_artifact = artifacts.get(str(candidate_binding_id), {})
        candidate_file = registered.get("candidate_file")
        visual_file = registered.get("visual_design_basis_file")
        registered_files = _rows(registered.get("files"))
        candidate_files = _rows(candidate.get("files"))
        candidate_file_rows = sorted(
            (
                str(row.get("role")),
                str(row.get("path")),
                row.get("sha256"),
                row.get("size_bytes"),
            )
            for row in candidate_files
        )
        registered_file_rows = sorted(
            (
                str(row.get("role")),
                str(row.get("path")),
                row.get("sha256"),
                row.get("size_bytes"),
            )
            for row in registered_files
        )
        semantic_authority = visual_basis.get("semantic_authority")
        semantic = _normalize_verifier_path_metadata(semantic, semantic_authority)
        semantic_evidence = semantic.get("evidence")
        claims = registered.get("claim_boundary")
        visual_citations = _rows(visual_basis.get("cited_frames"))
        semantic_frame_rows: list[dict[str, Any]] = []
        for row in visual_citations:
            file_row = row.get("file")
            if not isinstance(file_row, Mapping):
                return [error]
            semantic_frame_rows.append(
                {
                    "target_id": row.get("target_id"),
                    "camera_id": row.get("camera_id"),
                    "sha256": file_row.get("sha256"),
                    "size_bytes": file_row.get("size_bytes"),
                    "decoded_rgb_sha256": row.get("decoded_rgb_sha256"),
                }
            )
        cited_digest = semantic_frame_evidence_digest(semantic_frame_rows)
        source_citation_rows = {
            (
                row.get("camera_id"),
                row.get("sha256"),
                row.get("size_bytes"),
            )
            for row in source_citations
        }
        semantic_citation_rows = {
            (row["camera_id"], row["sha256"], row["size_bytes"]) for row in semantic_frame_rows
        }
        topology_sources = topology.get("source_files")
        labels_source = (
            topology_sources.get("interiorgs_labels")
            if isinstance(topology_sources, Mapping)
            else None
        )
        if (
            set(registered)
            != {
                "schema_version",
                "target_instance_id",
                "topology_receipt_digest",
                "visual_design_basis_digest",
                "semantic_review_attestation_digest",
                "semantic_authority_selection_digest",
                "semantic_authority",
                "visual_review_digest",
                "visual_review_collision_topology_receipt_digest",
                "visual_design_basis_file",
                "component_geometry_receipt_digest",
                "component_local_geometry_digest",
                "candidate_digest",
                "candidate_file",
                "derived_geometry",
                "geometry_conversion",
                "files",
                "claim_boundary",
                "receipt_digest",
            }
            or set(topology)
            != {
                "schema_version",
                "source_files",
                "coordinate_frame",
                "thresholds",
                "inspected_overlapping_mesh_count",
                "inspected_connected_component_count",
                "targets",
                "all_component_collision_identities_passed",
                "claim_boundary",
                "receipt_digest",
            }
            or set(visual_basis)
            != {
                "schema_version",
                "status",
                "scene_id",
                "source_receptacle_instance_id",
                "current_topology_receipt_digest",
                "visual_review_receipt",
                "render_manifest",
                "cited_frame_count",
                "cited_frames_digest",
                "cited_frames",
                "semantic_authority",
                "source_receptacle_observation",
                "engineered_twin_design_basis",
                "claim_boundary",
                "basis_digest",
            }
            or registered.get("target_instance_id") != source_instance_id
            or not isinstance(candidate_source, Mapping)
            or candidate_source.get("observation_id") != f"sage-component:{source_instance_id}"
            or not isinstance(candidate_source.get("bounds_world"), Mapping)
            or not isinstance(source_evidence.get("bounds_world"), Mapping)
            or not _finite_vectors_close(
                candidate_source.get("bounds_world", {}).get("minimum_m"),
                source_evidence.get("bounds_world", {}).get("minimum_m"),
                length=3,
            )
            or not _finite_vectors_close(
                candidate_source.get("bounds_world", {}).get("maximum_m"),
                source_evidence.get("bounds_world", {}).get("maximum_m"),
                length=3,
            )
            or not _finite_vectors_close(
                candidate_source.get("metric_dimensions_m"),
                source_evidence.get("metric_dimensions_m"),
                length=3,
            )
            or registered.get("topology_receipt_digest") != topology.get("receipt_digest")
            or registered.get("visual_design_basis_digest") != visual_basis.get("basis_digest")
            or registered.get("semantic_review_attestation_digest")
            != semantic.get("attestation_digest")
            or registered.get("semantic_authority_selection_digest")
            != semantic.get("selection_digest")
            or registered.get("visual_review_collision_topology_receipt_digest")
            != topology.get("receipt_digest")
            or registered.get("visual_review_digest")
            != semantic_evidence.get("visual_review_digest")
            or registered.get("candidate_digest") != candidate.get("candidate_digest")
            or not isinstance(candidate_file, Mapping)
            or candidate_file.get("sha256") != candidate_artifact.get("binding", {}).get("sha256")
            or candidate_file.get("path")
            != Path(str(candidate_artifact.get("binding", {}).get("relative_path", ""))).name
            or candidate_file.get("size_bytes") != len(candidate_artifact.get("content", b""))
            or not isinstance(visual_file, Mapping)
            or visual_file.get("sha256")
            != artifacts[str(visual_binding_id)].get("binding", {}).get("sha256")
            or visual_file.get("path")
            != Path(
                str(artifacts[str(visual_binding_id)].get("binding", {}).get("relative_path", ""))
            ).name
            or visual_file.get("size_bytes")
            != len(artifacts[str(visual_binding_id)].get("content", b""))
            or candidate_file_rows != registered_file_rows
            or not isinstance(labels_source, Mapping)
            or topology_target.get("component_collision_identity_passed") is not True
            or topology.get("all_component_collision_identities_passed") is not True
            or visual_basis.get("source_receptacle_instance_id") != source_instance_id
            or visual_basis.get("status") != "verified_visual_design_basis"
            or visual_basis.get("scene_id") != semantic.get("scene_id")
            or visual_basis.get("render_manifest", {}).get("source_digest")
            != source.get("source_sha256")
            or visual_basis.get("current_topology_receipt_digest") != topology.get("receipt_digest")
            or visual_basis.get("cited_frame_count") != len(visual_citations)
            or semantic_authority != semantic
            or not isinstance(semantic_authority, Mapping)
            or semantic_authority.get("semantic_authority_verified") is not True
            or semantic_authority.get("attestation_digest") != semantic.get("attestation_digest")
            or semantic_authority.get("selection_digest") != semantic.get("selection_digest")
            or semantic.get("source_target", {}).get("source_instance_id") != source_instance_id
            or semantic_evidence.get("collision_topology_receipt_digest")
            != topology.get("receipt_digest")
            or semantic_evidence.get("visual_review_digest")
            != registered.get("visual_review_digest")
            or visual_basis.get("cited_frames_digest") != cited_digest
            or semantic_evidence.get("cited_frames_digest") != cited_digest
            or not semantic_citation_rows
            or not semantic_citation_rows.issubset(source_citation_rows)
            or not isinstance(claims, Mapping)
            or claims.get("source_collision_component_used_as_runtime_geometry") is not False
            or claims.get("source_bytes_copied_to_output") is not False
            or claims.get("engineered_twin_not_source_scene_truth") is not True
            or claims.get("visual_semantics_authority_signed") is not True
            or claims.get("signed_visual_semantics_are_native_qualification") is not False
            or claims.get("native_simulator_qualified") is not False
            or claims.get("physical_equivalence_proven") is not False
        ):
            return [error]
    except (
        KeyError,
        OSError,
        SemanticReviewAttestationError,
        TaskPreinsertionReadinessError,
        TypeError,
        ValueError,
    ):
        return [error]
    return []


def _registered_receptacle_replay_errors(
    *,
    entity_id: str,
    candidate: Mapping[str, Any],
    candidate_binding_id: Any,
    authoring_receipt: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Replay the released registered-receptacle builder from retained inputs."""

    error = f"task_preinsertion_entity_engineered_registered_replay_invalid:{entity_id}"
    try:
        request_binding_id = authoring_receipt.get("builder_replay_request_binding_id")
        request = _typed_payload(
            artifacts,
            request_binding_id,
            schema_version=REGISTERED_RECEPTACLE_REPLAY_REQUEST_SCHEMA_VERSION,
        )
        if (
            not _binding_matches(
                artifacts,
                request_binding_id,
                sha256=authoring_receipt.get("builder_replay_request_sha256"),
            )
            or set(request)
            != {
                "schema_version",
                "replay_id",
                "input_bindings",
                "frame_bindings",
                "module_sources",
                "builder_arguments",
                "receipt_digest",
            }
            or not _is_identifier(request.get("replay_id"))
            or not _receipt_digest_valid(request, "receipt_digest")
        ):
            return [error]
        input_rows = request.get("input_bindings")
        required_inputs = {
            "labels",
            "collision",
            "topology",
            "visual_review",
            "render_manifest",
            "semantic_attestation",
            "semantic_selection",
        }
        if not isinstance(input_rows, Mapping) or set(input_rows) != required_inputs:
            return [error]
        replay_artifacts: dict[str, Mapping[str, Any]] = {}
        for purpose, row in input_rows.items():
            if (
                not isinstance(row, Mapping)
                or set(row) != {"binding_id", "sha256"}
                or not _binding_matches(artifacts, row.get("binding_id"), sha256=row.get("sha256"))
            ):
                return [error]
            replay_artifacts[str(purpose)] = artifacts[str(row["binding_id"])]
        if (
            replay_artifacts["labels"].get("binding", {}).get("content_type") != "opaque"
            or replay_artifacts["collision"].get("binding", {}).get("content_type") != "opaque"
            or replay_artifacts["topology"].get("payload", {}).get("schema_version")
            != REGISTERED_COLLISION_TOPOLOGY_SCHEMA_VERSION
        ):
            return [error]
        render_manifest = replay_artifacts["render_manifest"].get("payload")
        if not isinstance(render_manifest, Mapping):
            return [error]
        frame_rows = _rows(request.get("frame_bindings"))
        cameras = _rows(render_manifest.get("cameras"))
        if not frame_rows or len(frame_rows) != len(cameras):
            return [error]
        source_frame_root: Path | None = None
        frame_snapshots: list[tuple[str, Mapping[str, Any]]] = []
        seen_frame_filenames: set[str] = set()
        seen_cameras: set[str] = set()
        for row in frame_rows:
            if set(row) != {"camera_id", "binding_id", "sha256"}:
                return [error]
            camera_id = row.get("camera_id")
            binding_id = row.get("binding_id")
            artifact = artifacts.get(str(binding_id), {})
            matching = [camera for camera in cameras if camera.get("id") == camera_id]
            if (
                not _is_identifier(camera_id)
                or camera_id in seen_cameras
                or len(matching) != 1
                or not _binding_matches(artifacts, binding_id, sha256=row.get("sha256"))
                or artifact.get("binding", {}).get("content_type") != "opaque"
                or not isinstance(artifact.get("path"), Path)
            ):
                return [error]
            camera = matching[0]
            if (
                camera.get("digest") != row.get("sha256")
                or camera.get("bytes") != len(artifact.get("content", b""))
                or Path(str(camera.get("path"))).name != artifact["path"].name
            ):
                return [error]
            frame_filename = artifact["path"].name
            if frame_filename in seen_frame_filenames:
                return [error]
            if source_frame_root is None:
                source_frame_root = artifact["path"].parent
            elif artifact["path"].parent != source_frame_root:
                return [error]
            frame_snapshots.append((frame_filename, artifact))
            seen_frame_filenames.add(frame_filename)
            seen_cameras.add(str(camera_id))
        module_rows = _rows(request.get("module_sources"))
        expected_modules = {
            "registered_static_receptacle_asset": Path(__file__).with_name(
                "registered_static_receptacle_asset.py"
            ),
            "sage_collision_component_topology": Path(__file__).with_name(
                "sage_collision_component_topology.py"
            ),
            "engineered_receptacle_visual_basis": Path(__file__).with_name(
                "engineered_receptacle_visual_basis.py"
            ),
            "semantic_review_attestation": Path(__file__).with_name(
                "semantic_review_attestation.py"
            ),
        }
        if len(module_rows) != len(expected_modules):
            return [error]
        seen_modules: set[str] = set()
        for row in module_rows:
            module = row.get("module")
            binding_id = row.get("binding_id")
            artifact = artifacts.get(str(binding_id), {})
            if (
                set(row) != {"module", "binding_id", "sha256"}
                or module not in expected_modules
                or module in seen_modules
                or not _binding_matches(artifacts, binding_id, sha256=row.get("sha256"))
                or artifact.get("binding", {}).get("content_type") != "opaque"
                or artifact.get("content") != expected_modules[str(module)].read_bytes()
            ):
                return [error]
            seen_modules.add(str(module))
        arguments = request.get("builder_arguments")
        if not isinstance(arguments, Mapping) or set(arguments) != {
            "target_instance_id",
            "entity_id",
            "asset_id",
            "reference_world_pose",
            "rights",
            "authoring_identity",
            "physics_configuration",
            "simulator_name",
            "simulator_version",
        }:
            return [error]
        with tempfile.TemporaryDirectory(
            prefix="task-preinsertion-replay-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as raw:
            verifier_root = Path(raw)
            input_root = verifier_root / "inputs"
            frame_root = input_root / "frames"
            labels_snapshot = _materialize_verifier_snapshot(
                root=input_root,
                filename="labels.json",
                artifact=replay_artifacts["labels"],
            )
            collision_snapshot = _materialize_verifier_snapshot(
                root=input_root,
                filename="collision.usda",
                artifact=replay_artifacts["collision"],
            )
            visual_review_snapshot = _materialize_verifier_snapshot(
                root=input_root,
                filename="visual_review.json",
                artifact=replay_artifacts["visual_review"],
            )
            render_manifest_snapshot = _materialize_verifier_snapshot(
                root=input_root,
                filename="render_manifest.json",
                artifact=replay_artifacts["render_manifest"],
            )
            attestation_snapshot = _materialize_verifier_snapshot(
                root=input_root,
                filename="semantic_attestation.json",
                artifact=replay_artifacts["semantic_attestation"],
            )
            selection_snapshot = _materialize_verifier_snapshot(
                root=input_root,
                filename="semantic_selection.json",
                artifact=replay_artifacts["semantic_selection"],
            )
            for frame_filename, frame_artifact in frame_snapshots:
                _materialize_verifier_snapshot(
                    root=frame_root,
                    filename=frame_filename,
                    artifact=frame_artifact,
                )
            output_root = verifier_root / "asset"
            replayed = build_registered_static_receptacle_asset(
                labels_path=labels_snapshot,
                sage_collision_usd_path=collision_snapshot,
                topology_receipt=replay_artifacts["topology"]["payload"],
                visual_review_receipt_path=visual_review_snapshot,
                render_manifest_path=render_manifest_snapshot,
                frame_root=frame_root,
                semantic_review_attestation_path=attestation_snapshot,
                semantic_authority_selection_path=selection_snapshot,
                target_instance_id=arguments.get("target_instance_id"),
                entity_id=arguments.get("entity_id"),
                asset_id=arguments.get("asset_id"),
                reference_world_pose=arguments.get("reference_world_pose"),
                rights=arguments.get("rights"),
                authoring_identity=arguments.get("authoring_identity"),
                physics_configuration=arguments.get("physics_configuration"),
                simulator_name=arguments.get("simulator_name"),
                simulator_version=arguments.get("simulator_version"),
                output_root=output_root,
            )
            registered = _typed_payload(
                artifacts,
                authoring_receipt.get("registered_asset_receipt_binding_id"),
                schema_version=REGISTERED_RECEPTACLE_RECEIPT_SCHEMA_VERSION,
            )
            retained_visual_basis = _typed_payload(
                artifacts,
                authoring_receipt.get("visual_basis_binding_id"),
                schema_version=ENGINEERED_RECEPTACLE_VISUAL_BASIS_SCHEMA_VERSION,
            )
            replayed_visual_bytes, _ = _read_absolute_once(
                output_root / REGISTERED_RECEPTACLE_VISUAL_BASIS_FILENAME,
                label="registered_receptacle_replay_visual_basis",
                maximum_size=_MAX_ARTIFACT_BYTES,
            )
            replayed_visual_basis = _strict_json_object(
                replayed_visual_bytes,
                error="task_preinsertion_registered_receptacle_replay_visual_basis_json_invalid",
            )
            normalized_visual_basis = _normalize_verifier_path_metadata(
                replayed_visual_basis,
                retained_visual_basis,
            )
            if isinstance(normalized_visual_basis, dict):
                normalized_visual_basis["basis_digest"] = canonical_digest(
                    normalized_visual_basis,
                    digest_field="basis_digest",
                )
            replayed_receipt = _normalize_verifier_path_metadata(
                replayed.get("receipt"),
                registered,
            )
            if isinstance(replayed_receipt, dict):
                replayed_receipt["visual_design_basis_digest"] = registered.get(
                    "visual_design_basis_digest"
                )
                replayed_receipt["visual_design_basis_file"] = registered.get(
                    "visual_design_basis_file"
                )
                replayed_receipt["receipt_digest"] = canonical_digest(
                    replayed_receipt,
                    digest_field="receipt_digest",
                )
            if (
                replayed.get("candidate") != candidate
                or normalized_visual_basis != retained_visual_basis
                or replayed_receipt != registered
                or arguments.get("entity_id") != entity_id
                or arguments.get("asset_id") != candidate.get("asset_id")
            ):
                return [error]
            expected_output_bindings = {
                REGISTERED_RECEPTACLE_CANDIDATE_FILENAME: candidate_binding_id,
                REGISTERED_RECEPTACLE_RECEIPT_FILENAME: authoring_receipt.get(
                    "registered_asset_receipt_binding_id"
                ),
                REGISTERED_RECEPTACLE_VISUAL_BASIS_FILENAME: authoring_receipt.get(
                    "visual_basis_binding_id"
                ),
            }
            candidate_parent = artifacts[str(candidate_binding_id)]["path"].parent
            for row in _rows(candidate.get("files")):
                expected_output_bindings[str(row.get("path"))] = next(
                    (
                        binding_id
                        for binding_id, artifact in artifacts.items()
                        if artifact.get("path") == candidate_parent / str(row.get("path"))
                    ),
                    None,
                )
            for filename, binding_id in expected_output_bindings.items():
                artifact = artifacts.get(str(binding_id), {})
                output_path = output_root / filename
                output_content, _ = _read_absolute_once(
                    output_path,
                    label="registered_receptacle_replay_output",
                    maximum_size=_MAX_ARTIFACT_BYTES,
                )
                if filename == REGISTERED_RECEPTACLE_VISUAL_BASIS_FILENAME:
                    output_matches = normalized_visual_basis == artifact.get("payload")
                elif filename == REGISTERED_RECEPTACLE_RECEIPT_FILENAME:
                    output_matches = replayed_receipt == artifact.get("payload")
                else:
                    output_matches = artifact.get("content") == output_content
                if not _is_identifier(binding_id) or not output_matches:
                    return [error]
    except (
        KeyError,
        OSError,
        RegisteredStaticReceptacleAssetError,
        TaskPreinsertionReadinessError,
        TypeError,
        ValueError,
    ):
        return [error]
    return []


def _engineered_candidate_errors(
    *,
    entity_id: str,
    runtime: Mapping[str, Any],
    source: Mapping[str, Any],
    source_evidence: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    binding_id = runtime.get("evidence_binding_id")
    candidate = _typed_payload(
        artifacts,
        binding_id,
        schema_version=TASK_ENTITY_ASSET_CANDIDATE_SCHEMA_VERSION,
    )
    errors: list[str] = []
    try:
        replayed = materialize_task_entity_asset_candidate(_asset_candidate_input(candidate))
    except (TaskEntityAssetCandidateError, TypeError, ValueError):
        replayed = {}
    candidate_source = candidate.get("source_observation")
    source_bounds = source_evidence.get("bounds_world")
    candidate_bounds = (
        candidate_source.get("bounds_world") if isinstance(candidate_source, Mapping) else None
    )
    source_dimensions = source_evidence.get("metric_dimensions_m")
    candidate_dimensions = (
        candidate_source.get("metric_dimensions_m")
        if isinstance(candidate_source, Mapping)
        else None
    )
    source_minimum = source_bounds.get("minimum_m") if isinstance(source_bounds, Mapping) else None
    source_maximum = source_bounds.get("maximum_m") if isinstance(source_bounds, Mapping) else None
    candidate_minimum = (
        candidate_bounds.get("minimum_m") if isinstance(candidate_bounds, Mapping) else None
    )
    candidate_maximum = (
        candidate_bounds.get("maximum_m") if isinstance(candidate_bounds, Mapping) else None
    )
    observation_geometry_joined = bool(
        isinstance(candidate_source, Mapping)
        and candidate_source.get("observation_id")
        == f"sage-component:{source_evidence.get('source_instance_id')}"
        and _finite_vector(source_minimum, length=3)
        and _finite_vector(source_maximum, length=3)
        and _finite_vector(candidate_minimum, length=3)
        and _finite_vector(candidate_maximum, length=3)
        and _finite_vector(source_dimensions, length=3)
        and _finite_vector(candidate_dimensions, length=3)
        and all(
            abs(float(candidate_minimum[index]) - float(source_minimum[index])) <= 1.0e-7
            and abs(float(candidate_maximum[index]) - float(source_maximum[index])) <= 1.0e-7
            and abs(float(candidate_dimensions[index]) - float(source_dimensions[index])) <= 1.0e-7
            for index in range(3)
        )
    )
    if (
        replayed != candidate
        or candidate.get("entity_id") != entity_id
        or candidate.get("asset_id") != runtime.get("asset_id")
        or candidate.get("asset_class") != "rigid_receptacle"
        or not observation_geometry_joined
        or candidate.get("status") != "simready_candidate_pending_native_qualification"
        or candidate.get("claims", {}).get("simready_candidate") is not True
        or candidate.get("claims", {}).get("native_simulator_qualified") is not False
        or candidate.get("claims", {}).get("physically_equivalent_real_material") is not False
        or runtime.get("sha256") != candidate.get("candidate_digest")
    ):
        errors.append(f"task_preinsertion_entity_engineered_candidate_invalid:{entity_id}")
        return errors
    candidate_artifact = artifacts.get(str(binding_id), {})
    candidate_path = candidate_artifact.get("path")
    candidate_parent = candidate_path.parent if isinstance(candidate_path, Path) else None
    bindings_by_path = {
        artifact["path"]: artifact
        for artifact in artifacts.values()
        if isinstance(artifact.get("path"), Path)
    }
    files_by_role: dict[str, Mapping[str, Any]] = {}
    for file_row in _rows(candidate.get("files")):
        path = file_row.get("path")
        artifact = (
            bindings_by_path.get(candidate_parent / str(path), {})
            if candidate_parent is not None
            else {}
        )
        if (
            not isinstance(path, str)
            or artifact.get("binding", {}).get("sha256") != file_row.get("sha256")
            or len(artifact.get("content", b"")) != file_row.get("size_bytes")
        ):
            errors.append(f"task_preinsertion_entity_engineered_candidate_file_invalid:{entity_id}")
            break
        files_by_role[str(file_row.get("role"))] = artifact
    structure_errors = _rigid_receptacle_structure_errors(
        candidate=candidate,
        files_by_role=files_by_role,
    )
    errors.extend(f"{item}:{entity_id}" for item in structure_errors)

    authoring_receipt = _typed_payload(
        artifacts,
        runtime.get("authoring_receipt_binding_id"),
        schema_version=ENGINEERED_ASSET_EVIDENCE_SCHEMA_VERSION,
    )
    registered_fields = {
        "registered_asset_receipt_binding_id",
        "registered_asset_receipt_sha256",
        "topology_receipt_binding_id",
        "topology_receipt_sha256",
        "visual_basis_binding_id",
        "visual_basis_sha256",
        "semantic_attestation_binding_id",
        "semantic_attestation_sha256",
        "semantic_selection_binding_id",
        "semantic_selection_sha256",
        "builder_source_binding_id",
        "builder_source_sha256",
        "builder_replay_request_binding_id",
        "builder_replay_request_sha256",
    }
    if (
        set(authoring_receipt)
        != {
            "schema_version",
            "evidence_id",
            "entity_id",
            "asset_id",
            "candidate_binding_id",
            "candidate_digest",
            *registered_fields,
            "contract_replay_passed",
            "all_candidate_files_bound",
            "static_asset_structure_readback_passed",
            "native_simulator_qualified",
            "receipt_digest",
        }
        or not _is_identifier(authoring_receipt.get("evidence_id"))
        or authoring_receipt.get("entity_id") != entity_id
        or authoring_receipt.get("asset_id") != runtime.get("asset_id")
        or authoring_receipt.get("candidate_binding_id") != binding_id
        or authoring_receipt.get("candidate_digest") != candidate.get("candidate_digest")
        or any(
            not _is_identifier(authoring_receipt.get(field))
            for field in registered_fields
            if field.endswith("_binding_id")
        )
        or any(
            not _is_digest(authoring_receipt.get(field))
            for field in registered_fields
            if field.endswith("_sha256")
        )
        or authoring_receipt.get("contract_replay_passed") is not True
        or authoring_receipt.get("all_candidate_files_bound") is not True
        or authoring_receipt.get("static_asset_structure_readback_passed")
        is not (not structure_errors)
        or authoring_receipt.get("native_simulator_qualified") is not False
        or not _receipt_digest_valid(authoring_receipt, "receipt_digest")
    ):
        errors.append(f"task_preinsertion_entity_engineered_authoring_evidence_invalid:{entity_id}")
    errors.extend(
        _registered_receptacle_evidence_errors(
            entity_id=entity_id,
            candidate=candidate,
            candidate_binding_id=binding_id,
            source=source,
            authoring_receipt=authoring_receipt,
            artifacts=artifacts,
        )
    )
    errors.extend(
        _registered_receptacle_replay_errors(
            entity_id=entity_id,
            candidate=candidate,
            candidate_binding_id=binding_id,
            authoring_receipt=authoring_receipt,
            artifacts=artifacts,
        )
    )
    return errors


def _entity_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    rights_sources: Mapping[str, Mapping[str, Any]],
    coordinate_frame_id: Any,
) -> tuple[list[str], dict[str, Any], dict[str, Mapping[str, Any]]]:
    errors: list[str] = []
    if (
        set(value)
        != {
            "schema_version",
            "scene_id",
            "task_id",
            "task_kind",
            "entities",
            "inventory_digest",
        }
        or value.get("scene_id") != manifest["scene_id"]
        or value.get("task_id") != manifest["task_id"]
        or value.get("task_kind") != manifest["task_kind"]
        or not _receipt_digest_valid(value, "inventory_digest")
    ):
        errors.append("task_preinsertion_entity_inventory_invalid")
    entities: dict[str, Mapping[str, Any]] = {}
    lineage: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    support_relations: dict[str, Mapping[str, Any]] = {}
    entity_fields = {
        "entity_id",
        "semantic_role",
        "physics_type",
        "source_observation",
        "runtime_asset",
    }
    source_fields = {
        "classification",
        "source_id",
        "source_sha256",
        "observed",
        "evidence_binding_id",
    }
    runtime_fields = {
        "origin",
        "status",
        "asset_id",
        "sha256",
        "evidence_binding_id",
        "authoring_receipt_binding_id",
        "design_basis_observation_binding_id",
        "observed_source_truth",
        "physical_equivalence_claimed",
    }
    entity_rows = _rows(value.get("entities"))
    if not entity_rows or len(entity_rows) > 256:
        errors.append("task_preinsertion_entity_inventory_size_invalid")
    for row in entity_rows:
        entity_id = row.get("entity_id")
        role = row.get("semantic_role")
        physics = row.get("physics_type")
        if (
            set(row) != entity_fields
            or not _is_identifier(entity_id)
            or entity_id in entities
            or not isinstance(role, str)
            or role not in _PHYSICS_BY_ROLE
            or not isinstance(physics, str)
            or physics not in _PHYSICS_BY_ROLE.get(role, frozenset())
        ):
            errors.append(f"task_preinsertion_entity_invalid:{entity_id or 'missing'}")
            continue
        source = row.get("source_observation")
        runtime = row.get("runtime_asset")
        if not isinstance(source, Mapping) or set(source) != source_fields:
            errors.append(f"task_preinsertion_entity_source_invalid:{entity_id}")
            source = {}
        if not isinstance(runtime, Mapping) or set(runtime) != runtime_fields:
            errors.append(f"task_preinsertion_entity_runtime_invalid:{entity_id}")
            runtime = {}
        classification = source.get("classification")
        origin = runtime.get("origin")
        source_binding_id = source.get("evidence_binding_id")
        source_evidence = _typed_payload(
            artifacts,
            source_binding_id,
            schema_version=SOURCE_EVIDENCE_SCHEMA_VERSION,
        )
        source_evidence_valid = bool(
            set(source_evidence)
            == {
                "schema_version",
                "evidence_id",
                "entity_id",
                "source_id",
                "source_sha256",
                "classification",
                "observed",
                "design_basis_only",
                "source_instance_id",
                "coordinate_frame_id",
                "bounds_world",
                "metric_dimensions_m",
                "rest_state",
                "support_relation",
                "cited_visual_evidence",
                "cited_visual_evidence_digest",
                "semantic_authority",
                "receipt_digest",
            }
            and source_evidence.get("entity_id") == entity_id
            and _is_identifier(source_evidence.get("evidence_id"))
            and source_evidence.get("source_id") == source.get("source_id")
            and source_evidence.get("source_sha256") == source.get("source_sha256")
            and source_evidence.get("classification") == classification
            and source_evidence.get("observed") == source.get("observed")
            and source_evidence.get("design_basis_only")
            is (runtime.get("origin") == "engineered_composed_asset")
            and (
                _source_observation_authority_signature_valid(source_evidence)
                if classification == "observed_source"
                else source_evidence.get("semantic_authority") is None
            )
            and _receipt_digest_valid(source_evidence, "receipt_digest")
        )
        bounds = source_evidence.get("bounds_world")
        minimum = bounds.get("minimum_m") if isinstance(bounds, Mapping) else None
        maximum = bounds.get("maximum_m") if isinstance(bounds, Mapping) else None
        dimensions = source_evidence.get("metric_dimensions_m")
        geometry_evidence_valid = bool(
            isinstance(bounds, Mapping)
            and set(bounds) == {"minimum_m", "maximum_m"}
            and _finite_vector(minimum, length=3)
            and _finite_vector(maximum, length=3)
            and _finite_vector(dimensions, length=3)
            and all(float(maximum[index]) > float(minimum[index]) for index in range(3))
            and all(float(dimensions[index]) > 0.0 for index in range(3))
            and all(
                abs(float(maximum[index]) - float(minimum[index]) - float(dimensions[index]))
                <= 1.0e-7
                for index in range(3)
            )
        )
        citations = source_evidence.get("cited_visual_evidence")
        citation_rows = _rows(citations)
        citation_identities: set[tuple[str, str]] = set()
        visual_evidence_valid = isinstance(citations, list) and len(citation_rows) == len(citations)
        for citation in citation_rows:
            citation_binding_id = citation.get("binding_id")
            citation_artifact = artifacts.get(str(citation_binding_id), {})
            provenance_binding_id = citation.get("provenance_binding_id")
            provenance = _typed_payload(
                artifacts,
                provenance_binding_id,
                schema_version=VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION,
            )
            decoded = _decoded_visual_identity(citation_artifact.get("content", b""))
            identity = (
                str(citation.get("camera_id")),
                str(citation_binding_id),
            )
            if (
                set(citation)
                != {
                    "binding_id",
                    "camera_id",
                    "sha256",
                    "size_bytes",
                    "width",
                    "height",
                    "decoded_rgb_sha256",
                    "provenance_binding_id",
                    "provenance_sha256",
                }
                or not _is_identifier(citation_binding_id)
                or not _is_identifier(citation.get("camera_id"))
                or identity in citation_identities
                or citation_binding_id not in artifacts
                or citation_artifact.get("binding", {}).get("content_type") != "opaque"
                or not str(citation_artifact.get("binding", {}).get("relative_path", ""))
                .lower()
                .endswith(".png")
                or citation_artifact.get("binding", {}).get("sha256") != citation.get("sha256")
                or type(citation.get("size_bytes")) is not int
                or citation.get("size_bytes", 0) <= 0
                or len(citation_artifact.get("content", b"")) != citation.get("size_bytes")
                or decoded is None
                or decoded.get("width") != citation.get("width")
                or decoded.get("height") != citation.get("height")
                or decoded.get("decoded_rgb_sha256") != citation.get("decoded_rgb_sha256")
                or not _binding_matches(
                    artifacts,
                    provenance_binding_id,
                    sha256=citation.get("provenance_sha256"),
                )
                or set(provenance)
                != {
                    "schema_version",
                    "evidence_id",
                    "entity_id",
                    "source_instance_id",
                    "source_id",
                    "source_sha256",
                    "coordinate_frame_id",
                    "camera_id",
                    "frame_binding_id",
                    "frame_sha256",
                    "frame_size_bytes",
                    "width",
                    "height",
                    "decoded_rgb_sha256",
                    "producer_identity",
                    "receipt_digest",
                }
                or not _is_identifier(provenance.get("evidence_id"))
                or provenance.get("entity_id") != entity_id
                or provenance.get("source_instance_id") != source_evidence.get("source_instance_id")
                or provenance.get("source_id") != source.get("source_id")
                or provenance.get("source_sha256") != source.get("source_sha256")
                or provenance.get("coordinate_frame_id") != coordinate_frame_id
                or provenance.get("camera_id") != citation.get("camera_id")
                or provenance.get("frame_binding_id") != citation_binding_id
                or provenance.get("frame_sha256") != citation.get("sha256")
                or provenance.get("frame_size_bytes") != citation.get("size_bytes")
                or provenance.get("width") != citation.get("width")
                or provenance.get("height") != citation.get("height")
                or provenance.get("decoded_rgb_sha256") != citation.get("decoded_rgb_sha256")
                or not isinstance(provenance.get("producer_identity"), Mapping)
                or set(provenance.get("producer_identity", {}))
                != {"kind", "producer", "version", "configuration_sha256"}
                or provenance.get("producer_identity", {}).get("kind")
                not in {"calibrated_capture", "registered_scene_render"}
                or not _is_identifier(provenance.get("producer_identity", {}).get("producer"))
                or not isinstance(provenance.get("producer_identity", {}).get("version"), str)
                or not provenance.get("producer_identity", {}).get("version")
                or not _is_digest(
                    provenance.get("producer_identity", {}).get("configuration_sha256")
                )
                or not _receipt_digest_valid(provenance, "receipt_digest")
            ):
                visual_evidence_valid = False
            citation_identities.add(identity)
        if classification == "observed_source" and not citation_rows:
            visual_evidence_valid = False
        if classification == "runtime_embodiment" and citation_rows:
            visual_evidence_valid = False
        try:
            expected_visual_digest = canonical_digest({"cited_visual_evidence": citation_rows})
        except (RecursionError, TypeError, ValueError):
            expected_visual_digest = None
        visual_evidence_valid = bool(
            visual_evidence_valid
            and source_evidence.get("cited_visual_evidence_digest") == expected_visual_digest
        )
        if (
            not isinstance(classification, str)
            or classification not in {"observed_source", "runtime_embodiment"}
            or (
                classification == "runtime_embodiment"
                and (role != "robot" or origin != "runtime_embodiment")
            )
            or (role != "robot" and classification != "observed_source")
            or not _is_identifier(source.get("source_id"))
            or not _is_digest(source.get("source_sha256"))
            or not isinstance(source.get("observed"), bool)
            or not _is_identifier(source_binding_id)
            or source_binding_id not in artifacts
            or not source_evidence_valid
            or not _is_identifier(source_evidence.get("source_instance_id"))
            or source_evidence.get("coordinate_frame_id") != coordinate_frame_id
            or not geometry_evidence_valid
            or not isinstance(source_evidence.get("rest_state"), str)
            or not source_evidence.get("rest_state", "").strip()
            or not isinstance(source_evidence.get("support_relation"), Mapping)
            or set(source_evidence.get("support_relation", {})) != {"relation", "support_entity_id"}
            or source_evidence.get("support_relation", {}).get("relation")
            not in {"supported_by", "static_scene_anchor", "runtime_mount"}
            or (
                source_evidence.get("support_relation", {}).get("relation") == "supported_by"
                and not _is_identifier(
                    source_evidence.get("support_relation", {}).get("support_entity_id")
                )
            )
            or (
                source_evidence.get("support_relation", {}).get("relation") != "supported_by"
                and source_evidence.get("support_relation", {}).get("support_entity_id") is not None
            )
            or not visual_evidence_valid
            or (classification == "observed_source" and source.get("observed") is not True)
            or (classification == "runtime_embodiment" and source.get("observed") is not False)
            or (
                classification == "observed_source"
                and source.get("source_id") not in rights_sources
            )
            or (
                classification == "observed_source"
                and source.get("source_id") in rights_sources
                and source.get("source_sha256")
                != rights_sources[str(source.get("source_id"))].get("sha256")
            )
        ):
            errors.append(f"task_preinsertion_entity_source_invalid:{entity_id}")
        status = runtime.get("status")
        runtime_binding_id = runtime.get("evidence_binding_id")
        design_binding_id = runtime.get("design_basis_observation_binding_id")
        if status == "pending_asset_slot":
            pending_ids.append(str(entity_id))
            if (
                origin != "pending_asset_slot"
                or any(
                    runtime.get(field) is not None
                    for field in (
                        "asset_id",
                        "sha256",
                        "evidence_binding_id",
                        "authoring_receipt_binding_id",
                        "design_basis_observation_binding_id",
                    )
                )
                or runtime.get("observed_source_truth") is not False
                or runtime.get("physical_equivalence_claimed") is not False
            ):
                errors.append(f"task_preinsertion_entity_pending_slot_invalid:{entity_id}")
        elif status in {"ready", "candidate_ready_pending_native"}:
            if (
                not isinstance(origin, str)
                or origin
                not in {"registered_source", "engineered_composed_asset", "runtime_embodiment"}
                or not _is_identifier(runtime.get("asset_id"))
                or not _is_digest(runtime.get("sha256"))
                or not _is_identifier(runtime_binding_id)
                or runtime_binding_id not in artifacts
                or (
                    origin != "engineered_composed_asset"
                    and runtime_binding_id in artifacts
                    and runtime.get("sha256")
                    != artifacts[str(runtime_binding_id)]["binding"]["sha256"]
                )
                or runtime.get("physical_equivalence_claimed") is not False
            ):
                errors.append(f"task_preinsertion_entity_runtime_invalid:{entity_id}")
            if origin == "registered_source" and (
                status != "ready"
                or runtime.get("authoring_receipt_binding_id") is not None
                or classification != "observed_source"
                or runtime.get("observed_source_truth") is not True
                or design_binding_id is not None
            ):
                errors.append(f"task_preinsertion_entity_registered_source_invalid:{entity_id}")
            if origin == "engineered_composed_asset" and (
                status != "candidate_ready_pending_native"
                or runtime.get("observed_source_truth") is not False
                or not _is_identifier(runtime.get("authoring_receipt_binding_id"))
                or runtime.get("authoring_receipt_binding_id") not in artifacts
                or not _is_identifier(design_binding_id)
                or design_binding_id not in artifacts
                or design_binding_id != source_binding_id
                or classification != "observed_source"
            ):
                errors.append(f"task_preinsertion_entity_engineered_asset_invalid:{entity_id}")
            if origin == "engineered_composed_asset":
                errors.extend(
                    _engineered_candidate_errors(
                        entity_id=str(entity_id),
                        runtime=runtime,
                        source=source,
                        source_evidence=source_evidence,
                        artifacts=artifacts,
                    )
                )
            if origin == "runtime_embodiment" and (
                status != "ready"
                or runtime.get("authoring_receipt_binding_id") is not None
                or classification != "runtime_embodiment"
                or runtime.get("observed_source_truth") is not False
                or design_binding_id is not None
            ):
                errors.append(f"task_preinsertion_entity_embodiment_invalid:{entity_id}")
        else:
            errors.append(f"task_preinsertion_entity_runtime_status_invalid:{entity_id}")
        entities[str(entity_id)] = row
        if isinstance(source_evidence.get("support_relation"), Mapping):
            support_relations[str(entity_id)] = source_evidence["support_relation"]
        lineage.append(
            {
                "entity_id": entity_id,
                "semantic_role": role,
                "source_observation_classification": classification,
                "source_observed": source.get("observed"),
                "source_instance_id": source_evidence.get("source_instance_id"),
                "source_bounds_world": bounds,
                "source_metric_dimensions_m": dimensions,
                "source_rest_state": source_evidence.get("rest_state"),
                "source_support_relation": source_evidence.get("support_relation"),
                "source_visual_evidence_digest": source_evidence.get(
                    "cited_visual_evidence_digest"
                ),
                "source_semantic_authority_id": (
                    source_evidence.get("semantic_authority", {}).get("authority_id")
                    if isinstance(source_evidence.get("semantic_authority"), Mapping)
                    else None
                ),
                "runtime_asset_origin": origin,
                "runtime_asset_status": status,
                "runtime_asset_is_observed_source_truth": runtime.get("observed_source_truth"),
                "physical_equivalence_claimed": runtime.get("physical_equivalence_claimed"),
            }
        )
    for entity_id, relation in support_relations.items():
        relation_kind = relation.get("relation")
        support_entity_id = relation.get("support_entity_id")
        entity_role = entities.get(entity_id, {}).get("semantic_role")
        if relation_kind == "supported_by":
            support_role = entities.get(str(support_entity_id), {}).get("semantic_role")
            if support_entity_id == entity_id or support_role != "support_surface":
                errors.append(f"task_preinsertion_entity_source_invalid:{entity_id}")
        elif relation_kind == "static_scene_anchor":
            if entity_role not in {"support_surface", "obstacle"}:
                errors.append(f"task_preinsertion_entity_source_invalid:{entity_id}")
        elif relation_kind == "runtime_mount" and entity_role != "robot":
            errors.append(f"task_preinsertion_entity_source_invalid:{entity_id}")
    role_index: dict[str, list[str]] = {}
    for entity_id, row in entities.items():
        role_index.setdefault(str(row.get("semantic_role")), []).append(entity_id)
    required_roles = _REQUIRED_ROLES_BY_TASK_KIND[manifest["task_kind"]]
    if not required_roles.issubset(role_index):
        errors.append("task_preinsertion_entity_required_roles_missing")
    if len(role_index.get("robot", [])) != 1:
        errors.append("task_preinsertion_entity_robot_cardinality_invalid")
    slot_id = manifest["asset_slot"]["entity_id"]
    slot_row = entities.get(slot_id, {})
    target_role = manifest["asset_slot"]["semantic_role"]
    if (
        pending_ids != [slot_id]
        or slot_row.get("semantic_role") != target_role
        or slot_row.get("physics_type") != manifest["asset_slot"]["physics_type"]
        or role_index.get(str(target_role)) != [slot_id]
    ):
        errors.append("task_preinsertion_entity_asset_slot_join_invalid")
    return (
        sorted(set(errors)),
        {
            "entity_ids": sorted(entities),
            "semantic_role_index": {role: sorted(ids) for role, ids in sorted(role_index.items())},
            "lineage": sorted(lineage, key=lambda row: str(row["entity_id"])),
        },
        entities,
    )


def _placement_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    errors: list[str] = []
    if (
        set(value)
        != {
            "schema_version",
            "scene_id",
            "task_id",
            "status",
            "placements",
            "receipt_digest",
        }
        or value.get("scene_id") != manifest["scene_id"]
        or value.get("task_id") != manifest["task_id"]
        or value.get("status") != "passed"
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_placement_suite_invalid")
    cells: dict[str, str] = {}
    destination_placements: dict[str, Mapping[str, Any]] = {}
    selection_digests: dict[str, str] = {}
    placement_seeds: dict[str, int] = {}
    task_entity_placements: dict[str, dict[str, Mapping[str, Any]]] = {}
    placement_rows = _rows(value.get("placements"))
    if not placement_rows or len(placement_rows) > 32:
        errors.append("task_preinsertion_placement_inventory_invalid")
    for row in placement_rows:
        if set(row) != {
            "cell_id",
            "receipt_binding_id",
            "receipt_digest",
            "robot_entity_id",
        }:
            errors.append("task_preinsertion_placement_binding_invalid")
            continue
        cell_id = row.get("cell_id")
        binding_id = row.get("receipt_binding_id")
        artifact = artifacts.get(str(binding_id), {})
        receipt = artifact.get("payload")
        if (
            not _is_identifier(cell_id)
            or cell_id in cells
            or not _is_identifier(binding_id)
            or not isinstance(receipt, Mapping)
            or receipt.get("schema_version") != COMPOSED_PLACEMENT_SCHEMA_VERSION
            or row.get("receipt_digest") != receipt.get("receipt_digest")
        ):
            errors.append(f"task_preinsertion_placement_binding_invalid:{cell_id or 'missing'}")
            continue
        request = receipt.get("request")
        try:
            if not isinstance(request, Mapping):
                raise ValueError("request_not_mapping")
            replayed = plan_composed_paired_entity_placement(
                support_regions=request.get("support_regions") or [],
                obstacle_aabbs=request.get("obstacle_aabbs") or [],
                entity_specs=request.get("entity_specs") or [],
                canonical_task_centers_m=request.get("canonical_task_centers_m") or [],
                robot_spec=(
                    request.get("robot_spec")
                    if isinstance(request.get("robot_spec"), Mapping)
                    else {}
                ),
                minimum_separations_m=(
                    request.get("minimum_separations_m")
                    if isinstance(request.get("minimum_separations_m"), Mapping)
                    else {}
                ),
                grid_spacing_m=request.get("grid_spacing_m"),
                frozen_seed=request.get("frozen_seed_uint64"),
                maximum_combination_count=request.get("maximum_combination_count"),
            )
        except Exception:
            errors.append(f"task_preinsertion_placement_replay_invalid:{cell_id}")
            continue
        if (
            replayed != receipt
            or replayed.get("status") != "geometry_plausibility_candidate_selected"
            or replayed.get("blockers") != []
        ):
            errors.append(f"task_preinsertion_placement_replay_mismatch:{cell_id}")
            continue
        selected_rows = _rows(
            replayed.get("selection", {}).get("entity_placements")
            if isinstance(replayed.get("selection"), Mapping)
            else None
        )
        selected_ids = {str(item.get("subject_id")) for item in selected_rows}
        destination_ids = {
            entity_id
            for entity_id, entity in entities.items()
            if entity.get("semantic_role") == "destination_receptacle"
        }
        robot_ids = {
            entity_id
            for entity_id, entity in entities.items()
            if entity.get("semantic_role") == "robot"
        }
        robot_placement = (
            replayed.get("selection", {}).get("robot_base_placement")
            if isinstance(replayed.get("selection"), Mapping)
            else None
        )
        dimension_join_valid = True
        for spec in _rows(request.get("entity_specs")):
            spec_entity_id = spec.get("entity_id")
            entity = entities.get(str(spec_entity_id), {})
            source = entity.get("source_observation")
            source_receipt = _typed_payload(
                artifacts,
                source.get("evidence_binding_id") if isinstance(source, Mapping) else None,
                schema_version=SOURCE_EVIDENCE_SCHEMA_VERSION,
            )
            dimensions = source_receipt.get("metric_dimensions_m")
            footprint = spec.get("footprint_xy_m")
            height = spec.get("height_m")
            if (
                not _finite_vector(dimensions, length=3)
                or not _finite_vector(footprint, length=2)
                or isinstance(height, bool)
                or not isinstance(height, (int, float))
                or any(
                    abs(float(footprint[index]) - float(dimensions[index])) > 1.0e-7
                    for index in range(2)
                )
                or abs(float(height) - float(dimensions[2])) > 1.0e-7
            ):
                dimension_join_valid = False
        if (
            manifest["asset_slot"]["entity_id"] not in selected_ids
            or not selected_ids.issubset(entities)
            or (
                manifest["task_kind"] == TASK_KIND_DEFORMABLE_TRANSFER
                and len(selected_ids & destination_ids) != 1
            )
            or robot_ids != {row.get("robot_entity_id")}
            or not isinstance(robot_placement, Mapping)
            or robot_placement.get("subject_id") != "robot_base"
        ):
            errors.append(f"task_preinsertion_placement_entity_join_invalid:{cell_id}")
            continue
        if not dimension_join_valid:
            errors.append(f"task_preinsertion_placement_entity_dimensions_join_invalid:{cell_id}")
            continue
        placement_seed = request.get("frozen_seed_uint64")
        if type(placement_seed) is not int or not 0 <= placement_seed < 2**64:
            errors.append(f"task_preinsertion_placement_seed_invalid:{cell_id}")
            continue
        destination_rows = [
            item for item in selected_rows if item.get("subject_id") in destination_ids
        ]
        if manifest["task_kind"] == TASK_KIND_DEFORMABLE_TRANSFER:
            if len(destination_rows) != 1:
                errors.append(f"task_preinsertion_placement_destination_join_invalid:{cell_id}")
                continue
        if len(destination_rows) == 1:
            destination_placements[str(cell_id)] = destination_rows[0]
        cells[str(cell_id)] = str(receipt["receipt_digest"])
        placement_seeds[str(cell_id)] = placement_seed
        task_relevant_ids = {str(manifest["asset_slot"]["entity_id"]), *destination_ids}
        task_entity_placements[str(cell_id)] = {
            str(item["subject_id"]): dict(item)
            for item in selected_rows
            if str(item.get("subject_id")) in task_relevant_ids
        }
        try:
            selection_digests[str(cell_id)] = canonical_digest(
                {"selection": replayed.get("selection")}
            )
        except (RecursionError, TypeError, ValueError):
            errors.append(f"task_preinsertion_placement_selection_invalid:{cell_id}")
    if not cells:
        errors.append("task_preinsertion_placement_cells_missing")
    return (
        sorted(set(errors)),
        {
            "cell_receipt_digests": dict(sorted(cells.items())),
            "destination_placements": {
                cell_id: dict(row) for cell_id, row in sorted(destination_placements.items())
            },
            "selection_digests": dict(sorted(selection_digests.items())),
            "placement_seeds": dict(sorted(placement_seeds.items())),
            "task_entity_placements": {
                cell_id: dict(sorted(rows.items()))
                for cell_id, rows in sorted(task_entity_placements.items())
            },
        },
        cells,
    )


def _rigid_task_spec_valid(spec: Mapping[str, Any]) -> bool:
    fields = {
        "schema_version",
        "task_kind",
        "destination_position_world_m",
        "support_plane_z_m",
        "settle_window_samples",
        "require_sealed_start_pose",
    }
    return bool(
        set(spec) == fields
        and spec.get("schema_version") == TASK_SPEC_SCHEMA_VERSION
        and spec.get("task_kind") == TASK_KIND_RIGID_PICK_PLACE
        and _finite_vector(spec.get("destination_position_world_m"), length=3)
        and not isinstance(spec.get("support_plane_z_m"), bool)
        and isinstance(spec.get("support_plane_z_m"), (int, float))
        and math.isfinite(float(spec.get("support_plane_z_m")))
        and type(spec.get("settle_window_samples")) is int
        and spec.get("settle_window_samples", 0) > 0
        and spec.get("require_sealed_start_pose") is True
    )


def _articulated_task_spec_valid(spec: Mapping[str, Any]) -> bool:
    fields = {
        "schema_version",
        "task_kind",
        "target_joint_id",
        "joint_reset_positions_rad",
        "target_success_interval_rad",
        "joint_hard_limits_rad",
        "settle_window_samples",
        "maximum_settled_target_speed_rad_s",
        "non_task_joint_motion_tolerance_rad",
        "movement_epsilon_rad",
        "reset_tolerance_rad",
    }
    if set(spec) != fields:
        return False
    try:
        validate_articulated_task_spec(spec)
    except TaskNeutralScoringError:
        return False
    return True


def _deformable_destination_contract(
    *,
    placement_row: Any,
    interior_dimensions: Any,
    floor_thickness: Any,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if (
        not isinstance(placement_row, Mapping)
        or not _finite_vector(placement_row.get("center_world_m"), length=3)
        or not _finite_vector(placement_row.get("aabb_min_m"), length=3)
        or not _finite_vector(interior_dimensions, length=3)
        or isinstance(floor_thickness, bool)
        or not isinstance(floor_thickness, (int, float))
        or not math.isfinite(float(floor_thickness))
    ):
        return None
    center = placement_row["center_world_m"]
    aabb_minimum = placement_row["aabb_min_m"]
    reference_position = [
        float(center[0]),
        float(center[1]),
        float(aabb_minimum[2]),
    ]
    reference_pose = {
        "position_m": reference_position,
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    destination_volume = {
        "center_world_m": [
            reference_position[0],
            reference_position[1],
            reference_position[2] + float(floor_thickness) + float(interior_dimensions[2]) / 2.0,
        ],
        "half_extents_m": [float(interior_dimensions[index]) / 2.0 for index in range(3)],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    return reference_pose, destination_volume


def _scorer_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    entities: Mapping[str, Mapping[str, Any]],
    task_evidence: Mapping[str, Any],
    placement_evidence: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    fields = {
        "schema_version",
        "task_id",
        "task_kind",
        "target_entity_id",
        "destination_entity_id",
        "deterministic",
        "policy_self_grading_allowed",
        "caller_asserted_outcomes_accepted",
        "prompt",
        "task_spec",
        "cell_task_specs",
        "scorer_source_binding_id",
        "receipt_digest",
    }
    target_id = value.get("target_entity_id")
    source_binding_id = value.get("scorer_source_binding_id")
    scorer_source = artifacts.get(str(source_binding_id), {})
    scorer_source_path = scorer_source.get("path")
    expected_scorer_path = Path(__file__).with_name("adp_task_scoring.py")
    if (
        set(value) != fields
        or value.get("task_id") != manifest["task_id"]
        or value.get("task_kind") != manifest["task_kind"]
        or target_id != manifest["asset_slot"]["entity_id"]
        or target_id not in entities
        or value.get("deterministic") is not True
        or value.get("policy_self_grading_allowed") is not False
        or value.get("caller_asserted_outcomes_accepted") is not False
        or not isinstance(value.get("prompt"), str)
        or value.get("prompt") != task_evidence.get("prompt")
        or not _is_identifier(source_binding_id)
        or source_binding_id not in artifacts
        or scorer_source.get("binding", {}).get("content_type") != "opaque"
        or not isinstance(scorer_source_path, Path)
        or scorer_source_path.name != "adp_task_scoring.py"
        or scorer_source.get("content") != expected_scorer_path.read_bytes()
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_scorer_contract_invalid")

    placement_cells = placement_evidence.get("cell_receipt_digests")
    if not isinstance(placement_cells, Mapping):
        placement_cells = {}
    cell_specs: dict[str, Mapping[str, Any]] = {}
    cell_spec_digests: dict[str, str] = {}
    rows = _rows(value.get("cell_task_specs"))
    if not rows or len(rows) > 64:
        errors.append("task_preinsertion_scorer_cell_task_specs_invalid")
    for row in rows:
        cell_id = row.get("cell_id")
        spec = row.get("task_spec")
        spec_digest = _canonical_digest_or_none(spec) if isinstance(spec, Mapping) else None
        if (
            set(row) != {"cell_id", "task_spec", "task_spec_digest"}
            or not _is_identifier(cell_id)
            or cell_id in cell_specs
            or not isinstance(spec, Mapping)
            or row.get("task_spec_digest") != spec_digest
        ):
            errors.append(f"task_preinsertion_scorer_cell_task_spec_invalid:{cell_id or 'missing'}")
            continue
        cell_specs[str(cell_id)] = spec
        cell_spec_digests[str(cell_id)] = str(spec_digest)
    if set(cell_specs) != set(placement_cells) or "canonical" not in cell_specs:
        errors.append("task_preinsertion_scorer_cell_task_spec_set_invalid")

    destination_ids = {
        entity_id
        for entity_id, row in entities.items()
        if row.get("semantic_role") == "destination_receptacle"
    }
    robot_ids = {
        entity_id for entity_id, row in entities.items() if row.get("semantic_role") == "robot"
    }
    destination_cells = placement_evidence.get("destination_placements")
    if not isinstance(destination_cells, Mapping):
        destination_cells = {}

    destination_candidate: Mapping[str, Any] = {}
    interior_dimensions: Any = None
    floor_thickness: Any = None
    if manifest["task_kind"] == TASK_KIND_DEFORMABLE_TRANSFER:
        destination_entity = entities.get(str(value.get("destination_entity_id")), {})
        destination_runtime = destination_entity.get("runtime_asset")
        destination_candidate = _typed_payload(
            artifacts,
            destination_runtime.get("evidence_binding_id")
            if isinstance(destination_runtime, Mapping)
            else None,
            schema_version=TASK_ENTITY_ASSET_CANDIDATE_SCHEMA_VERSION,
        )
        configuration = destination_candidate.get("receptacle_configuration")
        geometry = configuration.get("geometry") if isinstance(configuration, Mapping) else None
        interior_dimensions = (
            geometry.get("interior_dimensions_m") if isinstance(geometry, Mapping) else None
        )
        floor_thickness = (
            geometry.get("floor_thickness_m") if isinstance(geometry, Mapping) else None
        )

    for cell_id, spec in sorted(cell_specs.items()):
        try:
            if manifest["task_kind"] == TASK_KIND_DEFORMABLE_TRANSFER:
                validate_deformable_task_spec(spec)
                if (
                    spec.get("prompt") != value.get("prompt")
                    or spec.get("deformable_entity_id") != target_id
                    or spec.get("destination_entity_id") != value.get("destination_entity_id")
                    or spec.get("destination_entity_id") not in destination_ids
                    or spec.get("robot_entity_id") not in robot_ids
                ):
                    errors.append(f"task_preinsertion_scorer_entity_join_invalid:{cell_id}")
                expected_destination = _deformable_destination_contract(
                    placement_row=destination_cells.get(cell_id),
                    interior_dimensions=interior_dimensions,
                    floor_thickness=floor_thickness,
                )
                if (
                    expected_destination is None
                    or spec.get("receptacle_reference_pose_world") != expected_destination[0]
                    or spec.get("destination_interior_obb") != expected_destination[1]
                ):
                    errors.append(
                        f"task_preinsertion_scorer_destination_placement_join_invalid:{cell_id}"
                    )
            elif manifest["task_kind"] == TASK_KIND_ARTICULATED_OPEN_CLOSE:
                if not _articulated_task_spec_valid(spec):
                    errors.append(f"task_preinsertion_scorer_task_spec_invalid:{cell_id}")
                if value.get("destination_entity_id") is not None:
                    errors.append("task_preinsertion_scorer_destination_join_invalid")
            else:
                if not _rigid_task_spec_valid(spec):
                    errors.append(f"task_preinsertion_scorer_task_spec_invalid:{cell_id}")
                destination_id = value.get("destination_entity_id")
                if destination_id is not None:
                    placement_row = destination_cells.get(cell_id)
                    expected_position = (
                        placement_row.get("center_world_m")
                        if isinstance(placement_row, Mapping)
                        else None
                    )
                    if (
                        destination_id not in destination_ids
                        or not _finite_vector(expected_position, length=3)
                        or spec.get("destination_position_world_m")
                        != [float(item) for item in expected_position]
                    ):
                        errors.append(
                            f"task_preinsertion_scorer_destination_placement_join_invalid:{cell_id}"
                        )
        except TaskNeutralScoringError as exc:
            errors.extend(
                f"task_preinsertion_scorer_task_spec_invalid:{cell_id}:{item}"
                for item in exc.errors
            )

    canonical_spec = cell_specs.get("canonical")
    canonical_spec_digest = _canonical_digest_or_none(canonical_spec)
    if (
        not isinstance(value.get("task_spec"), Mapping)
        or value.get("task_spec") != canonical_spec
        or canonical_spec_digest != task_evidence.get("task_spec_digest")
    ):
        errors.append("task_preinsertion_scorer_task_spec_freeze_join_invalid")
    try:
        prompt_suite_digest = prompt_task_spec_freeze_digest(
            task_kind=str(manifest["task_kind"]),
            prompt=str(value.get("prompt") or ""),
            cell_task_spec_digests=cell_spec_digests,
        )
    except TaskPreinsertionReadinessError:
        prompt_suite_digest = None
    if prompt_suite_digest != task_evidence.get("prompt_task_spec_digest"):
        errors.append("task_preinsertion_scorer_prompt_task_spec_freeze_join_invalid")
    if value.get("prompt") != task_evidence.get("prompt"):
        errors.append("task_preinsertion_scorer_prompt_freeze_join_invalid")
    if manifest["task_kind"] == TASK_KIND_DEFORMABLE_TRANSFER:
        canonical_destination = (
            canonical_spec.get("receptacle_reference_pose_world")
            if isinstance(canonical_spec, Mapping)
            else None
        )
        if destination_candidate.get("transform", {}).get("world_pose") != {
            "position_world_m": (
                canonical_destination.get("position_m")
                if isinstance(canonical_destination, Mapping)
                else None
            ),
            "orientation_xyzw": (
                canonical_destination.get("orientation_xyzw")
                if isinstance(canonical_destination, Mapping)
                else None
            ),
        }:
            errors.append("task_preinsertion_scorer_destination_candidate_join_invalid")
    return sorted(set(errors)), {
        "task_spec_digest": canonical_spec_digest,
        "cell_task_spec_digests": dict(sorted(cell_spec_digests.items())),
        "prompt_task_spec_digest": prompt_suite_digest,
        "scorer_source_binding_id": source_binding_id,
        "deterministic": value.get("deterministic"),
    }


def _camera_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    coordinate_frame_id: Any,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if (
        set(value)
        != {
            "schema_version",
            "scene_id",
            "task_id",
            "status",
            "native_application_claimed",
            "cameras",
            "receipt_digest",
        }
        or value.get("scene_id") != manifest["scene_id"]
        or value.get("task_id") != manifest["task_id"]
        or value.get("status") != "frozen_pending_native_application"
        or value.get("native_application_claimed") is not False
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_camera_contract_invalid")
    camera_fields = {
        "camera_id",
        "role",
        "policy_input",
        "review_only",
        "scoring_input",
        "pose_frame",
        "extrinsics_binding_id",
        "extrinsics_sha256",
        "intrinsics",
        "visibility_thresholds",
    }
    cameras: dict[str, Mapping[str, Any]] = {}
    camera_rows = _rows(value.get("cameras"))
    if len(camera_rows) != 3:
        errors.append("task_preinsertion_camera_roles_invalid")
    for row in camera_rows:
        role = row.get("role")
        intrinsics = row.get("intrinsics")
        thresholds = row.get("visibility_thresholds")
        extrinsics = _typed_payload(
            artifacts,
            row.get("extrinsics_binding_id"),
            schema_version=CAMERA_EXTRINSICS_SCHEMA_VERSION,
        )
        orientation = extrinsics.get("orientation_xyzw")
        orientation_norm = (
            math.sqrt(sum(float(item) ** 2 for item in orientation))
            if _finite_vector(orientation, length=4)
            else 0.0
        )
        if (
            set(row) != camera_fields
            or not isinstance(role, str)
            or role not in {"external", "wrist", "overview"}
            or role in cameras
            or not _is_identifier(row.get("camera_id"))
            or row.get("policy_input") is not (role in {"external", "wrist"})
            or row.get("review_only") is not (role == "overview")
            or row.get("scoring_input") is not False
            or not _is_identifier(row.get("pose_frame"))
            or row.get("pose_frame") != coordinate_frame_id
            or not _is_digest(row.get("extrinsics_sha256"))
            or not _binding_matches(
                artifacts,
                row.get("extrinsics_binding_id"),
                sha256=row.get("extrinsics_sha256"),
            )
            or set(extrinsics)
            != {
                "schema_version",
                "evidence_id",
                "camera_id",
                "pose_frame",
                "translation_m",
                "orientation_xyzw",
                "calibrated_at",
                "receipt_digest",
            }
            or not _is_identifier(extrinsics.get("evidence_id"))
            or extrinsics.get("camera_id") != row.get("camera_id")
            or extrinsics.get("pose_frame") != row.get("pose_frame")
            or not _finite_vector(extrinsics.get("translation_m"), length=3)
            or not _finite_vector(orientation, length=4)
            or abs(orientation_norm - 1.0) > 1.0e-6
            or not isinstance(extrinsics.get("calibrated_at"), str)
            or not extrinsics.get("calibrated_at")
            or not _receipt_digest_valid(extrinsics, "receipt_digest")
            or not isinstance(intrinsics, Mapping)
            or set(intrinsics) != {"fx", "fy", "cx", "cy", "width", "height"}
            or any(
                isinstance(intrinsics.get(field), bool)
                or not isinstance(intrinsics.get(field), (int, float))
                or not math.isfinite(float(intrinsics[field]))
                or float(intrinsics[field]) <= 0
                for field in intrinsics
            )
            or not isinstance(thresholds, Mapping)
            or not thresholds
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or not 0 < float(item) <= 1
                for item in thresholds.values()
            )
        ):
            errors.append(f"task_preinsertion_camera_invalid:{role or 'missing'}")
            continue
        cameras[str(role)] = row
    if set(cameras) != {"external", "wrist", "overview"}:
        errors.append("task_preinsertion_camera_roles_invalid")
    return sorted(set(errors)), {
        "camera_role_ids": {role: row.get("camera_id") for role, row in sorted(cameras.items())},
        "native_application_claimed": value.get("native_application_claimed"),
    }


def _scenario_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    placement_cells: Mapping[str, str],
    placement_evidence: Mapping[str, Any],
    scorer_evidence: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    if (
        set(value)
        != {
            "schema_version",
            "scene_id",
            "task_id",
            "status",
            "candidate_ids",
            "controls_required_in_every_scored_cell",
            "upper_bound_matrix_launched",
            "cells",
            "receipt_digest",
        }
        or value.get("scene_id") != manifest["scene_id"]
        or value.get("task_id") != manifest["task_id"]
        or value.get("candidate_ids") != manifest["candidate_ids"]
        or value.get("status") != "frozen"
        or value.get("controls_required_in_every_scored_cell") is not True
        or value.get("upper_bound_matrix_launched") is not False
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_scenario_contract_invalid")
    cells: dict[str, Mapping[str, Any]] = {}
    families: set[str] = set()
    seeds: set[int] = set()
    placement_seeds = placement_evidence.get("placement_seeds")
    if not isinstance(placement_seeds, Mapping):
        placement_seeds = {}
    cell_rows = _rows(value.get("cells"))
    if not cell_rows or len(cell_rows) > 64:
        errors.append("task_preinsertion_scenario_cell_set_invalid")
    for row in cell_rows:
        cell_id = row.get("cell_id")
        resolved = _typed_payload(
            artifacts,
            row.get("resolved_parameters_binding_id"),
            schema_version=RESOLVED_SCENARIO_CELL_SCHEMA_VERSION,
        )
        if (
            set(row)
            != {
                "cell_id",
                "seed",
                "family",
                "placement_receipt_digest",
                "resolved_parameters_binding_id",
                "resolved_parameters_sha256",
            }
            or not _is_identifier(cell_id)
            or cell_id in cells
            or type(row.get("seed")) is not int
            or not 0 <= row.get("seed", -1) < 2**64
            or row.get("seed") in seeds
            or not isinstance(row.get("family"), str)
            or row.get("family")
            not in {
                "canonical",
                "placement_approach",
                "illumination",
                "camera_sensor",
                "bounded_physics",
                "appearance_material_cousin",
                "held_out_composed",
            }
            or row.get("placement_receipt_digest") != placement_cells.get(str(cell_id))
            or not _is_digest(row.get("resolved_parameters_sha256"))
            or not _binding_matches(
                artifacts,
                row.get("resolved_parameters_binding_id"),
                sha256=row.get("resolved_parameters_sha256"),
            )
            or set(resolved)
            != {
                "schema_version",
                "evidence_id",
                "cell_id",
                "seed",
                "family",
                "resolved_parameters",
                "receipt_digest",
            }
            or not _is_identifier(resolved.get("evidence_id"))
            or resolved.get("cell_id") != cell_id
            or resolved.get("seed") != row.get("seed")
            or resolved.get("family") != row.get("family")
            or not isinstance(resolved.get("resolved_parameters"), Mapping)
            or not resolved.get("resolved_parameters")
            or not _receipt_digest_valid(resolved, "receipt_digest")
        ):
            errors.append(f"task_preinsertion_scenario_cell_invalid:{cell_id or 'missing'}")
            continue
        parameters = resolved.get("resolved_parameters")
        family = str(row["family"])
        family_parameter_fields = {
            "placement_variant": "placement_approach",
            "approach_variant": "placement_approach",
            "illumination_variant": "illumination",
            "camera_sensor_variant": "camera_sensor",
            "bounded_physics_variant": "bounded_physics",
            "appearance_material_cousin_variant": "appearance_material_cousin",
        }
        family_semantics_valid = bool(
            isinstance(parameters, Mapping)
            and set(parameters) == set(family_parameter_fields)
            and all(_is_identifier(parameters.get(field)) for field in family_parameter_fields)
        )
        if family_semantics_valid:
            changed_families = {
                family_id
                for field, family_id in family_parameter_fields.items()
                if parameters.get(field) != "canonical"
            }
            if family == "canonical":
                family_semantics_valid = not changed_families
            elif family == "held_out_composed":
                family_semantics_valid = bool(
                    len(changed_families) >= 2
                    and "placement_approach" in changed_families
                    and parameters.get("placement_variant") != "canonical"
                )
            elif family == "placement_approach":
                family_semantics_valid = bool(
                    changed_families == {"placement_approach"}
                    and parameters.get("placement_variant") != "canonical"
                )
            else:
                family_semantics_valid = changed_families == {family}
        if not family_semantics_valid:
            errors.append(f"task_preinsertion_scenario_family_semantics_invalid:{cell_id}")
        if row.get("seed") != placement_seeds.get(str(cell_id)):
            errors.append(f"task_preinsertion_scenario_placement_seed_join_invalid:{cell_id}")
        cells[str(cell_id)] = row
        families.add(family)
        seeds.add(int(row["seed"]))
    scorer_cells = scorer_evidence.get("cell_task_spec_digests")
    if not isinstance(scorer_cells, Mapping):
        scorer_cells = {}
    canonical_ids = [cell_id for cell_id, row in cells.items() if row.get("family") == "canonical"]
    if (
        not cells
        or set(canonical_ids) != {"canonical"}
        or set(cells) != set(placement_cells)
        or set(cells) != set(scorer_cells)
    ):
        errors.append("task_preinsertion_scenario_cell_set_invalid")
    selection_digests = placement_evidence.get("selection_digests")
    if not isinstance(selection_digests, Mapping):
        selection_digests = {}
    canonical_receipt_digest = placement_cells.get("canonical")
    canonical_selection_digest = selection_digests.get("canonical")
    task_entity_placements = placement_evidence.get("task_entity_placements")
    if not isinstance(task_entity_placements, Mapping):
        task_entity_placements = {}
    canonical_task_placements = task_entity_placements.get("canonical")
    if not isinstance(canonical_task_placements, Mapping):
        canonical_task_placements = {}
    for cell_id, row in cells.items():
        family = row.get("family")
        receipt_digest = placement_cells.get(cell_id)
        selection_digest = selection_digests.get(cell_id)
        cell_task_placements = task_entity_placements.get(cell_id)
        if not isinstance(cell_task_placements, Mapping):
            cell_task_placements = {}
        same_task_entity_set = bool(
            canonical_task_placements
            and set(cell_task_placements) == set(canonical_task_placements)
        )
        changed_task_entities = {
            entity_id
            for entity_id, canonical_placement in canonical_task_placements.items()
            if cell_task_placements.get(entity_id) != canonical_placement
        }
        if family in {"placement_approach", "held_out_composed"}:
            if (
                receipt_digest == canonical_receipt_digest
                or selection_digest == canonical_selection_digest
            ):
                errors.append(
                    f"task_preinsertion_scenario_placement_distinctness_invalid:{cell_id}"
                )
            if not same_task_entity_set or not changed_task_entities:
                errors.append(
                    f"task_preinsertion_scenario_task_placement_distinctness_invalid:{cell_id}"
                )
        elif family != "canonical" and (
            receipt_digest != canonical_receipt_digest
            or selection_digest != canonical_selection_digest
        ):
            errors.append(f"task_preinsertion_scenario_single_family_isolation_invalid:{cell_id}")
    return sorted(set(errors)), {
        "cell_ids": sorted(cells),
        "families": sorted(families),
        "candidate_ids": value.get("candidate_ids"),
    }


def _runtime_gate(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, Any], list[str]]:
    errors: list[str] = []
    fields = {
        "schema_version",
        "task_kind",
        "status",
        "request_binding_id",
        "observations_binding_id",
        "matrix_binding_id",
        "native_execution_completed",
        "native_qualified",
        "scene_run_admitted",
        "receipt_digest",
    }
    if (
        set(value) != fields
        or value.get("task_kind") != manifest["task_kind"]
        or value.get("status") != "static_preflight_passed_dynamic_native_required"
        or value.get("native_execution_completed") is not False
        or value.get("native_qualified") is not False
        or value.get("scene_run_admitted") is not False
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_runtime_contract_invalid")
    request = _typed_payload(
        artifacts,
        value.get("request_binding_id"),
        schema_version=DEFORMABLE_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    )
    observations = _typed_payload(
        artifacts,
        value.get("observations_binding_id"),
        schema_version=PREFLIGHT_OBSERVATIONS_SCHEMA_VERSION,
    )
    matrix = _typed_payload(
        artifacts,
        value.get("matrix_binding_id"),
        schema_version=DEFORMABLE_PREFLIGHT_MATRIX_SCHEMA_VERSION,
    )
    try:
        replayed = build_deformable_native_capability_preflight(
            request=request,
            observations=observations,
        )
    except Exception:
        replayed = {}
    policy_rows = _rows(request.get("policy_identities"))
    observed_candidates = sorted(str(row.get("candidate_id")) for row in policy_rows)
    dynamic_rows = _rows(matrix.get("dynamic_native_canary_gates"))
    dynamic_ids = sorted(str(row.get("check_id")) for row in dynamic_rows)
    if (
        replayed != matrix
        or matrix.get("status") != "static_preflight_passed_native_canary_required"
        or matrix.get("static_checks_passed") is not True
        or matrix.get("blockers") != []
        or matrix.get("native_canary_completed") is not False
        or matrix.get("scene_run_admitted") is not False
        or set(dynamic_ids) != _EXPECTED_PREFLIGHT_DYNAMIC_GATES
        or any(row.get("status") != "pending_native_canary" for row in dynamic_rows)
        or observed_candidates != manifest["candidate_ids"]
    ):
        errors.append("task_preinsertion_runtime_preflight_replay_invalid")
    task_dependent_native_gates = list(_DEPENDENT_NATIVE_GATES_BY_TASK_KIND[manifest["task_kind"]])
    return (
        sorted(set(errors)),
        {
            "preflight_receipt_digest": matrix.get("receipt_digest"),
            "static_check_ids": sorted(
                str(row.get("check_id")) for row in _rows(matrix.get("static_checks"))
            ),
            "preflight_dynamic_canary_gate_ids": dynamic_ids,
            "task_dependent_native_gate_ids": task_dependent_native_gates,
            "native_execution_completed": value.get("native_execution_completed"),
            "native_qualified": value.get("native_qualified"),
            "scene_run_admitted": value.get("scene_run_admitted"),
        },
        task_dependent_native_gates,
    )


def _trust_gate(
    value: Mapping[str, Any], *, artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    source_id = value.get("verifier_source_binding_id")
    source = artifacts.get(str(source_id), {})
    source_path = source.get("path")
    expected_source_path = Path(__file__).with_name("trusted_execution_envelope.py")
    if (
        set(value)
        != {
            "schema_version",
            "status",
            "verifier_source_binding_id",
            "envelope_schema_version",
            "runner_public_key_sha256",
            "signed_return_required",
            "configured_key_match_required",
            "lifecycle_artifacts_verifier_owned",
            "provider_zero_verifier_owned",
            "native_execution_claimed",
            "receipt_digest",
        }
        or value.get("status") != "configured_pending_signed_execution"
        or not _is_identifier(source_id)
        or source_id not in artifacts
        or source.get("binding", {}).get("content_type") != "opaque"
        or not isinstance(source_path, Path)
        or source_path.name != "trusted_execution_envelope.py"
        or source.get("content") != expected_source_path.read_bytes()
        or value.get("envelope_schema_version") != TRUSTED_ENVELOPE_SCHEMA_VERSION
        or not _is_digest(value.get("runner_public_key_sha256"))
        or any(
            value.get(field) is not True
            for field in (
                "signed_return_required",
                "configured_key_match_required",
                "lifecycle_artifacts_verifier_owned",
                "provider_zero_verifier_owned",
            )
        )
        or value.get("native_execution_claimed") is not False
        or not _receipt_digest_valid(value, "receipt_digest")
    ):
        errors.append("task_preinsertion_trust_policy_invalid")
    return sorted(set(errors)), {
        "verifier_source_binding_id": source_id,
        "runner_public_key_sha256": value.get("runner_public_key_sha256"),
        "native_execution_claimed": value.get("native_execution_claimed"),
    }


def _gate_row(
    *, gate_id: str, blockers: Sequence[str], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    typed = sorted(set(str(item) for item in blockers if str(item)))
    return {
        "gate_id": gate_id,
        "required": True,
        "status": "blocked" if typed else "passed",
        "blockers": typed,
        "evidence": dict(evidence),
    }


def collect_task_preinsertion_readiness(
    input_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify one on-disk graph and expose at most one movable-asset slot.

    ``input_manifest_path`` is deliberately the only evidence input.  There is
    no mapping overload: every fact used by the collector must have retained
    bytes and an expected file digest in the manifest.
    """

    manifest_bytes, manifest_path = _read_absolute_once(
        input_manifest_path,
        label="manifest",
        maximum_size=_MAX_MANIFEST_BYTES,
    )
    manifest = _manifest_contract(
        _strict_json_object(manifest_bytes, error="task_preinsertion_manifest_json_invalid")
    )
    artifacts, binding_receipts, binding_blockers = _load_bindings(
        manifest=manifest, root=manifest_path.parent
    )
    gates: list[dict[str, Any]] = []
    gates.append(
        _gate_row(
            gate_id="artifact_bindings",
            blockers=binding_blockers,
            evidence={
                "expected_binding_count": len(manifest["bindings"]),
                "verified_binding_count": len(binding_receipts),
            },
        )
    )

    scene, _ = _core_payload(artifacts, "scene")
    scene_errors, scene_evidence = _scene_gate(scene, manifest=manifest, artifacts=artifacts)
    gates.append(_gate_row(gate_id="scene", blockers=scene_errors, evidence=scene_evidence))

    task, _ = _core_payload(artifacts, "task")
    task_errors, task_evidence = _task_gate(task, manifest=manifest)
    gates.append(_gate_row(gate_id="task", blockers=task_errors, evidence=task_evidence))

    scene_sources = {
        str(row.get("rights_source_id")): row
        for row in (scene.get("appearance"), scene.get("collision"))
        if isinstance(row, Mapping) and _is_identifier(row.get("rights_source_id"))
    }
    rights, _ = _core_payload(artifacts, "rights")
    rights_errors, rights_evidence, rights_sources = _rights_gate(
        rights, required_sources=scene_sources, artifacts=artifacts
    )
    gates.append(_gate_row(gate_id="rights", blockers=rights_errors, evidence=rights_evidence))

    entity_inventory, _ = _core_payload(artifacts, "entities")
    entity_errors, entity_evidence, entities = _entity_gate(
        entity_inventory,
        manifest=manifest,
        artifacts=artifacts,
        rights_sources=rights_sources,
        coordinate_frame_id=scene_evidence.get("coordinate_frame_id"),
    )
    gates.append(_gate_row(gate_id="entities", blockers=entity_errors, evidence=entity_evidence))

    placement, _ = _core_payload(artifacts, "placement")
    placement_errors, placement_evidence, placement_cells = _placement_gate(
        placement, manifest=manifest, artifacts=artifacts, entities=entities
    )
    gates.append(
        _gate_row(gate_id="placement", blockers=placement_errors, evidence=placement_evidence)
    )

    scorer, _ = _core_payload(artifacts, "scorer")
    scorer_errors, scorer_evidence = _scorer_gate(
        scorer,
        manifest=manifest,
        artifacts=artifacts,
        entities=entities,
        task_evidence=task_evidence,
        placement_evidence=placement_evidence,
    )
    gates.append(_gate_row(gate_id="scorer", blockers=scorer_errors, evidence=scorer_evidence))

    cameras, _ = _core_payload(artifacts, "cameras")
    camera_errors, camera_evidence = _camera_gate(
        cameras,
        manifest=manifest,
        coordinate_frame_id=scene_evidence.get("coordinate_frame_id"),
        artifacts=artifacts,
    )
    gates.append(_gate_row(gate_id="cameras", blockers=camera_errors, evidence=camera_evidence))

    scenario, _ = _core_payload(artifacts, "scenario")
    scenario_errors, scenario_evidence = _scenario_gate(
        scenario,
        manifest=manifest,
        placement_cells=placement_cells,
        placement_evidence=placement_evidence,
        scorer_evidence=scorer_evidence,
        artifacts=artifacts,
    )
    gates.append(
        _gate_row(gate_id="scenario", blockers=scenario_errors, evidence=scenario_evidence)
    )

    runtime, _ = _core_payload(artifacts, "runtime")
    runtime_errors, runtime_evidence, dynamic_native_gates = _runtime_gate(
        runtime, manifest=manifest, artifacts=artifacts
    )
    gates.append(_gate_row(gate_id="runtime", blockers=runtime_errors, evidence=runtime_evidence))

    trust, _ = _core_payload(artifacts, "trust")
    trust_errors, trust_evidence = _trust_gate(trust, artifacts=artifacts)
    gates.append(_gate_row(gate_id="trust", blockers=trust_errors, evidence=trust_evidence))

    blockers = sorted({blocker for gate in gates for blocker in gate["blockers"]})
    prerequisites_passed = not blockers
    unresolved_slots = []
    if prerequisites_passed:
        slot = manifest["asset_slot"]
        unresolved_slots = [
            {
                "slot_id": f"asset-insertion:{slot['entity_id']}",
                "entity_id": slot["entity_id"],
                "semantic_role": slot["semantic_role"],
                "physics_type": slot["physics_type"],
                "status": "typed_unresolved_asset_slot",
                "blocker_code": slot["blocker_code"],
                "required_input": "exact_simready_asset_bytes_configuration_provenance_and_insertion_receipt",
                "dependent_native_gate_ids": dynamic_native_gates,
                "native_qualification_claimed": False,
            }
        ]

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": (
            "preinsertion_ready_one_asset_slot_unresolved"
            if prerequisites_passed
            else "blocked_non_asset_prerequisites"
        ),
        "run_id": manifest["run_id"],
        "scene_id": manifest["scene_id"],
        "task_id": manifest["task_id"],
        "task_kind": manifest["task_kind"],
        "candidate_ids": manifest["candidate_ids"],
        "input_manifest": {
            "path": manifest_path.name,
            "sha256": _sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
            "manifest_digest": manifest["manifest_digest"],
        },
        "artifact_bindings": binding_receipts,
        "prerequisite_gates": gates,
        "all_non_movable_asset_prerequisites_passed": prerequisites_passed,
        "ready_for_asset_insertion": prerequisites_passed,
        "unresolved_slots": unresolved_slots,
        "blockers": blockers,
        "entity_lineage": entity_evidence.get("lineage", []),
        "claim_boundary": {
            "static_path_bound_prerequisites_only": True,
            "native_execution_observed": False,
            "native_asset_qualified": False,
            "native_scene_composition_qualified": False,
            "controls_or_policy_outcomes_observed": False,
            "provider_zero_proved": False,
            "physical_equivalence_or_performance_proved": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


__all__ = [
    "CAMERA_EXTRINSICS_SCHEMA_VERSION",
    "CAMERA_SCHEMA_VERSION",
    "ENGINEERED_ASSET_EVIDENCE_SCHEMA_VERSION",
    "ENTITY_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "PLACEMENT_SCHEMA_VERSION",
    "PREFLIGHT_OBSERVATIONS_SCHEMA_VERSION",
    "REGISTRATION_TRANSFORM_SCHEMA_VERSION",
    "REGISTERED_RECEPTACLE_REPLAY_REQUEST_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "RIGHTS_SCHEMA_VERSION",
    "RIGHTS_EVIDENCE_SCHEMA_VERSION",
    "RIGHTS_AUTHORITY_PUBLIC_KEY_SHA256_ENV",
    "RIGHTS_INTERPRETATION_VERSION",
    "REGISTRATION_EVIDENCE_SCHEMA_VERSION",
    "RUNTIME_SCHEMA_VERSION",
    "SCENARIO_SCHEMA_VERSION",
    "RESOLVED_SCENARIO_CELL_SCHEMA_VERSION",
    "SCENE_SCHEMA_VERSION",
    "SCORER_SCHEMA_VERSION",
    "SOURCE_EVIDENCE_SCHEMA_VERSION",
    "TASK_SCHEMA_VERSION",
    "TASK_FREEZE_AUTHORITY_PUBLIC_KEY_SHA256_ENV",
    "TRUST_SCHEMA_VERSION",
    "TOPOLOGY_EVIDENCE_SCHEMA_VERSION",
    "TOPOLOGY_SURVEY_SCHEMA_VERSION",
    "VISUAL_OBSERVATION_PROVENANCE_SCHEMA_VERSION",
    "TaskPreinsertionReadinessError",
    "collect_task_preinsertion_readiness",
    "prompt_task_spec_freeze_digest",
    "rights_evidence_signature_message",
    "source_observation_signature_message",
    "task_freeze_signature_message",
]
