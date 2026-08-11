"""Package, then independently verify, a deformable native canary.

This module deliberately has two trust boundaries:

``build_deformable_native_canary_bundle`` replays retained local inputs and
creates a portable worker package.  It never creates a paid-resource grant and
never authorizes execution.  Every dynamic gate remains pending.

``verify_deformable_native_canary_return`` accepts returned evidence only after
a configured runner's Ed25519 envelope binds the exact ZIP and immutable run,
package, worker, container, instance, and lifecycle identities.  Worker-owned
provider receipts remain diagnostics: they cannot prove provider-zero or admit
a backend refreeze.  A worker-reported ``success`` value is never consulted.

The contract is entity- and embodiment-neutral within the deformable-transfer
task family.  Entity IDs, the selected robot, action dimensions, supports, and
obstacles come from canonical upstream contracts instead of being hard-coded.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .common import utc_now_iso, write_json
from .composed_paired_entity_placement import (
    RECEIPT_SCHEMA_VERSION as PLACEMENT_RECEIPT_SCHEMA_VERSION,
)
from .composed_paired_entity_placement import plan_composed_paired_entity_placement
from .decision_evidence_contracts import canonical_digest
from .deformable_native_capability_preflight import (
    DYNAMIC_NATIVE_CANARY_GATES,
    FROZEN_CANDIDATES,
    MATRIX_SCHEMA_VERSION as PREFLIGHT_MATRIX_SCHEMA_VERSION,
    build_deformable_native_capability_preflight,
)
from .native_task_arena_scene_plan import SCHEMA_VERSION as SCENE_PLAN_SCHEMA_VERSION
from .native_task_entity_asset_authoring_bundle import (
    BUNDLE_FILENAME as AUTHORING_BUNDLE_FILENAME,
)
from .native_task_entity_asset_authoring_bundle import (
    INPUT_SCHEMA_VERSION as AUTHORING_INPUT_SCHEMA_VERSION,
)
from .native_task_entity_asset_authoring_bundle import (
    RECEIPT_SCHEMA_VERSION as AUTHORING_RECEIPT_SCHEMA_VERSION,
)
from .native_task_entity_asset_authoring_bundle import (
    SOURCE_ROOT_NAME as AUTHORING_SOURCE_ROOT_NAME,
)
from .native_task_entity_asset_authoring_bundle import (
    materialize_native_asset_authoring_runtime_identity,
)
from .native_task_entity_asset_authoring_bundle import (
    verify_native_task_entity_asset_authoring_bundle,
)
from .native_task_entity_contract import (
    TASK_KIND_DEFORMABLE_TRANSFER,
    NativeTaskEntityContractError,
    materialize_native_task_entity_contract,
)
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    require_paid_resource_admission,
    require_paid_resource_admission_grant,
)
from .task_entity_asset_candidate import materialize_task_entity_asset_candidate
from .trusted_execution_envelope import verify_trusted_execution_envelope


EXECUTION_REQUEST_SCHEMA_VERSION = "deformable_native_canary_execution_request.v2"
WORKER_CONTRACT_SCHEMA_VERSION = "deformable_native_canary_worker_contract.v2"
BUNDLE_RECEIPT_SCHEMA_VERSION = "deformable_native_canary_bundle_receipt.v2"
DISCLOSURE_SCHEMA_VERSION = "deformable_native_canary_disclosure_manifest.v1"
RETURN_MANIFEST_SCHEMA_VERSION = "deformable_native_canary_return_manifest.v2"
RETURN_VERIFICATION_SCHEMA_VERSION = "deformable_native_canary_return_verification.v2"
RUN_BINDING_SCHEMA_VERSION = "deformable_native_canary_run_binding.v1"

BUNDLE_FILENAME = "deformable_native_canary_bundle.v2.zip"
RECEIPT_FILENAME = "deformable_native_canary_bundle_receipt.v2.json"
WORKER_CONTRACT_FILENAME = "deformable_native_canary_worker_contract.v2.json"
DISCLOSURE_FILENAME = "deformable_native_canary_disclosure_manifest.v1.json"
RETURN_MANIFEST_FILENAME = "native_return/return_manifest.v2.json"
INPUT_ROOT_NAME = "deformable_native_canary_input"

CANARY_STAGE_IDS = ("blank_stage_asset_runtime", "scene_bound_native_execution")
REQUIRED_CAMERA_ROLES = ("external", "wrist", "overview")
ADDITIONAL_DYNAMIC_NATIVE_GATES = (
    "dynamic_rigid_receptacle_support_and_pose_stability",
    "dynamic_full_phase_ik_reachability",
    "dynamic_policy_adapter_runtime_smoke",
)
DYNAMIC_NATIVE_GATE_IDS = tuple(
    [row[0] for row in DYNAMIC_NATIVE_CANARY_GATES] + list(ADDITIONAL_DYNAMIC_NATIVE_GATES)
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_RETURN_ROLE_RE = re.compile(
    r"^(blank_stage_report|scene_stage_report|provider_admission|prelaunch_inventory|"
    r"allocation_receipt|watchdog_receipt|billing_receipt|billing_response|upload_receipt|"
    r"scene_disclosure_receipt|prelaunch_inventory_response|provider_zero_inventory_response|"
    r"teardown_receipt|provider_zero_inventory|camera_manifest:(external|wrist|overview)|"
    r"camera_frame:(external|wrist|overview):[0-9]{6}|"
    r"camera_video:(external|wrist|overview)|camera_ffprobe:(external|wrist|overview))$"
)
_JSON_LIMIT = 4 * 1024 * 1024
_PACKAGE_LIMIT = 256 * 1024 * 1024
_RETURN_LIMIT = 512 * 1024 * 1024
_ARCHIVE_MEMBER_LIMIT = 256
_SECRET_MARKERS = (
    b"-----begin private key-----",
    b"-----begin openssh private key-----",
    b"authorization: bearer ",
    b"x-api-key:",
)
_GEOMETRY_ROLES = frozenset({"rest_geometry", "visual_geometry", "collision_geometry"})
_JSON_ASSET_ROLES = frozenset({"material_definition", "physics_configuration"})
_ASSET_ROLE_SET = frozenset({*_GEOMETRY_ROLES, *_JSON_ASSET_ROLES, "texture", "runtime_usd"})
_LIFECYCLE_ARTIFACT_ROLES = frozenset(
    {
        "provider_admission",
        "prelaunch_inventory",
        "prelaunch_inventory_response",
        "allocation_receipt",
        "watchdog_receipt",
        "billing_receipt",
        "billing_response",
        "upload_receipt",
        "teardown_receipt",
        "provider_zero_inventory",
        "provider_zero_inventory_response",
    }
)
_PROVIDER_PROOF_BLOCKER = "verifier_owned_provider_lifecycle_and_provider_zero_proof_missing"


class DeformableNativeCanaryBundleError(ValueError):
    """Stable, sorted failures at either canary trust boundary."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_file_once_no_follow(
    path_value: str | Path, *, label: str, maximum_size: int
) -> tuple[bytes, Path]:
    """Read one regular file descriptor once and reject identity drift."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_{label}_no_follow_unavailable"]
        )
    path = Path(os.path.abspath(Path(path_value).expanduser()))
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > maximum_size:
            raise DeformableNativeCanaryBundleError(
                [f"deformable_native_canary_{label}_file_invalid"]
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
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
        if identity_before != identity_after or len(content) != before.st_size:
            raise DeformableNativeCanaryBundleError(
                [f"deformable_native_canary_{label}_changed_while_reading"]
            )
        return content, path
    except DeformableNativeCanaryBundleError:
        raise
    except OSError as exc:
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_{label}_file_invalid"]
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _digest_without(value: Mapping[str, Any], *fields: str) -> str:
    projected = dict(value)
    for field in fields:
        projected.pop(field, None)
    return canonical_digest(projected)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _strings(value: Any) -> list[str]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _valid_digest(value: Any) -> bool:
    return bool(_DIGEST_RE.fullmatch(str(value or "")))


def _same_vector(first: Any, second: Any, *, length: int) -> bool:
    if (
        isinstance(first, (str, bytes, Mapping))
        or isinstance(second, (str, bytes, Mapping))
        or not isinstance(first, Sequence)
        or not isinstance(second, Sequence)
        or len(first) != length
        or len(second) != length
    ):
        return False
    try:
        return all(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
            for left, right in zip(first, second, strict=True)
        )
    except (TypeError, ValueError):
        return False


def _strict_json_bytes(value: bytes, *, error: str) -> dict[str, Any]:
    if not value or len(value) > _JSON_LIMIT or value.startswith((b"\xef\xbb\xbf", b"\xff\xfe")):
        raise DeformableNativeCanaryBundleError([error])

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = item
        return result

    try:
        decoded = value.decode("utf-8")
        result = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeformableNativeCanaryBundleError([error]) from exc
    if not isinstance(result, dict):
        raise DeformableNativeCanaryBundleError([error])
    return result


def _read_json(path_value: str | Path, *, label: str) -> tuple[dict[str, Any], bytes, Path]:
    path = Path(path_value).expanduser()
    if path.is_symlink():
        raise DeformableNativeCanaryBundleError([f"deformable_native_canary_{label}_path_invalid"])
    path = path.resolve()
    try:
        if not path.is_file() or path.stat().st_size > _JSON_LIMIT:
            raise OSError
        content = path.read_bytes()
    except OSError as exc:
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_{label}_file_invalid"]
        ) from exc
    return (
        _strict_json_bytes(content, error=f"deformable_native_canary_{label}_json_invalid"),
        content,
        path,
    )


def _safe_archive_infos(
    archive: zipfile.ZipFile, *, prefix: str, maximum_size: int
) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    errors: list[str] = []
    names = [info.filename for info in infos]
    if not infos or len(infos) > _ARCHIVE_MEMBER_LIMIT or len(names) != len(set(names)):
        errors.append(f"{prefix}_archive_inventory_invalid")
    total = 0
    for info in infos:
        name = PurePosixPath(info.filename)
        mode = (info.external_attr >> 16) & 0o170000
        total += info.file_size
        if (
            name.is_absolute()
            or not name.parts
            or len(info.filename) > 240
            or ".." in name.parts
            or "\\" in info.filename
            or info.is_dir()
            or mode == 0o120000
            or info.file_size < 0
            or info.compress_size < 0
            or (info.compress_size and info.file_size > info.compress_size * 100)
        ):
            errors.append(f"{prefix}_archive_member_invalid")
    if total > maximum_size:
        errors.append(f"{prefix}_archive_size_invalid")
    if errors:
        raise DeformableNativeCanaryBundleError(errors)
    return infos


def _valid_png(content: bytes) -> tuple[int, int] | None:
    if len(content) < 45 or content[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    offset = 8
    dimensions: tuple[int, int] | None = None
    seen_iend = False
    while offset + 12 <= len(content):
        length = struct.unpack(">I", content[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(content):
            return None
        kind = content[offset + 4 : offset + 8]
        payload = content[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", content[offset + 8 + length : end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            return None
        if offset == 8:
            if kind != b"IHDR" or length != 13:
                return None
            width, height = struct.unpack(">II", payload[:8])
            if width <= 0 or height <= 0:
                return None
            dimensions = (width, height)
        if kind == b"IEND":
            if length != 0 or end != len(content):
                return None
            seen_iend = True
            break
        offset = end
    return dimensions if seen_iend else None


def _valid_mp4_h264(content: bytes) -> bool:
    offset = 0
    boxes: set[bytes] = set()
    while offset + 8 <= len(content):
        size = struct.unpack(">I", content[offset : offset + 4])[0]
        kind = content[offset + 4 : offset + 8]
        if size == 1:
            if offset + 16 > len(content):
                return False
            size = struct.unpack(">Q", content[offset + 8 : offset + 16])[0]
        if size < 8 or offset + size > len(content):
            return False
        boxes.add(kind)
        offset += size
    return bool(
        offset == len(content)
        and {b"ftyp", b"moov", b"mdat"}.issubset(boxes)
        and (b"avc1" in content or b"avc3" in content)
    )


def _decoded_png_rgb(content: bytes) -> dict[str, Any] | None:
    """Fully decode one PNG and bind its exact RGB pixels.

    The structural parser above is useful for cheap asset checks, but it cannot
    establish that IDAT data is present or decodable.  Camera evidence crosses
    a stronger boundary: verifier-owned Pillow must decode the exact archive
    bytes before they can participate in a frame/video derivation join.
    """

    if _valid_png(content) is None:
        return None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG":
                return None
            width, height = (int(value) for value in image.size)
            if width < 1 or height < 1 or width * height > 16_777_216:
                return None
            rgb = image.convert("RGB")
            rgb.load()
            payload = rgb.tobytes()
    except (OSError, ValueError):
        return None
    if len(payload) != width * height * 3:
        return None
    return {
        "width": width,
        "height": height,
        "raw_rgb_sha256": _sha256_bytes(payload),
    }


def _decoded_h264_rgb(content: bytes) -> dict[str, Any] | None:
    """Decode verifier-owned H.264 bytes and bind every RGB sample.

    A worker-authored ffprobe JSON document is retained as diagnostic metadata,
    not as proof that the video is decodable or derived from the lossless frame
    manifest.  OpenCV's FFmpeg backend reads an immutable temporary snapshot of
    the exact returned bytes; the caller cannot inject an alternate decoder or
    path through the public verification API.
    """

    if not _valid_mp4_h264(content):
        return None
    try:
        import cv2
    except ImportError:
        return None
    decoded: dict[str, Any] | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="blueprint-canary-h264-") as temporary:
            snapshot = Path(temporary) / "review.mp4"
            snapshot.write_bytes(content)
            capture = cv2.VideoCapture(str(snapshot), cv2.CAP_FFMPEG)
            try:
                if not capture.isOpened():
                    return None
                fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
                fourcc = "".join(
                    chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
                ).lower()
                if fourcc not in {"avc1", "avc3", "h264"}:
                    return None
                dimensions: tuple[int, int] | None = None
                raw_rgb_sha256_by_sample: list[str] = []
                while len(raw_rgb_sha256_by_sample) <= 10_000:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                        return None
                    current = (int(frame.shape[1]), int(frame.shape[0]))
                    if min(current) < 1 or current[0] * current[1] > 16_777_216:
                        return None
                    if dimensions is None:
                        dimensions = current
                    elif dimensions != current:
                        return None
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_rgb_sha256_by_sample.append(_sha256_bytes(rgb.tobytes()))
                if dimensions is not None and raw_rgb_sha256_by_sample:
                    decoded = {
                        "width": dimensions[0],
                        "height": dimensions[1],
                        "sample_count": len(raw_rgb_sha256_by_sample),
                        "raw_rgb_sha256_by_sample": raw_rgb_sha256_by_sample,
                    }
            finally:
                capture.release()
    except (cv2.error, OSError, ValueError):
        return None
    return decoded


def _valid_usd(content: bytes) -> bool:
    return bool(
        (content.startswith(b"#usda ") or content.startswith(b"PXR-USDC"))
        and not _contains_forbidden_raw_dataset_payload(content)
    )


def _contains_forbidden_raw_dataset_payload(content: bytes) -> bool:
    """Detect raw PLY/Gaussian payloads even when renamed or USD-prefixed."""

    lowered = content.lower()
    normalized = lowered.replace(b"\r\n", b"\n")
    ply_header = (
        (normalized.startswith(b"ply\n") or b"\nply\n" in normalized)
        and b"format " in normalized
        and b"end_header" in normalized
    )
    gaussian_properties = b"property float opacity" in normalized and (
        b"property float scale_" in normalized
        or b"property float f_dc_" in normalized
        or b"property float rot_" in normalized
    )
    return ply_header or gaussian_properties


def _valid_geometry(content: bytes, suffix: str) -> bool:
    suffix = suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return _valid_usd(content)
    if suffix == ".obj":
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return any(line.startswith("v ") for line in text.splitlines()) and any(
            line.startswith("f ") for line in text.splitlines()
        )
    if suffix == ".stl":
        if len(content) >= 84:
            triangles = struct.unpack("<I", content[80:84])[0]
            if len(content) == 84 + 50 * triangles and triangles > 0:
                return True
        stripped = content.lstrip()
        return (
            stripped.startswith(b"solid ") and b"facet normal" in content and b"endsolid" in content
        )
    return False


def _asset_content_valid(role: str, name: str, content: bytes) -> bool:
    lowered = content.lower()
    if _contains_forbidden_raw_dataset_payload(content) or any(
        marker in lowered for marker in _SECRET_MARKERS
    ):
        return False
    suffix = PurePosixPath(name).suffix.lower()
    if role == "runtime_usd":
        return suffix in {".usd", ".usda", ".usdc"} and _valid_usd(content)
    if role in _GEOMETRY_ROLES:
        return _valid_geometry(content, suffix)
    if role in _JSON_ASSET_ROLES:
        if suffix != ".json":
            return False
        try:
            parsed = _strict_json_bytes(
                content, error="deformable_native_canary_asset_json_invalid"
            )
        except DeformableNativeCanaryBundleError:
            return False
        allowed = (
            {
                "schema_version",
                "material",
                "materials",
                "name",
                "base_color",
                "roughness",
                "metallic",
                "texture",
            }
            if role == "material_definition"
            else {
                "schema_version",
                "representation",
                "material",
                "solver",
                "collision",
                "reset",
                "receptacle",
                "anchoring",
            }
        )
        return len(parsed) <= 64 and set(parsed).issubset(allowed)
    if role == "texture":
        return suffix == ".png" and _valid_png(content) is not None
    return False


def _validate_task_entities(scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if scene_plan.get("schema_version") != SCENE_PLAN_SCHEMA_VERSION:
        errors.append("deformable_native_canary_scene_plan_schema_invalid")
    if scene_plan.get("plan_digest") != canonical_digest(scene_plan, digest_field="plan_digest"):
        errors.append("deformable_native_canary_scene_plan_digest_invalid")
    if scene_plan.get("task_kind") != TASK_KIND_DEFORMABLE_TRANSFER:
        errors.append("deformable_native_canary_task_kind_unsupported")
    try:
        contract = materialize_native_task_entity_contract(
            task_kind=str(scene_plan.get("task_kind") or ""),
            task_entities=scene_plan.get("task_entities") or [],
        )
    except NativeTaskEntityContractError as exc:
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_task_entities_invalid:{item}" for item in exc.errors]
        ) from exc
    if contract["task_entities"] != scene_plan.get("task_entities"):
        errors.append("deformable_native_canary_task_entities_not_normalized")
    if contract["contract_digest"] != scene_plan.get("task_entity_contract_digest"):
        errors.append("deformable_native_canary_task_entity_digest_mismatch")
    if contract["semantic_role_index"] != scene_plan.get("task_entity_role_index"):
        errors.append("deformable_native_canary_task_entity_role_index_mismatch")
    for entity in contract["task_entities"]:
        for value in (
            _mapping(entity.get("source_observation")).get("source_reference"),
            _mapping(entity.get("runtime_asset")).get("source_reference"),
            _mapping(entity.get("provenance")).get("source_path"),
        ):
            path = PurePosixPath(str(value or ""))
            if path.is_absolute() or ".." in path.parts or "\\" in str(value or ""):
                errors.append(
                    f"deformable_native_canary_task_entity_reference_not_portable:{entity['entity_id']}"
                )

    task_spec = _mapping(scene_plan.get("task_spec"))
    joins = (
        ("deformable_entity_id", "movable_deformable"),
        ("destination_entity_id", "destination_receptacle"),
        ("robot_entity_id", "robot"),
    )
    for field, role in joins:
        if task_spec.get(field) not in contract["semantic_role_index"].get(role, []):
            errors.append(f"deformable_native_canary_task_entity_binding_invalid:{field}")
    for field in (
        "settle_window_samples",
        "maximum_node_speed_mps",
        "maximum_principal_strain",
        "minimum_robot_clearance_m",
        "maximum_receptacle_translation_drift_m",
        "maximum_receptacle_rotation_drift_rad",
    ):
        try:
            number = float(task_spec.get(field))
        except (TypeError, ValueError):
            number = math.nan
        if not math.isfinite(number) or number <= 0:
            errors.append(f"deformable_native_canary_task_threshold_invalid:{field}")
    if isinstance(task_spec.get("settle_window_samples"), bool) or not isinstance(
        task_spec.get("settle_window_samples"), int
    ):
        errors.append("deformable_native_canary_task_threshold_invalid:settle_window_samples")

    robot = _mapping(scene_plan.get("robot"))
    action = _mapping(robot.get("action_seam"))
    if (
        not str(robot.get("robot_id") or "").strip()
        or not str(action.get("kind") or "").strip()
        or isinstance(action.get("action_dimension"), bool)
        or not isinstance(action.get("action_dimension"), int)
        or action.get("action_dimension", 0) <= 0
    ):
        errors.append("deformable_native_canary_action_seam_invalid")

    cameras = _rows(scene_plan.get("cameras"))
    by_role = {str(row.get("role") or ""): row for row in cameras}
    if len(by_role) != len(cameras) or not set(REQUIRED_CAMERA_ROLES).issubset(by_role):
        errors.append("deformable_native_canary_camera_roles_invalid")
    for role in REQUIRED_CAMERA_ROLES:
        row = _mapping(by_role.get(role))
        intrinsics = _mapping(row.get("intrinsics"))
        if (
            row.get("policy_input") is not (role in {"external", "wrist"})
            or row.get("scoring_input") is not False
            or row.get("review_only") is not (role == "overview")
            or not str(row.get("pose_frame") or "").strip()
            or any(
                isinstance(intrinsics.get(field), bool)
                or not isinstance(intrinsics.get(field), (int, float))
                or not math.isfinite(float(intrinsics[field]))
                or float(intrinsics[field]) <= 0
                for field in ("fx", "fy", "width", "height")
            )
        ):
            errors.append(f"deformable_native_canary_camera_role_invalid:{role}")
    scenario = _mapping(scene_plan.get("scenario"))
    if (
        not str(scenario.get("cell_id") or "").strip()
        or not _valid_digest(scenario.get("instance_digest"))
        or isinstance(scenario.get("seed"), bool)
        or not isinstance(scenario.get("seed"), int)
    ):
        errors.append("deformable_native_canary_scenario_invalid")
    if errors:
        raise DeformableNativeCanaryBundleError(errors)
    return contract


def _replay_placement(receipt: Mapping[str, Any], scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema_version") != PLACEMENT_RECEIPT_SCHEMA_VERSION:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_placement_schema_invalid"]
        )
    request = _mapping(receipt.get("request"))
    try:
        replayed = plan_composed_paired_entity_placement(
            support_regions=request.get("support_regions") or [],
            obstacle_aabbs=request.get("obstacle_aabbs") or [],
            entity_specs=request.get("entity_specs") or [],
            canonical_task_centers_m=request.get("canonical_task_centers_m") or [],
            robot_spec=_mapping(request.get("robot_spec")),
            minimum_separations_m=_mapping(request.get("minimum_separations_m")),
            grid_spacing_m=request.get("grid_spacing_m"),
            frozen_seed=request.get("frozen_seed_uint64"),
            maximum_combination_count=request.get("maximum_combination_count"),
        )
    except Exception as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_placement_replay_invalid"]
        ) from exc
    if replayed != receipt or replayed.get("status") != "geometry_plausibility_candidate_selected":
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_placement_replay_mismatch"]
        )
    scenario = _mapping(scene_plan.get("scenario"))
    if scenario.get("instance_digest") != replayed.get("receipt_digest") or scenario.get(
        "seed"
    ) != request.get("frozen_seed_uint64"):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_placement_scenario_join_mismatch"]
        )
    selected = {
        str(row.get("subject_id") or ""): row
        for row in _rows(_mapping(replayed.get("selection")).get("entity_placements"))
    }
    task_spec = _mapping(scene_plan.get("task_spec"))
    expected = {task_spec.get("deformable_entity_id"), task_spec.get("destination_entity_id")}
    if set(selected) != expected:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_placement_entity_join_mismatch"]
        )
    entities = {
        str(row.get("entity_id") or ""): row for row in _rows(scene_plan.get("task_entities"))
    }
    for entity_id, selected_row in selected.items():
        position = _mapping(
            _mapping(entities[entity_id].get("initial_state")).get("pose_world")
        ).get("position_world_m")
        if not _same_vector(position, selected_row.get("center_world_m"), length=3):
            raise DeformableNativeCanaryBundleError(
                [f"deformable_native_canary_placement_pose_join_mismatch:{entity_id}"]
            )
    robot_base = _mapping(_mapping(replayed.get("selection")).get("robot_base_placement"))
    robot_position = _mapping(_mapping(scene_plan.get("robot")).get("base_pose_world")).get(
        "position_world_m"
    )
    if not _same_vector(robot_position, robot_base.get("aabb_min_m"), length=3):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_placement_robot_pose_join_mismatch"]
        )
    return replayed


def _replay_preflight(
    request: Mapping[str, Any], observations: Mapping[str, Any], matrix: Mapping[str, Any]
) -> dict[str, Any]:
    replayed = build_deformable_native_capability_preflight(
        request=request, observations=observations
    )
    if replayed != matrix:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_preflight_full_replay_mismatch"]
        )
    if (
        replayed.get("schema_version") != PREFLIGHT_MATRIX_SCHEMA_VERSION
        or replayed.get("status") != "static_preflight_passed_native_canary_required"
        or replayed.get("static_checks_passed") is not True
        or replayed.get("blockers") != []
        or replayed.get("native_canary_completed") is not False
        or replayed.get("scene_run_admitted") is not False
    ):
        raise DeformableNativeCanaryBundleError(["deformable_native_canary_preflight_not_passing"])
    observed_gates = tuple(
        str(row.get("check_id") or "") for row in _rows(replayed.get("dynamic_native_canary_gates"))
    )
    expected_gates = tuple(row[0] for row in DYNAMIC_NATIVE_CANARY_GATES)
    if observed_gates != expected_gates or any(
        row.get("status") != "pending_native_canary"
        for row in _rows(replayed.get("dynamic_native_canary_gates"))
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_preflight_dynamic_gate_inventory_invalid"]
        )
    return replayed


def _authoring_bundle(
    receipt_path: Path, *, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], bytes, list[dict[str, Any]]]:
    try:
        receipt = verify_native_task_entity_asset_authoring_bundle(
            receipt_path,
            expected_task_entity_contract_digest=str(contract["contract_digest"]),
        )
    except Exception as exc:
        errors = getattr(exc, "errors", ("authoring_bundle_invalid",))
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_authoring_invalid:{item}" for item in errors]
        ) from exc
    raw_bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser()
    if raw_bundle_path.is_symlink():
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_bundle_path_invalid"]
        )
    bundle_path = raw_bundle_path.resolve()
    if bundle_path.stat().st_size > _PACKAGE_LIMIT:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_bundle_path_invalid"]
        )
    bundle_bytes = bundle_path.read_bytes()
    manifest_name = PurePosixPath(
        AUTHORING_SOURCE_ROOT_NAME, "native_task_entity_asset_authoring_input.v1.json"
    ).as_posix()
    disclosure_rows: list[dict[str, Any]] = []
    forbidden_digests = {
        str(row["source_observation"]["source_sha256"]) for row in contract["task_entities"]
    }
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            infos = _safe_archive_infos(
                archive,
                prefix="deformable_native_canary_authoring",
                maximum_size=_PACKAGE_LIMIT,
            )
            names = {info.filename for info in infos}
            if manifest_name not in names:
                raise DeformableNativeCanaryBundleError(
                    ["deformable_native_canary_authoring_manifest_missing"]
                )
            manifest_bytes = archive.read(manifest_name)
            manifest = _strict_json_bytes(
                manifest_bytes, error="deformable_native_canary_authoring_manifest_invalid"
            )
            staged: dict[str, Mapping[str, Any]] = {}
            for plan in _rows(manifest.get("entity_authoring_plans")):
                for file_row in _rows(plan.get("staged_files")):
                    role = str(file_row.get("role") or "")
                    name = PurePosixPath(
                        AUTHORING_SOURCE_ROOT_NAME,
                        str(file_row.get("archive_relative_path") or ""),
                    ).as_posix()
                    if role not in _ASSET_ROLE_SET or name in staged:
                        raise DeformableNativeCanaryBundleError(
                            ["deformable_native_canary_authoring_file_inventory_invalid"]
                        )
                    staged[name] = file_row
            if names != {manifest_name, *staged}:
                raise DeformableNativeCanaryBundleError(
                    ["deformable_native_canary_authoring_file_inventory_invalid"]
                )
            for name, row in sorted(staged.items()):
                content = archive.read(name)
                digest = _sha256_bytes(content)
                role = str(row["role"])
                if (
                    len(content) != row.get("size_bytes")
                    or digest != row.get("sha256")
                    or digest in forbidden_digests
                    or not _asset_content_valid(role, name, content)
                ):
                    raise DeformableNativeCanaryBundleError(
                        [f"deformable_native_canary_authoring_file_invalid:{role}"]
                    )
                disclosure_rows.append(
                    {
                        "nested_path": name,
                        "role": role,
                        "size_bytes": len(content),
                        "sha256": digest,
                        "classification": "derived_runtime_asset",
                    }
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_archive_invalid"]
        ) from exc
    if (
        manifest.get("schema_version") != AUTHORING_INPUT_SCHEMA_VERSION
        or manifest.get("input_digest") != canonical_digest(manifest, digest_field="input_digest")
        or receipt.get("schema_version") != AUTHORING_RECEIPT_SCHEMA_VERSION
        or receipt.get("input_digest") != manifest.get("input_digest")
        or manifest.get("task_entity_contract_digest") != contract["contract_digest"]
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_manifest_join_invalid"]
        )
    identity = _mapping(manifest.get("runtime_identity"))
    identity_raw = dict(identity)
    identity_raw.pop("runtime_identity_digest", None)
    try:
        normalized_identity = materialize_native_asset_authoring_runtime_identity(identity_raw)
    except Exception as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_runtime_identity_invalid"]
        ) from exc
    if normalized_identity != identity:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_runtime_identity_not_normalized"]
        )
    plans = _rows(manifest.get("entity_authoring_plans"))
    expected_ids = {
        row["entity_id"]
        for row in contract["task_entities"]
        if row["semantic_role"] in {"movable_deformable", "destination_receptacle"}
    }
    if {str(row.get("entity_id") or "") for row in plans} != expected_ids:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_entity_join_invalid"]
        )
    for plan in plans:
        candidate = _mapping(plan.get("candidate_record"))
        candidate_raw = dict(candidate)
        for field in (
            "status",
            "claims",
            "pending_gates",
            "physically_unresolved",
            "candidate_digest",
        ):
            candidate_raw.pop(field, None)
        try:
            normalized_candidate = materialize_task_entity_asset_candidate(candidate_raw)
        except Exception as exc:
            raise DeformableNativeCanaryBundleError(
                ["deformable_native_canary_authoring_candidate_invalid"]
            ) from exc
        staged_identity = sorted(
            (row.get("role"), row.get("size_bytes"), row.get("sha256"))
            for row in _rows(plan.get("staged_files"))
        )
        candidate_identity = sorted(
            (row.get("role"), row.get("size_bytes"), row.get("sha256"))
            for row in _rows(candidate.get("files"))
        )
        if (
            normalized_candidate != candidate
            or plan.get("candidate_digest") != candidate.get("candidate_digest")
            or plan.get("entity_id") != candidate.get("entity_id")
            or staged_identity != candidate_identity
        ):
            raise DeformableNativeCanaryBundleError(
                ["deformable_native_canary_authoring_candidate_join_invalid"]
            )
    return dict(receipt), manifest, bundle_bytes, disclosure_rows


def materialize_deformable_native_canary_execution_request(
    value: Mapping[str, Any], *, authoring_receipt_digest: str
) -> dict[str, Any]:
    """Normalize immutable limits; this function does not issue a paid grant."""

    try:
        raw = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_execution_request_invalid"]
        ) from exc
    if not isinstance(raw, dict):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_execution_request_invalid"]
        )
    expected_fields = {
        "schema_version",
        "run_id",
        "provider_id",
        "authority_binding",
        "scene_runtime_reference",
        "resource_limits",
        "visibility_thresholds",
        "authoring_receipt_digest",
        "execution_request_digest",
    }
    errors: list[str] = []
    if set(raw) - expected_fields:
        errors.append("deformable_native_canary_execution_request_fields_unexpected")
    if raw.get("schema_version") != EXECUTION_REQUEST_SCHEMA_VERSION:
        errors.append("deformable_native_canary_execution_request_schema_invalid")
    run_id = str(raw.get("run_id") or "")
    if not _IDENTIFIER_RE.fullmatch(run_id):
        errors.append("deformable_native_canary_execution_run_id_invalid")
    if raw.get("provider_id") != "vast":
        errors.append("deformable_native_canary_execution_provider_invalid")
    if raw.get("authoring_receipt_digest") != authoring_receipt_digest:
        errors.append("deformable_native_canary_execution_authoring_join_mismatch")

    authority = _mapping(raw.get("authority_binding"))
    if (
        set(authority) != {"authority_receipt_digest", "authority_reference"}
        or not _valid_digest(authority.get("authority_receipt_digest"))
        or not _IDENTIFIER_RE.fullmatch(str(authority.get("authority_reference") or ""))
    ):
        errors.append("deformable_native_canary_execution_authority_binding_invalid")
    scene = _mapping(raw.get("scene_runtime_reference"))
    if (
        set(scene) != {"package_id", "package_sha256", "disclosure_receipt_digest"}
        or not _IDENTIFIER_RE.fullmatch(str(scene.get("package_id") or ""))
        or not _valid_digest(scene.get("package_sha256"))
        or not _valid_digest(scene.get("disclosure_receipt_digest"))
    ):
        errors.append("deformable_native_canary_scene_runtime_reference_invalid")

    limits = _mapping(raw.get("resource_limits"))
    try:
        attempt_cap = float(limits.get("attempt_spend_cap_usd"))
        goal_cap = float(limits.get("goal_total_spend_cap_usd"))
        spent_before = float(limits.get("goal_spend_before_attempt_usd"))
        ttl = int(limits.get("ttl_seconds"))
        watchdog = int(limits.get("watchdog_seconds"))
    except (TypeError, ValueError):
        attempt_cap = goal_cap = spent_before = math.nan
        ttl = watchdog = -1
    allowed_ids = limits.get("allowed_active_instance_ids")
    expected_limit_fields = {
        "maximum_paid_attempts",
        "automatic_retry_count",
        "attempt_spend_cap_usd",
        "goal_total_spend_cap_usd",
        "goal_spend_before_attempt_usd",
        "ttl_seconds",
        "watchdog_seconds",
        "allowed_active_instance_ids",
    }
    if (
        set(limits) != expected_limit_fields
        or limits.get("maximum_paid_attempts") != 1
        or limits.get("automatic_retry_count") != 0
        or any(
            isinstance(limits.get(field), bool) or not isinstance(limits.get(field), (int, float))
            for field in (
                "attempt_spend_cap_usd",
                "goal_total_spend_cap_usd",
                "goal_spend_before_attempt_usd",
            )
        )
        or isinstance(limits.get("ttl_seconds"), bool)
        or not isinstance(limits.get("ttl_seconds"), int)
        or isinstance(limits.get("watchdog_seconds"), bool)
        or not isinstance(limits.get("watchdog_seconds"), int)
        or not all(math.isfinite(number) for number in (attempt_cap, goal_cap, spent_before))
        or attempt_cap <= 0
        or goal_cap <= 0
        or spent_before < 0
        or spent_before + attempt_cap > goal_cap + 1e-9
        or not 60 <= ttl <= 14_400
        or not 10 <= watchdog < ttl
        or not isinstance(allowed_ids, list)
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in allowed_ids
        )
        or len(allowed_ids) != len(set(allowed_ids))
    ):
        errors.append("deformable_native_canary_execution_resource_limits_invalid")

    thresholds = _mapping(raw.get("visibility_thresholds"))
    required_thresholds = {
        "external": ("minimum_deformable_pixel_fraction", "minimum_destination_pixel_fraction"),
        "wrist": ("minimum_deformable_pixel_fraction", "minimum_destination_pixel_fraction"),
        "overview": ("minimum_motion_enclosure_fraction",),
    }
    if set(thresholds) != set(required_thresholds):
        errors.append("deformable_native_canary_visibility_threshold_roles_invalid")
    for role, names in required_thresholds.items():
        values = _mapping(thresholds.get(role))
        if set(values) != set(names):
            errors.append(f"deformable_native_canary_visibility_threshold_fields_invalid:{role}")
        for name in names:
            try:
                number = float(values.get(name))
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number) or not 0 < number <= 1:
                errors.append(
                    f"deformable_native_canary_visibility_threshold_invalid:{role}:{name}"
                )
    if errors:
        raise DeformableNativeCanaryBundleError(errors)
    normalized = {
        "schema_version": EXECUTION_REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "provider_id": "vast",
        "authority_binding": dict(authority),
        "scene_runtime_reference": dict(scene),
        "resource_limits": {
            "maximum_paid_attempts": 1,
            "automatic_retry_count": 0,
            "attempt_spend_cap_usd": attempt_cap,
            "goal_total_spend_cap_usd": goal_cap,
            "goal_spend_before_attempt_usd": spent_before,
            "ttl_seconds": ttl,
            "watchdog_seconds": watchdog,
            "allowed_active_instance_ids": sorted(allowed_ids),
        },
        "visibility_thresholds": json.loads(json.dumps(thresholds, sort_keys=True)),
        "authoring_receipt_digest": authoring_receipt_digest,
        "execution_request_digest": "",
    }
    normalized["execution_request_digest"] = canonical_digest(
        normalized, digest_field="execution_request_digest"
    )
    supplied_digest = raw.get("execution_request_digest")
    if supplied_digest not in (None, "", normalized["execution_request_digest"]):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_execution_request_digest_invalid"]
        )
    return normalized


def _validate_scene_runtime_disclosure(
    *,
    package_path_value: str | Path,
    disclosure_receipt_path: str | Path,
    expected_reference: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, bytes]:
    receipt, receipt_bytes, _ = _read_json(
        disclosure_receipt_path, label="scene_runtime_disclosure"
    )
    package_bytes, _package_path = _read_file_once_no_follow(
        package_path_value,
        label="scene_runtime_package",
        maximum_size=_PACKAGE_LIMIT,
    )
    package_sha256 = _sha256_bytes(package_bytes)
    members = _rows(receipt.get("members"))
    member_paths = [str(row.get("path") or "") for row in members]
    raw_source_digests = receipt.get("raw_source_digests")
    errors: list[str] = []
    if (
        set(receipt)
        != {
            "schema_version",
            "package_sha256",
            "provider_terms_receipt_digest",
            "output_rights_receipt_digest",
            "provider_training_permitted",
            "retention_mode",
            "raw_source_digests",
            "members",
            "receipt_digest",
        }
        or receipt.get("schema_version") != "deformable_native_canary_scene_disclosure.v1"
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or receipt.get("package_sha256") != package_sha256
        or receipt.get("package_sha256") != expected_reference.get("package_sha256")
        or receipt.get("receipt_digest") != expected_reference.get("disclosure_receipt_digest")
        or not _valid_digest(receipt.get("provider_terms_receipt_digest"))
        or not _valid_digest(receipt.get("output_rights_receipt_digest"))
        or receipt.get("provider_training_permitted") is not False
        or receipt.get("retention_mode") != "bounded_ephemeral_private_processing"
        or not isinstance(raw_source_digests, list)
        or not raw_source_digests
        or len(raw_source_digests) != len(set(raw_source_digests))
        or any(not _valid_digest(value) for value in raw_source_digests or [])
        or not members
        or len(member_paths) != len(set(member_paths))
    ):
        errors.append("deformable_native_canary_scene_runtime_disclosure_invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(package_bytes)) as archive:
            infos = _safe_archive_infos(
                archive,
                prefix="deformable_native_canary_scene_runtime",
                maximum_size=_PACKAGE_LIMIT,
            )
            names = {info.filename for info in infos}
            if names != set(member_paths):
                errors.append("deformable_native_canary_scene_runtime_member_set_invalid")
            for row in members:
                path = str(row.get("path") or "")
                kind = row.get("content_kind")
                content = archive.read(path) if path in names else b""
                digest = _sha256_bytes(content)
                pure = PurePosixPath(path)
                if (
                    set(row)
                    != {
                        "path",
                        "content_kind",
                        "size_bytes",
                        "sha256",
                        "derivation_receipt_digest",
                    }
                    or kind
                    not in {
                        "derived_scene_usd",
                        "derived_collision_usd",
                        "runtime_configuration",
                    }
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or pure.suffix.lower() in {".ply", ".labels", ".structure"}
                    or len(content) != row.get("size_bytes")
                    or digest != row.get("sha256")
                    or digest in set(raw_source_digests or [])
                    or _contains_forbidden_raw_dataset_payload(content)
                    or any(marker in content.lower() for marker in _SECRET_MARKERS)
                    or not _valid_digest(row.get("derivation_receipt_digest"))
                    or (
                        kind in {"derived_scene_usd", "derived_collision_usd"}
                        and (
                            pure.suffix.lower() not in {".usd", ".usda", ".usdc"}
                            or not _valid_usd(content)
                        )
                    )
                    or (kind == "runtime_configuration" and pure.suffix.lower() != ".json")
                ):
                    errors.append(f"deformable_native_canary_scene_runtime_member_invalid:{path}")
                if kind == "runtime_configuration":
                    try:
                        _strict_json_bytes(
                            content,
                            error="deformable_native_canary_scene_runtime_configuration_invalid",
                        )
                    except DeformableNativeCanaryBundleError as exc:
                        errors.extend(exc.errors)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_scene_runtime_archive_invalid"]
        ) from exc
    if errors:
        raise DeformableNativeCanaryBundleError(errors)
    return receipt, receipt_bytes, package_bytes


def _policy_projection(
    request: Mapping[str, Any], observations: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected = {
        str(row.get("candidate_id") or ""): row for row in _rows(request.get("policy_identities"))
    }
    observed = {
        str(row.get("candidate_id") or ""): row
        for row in _rows(observations.get("policy_identities"))
    }
    rows: list[dict[str, Any]] = []
    for candidate_id in FROZEN_CANDIDATES:
        requested = _mapping(expected.get(candidate_id))
        seen = _mapping(observed.get(candidate_id))
        checkpoint = _mapping(requested.get("checkpoint_identity"))
        rows.append(
            {
                "candidate_id": candidate_id,
                "adapter_module": requested.get("adapter_module"),
                "adapter_sha256": requested.get("adapter_sha256"),
                "checkpoint_identity_digest": canonical_digest(checkpoint),
                "observed_adapter_sha256": seen.get("adapter_sha256"),
                "observed_checkpoint_identity_digest": canonical_digest(
                    _mapping(seen.get("checkpoint_identity"))
                ),
            }
        )
    return rows


def _validate_preflight_authoring_identity_join(
    *,
    request: Mapping[str, Any],
    matrix: Mapping[str, Any],
    authoring_manifest: Mapping[str, Any],
) -> None:
    checks = {str(row.get("check_id") or ""): row for row in _rows(matrix.get("static_checks"))}
    source_evidence = _mapping(
        _mapping(checks.get("runtime_source_roots_and_revisions")).get("evidence")
    )
    repositories = _mapping(source_evidence.get("repositories"))
    identity = _mapping(authoring_manifest.get("runtime_identity"))
    identity_sources = _mapping(identity.get("runtime_sources"))
    errors: list[str] = []
    for identity_key, preflight_key in (("isaac_lab", "isaaclab"), ("arena", "arena")):
        expected = _mapping(repositories.get(preflight_key))
        observed = _mapping(identity_sources.get(identity_key))
        for field in ("repository", "revision", "tree"):
            if observed.get(field) != expected.get(field):
                errors.append(
                    f"deformable_native_canary_runtime_identity_join_mismatch:{identity_key}:{field}"
                )
    simulator = _mapping(identity.get("simulator"))
    requested_simulator = _mapping(request.get("simulator_runtime_identity"))
    if identity.get("runtime_id") != requested_simulator.get("runtime_id") or simulator.get(
        "container_image"
    ) != requested_simulator.get("container_image"):
        errors.append("deformable_native_canary_simulator_identity_join_mismatch")
    python = _mapping(identity.get("python"))
    requested_python = _mapping(request.get("runtime_python"))
    if (
        python.get("python_tag") != requested_python.get("python_tag")
        or python.get("abi_tag") != requested_python.get("abi_tag")
        or python.get("platform_tag") not in (requested_python.get("platform_tags") or [])
    ):
        errors.append("deformable_native_canary_python_identity_join_mismatch")
    if errors:
        raise DeformableNativeCanaryBundleError(errors)


def _worker_contract(
    *,
    scene_plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    placement: Mapping[str, Any],
    preflight: Mapping[str, Any],
    authoring_receipt: Mapping[str, Any],
    authoring_manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
    policy_identities: Sequence[Mapping[str, Any]],
    input_byte_digests: Mapping[str, str],
) -> dict[str, Any]:
    gates = [
        {
            "gate_id": gate_id,
            "stage_id": (
                "blank_stage_asset_runtime"
                if gate_id
                in {
                    "dynamic_usd_composition_and_deformable_cooking",
                    "dynamic_cuda_warp_execution",
                    "dynamic_nodal_reset_repeatability",
                }
                else "scene_bound_native_execution"
            ),
            "status": "pending_verified_native_return",
        }
        for gate_id in DYNAMIC_NATIVE_GATE_IDS
    ]
    task_spec = _mapping(scene_plan.get("task_spec"))
    run_binding = {
        "schema_version": RUN_BINDING_SCHEMA_VERSION,
        "run_id": execution["run_id"],
        "scene_plan_digest": scene_plan.get("plan_digest"),
        "placement_receipt_digest": placement.get("receipt_digest"),
        "authoring_receipt_digest": authoring_receipt.get("receipt_digest"),
        "execution_request_digest": execution.get("execution_request_digest"),
    }
    worker: dict[str, Any] = {
        "schema_version": WORKER_CONTRACT_SCHEMA_VERSION,
        "status": "packaged_pending_paid_allocator_and_native_return",
        "run_id": execution["run_id"],
        "run_digest": canonical_digest(run_binding),
        "task_identity": {
            "scene_id": scene_plan.get("scene_id"),
            "task_id": scene_plan.get("task_id"),
            "task_kind": scene_plan.get("task_kind"),
            "scenario": scene_plan.get("scenario"),
            "task_spec": task_spec,
        },
        "task_entity_contract": contract,
        "robot_contract": scene_plan.get("robot"),
        "camera_contracts": scene_plan.get("cameras"),
        "authoring_operations": authoring_manifest.get("entity_authoring_plans"),
        "policy_identities": list(policy_identities),
        "input_bindings": {
            **dict(input_byte_digests),
            "scene_plan_digest": scene_plan.get("plan_digest"),
            "preflight_receipt_digest": preflight.get("receipt_digest"),
            "placement_receipt_digest": placement.get("receipt_digest"),
            "authoring_receipt_digest": authoring_receipt.get("receipt_digest"),
            "authoring_bundle_sha256": authoring_receipt.get("bundle_sha256"),
            "execution_request_digest": execution.get("execution_request_digest"),
        },
        "stages": [
            {
                "stage_id": "blank_stage_asset_runtime",
                "scene_bytes_permitted": False,
                "policy_invocation_permitted": False,
                "required_operations": [
                    "compose_and_parse_exact_authored_assets",
                    "cook_and_read_back_deformable_schemas_and_topology",
                    "execute_physx_and_warp_on_selected_cuda_device",
                    "repeat_native_free_nodal_reset_twice",
                ],
            },
            {
                "stage_id": "scene_bound_native_execution",
                "scene_runtime_reference": execution["scene_runtime_reference"],
                "required_operations": [
                    "compose_exact_registered_scene_and_task_entities",
                    "read_back_entity_poses_stable_support_and_no_penetration",
                    "solve_every_frozen_manipulation_phase",
                    "establish_genuine_native_gripper_deformable_contact",
                    "release_and_retreat_without_post_start_object_state_writes",
                    "capture_synchronized_external_wrist_and_overview_media",
                    "load_and_smoke_both_frozen_policy_adapters_without_scoring",
                ],
            },
        ],
        "dynamic_gates": gates,
        "paid_allocator_requirements": {
            "entrypoint": "python -m blueprint_pipeline.paid_resource_allocator gpu-canary",
            "resource_class": "gpu_canary",
            "canonical_admission_schema": PAID_LANE_ADMISSION_SCHEMA_VERSION,
            "limits": execution["resource_limits"],
            "prelaunch_inventory_must_contain_only_allowed_instance_ids": True,
            "independent_watchdog_must_be_armed_before_create": True,
            "teardown_on_every_terminal_path": True,
            "provider_zero_api_inventory_after_teardown": True,
            "paid_null_and_exact_cost_retention_required": True,
        },
        "media_requirements": {
            "roles": list(REQUIRED_CAMERA_ROLES),
            "visibility_thresholds": execution["visibility_thresholds"],
            "lossless_png_frames_and_manifests_required": True,
            "h264_review_video_and_ffprobe_report_required": True,
            "overview_is_neither_policy_nor_scoring_input": True,
        },
        "authorization_capability_issued": False,
        "provider_mutation_performed": False,
        "native_simulator_executed": False,
        "native_backend_refreeze_admitted": False,
        "claim_boundary": {
            "input_package_only": True,
            "native_simulator_qualified": False,
            "simready_asset_qualified": False,
            "visual_alignment_qualified": False,
            "physical_material_equivalence": False,
            "real_robot_performance": False,
        },
        "worker_contract_digest": "",
    }
    worker["worker_contract_digest"] = canonical_digest(
        worker, digest_field="worker_contract_digest"
    )
    return worker


def _portable_authoring_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "bundle_path"}


def _deterministic_zip(path: Path, members: Mapping[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        for name, content in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_STORED)


def build_deformable_native_canary_bundle(
    *,
    output_dir: str | Path,
    authoring_bundle_receipt_path: str | Path,
    scene_plan_path: str | Path,
    preflight_request_path: str | Path,
    preflight_observations_path: str | Path,
    preflight_matrix_path: str | Path,
    placement_receipt_path: str | Path,
    execution_request_path: str | Path,
    scene_runtime_package_path: str | Path,
    scene_runtime_disclosure_receipt_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create a portable package while issuing no execution authority."""

    scene, scene_bytes, _ = _read_json(scene_plan_path, label="scene_plan")
    preflight_request, request_bytes, _ = _read_json(
        preflight_request_path, label="preflight_request"
    )
    preflight_observations, observations_bytes, _ = _read_json(
        preflight_observations_path, label="preflight_observations"
    )
    preflight_matrix, matrix_bytes, _ = _read_json(preflight_matrix_path, label="preflight_matrix")
    placement, placement_bytes, _ = _read_json(placement_receipt_path, label="placement_receipt")
    execution_raw, execution_bytes, _ = _read_json(
        execution_request_path, label="execution_request"
    )
    authoring_receipt_raw, authoring_receipt_bytes, authoring_receipt_path = _read_json(
        authoring_bundle_receipt_path, label="authoring_receipt"
    )

    contract = _validate_task_entities(scene)
    replayed_preflight = _replay_preflight(
        preflight_request, preflight_observations, preflight_matrix
    )
    replayed_placement = _replay_placement(placement, scene)
    authoring_receipt, authoring_manifest, authoring_bytes, nested_disclosure = _authoring_bundle(
        authoring_receipt_path, contract=contract
    )
    if authoring_receipt_raw != authoring_receipt:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_authoring_receipt_replay_mismatch"]
        )
    _validate_preflight_authoring_identity_join(
        request=preflight_request,
        matrix=replayed_preflight,
        authoring_manifest=authoring_manifest,
    )
    execution = materialize_deformable_native_canary_execution_request(
        execution_raw, authoring_receipt_digest=authoring_receipt["receipt_digest"]
    )
    (
        scene_runtime_disclosure,
        scene_runtime_disclosure_bytes,
        scene_runtime_package_bytes,
    ) = _validate_scene_runtime_disclosure(
        package_path_value=scene_runtime_package_path,
        disclosure_receipt_path=scene_runtime_disclosure_receipt_path,
        expected_reference=execution["scene_runtime_reference"],
    )

    selected_robot = str(_mapping(scene.get("robot")).get("robot_id") or "")
    if preflight_request.get("selected_robot_id") != selected_robot:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_robot_preflight_join_mismatch"]
        )
    runtime_identity = _mapping(authoring_manifest.get("runtime_identity"))
    if _mapping(runtime_identity.get("selected_robot")).get("robot_id") != selected_robot:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_robot_authoring_join_mismatch"]
        )
    policies = _policy_projection(preflight_request, preflight_observations)

    input_digests = {
        "scene_plan_file_sha256": _sha256_bytes(scene_bytes),
        "preflight_request_file_sha256": _sha256_bytes(request_bytes),
        "preflight_observations_file_sha256": _sha256_bytes(observations_bytes),
        "preflight_matrix_file_sha256": _sha256_bytes(matrix_bytes),
        "placement_receipt_file_sha256": _sha256_bytes(placement_bytes),
        "execution_request_file_sha256": _sha256_bytes(execution_bytes),
        "authoring_receipt_file_sha256": _sha256_bytes(authoring_receipt_bytes),
        "scene_runtime_package_file_sha256": _sha256_bytes(scene_runtime_package_bytes),
        "scene_runtime_disclosure_file_sha256": _sha256_bytes(scene_runtime_disclosure_bytes),
        "scene_runtime_disclosure_receipt_digest": scene_runtime_disclosure["receipt_digest"],
    }
    worker = _worker_contract(
        scene_plan=scene,
        contract=contract,
        placement=replayed_placement,
        preflight=replayed_preflight,
        authoring_receipt=authoring_receipt,
        authoring_manifest=authoring_manifest,
        execution=execution,
        policy_identities=policies,
        input_byte_digests=input_digests,
    )

    portable_inputs = {
        f"{INPUT_ROOT_NAME}/{WORKER_CONTRACT_FILENAME}": json.dumps(
            worker, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode(),
        f"{INPUT_ROOT_NAME}/{AUTHORING_BUNDLE_FILENAME}": authoring_bytes,
        f"{INPUT_ROOT_NAME}/authoring_receipt.json": json.dumps(
            _portable_authoring_receipt(authoring_receipt),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(),
        f"{INPUT_ROOT_NAME}/scene_runtime_disclosure_receipt.json": (
            scene_runtime_disclosure_bytes
        ),
    }
    disclosure: dict[str, Any] = {
        "schema_version": DISCLOSURE_SCHEMA_VERSION,
        "status": "portable_payload_bytes_validated",
        "payload_members": [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": _sha256_bytes(content),
                "classification": (
                    "derived_task_entity_asset_bundle"
                    if name.endswith(AUTHORING_BUNDLE_FILENAME)
                    else "portable_contract_json"
                ),
            }
            for name, content in sorted(portable_inputs.items())
        ],
        "nested_authoring_members": nested_disclosure,
        "external_scene_runtime_reference": execution["scene_runtime_reference"],
        "external_scene_bytes_included": False,
        "raw_dataset_file_format_members_detected": [],
        "provider_training_permitted": False,
        "disclosure_manifest_digest": "",
    }
    disclosure["disclosure_manifest_digest"] = canonical_digest(
        disclosure, digest_field="disclosure_manifest_digest"
    )
    disclosure_name = f"{INPUT_ROOT_NAME}/{DISCLOSURE_FILENAME}"
    portable_inputs[disclosure_name] = json.dumps(
        disclosure, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()

    output = Path(output_dir).expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise DeformableNativeCanaryBundleError(["deformable_native_canary_output_exists"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        bundle_path = staging / BUNDLE_FILENAME
        _deterministic_zip(bundle_path, portable_inputs)
        receipt: dict[str, Any] = {
            "schema_version": BUNDLE_RECEIPT_SCHEMA_VERSION,
            "generated_at": generated_at or utc_now_iso(),
            "status": "packaged_pending_canonical_paid_admission_and_native_return",
            "bundle_name": BUNDLE_FILENAME,
            "bundle_path": str(output / BUNDLE_FILENAME),
            "bundle_size_bytes": bundle_path.stat().st_size,
            "bundle_sha256": _sha256_file(bundle_path),
            "worker_contract_digest": worker["worker_contract_digest"],
            "disclosure_manifest_digest": disclosure["disclosure_manifest_digest"],
            "execution_request_digest": execution["execution_request_digest"],
            "run_digest": worker["run_digest"],
            "dynamic_gate_ids": list(DYNAMIC_NATIVE_GATE_IDS),
            "authorization_capability_issued": False,
            "provider_mutation_performed": False,
            "native_simulator_executed": False,
            "native_backend_refreeze_admitted": False,
            "receipt_digest": "",
        }
        receipt["receipt_digest"] = _digest_without(receipt, "bundle_path", "receipt_digest")
        write_json(staging / RECEIPT_FILENAME, receipt)
        staging.replace(output)
        return receipt
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_disclosure(
    archive: zipfile.ZipFile, names: set[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    disclosure_name = f"{INPUT_ROOT_NAME}/{DISCLOSURE_FILENAME}"
    worker_name = f"{INPUT_ROOT_NAME}/{WORKER_CONTRACT_FILENAME}"
    if disclosure_name not in names or worker_name not in names:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_bundle_required_member_missing"]
        )
    disclosure = _strict_json_bytes(
        archive.read(disclosure_name), error="deformable_native_canary_disclosure_invalid"
    )
    worker = _strict_json_bytes(
        archive.read(worker_name), error="deformable_native_canary_worker_contract_invalid"
    )
    payload_rows = _rows(disclosure.get("payload_members"))
    expected_names = {disclosure_name, *(str(row.get("path") or "") for row in payload_rows)}
    errors: list[str] = []
    if names != expected_names:
        errors.append("deformable_native_canary_disclosure_member_set_mismatch")
    if (
        set(disclosure)
        != {
            "schema_version",
            "status",
            "payload_members",
            "nested_authoring_members",
            "external_scene_runtime_reference",
            "external_scene_bytes_included",
            "raw_dataset_file_format_members_detected",
            "provider_training_permitted",
            "disclosure_manifest_digest",
        }
        or disclosure.get("schema_version") != DISCLOSURE_SCHEMA_VERSION
        or disclosure.get("disclosure_manifest_digest")
        != canonical_digest(disclosure, digest_field="disclosure_manifest_digest")
        or disclosure.get("external_scene_bytes_included") is not False
        or disclosure.get("raw_dataset_file_format_members_detected") != []
        or disclosure.get("provider_training_permitted") is not False
    ):
        errors.append("deformable_native_canary_disclosure_contract_invalid")
    for row in payload_rows:
        if set(row) != {"path", "size_bytes", "sha256", "classification"}:
            errors.append("deformable_native_canary_disclosure_payload_row_invalid")
        name = str(row.get("path") or "")
        if name not in names:
            continue
        content = archive.read(name)
        if len(content) != row.get("size_bytes") or _sha256_bytes(content) != row.get("sha256"):
            errors.append("deformable_native_canary_disclosure_payload_identity_mismatch")
    authoring_name = f"{INPUT_ROOT_NAME}/{AUTHORING_BUNDLE_FILENAME}"
    authoring_receipt_name = f"{INPUT_ROOT_NAME}/authoring_receipt.json"
    if authoring_name in names and authoring_receipt_name in names:
        authoring_bytes = archive.read(authoring_name)
        portable_receipt = _strict_json_bytes(
            archive.read(authoring_receipt_name),
            error="deformable_native_canary_portable_authoring_receipt_invalid",
        )
        nested_rows: list[dict[str, Any]] = []
        forbidden = {
            str(row["source_observation"]["source_sha256"])
            for row in _rows(_mapping(worker.get("task_entity_contract")).get("task_entities"))
        }
        try:
            with zipfile.ZipFile(io.BytesIO(authoring_bytes)) as nested:
                nested_infos = _safe_archive_infos(
                    nested,
                    prefix="deformable_native_canary_nested_authoring",
                    maximum_size=_PACKAGE_LIMIT,
                )
                nested_names = {info.filename for info in nested_infos}
                nested_manifest_name = PurePosixPath(
                    AUTHORING_SOURCE_ROOT_NAME,
                    "native_task_entity_asset_authoring_input.v1.json",
                ).as_posix()
                if nested_manifest_name not in nested_names:
                    errors.append("deformable_native_canary_nested_authoring_manifest_missing")
                else:
                    nested_manifest = _strict_json_bytes(
                        nested.read(nested_manifest_name),
                        error="deformable_native_canary_nested_authoring_manifest_invalid",
                    )
                    expected_nested_names = {nested_manifest_name}
                    for plan in _rows(nested_manifest.get("entity_authoring_plans")):
                        for file_row in _rows(plan.get("staged_files")):
                            role = str(file_row.get("role") or "")
                            nested_path = PurePosixPath(
                                AUTHORING_SOURCE_ROOT_NAME,
                                str(file_row.get("archive_relative_path") or ""),
                            ).as_posix()
                            expected_nested_names.add(nested_path)
                            content = (
                                nested.read(nested_path) if nested_path in nested_names else b""
                            )
                            digest = _sha256_bytes(content)
                            if (
                                role not in _ASSET_ROLE_SET
                                or len(content) != file_row.get("size_bytes")
                                or digest != file_row.get("sha256")
                                or digest in forbidden
                                or not _asset_content_valid(role, nested_path, content)
                            ):
                                errors.append(
                                    f"deformable_native_canary_nested_authoring_file_invalid:{role}"
                                )
                            nested_rows.append(
                                {
                                    "nested_path": nested_path,
                                    "role": role,
                                    "size_bytes": len(content),
                                    "sha256": digest,
                                    "classification": "derived_runtime_asset",
                                }
                            )
                    if nested_names != expected_nested_names:
                        errors.append(
                            "deformable_native_canary_nested_authoring_member_set_invalid"
                        )
                    if (
                        nested_manifest.get("input_digest")
                        != canonical_digest(nested_manifest, digest_field="input_digest")
                        or nested_manifest.get("entity_authoring_plans")
                        != worker.get("authoring_operations")
                        or portable_receipt.get("input_digest")
                        != nested_manifest.get("input_digest")
                    ):
                        errors.append("deformable_native_canary_nested_authoring_join_invalid")
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            raise DeformableNativeCanaryBundleError(
                ["deformable_native_canary_nested_authoring_archive_invalid"]
            ) from exc
        if (
            portable_receipt.get("schema_version") != AUTHORING_RECEIPT_SCHEMA_VERSION
            or portable_receipt.get("status") != "ready_for_native_authoring_canary"
            or portable_receipt.get("native_simulator_executed") is not False
            or portable_receipt.get("native_qualification_claimed") is not False
            or portable_receipt.get("raw_dataset_source_bytes_included") is not False
            or portable_receipt.get("receipt_digest")
            != _digest_without(portable_receipt, "bundle_path", "receipt_digest")
            or portable_receipt.get("bundle_sha256") != _sha256_bytes(authoring_bytes)
            or portable_receipt.get("bundle_size_bytes") != len(authoring_bytes)
            or sorted(nested_rows, key=lambda row: row["nested_path"])
            != sorted(
                _rows(disclosure.get("nested_authoring_members")),
                key=lambda row: str(row.get("nested_path") or ""),
            )
        ):
            errors.append("deformable_native_canary_nested_disclosure_replay_mismatch")
    else:
        errors.append("deformable_native_canary_authoring_payload_missing")
    if (
        worker.get("schema_version") != WORKER_CONTRACT_SCHEMA_VERSION
        or worker.get("worker_contract_digest")
        != canonical_digest(worker, digest_field="worker_contract_digest")
        or worker.get("authorization_capability_issued") is not False
        or worker.get("provider_mutation_performed") is not False
        or worker.get("native_simulator_executed") is not False
        or worker.get("native_backend_refreeze_admitted") is not False
        or [row.get("gate_id") for row in _rows(worker.get("dynamic_gates"))]
        != list(DYNAMIC_NATIVE_GATE_IDS)
        or any(
            row.get("status") != "pending_verified_native_return"
            for row in _rows(worker.get("dynamic_gates"))
        )
    ):
        errors.append("deformable_native_canary_worker_contract_replay_invalid")
    if errors:
        raise DeformableNativeCanaryBundleError(errors)
    return worker, disclosure


def verify_deformable_native_canary_bundle(
    receipt_path: str | Path,
    *,
    expected_receipt_digest: str | None = None,
    expected_worker_contract_digest: str | None = None,
) -> dict[str, Any]:
    """Fully rederive portable package identities without executing it."""

    receipt, _receipt_bytes, _ = _read_json(receipt_path, label="bundle_receipt")
    errors: list[str] = []
    if (
        receipt.get("schema_version") != BUNDLE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "packaged_pending_canonical_paid_admission_and_native_return"
        or receipt.get("bundle_name") != BUNDLE_FILENAME
        or receipt.get("authorization_capability_issued") is not False
        or receipt.get("provider_mutation_performed") is not False
        or receipt.get("native_simulator_executed") is not False
        or receipt.get("native_backend_refreeze_admitted") is not False
        or receipt.get("dynamic_gate_ids") != list(DYNAMIC_NATIVE_GATE_IDS)
    ):
        errors.append("deformable_native_canary_bundle_receipt_contract_invalid")
    if receipt.get("receipt_digest") != _digest_without(receipt, "bundle_path", "receipt_digest"):
        errors.append("deformable_native_canary_bundle_receipt_digest_invalid")
    if expected_receipt_digest and receipt.get("receipt_digest") != expected_receipt_digest:
        errors.append("deformable_native_canary_bundle_receipt_expected_digest_mismatch")
    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    try:
        if (
            not bundle_path.is_file()
            or bundle_path.stat().st_size != receipt.get("bundle_size_bytes")
            or bundle_path.stat().st_size > _PACKAGE_LIMIT
            or _sha256_file(bundle_path) != receipt.get("bundle_sha256")
        ):
            errors.append("deformable_native_canary_bundle_bytes_identity_mismatch")
        if errors:
            raise DeformableNativeCanaryBundleError(errors)
        with zipfile.ZipFile(bundle_path) as archive:
            infos = _safe_archive_infos(
                archive, prefix="deformable_native_canary_bundle", maximum_size=_PACKAGE_LIMIT
            )
            worker, disclosure = _verify_disclosure(archive, {info.filename for info in infos})
    except (OSError, zipfile.BadZipFile) as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_bundle_archive_invalid"]
        ) from exc
    if worker["worker_contract_digest"] != receipt.get("worker_contract_digest"):
        errors.append("deformable_native_canary_worker_contract_digest_mismatch")
    if worker.get("run_digest") != receipt.get("run_digest"):
        errors.append("deformable_native_canary_run_digest_mismatch")
    if (
        expected_worker_contract_digest
        and worker["worker_contract_digest"] != expected_worker_contract_digest
    ):
        errors.append("deformable_native_canary_worker_contract_expected_digest_mismatch")
    if disclosure["disclosure_manifest_digest"] != receipt.get("disclosure_manifest_digest"):
        errors.append("deformable_native_canary_disclosure_digest_mismatch")
    if errors:
        raise DeformableNativeCanaryBundleError(errors)
    return receipt


def _parse_timestamp(value: Any, *, error: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DeformableNativeCanaryBundleError([error]) from exc
    if parsed.tzinfo is None:
        raise DeformableNativeCanaryBundleError([error])
    return parsed.astimezone(timezone.utc)


def _report_digest_valid(report: Mapping[str, Any], schema: str) -> bool:
    return bool(
        report.get("schema_version") == schema
        and report.get("report_digest") == canonical_digest(report, digest_field="report_digest")
    )


def _gate_blockers(
    *,
    worker: Mapping[str, Any],
    blank: Mapping[str, Any],
    scene: Mapping[str, Any],
    camera_artifacts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    blockers: list[str] = []
    authoring = {
        str(row.get("entity_id") or ""): row for row in _rows(worker.get("authoring_operations"))
    }
    asset_rows = {
        str(row.get("entity_id") or ""): row for row in _rows(blank.get("asset_readbacks"))
    }
    if set(asset_rows) != set(authoring):
        blockers.append("dynamic_usd_composition_and_deformable_cooking")
    else:
        for entity_id, expected in authoring.items():
            observed = asset_rows[entity_id]
            required_schemas = set(expected.get("operation", {}).get("required_schemas") or [])
            if (
                observed.get("candidate_digest") != expected.get("candidate_digest")
                or observed.get("runtime_asset_sha256")
                != next(
                    (
                        row.get("sha256")
                        for row in _rows(expected.get("staged_files"))
                        if row.get("role") == "runtime_usd"
                    ),
                    None,
                )
                or not required_schemas.issubset(set(observed.get("applied_schemas") or []))
                or observed.get("observed_prim_type")
                != _mapping(expected.get("operation")).get("expected_prim_type")
                or int(observed.get("cooked_element_count") or 0) <= 0
            ):
                blockers.append("dynamic_usd_composition_and_deformable_cooking")
                break
    device = _mapping(blank.get("device_execution"))
    if (
        not str(device.get("cuda_device_uuid") or "").strip()
        or int(device.get("physx_gpu_step_count") or 0) <= 0
        or int(device.get("warp_launch_count") or 0) <= 0
    ):
        blockers.append("dynamic_cuda_warp_execution")
    reset = _mapping(blank.get("reset_readback"))
    reset_digests = [
        reset.get("initial_nodal_state_sha256"),
        reset.get("first_reset_nodal_state_sha256"),
        reset.get("second_reset_nodal_state_sha256"),
    ]
    if (
        not all(_valid_digest(value) for value in reset_digests)
        or len(set(reset_digests)) != 1
        or reset.get("post_start_direct_object_state_write_count") != 0
    ):
        blockers.append("dynamic_nodal_reset_repeatability")

    task_spec = _mapping(_mapping(worker.get("task_identity")).get("task_spec"))
    contact = _mapping(scene.get("contact_readback"))
    actions = _rows(contact.get("action_trace"))
    robot_trace = _rows(contact.get("robot_state_trace"))
    contact_events = _rows(contact.get("native_contact_events"))
    phase_samples = {
        str(row.get("phase") or ""): row for row in _rows(contact.get("phase_samples"))
    }
    action_dimension = int(
        _mapping(_mapping(worker.get("robot_contract")).get("action_seam")).get("action_dimension")
        or 0
    )

    def numeric_vector(value: Any, length: int | None = None) -> list[float] | None:
        if (
            isinstance(value, (str, bytes, Mapping))
            or not isinstance(value, Sequence)
            or (length is not None and len(value) != length)
        ):
            return None
        try:
            normalized = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return normalized if all(math.isfinite(item) for item in normalized) else None

    action_vectors = [numeric_vector(row.get("action"), action_dimension) for row in actions]
    arm_vectors = [numeric_vector(row.get("arm_joint_positions_rad")) for row in robot_trace]
    gripper_vectors = [
        numeric_vector(row.get("gripper_joint_positions_rad")) for row in robot_trace
    ]
    arm_delta = 0.0
    gripper_delta = 0.0
    if (
        len(arm_vectors) >= 2
        and all(arm_vectors)
        and len({len(row or []) for row in arm_vectors}) == 1
    ):
        arm_delta = max(
            abs(right - left)
            for left, right in zip(arm_vectors[0] or [], arm_vectors[-1] or [], strict=True)
        )
    if (
        len(gripper_vectors) >= 2
        and all(gripper_vectors)
        and len({len(row or []) for row in gripper_vectors}) == 1
    ):
        gripper_delta = max(
            abs(right - left)
            for left, right in zip(gripper_vectors[0] or [], gripper_vectors[-1] or [], strict=True)
        )
    lift = _mapping(phase_samples.get("lift"))
    release = _mapping(phase_samples.get("release"))
    retreat = _mapping(phase_samples.get("retreat"))
    try:
        phase_evidence_ok = bool(
            float(lift.get("deformable_height_delta_m")) > 0
            and int(release.get("gripper_deformable_contact_count")) == 0
            and float(retreat.get("robot_deformable_clearance_m"))
            >= float(task_spec.get("minimum_robot_clearance_m"))
        )
    except (TypeError, ValueError):
        phase_evidence_ok = False
    try:
        contact_events_ok = bool(
            contact_events
            and all(
                math.isfinite(float(row.get("normal_force_n")))
                and float(row.get("normal_force_n")) > 0
                for row in contact_events
            )
        )
    except (TypeError, ValueError):
        contact_events_ok = False
    if (
        not actions
        or not robot_trace
        or any(vector is None for vector in action_vectors)
        or arm_delta <= 0
        or gripper_delta <= 0
        or not contact_events_ok
        or not phase_evidence_ok
        or _rows(contact.get("hidden_attachment_events"))
        or _rows(contact.get("post_start_direct_object_state_write_events"))
    ):
        blockers.append("dynamic_genuine_gripper_deformable_contact")
    settle = _mapping(scene.get("deformable_settle_readback"))
    try:
        settle_ok = bool(
            int(settle.get("settle_window_sample_count"))
            >= int(task_spec.get("settle_window_samples"))
            and float(settle.get("maximum_node_speed_mps"))
            <= float(task_spec.get("maximum_node_speed_mps"))
            and float(settle.get("maximum_principal_strain"))
            <= float(task_spec.get("maximum_principal_strain"))
            and int(settle.get("nan_count")) == 0
            and int(settle.get("solver_divergence_count")) == 0
        )
    except (TypeError, ValueError):
        settle_ok = False
    if not settle_ok:
        blockers.append("dynamic_deformable_settling_and_strain_readback")
    applied = _mapping(scene.get("applied_parameter_readback"))
    bindings = _mapping(worker.get("input_bindings"))
    if any(
        applied.get(field) != bindings.get(field)
        for field in (
            "scene_plan_digest",
            "placement_receipt_digest",
            "authoring_receipt_digest",
            "execution_request_digest",
        )
    ) or set(applied.get("entity_ids") or []) != {
        row["entity_id"]
        for row in _rows(_mapping(worker.get("task_entity_contract")).get("task_entities"))
    }:
        blockers.append("dynamic_applied_parameter_readback")
    expected_entities = {
        row["entity_id"]: row
        for row in _rows(_mapping(worker.get("task_entity_contract")).get("task_entities"))
    }
    entity_readbacks = {
        str(row.get("entity_id") or ""): row for row in _rows(scene.get("entity_readbacks"))
    }
    if set(entity_readbacks) != set(expected_entities) or any(
        entity_readbacks[entity_id].get("runtime_asset_sha256") != entity["runtime_asset"]["sha256"]
        or entity_readbacks[entity_id].get("pose_world") != entity["initial_state"]["pose_world"]
        for entity_id, entity in expected_entities.items()
    ):
        blockers.append("dynamic_applied_parameter_readback")
    robot_readback = _mapping(scene.get("robot_readback"))
    robot_contract = _mapping(worker.get("robot_contract"))
    if (
        robot_readback.get("robot_id") != robot_contract.get("robot_id")
        or robot_readback.get("action_seam") != robot_contract.get("action_seam")
        or robot_readback.get("base_pose_world") != robot_contract.get("base_pose_world")
    ):
        blockers.append("dynamic_applied_parameter_readback")

    receptacle = _mapping(scene.get("receptacle_readback"))
    try:
        receptacle_ok = bool(
            receptacle.get("initial_penetration_count") == 0
            and int(receptacle.get("support_contact_event_count")) > 0
            and float(receptacle.get("maximum_translation_drift_m"))
            <= float(task_spec.get("maximum_receptacle_translation_drift_m"))
            and float(receptacle.get("maximum_rotation_drift_rad"))
            <= float(task_spec.get("maximum_receptacle_rotation_drift_rad"))
        )
    except (TypeError, ValueError):
        receptacle_ok = False
    if not receptacle_ok:
        blockers.append("dynamic_rigid_receptacle_support_and_pose_stability")
    phases = {str(row.get("phase") or ""): row for row in _rows(scene.get("ik_readbacks"))}
    required_phases = {
        "pregrasp",
        "grasp",
        "lift",
        "transport",
        "deposit",
        "release",
        "retreat",
        "recovery",
    }
    if set(phases) != required_phases or any(
        row.get("solution_count", 0) <= 0 or row.get("collision_count") != 0
        for row in phases.values()
    ):
        blockers.append("dynamic_full_phase_ik_reachability")
    expected_policies = {row["candidate_id"]: row for row in _rows(worker.get("policy_identities"))}
    observed_policies = {
        str(row.get("candidate_id") or ""): row
        for row in _rows(scene.get("policy_adapter_readbacks"))
    }
    if set(observed_policies) != set(expected_policies) or any(
        observed_policies[candidate_id].get("adapter_sha256") != expected.get("adapter_sha256")
        or observed_policies[candidate_id].get("checkpoint_identity_digest")
        != expected.get("checkpoint_identity_digest")
        or int(observed_policies[candidate_id].get("load_count") or 0) != 1
        or int(observed_policies[candidate_id].get("preprocess_call_count") or 0) <= 0
        or int(observed_policies[candidate_id].get("action_adapter_call_count") or 0) <= 0
        or not _valid_digest(observed_policies[candidate_id].get("load_receipt_sha256"))
        or not _valid_digest(observed_policies[candidate_id].get("preprocessed_observation_sha256"))
        or numeric_vector(observed_policies[candidate_id].get("sample_action"), action_dimension)
        is None
        for candidate_id, expected in expected_policies.items()
    ):
        blockers.append("dynamic_policy_adapter_runtime_smoke")

    camera_rows = {str(row.get("role") or ""): row for row in _rows(scene.get("camera_readbacks"))}
    thresholds = _mapping(_mapping(worker.get("media_requirements")).get("visibility_thresholds"))
    camera_ok = set(camera_rows) >= set(REQUIRED_CAMERA_ROLES)
    for role in REQUIRED_CAMERA_ROLES:
        row = _mapping(camera_rows.get(role))
        artifacts = _mapping(camera_artifacts.get(role))
        expected_camera = next(
            (value for value in _rows(worker.get("camera_contracts")) if value.get("role") == role),
            {},
        )
        if (
            not artifacts
            or row.get("frame_manifest_digest") != artifacts.get("manifest_digest")
            or row.get("video_sha256") != artifacts.get("video_sha256")
            or row.get("ffprobe_report_digest") != artifacts.get("ffprobe_report_digest")
            or not _valid_digest(row.get("camera_calibration_digest"))
            or int(row.get("synchronized_frame_count") or 0) <= 0
            or row.get("intrinsics") != expected_camera.get("intrinsics")
            or row.get("pose_frame") != expected_camera.get("pose_frame")
        ):
            camera_ok = False
            continue
        role_thresholds = _mapping(thresholds.get(role))
        for field, minimum in role_thresholds.items():
            try:
                if float(row.get(field)) < float(minimum):
                    camera_ok = False
            except (TypeError, ValueError):
                camera_ok = False
    if not camera_ok:
        blockers.append("dynamic_renderer_camera_capture")
    return sorted(set(blockers))


def _return_artifacts(
    return_bundle_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(return_bundle_bytes)) as archive:
            infos = _safe_archive_infos(
                archive, prefix="deformable_native_canary_return", maximum_size=_RETURN_LIMIT
            )
            names = {info.filename for info in infos}
            if RETURN_MANIFEST_FILENAME not in names:
                raise DeformableNativeCanaryBundleError(
                    ["deformable_native_canary_return_manifest_missing"]
                )
            manifest = _strict_json_bytes(
                archive.read(RETURN_MANIFEST_FILENAME),
                error="deformable_native_canary_return_manifest_invalid",
            )
            artifacts: dict[str, bytes] = {}
            paths: dict[str, str] = {}
            for row in _rows(manifest.get("artifacts")):
                if set(row) != {"role", "path", "size_bytes", "sha256"}:
                    raise DeformableNativeCanaryBundleError(
                        ["deformable_native_canary_return_artifact_inventory_invalid"]
                    )
                role = str(row.get("role") or "")
                path = str(row.get("path") or "")
                if (
                    not _RETURN_ROLE_RE.fullmatch(role)
                    or role in artifacts
                    or path in paths.values()
                ):
                    raise DeformableNativeCanaryBundleError(
                        ["deformable_native_canary_return_artifact_inventory_invalid"]
                    )
                if path not in names or path == RETURN_MANIFEST_FILENAME:
                    raise DeformableNativeCanaryBundleError(
                        ["deformable_native_canary_return_artifact_missing"]
                    )
                content = archive.read(path)
                if len(content) != row.get("size_bytes") or _sha256_bytes(content) != row.get(
                    "sha256"
                ):
                    raise DeformableNativeCanaryBundleError(
                        ["deformable_native_canary_return_artifact_identity_mismatch"]
                    )
                if path.endswith(".json") and any(
                    marker in content.lower() for marker in _SECRET_MARKERS
                ):
                    raise DeformableNativeCanaryBundleError(
                        ["deformable_native_canary_return_secret_material_forbidden"]
                    )
                artifacts[role] = content
                paths[role] = path
            if names != {RETURN_MANIFEST_FILENAME, *paths.values()}:
                raise DeformableNativeCanaryBundleError(
                    ["deformable_native_canary_return_member_set_invalid"]
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_return_archive_invalid"]
        ) from exc
    if (
        set(manifest)
        != {
            "schema_version",
            "run_id",
            "instance_id",
            "package_receipt_digest",
            "package_bundle_sha256",
            "worker_contract_digest",
            "artifacts",
            "manifest_digest",
        }
        or manifest.get("schema_version") != RETURN_MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_digest")
        != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_return_manifest_digest_invalid"]
        )
    return manifest, artifacts, paths


def _json_artifact(artifacts: Mapping[str, bytes], role: str) -> dict[str, Any]:
    content = artifacts.get(role)
    if content is None:
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_return_artifact_missing:{role}"]
        )
    return _strict_json_bytes(content, error=f"deformable_native_canary_return_json_invalid:{role}")


def _vast_inventory_instance_ids(content: bytes, *, role: str) -> set[int]:
    response = _strict_json_bytes(
        content, error=f"deformable_native_canary_vast_response_invalid:{role}"
    )
    if set(response) != {"instances"} or not isinstance(response.get("instances"), list):
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_vast_response_invalid:{role}"]
        )
    instance_ids: set[int] = set()
    for row in response["instances"]:
        if not isinstance(row, Mapping):
            raise DeformableNativeCanaryBundleError(
                [f"deformable_native_canary_vast_response_invalid:{role}"]
            )
        instance_id = row.get("instance_id")
        if (
            isinstance(instance_id, bool)
            or not isinstance(instance_id, int)
            or instance_id <= 0
            or instance_id in instance_ids
        ):
            raise DeformableNativeCanaryBundleError(
                [f"deformable_native_canary_vast_response_invalid:{role}"]
            )
        instance_ids.add(instance_id)
    return instance_ids


def _verify_camera_artifacts(
    artifacts: Mapping[str, bytes], paths: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role in REQUIRED_CAMERA_ROLES:
        manifest_role = f"camera_manifest:{role}"
        video_role = f"camera_video:{role}"
        ffprobe_role = f"camera_ffprobe:{role}"
        frame_manifest = _json_artifact(artifacts, manifest_role)
        ffprobe = _json_artifact(artifacts, ffprobe_role)
        frames = _rows(frame_manifest.get("frames"))
        errors: list[str] = []
        if (
            frame_manifest.get("schema_version") != "deformable_native_canary_frame_manifest.v1"
            or frame_manifest.get("role") != role
            or frame_manifest.get("manifest_digest")
            != canonical_digest(frame_manifest, digest_field="manifest_digest")
            or not frames
        ):
            errors.append(f"deformable_native_canary_camera_manifest_invalid:{role}")
        timestamps: list[int] = []
        decoded_frames: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            frame_role = f"camera_frame:{role}:{index:06d}"
            content = artifacts.get(frame_role)
            decoded_frame = _decoded_png_rgb(content or b"")
            dimensions = (
                (decoded_frame["width"], decoded_frame["height"])
                if decoded_frame is not None
                else None
            )
            if (
                content is None
                or frame.get("path") != paths.get(frame_role)
                or frame.get("sha256") != _sha256_bytes(content)
                or frame.get("size_bytes") != len(content)
                or dimensions != (frame.get("width"), frame.get("height"))
                or isinstance(frame.get("timestamp_ns"), bool)
                or not isinstance(frame.get("timestamp_ns"), int)
            ):
                errors.append(f"deformable_native_canary_camera_frame_invalid:{role}:{index}")
            else:
                timestamps.append(frame["timestamp_ns"])
                decoded_frames.append(decoded_frame)
        if timestamps != sorted(set(timestamps)):
            errors.append(f"deformable_native_canary_camera_timestamps_invalid:{role}")
        video = artifacts.get(video_role, b"")
        decoded_video = _decoded_h264_rgb(video)
        if decoded_video is None:
            errors.append(f"deformable_native_canary_camera_video_invalid:{role}")
        elif (
            decoded_video["sample_count"] != len(decoded_frames)
            or any(
                (row["width"], row["height"]) != (decoded_video["width"], decoded_video["height"])
                for row in decoded_frames
            )
            or [row["raw_rgb_sha256"] for row in decoded_frames]
            != decoded_video["raw_rgb_sha256_by_sample"]
        ):
            errors.append(f"deformable_native_canary_camera_video_derivation_invalid:{role}")
        if (
            ffprobe.get("schema_version") != "deformable_native_canary_ffprobe_report.v1"
            or ffprobe.get("report_digest")
            != canonical_digest(ffprobe, digest_field="report_digest")
            or ffprobe.get("video_sha256") != _sha256_bytes(video)
            or not _valid_digest(ffprobe.get("ffprobe_binary_sha256"))
            or not str(ffprobe.get("ffprobe_version") or "").strip()
            or not any(
                row.get("codec_type") == "video" and row.get("codec_name") == "h264"
                for row in _rows(ffprobe.get("streams"))
            )
        ):
            errors.append(f"deformable_native_canary_camera_ffprobe_invalid:{role}")
        if errors:
            raise DeformableNativeCanaryBundleError(errors)
        result[role] = {
            "manifest_digest": frame_manifest["manifest_digest"],
            "video_sha256": _sha256_bytes(video),
            "ffprobe_report_digest": ffprobe["report_digest"],
            "verifier_decoded_sample_count": decoded_video["sample_count"],
            "exact_lossless_rgb_correspondence": True,
        }
    return result


def _verify_provider_lifecycle(
    *,
    worker: Mapping[str, Any],
    package_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifacts: Mapping[str, bytes],
) -> tuple[float, int, datetime, datetime, datetime]:
    # This legacy parser is intentionally terminal. Worker-returned JSON is not
    # an authority source for provider lifecycle or provider-zero; a future
    # verifier-owned proof must live behind a separate contract.
    raise DeformableNativeCanaryBundleError(
        ["deformable_native_canary_worker_provider_lifecycle_untrusted"]
    )
    admission = _json_artifact(artifacts, "provider_admission")
    pre = _json_artifact(artifacts, "prelaunch_inventory")
    allocation = _json_artifact(artifacts, "allocation_receipt")
    watchdog = _json_artifact(artifacts, "watchdog_receipt")
    billing = _json_artifact(artifacts, "billing_receipt")
    upload = _json_artifact(artifacts, "upload_receipt")
    scene_disclosure = _json_artifact(artifacts, "scene_disclosure_receipt")
    teardown = _json_artifact(artifacts, "teardown_receipt")
    zero = _json_artifact(artifacts, "provider_zero_inventory")
    pre_response = artifacts.get("prelaunch_inventory_response", b"")
    zero_response = artifacts.get("provider_zero_inventory_response", b"")
    billing_response = _json_artifact(artifacts, "billing_response")
    limits = _mapping(_mapping(worker.get("paid_allocator_requirements")).get("limits"))
    scene_ref = _mapping(
        next(
            row
            for row in _rows(worker.get("stages"))
            if row.get("stage_id") == "scene_bound_native_execution"
        ).get("scene_runtime_reference")
    )
    expected_binding = {
        "run_id": worker.get("run_id"),
        "provider_id": "vast",
        "canary_package_receipt_digest": package_receipt.get("receipt_digest"),
        "canary_package_sha256": package_receipt.get("bundle_sha256"),
        "attempt_spend_cap_usd": limits.get("attempt_spend_cap_usd"),
        "ttl_seconds": limits.get("ttl_seconds"),
        "watchdog_seconds": limits.get("watchdog_seconds"),
        "allowed_active_vast_instance_ids": limits.get("allowed_active_instance_ids"),
    }
    if admission.get("allocation_binding") != expected_binding:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_provider_admission_binding_invalid"]
        )
    try:
        grant = require_paid_resource_admission(
            admission,
            resource_class="gpu_canary",
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
        require_paid_resource_admission_grant(
            grant,
            resource_class="gpu_canary",
            allocation_binding_digest=admission.get("allocation_binding_digest"),
            require_allocation_binding=True,
            allowed_active_instance_ids=limits.get("allowed_active_instance_ids"),
        )
    except PaidResourceAdmissionBlocked as exc:
        raise DeformableNativeCanaryBundleError(
            [f"deformable_native_canary_provider_admission_invalid:{item}" for item in exc.blockers]
        ) from exc

    def valid_receipt(value: Mapping[str, Any], schema: str, digest_field: str) -> bool:
        return bool(
            value.get("schema_version") == schema
            and value.get(digest_field) == canonical_digest(value, digest_field=digest_field)
        )

    schemas = (
        (pre, "deformable_native_canary_vast_inventory.v1", "inventory_digest"),
        (allocation, "deformable_native_canary_allocation.v1", "allocation_digest"),
        (watchdog, "deformable_native_canary_watchdog.v1", "watchdog_digest"),
        (billing, "deformable_native_canary_billing.v1", "billing_digest"),
        (upload, "deformable_native_canary_upload.v1", "upload_digest"),
        (teardown, "deformable_native_canary_teardown.v1", "teardown_digest"),
        (zero, "deformable_native_canary_vast_inventory.v1", "inventory_digest"),
    )
    if any(not valid_receipt(*row) for row in schemas):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_provider_receipt_digest_invalid"]
        )
    allowed_ids = set(limits.get("allowed_active_instance_ids") or [])
    pre_ids = _vast_inventory_instance_ids(pre_response, role="prelaunch")
    zero_ids = _vast_inventory_instance_ids(zero_response, role="provider_zero")
    if (
        pre.get("provider") != "vast"
        or pre.get("response_sha256") != _sha256_bytes(pre_response)
        or pre.get("parser_id")
        != "blueprint_pipeline.deformable_native_canary_bundle:vast_inventory_response:v1"
        or set(pre.get("active_instance_ids") or []) != pre_ids
        or not pre_ids.issubset(allowed_ids)
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_prelaunch_inventory_invalid"]
        )
    instance_id = allocation.get("instance_id")
    if isinstance(instance_id, bool) or not isinstance(instance_id, int) or instance_id <= 0:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_allocation_instance_invalid"]
        )
    if (
        allocation.get("provider") != "vast"
        or allocation.get("run_id") != worker.get("run_id")
        or allocation.get("package_receipt_digest") != package_receipt.get("receipt_digest")
        or allocation.get("admission_sha256") != _sha256_bytes(artifacts["provider_admission"])
        or allocation.get("attempt_ordinal") != 1
        or allocation.get("retry_count") != 0
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_allocation_join_invalid"]
        )
    armed = _parse_timestamp(
        watchdog.get("armed_at"), error="deformable_native_canary_watchdog_time_invalid"
    )
    created = _parse_timestamp(
        allocation.get("created_at"), error="deformable_native_canary_allocation_time_invalid"
    )
    deadline = _parse_timestamp(
        watchdog.get("deadline_at"), error="deformable_native_canary_watchdog_time_invalid"
    )
    if (
        watchdog.get("instance_id") != instance_id
        or not armed < created < deadline
        or (deadline - created).total_seconds() > int(limits.get("ttl_seconds"))
        or watchdog.get("watchdog_seconds") != limits.get("watchdog_seconds")
        or isinstance(watchdog.get("watchdog_process_pid"), bool)
        or not isinstance(watchdog.get("watchdog_process_pid"), int)
        or watchdog.get("watchdog_process_pid", 0) <= 0
        or not _valid_digest(watchdog.get("watchdog_command_sha256"))
    ):
        raise DeformableNativeCanaryBundleError(["deformable_native_canary_watchdog_order_invalid"])
    progress_times = [
        _parse_timestamp(
            row.get("observed_at"), error="deformable_native_canary_watchdog_progress_invalid"
        )
        for row in _rows(watchdog.get("progress_events"))
    ]
    if (
        not progress_times
        or progress_times != sorted(set(progress_times))
        or progress_times[0] < created
        or progress_times[-1] > deadline
        or any(
            (right - left).total_seconds() > int(limits.get("watchdog_seconds"))
            for left, right in zip([created, *progress_times[:-1]], progress_times, strict=True)
        )
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_watchdog_progress_invalid"]
        )
    transmitted = {row.get("sha256") for row in _rows(upload.get("transmitted_artifacts"))}
    if transmitted != {package_receipt.get("bundle_sha256"), scene_ref.get("package_sha256")} or (
        upload.get("scene_disclosure_receipt_digest") != scene_ref.get("disclosure_receipt_digest")
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_upload_disclosure_join_invalid"]
        )
    disclosed_members = _rows(scene_disclosure.get("members"))
    disclosed_paths = [str(row.get("path") or "") for row in disclosed_members]
    raw_source_digests = scene_disclosure.get("raw_source_digests")
    if (
        set(scene_disclosure)
        != {
            "schema_version",
            "package_sha256",
            "provider_terms_receipt_digest",
            "output_rights_receipt_digest",
            "provider_training_permitted",
            "retention_mode",
            "raw_source_digests",
            "members",
            "receipt_digest",
        }
        or scene_disclosure.get("schema_version") != "deformable_native_canary_scene_disclosure.v1"
        or scene_disclosure.get("receipt_digest")
        != canonical_digest(scene_disclosure, digest_field="receipt_digest")
        or scene_disclosure.get("package_sha256") != scene_ref.get("package_sha256")
        or scene_disclosure.get("receipt_digest") != scene_ref.get("disclosure_receipt_digest")
        or not _valid_digest(scene_disclosure.get("provider_terms_receipt_digest"))
        or not _valid_digest(scene_disclosure.get("output_rights_receipt_digest"))
        or scene_disclosure.get("provider_training_permitted") is not False
        or scene_disclosure.get("retention_mode") != "bounded_ephemeral_private_processing"
        or not isinstance(raw_source_digests, list)
        or not raw_source_digests
        or len(raw_source_digests) != len(set(raw_source_digests))
        or any(not _valid_digest(value) for value in raw_source_digests)
        or not disclosed_members
        or len(disclosed_paths) != len(set(disclosed_paths))
        or any(
            set(row)
            != {
                "path",
                "content_kind",
                "sha256",
                "derivation_receipt_digest",
            }
            or row.get("content_kind")
            not in {"derived_scene_usd", "derived_collision_usd", "runtime_configuration"}
            or not _valid_digest(row.get("sha256"))
            or not _valid_digest(row.get("derivation_receipt_digest"))
            or row.get("sha256") in set(raw_source_digests or [])
            or PurePosixPath(str(row.get("path") or "")).is_absolute()
            or ".." in PurePosixPath(str(row.get("path") or "")).parts
            or PurePosixPath(str(row.get("path") or "")).suffix.lower()
            in {".ply", ".labels", ".structure"}
            for row in disclosed_members
        )
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_scene_disclosure_invalid"]
        )
    if upload.get("scene_disclosure_receipt_sha256") != _sha256_bytes(
        artifacts["scene_disclosure_receipt"]
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_upload_disclosure_bytes_mismatch"]
        )
    try:
        cost = float(billing_response.get("charged_usd"))
    except (TypeError, ValueError):
        cost = math.nan
    if (
        set(billing_response) != {"instance_id", "charged_usd"}
        or isinstance(billing_response.get("charged_usd"), bool)
        or not isinstance(billing_response.get("charged_usd"), (int, float))
        or billing.get("response_sha256") != _sha256_bytes(artifacts["billing_response"])
        or billing.get("parser_id")
        != "blueprint_pipeline.deformable_native_canary_bundle:vast_billing_response:v1"
        or billing.get("instance_id") != instance_id
        or billing_response.get("instance_id") != instance_id
        or billing.get("attempt_ordinal") != 1
        or billing.get("retry_count") != 0
        or not math.isfinite(cost)
        or cost < 0
        or cost > float(limits.get("attempt_spend_cap_usd")) + 1e-9
    ):
        raise DeformableNativeCanaryBundleError(["deformable_native_canary_billing_invalid"])
    teardown_requested = _parse_timestamp(
        teardown.get("requested_at"), error="deformable_native_canary_teardown_time_invalid"
    )
    teardown_confirmed = _parse_timestamp(
        teardown.get("confirmed_at"), error="deformable_native_canary_teardown_time_invalid"
    )
    zero_observed = _parse_timestamp(
        zero.get("observed_at"), error="deformable_native_canary_provider_zero_time_invalid"
    )
    if (
        teardown.get("instance_id") != instance_id
        or not created < teardown_requested <= teardown_confirmed <= zero_observed
        or teardown_confirmed > deadline
        or (teardown_requested - progress_times[-1]).total_seconds()
        > int(limits.get("watchdog_seconds"))
        or zero.get("provider") != "vast"
        or zero.get("response_sha256") != _sha256_bytes(zero_response)
        or zero.get("parser_id")
        != "blueprint_pipeline.deformable_native_canary_bundle:vast_inventory_response:v1"
        or set(zero.get("active_instance_ids") or []) != zero_ids
        or zero_ids
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_teardown_or_provider_zero_invalid"]
        )
    if manifest.get("instance_id") != instance_id:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_return_instance_join_invalid"]
        )
    return cost, instance_id, created, deadline, teardown_requested


def _typed_null_verification(
    *,
    package: Mapping[str, Any],
    trusted_execution: Mapping[str, Any],
    blockers: Sequence[str],
    return_manifest_digest: str | None = None,
    worker_contract_digest: str | None = None,
    instance_id: str | None = None,
    dynamic_gate_results: Sequence[Mapping[str, Any]] = (),
    payload_gate_evidence_satisfied: bool = False,
    lifecycle_artifact_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    retained_blockers = sorted(set(str(item) for item in blockers if str(item)))
    return_artifact = _mapping(trusted_execution.get("return_zip_artifact"))
    verification: dict[str, Any] = {
        "schema_version": RETURN_VERIFICATION_SCHEMA_VERSION,
        "status": "native_canary_return_retained_typed_null",
        "package_receipt_digest": package.get("receipt_digest"),
        "run_digest": package.get("run_digest"),
        "return_bundle_sha256": return_artifact.get("sha256"),
        "return_bundle_size_bytes": return_artifact.get("size_bytes"),
        "return_manifest_digest": return_manifest_digest,
        "worker_contract_digest": worker_contract_digest,
        "instance_id": instance_id,
        "charged_usd": None,
        "trusted_execution": dict(trusted_execution),
        "trusted_execution_structural_verified": bool(
            trusted_execution.get("structural_trust_verified")
        ),
        "lifecycle_artifact_digests": dict(lifecycle_artifact_digests or {}),
        "dynamic_gate_results": [dict(row) for row in dynamic_gate_results],
        "worker_payload_gate_evidence_satisfied": payload_gate_evidence_satisfied,
        "blockers": retained_blockers,
        "paid_null_retained": True,
        "explicit_refreeze_required_before_retry": any(
            item.startswith("native_gate_blocked:") for item in retained_blockers
        ),
        "native_backend_refreeze_admitted": False,
        "provider_zero_verified_from_empty_api_inventory": False,
        "claim_boundary": {
            "runner_signed_return_structure_verified": bool(
                trusted_execution.get("structural_trust_verified")
            ),
            "native_canary_execution_verified": False,
            "simready_asset_qualified": False,
            "visual_alignment_qualified": False,
            "physical_material_equivalence": False,
            "real_robot_performance": False,
        },
        "verification_digest": "",
    }
    verification["verification_digest"] = canonical_digest(
        verification, digest_field="verification_digest"
    )
    return verification


def verify_deformable_native_canary_return(
    *,
    package_receipt_path: str | Path,
    return_bundle_path: str | Path,
    trusted_execution_envelope_path: str | Path,
    expected_nonce: str,
    expected_worker_entrypoint: str,
    expected_worker_source_tree_digest: str,
    expected_worker_container_digest: str,
    expected_instance_id: str,
    expected_allocator_lifecycle_artifact_digests: Mapping[str, str],
) -> dict[str, Any]:
    """Parse worker payload only after runner-signed structural authorization.

    The runner signature binds exact returned ZIP bytes.  It does not establish
    provider lifecycle semantics or provider-zero, so this verifier cannot
    refreeze a backend until a separate verifier-owned lifecycle proof exists.
    """

    package = verify_deformable_native_canary_bundle(package_receipt_path)
    bundle_path = Path(str(package["bundle_path"])).expanduser().resolve()
    with zipfile.ZipFile(bundle_path) as archive:
        infos = _safe_archive_infos(
            archive, prefix="deformable_native_canary_bundle", maximum_size=_PACKAGE_LIMIT
        )
        worker, _disclosure = _verify_disclosure(archive, {info.filename for info in infos})

    trusted_execution = verify_trusted_execution_envelope(
        trusted_execution_envelope_path,
        return_zip_path=return_bundle_path,
        expected_nonce=expected_nonce,
        expected_run_digest=str(package.get("run_digest") or ""),
        expected_package_digest=str(package.get("bundle_sha256") or ""),
        expected_execution_request_digest=str(package.get("execution_request_digest") or ""),
        expected_worker_entrypoint=expected_worker_entrypoint,
        expected_worker_source_tree_digest=expected_worker_source_tree_digest,
        expected_worker_container_digest=expected_worker_container_digest,
        expected_instance_id=expected_instance_id,
        expected_allocator_lifecycle_artifact_digests=(
            expected_allocator_lifecycle_artifact_digests
        ),
    )
    if not trusted_execution.get("structural_trust_verified"):
        return _typed_null_verification(
            package=package,
            trusted_execution=trusted_execution,
            blockers=[
                f"trusted_execution_blocked:{item}"
                for item in _strings(trusted_execution.get("blockers"))
            ],
            worker_contract_digest=worker.get("worker_contract_digest"),
        )

    return_bytes, _return_path = _read_file_once_no_follow(
        return_bundle_path,
        label="return_bundle",
        maximum_size=_RETURN_LIMIT,
    )
    signed_return = _mapping(trusted_execution.get("return_zip_artifact"))
    if _sha256_bytes(return_bytes) != signed_return.get("sha256") or len(
        return_bytes
    ) != signed_return.get("size_bytes"):
        return _typed_null_verification(
            package=package,
            trusted_execution=trusted_execution,
            blockers=["trusted_execution_return_zip_changed_after_verification"],
            worker_contract_digest=worker.get("worker_contract_digest"),
        )

    manifest, artifacts, paths = _return_artifacts(return_bytes)
    structural_blockers: list[str] = []
    if (
        manifest.get("package_receipt_digest") != package.get("receipt_digest")
        or manifest.get("package_bundle_sha256") != package.get("bundle_sha256")
        or manifest.get("worker_contract_digest") != worker.get("worker_contract_digest")
        or manifest.get("run_id") != worker.get("run_id")
    ):
        structural_blockers.append("deformable_native_canary_return_package_join_invalid")
    if str(manifest.get("instance_id") or "") != expected_instance_id:
        structural_blockers.append("deformable_native_canary_return_instance_join_invalid")

    expected_lifecycle = dict(expected_allocator_lifecycle_artifact_digests)
    lifecycle_digests = {
        role: _sha256_bytes(artifacts[role])
        for role in _LIFECYCLE_ARTIFACT_ROLES
        if role in artifacts
    }
    if (
        set(expected_lifecycle) != _LIFECYCLE_ARTIFACT_ROLES
        or lifecycle_digests != expected_lifecycle
    ):
        structural_blockers.append("trusted_execution_allocator_lifecycle_artifact_join_invalid")

    scene_disclosure = _json_artifact(artifacts, "scene_disclosure_receipt")
    scene_reference = _mapping(
        next(
            row
            for row in _rows(worker.get("stages"))
            if row.get("stage_id") == "scene_bound_native_execution"
        ).get("scene_runtime_reference")
    )
    if scene_disclosure.get("receipt_digest") != scene_reference.get(
        "disclosure_receipt_digest"
    ) or _sha256_bytes(artifacts["scene_disclosure_receipt"]) != _mapping(
        worker.get("input_bindings")
    ).get("scene_runtime_disclosure_file_sha256"):
        structural_blockers.append("deformable_native_canary_return_scene_disclosure_join_invalid")
    if structural_blockers:
        return _typed_null_verification(
            package=package,
            trusted_execution=trusted_execution,
            blockers=structural_blockers,
            return_manifest_digest=str(manifest.get("manifest_digest") or ""),
            worker_contract_digest=worker.get("worker_contract_digest"),
            instance_id=str(manifest.get("instance_id") or ""),
            lifecycle_artifact_digests=lifecycle_digests,
        )

    blank = _json_artifact(artifacts, "blank_stage_report")
    scene = _json_artifact(artifacts, "scene_stage_report")
    blank_fields = {
        "schema_version",
        "stage_id",
        "worker_contract_digest",
        "started_at",
        "finished_at",
        "asset_readbacks",
        "device_execution",
        "reset_readback",
        "report_digest",
    }
    scene_fields = {
        "schema_version",
        "stage_id",
        "worker_contract_digest",
        "started_at",
        "finished_at",
        "worker_reported_success",
        "contact_readback",
        "deformable_settle_readback",
        "applied_parameter_readback",
        "entity_readbacks",
        "robot_readback",
        "receptacle_readback",
        "ik_readbacks",
        "policy_adapter_readbacks",
        "camera_readbacks",
        "report_digest",
    }
    if (
        set(blank) != blank_fields
        or set(scene) != scene_fields
        or not _report_digest_valid(blank, "deformable_native_canary_blank_stage_report.v1")
        or not _report_digest_valid(scene, "deformable_native_canary_scene_stage_report.v1")
        or blank.get("stage_id") != CANARY_STAGE_IDS[0]
        or scene.get("stage_id") != CANARY_STAGE_IDS[1]
        or blank.get("worker_contract_digest") != worker.get("worker_contract_digest")
        or scene.get("worker_contract_digest") != worker.get("worker_contract_digest")
    ):
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_stage_report_join_invalid"]
        )
    blank_start = _parse_timestamp(
        blank.get("started_at"), error="deformable_native_canary_stage_time_invalid"
    )
    blank_end = _parse_timestamp(
        blank.get("finished_at"), error="deformable_native_canary_stage_time_invalid"
    )
    scene_start = _parse_timestamp(
        scene.get("started_at"), error="deformable_native_canary_stage_time_invalid"
    )
    scene_end = _parse_timestamp(
        scene.get("finished_at"), error="deformable_native_canary_stage_time_invalid"
    )
    if not blank_start < blank_end <= scene_start < scene_end:
        raise DeformableNativeCanaryBundleError(["deformable_native_canary_stage_order_invalid"])
    cameras = _verify_camera_artifacts(artifacts, paths)
    try:
        native_gate_blockers = _gate_blockers(
            worker=worker, blank=blank, scene=scene, camera_artifacts=cameras
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DeformableNativeCanaryBundleError(
            ["deformable_native_canary_native_evidence_numeric_invalid"]
        ) from exc
    gate_results = [
        {
            "gate_id": gate_id,
            "status": (
                "blocked"
                if gate_id in native_gate_blockers
                else "signed_payload_satisfied_pending_independent_provider_proof"
            ),
        }
        for gate_id in DYNAMIC_NATIVE_GATE_IDS
    ]
    blockers = [
        _PROVIDER_PROOF_BLOCKER,
        *[f"native_gate_blocked:{gate_id}" for gate_id in native_gate_blockers],
    ]
    return _typed_null_verification(
        package=package,
        trusted_execution=trusted_execution,
        blockers=blockers,
        return_manifest_digest=manifest["manifest_digest"],
        worker_contract_digest=worker["worker_contract_digest"],
        instance_id=str(manifest["instance_id"]),
        dynamic_gate_results=gate_results,
        payload_gate_evidence_satisfied=not native_gate_blockers,
        lifecycle_artifact_digests=lifecycle_digests,
    )


__all__ = [
    "ADDITIONAL_DYNAMIC_NATIVE_GATES",
    "BUNDLE_FILENAME",
    "BUNDLE_RECEIPT_SCHEMA_VERSION",
    "CANARY_STAGE_IDS",
    "DISCLOSURE_FILENAME",
    "DISCLOSURE_SCHEMA_VERSION",
    "DYNAMIC_NATIVE_GATE_IDS",
    "DeformableNativeCanaryBundleError",
    "EXECUTION_REQUEST_SCHEMA_VERSION",
    "RECEIPT_FILENAME",
    "REQUIRED_CAMERA_ROLES",
    "RETURN_MANIFEST_FILENAME",
    "RETURN_MANIFEST_SCHEMA_VERSION",
    "RETURN_VERIFICATION_SCHEMA_VERSION",
    "WORKER_CONTRACT_FILENAME",
    "WORKER_CONTRACT_SCHEMA_VERSION",
    "build_deformable_native_canary_bundle",
    "materialize_deformable_native_canary_execution_request",
    "verify_deformable_native_canary_bundle",
    "verify_deformable_native_canary_return",
]
