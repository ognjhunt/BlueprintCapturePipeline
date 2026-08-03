"""Bind a local Polycam Developer Mode raw ZIP as derived source support.

This adapter inventories and hashes the original archive and every regular ZIP
member, then binds declared semantic lanes to exact member digests.  It never
extracts the archive, calls Polycam, or upgrades a provider export into
Blueprint Raw Contract evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest


POLYCAM_DEVELOPER_SOURCE_DECLARATION_SCHEMA_VERSION = (
    "polycam_developer_source_declaration.v1"
)
POLYCAM_DEVELOPER_SOURCE_PROFILE_SCHEMA_VERSION = "polycam_developer_source_profile.v1"
POLYCAM_DEVELOPER_SOURCE_PROFILE = "polycam_developer_mode_lidar_raw_zip"

MAX_ARCHIVE_BYTES = 50 * 1024 * 1024 * 1024
MAX_MEMBER_COUNT = 250_000
MAX_MEMBER_BYTES = 50 * 1024 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 500 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000.0
_CHUNK_BYTES = 1024 * 1024
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")

SEMANTIC_ROLES = {
    "source_rgb_frames",
    "source_video",
    "frame_timestamps",
    "camera_intrinsics",
    "camera_extrinsics",
    "depth",
    "confidence",
    "mesh_geometry",
    "mesh_info",
    "metric_units",
    "capture_identity",
    "device_identity",
    "provider_identity",
}
REQUIRED_ROLES = {
    "frame_timestamps",
    "camera_intrinsics",
    "camera_extrinsics",
    "depth",
    "confidence",
    "mesh_geometry",
    "mesh_info",
    "metric_units",
    "capture_identity",
    "device_identity",
    "provider_identity",
}
IDENTITY_ROLES = {"capture_identity", "device_identity", "provider_identity"}
_DECLARATION_FIELDS = {
    "schema_version",
    "source_profile",
    "provider_identity",
    "source_capture_identity",
    "provider_capture_identity",
    "provider_app_version",
    "provider_export_timestamp",
    "layout_profile",
    "capture_mode",
    "developer_mode_enabled",
    "blueprint_remote_upload_performed",
    "device_identity",
    "metric_units",
    "semantic_bindings",
}
_DEVICE_IDENTITY_FIELDS = {
    "manufacturer",
    "model",
    "operating_system",
    "lidar_capable",
}
_METRIC_UNIT_FIELDS = {"length_unit", "scale_to_meters", "authority"}

CLAIM_BOUNDARY: dict[str, Any] = {
    "provider_derived_support": True,
    "blueprint_raw_contract_truth": False,
    "blueprint_raw_contract_version": None,
    "encoder_attempt_evidence_present": False,
    "retained_frame_evidence_present": False,
    "provider_declared_metric_units_are_independent_scale_proof": False,
    "metric_scale_independently_proven": False,
    "metric_geometry_qualified": False,
    "collision_geometry_qualified": False,
    "isaac_compatibility_proven": False,
    "task_success_proven": False,
    "physical_success_proven": False,
    "deployment_readiness_proven": False,
    "remote_provider_calls_performed": False,
}


class PolycamDeveloperSourceProfileError(ValueError):
    """Stable fail-closed error for malformed or unsafe source archives."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(sorted(set(str(row) for row in blockers if str(row))))
        super().__init__(";".join(self.blockers))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _opaque_text(value: Any, *, field: str, maximum: int = 256) -> str:
    text = value.strip() if isinstance(value, str) else ""
    if (
        not text
        or len(text) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise PolycamDeveloperSourceProfileError([f"{field}_invalid"])
    return text


def _timestamp(value: Any) -> str:
    try:
        parsed = datetime.fromisoformat(_text(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PolycamDeveloperSourceProfileError(
            ["provider_export_timestamp_invalid"]
        ) from exc
    if parsed.tzinfo is None:
        raise PolycamDeveloperSourceProfileError(["provider_export_timestamp_invalid"])
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _implementation_digest() -> str:
    return _sha256_file(Path(__file__).resolve())


def _source_commit_sha(value: str | None) -> str:
    candidate = _text(value or os.getenv("BLUEPRINT_SOURCE_COMMIT")).lower()
    if not candidate:
        raise PolycamDeveloperSourceProfileError(["source_commit_sha_required"])
    if _SOURCE_COMMIT.fullmatch(candidate) is None:
        raise PolycamDeveloperSourceProfileError(["source_commit_sha_invalid"])
    return candidate


def _safe_member_name(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise PolycamDeveloperSourceProfileError(["archive_member_path_unsafe"])
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PolycamDeveloperSourceProfileError(["archive_member_path_unsafe"])
    normalized = path.as_posix()
    if normalized != value.rstrip("/"):
        raise PolycamDeveloperSourceProfileError(["archive_member_path_unsafe"])
    return normalized


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _validated_bindings(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise PolycamDeveloperSourceProfileError(["semantic_bindings_invalid"])
    unknown = sorted(set(str(key) for key in value) - SEMANTIC_ROLES)
    if unknown:
        raise PolycamDeveloperSourceProfileError(
            [f"semantic_role_unsupported:{role}" for role in unknown]
        )
    bindings: dict[str, list[str]] = {}
    for role in sorted(SEMANTIC_ROLES):
        members = value.get(role, [])
        if not isinstance(members, list) or not all(isinstance(row, str) for row in members):
            raise PolycamDeveloperSourceProfileError(
                [f"semantic_binding_invalid:{role}"]
            )
        try:
            normalized = sorted({_safe_member_name(row) for row in members})
        except PolycamDeveloperSourceProfileError as exc:
            raise PolycamDeveloperSourceProfileError(
                [f"semantic_binding_member_unsafe:{role}"]
            ) from exc
        bindings[role] = normalized
    return bindings


def compile_polycam_developer_source_declaration(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonicalize one user-managed Polycam source declaration."""

    if not isinstance(value, Mapping):
        raise PolycamDeveloperSourceProfileError(["source_declaration_invalid"])
    try:
        declaration = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise PolycamDeveloperSourceProfileError(
            ["source_declaration_not_json"]
        ) from exc
    if not isinstance(declaration, dict):
        raise PolycamDeveloperSourceProfileError(["source_declaration_invalid"])

    supplied_digest = declaration.pop("source_declaration_digest", None)
    unknown_fields = sorted(set(declaration) - _DECLARATION_FIELDS)
    if unknown_fields:
        raise PolycamDeveloperSourceProfileError(
            [f"source_declaration_field_unsupported:{field}" for field in unknown_fields]
        )
    if declaration.get("schema_version") != POLYCAM_DEVELOPER_SOURCE_DECLARATION_SCHEMA_VERSION:
        raise PolycamDeveloperSourceProfileError(["source_declaration_schema_invalid"])
    if declaration.get("source_profile") != POLYCAM_DEVELOPER_SOURCE_PROFILE:
        raise PolycamDeveloperSourceProfileError(["source_profile_invalid"])
    if declaration.get("provider_identity") != "polycam":
        raise PolycamDeveloperSourceProfileError(["provider_identity_invalid"])
    if declaration.get("capture_mode") != "space_lidar":
        raise PolycamDeveloperSourceProfileError(["capture_mode_invalid"])
    if declaration.get("developer_mode_enabled") is not True:
        raise PolycamDeveloperSourceProfileError(["developer_mode_not_attested"])
    if declaration.get("blueprint_remote_upload_performed") is not False:
        raise PolycamDeveloperSourceProfileError(
            ["blueprint_remote_upload_declaration_invalid"]
        )
    for field in (
        "source_capture_identity",
        "provider_capture_identity",
        "provider_app_version",
        "layout_profile",
    ):
        declaration[field] = _opaque_text(declaration.get(field), field=field)
    declaration["provider_export_timestamp"] = _timestamp(
        declaration.get("provider_export_timestamp")
    )

    device = declaration.get("device_identity")
    if not isinstance(device, Mapping):
        raise PolycamDeveloperSourceProfileError(["device_identity_invalid"])
    normalized_device = dict(device)
    unknown_device_fields = sorted(set(normalized_device) - _DEVICE_IDENTITY_FIELDS)
    if unknown_device_fields:
        raise PolycamDeveloperSourceProfileError(
            [f"device_identity_field_unsupported:{field}" for field in unknown_device_fields]
        )
    if _text(normalized_device.get("manufacturer")).lower() != "apple":
        raise PolycamDeveloperSourceProfileError(["device_manufacturer_invalid"])
    normalized_device["manufacturer"] = "Apple"
    normalized_device["model"] = _opaque_text(
        normalized_device.get("model"), field="device_model"
    )
    normalized_device["operating_system"] = _opaque_text(
        normalized_device.get("operating_system"), field="device_operating_system"
    )
    if normalized_device.get("lidar_capable") is not True:
        raise PolycamDeveloperSourceProfileError(["device_lidar_capability_invalid"])
    declaration["device_identity"] = normalized_device

    metric_units = declaration.get("metric_units")
    if not isinstance(metric_units, Mapping):
        raise PolycamDeveloperSourceProfileError(["metric_units_invalid"])
    normalized_units = dict(metric_units)
    unknown_metric_fields = sorted(set(normalized_units) - _METRIC_UNIT_FIELDS)
    if unknown_metric_fields:
        raise PolycamDeveloperSourceProfileError(
            [f"metric_units_field_unsupported:{field}" for field in unknown_metric_fields]
        )
    if normalized_units.get("authority") not in {
        None,
        "provider_declared_unqualified",
    }:
        raise PolycamDeveloperSourceProfileError(["metric_units_authority_invalid"])
    unit = _text(normalized_units.get("length_unit")).lower()
    if unit not in {"m", "meter", "meters", "metre", "metres"}:
        raise PolycamDeveloperSourceProfileError(["metric_length_unit_invalid"])
    scale = normalized_units.get("scale_to_meters")
    if isinstance(scale, bool):
        raise PolycamDeveloperSourceProfileError(["metric_scale_to_meters_invalid"])
    try:
        scale_number = float(scale)
    except (TypeError, ValueError) as exc:
        raise PolycamDeveloperSourceProfileError(
            ["metric_scale_to_meters_invalid"]
        ) from exc
    if not math.isfinite(scale_number) or scale_number != 1.0:
        raise PolycamDeveloperSourceProfileError(["metric_scale_to_meters_invalid"])
    normalized_units["length_unit"] = "meter"
    normalized_units["scale_to_meters"] = 1.0
    normalized_units["authority"] = "provider_declared_unqualified"
    declaration["metric_units"] = normalized_units
    declaration["semantic_bindings"] = _validated_bindings(
        declaration.get("semantic_bindings")
    )

    expected_digest = canonical_digest(declaration)
    if supplied_digest is not None and supplied_digest != expected_digest:
        raise PolycamDeveloperSourceProfileError(["source_declaration_digest_mismatch"])
    declaration["source_declaration_digest"] = expected_digest
    return declaration


def _inspect_archive(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.is_symlink():
        raise PolycamDeveloperSourceProfileError(["source_archive_symlink_forbidden"])
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise PolycamDeveloperSourceProfileError(["source_archive_invalid"])
    archive_size = path.stat().st_size
    if archive_size <= 0 or archive_size > MAX_ARCHIVE_BYTES:
        raise PolycamDeveloperSourceProfileError(["source_archive_size_invalid"])
    archive_digest = _sha256_file(path)
    inventory: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBER_COUNT:
                raise PolycamDeveloperSourceProfileError(
                    ["source_archive_member_count_exceeded"]
                )
            seen: set[str] = set()
            total_uncompressed = 0
            for info in infos:
                name = _safe_member_name(info.filename)
                if name in seen:
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_duplicate_member"]
                    )
                seen.add(name)
                if info.is_dir():
                    continue
                if _zip_member_is_symlink(info):
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_symlink_member_forbidden"]
                    )
                if info.flag_bits & 0x1:
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_encrypted_member_forbidden"]
                    )
                if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_member_size_exceeded"]
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_uncompressed_size_exceeded"]
                    )
                ratio = (
                    float(info.file_size) / float(info.compress_size)
                    if info.compress_size > 0
                    else (0.0 if info.file_size == 0 else math.inf)
                )
                if not math.isfinite(ratio) or ratio > MAX_COMPRESSION_RATIO:
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_compression_ratio_exceeded"]
                    )
                member_digest = hashlib.sha256()
                bytes_read = 0
                with archive.open(info, "r") as stream:
                    for chunk in iter(lambda: stream.read(_CHUNK_BYTES), b""):
                        bytes_read += len(chunk)
                        if bytes_read > info.file_size or bytes_read > MAX_MEMBER_BYTES:
                            raise PolycamDeveloperSourceProfileError(
                                ["source_archive_member_stream_size_mismatch"]
                            )
                        member_digest.update(chunk)
                if bytes_read != info.file_size:
                    raise PolycamDeveloperSourceProfileError(
                        ["source_archive_member_stream_size_mismatch"]
                    )
                inventory.append(
                    {
                        "member_path": name,
                        "size_bytes": info.file_size,
                        "compressed_size_bytes": info.compress_size,
                        "crc32": f"{info.CRC:08x}",
                        "sha256": f"sha256:{member_digest.hexdigest()}",
                    }
                )
    except zipfile.BadZipFile:
        raise PolycamDeveloperSourceProfileError(["source_archive_bad_zip"])
    return archive_digest, sorted(inventory, key=lambda row: row["member_path"])


def _smallest_missing_measurement(blockers: Sequence[str]) -> dict[str, str] | None:
    ordered = [
        ("source_appearance_missing", "Export at least one original RGB frame or source video member."),
        ("semantic_lane_missing:frame_timestamps", "Bind the per-frame timestamp metadata member or members."),
        ("semantic_lane_missing:camera_intrinsics", "Bind the camera intrinsics metadata member or members."),
        ("semantic_lane_missing:camera_extrinsics", "Bind the camera extrinsics metadata member or members."),
        ("semantic_lane_missing:depth", "Export and bind the LiDAR depth member or members."),
        ("semantic_lane_missing:confidence", "Export and bind the depth-confidence member or members."),
        ("semantic_lane_missing:mesh_geometry", "Export and bind the raw mesh geometry member."),
        ("semantic_lane_missing:mesh_info", "Export and bind the mesh metadata member."),
        ("semantic_lane_missing:metric_units", "Bind provider metadata that declares the mesh length unit."),
        ("semantic_lane_missing:capture_identity", "Bind provider metadata containing the capture identity."),
        ("semantic_lane_missing:device_identity", "Bind provider metadata containing the device identity."),
        ("semantic_lane_missing:provider_identity", "Bind provider metadata containing the provider identity."),
    ]
    blocker_set = set(blockers)
    for code, instruction in ordered:
        if code in blocker_set:
            return {"code": code, "instruction": instruction}
    if blockers:
        return {"code": sorted(blocker_set)[0], "instruction": "Correct the declared member binding and replay the unchanged archive."}
    return None


def build_polycam_developer_source_profile(
    *,
    archive_path: str | Path,
    declaration: Mapping[str, Any],
    source_commit_sha: str | None = None,
    implementation_digest: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, proof-bounded source profile for one raw ZIP."""

    compiled_declaration = compile_polycam_developer_source_declaration(declaration)
    # ``resolve`` would dereference the final component before the archive
    # safety check and make a caller-supplied symlink indistinguishable from
    # its target.  ``abspath`` normalizes the location without following it.
    archive = Path(os.path.abspath(str(Path(archive_path).expanduser())))
    archive_digest, inventory = _inspect_archive(archive)
    by_path = {str(row["member_path"]): row for row in inventory}
    bindings = compiled_declaration["semantic_bindings"]
    blockers: list[str] = []
    if not bindings["source_rgb_frames"] and not bindings["source_video"]:
        blockers.append("source_appearance_missing")
    for role in sorted(REQUIRED_ROLES):
        if not bindings[role]:
            blockers.append(f"semantic_lane_missing:{role}")
    bound_paths = {path for paths in bindings.values() for path in paths}
    missing_paths = sorted(bound_paths - set(by_path))
    blockers.extend(f"declared_member_missing:{path}" for path in missing_paths)

    semantic_receipt: dict[str, list[dict[str, str]]] = {}
    roles_by_path: dict[str, list[str]] = {path: [] for path in by_path}
    for role in sorted(SEMANTIC_ROLES):
        rows: list[dict[str, str]] = []
        for path in bindings[role]:
            member = by_path.get(path)
            if member is None:
                continue
            roles_by_path[path].append(role)
            rows.append({"member_path": path, "sha256": str(member["sha256"])})
        semantic_receipt[role] = rows
    for row in inventory:
        row["semantic_roles"] = sorted(roles_by_path[str(row["member_path"])])

    blockers = sorted(set(blockers))
    warnings: list[str] = []
    if not bindings["source_video"]:
        warnings.append("source_video_not_present_rgb_frames_bound")
    if not bindings["source_rgb_frames"]:
        warnings.append("source_rgb_frames_not_present_video_bound")
    unbound = sorted(set(by_path) - bound_paths)
    if unbound:
        warnings.append("archive_contains_unbound_members")
    status = "admitted_provider_derived_support" if not blockers else "abstained"
    resolved_implementation_digest = implementation_digest or _implementation_digest()
    if _SHA256.fullmatch(_text(resolved_implementation_digest)) is None:
        raise PolycamDeveloperSourceProfileError(["implementation_digest_invalid"])
    source_commit = _source_commit_sha(source_commit_sha)

    profile = {
        "schema_version": POLYCAM_DEVELOPER_SOURCE_PROFILE_SCHEMA_VERSION,
        "source_profile": POLYCAM_DEVELOPER_SOURCE_PROFILE,
        "status": status,
        "provider_identity": "polycam",
        "source_capture_identity": compiled_declaration["source_capture_identity"],
        "provider_capture_identity": compiled_declaration["provider_capture_identity"],
        "device_identity": compiled_declaration["device_identity"],
        "provider_app_version": compiled_declaration["provider_app_version"],
        "provider_export_timestamp": compiled_declaration["provider_export_timestamp"],
        "layout_profile": compiled_declaration["layout_profile"],
        "source_declaration_digest": compiled_declaration["source_declaration_digest"],
        "source_archive": {
            "original_filename": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": archive_digest,
            "member_count": len(inventory),
            "member_set_digest": canonical_digest({"members": inventory}),
            "original_archive_preserved": True,
            "archive_extracted_by_adapter": False,
        },
        "member_inventory": inventory,
        "semantic_bindings": semantic_receipt,
        "unbound_members": unbound,
        "metric_units": compiled_declaration["metric_units"],
        "identity_binding": {
            "capture_identity": compiled_declaration["source_capture_identity"],
            "provider_capture_identity": compiled_declaration["provider_capture_identity"],
            "provider_identity": "polycam",
            "device_identity": compiled_declaration["device_identity"],
            "source_declaration_digest": compiled_declaration["source_declaration_digest"],
            "archive_identity_member_digests": sorted(
                {
                    row["sha256"]
                    for role in IDENTITY_ROLES
                    for row in semantic_receipt[role]
                }
            ),
        },
        "producing_method": "deterministic_polycam_developer_raw_zip_adapter.v1",
        "implementation_digest": resolved_implementation_digest,
        "source_commit_sha": source_commit,
        "blockers": blockers,
        "warnings": sorted(warnings),
        "smallest_missing_measurement": _smallest_missing_measurement(blockers),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "proof_effect": "provider_source_profile_binding_only" if not blockers else "none",
        "claim_ceiling": "provider_derived_capture_support" if not blockers else "none",
        "legal_next_actions": (
            [
                "compile_provider_derived_reconstruction_dataset",
                "qualify_metric_scale_independently",
                "preserve_original_archive",
            ]
            if not blockers
            else ["preserve_original_archive", "supply_smallest_missing_measurement"]
        ),
    }
    profile["source_profile_digest"] = canonical_digest(
        profile, digest_field="source_profile_digest"
    )
    return profile


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PolycamDeveloperSourceProfileError(
            ["source_declaration_file_invalid"]
        ) from exc
    if not isinstance(value, dict):
        raise PolycamDeveloperSourceProfileError(["source_declaration_file_invalid"])
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise PolycamDeveloperSourceProfileError(["source_profile_output_symlink_forbidden"])
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise PolycamDeveloperSourceProfileError(["source_profile_output_conflict"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind a local Polycam Developer Mode raw ZIP as derived source support."
    )
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--declaration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-commit-sha",
        help="Immutable 40-hex repository commit; defaults to BLUEPRINT_SOURCE_COMMIT.",
    )
    args = parser.parse_args(argv)
    profile = build_polycam_developer_source_profile(
        archive_path=args.archive,
        declaration=_load_json(args.declaration),
        source_commit_sha=args.source_commit_sha,
    )
    _write_json(args.output, profile)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_profile_digest": profile["source_profile_digest"],
                "status": profile["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if profile["status"] == "admitted_provider_derived_support" else 2


__all__ = [
    "CLAIM_BOUNDARY",
    "POLYCAM_DEVELOPER_SOURCE_DECLARATION_SCHEMA_VERSION",
    "POLYCAM_DEVELOPER_SOURCE_PROFILE",
    "POLYCAM_DEVELOPER_SOURCE_PROFILE_SCHEMA_VERSION",
    "PolycamDeveloperSourceProfileError",
    "build_polycam_developer_source_profile",
    "compile_polycam_developer_source_declaration",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
