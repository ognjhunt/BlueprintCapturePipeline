"""CPU-only scene asset inspection for pre-GPU robot-eval setup.

This lane inspects local scene assets and writes conservative frame/proof
manifests. It never downloads assets, runs simulators, calls providers, or
marks collision/contact proof complete.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import struct
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from xml.etree import ElementTree as ET

from .common import PipelineError, ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


SCENE_ASSET_INSPECTION_SCHEMA_VERSION = "scene_asset_inspection.v1"
SCENE_FRAME_ESTIMATE_SCHEMA_VERSION = "scene_frame_estimate.v1"
CPU_PREFLIGHT_SCORECARD_SCHEMA_VERSION = "cpu_preflight_scorecard.v1"
SCENE_ASSET_INVENTORY_SCHEMA_VERSION = "scene_asset_inventory.v1"
SCENE_ASSET_DEPENDENCY_AUDIT_SCHEMA_VERSION = "scene_asset_dependency_audit.v1"
SCENE_ASSET_PREFLIGHT_SCHEMA_VERSION = "scene_asset_preflight.v1"
COLLIDER_PROXY_PLAN_SCHEMA_VERSION = "collider_proxy_plan.v1"
CPU_SCENE_PROXY_MANIFEST_SCHEMA_VERSION = "cpu_scene_proxy_manifest.v1"

SUPPORTED_SCENE_ASSET_SUFFIXES = {
    ".ply",
    ".usd",
    ".usda",
    ".usdc",
    ".glb",
    ".gltf",
    ".obj",
    ".urdf",
    ".mjcf",
    ".xml",
}
REMOTE_REF_PREFIXES = ("gs://", "http://", "https://", "s3://", "omniverse://")
TEXTURE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".exr",
    ".hdr",
    ".ktx",
    ".ktx2",
}
OWNER_SYSTEM_MATERIAL_SUFFIXES = {".mdl"}
OWNER_SYSTEM_MATERIAL_REFS = {"omnipbr.mdl"}

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "cpu_pre_gpu_scene_asset_preflight_only",
    "repo_local_only": True,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "gpu_required": False,
    "simulators_run": False,
    "simulator_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "non_ranking_operational_claim_validated": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "simulator_execution_completed",
        "robot_ready",
        "deployment_ready",
        "physics_contact_validated",
        "non_ranking_operational_claim_validated",
    ],
}

_PLY_SCALAR_TYPES: Dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _relative_if_file(base_dir: Path, target: Path) -> str | None:
    return _relative_to(base_dir, target) if target.is_file() else None


def _sha_payload(payload: Mapping[str, Any]) -> str:
    return sha256(
        repr(sorted(payload.items())).encode("utf-8", errors="replace")
    ).hexdigest()


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return "usd"
    if suffix == ".gltf":
        return "gltf"
    if suffix == ".glb":
        return "glb"
    if suffix == ".obj":
        return "obj"
    if suffix == ".urdf":
        return "urdf"
    if suffix == ".mjcf":
        return "mjcf"
    if suffix == ".xml":
        try:
            root = ET.parse(path).getroot()
            tag = root.tag.split("}")[-1].lower()
            if tag == "robot":
                return "urdf"
            if tag == "mujoco":
                return "mjcf"
        except Exception:
            return "xml"
    return suffix.lstrip(".") or "unknown"


def _is_remote_ref(value: str) -> bool:
    return value.lower().startswith(REMOTE_REF_PREFIXES)


def _path_suffix_from_ref(value: str) -> str:
    cleaned = value.split("?", 1)[0].split("#", 1)[0]
    return Path(cleaned).suffix.lower()


def _looks_like_supported_asset_ref(value: str) -> bool:
    suffix = _path_suffix_from_ref(value)
    return suffix in SUPPORTED_SCENE_ASSET_SUFFIXES or suffix in TEXTURE_SUFFIXES


def _resolve_local_ref(source_path: Path, ref: str) -> Path | None:
    text = _string(ref)
    if not text or _is_remote_ref(text) or any(char in text for char in ("\x00", "\r", "\n")):
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (source_path.parent / path).resolve()


def _is_owner_system_material_library_ref(ref: str, relationship: str) -> bool:
    suffix = _path_suffix_from_ref(ref)
    basename = Path(ref.split("?", 1)[0].split("#", 1)[0]).name.lower()
    return (
        suffix in OWNER_SYSTEM_MATERIAL_SUFFIXES
        or basename in OWNER_SYSTEM_MATERIAL_REFS
        or relationship == "owner_system_material_library"
    )


def _dependency_record(
    *,
    source_path: Path,
    ref: str,
    relationship: str,
    line_number: int | None = None,
) -> Dict[str, Any]:
    local_path = _resolve_local_ref(source_path, ref)
    exists = bool(local_path and local_path.is_file())
    owner_system_material_ref = _is_owner_system_material_library_ref(ref, relationship)
    missing_local_file = bool(local_path and not local_path.is_file())
    record = {
        "ref": ref,
        "relationship": relationship,
        "line_number": line_number,
        "remote_ref": _is_remote_ref(ref),
        "local_path": str(local_path) if local_path else None,
        "exists_local": exists,
        "missing_local_file": missing_local_file,
        "hard_missing_local_file": bool(missing_local_file and not owner_system_material_ref),
        "owner_system_material_library_ref": owner_system_material_ref,
        "unresolved_locally": _is_remote_ref(ref) or missing_local_file,
        "size_bytes": local_path.stat().st_size if exists and local_path else None,
        "sha256": _sha_file(local_path) if exists and local_path else None,
    }
    if record["remote_ref"]:
        record["warning"] = "remote_dependency_not_downloaded_by_cpu_preflight"
    elif owner_system_material_ref and record["missing_local_file"]:
        record["warning"] = "owner_system_material_library_not_local"
    elif record["missing_local_file"]:
        record["warning"] = "missing_local_dependency"
    return record


def _file_inventory_record(path: Path, *, source: str = "discovered_local_file") -> Dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "asset_type": _asset_type_for_path(path) if exists else path.suffix.lower().lstrip("."),
        "source": source,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": _sha_file(path) if exists else None,
    }


def _walk_payload_strings(value: Any) -> List[str]:
    out: List[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            out.extend(_walk_payload_strings(item))
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            out.extend(_walk_payload_strings(item))
    return out


def _finite_float_list(value: Any) -> List[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    out: List[float] = []
    for item in list(value)[:3]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _bounds_from_points(points: Sequence[Sequence[float]]) -> Dict[str, Any] | None:
    clean = []
    for point in points:
        xyz = _finite_float_list(point)
        if xyz is not None:
            clean.append(xyz)
    if not clean:
        return None
    mins = [min(point[index] for point in clean) for index in range(3)]
    maxs = [max(point[index] for point in clean) for index in range(3)]
    centroid = [
        sum(point[index] for point in clean) / float(len(clean))
        for index in range(3)
    ]
    return {
        "bounds": {"min": mins, "max": maxs},
        "centroid": centroid,
        "floor_z_estimate": _percentile([point[2] for point in clean], 0.02),
        "sampled_point_count": len(clean),
    }


def _collision_evidence(
    *,
    real_collider_proven: bool,
    proxy_estimated: bool = False,
    portable_collider_glb_present: bool = False,
    status: str | None = None,
    evidence: Sequence[str] = (),
) -> Dict[str, Any]:
    missing = not real_collider_proven
    return {
        "real_collider_proven": bool(real_collider_proven),
        "proxy_estimated": bool(proxy_estimated),
        "missing_collider": missing,
        "review_required": True,
        "portable_collider_glb_present": bool(portable_collider_glb_present),
        "cpu_proxy_collision_estimated": bool(proxy_estimated),
        "collision_ready_claim_allowed": False,
        "status": status
        or ("real_collider_metadata_present" if real_collider_proven else "missing_collider"),
        "evidence": list(evidence),
        "proof_boundary": "collider metadata or proxy planning is not physics/contact validation",
    }


def _semantic_hints_from_names(names: Sequence[str], *, source: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for name in names:
        text = _string(name)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append({"label": text, "source": source})
    return out[:100]


def _has_collider_name_token(value: str) -> bool:
    return bool(
        re.search(r"(^|[^a-z0-9])(collider|collision|physics)($|[^a-z0-9])", value.lower())
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = max(0.0, min(1.0, percentile)) * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    frac = index - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _ply_header(path: Path) -> tuple[List[str], int]:
    with path.open("rb") as handle:
        raw = handle.read(128 * 1024)
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = raw.find(marker)
    if end < 0:
        raise ValueError("PLY header missing end_header")
    header_end = end + len(marker)
    header_text = raw[:header_end].decode("ascii", errors="replace")
    return header_text.splitlines(), header_end


def _parse_ply_header(lines: Sequence[str]) -> Dict[str, Any]:
    fmt = "unknown"
    elements: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current)
        elif parts[0] == "property" and current is not None and len(parts) >= 3:
            if parts[1] == "list" and len(parts) >= 5:
                current["properties"].append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "value_type": parts[3],
                        "name": parts[4],
                    }
                )
            else:
                current["properties"].append(
                    {"kind": "scalar", "type": parts[1], "name": parts[2]}
                )
    return {"format": fmt, "elements": elements}


def _ply_scalar_size(type_name: str) -> int:
    return _PLY_SCALAR_TYPES.get(type_name, ("", 0))[1]


def _ply_record_size(properties: Sequence[Mapping[str, Any]]) -> int | None:
    total = 0
    for prop in properties:
        if prop.get("kind") != "scalar":
            return None
        total += _ply_scalar_size(_string(prop.get("type")))
    return total


def _unpack_scalar(raw: bytes, offset: int, type_name: str, endian: str) -> tuple[Any, int]:
    fmt, size = _PLY_SCALAR_TYPES[type_name]
    if not fmt or size <= 0:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return struct.unpack_from(endian + fmt, raw, offset)[0], offset + size


def _inspect_ascii_ply(path: Path, parsed: Mapping[str, Any], header_end: int) -> Dict[str, Any]:
    elements = [dict(item) for item in parsed.get("elements", []) if isinstance(item, Mapping)]
    vertex = next((item for item in elements if item.get("name") == "vertex"), {})
    properties = [dict(item) for item in vertex.get("properties", []) if isinstance(item, Mapping)]
    property_names = [_string(item.get("name")) for item in properties]
    xyz_indexes = [property_names.index(axis) if axis in property_names else -1 for axis in ("x", "y", "z")]
    if min(xyz_indexes) < 0:
        return {
            "bounds": None,
            "centroid": None,
            "floor_z_estimate": None,
            "estimate_method": "ascii_header_only_no_xyz_properties",
            "confidence": "low",
        }
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    totals = [0.0, 0.0, 0.0]
    z_values: List[float] = []
    count = 0
    with path.open("rb") as handle:
        handle.seek(header_end)
        for raw_line in handle:
            parts = raw_line.decode("utf-8", errors="ignore").split()
            if len(parts) <= max(xyz_indexes):
                continue
            try:
                point = [float(parts[index]) for index in xyz_indexes]
            except ValueError:
                continue
            for axis, value in enumerate(point):
                mins[axis] = min(mins[axis], value)
                maxs[axis] = max(maxs[axis], value)
                totals[axis] += value
            z_values.append(point[2])
            count += 1
            if count >= 200_000:
                break
    if count == 0:
        return {
            "bounds": None,
            "centroid": None,
            "floor_z_estimate": None,
            "estimate_method": "ascii_no_points_read",
            "confidence": "low",
        }
    return {
        "bounds": {"min": mins, "max": maxs},
        "centroid": [value / count for value in totals],
        "floor_z_estimate": _percentile(z_values, 0.02),
        "sampled_point_count": count,
        "estimate_method": "ascii_vertex_xyz_sample",
        "confidence": "medium",
    }


def _inspect_binary_chunk_bounds(
    path: Path,
    parsed: Mapping[str, Any],
    header_end: int,
) -> Dict[str, Any] | None:
    elements = [dict(item) for item in parsed.get("elements", []) if isinstance(item, Mapping)]
    if not elements:
        return None
    first = elements[0]
    if first.get("name") != "chunk":
        return None
    properties = [dict(item) for item in first.get("properties", []) if isinstance(item, Mapping)]
    names = [_string(item.get("name")) for item in properties]
    required = ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"]
    if not all(name in names for name in required):
        return None
    size = _ply_record_size(properties)
    if size is None:
        return None
    fmt = _string(parsed.get("format"))
    endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else "<"
    chunk_count = int(first.get("count") or 0)
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    centers = [0.0, 0.0, 0.0]
    min_z_values: List[float] = []
    with path.open("rb") as handle:
        handle.seek(header_end)
        for _ in range(chunk_count):
            raw = handle.read(size)
            if len(raw) != size:
                break
            offset = 0
            record: Dict[str, Any] = {}
            for prop in properties:
                value, offset = _unpack_scalar(raw, offset, _string(prop.get("type")), endian)
                record[_string(prop.get("name"))] = value
            low = [float(record["min_x"]), float(record["min_y"]), float(record["min_z"])]
            high = [float(record["max_x"]), float(record["max_y"]), float(record["max_z"])]
            for axis in range(3):
                mins[axis] = min(mins[axis], low[axis])
                maxs[axis] = max(maxs[axis], high[axis])
                centers[axis] += (low[axis] + high[axis]) * 0.5
            min_z_values.append(low[2])
    if not min_z_values:
        return None
    return {
        "bounds": {"min": mins, "max": maxs},
        "centroid": [value / len(min_z_values) for value in centers],
        "floor_z_estimate": _percentile(min_z_values, 0.02),
        "sampled_chunk_count": len(min_z_values),
        "estimate_method": "binary_chunk_min_max_bounds",
        "confidence": "medium",
        "limitations": [
            "Binary splat PLY used chunk-level bounds rather than decoded per-vertex positions.",
        ],
    }


def inspect_ply_asset(path: Path) -> Dict[str, Any]:
    lines, header_end = _ply_header(path)
    parsed = _parse_ply_header(lines)
    elements = [dict(item) for item in parsed.get("elements", []) if isinstance(item, Mapping)]
    element_counts = {_string(item.get("name")): int(item.get("count") or 0) for item in elements}
    fmt = _string(parsed.get("format"))
    estimate: Dict[str, Any]
    if fmt == "ascii":
        estimate = _inspect_ascii_ply(path, parsed, header_end)
    elif fmt in {"binary_little_endian", "binary_big_endian"}:
        estimate = _inspect_binary_chunk_bounds(path, parsed, header_end) or {
            "bounds": None,
            "centroid": None,
            "floor_z_estimate": None,
            "estimate_method": "binary_header_only_no_decoded_xyz",
            "confidence": "low",
            "limitations": [
                "Binary PLY did not expose xyz float vertices or chunk min/max bounds.",
            ],
        }
    else:
        estimate = {
            "bounds": None,
            "centroid": None,
            "floor_z_estimate": None,
            "estimate_method": "unsupported_ply_format",
            "confidence": "low",
        }
    return {
        "asset_type": "ply",
        "path": str(path.resolve()),
        "format": fmt,
        "header": {
            "element_counts": element_counts,
            "properties_by_element": {
                _string(item.get("name")): [
                    _string(prop.get("name"))
                    for prop in item.get("properties", [])
                    if isinstance(prop, Mapping)
                ]
                for item in elements
            },
        },
        "vertex_count": element_counts.get("vertex"),
        "bounds": estimate.get("bounds"),
        "centroid": estimate.get("centroid"),
        "floor_z_estimate": estimate.get("floor_z_estimate"),
        "estimate_method": estimate.get("estimate_method"),
        "confidence": estimate.get("confidence"),
        "limitations": [
            "Point or splat visual assets do not prove collision readiness.",
            *list(estimate.get("limitations") or []),
        ],
        "dependencies": [],
        "semantic_hints": [],
        "collision_evidence": _collision_evidence(
            real_collider_proven=False,
            proxy_estimated=bool(estimate.get("bounds")),
            status="visual_only_ply_no_collider",
            evidence=["ply_visual_points_or_splats"],
        ),
    }


def _usd_dependency_relationship(line: str, ref: str) -> str:
    lower = line.lower()
    suffix = _path_suffix_from_ref(ref)
    if suffix in OWNER_SYSTEM_MATERIAL_SUFFIXES:
        return "owner_system_material_library"
    if "sublayer" in lower:
        return "sublayer"
    if "payload" in lower:
        return "payload"
    if "reference" in lower:
        return "reference"
    if "inputs:file" in lower or "texture" in lower or suffix in TEXTURE_SUFFIXES:
        return "texture_or_material_asset"
    return "asset_reference"


def _extract_usd_dependencies(path: Path, text: str) -> List[Dict[str, Any]]:
    dependencies: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for ref in re.findall(r"@([^@\n]+)@", line):
            dependencies.append(
                _dependency_record(
                    source_path=path,
                    ref=ref,
                    relationship=_usd_dependency_relationship(line, ref),
                    line_number=line_number,
                )
            )
    seen: set[tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for dependency in dependencies:
        key = (_string(dependency.get("ref")), _string(dependency.get("relationship")))
        if key in seen:
            continue
        seen.add(key)
        out.append(dependency)
    return out


def _dedupe_dependencies(dependencies: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for dependency in dependencies:
        key = (
            _string(dependency.get("ref")),
            _string(dependency.get("relationship")),
            _string(dependency.get("local_path")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(dependency))
    return out


def _dependency_relationship_for_path(path_text: str, default: str) -> str:
    suffix = _path_suffix_from_ref(path_text)
    if suffix in TEXTURE_SUFFIXES:
        return "texture_or_material_asset"
    if suffix in OWNER_SYSTEM_MATERIAL_SUFFIXES:
        return "owner_system_material_library"
    if suffix in {".usd", ".usda", ".usdc"}:
        return "usd_layer_or_reference"
    return default


def _extract_openusd_dependencies(path: Path) -> List[Dict[str, Any]]:
    try:  # pragma: no cover - depends on optional pxr install.
        from pxr import UsdUtils
    except Exception:
        return []
    try:  # pragma: no cover - depends on optional pxr install.
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    except Exception:
        return []
    dependencies: List[Dict[str, Any]] = []
    source_resolved = str(path.resolve())
    for layer in layers:
        layer_path = _string(getattr(layer, "realPath", "") or getattr(layer, "identifier", ""))
        if not layer_path or layer_path == source_resolved:
            continue
        dependencies.append(
            _dependency_record(
                source_path=path,
                ref=layer_path,
                relationship="usd_layer_or_reference",
            )
        )
    for asset in assets:
        ref = _string(asset)
        if not ref:
            continue
        dependencies.append(
            _dependency_record(
                source_path=path,
                ref=ref,
                relationship=_dependency_relationship_for_path(ref, "openusd_asset_dependency"),
            )
        )
    for ref in unresolved:
        text = _string(ref)
        if not text:
            continue
        record = _dependency_record(
            source_path=path,
            ref=text,
            relationship=_dependency_relationship_for_path(text, "unresolved_openusd_dependency"),
        )
        record["warning"] = "unresolved_openusd_dependency"
        record["unresolved_locally"] = True
        dependencies.append(record)
    return _dedupe_dependencies(dependencies)


def _extract_usd_semantic_hints(text: str) -> List[Dict[str, str]]:
    names = re.findall(r'\b(?:def|over)\s+\w+\s+"([^"]+)"', text)
    names.extend(re.findall(r'\b(?:def|over)\s+"([^"]+)"', text))
    return _semantic_hints_from_names(names, source="usd_prim_name")


def _usd_dependency_summary(dependencies: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        "dependency_count": len(dependencies),
        "missing_local_file_count": sum(
            1 for item in dependencies if bool(item.get("missing_local_file"))
        ),
        "hard_missing_local_file_count": sum(
            1
            for item in dependencies
            if bool(item.get("hard_missing_local_file", item.get("missing_local_file")))
        ),
        "owner_system_material_warning_count": sum(
            1 for item in dependencies if bool(item.get("owner_system_material_library_ref"))
        ),
        "remote_ref_count": sum(1 for item in dependencies if bool(item.get("remote_ref"))),
        "unresolved_local_ref_count": sum(
            1 for item in dependencies if bool(item.get("unresolved_locally"))
        ),
    }


def _inspect_usd_with_pxr(path: Path) -> Dict[str, Any] | None:
    if importlib.util.find_spec("pxr") is None:
        return None
    try:  # pragma: no cover - optional dependency not available in CI
        from pxr import Usd, UsdGeom, UsdPhysics
    except Exception:
        return None
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception:
        return None
    if stage is None:
        return {
            "asset_type": "usd",
            "path": str(path.resolve()),
            "status": "blocked_openusd_stage_open_failed",
            "confidence": "low",
        }
    prim_count = 0
    mesh_count = 0
    material_count = 0
    collision_api_count = 0
    rigid_body_api_count = 0
    references_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if prim.GetTypeName() == "Material":
            material_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api_count += 1
        if prim.HasAuthoredReferences():
            references_count += 1
    bounds: Dict[str, Any] | None = None
    centroid: List[float] | None = None
    floor_z_estimate: float | None = None
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        aligned_range = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        if not aligned_range.IsEmpty():
            low = [float(aligned_range.GetMin()[index]) for index in range(3)]
            high = [float(aligned_range.GetMax()[index]) for index in range(3)]
            if all(math.isfinite(value) for value in [*low, *high]):
                bounds = {"min": low, "max": high}
                centroid = [(low[index] + high[index]) * 0.5 for index in range(3)]
                floor_z_estimate = low[2]
    except Exception:
        bounds = None
    return {
        "asset_type": "usd",
        "path": str(path.resolve()),
        "status": "inspected_with_openusd",
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "bounds": bounds,
        "centroid": centroid,
        "floor_z_estimate": floor_z_estimate,
        "estimate_method": "openusd_bbox_cache_world_bound" if bounds else "openusd_no_bounds",
        "prim_counts": {
            "total": prim_count,
            "mesh": mesh_count,
            "material": material_count,
            "referenced_prims": references_count,
            "physics_collision_api": collision_api_count,
            "physics_rigid_body_api": rigid_body_api_count,
        },
        "isaac_usd_import_candidate": True,
        "isaac_usd_collision_verified": collision_api_count > 0,
        "isaac_usd_collision_unverified": collision_api_count <= 0,
        "collision_verification_status": "verified_by_openusd_api"
        if collision_api_count > 0
        else "no_collision_api_found_by_openusd",
        "confidence": "medium",
        "limitations": [
            "OpenUSD API inspection is not Isaac Sim execution or physics/contact validation.",
            "OpenUSD world bounds are used only for CPU spawn sanity checks.",
        ],
    }


def inspect_usd_asset(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        raw_prefix = handle.read(4096)
    binary_usd = raw_prefix.startswith(b"PXR-USDC") or b"\x00" in raw_prefix
    pxr_result = _inspect_usd_with_pxr(path)
    if pxr_result is not None:
        openusd_dependencies = _extract_openusd_dependencies(path) if binary_usd else []
        text = "" if binary_usd else path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
        dependencies = openusd_dependencies if binary_usd else _extract_usd_dependencies(path, text)
        semantic_hints = [] if binary_usd else _extract_usd_semantic_hints(text)
        collision_verified = bool(pxr_result.get("isaac_usd_collision_verified"))
        pxr_result.setdefault("dependencies", dependencies)
        pxr_result.setdefault("dependency_summary", _usd_dependency_summary(dependencies))
        pxr_result.setdefault("semantic_hints", semantic_hints)
        pxr_result["collision_evidence"] = _collision_evidence(
            real_collider_proven=collision_verified,
            proxy_estimated=False,
            status=pxr_result.get("collision_verification_status")
            or ("usd_collision_api_present" if collision_verified else "usd_collision_unverified"),
            evidence=["UsdPhysics.CollisionAPI"] if collision_verified else [],
        )
        return pxr_result
    if binary_usd:
        return {
            "asset_type": "usd",
            "path": str(path.resolve()),
            "status": "openusd_required_for_binary_usd",
            "format": "usdc_or_binary_usd",
            "meters_per_unit": None,
            "up_axis": None,
            "prim_counts": {
                "def_or_over_lines_estimated": 0,
                "mesh_tokens_estimated": 0,
                "material_tokens_estimated": 0,
                "reference_tokens_estimated": 0,
                "physics_collision_tokens_estimated": 0,
                "rigid_body_tokens_estimated": 0,
            },
            "isaac_usd_import_candidate": True,
            "isaac_usd_collision_verified": False,
            "isaac_usd_collision_unverified": True,
            "collision_verification_status": "openusd_required_for_binary_usd",
            "dependencies": [],
            "dependency_summary": _usd_dependency_summary([]),
            "semantic_hints": [],
            "collision_evidence": _collision_evidence(
                real_collider_proven=False,
                proxy_estimated=False,
                status="openusd_required_for_binary_usd",
            ),
            "confidence": "low",
            "limitations": [
                "Binary USD/USDC requires pxr/OpenUSD or owner-system simulator inspection for dependency and collision details.",
                "USD import candidacy is not Isaac Sim execution or collision/contact proof.",
            ],
        }
    text = path.read_text(encoding="utf-8", errors="ignore")[:2_000_000]
    dependencies = _extract_usd_dependencies(path, text)
    semantic_hints = _extract_usd_semantic_hints(text)
    def count(pattern: str) -> int:
        return len(re.findall(pattern, text))

    has_collision_token = bool(re.search(r"CollisionAPI|PhysicsCollision|physics:collision", text))
    return {
        "asset_type": "usd",
        "path": str(path.resolve()),
        "status": "metadata_string_inspection_only",
        "meters_per_unit": 1.0 if "metersPerUnit = 1" in text else None,
        "up_axis": "Z" if 'upAxis = "Z"' in text else "Y" if 'upAxis = "Y"' in text else None,
        "prim_counts": {
            "def_or_over_lines_estimated": count(r"(?m)^\s*(def|over)\s+"),
            "mesh_tokens_estimated": count(r"\bMesh\b"),
            "material_tokens_estimated": count(r"\bMaterial\b"),
            "reference_tokens_estimated": count(r"@[^@\n]+@"),
            "physics_collision_tokens_estimated": count(r"CollisionAPI|PhysicsCollision|physics:collision"),
            "rigid_body_tokens_estimated": count(r"RigidBodyAPI|physics:rigidBody"),
        },
        "isaac_usd_import_candidate": True,
        "isaac_usd_collision_verified": False,
        "isaac_usd_collision_unverified": True,
        "collision_verification_status": "collision_api_tokens_present_unverified"
        if has_collision_token
        else "openusd_unavailable_collision_unverified",
        "dependencies": dependencies,
        "dependency_summary": _usd_dependency_summary(dependencies),
        "semantic_hints": semantic_hints,
        "collision_evidence": _collision_evidence(
            real_collider_proven=False,
            proxy_estimated=False,
            status="collision_api_tokens_present_unverified"
            if has_collision_token
            else "openusd_unavailable_collision_unverified",
            evidence=["usd_collision_tokens"] if has_collision_token else [],
        ),
        "confidence": "low",
        "limitations": [
            "pxr/OpenUSD is unavailable; string inspection cannot verify collision APIs.",
            "USD import candidacy is not Isaac Sim execution or collision/contact proof.",
        ],
    }


def _gltf_from_glb(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12 or header[:4] != b"glTF":
            raise ValueError("GLB header missing glTF magic")
        version, total_length = struct.unpack_from("<II", header, 4)
        chunk_header = handle.read(8)
        if len(chunk_header) != 8:
            raise ValueError("GLB missing JSON chunk")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:
            raise ValueError("GLB first chunk is not JSON")
        raw_json = handle.read(chunk_length)
    payload = json.loads(raw_json.decode("utf-8", errors="replace").rstrip("\x00 "))
    return {"payload": payload, "version": version, "total_length": total_length}


def _gltf_dependencies(path: Path, payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    dependencies: List[Dict[str, Any]] = []
    for item in payload.get("buffers") or []:
        if isinstance(item, Mapping) and _string(item.get("uri")):
            dependencies.append(
                _dependency_record(
                    source_path=path,
                    ref=_string(item.get("uri")),
                    relationship="buffer",
                )
            )
    for item in payload.get("images") or []:
        if isinstance(item, Mapping) and _string(item.get("uri")):
            dependencies.append(
                _dependency_record(
                    source_path=path,
                    ref=_string(item.get("uri")),
                    relationship="image_or_texture",
                )
            )
    return dependencies


def _bounds_from_min_max(low_value: Any, high_value: Any) -> Dict[str, Any] | None:
    low = _finite_float_list(low_value)
    high = _finite_float_list(high_value)
    if low is None or high is None:
        return None
    if any(high[index] < low[index] for index in range(3)):
        return None
    return {
        "bounds": {"min": low, "max": high},
        "centroid": [(low[index] + high[index]) * 0.5 for index in range(3)],
        "floor_z_estimate": low[2],
    }


def _gltf_position_accessor_indexes(payload: Mapping[str, Any]) -> List[int]:
    indexes: List[int] = []
    for mesh in payload.get("meshes") or []:
        if not isinstance(mesh, Mapping):
            continue
        for primitive in mesh.get("primitives") or []:
            if not isinstance(primitive, Mapping):
                continue
            attributes = _mapping(primitive.get("attributes"))
            raw_index = attributes.get("POSITION")
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if index not in indexes:
                indexes.append(index)
    return indexes


def _gltf_accessor_bounds(payload: Mapping[str, Any]) -> Dict[str, Any] | None:
    accessors = [item for item in payload.get("accessors") or [] if isinstance(item, Mapping)]
    if not accessors:
        return None
    preferred = _gltf_position_accessor_indexes(payload)
    fallback = [
        index
        for index, accessor in enumerate(accessors)
        if _string(accessor.get("type")).upper() == "VEC3"
    ]
    ordered_indexes = [*preferred, *[index for index in fallback if index not in preferred]]
    combined_low: List[float] | None = None
    combined_high: List[float] | None = None
    used_indexes: List[int] = []
    for index in ordered_indexes:
        if index < 0 or index >= len(accessors):
            continue
        accessor = accessors[index]
        estimate = _bounds_from_min_max(accessor.get("min"), accessor.get("max"))
        if estimate is None:
            continue
        bounds = _mapping(estimate.get("bounds"))
        low = [float(value) for value in bounds.get("min", [])[:3]]
        high = [float(value) for value in bounds.get("max", [])[:3]]
        if len(low) != 3 or len(high) != 3:
            continue
        combined_low = low if combined_low is None else [
            min(combined_low[axis], low[axis]) for axis in range(3)
        ]
        combined_high = high if combined_high is None else [
            max(combined_high[axis], high[axis]) for axis in range(3)
        ]
        used_indexes.append(index)
        if index in preferred:
            break
    if combined_low is None or combined_high is None:
        return None
    return {
        "bounds": {"min": combined_low, "max": combined_high},
        "centroid": [
            (combined_low[index] + combined_high[index]) * 0.5 for index in range(3)
        ],
        "floor_z_estimate": combined_low[2],
        "estimate_method": "gltf_position_accessor_min_max"
        if any(index in preferred for index in used_indexes)
        else "gltf_vec3_accessor_min_max",
        "accessor_indexes": used_indexes,
        "confidence": "medium" if any(index in preferred for index in used_indexes) else "low",
        "limitations": [
            "glTF accessor bounds are CPU metadata estimates and may not include node transforms or simulator import effects.",
        ],
    }


def _trimesh_gltf_bounds(path: Path) -> Dict[str, Any] | None:
    if importlib.util.find_spec("trimesh") is None:
        return None
    try:  # pragma: no cover - optional runtime dependency.
        import trimesh  # type: ignore[import-untyped]
    except Exception:
        return None
    try:  # pragma: no cover - optional runtime dependency.
        loaded = trimesh.load(str(path), force="scene", process=False)
        raw_bounds = getattr(loaded, "bounds", None)
    except Exception:
        return None
    if raw_bounds is None:
        return None
    try:
        low = [float(value) for value in raw_bounds[0][:3]]
        high = [float(value) for value in raw_bounds[1][:3]]
    except Exception:
        return None
    if len(low) != 3 or len(high) != 3:
        return None
    if not all(math.isfinite(value) for value in [*low, *high]):
        return None
    return {
        "bounds": {"min": low, "max": high},
        "centroid": [(low[index] + high[index]) * 0.5 for index in range(3)],
        "floor_z_estimate": low[2],
        "estimate_method": "trimesh_scene_bounds",
        "confidence": "medium",
        "limitations": [
            "trimesh bounds are CPU import estimates and not simulator load/contact proof.",
        ],
    }


def inspect_gltf_asset(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        glb = _gltf_from_glb(path)
        payload = _mapping(glb.get("payload"))
        asset_type = "glb"
        format_summary = {"version": glb.get("version"), "total_length": glb.get("total_length")}
    else:
        payload = _mapping(json.loads(path.read_text(encoding="utf-8", errors="ignore")))
        asset_type = "gltf"
        format_summary = {"version": _mapping(payload.get("asset")).get("version")}
    node_names = [
        _string(item.get("name"))
        for item in payload.get("nodes") or []
        if isinstance(item, Mapping) and _string(item.get("name"))
    ]
    mesh_names = [
        _string(item.get("name"))
        for item in payload.get("meshes") or []
        if isinstance(item, Mapping) and _string(item.get("name"))
    ]
    name_blob = " ".join([path.stem, *node_names, *mesh_names]).lower()
    collider_named = _has_collider_name_token(name_blob)
    dependencies = _gltf_dependencies(path, payload)
    bounds_estimate = _gltf_accessor_bounds(payload) or _trimesh_gltf_bounds(path) or {}
    return {
        "asset_type": asset_type,
        "path": str(path.resolve()),
        "status": "metadata_inspected",
        "format": format_summary,
        "node_count": len(payload.get("nodes") or []),
        "mesh_count": len(payload.get("meshes") or []),
        "dependencies": dependencies,
        "dependency_summary": _usd_dependency_summary(dependencies),
        "semantic_hints": _semantic_hints_from_names(
            [*node_names, *mesh_names],
            source=f"{asset_type}_node_or_mesh_name",
        ),
        "bounds": bounds_estimate.get("bounds"),
        "centroid": bounds_estimate.get("centroid"),
        "floor_z_estimate": bounds_estimate.get("floor_z_estimate"),
        "estimate_method": bounds_estimate.get("estimate_method") or "gltf_metadata_only",
        "confidence": bounds_estimate.get("confidence") or "low",
        "limitations": [
            "GLTF/GLB metadata inspection does not prove simulator load or contact behavior.",
            *list(bounds_estimate.get("limitations") or []),
        ],
        "collision_evidence": _collision_evidence(
            real_collider_proven=collider_named,
            proxy_estimated=bool(bounds_estimate.get("bounds")),
            portable_collider_glb_present=suffix == ".glb" and collider_named,
            status="portable_collider_name_present"
            if collider_named
            else "visual_gltf_collision_unverified",
            evidence=["collider_or_collision_name"] if collider_named else [],
        ),
    }


def inspect_obj_asset(path: Path) -> Dict[str, Any]:
    points: List[List[float]] = []
    names: List[str] = []
    dependencies: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4 and len(points) < 200_000:
                    try:
                        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
            elif line.startswith(("o ", "g ")):
                names.append(line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else "")
            elif line.startswith("mtllib "):
                ref = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) > 1 else ""
                if ref:
                    dependencies.append(
                        _dependency_record(
                            source_path=path,
                            ref=ref,
                            relationship="material_library",
                            line_number=line_number,
                        )
                    )
    estimate = _bounds_from_points(points) or {}
    name_blob = " ".join([path.stem, *names]).lower()
    collider_named = _has_collider_name_token(name_blob)
    return {
        "asset_type": "obj",
        "path": str(path.resolve()),
        "status": "metadata_inspected",
        "vertex_count_sampled": len(points),
        "bounds": estimate.get("bounds"),
        "centroid": estimate.get("centroid"),
        "floor_z_estimate": estimate.get("floor_z_estimate"),
        "estimate_method": "obj_vertex_sample" if estimate else "obj_metadata_only",
        "dependencies": dependencies,
        "dependency_summary": _usd_dependency_summary(dependencies),
        "semantic_hints": _semantic_hints_from_names(names, source="obj_object_or_group_name"),
        "confidence": "medium" if estimate else "low",
        "limitations": [
            "OBJ visual mesh inspection does not prove simulator load or contact behavior.",
        ],
        "collision_evidence": _collision_evidence(
            real_collider_proven=collider_named,
            proxy_estimated=bool(estimate.get("bounds")),
            status="obj_collider_name_present" if collider_named else "obj_collision_unverified",
            evidence=["collider_or_collision_name"] if collider_named else [],
        ),
    }


def _inspect_xml_asset(path: Path) -> Dict[str, Any]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return {
            "asset_type": _asset_type_for_path(path),
            "path": str(path.resolve()),
            "status": "blocked_xml_parse_failed",
            "reason": str(exc),
            "dependencies": [],
            "semantic_hints": [],
            "confidence": "low",
            "collision_evidence": _collision_evidence(
                real_collider_proven=False,
                status="xml_parse_failed_collision_unverified",
            ),
        }
    tag = root.tag.split("}")[-1].lower()
    asset_type = "urdf" if tag == "robot" else "mjcf" if tag == "mujoco" else _asset_type_for_path(path)
    dependencies: List[Dict[str, Any]] = []
    names: List[str] = []
    collision_count = 0
    geom_count = 0
    for node in root.iter():
        node_tag = node.tag.split("}")[-1].lower()
        if _string(node.attrib.get("name")):
            names.append(_string(node.attrib.get("name")))
        if node_tag == "collision":
            collision_count += 1
        if node_tag == "geom":
            geom_count += 1
        ref = _string(node.attrib.get("filename") or node.attrib.get("file"))
        if ref and _looks_like_supported_asset_ref(ref):
            dependencies.append(
                _dependency_record(
                    source_path=path,
                    ref=ref.replace("package://", ""),
                    relationship="mesh_reference",
                )
            )
    real_collider = collision_count > 0 or (asset_type == "mjcf" and geom_count > 0)
    return {
        "asset_type": asset_type,
        "path": str(path.resolve()),
        "status": "metadata_inspected",
        "root_tag": tag,
        "collision_element_count": collision_count,
        "geom_count": geom_count,
        "dependencies": dependencies,
        "dependency_summary": _usd_dependency_summary(dependencies),
        "semantic_hints": _semantic_hints_from_names(names, source=f"{asset_type}_element_name"),
        "bounds": None,
        "centroid": None,
        "floor_z_estimate": None,
        "confidence": "medium" if real_collider else "low",
        "limitations": [
            "URDF/MJCF collision metadata is not owner-system simulator execution or safety proof.",
        ],
        "collision_evidence": _collision_evidence(
            real_collider_proven=real_collider,
            proxy_estimated=False,
            status="portable_collision_metadata_present"
            if real_collider
            else "portable_collision_metadata_missing",
            evidence=[f"{asset_type}_collision_or_geom_elements"] if real_collider else [],
        ),
    }


def _candidate_paths_from_payload(capture_root: Path, payload: Mapping[str, Any]) -> List[Path]:
    paths: List[Path] = []
    for key in (
        "ply_scene_uri",
        "ply_scene_path",
        "usd_scene_uri",
        "usd_scene_path",
        "gltf_scene_uri",
        "gltf_scene_path",
        "glb_scene_uri",
        "glb_scene_path",
        "obj_scene_uri",
        "obj_scene_path",
        "urdf_scene_uri",
        "urdf_scene_path",
        "mjcf_scene_uri",
        "mjcf_scene_path",
        "collider_glb_uri",
        "collider_glb_path",
        "collider_mesh_glb_url",
    ):
        value = _string(payload.get(key))
        if value and not value.startswith(("gs://", "http://", "https://")):
            paths.append((capture_root / value).resolve() if not Path(value).is_absolute() else Path(value))
    assets = _mapping(payload.get("assets"))
    splats = _mapping(assets.get("splats"))
    for group in (splats.get("ply_urls"), splats.get("usd_urls")):
        if isinstance(group, Mapping):
            for value in group.values():
                text = _string(value)
                if text and not text.startswith(("gs://", "http://", "https://")):
                    paths.append((capture_root / text).resolve() if not Path(text).is_absolute() else Path(text))
    for value in _walk_payload_strings(payload):
        text = _string(value)
        if not text or _is_remote_ref(text) or not _looks_like_supported_asset_ref(text):
            continue
        candidate = Path(text)
        paths.append((capture_root / candidate).resolve() if not candidate.is_absolute() else candidate)
    return paths


def discover_scene_assets(capture_root: Path, explicit_assets: Sequence[str | Path] = ()) -> List[Path]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    candidates: List[Path] = []
    for item in explicit_assets:
        candidates.append(Path(item).expanduser())
    for rel in (
        "advanced_geometry/3dgs_compressed.ply",
        "geometry/alignment/canonical_pointcloud.ply",
        "simready/isaac_sim/site_scene.usda",
        "simready/isaac_sim/site_scene.usd",
        "simready/mujoco/site_scene.xml",
        "simready/pybullet/site_scene.urdf",
        "simulation_automation/cpu_scene_proxy_manifest.json",
    ):
        candidates.append(pipeline_dir / rel)
    for payload_path in (
        context.capture_root / "capture_descriptor.json",
        context.capture_root / "raw" / "manifest.json",
        context.capture_root / "raw" / "task_hypothesis.json",
        context.capture_root / "raw" / "capture_context.json",
        pipeline_dir / "worldlabs_world_manifest.json",
        pipeline_dir / "marble_sim_assets" / "marble_asset_manifest.json",
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json",
    ):
        payload = _read_optional_mapping(payload_path)
        if payload:
            candidates.extend(_candidate_paths_from_payload(context.capture_root, payload))
    for search_root in (context.capture_root / "raw", pipeline_dir):
        if search_root.is_dir():
            for path in sorted(search_root.rglob("*")):
                if path.is_relative_to(pipeline_dir / "simulation_automation"):
                    continue
                if path.is_file() and path.suffix.lower() in SUPPORTED_SCENE_ASSET_SUFFIXES:
                    candidates.append(path)
    seen: set[str] = set()
    out: List[Path] = []
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SCENE_ASSET_SUFFIXES:
            out.append(path)
    return out


def inspect_scene_asset(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return inspect_ply_asset(path)
    if suffix in {".usd", ".usda", ".usdc"}:
        return inspect_usd_asset(path)
    if suffix in {".glb", ".gltf"}:
        return inspect_gltf_asset(path)
    if suffix == ".obj":
        return inspect_obj_asset(path)
    if suffix in {".urdf", ".mjcf", ".xml"}:
        return _inspect_xml_asset(path)
    return {
        "asset_type": suffix.lstrip(".") or "unknown",
        "path": str(path.resolve()),
        "status": "unsupported_asset_type",
        "confidence": "low",
        "limitations": ["Unsupported asset type for CPU preflight inspection."],
    }


def _inspect_scene_assets_with_local_dependencies(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    queue: List[tuple[Path, str, str | None]] = [
        (Path(path), "discovered_scene_asset", None) for path in paths
    ]
    seen: set[str] = set()
    assets: List[Dict[str, Any]] = []
    while queue:
        path, discovery_source, referenced_by = queue.pop(0)
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        asset = inspect_scene_asset(resolved)
        asset["discovery_source"] = discovery_source
        if referenced_by:
            asset["referenced_by"] = referenced_by
        assets.append(asset)
        for dependency in asset.get("dependencies") or []:
            if not isinstance(dependency, Mapping):
                continue
            local_path = _string(dependency.get("local_path"))
            if not local_path or not dependency.get("exists_local"):
                continue
            dependency_path = Path(local_path).expanduser().resolve()
            if dependency_path.suffix.lower() not in SUPPORTED_SCENE_ASSET_SUFFIXES:
                continue
            queue.append((dependency_path, "local_dependency_reference", key))
    return assets


def _frame_from_asset(asset: Mapping[str, Any]) -> Dict[str, Any] | None:
    bounds = _mapping(asset.get("bounds"))
    if bounds.get("min") and bounds.get("max"):
        low = _finite_float_list(bounds.get("min"))
        high = _finite_float_list(bounds.get("max"))
        if low is None or high is None:
            return None
        spans = [high[index] - low[index] for index in range(3)]
        if any(span < 0.0 for span in spans):
            return None
        if spans[0] <= 1e-6 or spans[1] <= 1e-6:
            return None
        return {
            "source_asset": asset.get("path"),
            "source_asset_type": asset.get("asset_type"),
            "bounds": {"min": low, "max": high},
            "centroid": asset.get("centroid")
            or [(low[index] + high[index]) * 0.5 for index in range(3)],
            "floor_z_estimate": asset.get("floor_z_estimate")
            if asset.get("floor_z_estimate") is not None
            else low[2],
            "up_axis": asset.get("up_axis") or "Z",
            "confidence": asset.get("confidence") or "low",
            "estimate_method": asset.get("estimate_method") or "asset_bounds",
        }
    return None


def _frame_candidate_rank(asset: Mapping[str, Any], frame: Mapping[str, Any]) -> tuple[float, ...]:
    bounds = _mapping(frame.get("bounds"))
    low = _finite_float_list(bounds.get("min"))
    high = _finite_float_list(bounds.get("max"))
    spans = (
        [max(0.0, high[index] - low[index]) for index in range(3)]
        if low is not None and high is not None
        else [0.0, 0.0, 0.0]
    )
    planar_area = spans[0] * spans[1]
    collision = _mapping(asset.get("collision_evidence"))
    real_collider = 1.0 if collision.get("real_collider_proven") else 0.0
    portable_collider = 1.0 if collision.get("portable_collider_glb_present") else 0.0
    asset_type = _string(asset.get("asset_type")).lower()
    type_rank = {
        "usd": 5.0,
        "usda": 5.0,
        "usdc": 5.0,
        "glb": 4.0,
        "gltf": 4.0,
        "obj": 3.0,
        "ply": 1.0,
    }.get(asset_type, 0.0)
    confidence_rank = {
        "high": 3.0,
        "medium": 2.0,
        "low": 1.0,
    }.get(_string(asset.get("confidence")).lower(), 0.0)
    return (
        real_collider,
        portable_collider,
        type_rank,
        min(planar_area, 10_000.0),
        min(spans[2], 10_000.0),
        confidence_rank,
    )


def _select_scene_frame_source(assets: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    candidates: List[tuple[tuple[float, ...], Dict[str, Any], Mapping[str, Any]]] = []
    rejected: List[Dict[str, Any]] = []
    for asset in assets:
        frame = _frame_from_asset(asset)
        if frame is None:
            rejected.append(
                {
                    "path": asset.get("path"),
                    "asset_type": asset.get("asset_type"),
                    "reason": "missing_or_degenerate_xy_bounds",
                }
            )
            continue
        candidates.append((_frame_candidate_rank(asset, frame), frame, asset))
    if not candidates:
        return None, rejected
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = dict(candidates[0][1])
    selected["selection_rank"] = list(candidates[0][0])
    selected["selection_reason"] = "ranked_non_degenerate_local_scene_asset_bounds"
    selected["selection_policy"] = (
        "requires finite non-degenerate x/y bounds and prefers collider/simulator assets over thin pointcloud previews"
    )
    return selected, rejected


def _inventory_from_assets(assets: Sequence[Mapping[str, Any]], *, generated_at: str, context: Any) -> Dict[str, Any]:
    records = []
    for asset in assets:
        path_text = _string(asset.get("path"))
        path = Path(path_text) if path_text else None
        records.append(
            {
                **(_file_inventory_record(path) if path else {}),
                "asset_type": asset.get("asset_type"),
                "inspection_status": asset.get("status") or "inspected",
                "discovery_source": asset.get("discovery_source"),
                "referenced_by": asset.get("referenced_by"),
                "confidence": asset.get("confidence"),
                "bounds_present": bool(_mapping(asset.get("bounds")).get("min")),
                "dependency_count": len(asset.get("dependencies") or []),
                "collision_evidence": asset.get("collision_evidence") or {},
            }
        )
    return {
        "schema_version": SCENE_ASSET_INVENTORY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "complete" if records else "blocked",
        "supported_extensions": sorted(SUPPORTED_SCENE_ASSET_SUFFIXES),
        "asset_count": len(records),
        "assets": sorted(records, key=lambda item: _string(item.get("path"))),
        "source_policy": {
            "local_files_only": True,
            "remote_asset_downloads_performed": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _dependency_audit_from_assets(
    assets: Sequence[Mapping[str, Any]],
    *,
    generated_at: str,
    context: Any,
) -> Dict[str, Any]:
    dependencies: List[Dict[str, Any]] = []
    for asset in assets:
        source = _string(asset.get("path"))
        for dependency in asset.get("dependencies") or []:
            if isinstance(dependency, Mapping):
                dependencies.append({"source_asset": source, **dict(dependency)})
    warnings = []
    for dependency in dependencies:
        if dependency.get("remote_ref"):
            warnings.append(
                {
                    "kind": "remote_reference_not_downloaded",
                    "source_asset": dependency.get("source_asset"),
                    "ref": dependency.get("ref"),
                }
            )
        if dependency.get("missing_local_file"):
            warnings.append(
                {
                    "kind": "owner_system_material_library_not_local"
                    if dependency.get("owner_system_material_library_ref")
                    else "missing_local_dependency",
                    "source_asset": dependency.get("source_asset"),
                    "ref": dependency.get("ref"),
                    "local_path": dependency.get("local_path"),
                }
            )
    missing_count = sum(1 for item in dependencies if item.get("missing_local_file"))
    hard_missing_count = sum(
        1 for item in dependencies if item.get("hard_missing_local_file", item.get("missing_local_file"))
    )
    owner_system_material_warning_count = sum(
        1 for item in dependencies if item.get("owner_system_material_library_ref")
    )
    remote_count = sum(1 for item in dependencies if item.get("remote_ref"))
    unresolved_count = sum(1 for item in dependencies if item.get("unresolved_locally"))
    return {
        "schema_version": SCENE_ASSET_DEPENDENCY_AUDIT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "blocked"
        if hard_missing_count
        else "complete_with_remote_refs"
        if remote_count
        else "complete",
        "dependency_count": len(dependencies),
        "missing_local_file_count": missing_count,
        "hard_missing_local_file_count": hard_missing_count,
        "owner_system_material_warning_count": owner_system_material_warning_count,
        "remote_ref_count": remote_count,
        "unresolved_ref_count": unresolved_count,
        "dependencies": sorted(
            dependencies,
            key=lambda item: (
                _string(item.get("source_asset")),
                _string(item.get("relationship")),
                _string(item.get("ref")),
            ),
        ),
        "warnings": warnings,
        "source_policy": {
            "local_files_only": True,
            "remote_asset_downloads_performed": False,
            "remote_refs_are_handoff_inputs_only": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _object_geometry_proxy_obstacles(pipeline_dir: Path) -> List[Dict[str, Any]]:
    manifest = _read_optional_mapping(pipeline_dir / "evaluation_prep" / "object_geometry_manifest.json")
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        return []
    obstacles: List[Dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, Mapping):
            continue
        bbox = _mapping(item.get("placement_bbox") or item.get("bbox"))
        center = _finite_float_list(bbox.get("center"))
        extents = _finite_float_list(bbox.get("extents") or bbox.get("size"))
        if center is None or extents is None:
            continue
        half = [max(0.0, value) * 0.5 for value in extents]
        obstacles.append(
            {
                "obstacle_id": _string(item.get("object_id") or item.get("id") or item.get("label"))
                or f"object_proxy_{len(obstacles) + 1}",
                "label": _string(item.get("label")) or None,
                "type": "axis_aligned_box",
                "min_xyz": [center[index] - half[index] for index in range(3)],
                "max_xyz": [center[index] + half[index] for index in range(3)],
                "source": "evaluation_prep/object_geometry_manifest.json",
                "confidence": "derived_review_required",
            }
        )
    return obstacles


def _proxy_primitive_from_frame(frame_source: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    frame = _mapping(frame_source)
    bounds = _mapping(frame.get("bounds"))
    low = _finite_float_list(bounds.get("min"))
    high = _finite_float_list(bounds.get("max"))
    if low is None or high is None:
        return None
    floor_z = frame.get("floor_z_estimate")
    try:
        floor = float(floor_z)
    except (TypeError, ValueError):
        floor = low[2]
    return {
        "proxy_id": "scene_bounds_floor_proxy",
        "type": "floor_plane_plus_scene_aabb",
        "floor_z": floor,
        "bounds_min_xyz": low,
        "bounds_max_xyz": high,
        "source": frame.get("source_asset") or "scene_frame_estimate",
        "confidence": "estimated_review_required",
    }


def _collider_proxy_artifacts(
    *,
    assets: Sequence[Mapping[str, Any]],
    frame_source: Mapping[str, Any] | None,
    pipeline_dir: Path,
    generated_at: str,
    context: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    real_collider_assets: List[Dict[str, Any]] = []
    labels: List[str] = []
    for asset in assets:
        evidence = _mapping(asset.get("collision_evidence"))
        if evidence.get("real_collider_proven"):
            real_collider_assets.append(
                {
                    "path": asset.get("path"),
                    "asset_type": asset.get("asset_type"),
                    "status": evidence.get("status"),
                    "evidence": evidence.get("evidence") or [],
                }
            )
            labels.append("real_collider_proven")
        elif evidence.get("proxy_estimated"):
            labels.append("proxy_estimated")
        else:
            labels.append("missing_collider")
    proxy_primitive = _proxy_primitive_from_frame(frame_source)
    proxy_obstacles = _object_geometry_proxy_obstacles(pipeline_dir)
    proxy_estimated = bool(proxy_primitive)
    real_collider_proven = bool(real_collider_assets)
    missing_collider = not real_collider_proven
    labels.extend(
        [
            "real_collider_proven" if real_collider_proven else "missing_collider",
            "proxy_estimated" if proxy_estimated else "proxy_unavailable",
            "review_required",
        ]
    )
    labels = sorted(set(labels))
    plan = {
        "schema_version": COLLIDER_PROXY_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "real_collider_metadata_present"
        if real_collider_proven
        else "proxy_plan_ready" if proxy_estimated else "blocked_missing_collider_and_proxy",
        "real_collider_proven": real_collider_proven,
        "proxy_estimated": proxy_estimated,
        "missing_collider": missing_collider,
        "review_required": True,
        "labels": labels,
        "real_collider_assets": real_collider_assets,
        "proxy_plan": {
            "proxy_primitive": proxy_primitive,
            "object_proxy_obstacle_count": len(proxy_obstacles),
            "object_proxy_obstacles": proxy_obstacles,
            "conservative_policy": [
                "Use scene bounds and floor only for CPU spawn sanity checks.",
                "Treat generated proxy collisions as review inputs until owner simulator proof exists.",
            ],
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    proxy_manifest = {
        "schema_version": CPU_SCENE_PROXY_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_cpu_spawn_checks" if proxy_estimated else "blocked_missing_proxy",
        "real_collider_proven": real_collider_proven,
        "proxy_estimated": proxy_estimated,
        "missing_collider": missing_collider,
        "review_required": True,
        "floor_proxy": proxy_primitive,
        "proxy_obstacles": proxy_obstacles,
        "simulator_execution_proven": False,
        "physics_contact_validated": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    return plan, proxy_manifest


def build_scene_asset_preflight(
    *,
    capture_root: str | Path,
    scene_assets: Sequence[str | Path] = (),
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    automation_dir = context.pipeline_root / "simulation_automation"
    ensure_dir(automation_dir)
    generated_at = utc_now_iso()
    discovered_paths = discover_scene_assets(context.capture_root, scene_assets)
    assets = _inspect_scene_assets_with_local_dependencies(discovered_paths)
    frame_source, rejected_frame_sources = _select_scene_frame_source(assets)
    inventory = _inventory_from_assets(assets, generated_at=generated_at, context=context)
    dependency_audit = _dependency_audit_from_assets(
        assets,
        generated_at=generated_at,
        context=context,
    )
    usd_assets = [asset for asset in assets if asset.get("asset_type") == "usd"]
    binary_usd_openusd_handoff = any(
        asset.get("status") == "openusd_required_for_binary_usd" for asset in usd_assets
    )
    owner_system_usd_handoff_ready = bool(
        binary_usd_openusd_handoff and not dependency_audit["hard_missing_local_file_count"]
    )
    collider_proxy_plan, cpu_scene_proxy_manifest = _collider_proxy_artifacts(
        assets=assets,
        frame_source=frame_source,
        pipeline_dir=context.pipeline_root,
        generated_at=generated_at,
        context=context,
    )
    blockers: List[str] = []
    if not assets:
        blockers.append("missing_local_scene_asset")
    if frame_source is None and not owner_system_usd_handoff_ready:
        blockers.append("missing_scene_frame_estimate")
    if dependency_audit["hard_missing_local_file_count"]:
        blockers.append("missing_scene_asset_dependencies")

    preflight_ready = bool(frame_source or owner_system_usd_handoff_ready)
    portable_collider_present = any(
        _mapping(asset.get("collision_evidence")).get("portable_collider_glb_present")
        or _mapping(asset.get("collision_evidence")).get("real_collider_proven")
        for asset in assets
    )
    real_collider_proven = bool(collider_proxy_plan.get("real_collider_proven"))
    proxy_estimated = bool(collider_proxy_plan.get("proxy_estimated"))
    missing_collider = bool(collider_proxy_plan.get("missing_collider"))
    cpu_proxy_estimated = bool(frame_source and frame_source.get("bounds"))
    inspection = {
        "schema_version": SCENE_ASSET_INSPECTION_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "complete" if assets else "blocked",
        "asset_count": len(assets),
        "assets": assets,
        "blockers": blockers,
        "source_policy": {
            "local_files_only": True,
            "remote_asset_downloads_performed": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    frame_estimate = {
        "schema_version": SCENE_FRAME_ESTIMATE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "complete" if frame_source else "blocked",
        "frame": frame_source,
        "frame_candidate_policy": {
            "requires_non_degenerate_xy_bounds": True,
            "preferred_sources": [
                "real_collider_metadata",
                "portable_collider_glb",
                "usd_or_simulator_asset_bounds",
                "other_non_degenerate_local_asset_bounds",
            ],
            "rejected_frame_source_count": len(rejected_frame_sources),
            "rejected_frame_sources": rejected_frame_sources[:20],
        },
        "coordinate_frame": {
            "up_axis": _mapping(frame_source or {}).get("up_axis") or "Z",
            "units": "meters_estimated",
            "scale_proof": "unproven_unless_source_manifest_proves_metric_scale",
        },
        "blockers": [] if frame_source else ["missing_scene_bounds_or_usd_frame"],
        "limitations": [
            "Frame estimates are CPU-derived review inputs.",
            "They do not prove collision/contact behavior or generated-world rank fidelity.",
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    scorecard = {
        "schema_version": CPU_PREFLIGHT_SCORECARD_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_episode_setup" if preflight_ready else "blocked",
        "scene_asset_inspection_status": inspection["status"],
        "scene_frame_estimate_status": frame_estimate["status"],
        "binary_usd_openusd_handoff": binary_usd_openusd_handoff,
        "owner_system_usd_handoff_ready": owner_system_usd_handoff_ready,
        "isaac_usd_import_candidate": any(asset.get("isaac_usd_import_candidate") for asset in usd_assets),
        "isaac_usd_collision_verified": any(asset.get("isaac_usd_collision_verified") for asset in usd_assets),
        "isaac_usd_collision_unverified": any(asset.get("isaac_usd_collision_unverified") for asset in usd_assets),
        "portable_collider_glb_present": portable_collider_present,
        "portable_collider_glb_missing": not portable_collider_present,
        "cpu_proxy_collision_estimated": cpu_proxy_estimated,
        "simulator_execution_not_run": True,
        "blockers": [
            *blockers,
            *(["portable_collider_glb_missing"] if not portable_collider_present else []),
            *(["isaac_usd_collision_unverified"] if any(asset.get("isaac_usd_collision_unverified") for asset in usd_assets) else []),
            "simulator_execution_not_run",
        ],
        "scene_asset_inventory_path": "scene_asset_inventory.json",
        "scene_asset_dependency_audit_path": "scene_asset_dependency_audit.json",
        "scene_asset_preflight_path": "scene_asset_preflight.json",
        "collider_proxy_plan_path": "collider_proxy_plan.json",
        "cpu_scene_proxy_manifest_path": "cpu_scene_proxy_manifest.json",
        "dependency_warning_count": len(dependency_audit.get("warnings") or []),
        "missing_dependency_count": dependency_audit["hard_missing_local_file_count"],
        "owner_system_material_warning_count": dependency_audit[
            "owner_system_material_warning_count"
        ],
        "remote_ref_count": dependency_audit["remote_ref_count"],
        "real_collider_proven": real_collider_proven,
        "proxy_estimated": proxy_estimated,
        "missing_collider": missing_collider,
        "review_required": True,
        "collider_backend_labels": collider_proxy_plan.get("labels") or [],
        "proof_booleans": {
            "simulator_execution_proven": False,
            "rank_fidelity_result_proven": False,
            "physics_contact_validated": False,
            "non_ranking_operational_claim_validated": False,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    inspection["deterministic_fingerprint"] = _sha_payload(
        {"assets": assets, "blockers": blockers}
    )
    frame_estimate["deterministic_fingerprint"] = _sha_payload(
        {"frame": frame_source, "blockers": frame_estimate["blockers"]}
    )
    scorecard["deterministic_fingerprint"] = _sha_payload(
        {"scorecard": {k: v for k, v in scorecard.items() if k != "generated_at"}}
    )
    inventory["deterministic_fingerprint"] = _sha_payload(
        {"assets": inventory["assets"], "asset_count": inventory["asset_count"]}
    )
    dependency_audit["deterministic_fingerprint"] = _sha_payload(
        {
            "dependencies": dependency_audit["dependencies"],
            "missing": dependency_audit["missing_local_file_count"],
            "hard_missing": dependency_audit["hard_missing_local_file_count"],
            "remote": dependency_audit["remote_ref_count"],
        }
    )
    collider_proxy_plan["deterministic_fingerprint"] = _sha_payload(
        {
            "labels": collider_proxy_plan["labels"],
            "real_collider_assets": collider_proxy_plan["real_collider_assets"],
            "proxy_plan": collider_proxy_plan["proxy_plan"],
        }
    )
    cpu_scene_proxy_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "floor_proxy": cpu_scene_proxy_manifest["floor_proxy"],
            "proxy_obstacles": cpu_scene_proxy_manifest["proxy_obstacles"],
        }
    )
    preflight_manifest = {
        "schema_version": SCENE_ASSET_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "ready_for_episode_setup" if preflight_ready else "blocked",
        "asset_count": len(assets),
        "artifact_paths": {
            "scene_asset_inventory": "scene_asset_inventory.json",
            "scene_asset_dependency_audit": "scene_asset_dependency_audit.json",
            "scene_asset_inspection": "scene_asset_inspection.json",
            "scene_frame_estimate": "scene_frame_estimate.json",
            "cpu_preflight_scorecard": "cpu_preflight_scorecard.json",
            "collider_proxy_plan": "collider_proxy_plan.json",
            "cpu_scene_proxy_manifest": "cpu_scene_proxy_manifest.json",
        },
        "dependency_summary": {
            "missing_local_file_count": dependency_audit["missing_local_file_count"],
            "hard_missing_local_file_count": dependency_audit["hard_missing_local_file_count"],
            "owner_system_material_warning_count": dependency_audit[
                "owner_system_material_warning_count"
            ],
            "remote_ref_count": dependency_audit["remote_ref_count"],
            "unresolved_ref_count": dependency_audit["unresolved_ref_count"],
            "warning_count": len(dependency_audit.get("warnings") or []),
        },
        "collider_summary": {
            "real_collider_proven": real_collider_proven,
            "proxy_estimated": proxy_estimated,
            "missing_collider": missing_collider,
            "review_required": True,
            "binary_usd_openusd_handoff": binary_usd_openusd_handoff,
            "owner_system_usd_handoff_ready": owner_system_usd_handoff_ready,
            "labels": collider_proxy_plan.get("labels") or [],
        },
        "blockers": list(dict.fromkeys(scorecard["blockers"])),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    preflight_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "status": preflight_manifest["status"],
            "dependency_summary": preflight_manifest["dependency_summary"],
            "collider_summary": preflight_manifest["collider_summary"],
            "blockers": preflight_manifest["blockers"],
        }
    )

    write_json(automation_dir / "scene_asset_inventory.json", inventory)
    write_json(automation_dir / "scene_asset_dependency_audit.json", dependency_audit)
    write_json(automation_dir / "scene_asset_inspection.json", inspection)
    write_json(automation_dir / "scene_frame_estimate.json", frame_estimate)
    write_json(automation_dir / "cpu_preflight_scorecard.json", scorecard)
    write_json(automation_dir / "collider_proxy_plan.json", collider_proxy_plan)
    write_json(automation_dir / "cpu_scene_proxy_manifest.json", cpu_scene_proxy_manifest)
    write_json(automation_dir / "scene_asset_preflight.json", preflight_manifest)
    return {
        "schema_version": "scene_asset_preflight_result.v1",
        "capture_root": str(context.capture_root),
        "automation_dir": str(automation_dir),
        "status": scorecard["status"],
        "scene_asset_inventory_path": str((automation_dir / "scene_asset_inventory.json").resolve()),
        "scene_asset_dependency_audit_path": str(
            (automation_dir / "scene_asset_dependency_audit.json").resolve()
        ),
        "scene_asset_preflight_path": str((automation_dir / "scene_asset_preflight.json").resolve()),
        "scene_asset_inspection_path": str((automation_dir / "scene_asset_inspection.json").resolve()),
        "scene_frame_estimate_path": str((automation_dir / "scene_frame_estimate.json").resolve()),
        "cpu_preflight_scorecard_path": str((automation_dir / "cpu_preflight_scorecard.json").resolve()),
        "collider_proxy_plan_path": str((automation_dir / "collider_proxy_plan.json").resolve()),
        "cpu_scene_proxy_manifest_path": str(
            (automation_dir / "cpu_scene_proxy_manifest.json").resolve()
        ),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build CPU-only scene asset preflight manifests for a local capture"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument(
        "--scene-asset",
        action="append",
        default=[],
        help="Optional local PLY/USD asset path to inspect; repeatable",
    )
    args = parser.parse_args(argv)
    try:
        result = build_scene_asset_preflight(
            capture_root=args.capture_root,
            scene_assets=args.scene_asset,
        )
    except (OSError, ValueError, PipelineError) as exc:
        print(f"[scene-asset-preflight] FAILED: {exc}")
        return 1
    print(f"[scene-asset-preflight] scorecard={result['cpu_preflight_scorecard_path']}")
    print(f"[scene-asset-preflight] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
