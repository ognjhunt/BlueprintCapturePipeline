"""Automatic swappable-asset candidate selection from object index signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import try_parse_float, utc_now_iso


_ARTICULATED_APPLIANCE_KEYWORDS = {
    "dishwasher",
    "refrigerator",
    "fridge",
    "oven",
    "microwave",
    "washer",
    "dryer",
    "freezer",
}

_ARTICULATED_FURNITURE_KEYWORDS = {
    "drawer",
    "cabinet",
    "door",
    "cupboard",
    "closet",
    "wardrobe",
    "locker",
}

_MANIPULABLE_KEYWORDS = {
    "tote",
    "bin",
    "box",
    "crate",
    "carton",
    "package",
    "container",
    "cup",
    "mug",
    "bottle",
    "can",
    "tool",
    "part",
    "object",
}


@dataclass(frozen=True)
class SwapCandidate:
    object_id: str
    label: str
    sim_role: str
    articulation_required: bool
    articulation_reason: str
    must_be_separate_asset: bool
    asset_dir: str
    point_cloud_file: Optional[str]
    obb: Dict[str, Any]
    dimensions_est: Dict[str, float]
    physics_hints: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "asset_dir": self.asset_dir,
            "label": self.label,
            "sim_role": self.sim_role,
            "must_be_separate_asset": self.must_be_separate_asset,
            "articulation": {
                "required": self.articulation_required,
                "requirement_source": self.articulation_reason,
            },
            "point_cloud_file": self.point_cloud_file,
            "obb": dict(self.obb),
            "dimensions_est": dict(self.dimensions_est),
            "physics_hints": dict(self.physics_hints),
        }


def _normalized_text(*parts: Any) -> str:
    tokens = [str(part).strip().lower() for part in parts if part is not None]
    return " ".join(token for token in tokens if token)


def _object_id(entry: Mapping[str, Any]) -> str:
    for key in ("id", "object_id", "uuid", "identifier"):
        raw = entry.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    raise ValueError(f"Object entry missing id field: {entry}")


def _label(entry: Mapping[str, Any]) -> str:
    for key in ("label", "name", "class_name", "category"):
        raw = entry.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return "object"


def _bounding_box(entry: Mapping[str, Any]) -> Dict[str, Any]:
    obb = entry.get("boundingBox") if isinstance(entry.get("boundingBox"), Mapping) else None
    if obb is None:
        obb = entry.get("obb") if isinstance(entry.get("obb"), Mapping) else {}

    center = obb.get("center") if isinstance(obb.get("center"), list) else [0.0, 0.0, 0.0]
    extents = obb.get("extents") if isinstance(obb.get("extents"), list) else [0.25, 0.25, 0.25]
    axes = obb.get("axes") if isinstance(obb.get("axes"), list) else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    quat = (
        obb.get("orientationQuaternion")
        if isinstance(obb.get("orientationQuaternion"), list)
        else [1.0, 0.0, 0.0, 0.0]
    )

    center = [try_parse_float(center[idx] if idx < len(center) else 0.0) for idx in range(3)]
    extents = [max(0.02, try_parse_float(extents[idx] if idx < len(extents) else 0.25, 0.25)) for idx in range(3)]

    normalized_axes: List[List[float]] = []
    for axis_idx in range(3):
        axis = axes[axis_idx] if axis_idx < len(axes) and isinstance(axes[axis_idx], list) else [0.0, 0.0, 0.0]
        normalized_axes.append([try_parse_float(axis[col] if col < len(axis) else 0.0, 0.0) for col in range(3)])

    quat = [try_parse_float(quat[idx] if idx < len(quat) else 0.0) for idx in range(4)]
    if all(abs(value) < 1e-8 for value in quat):
        quat = [1.0, 0.0, 0.0, 0.0]

    return {
        "center": center,
        "extents": extents,
        "axes": normalized_axes,
        "orientationQuaternion": quat,
    }


def _dimensions_from_obb(obb: Mapping[str, Any]) -> Dict[str, float]:
    extents = obb.get("extents") if isinstance(obb.get("extents"), list) else [0.25, 0.25, 0.25]
    width = max(0.02, try_parse_float(extents[0] if len(extents) > 0 else 0.25, 0.25))
    height = max(0.02, try_parse_float(extents[1] if len(extents) > 1 else 0.25, 0.25))
    depth = max(0.02, try_parse_float(extents[2] if len(extents) > 2 else 0.25, 0.25))
    return {
        "width": width,
        "height": height,
        "depth": depth,
    }


def _manipulation_lookup(
    descriptor: CaptureDescriptor,
) -> tuple[set[str], list[str], set[str], list[str]]:
    manip_ids: set[str] = set()
    manip_labels: list[str] = []
    articulated_ids: set[str] = set()
    articulated_labels: list[str] = []

    for entry in descriptor.manipulation_candidates:
        instance_id = str(entry.get("instance_id") or entry.get("id") or "").strip()
        label = str(entry.get("label") or entry.get("name") or "").strip().lower()
        if instance_id:
            manip_ids.add(instance_id)
        if label:
            manip_labels.append(label)

    for entry in descriptor.articulation_hints:
        instance_id = str(entry.get("instance_id") or entry.get("id") or "").strip()
        label = str(entry.get("label") or entry.get("name") or "").strip().lower()
        if instance_id:
            articulated_ids.add(instance_id)
        if label:
            articulated_labels.append(label)

    return manip_ids, manip_labels, articulated_ids, articulated_labels


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _classify_role(
    text: str,
    *,
    force_manipulable: bool,
    force_articulated: bool,
) -> tuple[Optional[str], bool, str]:
    if force_articulated:
        if _contains_any(text, _ARTICULATED_APPLIANCE_KEYWORDS):
            return "articulated_appliance", True, "descriptor_articulation_hint"
        return "articulated_furniture", True, "descriptor_articulation_hint"

    if _contains_any(text, _ARTICULATED_APPLIANCE_KEYWORDS):
        return "articulated_appliance", True, "keyword"
    if _contains_any(text, _ARTICULATED_FURNITURE_KEYWORDS):
        return "articulated_furniture", True, "keyword"

    if force_manipulable or _contains_any(text, _MANIPULABLE_KEYWORDS):
        reason = "descriptor_manipulation_candidate" if force_manipulable else "keyword"
        return "manipulable_object", False, reason

    return None, False, "not_selected"


def _physics_hints(sim_role: str) -> Dict[str, Any]:
    if sim_role == "manipulable_object":
        return {"dynamic": True, "mass_kg": 1.0}
    if sim_role in {"articulated_furniture", "articulated_appliance"}:
        return {"dynamic": False, "kinematic": True}
    return {"dynamic": False}


def select_swap_candidates(
    *,
    descriptor: CaptureDescriptor,
    object_index_entries: List[Mapping[str, Any]],
) -> List[SwapCandidate]:
    """Select swap candidates using object index + descriptor signals."""

    manip_ids, manip_labels, articulated_ids, articulated_labels = _manipulation_lookup(descriptor)
    selected: List[SwapCandidate] = []

    for entry in object_index_entries:
        object_id = _object_id(entry)
        label = _label(entry)
        source_text = _normalized_text(
            label,
            entry.get("name"),
            entry.get("class_name"),
            entry.get("category"),
            entry.get("description"),
        )

        force_manipulable = object_id in manip_ids or any(token in source_text for token in manip_labels)
        force_articulated = object_id in articulated_ids or any(
            token in source_text for token in articulated_labels
        )

        sim_role, articulation_required, articulation_reason = _classify_role(
            source_text,
            force_manipulable=force_manipulable,
            force_articulated=force_articulated,
        )
        if sim_role is None:
            continue

        obb = _bounding_box(entry)
        selected.append(
            SwapCandidate(
                object_id=object_id,
                label=label,
                sim_role=sim_role,
                articulation_required=articulation_required,
                articulation_reason=articulation_reason,
                must_be_separate_asset=True,
                asset_dir=f"obj_{object_id}",
                point_cloud_file=(
                    str(entry.get("pointCloudFile")).strip()
                    if entry.get("pointCloudFile") is not None
                    else None
                ),
                obb=obb,
                dimensions_est=_dimensions_from_obb(obb),
                physics_hints=_physics_hints(sim_role),
            )
        )

    return selected


def build_swap_candidates_payload(
    *,
    descriptor: CaptureDescriptor,
    object_index_entries: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = select_swap_candidates(
        descriptor=descriptor,
        object_index_entries=object_index_entries,
    )
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "policy": "auto_by_signals",
        "generated_at": utc_now_iso(),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
