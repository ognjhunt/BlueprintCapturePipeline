"""Automatic swappable-asset candidate selection from object index signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .capture_bridge import CaptureDescriptor
from .common import try_parse_float, utc_now_iso

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - dependency availability is tested in preflight
    yaml = None


_DEFAULT_POLICY_NAME = "auto_by_signals_default"
_DEFAULT_POLICY_VERSION = "v1"

_DEFAULT_POLICY: Dict[str, Any] = {
    "defaults": {
        "articulated_appliance_keywords": [
            "dishwasher",
            "refrigerator",
            "fridge",
            "oven",
            "microwave",
            "washer",
            "dryer",
            "freezer",
            "appliance_door",
        ],
        "articulated_furniture_keywords": [
            "drawer",
            "cabinet",
            "door",
            "cupboard",
            "closet",
            "wardrobe",
            "locker",
        ],
        "manipulable_keywords": [
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
        ],
        "exclude_keywords": [
            "wall",
            "floor",
            "ceiling",
            "window",
            "stairs",
            "pillar",
            "column",
            "beam",
            "light_fixture",
            "outlet",
        ],
        "min_volume_m3": {
            "articulated_appliance": 0.01,
            "articulated_furniture": 0.006,
            "manipulable_object": 0.0002,
        },
    },
    "environments": {
        "kitchen": {
            "articulated_appliance_keywords": [
                "dishwasher",
                "fridge",
                "refrigerator",
                "microwave",
                "oven",
                "cabinet_door",
            ],
            "articulated_furniture_keywords": [
                "drawer",
                "cabinet",
                "pantry_door",
            ],
            "manipulable_keywords": [
                "mug",
                "cup",
                "bowl",
                "plate",
                "pot",
                "pan",
                "bottle",
            ],
            "exclude_keywords": [
                "countertop",
                "backsplash",
            ],
        },
        "warehouse": {
            "manipulable_keywords": [
                "tote",
                "bin",
                "carton",
                "package",
                "pallet",
                "crate",
                "container",
            ],
            "exclude_keywords": [
                "rack",
                "shelf",
                "conveyor",
                "docking_door",
                "safety_barrier",
            ],
        },
        "industrial_unknown": {
            "articulated_furniture_keywords": [
                "door",
                "gate",
                "locker",
                "cabinet",
            ],
            "manipulable_keywords": [
                "tote",
                "bin",
                "carton",
                "crate",
                "container",
                "part",
                "tray",
            ],
            "exclude_keywords": [
                "aisle",
                "lane",
                "traffic_zone",
                "floor_hazard",
                "barrier",
            ],
        },
        "bedroom": {
            "articulated_furniture_keywords": [
                "door",
                "closet_door",
                "drawer",
                "wardrobe",
                "dresser",
                "nightstand",
            ],
            "manipulable_keywords": [
                "box",
                "container",
                "basket",
                "hamper",
                "bag",
                "bin",
            ],
            "exclude_keywords": [
                "bedframe",
                "mattress",
                "wall",
                "floor",
                "ceiling",
            ],
        },
    },
}


@dataclass(frozen=True)
class SwapPolicyConfig:
    name: str
    source: str
    articulated_appliance_keywords: frozenset[str]
    articulated_furniture_keywords: frozenset[str]
    manipulable_keywords: frozenset[str]
    exclude_keywords: frozenset[str]
    min_volume_m3: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "articulated_appliance_keywords": sorted(self.articulated_appliance_keywords),
            "articulated_furniture_keywords": sorted(self.articulated_furniture_keywords),
            "manipulable_keywords": sorted(self.manipulable_keywords),
            "exclude_keywords": sorted(self.exclude_keywords),
            "min_volume_m3": dict(self.min_volume_m3),
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
    reference_crop: Optional[str] = None
    all_crops: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
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
        if self.reference_crop is not None:
            d["reference_crop"] = self.reference_crop
        if self.all_crops:
            d["all_crops"] = list(self.all_crops)
        return d


def _normalized_tokens(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        token = str(value).strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _policy_hints(descriptor: CaptureDescriptor) -> List[str]:
    hints = _normalized_tokens(
        [descriptor.environment_type_hint] + list(descriptor.swap_focus or [])
    )
    return hints


def _deep_copy_policy(payload: Mapping[str, Any]) -> Dict[str, Any]:
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), Mapping) else {}
    envs = payload.get("environments") if isinstance(payload.get("environments"), Mapping) else {}
    copied: Dict[str, Any] = {
        "defaults": {
            "articulated_appliance_keywords": list(defaults.get("articulated_appliance_keywords") or []),
            "articulated_furniture_keywords": list(defaults.get("articulated_furniture_keywords") or []),
            "manipulable_keywords": list(defaults.get("manipulable_keywords") or []),
            "exclude_keywords": list(defaults.get("exclude_keywords") or []),
            "min_volume_m3": dict(defaults.get("min_volume_m3") or {}),
        },
        "environments": {},
    }
    for key, value in envs.items():
        if not isinstance(value, Mapping):
            continue
        copied["environments"][str(key).strip().lower()] = {
            "articulated_appliance_keywords": list(value.get("articulated_appliance_keywords") or []),
            "articulated_furniture_keywords": list(value.get("articulated_furniture_keywords") or []),
            "manipulable_keywords": list(value.get("manipulable_keywords") or []),
            "exclude_keywords": list(value.get("exclude_keywords") or []),
            "min_volume_m3": dict(value.get("min_volume_m3") or {}),
        }
    return copied


def _merge_keywords(base: Iterable[Any], overlay: Iterable[Any]) -> List[str]:
    base_values = list(base) if base is not None else []
    if overlay is None:
        overlay_values: List[Any] = []
    elif isinstance(overlay, str):
        overlay_values = [overlay]
    else:
        overlay_values = list(overlay)
    return _normalized_tokens(base_values + overlay_values)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _merge_min_volume(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, float]:
    merged = {str(key): _safe_float(value, 0.0) for key, value in dict(base).items()}
    for key, value in dict(overlay).items():
        merged[str(key)] = _safe_float(value, 0.0)
    return merged


def _overlay_policy(base: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    defaults = base["defaults"]
    defaults["articulated_appliance_keywords"] = _merge_keywords(
        defaults["articulated_appliance_keywords"],
        overlay.get("articulated_appliance_keywords") if isinstance(overlay, Mapping) else [],
    )
    defaults["articulated_furniture_keywords"] = _merge_keywords(
        defaults["articulated_furniture_keywords"],
        overlay.get("articulated_furniture_keywords") if isinstance(overlay, Mapping) else [],
    )
    defaults["manipulable_keywords"] = _merge_keywords(
        defaults["manipulable_keywords"],
        overlay.get("manipulable_keywords") if isinstance(overlay, Mapping) else [],
    )
    defaults["exclude_keywords"] = _merge_keywords(
        defaults["exclude_keywords"],
        overlay.get("exclude_keywords") if isinstance(overlay, Mapping) else [],
    )
    if isinstance(overlay, Mapping):
        defaults["min_volume_m3"] = _merge_min_volume(
            defaults.get("min_volume_m3") or {},
            overlay.get("min_volume_m3") if isinstance(overlay.get("min_volume_m3"), Mapping) else {},
        )
    return base


def _load_policy_payload(policy_path: Optional[str]) -> tuple[Dict[str, Any], str, str]:
    payload = _deep_copy_policy(_DEFAULT_POLICY)
    policy_name = _DEFAULT_POLICY_NAME
    source = "builtin_default"

    if not policy_path:
        return payload, policy_name, source

    path = Path(policy_path)
    if not path.is_file():
        raise ValueError(f"swap policy config not found: {path}")
    if yaml is None:
        raise ValueError("PyYAML is required to load custom swap policy config")

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    if not isinstance(loaded, Mapping):
        raise ValueError(f"invalid swap policy payload type: {type(loaded).__name__}")

    schema_version = str(loaded.get("schema_version") or _DEFAULT_POLICY_VERSION).strip()
    if schema_version != _DEFAULT_POLICY_VERSION:
        raise ValueError(
            f"unsupported swap policy schema_version: {schema_version} (expected {_DEFAULT_POLICY_VERSION})"
        )

    policy_name = str(loaded.get("name") or loaded.get("policy_name") or _DEFAULT_POLICY_NAME).strip()
    source = str(path)
    defaults = loaded.get("defaults") if isinstance(loaded.get("defaults"), Mapping) else {}
    environments = (
        loaded.get("environments") if isinstance(loaded.get("environments"), Mapping) else {}
    )

    _overlay_policy(payload, defaults)
    for env_name, env_payload in environments.items():
        if not isinstance(env_payload, Mapping):
            continue
        env_key = str(env_name).strip().lower()
        existing_env = payload["environments"].get(env_key) if isinstance(payload["environments"], Mapping) else {}
        if not isinstance(existing_env, Mapping):
            existing_env = {}
        merged_env = {
            "articulated_appliance_keywords": _merge_keywords(
                existing_env.get("articulated_appliance_keywords") if isinstance(existing_env, Mapping) else [],
                env_payload.get("articulated_appliance_keywords"),
            ),
            "articulated_furniture_keywords": _merge_keywords(
                existing_env.get("articulated_furniture_keywords") if isinstance(existing_env, Mapping) else [],
                env_payload.get("articulated_furniture_keywords"),
            ),
            "manipulable_keywords": _merge_keywords(
                existing_env.get("manipulable_keywords") if isinstance(existing_env, Mapping) else [],
                env_payload.get("manipulable_keywords"),
            ),
            "exclude_keywords": _merge_keywords(
                existing_env.get("exclude_keywords") if isinstance(existing_env, Mapping) else [],
                env_payload.get("exclude_keywords"),
            ),
            "min_volume_m3": _merge_min_volume(
                existing_env.get("min_volume_m3")
                if isinstance(existing_env.get("min_volume_m3"), Mapping)
                else {},
                env_payload.get("min_volume_m3")
                if isinstance(env_payload.get("min_volume_m3"), Mapping)
                else {},
            ),
        }
        payload["environments"][env_key] = merged_env

    return payload, policy_name, source


def resolve_policy_config(
    *,
    descriptor: CaptureDescriptor,
    policy_path: Optional[str] = None,
) -> SwapPolicyConfig:
    payload, policy_name, source = _load_policy_payload(policy_path)
    hints = _policy_hints(descriptor)

    defaults = payload["defaults"]
    resolved: Dict[str, Any] = {
        "articulated_appliance_keywords": list(defaults["articulated_appliance_keywords"]),
        "articulated_furniture_keywords": list(defaults["articulated_furniture_keywords"]),
        "manipulable_keywords": list(defaults["manipulable_keywords"]),
        "exclude_keywords": list(defaults["exclude_keywords"]),
        "min_volume_m3": dict(defaults["min_volume_m3"]),
    }

    env_payloads = payload.get("environments") if isinstance(payload.get("environments"), Mapping) else {}
    for hint in hints:
        env_overlay = env_payloads.get(hint) if isinstance(env_payloads, Mapping) else None
        if not isinstance(env_overlay, Mapping):
            continue
        _overlay_policy({"defaults": resolved, "environments": {}}, env_overlay)

    return SwapPolicyConfig(
        name=policy_name,
        source=source,
        articulated_appliance_keywords=frozenset(
            _normalized_tokens(resolved["articulated_appliance_keywords"])
        ),
        articulated_furniture_keywords=frozenset(
            _normalized_tokens(resolved["articulated_furniture_keywords"])
        ),
        manipulable_keywords=frozenset(_normalized_tokens(resolved["manipulable_keywords"])),
        exclude_keywords=frozenset(_normalized_tokens(resolved["exclude_keywords"])),
        min_volume_m3={
            str(key): max(0.0, _safe_float(value, 0.0))
            for key, value in dict(resolved["min_volume_m3"]).items()
        },
    )


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
    policy: SwapPolicyConfig,
    force_manipulable: bool,
    force_articulated: bool,
) -> tuple[Optional[str], bool, str]:
    if _contains_any(text, policy.exclude_keywords) and not (force_manipulable or force_articulated):
        return None, False, "policy_excluded"

    if force_articulated:
        if _contains_any(text, policy.articulated_appliance_keywords):
            return "articulated_appliance", True, "descriptor_articulation_hint"
        return "articulated_furniture", True, "descriptor_articulation_hint"

    if _contains_any(text, policy.articulated_appliance_keywords):
        return "articulated_appliance", True, "keyword"
    if _contains_any(text, policy.articulated_furniture_keywords):
        return "articulated_furniture", True, "keyword"

    if force_manipulable or _contains_any(text, policy.manipulable_keywords):
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
    policy_path: Optional[str] = None,
    resolved_policy: Optional[SwapPolicyConfig] = None,
) -> List[SwapCandidate]:
    """Select swap candidates using object index + descriptor signals."""

    policy = resolved_policy or resolve_policy_config(descriptor=descriptor, policy_path=policy_path)
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
            policy=policy,
            force_manipulable=force_manipulable,
            force_articulated=force_articulated,
        )
        if sim_role is None:
            continue

        obb = _bounding_box(entry)
        dimensions = _dimensions_from_obb(obb)
        volume = dimensions["width"] * dimensions["height"] * dimensions["depth"]
        min_volume = max(0.0, _safe_float(policy.min_volume_m3.get(sim_role), 0.0))
        if min_volume > 0 and volume < min_volume and not (force_manipulable or force_articulated):
            continue

        # Extract reference crop paths from object index if present
        ref_crop = entry.get("reference_crop")
        ref_crop_str = str(ref_crop).strip() if ref_crop is not None else None
        raw_all_crops = entry.get("all_crops")
        all_crops_list = (
            [str(c) for c in raw_all_crops if c]
            if isinstance(raw_all_crops, list)
            else None
        )

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
                dimensions_est=dimensions,
                physics_hints=_physics_hints(sim_role),
                reference_crop=ref_crop_str,
                all_crops=all_crops_list,
            )
        )

    return selected


def build_swap_candidates_payload(
    *,
    descriptor: CaptureDescriptor,
    object_index_entries: List[Mapping[str, Any]],
    policy_path: Optional[str] = None,
) -> Dict[str, Any]:
    policy = resolve_policy_config(descriptor=descriptor, policy_path=policy_path)
    candidates = select_swap_candidates(
        descriptor=descriptor,
        object_index_entries=object_index_entries,
        policy_path=policy_path,
        resolved_policy=policy,
    )
    return {
        "schema_version": "v1",
        "scene_id": descriptor.scene_id,
        "capture_id": descriptor.capture_id,
        "policy": "auto_by_signals",
        "policy_details": policy.to_dict(),
        "environment_hints": _policy_hints(descriptor),
        "generated_at": utc_now_iso(),
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
