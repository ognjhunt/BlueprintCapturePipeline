"""Optional Palatial PhysReady twin generation lane.

This lane prepares task-critical object twin requests from capture-derived
objects and can submit them to Palatial only when explicitly enabled. Generated
assets remain model-derived support artifacts until local and owner-system
validation prove simulator/contact behavior.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import urllib.error
import urllib.request as _urllib_request
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

from .common import PipelineError, ensure_dir, parse_bool, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


PALATIAL_CANDIDATE_SCHEMA_VERSION = "palatial_physready_twin_candidates.v1"
PALATIAL_REQUEST_SCHEMA_VERSION = "palatial_physready_request_manifest.v1"
PALATIAL_RUN_SCHEMA_VERSION = "palatial_physready_run_manifest.v1"
PALATIAL_MATERIALIZATION_SCHEMA_VERSION = "palatial_physready_materialization.v1"
PALATIAL_VALIDATION_SCHEMA_VERSION = "palatial_physready_validation.v1"
PALATIAL_DEFAULT_GENERATE_URL = "https://dashboard.palatial.cloud/api/v1/external/generate"
PALATIAL_ENABLE_ENV = "BLUEPRINT_ENABLE_PALATIAL_PHYSREADY"
PALATIAL_API_KEY_ENV = "PALATIAL_API_KEY"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".heic", ".heif"}
SUPPORTED_EXPORT_SUFFIXES = {
    ".usd",
    ".usda",
    ".usdc",
    ".mjcf",
    ".xml",
    ".urdf",
    ".obj",
    ".glb",
    ".gltf",
    ".zip",
    ".json",
}
DEFAULT_TARGET_SIMS = ("isaac_sim", "mujoco")
TASK_CRITICAL_LABEL_TOKENS = {
    "microwave",
    "tote",
    "bin",
    "crate",
    "box",
    "package",
    "door",
    "drawer",
    "cabinet",
    "shelf",
    "cart",
    "handle",
    "knob",
    "button",
    "appliance",
    "tool",
}

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "palatial_physready_model_derived_support_asset_lane",
    "enabled_by_default": False,
    "raw_capture_authority_preserved": True,
    "generated_assets_are_capture_truth": False,
    "generated_assets_are_model_derived_support": True,
    "simulator_execution_proven": False,
    "physics_contact_validated": False,
    "robot_readiness_proven": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "palatial_asset_is_raw_capture_truth",
        "physics_contact_validated",
        "robot_policy_ready",
        "robot_deployment_ready",
        "simulator_execution_completed",
    ],
    "promotion_requires": [
        "source image and prompt lineage",
        "license and rights review",
        "local export checksums",
        "unit/scale sanity checks",
        "collision and articulation metadata inspection",
        "owner-system simulator load trace before robot-readiness claims",
    ],
}


class PalatialClientProtocol(Protocol):
    def generate_asset(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        """Submit one Palatial request and return provider JSON."""


def _env_truthy(name: str, *, env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    return parse_bool(source.get(name), default=False)


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
    seen: set[str] = set()
    out: List[str] = []
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


def _safe_slug(value: Any, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"n_{text}"
    return text[:80]


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def _looks_like_image_ref(value: str) -> bool:
    cleaned = value.split("?", 1)[0].split("#", 1)[0]
    return Path(cleaned).suffix.lower() in IMAGE_SUFFIXES


def _looks_like_export_ref(value: str) -> bool:
    cleaned = value.split("?", 1)[0].split("#", 1)[0]
    return Path(cleaned).suffix.lower() in SUPPORTED_EXPORT_SUFFIXES


def _is_remote_ref(value: str) -> bool:
    return value.lower().startswith(("http://", "https://", "gs://", "s3://"))


def _resolve_local_path(context: Any, value: str) -> Path | None:
    text = _string(value)
    if not text or _is_remote_ref(text) or any(char in text for char in ("\x00", "\r", "\n")):
        return None
    path = Path(text).expanduser()
    candidates = [path] if path.is_absolute() else [
        context.capture_root / path,
        context.pipeline_root / path,
        context.raw_root / path,
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve() if candidates else None


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _walk_strings(item)


def _reference_image_record(context: Any, uri: str, *, source: str) -> Dict[str, Any]:
    local_path = _resolve_local_path(context, uri)
    exists = bool(local_path and local_path.is_file())
    return {
        "uri": uri,
        "source": source,
        "local_path": str(local_path) if local_path else None,
        "exists_local": exists,
        "size_bytes": local_path.stat().st_size if exists and local_path else None,
        "sha256": _sha_file(local_path) if exists and local_path else None,
        "upload_allowed_only_when_live_gate_passes": True,
    }


def _collect_object_image_refs(
    obj: Mapping[str, Any],
    *,
    context: Any,
    include_capture_image_fallback: bool,
    max_images: int,
) -> List[Dict[str, Any]]:
    keyed_sources = {
        "reference_images",
        "source_images",
        "image_uris",
        "image_paths",
        "crop_paths",
        "crops",
        "thumbnail_path",
        "thumbnail_uri",
        "reference_frame_uri",
        "reference_frame_path",
        "mask_path",
        "visual_replacement_masks",
        "provenance",
    }
    refs: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in sorted(keyed_sources):
        for value in _walk_strings(obj.get(key)):
            text = _string(value)
            if not text or not _looks_like_image_ref(text) or text in seen:
                continue
            seen.add(text)
            refs.append(_reference_image_record(context, text, source=f"object.{key}"))
            if len(refs) >= max_images:
                return refs
    for value in _walk_strings(obj):
        text = _string(value)
        if not text or not _looks_like_image_ref(text) or text in seen:
            continue
        seen.add(text)
        refs.append(_reference_image_record(context, text, source="object.deep_scan"))
        if len(refs) >= max_images:
            return refs
    if refs or not include_capture_image_fallback:
        return refs
    for path in sorted(context.raw_root.rglob("*")) if context.raw_root.is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = _relative_to(context.capture_root, path)
        if rel in seen:
            continue
        seen.add(rel)
        record = _reference_image_record(context, rel, source="raw.capture_image_fallback")
        record["fallback_full_capture_frame"] = True
        refs.append(record)
        if len(refs) >= max_images:
            break
    return refs


def _bbox_from_object(obj: Mapping[str, Any]) -> Dict[str, Any]:
    for key in ("placement_bbox", "boundingBox", "bbox", "obb"):
        raw = _mapping(obj.get(key))
        center = raw.get("center")
        extents = raw.get("extents") or raw.get("size") or raw.get("dimensions")
        if center or extents:
            return {
                "center": center if isinstance(center, list) else None,
                "extents": extents if isinstance(extents, list) else None,
                "source_key": key,
            }
    return {"center": None, "extents": None, "source_key": None}


def _task_refs(task_anchor_manifest: Mapping[str, Any]) -> tuple[set[str], set[str], Dict[str, List[str]]]:
    target_ids: set[str] = set()
    articulation_ids: set[str] = set()
    task_text_by_object: Dict[str, List[str]] = {}
    tasks = task_anchor_manifest.get("tasks")
    if not isinstance(tasks, list):
        return target_ids, articulation_ids, task_text_by_object
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_text = _string(task.get("task_text") or task.get("name") or task.get("task_id"))
        ids = _string_list(task.get("target_object_ids"))
        art_ids = _string_list(task.get("articulation_required_ids"))
        target_ids.update(ids)
        articulation_ids.update(art_ids)
        for object_id in [*ids, *art_ids]:
            if task_text:
                task_text_by_object.setdefault(object_id, []).append(task_text)
    return target_ids, articulation_ids, task_text_by_object


def _label_tokens(*values: Any) -> set[str]:
    blob = " ".join(_string(value).lower() for value in values)
    return {token for token in re.split(r"[^a-z0-9]+", blob) if token}


def _desired_articulation(
    *,
    object_id: str,
    label: str,
    task_role: str,
    articulation_required: bool,
) -> Dict[str, Any]:
    tokens = _label_tokens(object_id, label, task_role)
    if "microwave" in tokens:
        return {
            "requested": True,
            "type": "hinged_door_with_handle",
            "notes": [
                "Preserve microwave door swing, handle/grasp region, inner cavity, and button panel.",
                "Treat electronic behavior as semantic metadata unless a task explicitly requires it.",
            ],
        }
    if tokens.intersection({"tote", "bin", "crate"}):
        return {
            "requested": articulation_required,
            "type": "rigid_container_with_optional_lid_or_grasp_handles",
            "notes": [
                "Prioritize graspable rigid body, rim/handle geometry, and optional removable lid if visible.",
            ],
        }
    if tokens.intersection({"door", "cabinet", "drawer"}):
        return {
            "requested": True,
            "type": "hinge_or_prismatic_joint_from_task_context",
            "notes": ["Infer hinge/slide axis conservatively from image views and task prompt."],
        }
    if tokens.intersection({"button", "knob", "handle"}):
        return {
            "requested": articulation_required,
            "type": "small_manipulation_affordance",
            "notes": ["Preserve contact surface and local collision even if joint metadata is uncertain."],
        }
    return {
        "requested": articulation_required,
        "type": "single_object_or_rigid_parts",
        "notes": ["Use rigid parts unless images/task text clearly support articulation."],
    }


def _prompt_for_candidate(
    *,
    label: str,
    object_id: str,
    bbox: Mapping[str, Any],
    articulation: Mapping[str, Any],
    task_texts: Sequence[str],
) -> str:
    extents = bbox.get("extents")
    scale_hint = f" Observed extents are approximately {extents} meters." if extents else ""
    task_hint = f" Robot task context: {'; '.join(task_texts[:3])}." if task_texts else ""
    articulation_hint = (
        f" Requested articulation: {articulation.get('type')}; "
        f"articulation required={bool(articulation.get('requested'))}."
    )
    return (
        f"Create a PhysReady digital twin of the captured {label or object_id}. "
        "Use the provided scan/capture images as visual reference, preserve contact-rich geometry, "
        "PBR material cues, physically plausible mass/inertia/friction, collision meshes, and "
        "simulator-ready scale. "
        f"{articulation_hint}{scale_hint}{task_hint} "
        "Return model-derived support assets for Isaac Sim/OpenUSD and MuJoCo where available."
    ).strip()


def _object_selected(
    *,
    object_id: str,
    label: str,
    task_role: str,
    target_ids: set[str],
    articulation_ids: set[str],
    requested_object_ids: set[str],
    requested_labels: set[str],
) -> tuple[bool, List[str]]:
    tokens = _label_tokens(object_id, label, task_role)
    reasons: List[str] = []
    explicit_filter_active = bool(requested_object_ids or requested_labels)
    explicit_match = object_id in requested_object_ids or bool(tokens.intersection(requested_labels))
    if explicit_filter_active and not explicit_match:
        return False, reasons
    if object_id in requested_object_ids:
        reasons.append("explicit_object_id")
    if tokens.intersection(requested_labels):
        reasons.append("explicit_label")
    if object_id in target_ids:
        reasons.append("task_target_object")
    if object_id in articulation_ids:
        reasons.append("articulation_required_object")
    if tokens.intersection(TASK_CRITICAL_LABEL_TOKENS):
        reasons.append("task_critical_label_token")
    return bool(reasons), reasons


def build_twin_candidates(
    *,
    capture_root: str | Path,
    object_geometry_manifest: Optional[Mapping[str, Any]] = None,
    task_anchor_manifest: Optional[Mapping[str, Any]] = None,
    object_ids: Sequence[str] = (),
    labels: Sequence[str] = (),
    max_candidates: int = 20,
    max_images_per_asset: int = 4,
    include_capture_image_fallback: bool = False,
    target_sims: Sequence[str] = DEFAULT_TARGET_SIMS,
    collision: str = "sdf",
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    eval_dir = context.pipeline_root / "evaluation_prep"
    object_geometry = dict(
        object_geometry_manifest
        or _read_optional_mapping(eval_dir / "object_geometry_manifest.json")
    )
    task_anchor = dict(
        task_anchor_manifest
        or _read_optional_mapping(eval_dir / "task_anchor_manifest.json")
    )
    target_ids, articulation_ids, task_text_by_object = _task_refs(task_anchor)
    requested_object_ids = set(_string_list(object_ids))
    requested_labels = set().union(*(_label_tokens(label) for label in labels)) if labels else set()
    objects = object_geometry.get("objects")
    object_records = objects if isinstance(objects, list) else []
    candidates: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for index, raw_obj in enumerate(object_records):
        if not isinstance(raw_obj, Mapping):
            continue
        object_id = _string(raw_obj.get("object_id") or raw_obj.get("id") or f"object_{index}")
        label = _string(raw_obj.get("label") or raw_obj.get("class_name") or object_id)
        task_role = _string(raw_obj.get("task_role"))
        selected, reasons = _object_selected(
            object_id=object_id,
            label=label,
            task_role=task_role,
            target_ids=target_ids,
            articulation_ids=articulation_ids,
            requested_object_ids=requested_object_ids,
            requested_labels=requested_labels,
        )
        if not selected:
            skipped.append({"object_id": object_id, "label": label, "reason": "not_task_critical"})
            continue
        bbox = _bbox_from_object(raw_obj)
        articulation = _desired_articulation(
            object_id=object_id,
            label=label,
            task_role=task_role,
            articulation_required=object_id in articulation_ids,
        )
        reference_images = _collect_object_image_refs(
            raw_obj,
            context=context,
            include_capture_image_fallback=include_capture_image_fallback,
            max_images=max_images_per_asset,
        )
        task_texts = task_text_by_object.get(object_id, [])
        candidate_id = f"palatial_{_safe_slug(object_id or label, fallback=f'object_{index}')}"
        prompt = _prompt_for_candidate(
            label=label,
            object_id=object_id,
            bbox=bbox,
            articulation=articulation,
            task_texts=task_texts,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_object_id": object_id,
                "label": label,
                "task_role": task_role or None,
                "selection_reasons": reasons,
                "task_context": task_texts,
                "bbox": bbox,
                "desired_articulation": articulation,
                "target_sims": list(target_sims),
                "collision": collision,
                "prompt": prompt,
                "reference_images": reference_images,
                "reference_image_count": len(reference_images),
                "missing_reference_images": not bool(reference_images),
                "source_object_provenance": _mapping(raw_obj.get("provenance")),
                "capture_truth_policy": {
                    "raw_capture_remains_authoritative": True,
                    "palatial_output_role": "model_derived_support_asset",
                },
            }
        )
        if len(candidates) >= max_candidates:
            break
    blockers: List[str] = []
    warnings: List[str] = []
    if not object_records:
        blockers.append("missing_object_geometry_manifest_objects")
    if not candidates:
        blockers.append("missing_palatial_twin_candidates")
    if any(candidate["missing_reference_images"] for candidate in candidates):
        warnings.append("one_or_more_candidates_missing_reference_images_text_only_request")
    manifest = {
        "schema_version": PALATIAL_CANDIDATE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "blocked" if blockers else "ready",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "skipped_objects": skipped[:100],
        "selection_policy": {
            "object_ids": sorted(requested_object_ids),
            "labels": sorted(requested_labels),
            "task_target_objects_selected": True,
            "articulation_required_objects_selected": True,
            "task_critical_label_tokens": sorted(TASK_CRITICAL_LABEL_TOKENS),
            "include_capture_image_fallback": include_capture_image_fallback,
        },
        "blockers": blockers,
        "warnings": warnings,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "scene_id": context.scene_id,
            "capture_id": context.capture_id,
            "candidates": [
                {
                    "source_object_id": item["source_object_id"],
                    "label": item["label"],
                    "prompt": item["prompt"],
                    "reference_images": [
                        image.get("sha256") or image.get("uri")
                        for image in item.get("reference_images", [])
                    ],
                }
                for item in candidates
            ],
        }
    )
    return manifest


def _target_sim_api_value(target_sims: Sequence[str]) -> str:
    mapped = []
    for sim in target_sims:
        value = _string(sim).lower()
        if value in {"isaac", "isaac_sim", "openusd", "usd"}:
            mapped.append("isaac")
        elif value in {"mujoco", "mjcf"}:
            mapped.append("mujoco")
        elif value:
            mapped.append(value)
    return ",".join(dict.fromkeys(mapped)) or "isaac"


def _api_payload_for_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = {
        "source": "blueprint_capture_pipeline",
        "source_object_id": candidate.get("source_object_id"),
        "candidate_id": candidate.get("candidate_id"),
        "label": candidate.get("label"),
        "desired_articulation": candidate.get("desired_articulation"),
        "capture_truth_policy": candidate.get("capture_truth_policy"),
    }
    return {
        "candidate_id": candidate.get("candidate_id"),
        "prompt": candidate.get("prompt"),
        "target_sim": _target_sim_api_value(_string_list(candidate.get("target_sims"))),
        "collision": _string(candidate.get("collision")) or "sdf",
        "source_type": "image" if candidate.get("reference_images") else "text",
        "metadata_json": json.dumps(metadata, sort_keys=True),
        "image_paths": [
            image.get("local_path")
            for image in candidate.get("reference_images") or []
            if isinstance(image, Mapping) and image.get("exists_local") and image.get("local_path")
        ],
    }


def _live_gate(
    *,
    allow_live_palatial: bool,
    env: Mapping[str, str] | None,
    api_key_env: str,
) -> Dict[str, Any]:
    source = env if env is not None else os.environ
    env_gate_allows = _env_truthy(PALATIAL_ENABLE_ENV, env=source)
    api_key_present = bool(_string(source.get(api_key_env)))
    missing = []
    if not env_gate_allows:
        missing.append(f"{PALATIAL_ENABLE_ENV}=true")
    if not allow_live_palatial:
        missing.append("--allow-live-palatial")
    if not api_key_present:
        missing.append(f"{api_key_env}=<secret>")
    return {
        "enabled_by_default": False,
        "env_gate": PALATIAL_ENABLE_ENV,
        "env_gate_allows": env_gate_allows,
        "allow_live_palatial_flag": bool(allow_live_palatial),
        "api_key_env": api_key_env,
        "api_key_present": api_key_present,
        "live_provider_calls_allowed": not missing,
        "missing_gates": missing,
    }


def _request_manifest(
    *,
    context: Any,
    candidates: Mapping[str, Any],
    generate_url: str,
    auth_mode: str,
    live_gate: Mapping[str, Any],
    token_price_usd: float,
    estimated_tokens_per_asset: float,
) -> Dict[str, Any]:
    planned = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "source_object_id": candidate.get("source_object_id"),
            "label": candidate.get("label"),
            "request": {
                key: value
                for key, value in _api_payload_for_candidate(candidate).items()
                if key != "image_paths"
            },
            "local_image_count": len(_api_payload_for_candidate(candidate).get("image_paths") or []),
        }
        for candidate in candidates.get("candidates", [])
        if isinstance(candidate, Mapping)
    ]
    estimated_tokens = round(len(planned) * estimated_tokens_per_asset, 4)
    return {
        "schema_version": PALATIAL_REQUEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "provider": "palatial_physready",
        "provider_product": "PhysReady",
        "candidate_manifest_path": "twin_candidate_manifest.json",
        "request_count": len(planned),
        "planned_requests": planned,
        "api_config": {
            "generate_url": generate_url,
            "auth_mode": auth_mode,
            "api_key_value_recorded": False,
            "public_api_contract_note": (
                "Palatial public docs and dashboard examples have differed; "
                "override PALATIAL_GENERATE_URL/PALATIAL_AUTH_MODE after key smoke tests."
            ),
        },
        "live_execution_gate": dict(live_gate),
        "pricing_estimate": {
            "billing_unit": "palatial_token_estimate",
            "estimated_tokens_per_asset": estimated_tokens_per_asset,
            "estimated_total_tokens": estimated_tokens,
            "token_price_usd": token_price_usd,
            "estimated_marginal_cost_usd": round(estimated_tokens * token_price_usd, 2),
            "requires_billing_dashboard_confirmation": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


class PalatialApiClient:
    """Small urllib Palatial transport with configurable URL/auth mode."""

    def __init__(self, *, generate_url: str, api_key: str, auth_mode: str = "x-api-key") -> None:
        self.generate_url = generate_url
        self.api_key = api_key
        self.auth_mode = auth_mode

    def _headers(self, content_type: str) -> Dict[str, str]:
        headers = {
            "Content-Type": content_type,
            "Accept": "application/json",
            "User-Agent": "BlueprintCapturePipeline/2.0 PalatialPhysReady",
        }
        if self.auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-api-key"] = self.api_key
        return headers

    def generate_asset(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        fields = {
            "prompt": _string(request.get("prompt")),
            "target_sim": _string(request.get("target_sim") or "isaac"),
            "collision": _string(request.get("collision") or "sdf"),
            "source_type": _string(request.get("source_type") or "text"),
            "metadata_json": _string(request.get("metadata_json")),
        }
        files = [
            Path(path)
            for path in _string_list(request.get("image_paths"))
            if Path(path).is_file()
        ]
        body, content_type = _multipart_form_data(fields=fields, file_paths=files)
        req = _urllib_request.Request(
            self.generate_url,
            data=body,
            headers=self._headers(content_type),
            method="POST",
        )
        try:
            with _urllib_request.urlopen(req, timeout=600) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            raise RuntimeError(f"palatial_api_{exc.code}:{raw_err or 'request_failed'}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"palatial_api_url_error:{exc.reason}") from exc
        parsed = json.loads(raw) if raw else {}
        return dict(parsed) if isinstance(parsed, Mapping) else {"raw_response": parsed}


def _multipart_form_data(*, fields: Mapping[str, str], file_paths: Sequence[Path]) -> tuple[bytes, str]:
    boundary = f"----BlueprintPalatial{sha256(repr(sorted(fields.items())).encode()).hexdigest()[:16]}"
    chunks: List[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for index, path in enumerate(file_paths):
        name = "image" if len(file_paths) == 1 else f"image_{index + 1}"
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime}\r\n\r\n".encode("utf-8"),
                path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _collect_export_refs(value: Any, *, source_key: str = "response") -> List[Dict[str, str]]:
    refs: List[Dict[str, str]] = []
    if isinstance(value, str):
        text = _string(value)
        if _looks_like_export_ref(text):
            refs.append({"ref": text, "source_key": source_key})
    elif isinstance(value, Mapping):
        for key, item in value.items():
            child_key = f"{source_key}.{key}"
            if key in {
                "download_url",
                "download_urls",
                "export_url",
                "export_urls",
                "url",
                "uri",
                "path",
                "asset_url",
                "exports",
                "files",
                "assets",
            }:
                refs.extend(_collect_export_refs(item, source_key=child_key))
            elif isinstance(item, str) and _looks_like_export_ref(item):
                refs.append({"ref": _string(item), "source_key": child_key})
            elif isinstance(item, (Mapping, list, tuple)):
                refs.extend(_collect_export_refs(item, source_key=child_key))
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for index, item in enumerate(value):
            refs.extend(_collect_export_refs(item, source_key=f"{source_key}[{index}]"))
    seen: set[str] = set()
    out: List[Dict[str, str]] = []
    for item in refs:
        ref = item["ref"]
        if ref in seen:
            continue
        seen.add(ref)
        out.append(item)
    return out


def _suffix_for_ref(ref: str) -> str:
    suffix = Path(ref.split("?", 1)[0].split("#", 1)[0]).suffix.lower()
    return suffix if suffix else ".bin"


def _download_or_copy_ref(
    *,
    ref: str,
    output_path: Path,
    context: Any,
    max_bytes: int,
) -> Dict[str, Any]:
    ensure_dir(output_path.parent)
    if ref.startswith(("http://", "https://")):
        request = _urllib_request.Request(ref, headers={"User-Agent": "BlueprintCapturePipeline/2.0"})
        bytes_written = 0
        try:
            with _urllib_request.urlopen(request, timeout=600) as response, output_path.open("wb") as handle:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise RuntimeError("palatial_export_exceeds_max_bytes")
                    except ValueError:
                        pass
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise RuntimeError("palatial_export_exceeds_max_bytes")
                    handle.write(chunk)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        action = "downloaded"
    else:
        local_path = _resolve_local_path(context, ref)
        if local_path is None or not local_path.is_file():
            raise FileNotFoundError(ref)
        if local_path.stat().st_size > max_bytes:
            raise RuntimeError("palatial_export_exceeds_max_bytes")
        shutil.copyfile(local_path, output_path)
        action = "copied_local_provider_response_ref"
    return {
        "action": action,
        "local_path": str(output_path.resolve()),
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha_file(output_path),
        "asset_type": output_path.suffix.lower().lstrip(".") or "unknown",
    }


def _materialize_responses(
    *,
    responses: Sequence[Mapping[str, Any]],
    context: Any,
    palatial_dir: Path,
    download_exports: bool,
    max_export_bytes: int,
) -> Dict[str, Any]:
    exports: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for response_index, response in enumerate(responses):
        candidate_id = _string(response.get("candidate_id")) or f"response_{response_index + 1}"
        refs = _collect_export_refs(response)
        for ref_index, ref_record in enumerate(refs):
            ref = ref_record["ref"]
            suffix = _suffix_for_ref(ref)
            filename = f"{_safe_slug(candidate_id, fallback='asset')}_{ref_index + 1}{suffix}"
            output_path = palatial_dir / "assets" / _safe_slug(candidate_id, fallback="asset") / filename
            record: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "source_ref": ref,
                "source_key": ref_record["source_key"],
                "materialized": False,
                "local_path": None,
            }
            if download_exports or not _is_remote_ref(ref):
                try:
                    record.update(
                        _download_or_copy_ref(
                            ref=ref,
                            output_path=output_path,
                            context=context,
                            max_bytes=max_export_bytes,
                        )
                    )
                    record["materialized"] = True
                except Exception as exc:
                    record["error"] = str(exc)
                    errors.append(
                        {
                            "candidate_id": candidate_id,
                            "source_ref": ref,
                            "error": str(exc),
                        }
                    )
            exports.append(record)
    materialized_count = sum(1 for item in exports if item.get("materialized"))
    return {
        "schema_version": PALATIAL_MATERIALIZATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": "materialized" if materialized_count else "blocked_no_materialized_exports",
        "response_count": len(responses),
        "export_ref_count": len(exports),
        "materialized_export_count": materialized_count,
        "download_exports_requested": download_exports,
        "exports": exports,
        "errors": errors,
        "source_policy": {
            "remote_asset_downloads_performed": bool(
                download_exports and any(_is_remote_ref(item.get("source_ref") or "") for item in exports)
            ),
            "local_provider_response_refs_can_be_copied": True,
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _validate_materialized_assets(materialization: Mapping[str, Any]) -> Dict[str, Any]:
    inspections: List[Dict[str, Any]] = []
    warnings: List[str] = []
    blockers: List[str] = []
    try:
        from .scene_asset_preflight import inspect_scene_asset
    except Exception:
        inspect_scene_asset = None  # type: ignore[assignment]
        warnings.append("scene_asset_preflight_inspector_unavailable")
    for export in materialization.get("exports") or []:
        if not isinstance(export, Mapping) or not export.get("materialized"):
            continue
        path = Path(_string(export.get("local_path")))
        if not path.is_file():
            blockers.append("materialized_export_missing_local_file")
            continue
        if inspect_scene_asset is None:
            inspections.append(
                {
                    "path": str(path.resolve()),
                    "status": "exists_not_inspected",
                    "sha256": _sha_file(path),
                }
            )
            continue
        try:
            inspections.append(inspect_scene_asset(path))
        except Exception as exc:
            inspections.append(
                {
                    "path": str(path.resolve()),
                    "status": "inspection_failed",
                    "error": str(exc),
                    "sha256": _sha_file(path),
                }
            )
            warnings.append("one_or_more_palatial_exports_failed_cpu_inspection")
    if not inspections:
        blockers.append("missing_materialized_palatial_exports")
    real_collision = any(
        _mapping(item.get("collision_evidence")).get("real_collider_proven")
        for item in inspections
    )
    return {
        "schema_version": PALATIAL_VALIDATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "blocked" if blockers else "prepared_for_review",
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "inspection_count": len(inspections),
        "inspections": inspections,
        "real_collider_metadata_present": real_collision,
        "physics_contact_validated": False,
        "simulator_execution_proven": False,
        "robot_readiness_proven": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _load_provider_responses(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    responses: List[Dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value).expanduser()
        payload = read_json_any(path)
        if isinstance(payload, list):
            responses.extend(dict(item) for item in payload if isinstance(item, Mapping))
        elif isinstance(payload, Mapping):
            if isinstance(payload.get("responses"), list):
                responses.extend(
                    dict(item) for item in payload["responses"] if isinstance(item, Mapping)
                )
            else:
                responses.append(dict(payload))
        else:
            raise ValueError(f"Expected object or list provider response at {path}")
    return responses


def build_palatial_physready_assets(
    *,
    capture_root: str | Path,
    object_geometry_manifest: Optional[Mapping[str, Any]] = None,
    task_anchor_manifest: Optional[Mapping[str, Any]] = None,
    object_ids: Sequence[str] = (),
    labels: Sequence[str] = (),
    target_sims: Sequence[str] = DEFAULT_TARGET_SIMS,
    collision: str = "sdf",
    max_candidates: int = 20,
    max_images_per_asset: int = 4,
    include_capture_image_fallback: bool = False,
    allow_live_palatial: bool = False,
    generate_url: Optional[str] = None,
    auth_mode: str = "x-api-key",
    api_key_env: str = PALATIAL_API_KEY_ENV,
    token_price_usd: float = 10.0,
    estimated_tokens_per_asset: float = 1.0,
    download_exports: bool = False,
    max_export_bytes: int = 1_000_000_000,
    provider_response_paths: Sequence[str | Path] = (),
    client: PalatialClientProtocol | None = None,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    palatial_dir = context.pipeline_root / "palatial_physready"
    ensure_dir(palatial_dir)
    env_source = env if env is not None else os.environ
    url = _string(generate_url or env_source.get("PALATIAL_GENERATE_URL")) or PALATIAL_DEFAULT_GENERATE_URL
    auth = _string(env_source.get("PALATIAL_AUTH_MODE") or auth_mode or "x-api-key").lower()
    if auth not in {"x-api-key", "bearer"}:
        raise ValueError("auth_mode must be x-api-key or bearer")

    candidates = build_twin_candidates(
        capture_root=context.capture_root,
        object_geometry_manifest=object_geometry_manifest,
        task_anchor_manifest=task_anchor_manifest,
        object_ids=object_ids,
        labels=labels,
        max_candidates=max_candidates,
        max_images_per_asset=max_images_per_asset,
        include_capture_image_fallback=include_capture_image_fallback,
        target_sims=target_sims,
        collision=collision,
    )
    gate = _live_gate(
        allow_live_palatial=allow_live_palatial,
        env=env_source,
        api_key_env=api_key_env,
    )
    request_manifest = _request_manifest(
        context=context,
        candidates=candidates,
        generate_url=url,
        auth_mode=auth,
        live_gate=gate,
        token_price_usd=token_price_usd,
        estimated_tokens_per_asset=estimated_tokens_per_asset,
    )

    responses = _load_provider_responses(provider_response_paths)
    submissions: List[Dict[str, Any]] = []
    live_errors: List[Dict[str, Any]] = []
    live_calls_performed = False
    if gate["live_provider_calls_allowed"] and request_manifest["request_count"]:
        api_key = _string(env_source.get(api_key_env))
        live_client = client or PalatialApiClient(generate_url=url, api_key=api_key, auth_mode=auth)
        for candidate in candidates.get("candidates") or []:
            if not isinstance(candidate, Mapping):
                continue
            request_payload = _api_payload_for_candidate(candidate)
            try:
                response = dict(live_client.generate_asset(request_payload))
                response.setdefault("candidate_id", candidate.get("candidate_id"))
                response.setdefault("source_object_id", candidate.get("source_object_id"))
                responses.append(response)
                submissions.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "source_object_id": candidate.get("source_object_id"),
                        "status": "submitted",
                        "provider_asset_id": response.get("asset_id")
                        or response.get("project_id")
                        or response.get("id"),
                    }
                )
                live_calls_performed = True
            except Exception as exc:
                live_errors.append(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "source_object_id": candidate.get("source_object_id"),
                        "error": str(exc),
                    }
                )

    materialization = _materialize_responses(
        responses=responses,
        context=context,
        palatial_dir=palatial_dir,
        download_exports=download_exports,
        max_export_bytes=max_export_bytes,
    )
    validation = _validate_materialized_assets(materialization)

    blockers: List[str] = list(candidates.get("blockers") or [])
    warnings: List[str] = list(candidates.get("warnings") or [])
    if allow_live_palatial and not gate["live_provider_calls_allowed"]:
        blockers.extend(gate["missing_gates"])
    if live_errors:
        blockers.append("one_or_more_palatial_live_requests_failed")
    if responses and validation["status"] == "blocked":
        blockers.extend(validation["blockers"])

    if validation["status"] == "prepared_for_review":
        status = "materialized_for_review"
    elif live_errors:
        status = "failed_provider_submission"
    elif allow_live_palatial and not gate["live_provider_calls_allowed"]:
        status = "blocked_missing_live_gates"
    elif gate["live_provider_calls_allowed"] and submissions:
        status = "submitted_waiting_for_exports"
    elif candidates["status"] == "ready":
        status = "ready_for_manual_or_live_submission"
    else:
        status = "blocked"

    run_manifest = {
        "schema_version": PALATIAL_RUN_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "status": status,
        "candidate_manifest_path": "twin_candidate_manifest.json",
        "request_manifest_path": "palatial_request_manifest.json",
        "materialization_manifest_path": "materialization_manifest.json",
        "validation_manifest_path": "validation_manifest.json",
        "candidate_count": candidates["candidate_count"],
        "request_count": request_manifest["request_count"],
        "provider_response_count": len(responses),
        "submission_count": len(submissions),
        "submissions": submissions,
        "live_errors": live_errors,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "live_provider_calls_allowed": bool(gate["live_provider_calls_allowed"]),
        "live_provider_calls_performed": live_calls_performed,
        "remote_asset_downloads_performed": bool(
            materialization.get("source_policy", {}).get("remote_asset_downloads_performed")
        ),
        "local_exports_materialized": materialization["materialized_export_count"],
        "easy_on_off_switch": {
            "off_default": True,
            "enable_command_pattern": (
                f"{PALATIAL_ENABLE_ENV}=true {PALATIAL_API_KEY_ENV}=<secret> "
                "blueprint-build-palatial-physready --capture-root <capture-root> "
                "--allow-live-palatial"
            ),
            "disable_rule": f"unset {PALATIAL_ENABLE_ENV} or omit --allow-live-palatial",
        },
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "live_provider_calls_performed": live_calls_performed,
            "remote_asset_downloads_performed": bool(
                materialization.get("source_policy", {}).get("remote_asset_downloads_performed")
            ),
        },
    }
    run_manifest["deterministic_fingerprint"] = _sha_payload(
        {
            "status": status,
            "candidates": candidates.get("deterministic_fingerprint"),
            "submissions": submissions,
            "responses": [
                response.get("asset_id") or response.get("project_id") or response.get("id")
                for response in responses
            ],
            "materialized": [
                export.get("sha256") or export.get("source_ref")
                for export in materialization.get("exports", [])
            ],
        }
    )

    write_json(palatial_dir / "twin_candidate_manifest.json", candidates)
    write_json(palatial_dir / "palatial_request_manifest.json", request_manifest)
    write_json(palatial_dir / "materialization_manifest.json", materialization)
    write_json(palatial_dir / "validation_manifest.json", validation)
    write_json(palatial_dir / "palatial_physready_run_manifest.json", run_manifest)

    return {
        "schema_version": "palatial_physready_result.v1",
        "capture_root": str(context.capture_root),
        "palatial_dir": str(palatial_dir),
        "status": status,
        "candidate_manifest_path": str((palatial_dir / "twin_candidate_manifest.json").resolve()),
        "request_manifest_path": str((palatial_dir / "palatial_request_manifest.json").resolve()),
        "run_manifest_path": str((palatial_dir / "palatial_physready_run_manifest.json").resolve()),
        "materialization_manifest_path": str((palatial_dir / "materialization_manifest.json").resolve()),
        "validation_manifest_path": str((palatial_dir / "validation_manifest.json").resolve()),
        "claim_boundary": dict(run_manifest["claim_boundary"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build optional Palatial PhysReady twin request/materialization artifacts"
    )
    parser.add_argument("--capture-root", required=True, help="Local capture root path")
    parser.add_argument(
        "--object-id",
        action="append",
        default=[],
        help="Specific capture object id to twin; repeatable",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Specific object label token to twin, e.g. microwave or tote; repeatable",
    )
    parser.add_argument(
        "--target-sim",
        action="append",
        choices=("isaac_sim", "mujoco", "pybullet", "openusd"),
        default=[],
        help="Target simulator/export family; repeatable. Defaults to Isaac Sim and MuJoCo.",
    )
    parser.add_argument("--collision", default="sdf", help="Palatial collision setting")
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-images-per-asset", type=int, default=4)
    parser.add_argument(
        "--include-capture-image-fallback",
        action="store_true",
        help="Use raw capture images only when object-specific crop/image refs are missing",
    )
    parser.add_argument(
        "--allow-live-palatial",
        action="store_true",
        help=f"Permit Palatial API calls only when {PALATIAL_ENABLE_ENV}=true and API key is set",
    )
    parser.add_argument("--generate-url", default=None, help="Override Palatial generate endpoint")
    parser.add_argument(
        "--auth-mode",
        choices=("x-api-key", "bearer"),
        default="x-api-key",
        help="Palatial auth header style",
    )
    parser.add_argument("--api-key-env", default=PALATIAL_API_KEY_ENV)
    parser.add_argument("--token-price-usd", type=float, default=10.0)
    parser.add_argument("--estimated-tokens-per-asset", type=float, default=1.0)
    parser.add_argument(
        "--provider-response",
        action="append",
        default=[],
        help="Local JSON provider response to materialize; repeatable",
    )
    parser.add_argument(
        "--download-exports",
        action="store_true",
        help="Download remote export URLs from provider responses",
    )
    parser.add_argument("--max-export-bytes", type=int, default=1_000_000_000)
    args = parser.parse_args(argv)
    try:
        result = build_palatial_physready_assets(
            capture_root=args.capture_root,
            object_ids=args.object_id,
            labels=args.label,
            target_sims=args.target_sim or DEFAULT_TARGET_SIMS,
            collision=args.collision,
            max_candidates=args.max_candidates,
            max_images_per_asset=args.max_images_per_asset,
            include_capture_image_fallback=args.include_capture_image_fallback,
            allow_live_palatial=args.allow_live_palatial,
            generate_url=args.generate_url,
            auth_mode=args.auth_mode,
            api_key_env=args.api_key_env,
            token_price_usd=args.token_price_usd,
            estimated_tokens_per_asset=args.estimated_tokens_per_asset,
            provider_response_paths=args.provider_response,
            download_exports=args.download_exports,
            max_export_bytes=args.max_export_bytes,
        )
    except (PipelineError, OSError, ValueError, RuntimeError) as exc:
        print(f"[palatial-physready] FAILED: {exc}")
        return 1
    print(f"[palatial-physready] run_manifest={result['run_manifest_path']}")
    print(f"[palatial-physready] request_manifest={result['request_manifest_path']}")
    print(f"[palatial-physready] status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
