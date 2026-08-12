"""Seal exact-mask composites of Aura residual candidates for 1--5 tasks.

Aura's native render is valuable evidence, but it is not allowed to change an
observed retained-scene pixel outside the exact residual mask.  This module
keeps both kinds of output: it verifies every raw Aura candidate frame, then
constructs a composited frame that uses Aura only inside the packet's exact
mask and copies the retained GPU frame byte-for-byte everywhere else.  The
result is a sealed camera manifest for locality and multi-view review; it is
not an inpainting-success, removal, simulator, or physical claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_aura_exact_residual_preflight import (
    SCHEMA_VERSION as PREFLIGHT_SCHEMA,
)


SCHEMA_VERSION = "public_scene_aura_exact_residual_composite.v1"
RENDER_SCHEMA = "sealed_camera_render_manifest.v1"
RAW_RESULT_SCHEMA = "public_scene_aura_exact_residual_raw_result.v1"
VAST_TEARDOWN_SCHEMA = "vast_teardown_manifest.v1"
VAST_FINAL_VALIDATION_SCHEMA = "vast_final_validation.v1"


class AuraExactResidualCompositeError(ValueError):
    """Stable failures for exact-mask residual result composition."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraExactResidualCompositeError([code]) from exc
    if not isinstance(value, dict):
        raise AuraExactResidualCompositeError([code])
    return value


def _file(path_value: Any, *, code: str) -> Path:
    path = Path(str(path_value or "")).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise AuraExactResidualCompositeError([code])
    return path


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    result = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    result["relative_path" if root is not None else "path"] = (
        path.relative_to(root).as_posix() if root is not None else str(path)
    )
    return result


def _bound_absolute(record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise AuraExactResidualCompositeError([code])
    path = _file(record.get("path"), code=code)
    if path.stat().st_size != record.get("size_bytes") or _sha256(path) != record.get("sha256"):
        raise AuraExactResidualCompositeError([code])
    return path


def _bound_relative(root: Path, record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise AuraExactResidualCompositeError([code])
    relative = str(record.get("relative_path") or "")
    expected = record.get("sha256") if record.get("sha256") is not None else record.get("digest")
    if (
        not relative
        or relative.startswith("/")
        or ".." in Path(relative).parts
        or (
            record.get("sha256") is not None
            and record.get("digest") is not None
            and record.get("sha256") != record.get("digest")
        )
    ):
        raise AuraExactResidualCompositeError([code])
    path = (root / relative).resolve()
    if (
        (root != path and root not in path.parents)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != expected
    ):
        raise AuraExactResidualCompositeError([code])
    return path


def _image(path: Path, *, mode: str, code: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert(mode), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise AuraExactResidualCompositeError([code]) from exc


def _preflight(path: Path) -> dict[str, Any]:
    value = _read(path, code="aura_exact_residual_preflight_unreadable")
    if (
        value.get("schema_version") != PREFLIGHT_SCHEMA
        or value.get("status") != "prepared_no_upload_no_execution"
        or value.get("preflight_digest")
        != canonical_digest(value, digest_field="preflight_digest")
        or value.get("execution", {}).get("provider_mutations_performed") != 0
        or value.get("execution", {}).get("aura_inpainting_executed") is not False
        or value.get("required_result_checks", {}).get("outside_mask_pixel_delta_required")
        != 0
        or value.get("required_result_checks", {}).get("locality_mask_dilation_pixels")
        != 0
    ):
        raise AuraExactResidualCompositeError(["aura_exact_residual_preflight_invalid"])
    return value


def _input_rows(preflight: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = preflight.get("camera_inputs")
    if not isinstance(rows, list) or not rows:
        raise AuraExactResidualCompositeError(["aura_exact_residual_camera_inputs_missing"])
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AuraExactResidualCompositeError(["aura_exact_residual_camera_input_invalid"])
        task_id = str(row.get("task_id") or "")
        camera_id = str(row.get("camera_id") or "")
        key = (task_id, camera_id)
        if not task_id or not camera_id or key in result:
            raise AuraExactResidualCompositeError(["aura_exact_residual_camera_input_invalid"])
        result[key] = dict(row)
    return result


def _raw_result_rows(
    *, preflight: Mapping[str, Any], raw_result_path: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    result = _read(raw_result_path, code="aura_exact_residual_raw_result_unreadable")
    if (
        result.get("schema_version") != RAW_RESULT_SCHEMA
        or result.get("status") != "aura_native_residual_frames_rendered"
        or result.get("preflight_digest") != preflight.get("preflight_digest")
        or result.get("aura_inpainting_executed") is not True
        or result.get("provider_mutations_performed") != 1
        or result.get("learned_policy_outcomes_accessed") is not False
        or result.get("result_digest") != canonical_digest(result, digest_field="result_digest")
    ):
        raise AuraExactResidualCompositeError(["aura_exact_residual_raw_result_invalid"])
    rows = result.get("frames")
    if not isinstance(rows, list) or not rows:
        raise AuraExactResidualCompositeError(["aura_exact_residual_raw_frames_missing"])
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise AuraExactResidualCompositeError(["aura_exact_residual_raw_frame_invalid"])
        key = (str(row.get("task_id") or ""), str(row.get("camera_id") or ""))
        if not all(key) or key in indexed:
            raise AuraExactResidualCompositeError(["aura_exact_residual_raw_frame_invalid"])
        indexed[key] = dict(row)
    if set(indexed) != set(_input_rows(preflight)):
        raise AuraExactResidualCompositeError(["aura_exact_residual_raw_camera_set_mismatch"])
    return indexed, _provider_closeout(result.get("provider_closeout"))


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _provider_closeout(value: Any) -> dict[str, Any]:
    """Accept a paid native result only after independent provider-zero evidence.

    The raw Aura result cannot turn a set of caller booleans into a completion
    claim.  It has to bind the four independently retained files which close
    the allocator lane: the adapter result, terminal teardown, final validator,
    and the independent watchdog's provider-inventory observation.
    """

    if not isinstance(value, Mapping):
        raise AuraExactResidualCompositeError(["aura_exact_residual_provider_zero_not_proven"])
    try:
        adapter_path = _bound_absolute(
            value.get("adapter_result"), code="aura_exact_residual_provider_zero_not_proven"
        )
        teardown_path = _bound_absolute(
            value.get("teardown_manifest"), code="aura_exact_residual_provider_zero_not_proven"
        )
        final_path = _bound_absolute(
            value.get("final_validation"), code="aura_exact_residual_provider_zero_not_proven"
        )
        watchdog_path = _bound_absolute(
            value.get("watchdog_receipt"), code="aura_exact_residual_provider_zero_not_proven"
        )
    except AuraExactResidualCompositeError as exc:
        raise AuraExactResidualCompositeError(["aura_exact_residual_provider_zero_not_proven"]) from exc
    adapter = _read(adapter_path, code="aura_exact_residual_provider_zero_not_proven")
    teardown = _read(teardown_path, code="aura_exact_residual_provider_zero_not_proven")
    final = _read(final_path, code="aura_exact_residual_provider_zero_not_proven")
    watchdog = _read(watchdog_path, code="aura_exact_residual_provider_zero_not_proven")
    instance_ids = teardown.get("vast_instance_ids")
    actions = teardown.get("teardown_actions_performed")
    destroyed_ids = {
        row.get("instance_id")
        for row in actions or []
        if isinstance(row, Mapping)
        and row.get("action") == "destroy_instance"
        and row.get("status") == "completed"
        and isinstance(row.get("http_status_code"), int)
        and 200 <= row["http_status_code"] < 300
    }
    final_inventory = watchdog.get("final_inventory")
    final_global_inventory = watchdog.get("final_global_inventory")
    if (
        adapter.get("api_call_performed") is not True
        or adapter.get("provider_create_attempted") is not True
        or adapter.get("final_validation_status") != "passed"
        or adapter.get("continuing_spend_from_this_run") is not False
        or adapter.get("all_staged_objects_absent") is not True
        or not _positive_number(adapter.get("estimated_cost_usd"))
        or not isinstance(adapter.get("hard_ttl_seconds"), int)
        or adapter["hard_ttl_seconds"] <= 0
        or teardown.get("schema_version") != VAST_TEARDOWN_SCHEMA
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or not isinstance(instance_ids, list)
        or not instance_ids
        or set(instance_ids) != destroyed_ids
        or final.get("schema_version") != VAST_FINAL_VALIDATION_SCHEMA
        or final.get("status") != "passed"
        or final.get("all_vast_instances_destroyed_by_adapter") is not True
        or final.get("continuing_spend_from_this_run") is not False
        or not _positive_number(final.get("estimated_cost_usd"))
        or final.get("estimated_cost_usd") != adapter.get("estimated_cost_usd")
        or watchdog.get("status") != "provider_terminal"
        or watchdog.get("independent_process") is not True
        or watchdog.get("provider_absence_confirmed") is not True
        or not isinstance(final_inventory, Mapping)
        or final_inventory.get("api_confirmed") is not True
        or final_inventory.get("live_resource_count") != 0
        or not isinstance(final_global_inventory, Mapping)
        or final_global_inventory.get("api_confirmed") is not True
        or final_global_inventory.get("live_resource_count") != 0
        or watchdog.get("recorded_vast_instance_teardown", {}).get("provider_absence_confirmed")
        is not True
    ):
        raise AuraExactResidualCompositeError(["aura_exact_residual_provider_zero_not_proven"])
    return {
        "adapter_result": _record(adapter_path),
        "teardown_manifest": _record(teardown_path),
        "final_validation": _record(final_path),
        "watchdog_receipt": _record(watchdog_path),
        "actual_cost_usd": adapter["estimated_cost_usd"],
        "hard_ttl_seconds": adapter["hard_ttl_seconds"],
        "destroyed_vast_instance_ids": sorted(instance_ids),
    }


def materialize_aura_exact_residual_composite(
    *, preflight_path: str | Path, raw_result_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """Composite verified raw Aura outputs into the exact frozen residual masks."""

    preflight_file = _file(preflight_path, code="aura_exact_residual_preflight_missing")
    raw_result_file = _file(raw_result_path, code="aura_exact_residual_raw_result_missing")
    preflight = _preflight(preflight_file)
    inputs = _input_rows(preflight)
    raw_rows, provider_closeout = _raw_result_rows(
        preflight=preflight, raw_result_path=raw_result_file
    )
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise AuraExactResidualCompositeError(["aura_exact_residual_composite_output_not_empty"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    before_dirs: dict[str, Path] = {}
    mask_dirs: dict[str, Path] = {}
    for (task_id, camera_id), input_row in sorted(inputs.items()):
        raw = raw_rows[(task_id, camera_id)]
        before = _bound_absolute(
            input_row["retained_scene_before"], code="aura_exact_residual_before_frame_invalid"
        )
        mask = _bound_absolute(
            input_row["exact_residual_mask"], code="aura_exact_residual_mask_invalid"
        )
        raw_path = _bound_absolute(raw.get("native_aura_frame"), code="aura_exact_residual_raw_frame_invalid")
        before_rgb = _image(before, mode="RGB", code="aura_exact_residual_before_frame_invalid")
        raw_rgb = _image(raw_path, mode="RGB", code="aura_exact_residual_raw_frame_invalid")
        mask_pixels = _image(mask, mode="L", code="aura_exact_residual_mask_invalid")
        if (
            before_rgb.shape != raw_rgb.shape
            or mask_pixels.shape != before_rgb.shape[:2]
            or set(mask_pixels.tobytes()) - {0, 255}
            or not np.any(mask_pixels)
        ):
            raise AuraExactResidualCompositeError(["aura_exact_residual_frame_shape_or_mask_invalid"])
        composed = before_rgb.copy()
        edit = mask_pixels > 0
        composed[edit] = raw_rgb[edit]
        task_root = output / task_id
        frames_root = task_root / "frames"
        before_root = task_root / "before"
        masks_root = task_root / "masks"
        frames_root.mkdir(parents=True, exist_ok=True)
        before_root.mkdir(exist_ok=True)
        masks_root.mkdir(exist_ok=True)
        before_copy = before_root / f"{camera_id}.png"
        mask_copy = masks_root / f"{camera_id}.png"
        composed_path = frames_root / f"{camera_id}.png"
        Image.fromarray(before_rgb, mode="RGB").save(before_copy)
        Image.fromarray(mask_pixels, mode="L").save(mask_copy)
        Image.fromarray(composed, mode="RGB").save(composed_path)
        outside = ~edit
        changed = int(np.count_nonzero(np.any(composed[outside] != before_rgb[outside], axis=1)))
        if changed != 0:
            raise AuraExactResidualCompositeError(["aura_exact_residual_outside_mask_changed"])
        before_dirs[task_id] = before_root
        mask_dirs[task_id] = masks_root
        frame_record = {
            "camera_id": camera_id,
            "relative_path": (Path("frames") / composed_path.name).as_posix(),
            "size_bytes": composed_path.stat().st_size,
            "digest": _sha256(composed_path),
            "width": int(composed.shape[1]),
            "height": int(composed.shape[0]),
        }
        manifest_rows.setdefault(task_id, []).append(frame_record)
        rows.append(
            {
                "task_id": task_id,
                "camera_id": camera_id,
                "retained_scene_before": _record(before_copy, root=task_root),
                "exact_residual_mask": _record(mask_copy, root=task_root),
                "native_aura_frame": _record(raw_path),
                "exact_mask_composited_frame": _record(composed_path, root=task_root),
                "exact_mask_pixel_count": int(np.count_nonzero(edit)),
                "outside_mask_pixel_count": int(np.count_nonzero(outside)),
                "outside_mask_changed_pixels": changed,
            }
        )
    manifests: list[dict[str, Any]] = []
    for task_id, renders in sorted(manifest_rows.items()):
        task_root = output / task_id
        manifest: dict[str, Any] = {
            "schema_version": RENDER_SCHEMA,
            "status": "rendered_exact_cameras",
            "authorization_class": "method_input",
            "source_layer_role": "shared_retained_scene_exact_residual_composited",
            "rendered_by": "exact_residual_mask_compositor_over_native_aura_candidate",
            "render_settings": {
                "dimensions": {"width": renders[0]["width"], "height": renders[0]["height"]},
                "background_rgb": "preserved_from_retained_scene",
            },
            "scene": {"publisher_scene_id": "840920", "target_instance_id": task_id},
            "preflight_digest": preflight["preflight_digest"],
            "raw_aura_result": _record(raw_result_file),
            "renders": sorted(renders, key=lambda row: row["camera_id"]),
            "sealed_camera_render_manifest_digest": "",
        }
        manifest["sealed_camera_render_manifest_digest"] = canonical_digest(
            manifest, digest_field="sealed_camera_render_manifest_digest"
        )
        manifest_path = task_root / "sealed_camera_render_manifest.v1.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        manifests.append(
            {
                "task_id": task_id,
                "manifest": _record(manifest_path),
                "sealed_camera_render_manifest_digest": manifest[
                    "sealed_camera_render_manifest_digest"
                ],
                "before_dir": str(before_dirs[task_id]),
                "mask_dir": str(mask_dirs[task_id]),
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "exact_mask_composites_materialized_unqualified",
        "preflight": {**_record(preflight_file), "preflight_digest": preflight["preflight_digest"]},
        "raw_aura_result": {**_record(raw_result_file), "result_digest": _read(raw_result_file, code="aura_exact_residual_raw_result_unreadable")["result_digest"]},
        "provider_closeout": provider_closeout,
        "frames": rows,
        "task_render_manifests": manifests,
        "replacement_object_count": preflight["replacement_object_count"],
        "outside_mask_pixel_delta_required": 0,
        "outside_mask_changed_pixels_total": sum(row["outside_mask_changed_pixels"] for row in rows),
        "multi_view_consistency_required": True,
        "claim_boundary": {
            "native_aura_frames_retained": True,
            "outside_mask_pixels_copied_exactly_from_retained_scene": True,
            "inpainting_result_qualified": False,
            "gaussian_source_removal_qualified": False,
            "native_simulator_import_qualified": False,
        },
        "composite_digest": "",
    }
    receipt["composite_digest"] = canonical_digest(receipt, digest_field="composite_digest")
    (output / f"{SCHEMA_VERSION}.json").write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


__all__ = [
    "AuraExactResidualCompositeError",
    "SCHEMA_VERSION",
    "materialize_aura_exact_residual_composite",
]
