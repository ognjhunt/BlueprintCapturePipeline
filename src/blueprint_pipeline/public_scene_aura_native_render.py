"""Bind AuraFusion360's native 2DGS renders to the frozen scene cameras."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_aura_adapter import SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION
from .public_scene_aura_execution import SCHEMA_VERSION as EXECUTION_SCHEMA_VERSION
from .sealed_camera_render import (
    PROJECTION_PIXEL_CONVENTION,
    RENDER_MANIFEST_SCHEMA_VERSION,
)


class AuraNativeRenderManifestError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuraNativeRenderManifestError([code]) from exc
    if not isinstance(value, dict):
        raise AuraNativeRenderManifestError([code])
    return value


def _under(path: str | Path, root: Path, code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(root) or resolved.is_symlink():
        raise AuraNativeRenderManifestError([code])
    return resolved


def materialize_aura_native_render_manifest(
    *,
    adapter_receipt_path: str | Path,
    execution_receipt_path: str | Path,
    evidence_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Open and bind method-native frames; caller assertions cannot admit them."""

    evidence = Path(evidence_root).expanduser().resolve()
    if not evidence.is_dir():
        raise AuraNativeRenderManifestError(["aura_native_evidence_root_missing"])
    adapter_path = Path(adapter_receipt_path).expanduser().resolve()
    execution_path = Path(execution_receipt_path).expanduser().resolve()
    output = _under(output_path, evidence, "aura_native_output_outside_evidence_root")
    adapter = _read(adapter_path, "aura_native_adapter_receipt_invalid")
    execution = _read(execution_path, "aura_native_execution_receipt_invalid")
    if (
        adapter.get("schema_version") != ADAPTER_SCHEMA_VERSION
        or adapter.get("status") != "prepared_unexecuted"
        or canonical_digest(adapter, digest_field="receipt_digest")
        != adapter.get("receipt_digest")
    ):
        raise AuraNativeRenderManifestError(["aura_native_adapter_receipt_invalid"])
    if (
        execution.get("schema_version") != EXECUTION_SCHEMA_VERSION
        or execution.get("status") != "executed_candidate"
        or canonical_digest(execution, digest_field="receipt_digest")
        != execution.get("receipt_digest")
    ):
        raise AuraNativeRenderManifestError(["aura_native_execution_receipt_invalid"])
    prepared = execution.get("prepared_adapter") or {}
    if (
        prepared.get("receipt_digest") != adapter.get("receipt_digest")
        or prepared.get("sha256") != _sha256(adapter_path)
        or prepared.get("size_bytes") != adapter_path.stat().st_size
    ):
        raise AuraNativeRenderManifestError(["aura_native_adapter_binding_mismatch"])

    scene = adapter.get("scene") or {}
    execution_scene = execution.get("scene") or {}
    scene_id = str(scene.get("publisher_scene_id") or "")
    target_id = str(scene.get("target_instance_id") or "")
    if (
        not scene_id.isdigit()
        or not (
            target_id.isdigit()
            or (target_id.startswith("ins") and target_id[3:].isdigit())
        )
        or str(execution_scene.get("publisher_scene_id") or "") != scene_id
        or str(execution_scene.get("target_instance_id") or "") != target_id
    ):
        raise AuraNativeRenderManifestError(
            ["aura_native_scene_target_binding_invalid"]
        )
    camera_ids = sorted(
        Path(str(row.get("relative_path") or "")).stem
        for row in (adapter.get("artifacts") or [])
        if str(row.get("relative_path") or "").startswith("data/Other-360/")
        and "/images/" in str(row.get("relative_path") or "")
        and str(row.get("relative_path") or "").endswith(".png")
    )
    camera_count = int(scene.get("camera_count") or 0)
    if (
        not camera_ids
        or len(camera_ids) != camera_count
        or len(set(camera_ids)) != len(camera_ids)
    ):
        raise AuraNativeRenderManifestError(["aura_native_camera_inventory_invalid"])

    runtime_result = (execution.get("execution") or {}).get("runtime_result") or {}
    runtime_path = _under(
        str(runtime_result.get("path") or ""),
        evidence,
        "aura_native_runtime_result_outside_evidence_root",
    )
    if (
        not runtime_path.is_file()
        or runtime_path.stat().st_size != runtime_result.get("size_bytes")
        or _sha256(runtime_path) != runtime_result.get("sha256")
    ):
        raise AuraNativeRenderManifestError(["aura_native_runtime_result_changed"])
    artifact_root = runtime_path.parent
    if output.parent.resolve() != artifact_root:
        raise AuraNativeRenderManifestError(["aura_native_manifest_not_at_artifact_root"])

    frame_records = (execution.get("execution") or {}).get("final_frames") or []
    source_resolution = scene.get("source_resolution") or []
    if len(frame_records) != camera_count or len(source_resolution) != 2:
        raise AuraNativeRenderManifestError(["aura_native_frame_set_incomplete"])
    renders: list[dict[str, Any]] = []
    for index, (camera_id, record) in enumerate(zip(camera_ids, frame_records, strict=True)):
        expected_relative = f"artifacts/final_frames/{index:05d}.png"
        if record.get("relative_path") != expected_relative:
            raise AuraNativeRenderManifestError(["aura_native_frame_order_invalid"])
        frame = (artifact_root / expected_relative).resolve()
        if (
            not frame.is_relative_to(artifact_root)
            or not frame.is_file()
            or frame.is_symlink()
            or frame.stat().st_size != record.get("size_bytes")
            or _sha256(frame) != record.get("sha256")
        ):
            raise AuraNativeRenderManifestError([f"aura_native_frame_changed:{camera_id}"])
        with Image.open(frame) as image:
            pixels = np.asarray(image.convert("RGB"))
            width, height = image.size
        if [width, height] != source_resolution or float(pixels.std()) == 0.0:
            raise AuraNativeRenderManifestError([f"aura_native_frame_invalid:{camera_id}"])
        renders.append(
            {
                "camera_id": camera_id,
                "relative_path": expected_relative,
                "digest": record["sha256"],
                "width": width,
                "height": height,
                "pixel_std": round(float(pixels.std()), 4),
            }
        )

    source = execution.get("source") or {}
    point_cloud = (execution.get("execution") or {}).get("final_point_cloud") or {}
    manifest: dict[str, Any] = {
        "schema_version": RENDER_MANIFEST_SCHEMA_VERSION,
        "status": "rendered_exact_cameras",
        "rendered_by": "aurafusion360_native_2d_gaussian_rasterizer",
        "camera_set_label": f"adp009b_{scene_id}_{target_id}_frozen_{camera_count}",
        "scene": {
            "publisher_scene_id": scene_id,
            "target_instance_id": target_id,
        },
        "provider_splat_import_receipt_digest": execution.get("receipt_digest"),
        "provider_reconstruction_alignment_digest": scene.get("input_receipt_digest"),
        "splat_digest": point_cloud.get("sha256"),
        "splat_representation": "2d_gaussian_surfels_scale_0_scale_1",
        "projection_pixel_convention": PROJECTION_PIXEL_CONVENTION,
        "renderer_identity": {
            "repository": source.get("repository"),
            "commit": source.get("commit"),
            "tree": source.get("tree"),
            "source_modified": False,
            "camera_order_derivation": "sorted_adapter_image_stems_with_shuffle_false",
            "renderer_independent_of_method": False,
        },
        "renders": renders,
        "render_count": len(renders),
        "rendered_by_isaac_rtx": False,
        "hidden_pixels_read_by_renderer": False,
        "proof_effect": "method_native_exact_camera_render_for_independent_locality_measurement",
        "claim_ceiling": "aurafusion360_visual_candidate",
    }
    manifest["sealed_camera_render_manifest_digest"] = canonical_digest(
        manifest, digest_field="sealed_camera_render_manifest_digest"
    )
    output.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-receipt", required=True)
    parser.add_argument("--execution-receipt", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = materialize_aura_native_render_manifest(
        adapter_receipt_path=args.adapter_receipt,
        execution_receipt_path=args.execution_receipt,
        evidence_root=args.evidence_root,
        output_path=args.output,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
