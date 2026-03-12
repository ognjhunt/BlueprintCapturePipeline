"""Normalize remote world-model outputs into the Stage 1 contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping
from urllib import parse as urllib_parse


REQUIRED_STAGE1_ARTIFACTS = (
    "export_last.usdz",
    "nvblox_mesh.ply",
    "visual_mesh.glb",
    "mesh_manifest.json",
    "occupancy.bin",
    "object_point_cloud_index.json",
    "capture_quality_report.json",
)

_ALIAS_KEYS = {
    "export_last.usdz": ("export_last.usdz", "export_last_usdz", "usdz", "primary_visual_usdz"),
    "nvblox_mesh.ply": ("nvblox_mesh.ply", "nvblox_mesh_ply", "collision_mesh", "collision_mesh_ply"),
    "visual_mesh.glb": ("visual_mesh.glb", "visual_mesh_glb", "visual_mesh", "glb"),
    "occupancy.bin": ("occupancy.bin", "occupancy_bin", "occupancy"),
}


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _coerce_local_path(raw: Any) -> Path | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    parsed = urllib_parse.urlparse(text)
    if parsed.scheme in {"http", "https", "gs"}:
        return None
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(text)


def _resolve_artifact_path(result_manifest: Mapping[str, Any], target_name: str) -> Path | None:
    normalized = result_manifest.get("normalized_contract")
    if isinstance(normalized, Mapping):
        direct = _coerce_local_path(normalized.get(target_name))
        if direct is not None:
            return direct
    artifacts = result_manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        for alias in _ALIAS_KEYS.get(target_name, (target_name,)):
            direct = _coerce_local_path(artifacts.get(alias))
            if direct is not None:
                return direct
    direct = _coerce_local_path(result_manifest.get(target_name))
    if direct is not None:
        return direct
    return None


def _copy_required_artifact(
    result_manifest: Mapping[str, Any],
    *,
    artifact_name: str,
    output_dir: Path,
    notes: list[str],
) -> None:
    source = _resolve_artifact_path(result_manifest, artifact_name)
    if source is None or not source.is_file():
        raise RuntimeError(f"missing normalized artifact '{artifact_name}'")
    destination = output_dir / artifact_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
        notes.append(f"copied:{artifact_name}")
    else:
        notes.append(f"retained:{artifact_name}")


def _normalize_json_payload(
    *,
    result_manifest: Mapping[str, Any],
    payload_key: str,
    artifact_name: str,
    fallback_payload: Mapping[str, Any],
    output_dir: Path,
    notes: list[str],
) -> None:
    if (output_dir / artifact_name).is_file():
        return
    payload = result_manifest.get(payload_key)
    if isinstance(payload, Mapping):
        _write_json(output_dir / artifact_name, payload)
        notes.append(f"materialized:{artifact_name}:{payload_key}")
        return
    if fallback_payload:
        _write_json(output_dir / artifact_name, fallback_payload)
        notes.append(f"materialized:{artifact_name}:fallback")
        return
    raise RuntimeError(f"missing normalized artifact '{artifact_name}'")


def normalize_remote_result(
    *,
    backend: str,
    result_manifest: Mapping[str, Any],
    output_dir: Path,
    backend_report_path: Path,
) -> Mapping[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    for artifact_name in ("export_last.usdz", "nvblox_mesh.ply", "visual_mesh.glb", "occupancy.bin"):
        _copy_required_artifact(
            result_manifest,
            artifact_name=artifact_name,
            output_dir=output_dir,
            notes=notes,
        )

    mesh_manifest_fallback = {
        "primary_visual_asset": "export_last.usdz",
        "source": backend,
        "notes": "created_by_world_model_contract_adapter",
    }
    _normalize_json_payload(
        result_manifest=result_manifest,
        payload_key="mesh_manifest",
        artifact_name="mesh_manifest.json",
        fallback_payload=mesh_manifest_fallback,
        output_dir=output_dir,
        notes=notes,
    )
    _normalize_json_payload(
        result_manifest=result_manifest,
        payload_key="object_point_cloud_index",
        artifact_name="object_point_cloud_index.json",
        fallback_payload={},
        output_dir=output_dir,
        notes=notes,
    )
    _normalize_json_payload(
        result_manifest=result_manifest,
        payload_key="capture_quality_report",
        artifact_name="capture_quality_report.json",
        fallback_payload={},
        output_dir=output_dir,
        notes=notes,
    )

    for artifact_name in REQUIRED_STAGE1_ARTIFACTS:
        path = output_dir / artifact_name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"required Stage 1 artifact missing after normalization: {artifact_name}")

    existing_report = _load_json(backend_report_path)
    report = dict(existing_report)
    report["normalization"] = {
        "backend": backend,
        "required_artifacts": list(REQUIRED_STAGE1_ARTIFACTS),
        "notes": notes,
        "native_manifest_present": bool(result_manifest),
    }
    report["native_result_manifest"] = result_manifest
    _write_json(backend_report_path, report)
    return report
