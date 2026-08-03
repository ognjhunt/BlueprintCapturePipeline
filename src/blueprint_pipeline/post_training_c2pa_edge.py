"""Sidecar-only C2PA stamping for post-training compatibility exports."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from .c2pa_stamping import SCHEMA_VERSION as C2PA_SCHEMA_VERSION
from .c2pa_stamping import apply_edge_stamping


ALLOWED_MEDIA_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_relative_paths(output_dir: Path) -> list[str]:
    roots = (
        output_dir / "exports" / "video_bundle" / "objects",
        output_dir / "exports" / "lerobot_v3" / "videos",
        output_dir / "exports" / "gr00t_lerobot" / "videos",
    )
    media: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in ALLOWED_MEDIA_SUFFIXES:
                media.append(str(path.relative_to(output_dir)))
    return media


def apply_post_training_c2pa_edge_stamping(
    output_dir: Path,
    manifest: MutableMapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    """Stamp media sidecars without mutating content-addressed asset bytes."""

    raw_context = manifest.get("context")
    context = dict(raw_context) if isinstance(raw_context, Mapping) else {}
    ledger_refs: dict[str, Any] = {
        "scene_id": str(context.get("scene_id") or "unknown"),
        "capture_id": str(context.get("capture_id") or "unknown"),
    }
    for ref_key, artifact_name in (
        ("consent_evidence_digest", "consent_evidence.json"),
        ("signed_chain_manifest_sha256", "canonical_training_quality_pipeline.json"),
    ):
        artifact_path = output_dir / artifact_name
        if artifact_path.is_file():
            ledger_refs[ref_key] = f"sha256:{_sha_file(artifact_path)}"
    holdout_path = output_dir / "holdout_split.json"
    if holdout_path.is_file():
        try:
            holdout_sha = str(
                json.loads(holdout_path.read_text(encoding="utf-8")).get("split_sha256") or ""
            )
        except (OSError, json.JSONDecodeError):
            holdout_sha = ""
        if holdout_sha:
            ledger_refs["holdout_split_sha256"] = holdout_sha
    try:
        record = apply_edge_stamping(
            package_dir=output_dir,
            media_relative_paths=_media_relative_paths(output_dir),
            ledger_refs=ledger_refs,
            env=env,
        )
        summary = {
            key: record.get(key)
            for key in (
                "schema_version",
                "status",
                "sidecar_only",
                "internal_ledger_authoritative",
                "total_media_count",
                "stamped_count",
                "blockers",
                "record_path",
            )
        }
    except Exception as exc:  # noqa: BLE001 - stamping must never block the export
        summary = {
            "schema_version": C2PA_SCHEMA_VERSION,
            "status": "failed",
            "sidecar_only": True,
            "internal_ledger_authoritative": True,
            "total_media_count": 0,
            "stamped_count": 0,
            "blockers": [f"c2pa_stamping_exception:{type(exc).__name__}"],
            "record_path": None,
        }
    manifest["c2pa_edge_stamping"] = summary


__all__ = ["apply_post_training_c2pa_edge_stamping"]
