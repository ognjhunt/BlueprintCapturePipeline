"""Helpers for resolving local capture folder structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import PipelineError


@dataclass(frozen=True)
class LocalCaptureContext:
    capture_root: Path
    raw_root: Path
    pipeline_root: Path
    descriptor_path: Path
    raw_complete_path: Path
    storage_root: Path
    bucket: str
    scene_id: str
    capture_id: str

    @property
    def capture_prefix(self) -> str:
        return f"scenes/{self.scene_id}/captures/{self.capture_id}"

    @property
    def descriptor_uri(self) -> str:
        return f"gs://{self.bucket}/{self.capture_prefix}/capture_descriptor.json"

    @property
    def raw_prefix_uri(self) -> str:
        return f"gs://{self.bucket}/{self.capture_prefix}/raw"


def resolve_local_capture_context(path: str | Path) -> LocalCaptureContext:
    candidate = Path(path).resolve()
    parts = candidate.parts
    if "scenes" not in parts:
        raise PipelineError(
            "Path is not inside a scenes/<scene>/captures/<capture> tree: "
            f"{candidate}. Stage raw download folders first, for example with "
            "`python3 scripts/stage_capture_bundle.py --source-bundle <download-folder> "
            "--storage-root <storage-root> --bucket <bucket>`, so the capture lives under "
            "`<storage-root>/<bucket>/scenes/<scene_id>/captures/<capture_id>`."
        )
    idx = parts.index("scenes")
    if idx < 1 or len(parts) <= idx + 3 or parts[idx + 2] != "captures":
        raise PipelineError(
            "Path does not match scenes/<scene_id>/captures/<capture_id>: "
            f"{candidate}. Expected layout: "
            "`<storage-root>/<bucket>/scenes/<scene_id>/captures/<capture_id>`."
        )

    scene_id = parts[idx + 1]
    capture_id = parts[idx + 3]
    bucket = parts[idx - 1]

    if idx - 1 == 0:
        storage_root = Path(parts[0])
    else:
        storage_root = Path(*parts[: idx - 1])

    capture_root = Path(*parts[: idx + 4])
    return LocalCaptureContext(
        capture_root=capture_root,
        raw_root=capture_root / "raw",
        pipeline_root=capture_root / "pipeline",
        descriptor_path=capture_root / "capture_descriptor.json",
        raw_complete_path=capture_root / "raw" / "capture_upload_complete.json",
        storage_root=storage_root,
        bucket=bucket,
        scene_id=scene_id,
        capture_id=capture_id,
    )
