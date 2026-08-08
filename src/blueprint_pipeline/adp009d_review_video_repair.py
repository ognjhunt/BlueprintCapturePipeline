"""Re-derive reviewable ADP-009D videos from retained lossless frame manifests.

This is intentionally a derived-media operation.  It never overwrites the
immutable episode output and it refuses to encode until every ordered PNG still
matches the digest-bound frame manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import (
    FRAME_MANIFEST_SCHEMA_VERSION,
    _encode_episode_video,
)

SCHEMA_VERSION = "adp009d_review_video_repair.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_source_path(source_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("review_video_source_path_unsafe")
    resolved = (source_root / relative).resolve()
    try:
        resolved.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueError("review_video_source_path_unsafe") from exc
    return resolved


def _manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    inputs = manifest.get("policy_input_frames")
    terminal = manifest.get("terminal_observation")
    if not isinstance(inputs, list) or not isinstance(terminal, Mapping):
        raise ValueError("review_video_frame_manifest_rows_invalid")
    rows = [dict(row) for row in inputs if isinstance(row, Mapping)]
    if len(rows) != len(inputs):
        raise ValueError("review_video_frame_manifest_rows_invalid")
    rows.append(dict(terminal))
    expected_order = [str(row.get("relative_path") or "") for row in rows]
    if manifest.get("video_frame_order") != expected_order:
        raise ValueError("review_video_frame_order_mismatch")
    return rows


def rederive_review_video(
    *,
    frame_manifest_path: Path,
    output_dir: Path,
    frames_per_second: float = 4.0,
) -> dict[str, Any]:
    """Create a new H.264 derivative without mutating the source episode."""

    manifest_path = frame_manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("review_video_frame_manifest_not_mapping")
    if manifest.get("schema_version") != FRAME_MANIFEST_SCHEMA_VERSION:
        raise ValueError("review_video_frame_manifest_schema_invalid")
    expected_digest = canonical_digest(
        manifest, digest_field="frame_manifest_digest"
    )
    if manifest.get("frame_manifest_digest") != expected_digest:
        raise ValueError("review_video_frame_manifest_digest_mismatch")

    # Canonical manifests live at <source_root>/media/<episode>/frame_manifest.json.
    if len(manifest_path.parents) < 3 or manifest_path.parent.parent.name != "media":
        raise ValueError("review_video_frame_manifest_location_invalid")
    source_root = manifest_path.parents[2]
    frame_paths: list[Path] = []
    for row in _manifest_rows(manifest):
        path = _safe_source_path(source_root, str(row.get("relative_path") or ""))
        if not path.is_file() or _sha256(path) != row.get("png_sha256"):
            raise ValueError("review_video_source_png_digest_mismatch")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (
                int(row.get("width", -1)),
                int(row.get("height", -1)),
            ):
                raise ValueError("review_video_source_png_shape_invalid")
        frame_paths.append(path)

    episode_id = str(manifest.get("episode_id") or "").strip()
    if not episode_id or Path(episode_id).name != episode_id:
        raise ValueError("review_video_episode_id_invalid")
    target_dir = output_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    video_path = target_dir / f"{episode_id}.mp4"
    receipt_path = target_dir / f"{episode_id}.review_video_repair.json"
    if video_path.exists() or video_path.is_symlink() or receipt_path.exists():
        raise FileExistsError("review_video_repair_overwrite_forbidden")

    video = _encode_episode_video(
        frame_paths,
        video_path=video_path,
        frames_per_second=frames_per_second,
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "episode_id": episode_id,
        "source_frame_manifest": str(manifest_path),
        "source_frame_manifest_digest": expected_digest,
        "source_frame_count": len(frame_paths),
        "source_lossless_frames_are_authoritative": True,
        "historical_mp4v_derivative_invalid_for_macos_review": True,
        "output_video": {
            **video,
            "relative_path": video_path.name,
        },
        "physical_truth_claim": False,
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames-per-second", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipts = [
        rederive_review_video(
            frame_manifest_path=path,
            output_dir=args.output_dir,
            frames_per_second=args.frames_per_second,
        )
        for path in args.frame_manifests
    ]
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA_VERSION", "main", "rederive_review_video"]
