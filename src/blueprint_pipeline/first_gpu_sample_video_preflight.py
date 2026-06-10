"""Preflight source videos before first-GPU sample staging."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json


FIRST_GPU_SAMPLE_VIDEO_PREFLIGHT_SCHEMA_VERSION = "first_gpu_sample_video_preflight.v1"
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v"}
DEFAULT_MAX_DURATION_SECONDS = 30.0
DEFAULT_MAX_SIZE_BYTES = 100_000_000


def _string(value: Any) -> str:
    return str(value or "").strip()


def _append_unique(target: List[Path], paths: Iterable[Path]) -> None:
    seen = {str(item) for item in target}
    for path in paths:
        key = str(path)
        if key not in seen:
            target.append(path)
            seen.add(key)


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _discover_videos(search_roots: Sequence[str | Path]) -> List[Path]:
    discovered: List[Path] = []
    for search_root in search_roots:
        root = Path(search_root).expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix.lower() in VIDEO_SUFFIXES:
                _append_unique(discovered, [root])
            continue
        paths = (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        _append_unique(discovered, sorted(paths))
    return discovered


def _ffprobe_media_metadata(path: Path) -> Dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {
            "status": "unavailable",
            "tool": "ffprobe",
            "blockers": ["ffprobe_not_found"],
        }
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "status": "failed",
            "tool": ffprobe,
            "exit_code": proc.returncode,
            "stderr_tail": proc.stderr[-400:],
            "blockers": ["ffprobe_failed"],
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "tool": ffprobe,
            "exit_code": proc.returncode,
            "blockers": ["ffprobe_output_not_json"],
        }
    format_info = payload.get("format") if isinstance(payload.get("format"), Mapping) else {}
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next(
        (
            item
            for item in streams
            if isinstance(item, Mapping) and item.get("codec_type") == "video"
        ),
        {},
    )
    duration = _float_or_none(format_info.get("duration")) or _float_or_none(
        video_stream.get("duration") if isinstance(video_stream, Mapping) else None
    )
    return {
        "status": "ready",
        "tool": ffprobe,
        "duration_seconds": duration,
        "width": _int_or_none(video_stream.get("width") if isinstance(video_stream, Mapping) else None),
        "height": _int_or_none(
            video_stream.get("height") if isinstance(video_stream, Mapping) else None
        ),
        "codec_name": _string(video_stream.get("codec_name") if isinstance(video_stream, Mapping) else None)
        or None,
        "format_name": _string(format_info.get("format_name")) or None,
        "blockers": [],
    }


def _audit_video(
    path: Path,
    *,
    max_duration_seconds: float,
    max_size_bytes: int,
    require_probe: bool,
) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    staging_blockers: List[str] = []
    worldlabs_blockers: List[str] = []
    warnings: List[str] = []
    suffix = resolved.suffix.lower()
    exists = resolved.is_file()
    size_bytes = resolved.stat().st_size if exists else 0
    if not exists:
        staging_blockers.append("source_video_missing")
    if suffix not in VIDEO_SUFFIXES:
        staging_blockers.append("unsupported_video_suffix")
    if exists and size_bytes <= 0:
        worldlabs_blockers.append("source_video_empty")
    if exists and size_bytes > max_size_bytes:
        worldlabs_blockers.append("source_video_exceeds_worldlabs_size_limit")

    metadata = (
        _ffprobe_media_metadata(resolved)
        if exists and suffix in VIDEO_SUFFIXES
        else {
            "status": "skipped",
            "blockers": [],
        }
    )
    metadata_status = _string(metadata.get("status"))
    duration = _float_or_none(metadata.get("duration_seconds"))
    if metadata_status != "ready":
        if require_probe:
            worldlabs_blockers.append(f"media_probe_{metadata_status or 'not_ready'}")
        else:
            warnings.append(f"media_probe_{metadata_status or 'not_ready'}")
    elif duration is None:
        worldlabs_blockers.append("source_video_duration_unknown")
    elif duration > max_duration_seconds:
        worldlabs_blockers.append("source_video_exceeds_worldlabs_duration_limit")

    return {
        "path": str(resolved),
        "exists": exists,
        "suffix": suffix,
        "size_bytes": size_bytes,
        "max_size_bytes": max_size_bytes,
        "max_duration_seconds": max_duration_seconds,
        "media_metadata": metadata,
        "ready_for_capture_staging": not staging_blockers,
        "ready_for_worldlabs_first_clip": not staging_blockers and not worldlabs_blockers,
        "staging_blockers": staging_blockers,
        "worldlabs_blockers": worldlabs_blockers,
        "warnings": warnings,
        "next_commands": {
            "stage_sample": (
                "blueprint-stage-first-gpu-sample-video "
                f"--source-video {_shell_quote(resolved)} "
                "--storage-root output/first-gpu-sample-storage "
                "--bucket local-blueprint --scene-id <scene-id> --capture-id <capture-id>"
            )
            if not staging_blockers
            else None,
        },
        "proof_boundary": (
            "Source-video preflight checks file suitability only. It does not prove privacy "
            "clearance, scene geometry, WebApp upstream truth, simulator execution, or robot readiness."
        ),
    }


def build_first_gpu_sample_video_preflight(
    *,
    source_videos: Sequence[str | Path] = (),
    search_roots: Sequence[str | Path] = (),
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    require_probe: bool = False,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    candidates: List[Path] = []
    _append_unique(candidates, [Path(item).expanduser() for item in source_videos])
    _append_unique(candidates, _discover_videos(search_roots))
    audits = [
        _audit_video(
            candidate,
            max_duration_seconds=max_duration_seconds,
            max_size_bytes=max_size_bytes,
            require_probe=require_probe,
        )
        for candidate in candidates
    ]
    ready_for_staging = [item for item in audits if item["ready_for_capture_staging"]]
    ready_for_worldlabs = [item for item in audits if item["ready_for_worldlabs_first_clip"]]
    blockers: List[str] = []
    if not audits:
        blockers.append("no_source_videos_found")
    elif not ready_for_staging:
        blockers.append("no_source_videos_ready_for_capture_staging")
    elif not ready_for_worldlabs:
        blockers.append("no_source_videos_ready_for_worldlabs_first_clip")
    result = {
        "schema_version": FIRST_GPU_SAMPLE_VIDEO_PREFLIGHT_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "ready" if not blockers else "blocked",
        "source_video_count": len(audits),
        "ready_for_capture_staging_count": len(ready_for_staging),
        "ready_for_worldlabs_first_clip_count": len(ready_for_worldlabs),
        "max_duration_seconds": max_duration_seconds,
        "max_size_bytes": max_size_bytes,
        "require_probe": require_probe,
        "source_videos": [str(Path(item).expanduser()) for item in source_videos],
        "search_roots": [str(Path(item).expanduser()) for item in search_roots],
        "candidates": audits,
        "blockers": blockers,
        "claim_boundary": {
            "artifact_purpose": "first_gpu_sample_video_source_preflight",
            "live_provider_calls_performed": False,
            "webapp_requests_submitted": False,
            "simulator_execution_performed": False,
            "gpu_provisioning_performed": False,
            "robot_readiness_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    if output_path:
        output = Path(output_path).expanduser()
        ensure_dir(output.parent)
        write_json(output, result)
        result["output_path"] = str(output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit collected source videos before first-GPU sample staging"
    )
    parser.add_argument("--source-video", action="append", default=[])
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--max-duration-seconds", type=float, default=DEFAULT_MAX_DURATION_SECONDS)
    parser.add_argument("--max-size-bytes", type=int, default=DEFAULT_MAX_SIZE_BYTES)
    parser.add_argument(
        "--require-probe",
        action="store_true",
        help="Treat missing or failed ffprobe metadata as a blocker.",
    )
    parser.add_argument(
        "--output",
        default="output/first_gpu_sample_video_preflight_manifest.json",
    )
    args = parser.parse_args(argv)
    result = build_first_gpu_sample_video_preflight(
        source_videos=args.source_video,
        search_roots=args.search_root,
        max_duration_seconds=args.max_duration_seconds,
        max_size_bytes=args.max_size_bytes,
        require_probe=args.require_probe,
        output_path=args.output,
    )
    print(f"[first-gpu-sample-video-preflight] status={result['status']}")
    print(f"[first-gpu-sample-video-preflight] candidates={result['source_video_count']}")
    print(
        "[first-gpu-sample-video-preflight] ready_worldlabs="
        + str(result["ready_for_worldlabs_first_clip_count"])
    )
    print(f"[first-gpu-sample-video-preflight] manifest={result['output_path']}")
    if result["blockers"]:
        print("[first-gpu-sample-video-preflight] blockers=" + ",".join(result["blockers"]))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
