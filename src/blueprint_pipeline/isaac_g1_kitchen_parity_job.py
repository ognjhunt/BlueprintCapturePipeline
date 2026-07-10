"""End-to-end MuJoCo-parity G1 kitchen eval on Isaac Sim + GPU (productionized).

Replicates the MuJoCo G1 walk-to-target eval (policy + per-step trace + outcome + WAM-ready
package + MP4) but executes it inside Isaac Sim on a real GPU, against the sim-ready Lightwheel
kitchen USD and the official Isaac G1 USD. Stage A drives the SAME deterministic controller as
MuJoCo (``isaac_g1_policy``); Stage B swaps in GR00T N1.7 SONIC via ``--policy groot_sonic``.

Chain:
  scenarios + kitchen asset dir + G1 USD ref
    -> bundle (runner + policy module + request.json + kitchen assets)
    -> stage to object store (signed GET/PUT)        [reused from the splat job]
    -> provider GPU VM/pod runs run_isaac_g1_kitchen_parity_eval.py via a hardened bootstrap
       (DigitalOcean by default for this high-reliability Isaac review lane; RunPod remains
       explicit compatibility; Vast paid launch requires an explicit unstable override)
    -> RTX MP4s + traces + parity outcome JSON uploaded
    -> collect -> assemble the WAM-ready harness package with an honest claim boundary.

Paid GPU launches are gated behind ``allow_paid=True``. Secrets are file-based and never
logged. Truth boundary: Isaac RTX kinematic walk-to-target preview parity (Stage A) — not
dynamic locomotion, not a learned policy, not readiness.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso
from .gpu_render_providers import RenderLaunchSpec, get_render_provider
from .isaac_g1_policy import groot_sonic_isaac_bridge_readiness
from .isaac_particlefield_render_job import (
    DEFAULT_WARM_CANDIDATES, stage_bundle, watch_and_collect,
)
from .launch_provenance import (
    evaluate_dirty_tree_paid_launch_gate,
    git_worktree_evidence,
)
from .paid_lane_guard import (
    bind_pending_teardown_instance,
    cancel_pending_teardown,
    close_pending_teardown,
    open_pending_teardown,
    provider_state_from_inspect,
)
from .provider_race import boot_marker_present, race_launch
from .provider_reliability_manifest import (
    TEARDOWN_STATUS_SOURCE_PROVIDER_API,
    build_teardown_proof,
)
from .security_controls import (
    exact_https_origin,
    fetch_bounded_https,
    origins_from_env,
)

SCHEMA_VERSION = "isaac_g1_kitchen_parity_job.v1"
JOB_MANIFEST_FILENAME = "isaac_g1_kitchen_parity_job_manifest.json"
LAUNCH_ATTEMPT_TRACE_FILENAME = "isaac_g1_kitchen_parity_launch_attempts.json"
WORKER_BUNDLE_DIR = "/workspace/bundle"
ISAAC_G1_KITCHEN_PARITY_LANE = "isaac_g1_kitchen_parity"
PROVIDER_CAPACITY_UNAVAILABLE_BLOCKERS = frozenset({
    "digitalocean_gpu_size_region_unavailable",
})
DEFAULT_G1_USD_RELATIVE = "Isaac/Robots/Unitree/G1/g1.usd"
DEFAULT_KITCHEN_MAIN_USD = "Collected_KitchenRoom/KitchenRoom.usd"
DEFAULT_ISAAC_REVIEW_PROVIDER = "digitalocean"
DEFAULT_VAST_MAX_HOURLY_RATE_USD = 5.0
ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV = "BLUEPRINT_ALLOW_UNSTABLE_VAST_ISAAC_RENDER"
ISAAC_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF"
ISAAC_WORKER_IMAGE_REF_FILE_ENV = "BLUEPRINT_ISAAC_EVAL_WORKER_IMAGE_REF_FILE"
ROBOT_EVAL_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
DEFAULT_ISAAC_WORKER_IMAGE_REF_FILE = "~/.blueprint-secrets/isaac_eval_worker_image_ref"
ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV = "BLUEPRINT_ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD"
DEFAULT_PARITY_IMAGE_REF = "docker.io/nijelhunt/blueprint-isaac-eval-worker:20260626-faststart-amd64"
ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV = "BLUEPRINT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC"
DEFAULT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC = "output/isaac_worker_image_manifest_diagnostic.json"
ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV = "BLUEPRINT_ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START"
COLD_RACE_CONTENDERS_ENV = "BLUEPRINT_COLD_RACE_CONTENDERS"
DEFAULT_COLD_RACE_CONTENDERS = 2
DEFAULT_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS = 600
PROVIDER_ARTIFACT_ALLOWED_ORIGINS_ENV = (
    "BLUEPRINT_PROVIDER_ARTIFACT_ALLOWED_ORIGINS"
)
MAX_KITCHEN_ASSET_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_WARM_READINESS_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
ISAAC_G1_MAX_SPEND_USD_ENV = "BLUEPRINT_ISAAC_G1_MAX_SPEND_USD"
ISAAC_G1_GROOT_POLICY_COMMAND_ENV = "BLUEPRINT_ISAAC_G1_GROOT_POLICY_COMMAND"
UNITREE_GROOT_POLICY_COMMAND_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_COMMAND"
UNITREE_GROOT_POLICY_SERVER_URL_ENV = "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL"
ISAAC_G1_GROOT_POLICY_RUNTIME_MODE_ENV = "BLUEPRINT_ISAAC_G1_GROOT_POLICY_RUNTIME_MODE"
ISAAC_G1_GROOT_POLICY_PREBAKED_IMAGE_CONFIRMED_ENV = (
    "BLUEPRINT_ISAAC_G1_GROOT_POLICY_PREBAKED_IMAGE_CONFIRMED"
)
ISAAC_G1_GROOT_POLICY_COMMAND_TIMEOUT_ENV = (
    "BLUEPRINT_ISAAC_G1_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS"
)
PARITY_BUNDLE_REQUIRED_FILES = (
    "run_isaac_g1_kitchen_parity_eval.py",
    "isaac_g1_policy.py",
    "stance_configuration_agent.py",
    "render_visual_qc.py",
    "warm_render_server.py",
    "warm_render_broker.py",
    "g1_render_noise_audit.py",
    "blueprint_pipeline/__init__.py",
    "blueprint_pipeline/common.py",
    "blueprint_pipeline/unitree_groot_n17_sonic_policy_runtime.py",
    "blueprint_pipeline/unitree_groot_n17_sonic_policy_server_command.py",
    "request.json",
    "scene_placement/__init__.py",
    "scene_placement/types.py",
    "scene_placement/target_resolver.py",
    "scene_placement/usd_index.py",
    "scene_placement/perception_index.py",
    "scene_placement/perception_fusion.py",
    "scene_placement/perception_views.py",
    "scene_placement/perception_adapter.py",
    "scene_placement/placement.py",
    "scene_placement/validation.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_worktree_evidence(*, max_dirty_entries: int = 200) -> dict:
    return git_worktree_evidence(max_dirty_entries=max_dirty_entries)


def _zip_dir(src_dir: Path, zip_path: Path) -> Path:
    """Zip a directory tree (files only), paths relative to src_dir."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(src_dir.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(src_dir).as_posix())
    return zip_path


def _assert_parity_bundle_namelist(names: set[str]) -> None:
    missing = [name for name in PARITY_BUNDLE_REQUIRED_FILES if name not in names]
    if missing:
        raise RuntimeError(f"parity_bundle_missing_required_files:{','.join(missing)}")


def _inspect_kitchen_asset_namelist(
    names: Sequence[str],
    *,
    source: str,
    byte_size: int | None = None,
) -> dict:
    """Validate a staged kitchen asset zip/tree before a GPU worker tries to open it."""
    files = sorted(
        {
            str(name).lstrip("/")
            for name in names
            if str(name).strip() and not str(name).endswith("/")
        }
    )
    candidates = [DEFAULT_KITCHEN_MAIN_USD, "KitchenRoom.usd"]
    selected = next((candidate for candidate in candidates if candidate in files), "")
    if not selected:
        kitchen_room_files = sorted(
            (name for name in files if name.endswith("/KitchenRoom.usd") or name == "KitchenRoom.usd"),
            key=lambda name: (len(Path(name).parts), name),
        )
        selected = kitchen_room_files[0] if kitchen_room_files else ""
    blockers: list[str] = []
    if not files:
        blockers.append("kitchen_asset_empty")
    if not selected:
        blockers.append("kitchen_main_usd_missing")
    layout = "unknown"
    if selected == DEFAULT_KITCHEN_MAIN_USD:
        layout = "collected_kitchen_room"
    elif selected == "KitchenRoom.usd":
        layout = "root_kitchen_room"
    elif selected:
        layout = "nested_kitchen_room"
    return {
        "schema_version": "kitchen_asset_layout_validation.v1",
        "status": "PASS" if not blockers else "FAIL",
        "source": source,
        "blockers": blockers,
        "file_count": len(files),
        "zip_bytes": byte_size,
        "selected_kitchen_main_usd_relative": selected or None,
        "expected_worker_kitchen_usd": (
            f"{WORKER_BUNDLE_DIR}/kitchen/{selected}" if selected else None
        ),
        "layout": layout,
        "sample_files": files[:40],
        "raw_url_values_recorded": False,
        "claim_boundary": (
            "Kitchen asset layout validation proves only that the staged asset bundle contains "
            "a usable KitchenRoom.usd path for the worker request. It does not prove Isaac can "
            "render the scene, task success, WAM quality, physical reach, safety, or deployment readiness."
        ),
    }


def _inspect_kitchen_asset_dir_layout(path: str | Path) -> dict:
    root = Path(path)
    if not root.is_dir():
        return {
            "schema_version": "kitchen_asset_layout_validation.v1",
            "status": "FAIL",
            "source": "local_asset_dir",
            "blockers": ["kitchen_asset_dir_missing"],
            "path": str(root),
            "raw_url_values_recorded": False,
        }
    names = [
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    ]
    detail = _inspect_kitchen_asset_namelist(names, source="local_asset_dir")
    detail["path"] = str(root)
    return detail


def _fetch_provider_artifact_bytes(
    url: str,
    *,
    timeout: int,
    max_bytes: int,
) -> bytes:
    allowed_origins = origins_from_env(PROVIDER_ARTIFACT_ALLOWED_ORIGINS_ENV)
    if not allowed_origins:
        allowed_origins = (exact_https_origin(url),)
    return fetch_bounded_https(
        url,
        timeout_seconds=timeout,
        max_bytes=max_bytes,
        allowed_origins=allowed_origins,
        max_redirects=0,
    ).body


def _inspect_kitchen_asset_url_layout(kitchen_url: str, *, timeout: int = 1800) -> dict:
    if not str(kitchen_url or "").strip():
        return {
            "schema_version": "kitchen_asset_layout_validation.v1",
            "status": "FAIL",
            "source": "reused_existing_url",
            "blockers": ["kitchen_url_missing"],
            "raw_url_values_recorded": False,
        }
    try:
        data = _fetch_provider_artifact_bytes(
            kitchen_url,
            timeout=timeout,
            max_bytes=MAX_KITCHEN_ASSET_ARCHIVE_BYTES,
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            detail = _inspect_kitchen_asset_namelist(
                zf.namelist(),
                source="reused_existing_url",
                byte_size=len(data),
            )
        return detail
    except urllib.error.HTTPError as exc:
        return {
            "schema_version": "kitchen_asset_layout_validation.v1",
            "status": "FAIL",
            "source": "reused_existing_url",
            "blockers": ["kitchen_url_fetch_failed"],
            "http_status": exc.code,
            "raw_url_values_recorded": False,
        }
    except zipfile.BadZipFile:
        return {
            "schema_version": "kitchen_asset_layout_validation.v1",
            "status": "FAIL",
            "source": "reused_existing_url",
            "blockers": ["kitchen_url_not_zip"],
            "raw_url_values_recorded": False,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "schema_version": "kitchen_asset_layout_validation.v1",
            "status": "FAIL",
            "source": "reused_existing_url",
            "blockers": ["kitchen_url_inspection_failed"],
            "error_type": type(exc).__name__,
            "raw_url_values_recorded": False,
        }


def _write_job_manifest(out_dir: str | Path, manifest: dict) -> Path:
    """Persist the top-level job result so CLI failures survive lost terminal output."""
    out_path = Path(out_dir) / JOB_MANIFEST_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out_path


def _write_launch_attempt_trace(job_dir: str | Path, trace: dict) -> Path:
    out_path = Path(job_dir) / LAUNCH_ATTEMPT_TRACE_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")
    return out_path


def _int_or_none(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _probe_video_file(path: str | Path) -> dict:
    """Best-effort local media metadata for collected provider videos."""
    video_path = Path(path)
    if not video_path.is_file():
        return {
            "status": "missing",
            "path": str(video_path),
            "blockers": ["video_file_missing"],
        }
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {
            "status": "unavailable",
            "path": str(video_path),
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
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "status": "failed",
            "path": str(video_path),
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
            "path": str(video_path),
            "tool": ffprobe,
            "blockers": ["ffprobe_output_not_json"],
        }
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    video_stream = next(
        (
            item
            for item in streams
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        {},
    )
    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}
    fps = None
    rate = str(video_stream.get("r_frame_rate") or "").strip()
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = _float_or_none(den)
        if den_f:
            fps = (_float_or_none(num) or 0.0) / den_f
    else:
        fps = _float_or_none(rate)
    return {
        "status": "ready",
        "path": str(video_path),
        "tool": ffprobe,
        "width": _int_or_none(video_stream.get("width")),
        "height": _int_or_none(video_stream.get("height")),
        "frame_count": _int_or_none(video_stream.get("nb_frames")),
        "fps": fps,
        "duration_seconds": (
            _float_or_none(video_stream.get("duration"))
            or _float_or_none(format_info.get("duration"))
        ),
        "codec_name": video_stream.get("codec_name") or None,
    }


REVIEW_MP4_FRAME_PATTERNS = {
    "overview": "overview_[0-9][0-9][0-9][0-9].png",
    "robot_pov": "robot_pov_[0-9][0-9][0-9][0-9].png",
    "placement_topdown": "placement_topdown_[0-9][0-9][0-9][0-9].png",
}

REVIEW_MP4_FRAME_SEQUENCES = {
    "overview": "overview_%04d.png",
    "robot_pov": "robot_pov_%04d.png",
    "placement_topdown": "placement_topdown_%04d.png",
}


def _ffmpeg_mp4_command(
    *,
    ffmpeg: str,
    frames_sequence: str,
    fps: int,
    out_path: str,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-framerate",
        str(int(fps)),
        "-start_number",
        "0",
        "-i",
        frames_sequence,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]


def _repair_collected_review_mp4s(
    *,
    render_out_dir: Path,
    result: dict,
    fps: int,
    optional_videos: Sequence[str] = (),
    expected_frame_count: int | None = None,
) -> dict:
    """Assemble missing collected MP4s locally when the provider image lacked ffmpeg.

    When ``expected_frame_count`` is known, a repair over fewer frames is labeled
    ``repaired_truncated`` with a blocker instead of ``repaired`` — a locally
    assembled MP4 must never make a partially-uploaded provider render read as a
    complete one. The truncated video is still written for human review.
    """
    scenarios = result.get("scenarios", []) if isinstance(result, dict) else []
    ffmpeg = shutil.which("ffmpeg")
    repairs: list[dict] = []
    optional_video_set = {str(name) for name in optional_videos}
    expected = int(expected_frame_count) if expected_frame_count else None
    for sc in scenarios:
        sid = sc.get("scenario_id")
        sdir = render_out_dir / str(sid)
        frames_dir = sdir / "frames"
        for name, pattern in REVIEW_MP4_FRAME_PATTERNS.items():
            out_path = sdir / f"{name}.mp4"
            frame_paths = sorted(frames_dir.glob(pattern))
            rec: dict = {
                "scenario_id": sid,
                "video": name,
                "path": str(out_path),
                "frame_pattern": pattern,
                "frame_count": len(frame_paths),
                "expected_frame_count": expected,
            }
            truncated = expected is not None and 0 < len(frame_paths) < expected
            if out_path.is_file():
                rec["status"] = "already_present"
            elif not frame_paths:
                if name in optional_video_set:
                    rec["status"] = "skipped_optional"
                    rec["optional"] = True
                else:
                    rec["status"] = "missing_frames"
                    rec["blockers"] = ["video_frames_missing"]
            elif not ffmpeg:
                rec["status"] = "unavailable"
                rec["blockers"] = ["local_ffmpeg_not_found"]
            else:
                cmd = _ffmpeg_mp4_command(
                    ffmpeg=ffmpeg,
                    frames_sequence=str(frames_dir / REVIEW_MP4_FRAME_SEQUENCES[name]),
                    fps=int(fps),
                    out_path=str(out_path),
                )
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                rec["tool"] = ffmpeg
                rec["exit_code"] = proc.returncode
                if proc.returncode == 0 and out_path.is_file():
                    if truncated:
                        rec["status"] = "repaired_truncated"
                        rec["blockers"] = [
                            "mp4_repair_truncated_frames:"
                            f"{name}:{len(frame_paths)}<{expected}"
                        ]
                    else:
                        rec["status"] = "repaired"
                else:
                    rec["status"] = "failed"
                    rec["blockers"] = ["local_ffmpeg_mp4_repair_failed"]
                    rec["stderr_tail"] = proc.stderr[-500:]
            repairs.append(rec)
    blockers = sorted({
        str(blocker)
        for rec in repairs
        for blocker in (rec.get("blockers") or [])
    })
    return {
        "schema_version": "isaac_collected_review_mp4_repair.v1",
        "status": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "ffmpeg": ffmpeg,
        "fps": int(fps),
        "repairs": repairs,
        "claim_boundary": (
            "Local MP4 repair assembles already-collected provider PNG frames. It does not alter "
            "rendered frames, task outcomes, or evaluator success labels."
        ),
    }


def parity_image() -> str:
    """Isaac worker image for the parity eval (defaults to the same Isaac eval worker)."""
    image_config = _configured_isaac_worker_image_ref()
    return str(image_config.get("image_ref") or DEFAULT_PARITY_IMAGE_REF)


def _string(value) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _configured_isaac_worker_image_ref() -> dict:
    explicit = _string(os.getenv(ISAAC_WORKER_IMAGE_REF_ENV))
    if explicit:
        return {
            "image_ref": explicit,
            "source": ISAAC_WORKER_IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": None,
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    file_value = _string(os.getenv(ISAAC_WORKER_IMAGE_REF_FILE_ENV))
    image_ref_file = Path(file_value or DEFAULT_ISAAC_WORKER_IMAGE_REF_FILE).expanduser()
    if image_ref_file.is_file():
        image_ref = image_ref_file.read_text(encoding="utf-8").strip()
        return {
            "image_ref": image_ref,
            "source": ISAAC_WORKER_IMAGE_REF_FILE_ENV
            if file_value
            else "default_blueprint_secret_file_path",
            "configured": bool(image_ref),
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": True,
            "raw_secret_values_recorded": False,
        }
    generic = _string(os.getenv(ROBOT_EVAL_WORKER_IMAGE_REF_ENV))
    if generic:
        return {
            "image_ref": generic,
            "source": ROBOT_EVAL_WORKER_IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    return {
        "image_ref": "",
        "source": None,
        "configured": False,
        "image_ref_file": str(image_ref_file),
        "image_ref_file_present": False,
        "raw_secret_values_recorded": False,
    }


def _isaac_worker_image_size_diagnostic(image_ref: str) -> dict:
    explicit = _string(os.getenv(ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV))
    selected = explicit or DEFAULT_ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC
    path = Path(selected).expanduser()
    base = {
        "env_var": ISAAC_WORKER_IMAGE_MANIFEST_DIAGNOSTIC_ENV,
        "path": str(path),
        "path_source": "env" if explicit else "default_output_path",
        "path_present": path.is_file(),
        "raw_secret_values_recorded": False,
    }
    if not path.is_file():
        return {
            **base,
            "status": "missing",
            "metadata_available_for_selected_image": False,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "status": "unreadable",
            "metadata_available_for_selected_image": False,
            "error_type": type(exc).__name__,
        }
    manifest = dict(payload) if isinstance(payload, dict) else {}
    manifest_image_ref = _string(manifest.get("image_ref"))
    if manifest_image_ref and image_ref and manifest_image_ref != image_ref:
        return {
            **base,
            "status": "ignored_image_ref_mismatch",
            "metadata_available_for_selected_image": False,
            "manifest_image_ref": manifest_image_ref,
            "selected_image_ref": image_ref,
        }
    return {
        **base,
        "status": _string(manifest.get("status")) or "completed",
        "metadata_available_for_selected_image": True,
        "image_ref": manifest_image_ref or image_ref,
        "layer_count": manifest.get("layer_count"),
        "total_compressed_size_bytes": manifest.get("total_compressed_size_bytes"),
        "largest_layer_size_bytes": manifest.get("largest_layer_size_bytes"),
        "large_image_pull_risk": bool(manifest.get("large_image_pull_risk")),
        "proof_boundary": (
            "Worker image manifest metadata only. This does not prove container "
            "startup, Isaac Sim execution, rendered RGB quality, WAM quality, or "
            "robot readiness."
        ),
    }


def _gemini_api_key_from_env() -> str:
    """Gemini key for worker-side visual QC, read from local env and never serialized to artifacts."""
    return (os.getenv("GOOGLE_GENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


def _vast_max_hourly_rate_from_env(default: float = DEFAULT_VAST_MAX_HOURLY_RATE_USD) -> float:
    value = (
        os.getenv("BLUEPRINT_ISAAC_G1_PARITY_VAST_MAX_HOURLY_RATE")
        or os.getenv("BLUEPRINT_VAST_RENDER_MAX_HOURLY_RATE")
        or ""
    ).strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return parsed if parsed > 0 else float(default)


# diagnostics-streaming pod bootstrap for the parity runner
BOOTSTRAP = r'''
import os, sys, io, time, json, zipfile, threading, subprocess, urllib.request, pathlib, shutil, signal
OUT="/workspace/out"; BUNDLE="/workspace/bundle"
for d in (OUT, BUNDLE): pathlib.Path(d).mkdir(parents=True, exist_ok=True)
for p in pathlib.Path(OUT).iterdir():
    # tee already holds runner_console.log open; unlinking it detaches the inode and the
    # console then never reaches the output zip (every crash so far collected an empty tail).
    if p.name == "runner_console.log": continue
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except Exception: pass
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
GETB=os.environ.get("BLUEPRINT_EVAL_MANIFEST_URI","")
SESSION=os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID","")
def putout():
    try:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            for p in pathlib.Path(OUT).rglob("*"):
                if p.is_file():
                    try: z.write(p, p.relative_to(OUT).as_posix())
                    except Exception: pass
        req=urllib.request.Request(PUT, data=buf.getvalue(), method="PUT", headers={"Content-Type":"application/zip"})
        urllib.request.urlopen(req, timeout=180).read()
    except Exception: pass
def mark(ph, **k):
    try: json.dump({"phase":ph, "launch_session_id":SESSION, **k}, open(OUT+"/bootstrap.json","w"))
    except Exception: pass
    putout()
def hb():
    while True:
        time.sleep(25); putout()
def fetch_bytes(url, *, phase, timeout=1800, progress_step=67108864):
    chunks=[]; total=0; next_mark=progress_step
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        content_length=resp.headers.get("Content-Length") or resp.headers.get("content-length")
        mark(phase+"_connected", content_length_bytes=content_length)
        while True:
            chunk=resp.read(4194304)
            if not chunk: break
            chunks.append(chunk); total += len(chunk)
            if total >= next_mark:
                mark(phase+"_progress", bytes_read=total, content_length_bytes=content_length)
                next_mark = total + progress_step
    mark(phase+"_fetched", bytes_read=total)
    return b"".join(chunks)
def runner_timeout_seconds():
    raw=os.environ.get("PARITY_RUNNER_TIMEOUT_SECONDS","")
    if not raw: return 0
    try: return max(1, int(float(raw)))
    except Exception: return 0
threading.Thread(target=hb, daemon=True).start()
try:
    mark("bootstrap_fetching")
    data=fetch_bytes(GETB, phase="bootstrap_fetch", timeout=600, progress_step=16777216)
    zipfile.ZipFile(io.BytesIO(data)).extractall(BUNDLE)
    mark("bootstrap_extracted", files=sorted(os.listdir(BUNDLE)))
    # kitchen assets are staged separately (large, reused across iterations) and fetched into BUNDLE/kitchen
    KURL=os.environ.get("KITCHEN_BUNDLE_URL","")
    if KURL:
        mark("kitchen_fetching")
        kdir=BUNDLE+"/kitchen"; pathlib.Path(kdir).mkdir(parents=True, exist_ok=True)
        kdata=fetch_bytes(KURL, phase="kitchen_fetch", timeout=1800)
        mark("kitchen_extracting", bytes_read=len(kdata))
        zipfile.ZipFile(io.BytesIO(kdata)).extractall(kdir)
        mark("kitchen_extracted", kitchen_files=len(list(pathlib.Path(kdir).rglob("*"))))
except Exception as exc:
    mark("bootstrap_failed", error=repr(exc))
    raise
try: subprocess.call(["/isaac-sim/python.sh","-m","pip","install","-q","pillow","google-genai"])  # frame save + Gemini QC deps (best-effort)
except Exception: pass
try: subprocess.call(["bash","-c","command -v ffmpeg >/dev/null 2>&1 || (apt-get update -y >/dev/null 2>&1 && apt-get install -y ffmpeg >/dev/null 2>&1)"])  # mp4 assembly (best-effort)
except Exception: pass
cmd=["/isaac-sim/python.sh", BUNDLE+"/run_isaac_g1_kitchen_parity_eval.py",
     "--request", BUNDLE+"/request.json", "--out-dir", OUT,
     "--policy", os.environ.get("PARITY_POLICY","blueprint_default_walk_to_target_smoke_policy"),
     "--steps", os.environ.get("PARITY_STEPS","64"),
     "--width", os.environ.get("RENDER_WIDTH","1280"), "--height", os.environ.get("RENDER_HEIGHT","960"),
     "--fps", os.environ.get("RENDER_FPS","20"),
     "--warmup-frames", os.environ.get("RENDER_WARMUP","6"),
     "--per-scenario-seconds", os.environ.get("PARITY_PER_SCENARIO_SECONDS","420"),
     "--focus-radius", os.environ.get("PARITY_FOCUS_RADIUS","0"),
     "--settle-seconds", os.environ.get("PARITY_SETTLE_SECONDS","0")]
if os.environ.get("PARITY_GROOT_POLICY_COMMAND",""): cmd += ["--groot-policy-command", os.environ["PARITY_GROOT_POLICY_COMMAND"]]
if os.environ.get("PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS",""): cmd += ["--groot-policy-command-timeout-seconds", os.environ["PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS"]]
if os.environ.get("PARITY_GROOT_POLICY_INITIAL_FRAME",""): cmd += ["--groot-policy-initial-frame", os.environ["PARITY_GROOT_POLICY_INITIAL_FRAME"]]
if os.environ.get("PARITY_DYNAMIC_EPISODE_TERMINATION","")=="1": cmd.append("--dynamic-episode-termination")
elif os.environ.get("PARITY_DYNAMIC_EPISODE_TERMINATION","")=="0": cmd.append("--no-dynamic-episode-termination")
if os.environ.get("PARITY_EPISODE_MAX_STEPS",""): cmd += ["--episode-max-steps", os.environ["PARITY_EPISODE_MAX_STEPS"]]
if os.environ.get("PARITY_DYNAMIC_EPISODE_CHECK_EVERY",""): cmd += ["--dynamic-episode-check-every", os.environ["PARITY_DYNAMIC_EPISODE_CHECK_EVERY"]]
if os.environ.get("PARITY_CAPTURE_EVERY",""): cmd += ["--capture-every", os.environ["PARITY_CAPTURE_EVERY"]]
if os.environ.get("PARITY_KEEP_OBJECTS",""): cmd += ["--keep-objects", os.environ["PARITY_KEEP_OBJECTS"]]
if os.environ.get("PARITY_NO_PROBE","")=="1": cmd.append("--no-collision-probe")
if os.environ.get("PARITY_CHEAP_COLLISION","")=="1": cmd.append("--cheap-collision")
if os.environ.get("PARITY_ARTICULATED","")=="1": cmd.append("--articulated")
if os.environ.get("PARITY_PHYSICS_ARTICULATION_DRIVE","")=="1": cmd.append("--physics-articulation-drive")
if os.environ.get("PARITY_DYNAMIC_STANDING_CONTACT_STEPS",""): cmd += ["--dynamic-standing-contact-steps", os.environ["PARITY_DYNAMIC_STANDING_CONTACT_STEPS"]]
if os.environ.get("PARITY_MANIPULATION_CAM","")=="1": cmd.append("--manipulation-cam")
if os.environ.get("PARITY_MANIPULATION_LOOK_AT",""): cmd += ["--manipulation-look-at", os.environ["PARITY_MANIPULATION_LOOK_AT"]]
if os.environ.get("PARITY_RENDER_SUBFRAMES",""): cmd += ["--render-subframes", os.environ["PARITY_RENDER_SUBFRAMES"]]
if os.environ.get("PARITY_NO_SOFTWARE_DENOISE","")=="1": cmd.append("--no-software-denoise")
if os.environ.get("PARITY_MANIPULATION_REACH","")=="1": cmd.append("--manipulation-reach")
if os.environ.get("PARITY_MANIPULATION_REACH_ARM",""): cmd += ["--manipulation-reach-arm", os.environ["PARITY_MANIPULATION_REACH_ARM"]]
if os.environ.get("PARITY_FILL_LIGHT_INTENSITY",""): cmd += ["--fill-light-intensity", os.environ["PARITY_FILL_LIGHT_INTENSITY"]]
if os.environ.get("PARITY_NEUTRAL_ENVIRONMENT","")=="1": cmd.append("--neutral-environment")
if os.environ.get("PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE","")=="1": cmd.append("--robot-review-material-override")
if os.environ.get("PARITY_ROBOT_REVIEW_MATERIAL_MODE",""): cmd += ["--robot-review-material-mode", os.environ["PARITY_ROBOT_REVIEW_MATERIAL_MODE"]]
if os.environ.get("PARITY_COLLISION_APPROXIMATION",""): cmd += ["--collision-approximation", os.environ["PARITY_COLLISION_APPROXIMATION"]]
if os.environ.get("PARITY_VERIFY_CAM","")=="1": cmd.append("--verify-cam")
if os.environ.get("PARITY_MANIPULATION_STAND","")=="1": cmd.append("--manipulation-stand")
if os.environ.get("PARITY_NO_PLACEMENT_TOPDOWN_CAPTURE","")=="1": cmd.append("--no-placement-topdown-capture")
if os.environ.get("PARITY_KINEMATIC_ARM_POSE","")=="1": cmd.append("--kinematic-arm-pose")
if os.environ.get("PARITY_RENDER_NOISE_AUDIT","")=="1":
    cmd.append("--render-noise-audit")  # variant-matrix render-quality audit instead of the scenario eval
    if os.environ.get("PARITY_AUDIT_HIGH_SPP",""): cmd += ["--audit-high-spp", os.environ["PARITY_AUDIT_HIGH_SPP"]]
    if os.environ.get("PARITY_AUDIT_WARMUP_FRAMES",""): cmd += ["--audit-warmup-frames", os.environ["PARITY_AUDIT_WARMUP_FRAMES"]]
    if os.environ.get("PARITY_AUDIT_BOOST_LIGHT_INTENSITY",""): cmd += ["--audit-boost-light-intensity", os.environ["PARITY_AUDIT_BOOST_LIGHT_INTENSITY"]]
if os.environ.get("PARITY_SERVE","")=="1":
    cmd.append("--serve")  # warm mode: boot Isaac + load scene ONCE, then serve jobs from the inbox env
    if os.environ.get("PARITY_SERVE_IDLE_TIMEOUT",""): cmd += ["--serve-idle-timeout", os.environ["PARITY_SERVE_IDLE_TIMEOUT"]]
    if os.environ.get("PARITY_SERVE_MAX_JOBS",""): cmd += ["--serve-max-jobs", os.environ["PARITY_SERVE_MAX_JOBS"]]
mark("runner_starting", cmd=cmd)
timeout=runner_timeout_seconds()
started=time.monotonic()
proc=subprocess.Popen(cmd, start_new_session=True)
try:
    rc=proc.wait(timeout=timeout if timeout > 0 else None)
    mark("runner_done", rc=rc, elapsed_seconds=round(time.monotonic()-started,1),
         timeout_seconds=timeout)
except subprocess.TimeoutExpired:
    try: os.killpg(proc.pid, signal.SIGTERM)
    except Exception: pass
    try:
        rc=proc.wait(timeout=30)
    except Exception:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except Exception: pass
        try: rc=proc.wait(timeout=10)
        except Exception: rc=None
    mark("runner_timeout", rc=rc, timeout_seconds=timeout,
         elapsed_seconds=round(time.monotonic()-started,1), cmd=cmd)
# Keep the container process alive after runner completion so RunPod does not restart it and
# clobber the final output object before the parent collector observes runner_done.
while True:
    time.sleep(30); putout()
'''

IMAGE_STARTUP_CANARY_BOOTSTRAP = r'''
import os, io, json, time, zipfile, pathlib, urllib.request, shutil, sys
from datetime import datetime, timezone

OUT="/workspace/out"
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
for p in pathlib.Path(OUT).iterdir():
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except Exception:
        pass
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
SESSION=os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID","")

def putout():
    try:
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            for p in pathlib.Path(OUT).rglob("*"):
                if p.is_file():
                    try:
                        z.write(p, p.relative_to(OUT).as_posix())
                    except Exception:
                        pass
        req=urllib.request.Request(PUT, data=buf.getvalue(), method="PUT", headers={"Content-Type":"application/zip"})
        urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        pass

def mark(phase, **extra):
    payload={"phase":phase, "launch_session_id":SESSION, **extra}
    pathlib.Path(OUT, "bootstrap.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    putout()

mark("runner_starting", image_startup_canary=True)
canary={
    "schema_version": "isaac_g1_parity_image_startup_canary.v1",
    "status": "completed",
    "image_startup_canary": True,
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "launch_session_id": SESSION,
    "python_executable": sys.executable,
    "python3_path": shutil.which("python3"),
    "isaac_python_path": "/isaac-sim/python.sh" if pathlib.Path("/isaac-sim/python.sh").exists() else None,
    "blueprint_worker_image_family": os.environ.get("BLUEPRINT_WORKER_IMAGE_FAMILY"),
    "simulator_framework": os.environ.get("BLUEPRINT_SIMULATOR_FRAMEWORK"),
    "claim_boundary": (
        "This canary proves only that the selected worker image reached user command "
        "execution and uploaded a provider output artifact for this launch session. It "
        "does not prove Isaac Sim startup, scene loading, RTX rendering, policy execution, "
        "WAM quality, or robot readiness."
    ),
}
pathlib.Path(OUT, "isaac_g1_kitchen_parity_result.json").write_text(json.dumps(canary, indent=2), encoding="utf-8")
pathlib.Path(OUT, "isaac_g1_parity_image_startup_canary.json").write_text(json.dumps(canary, indent=2), encoding="utf-8")
mark("runner_done", rc=0, image_startup_canary=True)
while True:
    time.sleep(30)
    putout()
'''

_EARLY_MARKER = r'''
import os, io, json, zipfile, pathlib, urllib.request
OUT="/workspace/out"; pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
SESSION=os.environ.get("BLUEPRINT_LAUNCH_SESSION_ID","")
json.dump({"phase":"container_bash_started","launch_session_id":SESSION}, open(OUT+"/bootstrap.json","w"))
PUT=os.environ.get("BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL","")
buf=io.BytesIO()
with zipfile.ZipFile(buf,"w") as z: z.write(OUT+"/bootstrap.json","bootstrap.json")
try: urllib.request.urlopen(urllib.request.Request(PUT,data=buf.getvalue(),method="PUT",headers={"Content-Type":"application/zip"}),timeout=60).read()
except Exception: pass
'''


def docker_start_cmd(*, image_startup_canary: bool = False) -> list[str]:
    worker_script = IMAGE_STARTUP_CANARY_BOOTSTRAP if image_startup_canary else BOOTSTRAP
    worker_script_name = "parity_image_startup_canary.py" if image_startup_canary else "boot.py"
    worker_python_cmd = (
        f'(python3 /workspace/{worker_script_name} || '
        f'python /workspace/{worker_script_name} || '
        f'/isaac-sim/python.sh /workspace/{worker_script_name})'
        if image_startup_canary
        else f"/isaac-sim/python.sh /workspace/{worker_script_name}"
    )
    script = (
        "set +e\n"
        "mkdir -p /workspace/out\n"
        "cat > /workspace/early.py <<'EARLYEOF'\n" + _EARLY_MARKER + "\nEARLYEOF\n"
        "(python3 /workspace/early.py 2>/dev/null || /isaac-sim/python.sh /workspace/early.py 2>/dev/null) || true\n"
        f"cat > /workspace/{worker_script_name} <<'PYEOF'\n" + worker_script + "\nPYEOF\n"
        f"{worker_python_cmd} 2>&1 | tee /workspace/out/runner_console.log\n"
    )
    return ["-lc", script]


# ----------------------------- request + bundle -----------------------------

def build_request(*, scenarios: Sequence[dict], kitchen_main_usd_relative: str = DEFAULT_KITCHEN_MAIN_USD,
                  g1_usd: str = DEFAULT_G1_USD_RELATIVE, policy_id: str, steps: int,
                  render_noise_audit_plan: dict | None = None) -> dict:
    """The runner's request.json. kitchen_usd is the worker-absolute path inside the extracted
    bundle; g1_usd is a relative Isaac asset path resolved against the assets root on the worker."""
    request = {
        "schema_version": "isaac_g1_kitchen_parity_request.v1",
        "kitchen_usd": f"{WORKER_BUNDLE_DIR}/kitchen/{kitchen_main_usd_relative}",
        "g1_usd": g1_usd,
        "policy_id": policy_id,
        "steps": steps,
        "scenarios": list(scenarios),
    }
    if render_noise_audit_plan is not None:
        request["render_noise_audit"] = dict(render_noise_audit_plan)
    return request


def build_parity_bundle(*, scenarios: Sequence[dict], out_dir: Path,
                        kitchen_asset_dir: str | Path | None = None,
                        kitchen_main_usd_relative: str = DEFAULT_KITCHEN_MAIN_USD,
                        g1_usd: str = DEFAULT_G1_USD_RELATIVE,
                        policy_id: str = "blueprint_default_walk_to_target_smoke_policy",
                        steps: int = 64,
                        render_noise_audit_plan: dict | None = None) -> Path:
    """Assemble the GPU bundle: the runner + the policy module + request.json + (optional) the
    kitchen asset tree under kitchen/. The runner imports the shipped policy module on the worker."""
    bundle = out_dir / "bundle"
    (bundle / "kitchen").mkdir(parents=True, exist_ok=True)
    runner = _repo_root() / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"
    (bundle / "run_isaac_g1_kitchen_parity_eval.py").write_bytes(runner.read_bytes())
    # Flat single-file modules the runner imports bundle-first on the worker (the
    # policy, stance search agent, placement QC, warm transport, noise audit).
    for flat_module_name in (
        "isaac_g1_policy.py",
        "stance_configuration_agent.py",
        "render_visual_qc.py",
        "warm_render_server.py",
        "warm_render_broker.py",
        "g1_render_noise_audit.py",
    ):
        flat_module = _repo_root() / "src" / "blueprint_pipeline" / flat_module_name
        (bundle / flat_module_name).write_bytes(flat_module.read_bytes())
    package_dst = bundle / "blueprint_pipeline"
    package_dst.mkdir(parents=True, exist_ok=True)
    (package_dst / "__init__.py").write_text("", encoding="utf-8")
    for module_name in (
        "common.py",
        "unitree_groot_n17_sonic_policy_runtime.py",
        "unitree_groot_n17_sonic_policy_server_command.py",
    ):
        src_module = _repo_root() / "src" / "blueprint_pipeline" / module_name
        (package_dst / module_name).write_bytes(src_module.read_bytes())
    # Ship the scene_placement package alongside the runner so its dynamic task->object resolution
    # works on the worker (the runner imports `scene_placement` from the bundle dir; it falls back to
    # the repo's `blueprint_pipeline.scene_placement` in tests). Without this the worker has no
    # `blueprint_pipeline` on its path and the runner silently degrades to the scenario's literal
    # target. The package is pure-Python + intra-package imports only, so a flat copy is importable.
    import shutil
    sp_src = _repo_root() / "src" / "blueprint_pipeline" / "scene_placement"
    sp_dst = bundle / "scene_placement"
    if sp_dst.exists():
        shutil.rmtree(sp_dst)
    shutil.copytree(sp_src, sp_dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    request = build_request(scenarios=scenarios, kitchen_main_usd_relative=kitchen_main_usd_relative,
                            g1_usd=g1_usd, policy_id=policy_id, steps=steps,
                            render_noise_audit_plan=render_noise_audit_plan)
    (bundle / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    if kitchen_asset_dir is not None:
        src = Path(kitchen_asset_dir)
        for item in src.rglob("*"):
            if item.is_file():
                dst = bundle / "kitchen" / item.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(item.read_bytes())
    bundle_files = sorted(
        item.relative_to(bundle).as_posix()
        for item in bundle.rglob("*")
        if item.is_file()
    )
    _assert_parity_bundle_namelist(set(bundle_files))
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "isaac_g1_kitchen_parity_bundle.v1",
                "required_files": list(PARITY_BUNDLE_REQUIRED_FILES),
                "files": bundle_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    zip_path = out_dir / "isaac_g1_kitchen_parity_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(bundle.rglob("*")):
            if item.is_file():
                zf.write(item, item.relative_to(bundle).as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        _assert_parity_bundle_namelist(set(zf.namelist()))
    return zip_path


def build_launch_spec(job_dir: Path, *, image: str, policy_id: str, steps: int, width: int = 1280,
                      height: int = 960, fps: int = 20, container_disk_gb: int = 140,
                      volume_gb: int = 80, kitchen_url: str | None = None, warmup: int = 6,
                      per_scenario_seconds: int = 420, no_collision_probe: bool = False,
                      focus_radius: float = 0.0, keep_objects: str = "", settle_seconds: int = 0,
                      cheap_collision: bool = False, articulated: bool = False,
                      physics_articulation_drive: bool = False,
                      dynamic_standing_contact_steps: int = 0,
                      manipulation_cam: bool = False, manipulation_look_at: str = "",
                      render_subframes: int = 0, manipulation_reach: bool = False,
                      manipulation_reach_arm: str = "auto",
                      dynamic_episode_termination: bool = True,
                      episode_max_steps: int = 0,
                      dynamic_episode_check_every: int = 1,
                      capture_every: int = 1,
                      fill_light_intensity: float = 0.0,
                      neutral_environment: bool = False,
                      robot_review_material_override: bool = False,
                      robot_review_material_mode: str = "",
                      kinematic_arm_pose: bool = False,
                      collision_approximation: str = "",
                      verify_cam: bool = False,
                      manipulation_stand: bool = False,
                      placement_topdown_capture: bool = True,
                      render_noise_audit: bool = False,
                      audit_high_spp: int = 0,
                      audit_warmup_frames: int = 0,
                      audit_boost_light_intensity: float = 0.0,
                      gemini_api_key: str | None = None,
                      groot_policy_command: str = "",
                      groot_policy_command_timeout_seconds: float = 120.0,
                      serve: bool = False, inbox_get_url: str = "",
                      warm_broker_base_url: str = "",
                      warm_broker_token: str = "",
                      serve_idle_timeout_s: float = 1800.0,
                      serve_max_jobs: int | None = None,
                      vast_max_hourly_rate_usd: float | None = None,
                      image_startup_canary: bool = False,
                      runner_timeout_seconds: int = 0) -> RenderLaunchSpec:
    bundle_url = (job_dir / "provider_bundle_url.txt").read_text().strip()
    put_url = (job_dir / "provider_output_put_url.txt").read_text().strip()
    env = {
        "ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "BLUEPRINT_EVAL_MANIFEST_URI": bundle_url,
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": put_url,
        "PARITY_POLICY": policy_id, "PARITY_STEPS": str(steps),
        "RENDER_WIDTH": str(width), "RENDER_HEIGHT": str(height), "RENDER_FPS": str(fps),
        "RENDER_WARMUP": str(warmup), "PARITY_PER_SCENARIO_SECONDS": str(per_scenario_seconds),
    }
    if runner_timeout_seconds and runner_timeout_seconds > 0:
        env["PARITY_RUNNER_TIMEOUT_SECONDS"] = str(int(runner_timeout_seconds))
    if no_collision_probe:
        env["PARITY_NO_PROBE"] = "1"
    if focus_radius and focus_radius > 0:
        env["PARITY_FOCUS_RADIUS"] = str(focus_radius)
    if keep_objects:
        env["PARITY_KEEP_OBJECTS"] = keep_objects
    if settle_seconds and settle_seconds > 0:
        env["PARITY_SETTLE_SECONDS"] = str(settle_seconds)
    if cheap_collision:
        env["PARITY_CHEAP_COLLISION"] = "1"
    if articulated or physics_articulation_drive or dynamic_standing_contact_steps > 0:
        env["PARITY_ARTICULATED"] = "1"
    if physics_articulation_drive or dynamic_standing_contact_steps > 0:
        env["PARITY_PHYSICS_ARTICULATION_DRIVE"] = "1"
    if dynamic_standing_contact_steps > 0:
        env["PARITY_DYNAMIC_STANDING_CONTACT_STEPS"] = str(int(dynamic_standing_contact_steps))
    if manipulation_cam:
        env["PARITY_MANIPULATION_CAM"] = "1"
    if manipulation_look_at:
        env["PARITY_MANIPULATION_LOOK_AT"] = str(manipulation_look_at)
    if render_subframes and render_subframes > 0:
        env["PARITY_RENDER_SUBFRAMES"] = str(render_subframes)
    for key in (
        "PARITY_RENDER_QUALITY_MODE",
        "PARITY_PATH_TRACING_SAMPLES_PER_PIXEL",
        "PARITY_PATH_TRACED_RT_SUBFRAMES",
    ):
        value = os.getenv(key, "").strip()
        if value:
            env[key] = value
    if manipulation_reach:
        env["PARITY_MANIPULATION_REACH"] = "1"
    if manipulation_reach and manipulation_reach_arm:
        env["PARITY_MANIPULATION_REACH_ARM"] = str(manipulation_reach_arm)
    env["PARITY_DYNAMIC_EPISODE_TERMINATION"] = "1" if dynamic_episode_termination else "0"
    if episode_max_steps and episode_max_steps > 0:
        env["PARITY_EPISODE_MAX_STEPS"] = str(int(episode_max_steps))
    if dynamic_episode_check_every and dynamic_episode_check_every > 1:
        env["PARITY_DYNAMIC_EPISODE_CHECK_EVERY"] = str(int(dynamic_episode_check_every))
    if capture_every and capture_every > 1:
        env["PARITY_CAPTURE_EVERY"] = str(int(capture_every))
    if fill_light_intensity and fill_light_intensity > 0:
        env["PARITY_FILL_LIGHT_INTENSITY"] = str(fill_light_intensity)
    if neutral_environment:
        env["PARITY_NEUTRAL_ENVIRONMENT"] = "1"
    if robot_review_material_override:
        env["PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE"] = "1"
    if robot_review_material_mode:
        env["PARITY_ROBOT_REVIEW_MATERIAL_MODE"] = str(robot_review_material_mode)
    if kinematic_arm_pose:
        env["PARITY_KINEMATIC_ARM_POSE"] = "1"
    if collision_approximation:
        env["PARITY_COLLISION_APPROXIMATION"] = str(collision_approximation)
    if verify_cam:
        env["PARITY_VERIFY_CAM"] = "1"
    if manipulation_stand:
        env["PARITY_MANIPULATION_STAND"] = "1"
    if not placement_topdown_capture:
        env["PARITY_NO_PLACEMENT_TOPDOWN_CAPTURE"] = "1"
    if render_noise_audit:
        env["PARITY_RENDER_NOISE_AUDIT"] = "1"
        if audit_high_spp and audit_high_spp > 0:
            env["PARITY_AUDIT_HIGH_SPP"] = str(int(audit_high_spp))
        if audit_warmup_frames and audit_warmup_frames > 0:
            env["PARITY_AUDIT_WARMUP_FRAMES"] = str(int(audit_warmup_frames))
        if audit_boost_light_intensity and audit_boost_light_intensity > 0:
            env["PARITY_AUDIT_BOOST_LIGHT_INTENSITY"] = str(float(audit_boost_light_intensity))
    if kitchen_url:
        env["KITCHEN_BUNDLE_URL"] = kitchen_url
    if gemini_api_key:
        env["GOOGLE_GENAI_API_KEY"] = gemini_api_key
        env["GEMINI_API_KEY"] = gemini_api_key
    if groot_policy_command:
        env["PARITY_GROOT_POLICY_COMMAND"] = str(groot_policy_command)
        env["PARITY_GROOT_POLICY_COMMAND_TIMEOUT_SECONDS"] = str(
            float(groot_policy_command_timeout_seconds)
        )
    if serve:
        env["PARITY_SERVE"] = "1"
        env["PARITY_SERVE_IDLE_TIMEOUT"] = str(int(serve_idle_timeout_s))
        if serve_max_jobs is not None:
            env["PARITY_SERVE_MAX_JOBS"] = str(int(serve_max_jobs))
        if inbox_get_url:
            raise ValueError(
                "single_object_warm_inbox_retired_use_durable_broker"
            )
        if bool(warm_broker_base_url) != bool(warm_broker_token):
            raise ValueError("warm_render_broker_url_and_token_required_together")
        if warm_broker_base_url:
            env["BLUEPRINT_WARM_RENDER_BROKER_BASE_URL"] = warm_broker_base_url
            env["BLUEPRINT_WARM_RENDER_BROKER_TOKEN"] = warm_broker_token
    return RenderLaunchSpec(
        name="blueprint-isaac-g1-kitchen-parity", image=image, env=env,
        bootstrap_argv=docker_start_cmd(image_startup_canary=image_startup_canary),
        entrypoint=["bash"],
        container_disk_gb=container_disk_gb, volume_gb=volume_gb,
        max_hourly_rate_usd=(
            float(vast_max_hourly_rate_usd)
            if vast_max_hourly_rate_usd is not None and vast_max_hourly_rate_usd > 0
            else _vast_max_hourly_rate_from_env()
        ),
    )


# ----------------------------- WAM-ready harness package -----------------------------

def build_harness_package(
    *,
    result: dict,
    render_out_dir: Path,
    out_dir: Path,
    requested_render_settings: dict | None = None,
) -> dict:
    """Assemble the WAM-ready harness package from the collected Isaac run: per-scenario MP4s +
    traces + outcome become the inputs the WAM video-fidelity evaluator consumes. Running the
    OSCAR/COSMOS model itself is a separate GPU/checkpoint-gated step; this packages its inputs
    and records the honest claim boundary."""
    scenarios = result.get("scenarios", []) if isinstance(result, dict) else []
    items = []
    for sc in scenarios:
        sid = sc.get("scenario_id")
        sdir = render_out_dir / str(sid)
        overview_mp4 = sdir / "overview.mp4"
        robot_pov_mp4 = sdir / "robot_pov.mp4"
        items.append({
            "scenario_id": sid,
            "task_success": sc.get("task_success"),
            "task_success_contract": sc.get("task_success_contract"),
            "review_task_success": sc.get("review_task_success"),
            "success_claim_ledger": sc.get("success_claim_ledger"),
            "trace_jsonl": str(sdir / "trace.jsonl"),
            "overview_mp4": str(overview_mp4),
            "robot_pov_mp4": str(robot_pov_mp4),
            "wam_reference_video": str(overview_mp4),
            "media_metadata": {
                "overview_mp4": _probe_video_file(overview_mp4),
                "robot_pov_mp4": _probe_video_file(robot_pov_mp4),
            },
        })
    package = {
        "schema_version": "isaac_g1_kitchen_parity_harness.v1",
        "policy_id": result.get("policy_id") if isinstance(result, dict) else None,
        "rendered_by_isaac_rtx": True,
        "requested_render_settings": requested_render_settings or {},
        "scenarios_executed": result.get("scenarios_executed") if isinstance(result, dict) else 0,
        "scenarios_passed": result.get("scenarios_passed") if isinstance(result, dict) else 0,
        "wam_evaluator": {
            "evaluator_id": "oscar_cosmos_wam_evaluator",
            "evaluates": "video_rollout_fidelity_not_task_success",
            "status": "inputs_ready_pending_model_run",
            "inputs": items,
        },
        "claim_boundary": (
            "Isaac RTX kinematic walk-to-target preview parity with the MuJoCo lane. The WAM "
            "evaluator judges generated-rollout video fidelity, not task success or readiness. "
            "task_success here means only that the scenario's declared task_success_contract "
            "(e.g. root navigation within tolerance, or visible reach) passed in the trace — "
            "not that the described manipulation happened, not contact, not object state "
            "change, and not physical readiness. scenarios_passed counts those contract "
            "passes, nothing stronger."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "isaac_g1_kitchen_parity_harness.json").write_text(json.dumps(package, indent=2))
    return package


# ----------------------------- launch with flaky-pod retry -----------------------------

def _request_with_launch_session_nonce(request: dict, launch_session_id: str) -> dict:
    """Return a provider request copy whose worker environment carries this launch nonce."""
    request_copy = deepcopy(request)
    env = request_copy.get("env")
    if not isinstance(env, dict):
        env = {}
        request_copy["env"] = env
    env["BLUEPRINT_LAUNCH_SESSION_ID"] = launch_session_id
    create_payload = request_copy.get("create_payload")
    if isinstance(create_payload, dict):
        create_env = create_payload.get("env")
        if not isinstance(create_env, dict):
            create_env = {}
            create_payload["env"] = create_env
        create_env["BLUEPRINT_LAUNCH_SESSION_ID"] = launch_session_id
    return request_copy


def _provider_names(provider: str | None) -> list[str]:
    names = [
        p.strip().lower()
        for p in str(provider or DEFAULT_ISAAC_REVIEW_PROVIDER).split(",")
        if p.strip()
    ]
    return names or [DEFAULT_ISAAC_REVIEW_PROVIDER]


def _env_truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _apply_paid_provider_policy(provider_names: Sequence[str], *, allow_paid: bool) -> tuple[list[str], dict]:
    requested = [str(p).strip().lower() for p in provider_names if str(p).strip()]
    if not allow_paid or "vast" not in requested or _env_truthy(ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV):
        return requested, {
            "status": "allowed",
            "requested_providers": requested,
            "paid_launch": bool(allow_paid),
        }
    filtered = [p for p in requested if p != "vast"]
    policy = {
        "status": "degraded" if filtered else "blocked",
        "requested_providers": requested,
        "effective_providers": filtered,
        "disabled_paid_providers": ["vast"],
        "blocker": "vast_provider_disabled_for_paid_isaac_review",
        "override_env": ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV,
        "reason": (
            "Vast is not used as a default paid Isaac review-render fallback because this lane "
            "requires reliable bootstrap/output markers before WAM can consume a seed."
        ),
    }
    return filtered, policy


def _isaac_g1_prelaunch_spend_guard(
    *,
    allow_paid: bool,
    provider_name: str,
    max_spend_usd: float | None,
    max_seconds: int,
    max_hourly_rate_usd: float | None,
    contender_count: int = 1,
) -> dict:
    env_budget = _float_or_none(os.getenv(ISAAC_G1_MAX_SPEND_USD_ENV))
    requested_budget = max_spend_usd if max_spend_usd is not None else env_budget
    hourly_rate = (
        float(max_hourly_rate_usd)
        if max_hourly_rate_usd is not None and max_hourly_rate_usd > 0
        else DEFAULT_VAST_MAX_HOURLY_RATE_USD
    )
    seconds = max(0, int(max_seconds or 0))
    contenders = max(1, int(contender_count or 1))
    estimated_max_spend_usd = round((hourly_rate * (seconds / 3600.0)) * contenders, 4)
    blockers: list[str] = []
    if not allow_paid:
        blockers.append("paid_launch_not_requested")
    if requested_budget is None:
        blockers.append("isaac_g1_max_spend_usd_missing")
    elif requested_budget <= 0:
        blockers.append("isaac_g1_max_spend_usd_must_be_positive")
    elif estimated_max_spend_usd > float(requested_budget):
        blockers.append("isaac_g1_estimated_spend_exceeds_budget")
    can_launch = bool(allow_paid and not blockers)
    return {
        "schema_version": "isaac_g1_kitchen_parity_prelaunch_spend_guard.v1",
        "status": "passed" if can_launch else "blocked",
        "provider": provider_name,
        "allow_paid": bool(allow_paid),
        "required_before_provider_launch": True,
        "can_launch": can_launch,
        "requested_budget_usd": requested_budget,
        "budget_source": "argument"
        if max_spend_usd is not None
        else ("env" if env_budget is not None else "missing"),
        "estimated_max_spend_usd": estimated_max_spend_usd,
        "max_hourly_rate_usd": hourly_rate,
        "max_seconds": seconds,
        "contender_count": contenders,
        "blockers": blockers,
        "claim_boundary": {
            "spend_guard_only": True,
            "can_launch_is_not_provider_success": True,
            "can_launch_is_not_task_success": True,
            "no_provider_api_call_before_can_launch": True,
        },
    }


def _paid_worker_image_policy(
    *,
    image: str | None,
    allow_paid: bool,
    provider_names: Sequence[str],
    cold: bool,
    warm_only: bool,
    image_startup_canary: bool,
) -> tuple[str, dict]:
    cli_image = _string(image)
    image_config = _configured_isaac_worker_image_ref()
    direct_base_image_allowed = _env_truthy(ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV)
    if cli_image:
        selected_image = cli_image
        configured = True
        source = "cli_image_arg"
    else:
        selected_image = _string(image_config.get("image_ref")) or DEFAULT_PARITY_IMAGE_REF
        configured = bool(image_config.get("configured"))
        source = image_config.get("source") or "default_historical_parity_image"
    image_size_diagnostic = _isaac_worker_image_size_diagnostic(selected_image)
    blockers: list[str] = []
    if allow_paid and not configured and not direct_base_image_allowed:
        blockers.append("prebuilt_isaac_eval_worker_image_ref_missing")
    cold_start_possible = bool(cold or not warm_only)
    large_runpod_fresh_start = bool(
        allow_paid
        and "runpod" in {str(p).strip().lower() for p in provider_names}
        and cold_start_possible
        and image_size_diagnostic.get("large_image_pull_risk")
        and not image_startup_canary
        and not _env_truthy(ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV)
    )
    if large_runpod_fresh_start:
        blockers.append("large_worker_image_requires_canary_or_warm_provider")
    policy = {
        "status": "blocked" if blockers else "allowed",
        "selected_image_ref": selected_image,
        "image_ref_source": source,
        "prebuilt_worker_image_ref_configured": configured,
        "worker_image_ref_file": image_config.get("image_ref_file"),
        "worker_image_ref_file_present": bool(image_config.get("image_ref_file_present")),
        "worker_image_manifest_diagnostic": image_size_diagnostic,
        "direct_isaac_base_image_runpod_allowed": direct_base_image_allowed,
        "direct_isaac_base_image_override_env": ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV,
        "large_runpod_image_fresh_start_allowed": _env_truthy(
            ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV
        ) or bool(image_startup_canary),
        "image_startup_canary": bool(image_startup_canary),
        "runpod_cold_start_possible": cold_start_possible,
        "large_runpod_image_fresh_start_override_env": ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV,
        "blockers": blockers,
        "blocker": blockers[0] if blockers else None,
        "claim_boundary": (
            "Worker image policy only gates provider startup risk. It does not prove "
            "container startup, Isaac execution, rendered RGB quality, WAM quality, or "
            "robot readiness."
        ),
    }
    return selected_image, policy


def _groot_sonic_policy_runtime_policy(
    *,
    policy_id: str,
    selected_image: str,
    allow_paid: bool,
    image_startup_canary: bool,
    effective_groot_policy_command: str,
    effective_groot_policy_command_timeout_seconds: float,
) -> dict:
    requested = str(policy_id).strip() in {
        "groot_sonic",
        "groot",
        "groot_n17_sonic",
        "unitree_groot_n17_sonic_policy",
    }
    command_configured = bool(_string(effective_groot_policy_command))
    server_url_configured = bool(_string(os.getenv(UNITREE_GROOT_POLICY_SERVER_URL_ENV)))
    runtime_mode = _string(os.getenv(ISAAC_G1_GROOT_POLICY_RUNTIME_MODE_ENV)).lower()
    prebaked_image_confirmed = _env_truthy(ISAAC_G1_GROOT_POLICY_PREBAKED_IMAGE_CONFIRMED_ENV)
    selected_image_text = _string(selected_image).lower()
    image_looks_like_plain_isaac_worker = (
        "isaac-eval-worker" in selected_image_text
        and "groot" not in selected_image_text
        and "sonic" not in selected_image_text
    )

    if image_startup_canary or not requested:
        return {
            "status": "not_applicable",
            "policy_id": policy_id,
            "groot_sonic_policy_requested": requested,
            "image_startup_canary": bool(image_startup_canary),
            "blockers": [],
            "raw_secret_values_recorded": False,
        }

    required_contract = groot_sonic_isaac_bridge_readiness(
        {
            "action": {
                "hand_targets": {
                    "left_hand_joints": [0.0],
                    "right_hand_joints": [0.0],
                }
            }
        }
    )
    base = {
        "policy_id": policy_id,
        "groot_sonic_policy_requested": True,
        "policy_command_configured": command_configured,
        "policy_command_value_redacted": "<configured>" if command_configured else None,
        "policy_command_env_candidates": [
            ISAAC_G1_GROOT_POLICY_COMMAND_ENV,
            UNITREE_GROOT_POLICY_COMMAND_ENV,
        ],
        "policy_command_timeout_seconds": effective_groot_policy_command_timeout_seconds,
        "selected_image_ref": selected_image,
        "selected_image_looks_like_plain_isaac_worker": image_looks_like_plain_isaac_worker,
        "policy_server_url_configured": server_url_configured,
        "policy_server_url_env": UNITREE_GROOT_POLICY_SERVER_URL_ENV,
        "runtime_mode_env": ISAAC_G1_GROOT_POLICY_RUNTIME_MODE_ENV,
        "runtime_mode": runtime_mode or None,
        "prebaked_image_confirmed_env": (
            ISAAC_G1_GROOT_POLICY_PREBAKED_IMAGE_CONFIRMED_ENV
        ),
        "prebaked_image_confirmed": prebaked_image_confirmed,
        "runtime_dependency_install_disallowed_for_paid_launch": bool(allow_paid),
        "required_isaac_control_contract": required_contract,
        "raw_secret_values_recorded": False,
        "claim_boundary": (
            "This is a spend/readiness gate for locating the GR00T/SONIC policy runtime "
            "from an Isaac worker. It does not prove policy quality, simulator task success, "
            "physical robot readiness, or prior Unitree GR00T/SONIC provider action-command "
            "evidence."
        ),
    }
    blockers: list[str] = []
    if not command_configured:
        blockers.append("groot_sonic_policy_not_connected_to_isaac_parity_runner")
    runtime_location_proven = False
    runtime_location_source = None
    if server_url_configured:
        runtime_location_proven = True
        runtime_location_source = "external_policy_server_url"
    elif runtime_mode in {"prebaked_worker_image", "prebaked_image", "sealed_worker_image"}:
        if prebaked_image_confirmed:
            runtime_location_proven = True
            runtime_location_source = "prebaked_worker_image_contract"
        else:
            blockers.append("groot_sonic_prebaked_image_contract_not_confirmed")
    else:
        blockers.append("groot_sonic_policy_runtime_presence_not_proven_for_selected_image")

    if allow_paid and blockers:
        return {
            **base,
            "status": "blocked",
            "runtime_location_proven": runtime_location_proven,
            "runtime_location_source": runtime_location_source,
            "blockers": blockers,
            "blocker": blockers[0],
            "reason": (
                "A paid Isaac GR00T/SONIC run needs proof that the worker can reach a live "
                "PolicyServer or that the selected image already contains the GR00T runtime, "
                "server dependencies, and checkpoints. A command string alone is not enough."
            ),
            "safe_next_path": (
                "Set BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL for an external "
                "policy server, or run a sealed combined Isaac+GR00T image and set "
                "BLUEPRINT_ISAAC_G1_GROOT_POLICY_RUNTIME_MODE=prebaked_worker_image plus "
                "BLUEPRINT_ISAAC_G1_GROOT_POLICY_PREBAKED_IMAGE_CONFIRMED=true."
            ),
        }

    status = "configured" if command_configured else "blocked"
    if blockers and not allow_paid:
        status = "configured_unproven_no_spend_plan" if command_configured else "blocked"
    return {
        **base,
        "status": status,
        "runtime_location_proven": runtime_location_proven,
        "runtime_location_source": runtime_location_source,
        "blockers": blockers,
        "blocker": blockers[0] if blockers else None,
        "reason": (
            "No-spend plan only; runtime location is recorded but not paid-launched."
            if blockers and not allow_paid
            else None
        ),
    }


def _provider_startup_pre_runtime(snapshot: dict) -> bool:
    return (
        snapshot.get("status") == "observed"
        and int(snapshot.get("http") or 0) == 200
        and snapshot.get("runtime_present") is False
        and snapshot.get("public_ip_present") is False
    )


def _unique_strings(values: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value and value not in ordered:
            ordered.append(value)
    return ordered


def _launch_attempt_detail_blockers(attempts: Sequence[dict]) -> list[str]:
    blockers: list[str] = []
    for attempt in attempts:
        detail = attempt.get("detail") if isinstance(attempt, dict) else None
        if not isinstance(detail, dict):
            continue
        blockers.extend(str(item) for item in (detail.get("blockers") or []) if item)
    return _unique_strings(blockers)


def _launch_failure_blockers(attempts: Sequence[dict]) -> list[str]:
    """Classify launch failures without conflating provider capacity with dead pods."""
    detail_blockers = _launch_attempt_detail_blockers(attempts)
    capacity_blockers = [
        blocker
        for blocker in detail_blockers
        if blocker in PROVIDER_CAPACITY_UNAVAILABLE_BLOCKERS
    ]
    if capacity_blockers and all(
        (item.get("result") if isinstance(item, dict) else None) == "launch_call_failed"
        for item in attempts
    ):
        return _unique_strings([
            *capacity_blockers,
            "provider_capacity_unavailable_before_instance_created",
        ])
    final_blockers = ["all_launch_attempts_flaky"]
    if any(
        str(item.get("result") or "").startswith("startup_no_runtime_timeout")
        for item in attempts
        if isinstance(item, dict)
    ):
        final_blockers.append("provider_startup_no_runtime_timeout")
    return final_blockers


def _paid_launch_pending_teardown_max_age(
    *,
    marker_timeout: int,
    startup_no_runtime_timeout: int,
    max_attempts: int,
) -> int:
    per_attempt = max(
        int(marker_timeout or 0),
        int(startup_no_runtime_timeout or 0),
        60,
    )
    return max(300, per_attempt * max(1, int(max_attempts or 1)) + 1800)


def _teardown_proof_from_attempt(
    *,
    provider: Any,
    instance_id: str,
    teardown: Mapping[str, Any],
    action: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    provider_name = _string(getattr(provider, "name", "")) or "unknown"
    status = _string(teardown.get("status")).lower()
    action_text = _string(action).lower()
    if action_text == "stop":
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=status or "stopped_for_warm_reuse",
        )
    verification: dict[str, Any] = {}
    if hasattr(provider, "inspect"):
        try:
            verification = provider_state_from_inspect(provider.inspect(instance_id))
        except Exception as exc:  # noqa: BLE001 - failed verification is evidence, not a crash
            verification = {
                "api_confirmed": False,
                "provider_status": "",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    observed_status = _string(verification.get("provider_status")).lower()
    if verification.get("api_confirmed") is True and observed_status:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed_status,
            verified_at=generated_at or utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider=provider_name,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=generated_at or utc_now_iso() if status == "terminated" else None,
    )


def _teardown_proof_from_watch_result(
    *,
    provider_name: str,
    instance_id: str,
    watch: Mapping[str, Any],
) -> dict[str, Any]:
    teardown = _mapping(watch.get("teardown"))
    reason = _string(watch.get("teardown_reason")).lower()
    status = _string(teardown.get("status")).lower()
    if status in {"stopped", "preserved", "skipped"} or reason in {
        "left_running_by_request",
        "runner_done_preserved_for_warm_reuse",
    }:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=False,
            provider_terminal_status=None,
            keep_alive_requested=True,
            keep_alive_reason=reason or status or "kept_alive",
        )
    verification = _mapping(teardown.get("verification"))
    observed_status = _string(verification.get("provider_status")).lower()
    if verification.get("api_confirmed") is True and observed_status:
        return build_teardown_proof(
            provider=provider_name,
            allocation_id=instance_id,
            terminate_requested=True,
            provider_terminal_status=observed_status,
            verified_at=utc_now_iso(),
            status_source=TEARDOWN_STATUS_SOURCE_PROVIDER_API,
        )
    return build_teardown_proof(
        provider=provider_name,
        allocation_id=instance_id,
        terminate_requested=True,
        provider_terminal_status="terminated" if status == "terminated" else status or None,
        verified_at=utc_now_iso() if status == "terminated" else None,
    )


class _ColdCreateContender:
    """Provider proxy for same-provider cold-create racing."""

    def __init__(self, provider) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return self._provider.name

    def launch(self, job_dir, request, *, cold: bool = False, **kwargs):
        return self._provider.launch(job_dir, request, cold=True, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._provider, attr)


def resolve_cold_race_contenders(value: int | None = None) -> int:
    """How many same-provider cold creates to race. CLI/param wins over env."""
    raw = value if value is not None else os.getenv(COLD_RACE_CONTENDERS_ENV, "")
    try:
        count = int(str(raw).strip() or DEFAULT_COLD_RACE_CONTENDERS)
    except (TypeError, ValueError):
        count = DEFAULT_COLD_RACE_CONTENDERS
    return max(1, min(4, count))


def launch_with_marker_retry(prov, job_dir: Path, request: dict, *, max_attempts: int = 3,
                             marker_timeout: int = 150, poll: int = 15,
                             cold: bool = True,
                             allow_cold_fallback: bool = True,
                             startup_no_runtime_timeout: int = 0,
                             prelaunch_guard: dict | None = None) -> dict:
    """Launch a pod, then wait for its container's early heartbeat (``bootstrap.json`` on the
    output URL). RunPod cold pods are ~50% flaky — created + billing but the container never runs.
    If no marker appears within ``marker_timeout``, terminate that pod and retry, so we never pay
    for a dead cold pod. Warm-restart duds are stopped rather than deleted so a preserved pod can be
    reused later. Returns the launch of the first pod that actually started."""
    attempts: list[dict] = []
    trace = {
        "schema_version": "isaac_g1_kitchen_parity_launch_attempts.v1",
        "status": "starting",
        "provider": getattr(prov, "name", "unknown"),
        "marker_timeout_seconds": int(marker_timeout),
        "poll_seconds": int(poll),
        "max_attempts": int(max_attempts),
        "startup_no_runtime_timeout_seconds": int(startup_no_runtime_timeout),
        "cold": bool(cold),
        "allow_cold_fallback": bool(allow_cold_fallback),
        "prelaunch_guard": prelaunch_guard,
        "attempts": attempts,
        "proof_boundary": (
            "Launch-attempt trace only. It proves provider API/result observation, "
            "not container startup, Isaac execution, rendered RGB quality, WAM quality, "
            "or robot readiness."
        ),
    }
    trace_path = _write_launch_attempt_trace(job_dir, trace)
    if prelaunch_guard and prelaunch_guard.get("can_launch") is not True:
        blockers = [
            "isaac_g1_prelaunch_spend_guard_not_passed",
            *[str(item) for item in prelaunch_guard.get("blockers") or []],
        ]
        attempts.append(
            {
                "attempt": 0,
                "result": "prelaunch_blocked",
                "prelaunch_guard": prelaunch_guard,
                "blockers": blockers,
            }
        )
        trace["status"] = "prelaunch_blocked"
        trace["blockers"] = blockers
        _write_launch_attempt_trace(job_dir, trace)
        return {
            "status": "blocked",
            "blockers": blockers,
            "attempts": attempts,
            "attempt_trace_path": str(trace_path),
            "prelaunch_guard": prelaunch_guard,
        }
    for attempt in range(max_attempts):
        launch_session_id = uuid.uuid4().hex
        request_for_launch = _request_with_launch_session_nonce(request, launch_session_id)
        if prelaunch_guard:
            request_for_launch["prelaunch_spend_guard"] = prelaunch_guard
        (job_dir / "launch_session_nonce.txt").write_text(launch_session_id, encoding="utf-8")
        pending_teardown: dict[str, Any] | None = None
        if prelaunch_guard and prelaunch_guard.get("can_launch") is True:
            pending_teardown = open_pending_teardown(
                provider=getattr(prov, "name", "unknown"),
                lane=ISAAC_G1_KITCHEN_PARITY_LANE,
                run_id=launch_session_id,
                job_dir=job_dir,
                max_age_seconds=_paid_launch_pending_teardown_max_age(
                    marker_timeout=int(marker_timeout),
                    startup_no_runtime_timeout=int(startup_no_runtime_timeout),
                    max_attempts=int(max_attempts),
                ),
            )
            request_for_launch["pending_teardown_record"] = pending_teardown["path"]
        try:
            launch = prov.launch(
                job_dir,
                request_for_launch,
                cold=cold,
                allow_cold_fallback=allow_cold_fallback,
            )
        except Exception:
            if pending_teardown:
                cancel_pending_teardown(
                    pending_teardown["path"],
                    reason="provider_launch_raised_before_allocation",
                )
            raise
        if pending_teardown:
            launch["pending_teardown_record"] = pending_teardown["path"]
            if launch.get("instance_id"):
                bind_pending_teardown_instance(
                    pending_teardown["path"], str(launch["instance_id"])
                )
        if launch.get("status") != "launched":
            if pending_teardown and not launch.get("instance_id"):
                cancel_pending_teardown(
                    pending_teardown["path"],
                    reason="launch_returned_no_allocation",
                    evidence=launch,
                )
            attempts.append({"attempt": attempt, "result": "launch_call_failed", "detail": launch})
            trace["status"] = "launch_call_failed"
            _write_launch_attempt_trace(job_dir, trace)
            blockers = {str(b) for b in (launch.get("blockers") or [])}
            if "warm_restart_failed_cold_fallback_disabled" in blockers:
                return {
                    "status": "blocked",
                    "blockers": ["warm_restart_failed_cold_fallback_disabled"],
                    "attempts": attempts,
                    "attempt_trace_path": str(trace_path),
                }
            continue
        iid = launch["instance_id"]
        attempt_record = {
            "attempt": attempt,
            "instance_id": iid,
            "marker_seen": False,
            "launch_session_id": launch_session_id,
            "launch_mode": launch.get("mode"),
            "result": "waiting_for_marker",
            "marker_timeout_seconds": int(marker_timeout),
            "startup_no_runtime_timeout_seconds": int(startup_no_runtime_timeout),
            "pending_teardown_record": launch.get("pending_teardown_record"),
        }
        attempts.append(attempt_record)
        trace["status"] = "waiting_for_marker"
        _write_launch_attempt_trace(job_dir, trace)
        t0 = time.time()
        marker_seen = False
        startup_no_runtime = False
        while time.time() - t0 < marker_timeout:
            time.sleep(poll)
            marker_seen = boot_marker_present(
                job_dir,
                expected_launch_session_id=launch_session_id,
                urlopen=urllib.request.urlopen,
            )
            if marker_seen:
                break
            elapsed = time.time() - t0
            if startup_no_runtime_timeout and elapsed >= startup_no_runtime_timeout:
                try:
                    startup_snapshot = prov.inspect(iid)
                except Exception as exc:  # noqa: BLE001
                    startup_snapshot = {
                        "status": "unavailable",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_provider_response_recorded": False,
                    }
                attempt_record["startup_no_runtime_snapshot"] = startup_snapshot
                attempt_record["startup_no_runtime_elapsed_seconds"] = round(elapsed, 1)
                if _provider_startup_pre_runtime(startup_snapshot):
                    startup_no_runtime = True
                    break
        attempt_record["elapsed_seconds"] = round(time.time() - t0, 1)
        attempt_record["marker_seen"] = marker_seen
        if marker_seen:
            attempt_record["result"] = "marker_verified"
            trace["status"] = "marker_verified"
            _write_launch_attempt_trace(job_dir, trace)
            mode = launch.get("mode") or ("cold_create" if cold else "warm_or_cold")
            return {"status": "launched", "instance_id": iid, "mode": f"{mode}_marker_verified",
                    "attempts": attempts, "attempt_trace_path": str(trace_path),
                    "pending_teardown_record": launch.get("pending_teardown_record")}
        if str(launch.get("mode") or "").startswith("warm"):
            teardown = prov.stop(iid)
            attempt_record["teardown"] = teardown
            attempt_record["teardown_action"] = "stop"
            attempt_record["result"] = (
                "startup_no_runtime_timeout_stopped"
                if startup_no_runtime
                else "marker_timeout_stopped"
            )
        else:
            teardown = prov.terminate(iid)  # flaky cold pod (billing but not running) -> kill and retry
            attempt_record["teardown"] = teardown
            attempt_record["teardown_action"] = "terminate"
            attempt_record["result"] = (
                "startup_no_runtime_timeout_terminated"
                if startup_no_runtime
                else "marker_timeout_terminated"
            )
        if launch.get("pending_teardown_record"):
            proof = _teardown_proof_from_attempt(
                provider=prov,
                instance_id=iid,
                teardown=teardown if isinstance(teardown, Mapping) else {},
                action=attempt_record["teardown_action"],
            )
            attempt_record["teardown_proof"] = proof
            closure = close_pending_teardown(
                launch["pending_teardown_record"],
                proof,
            )
            attempt_record["pending_teardown_status"] = closure.get("status")
        trace["status"] = attempt_record["result"]
        _write_launch_attempt_trace(job_dir, trace)
    trace["status"] = "blocked"
    final_blockers = _launch_failure_blockers(attempts)
    trace["blockers"] = final_blockers
    _write_launch_attempt_trace(job_dir, trace)
    return {
        "status": "blocked",
        "blockers": final_blockers,
        "attempts": attempts,
        "attempt_trace_path": str(trace_path),
    }


# ----------------------------- orchestration -----------------------------

def _await_warm_serve_ready(job_dir: Path, *, instance_id: str, timeout_s: int = 1800,
                            poll_interval_s: float = 15.0,
                            launch_session_id: str | None = None) -> dict:
    """Poll the worker's uploaded output zip for the --serve readiness marker (Isaac booted + scene
    loaded + the loop accepting jobs). Returns {ready, elapsed_seconds, last_phase}. Does NOT tear the
    pod down — the warm pod must stay running for the caller's WarmPoolClient."""
    import io as _io
    import time as _time
    import zipfile as _zip

    get_url_file = job_dir / "provider_output_get_url.txt"
    if not get_url_file.is_file():
        return {"ready": False, "reason": "missing_output_get_url", "instance_id": instance_id}
    get_url = get_url_file.read_text().strip()
    expected_session = str(launch_session_id or "").strip()
    if not expected_session:
        nonce_file = job_dir / "launch_session_nonce.txt"
        if nonce_file.is_file():
            expected_session = nonce_file.read_text(encoding="utf-8").strip()
    start = _time.monotonic()
    last_phase = None
    while _time.monotonic() - start < timeout_s:
        try:
            data = _fetch_provider_artifact_bytes(
                get_url,
                timeout=60,
                max_bytes=MAX_WARM_READINESS_ARCHIVE_BYTES,
            )
            with _zip.ZipFile(_io.BytesIO(data)) as z:
                names = z.namelist()
                bootstrap_session = ""
                if "bootstrap.json" in names:
                    try:
                        bootstrap_detail = json.loads(z.read("bootstrap.json").decode())
                        if isinstance(bootstrap_detail, dict):
                            last_phase = bootstrap_detail.get("phase")
                            bootstrap_session = str(
                                bootstrap_detail.get("launch_session_id") or ""
                            ).strip()
                    except Exception:  # noqa: BLE001
                        bootstrap_detail = {}
                        bootstrap_session = ""
                        pass
                if "warm_serve_ready.json" in names:
                    detail = {}
                    try:
                        detail = json.loads(z.read("warm_serve_ready.json").decode())
                    except Exception:  # noqa: BLE001
                        pass
                    if expected_session and str(detail.get("launch_session_id") or "") != expected_session:
                        _time.sleep(poll_interval_s)
                        continue
                    return {"ready": True, "elapsed_seconds": round(_time.monotonic() - start, 1),
                            "last_phase": last_phase, "serve_detail": detail, "instance_id": instance_id}
                if (
                    last_phase in {"runner_done", "runner_timeout"}
                    and (
                        not expected_session
                        or bootstrap_session == expected_session
                    )
                ):
                    reason = (
                        "runner_timeout_without_warm_serve_ready"
                        if last_phase == "runner_timeout"
                        else "runner_completed_without_warm_serve_ready"
                    )
                    return {
                        "ready": False,
                        "reason": reason,
                        "elapsed_seconds": round(_time.monotonic() - start, 1),
                        "last_phase": last_phase,
                        "instance_id": instance_id,
                        "zip_entries": sorted(names),
                    }
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return {
                    "ready": False,
                    "reason": "presigned_url_expired_or_forbidden",
                    "http_status": exc.code,
                    "elapsed_seconds": round(_time.monotonic() - start, 1),
                    "last_phase": last_phase,
                    "instance_id": instance_id,
                }
        except Exception:  # noqa: BLE001 - output not posted yet / mid-upload
            pass
        _time.sleep(poll_interval_s)
    return {"ready": False, "reason": "serve_ready_timeout", "elapsed_seconds": round(_time.monotonic() - start, 1),
            "last_phase": last_phase, "instance_id": instance_id}


def run_isaac_g1_kitchen_parity_job(
    *, scenarios: Sequence[dict], out_dir: str | Path, kitchen_asset_dir: str | Path | None = None,
    kitchen_url: str | None = None,
    g1_usd: str = DEFAULT_G1_USD_RELATIVE, policy_id: str = "blueprint_default_walk_to_target_smoke_policy",
    steps: int = 64, provider: str = DEFAULT_ISAAC_REVIEW_PROVIDER, allow_paid: bool = False,
    allow_dirty_paid_launch: bool = False, cold: bool = False,
    image: str | None = None, key_prefix: str = "blueprint/isaac-g1-parity", max_seconds: int = 1500,
    marker_timeout: int = 900, max_attempts: int = 3,
    post_marker_progress_timeout: int = 360,
    startup_no_runtime_timeout: int = DEFAULT_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS,
    cold_race_contenders: int | None = None,
    width: int = 1280, height: int = 960, fps: int = 20,
    warmup: int = 6, per_scenario_seconds: int = 420,
    container_disk_gb: int = 140, volume_gb: int = 80,
    no_collision_probe: bool = False, focus_radius: float = 0.0, keep_objects: str = "",
    settle_seconds: int = 0, cheap_collision: bool = False, articulated: bool = False,
    physics_articulation_drive: bool = False,
    dynamic_standing_contact_steps: int = 0,
    manipulation_cam: bool = False, manipulation_look_at: str = "", render_subframes: int = 0,
    manipulation_reach: bool = False, manipulation_reach_arm: str = "auto",
    dynamic_episode_termination: bool = True, episode_max_steps: int = 0,
    dynamic_episode_check_every: int = 1, capture_every: int = 1,
    fill_light_intensity: float = 0.0,
    neutral_environment: bool = False,
    robot_review_material_override: bool = False,
    robot_review_material_mode: str = "",
    kinematic_arm_pose: bool = False,
    collision_approximation: str = "",
    verify_cam: bool = False,
    manipulation_stand: bool = False,
    placement_topdown_capture: bool = True,
    render_noise_audit: bool = False,
    audit_high_spp: int = 0,
    audit_warmup_frames: int = 0,
    audit_boost_light_intensity: float = 0.0,
    vast_max_hourly_rate_usd: float | None = None,
    max_spend_usd: float | None = None,
    warm_candidates: Sequence[str] | None = None,
    warm_only: bool = False,
    serve: bool = False, serve_idle_timeout_s: float = 1800.0,
    serve_max_jobs: int | None = None, serve_ready_timeout: int = 1800,
    image_startup_canary: bool = False,
    groot_policy_command: str = "",
    groot_policy_command_timeout_seconds: float | None = None,
) -> dict:
    """Full parity job. Without ``allow_paid`` it bundles + stages and returns a launchable plan.

    Warm serve mode (``serve=True``): presign a job-inbox channel, launch ONE pod with ``--serve``
    (boots Isaac + loads the scene once, then polls the inbox), wait for its readiness marker, and
    return WITH THE POD LEFT RUNNING (no watch_and_collect / teardown). The caller drives it with a
    :class:`~blueprint_pipeline.warm_render_server.WarmPoolClient` (submit jobs / collect results) and
    is responsible for tearing it down. ``serve`` allows an empty ``scenarios`` list (jobs arrive live)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_provider_names = _provider_names(provider)
    provider_names = list(requested_provider_names)
    manifest: dict = {"schema_version": SCHEMA_VERSION, "status": "blocked", "blockers": [],
                      "provider": ",".join(provider_names), "policy_id": policy_id,
                      "rendered_by": "isaac_rtx_g1_kitchen_parity",
                      "image_startup_canary": bool(image_startup_canary)}
    requested_render_settings = {
        "steps": int(steps),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "warmup_frames": int(warmup),
        "per_scenario_seconds": int(per_scenario_seconds),
        "expected_frame_count_per_scenario": int(steps),
    }
    manifest["requested_render_settings"] = requested_render_settings
    configured_warm_candidates = tuple(
        c.strip()
        for c in (os.getenv("BLUEPRINT_RUNPOD_WARM_CANDIDATES") or "").split(",")
        if c.strip()
    )
    warm_candidate_ids = tuple(warm_candidates or ()) + configured_warm_candidates + tuple(DEFAULT_WARM_CANDIDATES)
    try:
        providers = [
            get_render_provider(
                name,
                warm_candidates=warm_candidate_ids if name == "runpod" else (),
            )
            for name in provider_names
        ]
    except ValueError as exc:
        manifest["blockers"].append("unknown_render_provider")
        manifest["error"] = str(exc)
        return manifest
    if not scenarios and not serve and not image_startup_canary:
        manifest["blockers"].append("no_scenarios")
        return manifest
    manifest["scenario_ids"] = [s.get("scenario_id") or s.get("id") for s in scenarios]
    git_evidence = _git_worktree_evidence()
    manifest["git_evidence"] = git_evidence
    launch_gate = evaluate_dirty_tree_paid_launch_gate(
        git_evidence=git_evidence,
        allow_paid=allow_paid,
        allow_dirty_paid_launch=allow_dirty_paid_launch,
    )
    if not launch_gate["launch_allowed"]:
        manifest["blockers"].extend(launch_gate["blockers"])
        manifest["status"] = "blocked"
        manifest["note"] = launch_gate["note"]
        return manifest
    effective_provider_names, provider_policy = _apply_paid_provider_policy(
        provider_names,
        allow_paid=allow_paid,
    )
    manifest["provider_policy"] = provider_policy
    if not effective_provider_names:
        manifest["provider"] = ",".join(requested_provider_names)
        manifest["blockers"].append("vast_provider_disabled_for_paid_isaac_review")
        manifest["note"] = (
            "Vast paid Isaac review renders are disabled by default for this lane. "
            f"Set {ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV}=1 only for an intentional Vast experiment."
        )
        return manifest
    selected_image, image_policy = _paid_worker_image_policy(
        image=image,
        allow_paid=allow_paid,
        provider_names=effective_provider_names,
        cold=cold,
        warm_only=warm_only,
        image_startup_canary=image_startup_canary,
    )
    manifest["worker_image_policy"] = image_policy
    if image_policy.get("status") == "blocked":
        for blocker in image_policy.get("blockers") or []:
            if blocker not in manifest["blockers"]:
                manifest["blockers"].append(blocker)
        manifest["note"] = (
            "Paid Isaac review renders require a configured, published worker image and "
            "a startup-safe provider path. "
            f"Set {ISAAC_WORKER_IMAGE_REF_ENV}, {ISAAC_WORKER_IMAGE_REF_FILE_ENV}, "
            f"or {ROBOT_EVAL_WORKER_IMAGE_REF_ENV}; use "
            f"{ALLOW_DIRECT_ISAAC_BASE_IMAGE_RUNPOD_ENV}=true or "
            f"{ALLOW_LARGE_RUNPOD_IMAGE_FRESH_START_ENV}=true only for deliberate debug runs."
        )
        return manifest
    effective_groot_policy_command = (
        _string(groot_policy_command)
        or _string(os.getenv(ISAAC_G1_GROOT_POLICY_COMMAND_ENV))
        or _string(os.getenv(UNITREE_GROOT_POLICY_COMMAND_ENV))
    )
    effective_groot_policy_command_timeout_seconds = (
        float(groot_policy_command_timeout_seconds)
        if groot_policy_command_timeout_seconds is not None
        and groot_policy_command_timeout_seconds > 0
        else float(os.getenv(ISAAC_G1_GROOT_POLICY_COMMAND_TIMEOUT_ENV, "120") or 120)
    )
    groot_policy_requested = str(policy_id).strip() in {
        "groot_sonic",
        "groot",
        "groot_n17_sonic",
        "unitree_groot_n17_sonic_policy",
    }
    if not image_startup_canary and groot_policy_requested:
        manifest["policy_runtime_policy"] = _groot_sonic_policy_runtime_policy(
            policy_id=policy_id,
            selected_image=selected_image,
            allow_paid=allow_paid,
            image_startup_canary=image_startup_canary,
            effective_groot_policy_command=effective_groot_policy_command,
            effective_groot_policy_command_timeout_seconds=(
                effective_groot_policy_command_timeout_seconds
            ),
        )
        if manifest["policy_runtime_policy"].get("status") == "blocked":
            manifest["blockers"].extend(manifest["policy_runtime_policy"].get("blockers") or [])
            manifest["blockers"] = sorted(set(manifest["blockers"]))
            return manifest
    if effective_provider_names != provider_names:
        provider_names = effective_provider_names
        manifest["provider"] = ",".join(provider_names)
        manifest["providers"] = provider_names
        providers = [p for p in providers if p.name in set(provider_names)]
    prov = providers[0]
    multi_provider_race = len(providers) > 1 and not serve
    race_contender_count = resolve_cold_race_contenders(cold_race_contenders)
    single_provider_cold_race = (
        not multi_provider_race
        and not serve
        and not warm_only
        and prov.name == "runpod"
        and race_contender_count > 1
    )
    manifest["cold_race_contenders"] = (
        race_contender_count if single_provider_cold_race else 1
    )
    if multi_provider_race:
        manifest["providers"] = [p.name for p in providers]
        manifest["provider_available"] = [p.available() for p in providers]
    else:
        manifest["provider_available"] = prov.available()
        if allow_paid and callable(getattr(prov, "capacity_preflight", None)):
            capacity = prov.capacity_preflight()
            manifest["provider_capacity_preflight"] = capacity
            if capacity.get("status") == "blocked":
                manifest["blockers"].extend(capacity.get("blockers") or [])
                manifest["blockers"].append("provider_capacity_unavailable_before_staging")
                manifest["blockers"] = sorted(set(manifest["blockers"]))
                return manifest
    # Stage the large kitchen tree ONCE (reused across iterations); keep the code bundle tiny.
    # A caller may pass a previously-staged kitchen_url to skip the 1.2GB re-upload entirely.
    kitchen_main_usd_relative = DEFAULT_KITCHEN_MAIN_USD
    if kitchen_url:
        layout = _inspect_kitchen_asset_url_layout(kitchen_url)
        manifest["kitchen_layout_validation"] = layout
        if layout.get("status") != "PASS":
            manifest["blockers"].append("kitchen_asset_layout_validation_failed")
            return manifest
        kitchen_main_usd_relative = str(layout["selected_kitchen_main_usd_relative"])
        manifest["kitchen_staging"] = {
            "status": "reused_existing_url",
            "selected_kitchen_main_usd_relative": kitchen_main_usd_relative,
        }
    elif kitchen_asset_dir is not None:
        layout = _inspect_kitchen_asset_dir_layout(kitchen_asset_dir)
        manifest["kitchen_layout_validation"] = layout
        if layout.get("status") != "PASS":
            manifest["blockers"].append("kitchen_asset_layout_validation_failed")
            return manifest
        kitchen_main_usd_relative = str(layout["selected_kitchen_main_usd_relative"])
        kzip = out_dir / "kitchen_assets.zip"
        _zip_dir(Path(kitchen_asset_dir), kzip)
        kjob = out_dir / "kitchen_object_store"
        kstaged = stage_bundle(kzip, kjob, key_prefix=key_prefix + "/kitchen")
        manifest["kitchen_staging"] = {
            "status": kstaged["status"],
            "zip_bytes": kzip.stat().st_size,
            "selected_kitchen_main_usd_relative": kitchen_main_usd_relative,
        }
        if kstaged["status"] != "completed":
            manifest["blockers"].append("kitchen_staging_failed")
            manifest["kitchen_staging"]["stderr_tail"] = kstaged.get("stderr_tail")
            return manifest
        kitchen_url = (kjob / "provider_bundle_url.txt").read_text().strip()
    manifest["kitchen_assets_shipped"] = kitchen_url is not None
    render_noise_audit_plan = None
    if render_noise_audit:
        from .g1_render_noise_audit import build_variant_plan
        render_noise_audit_plan = build_variant_plan()
        manifest["render_noise_audit_requested"] = True
        manifest["render_noise_audit_variants"] = [
            v["variant_id"] for v in render_noise_audit_plan["variants"]
        ]
    bundle_zip = build_parity_bundle(scenarios=scenarios, out_dir=out_dir,
                                     kitchen_asset_dir=None,
                                     kitchen_main_usd_relative=kitchen_main_usd_relative,
                                     g1_usd=g1_usd,
                                     policy_id=policy_id, steps=steps,
                                     render_noise_audit_plan=render_noise_audit_plan)
    manifest["bundle_zip"] = str(bundle_zip)
    job_dir = out_dir / "object_store_real_run"
    staged = stage_bundle(bundle_zip, job_dir, key_prefix=key_prefix)
    manifest["staging"] = {"status": staged["status"]}
    if staged["status"] != "completed":
        manifest["blockers"].append("staging_failed")
        manifest["staging"]["stderr_tail"] = staged.get("stderr_tail")
        return manifest
    warm_broker_base_url = ""
    warm_broker_token = ""
    if serve:
        from blueprint_pipeline.wam_provider_object_store import presign_warm_inbox_channel
        inbox = presign_warm_inbox_channel(job_dir, key_prefix=key_prefix)
        manifest["warm_inbox"] = {
            "status": inbox.get("status"),
            "blockers": inbox.get("blockers"),
            "transport": inbox.get("transport"),
            "single_object_transport_enabled": inbox.get(
                "single_object_transport_enabled"
            ),
        }
        if inbox.get("status") != "completed":
            manifest["blockers"].append("durable_warm_render_broker_not_configured")
            return manifest
        warm_broker_base_url = Path(inbox["broker_base_url_file"]).read_text().strip()
        warm_broker_token = Path(inbox["broker_token_file"]).read_text().strip()
    runner_timeout_seconds = 0
    if not serve and max_seconds and max_seconds > 60:
        scenario_budget = max(1, len(scenarios)) * max(1, int(per_scenario_seconds))
        runner_timeout_seconds = max(
            300,
            min(int(max_seconds) - 60, scenario_budget + 420),
        )
    spec = build_launch_spec(job_dir, image=selected_image, policy_id=policy_id,
                             steps=steps, kitchen_url=kitchen_url, width=width, height=height,
                             fps=fps,
                             container_disk_gb=container_disk_gb, volume_gb=volume_gb,
                             serve=serve,
                             warm_broker_base_url=warm_broker_base_url,
                             warm_broker_token=warm_broker_token,
                             serve_idle_timeout_s=serve_idle_timeout_s, serve_max_jobs=serve_max_jobs,
                             warmup=warmup, per_scenario_seconds=per_scenario_seconds,
                             no_collision_probe=no_collision_probe, focus_radius=focus_radius,
                             keep_objects=keep_objects, settle_seconds=settle_seconds,
                             cheap_collision=cheap_collision, articulated=articulated,
                             physics_articulation_drive=physics_articulation_drive,
                             dynamic_standing_contact_steps=dynamic_standing_contact_steps,
                             manipulation_cam=manipulation_cam, manipulation_look_at=manipulation_look_at,
                             render_subframes=render_subframes, manipulation_reach=manipulation_reach,
                             manipulation_reach_arm=manipulation_reach_arm,
                             dynamic_episode_termination=dynamic_episode_termination,
                             episode_max_steps=episode_max_steps,
                             dynamic_episode_check_every=dynamic_episode_check_every,
                             capture_every=capture_every,
                             fill_light_intensity=fill_light_intensity,
                             neutral_environment=neutral_environment,
                             robot_review_material_override=robot_review_material_override,
                             robot_review_material_mode=robot_review_material_mode,
                             kinematic_arm_pose=kinematic_arm_pose,
                             collision_approximation=collision_approximation, verify_cam=verify_cam,
                             manipulation_stand=manipulation_stand,
                             placement_topdown_capture=placement_topdown_capture,
                             render_noise_audit=render_noise_audit,
                             audit_high_spp=audit_high_spp,
                             audit_warmup_frames=audit_warmup_frames,
                             audit_boost_light_intensity=audit_boost_light_intensity,
                             vast_max_hourly_rate_usd=vast_max_hourly_rate_usd,
                             gemini_api_key=_gemini_api_key_from_env(),
                             groot_policy_command=effective_groot_policy_command,
                             groot_policy_command_timeout_seconds=(
                                 effective_groot_policy_command_timeout_seconds
                             ),
                             image_startup_canary=image_startup_canary,
                             runner_timeout_seconds=runner_timeout_seconds)
    request_body = prov.build_request(spec, job_dir)
    manifest["launch_request_shape"] = {"provider": prov.name, "image": spec.image,
                                        "policy_id": policy_id, "steps": steps,
                                        "width": int(width), "height": int(height),
                                        "fps": int(fps),
                                        "runner_timeout_seconds": int(runner_timeout_seconds),
                                        "post_marker_progress_timeout": int(
                                            post_marker_progress_timeout or 0
                                        ),
                                        "container_disk_gb": int(container_disk_gb),
                                        "volume_gb": int(volume_gb),
                                        "vast_max_hourly_rate_usd": spec.max_hourly_rate_usd,
                                        "physics_articulation_drive": bool(
                                            physics_articulation_drive
                                            or dynamic_standing_contact_steps > 0
                                        ),
                                        "dynamic_standing_contact_steps": int(
                                            dynamic_standing_contact_steps
                                        ),
                                        "dynamic_episode_termination": bool(
                                            dynamic_episode_termination
                                        ),
                                        "episode_max_steps": int(episode_max_steps or 0),
                                        "dynamic_episode_check_every": int(
                                            dynamic_episode_check_every or 1
                                        ),
                                        "capture_every": int(capture_every or 1),
                                        "placement_topdown_capture": bool(
                                            placement_topdown_capture
                                        ),
                                        "robot_review_material_override": bool(
                                            robot_review_material_override
                                        ),
                                        "robot_review_material_mode": (
                                            str(robot_review_material_mode) or None
                                        ),
                                        "groot_policy_command_configured": bool(
                                            effective_groot_policy_command
                                        ),
                                        "image_startup_canary": bool(image_startup_canary)}
    if not allow_paid:
        manifest["status"] = "prepared"
        manifest["note"] = f"bundled + staged + launchable on {prov.name}; re-run with allow_paid=True to spend GPU"
        return manifest
    if multi_provider_race:
        available_pairs = [(p, p.available()) for p in providers]
        runnable_providers = [p for p, avail in available_pairs if avail.get("available")]
        manifest["provider_available"] = [avail for _p, avail in available_pairs]
        if not runnable_providers:
            manifest["blockers"].append("provider_credentials_missing")
            return manifest
    else:
        avail = prov.available()
        if not avail.get("available"):
            manifest["blockers"].append(avail.get("reason") or "provider_credentials_missing")
            return manifest
    prelaunch_contender_count = (
        len(runnable_providers)
        if multi_provider_race
        else race_contender_count
        if single_provider_cold_race
        else 1
    )
    prelaunch_spend_guard = _isaac_g1_prelaunch_spend_guard(
        allow_paid=allow_paid,
        provider_name=",".join([p.name for p in providers])
        if multi_provider_race
        else prov.name,
        max_spend_usd=max_spend_usd,
        max_seconds=max_seconds,
        max_hourly_rate_usd=spec.max_hourly_rate_usd,
        contender_count=prelaunch_contender_count,
    )
    manifest["prelaunch_spend_guard"] = prelaunch_spend_guard
    if prelaunch_spend_guard.get("can_launch") is not True:
        manifest["blockers"].append("isaac_g1_prelaunch_spend_guard_not_passed")
        manifest["blockers"].extend(prelaunch_spend_guard.get("blockers") or [])
        manifest["blockers"] = sorted(set(manifest["blockers"]))
        return manifest
    # cold ~10-15GB Isaac image pulls on congested nodes routinely exceed 150s before the container
    # starts bash; give the early marker a generous window (+ an extra attempt) so a slow pull is not
    # mistaken for a dead pod (which caused all-dud batches on both providers).
    # The worker image is ~10.7 GB (one 10.6 GB layer); a slow node needs >7 min just to pull it
    # before its container can write the bootstrap marker. Default the boot window to 900s so we
    # stop reaping nodes mid-pull (the 420s default lost every <~200 Mbps node). Configurable.
    collect_job_dir = job_dir
    collect_provider = prov
    if multi_provider_race or single_provider_cold_race:
        if multi_provider_race:
            race_contender_providers = list(runnable_providers)
        else:
            race_contender_providers = [prov] + [
                _ColdCreateContender(prov) for _ in range(race_contender_count - 1)
            ]
        race_stage_records: list[dict] = []

        def _race_request(provider_obj, contender_job_dir):
            contender_job_dir = Path(contender_job_dir)
            staged_contender = stage_bundle(
                bundle_zip,
                contender_job_dir,
                key_prefix=f"{key_prefix}/race/{provider_obj.name}",
            )
            race_stage_records.append({
                "provider": provider_obj.name,
                "job_dir": str(contender_job_dir),
                "status": staged_contender.get("status"),
            })
            if staged_contender.get("status") != "completed":
                raise RuntimeError(
                    f"race_staging_failed:{provider_obj.name}:{staged_contender.get('status')}"
                )
            launch_session_id = uuid.uuid4().hex
            (contender_job_dir / "launch_session_nonce.txt").write_text(
                launch_session_id,
                encoding="utf-8",
            )
            body = provider_obj.build_request(spec, contender_job_dir)
            request_for_launch = _request_with_launch_session_nonce(body, launch_session_id)
            request_for_launch["prelaunch_spend_guard"] = prelaunch_spend_guard
            return request_for_launch

        def _race_marker_check(provider_obj, launch_result):
            contender_job_dir = Path(str(launch_result.get("job_dir") or job_dir))
            nonce_file = contender_job_dir / "launch_session_nonce.txt"
            if not nonce_file.is_file():
                return False
            launch_session_id = nonce_file.read_text(encoding="utf-8").strip()
            if not launch_session_id:
                return False
            return boot_marker_present(
                contender_job_dir,
                expected_launch_session_id=launch_session_id,
                urlopen=urllib.request.urlopen,
            )

        race: dict = {}
        race_rounds = 0
        for _race_round in range(max(1, int(max_attempts))):
            race_rounds += 1
            race = race_launch(
                race_contender_providers,
                _race_request,
                marker_check=_race_marker_check,
                marker_timeout=marker_timeout,
                job_dir=job_dir,
                cold=cold,
                poll_interval=max(1.0, min(15.0, float(marker_timeout))),
                launch_kwargs=lambda _p: {"allow_cold_fallback": not warm_only},
                prelaunch_guard=prelaunch_spend_guard,
                pending_teardown_lane=ISAAC_G1_KITCHEN_PARITY_LANE,
                pending_teardown_max_age_seconds=_paid_launch_pending_teardown_max_age(
                    marker_timeout=int(marker_timeout),
                    startup_no_runtime_timeout=int(startup_no_runtime_timeout),
                    max_attempts=int(max_attempts),
                ),
            )
            if race.get("status") == "launched":
                break
        manifest["race_rounds"] = race_rounds
        manifest["race_staging"] = race_stage_records
        manifest["launch"] = {k: v for k, v in race.items() if k != "winner_provider"}
        if race.get("status") == "launched":
            launch = {
                "status": "launched",
                "instance_id": race.get("instance_id"),
                "mode": race.get("mode"),
                "winner_launch": race.get("winner_launch"),
                "pending_teardown_record": race.get("pending_teardown_record"),
            }
            collect_provider = race["winner_provider"]
            collect_job_dir = Path(
                str((race.get("winner_launch") or {}).get("job_dir") or job_dir)
            )
        else:
            launch = race
    else:
        launch = launch_with_marker_retry(prov, job_dir, request_body,
                                          marker_timeout=marker_timeout, max_attempts=max_attempts,
                                          cold=cold,
                                          allow_cold_fallback=not warm_only,
                                          startup_no_runtime_timeout=startup_no_runtime_timeout,
                                          prelaunch_guard=prelaunch_spend_guard)
        manifest["launch"] = launch
    if launch.get("status") != "launched":
        for blocker in launch.get("blockers") or []:
            if blocker not in manifest["blockers"]:
                manifest["blockers"].append(blocker)
        failed_launch_blocker = (
            "launch_failed_provider_capacity_unavailable"
            if any(
                blocker in PROVIDER_CAPACITY_UNAVAILABLE_BLOCKERS
                or blocker == "provider_capacity_unavailable_before_instance_created"
                for blocker in (launch.get("blockers") or [])
            )
            else "launch_failed_all_attempts_flaky"
        )
        if failed_launch_blocker not in manifest["blockers"]:
            manifest["blockers"].append(failed_launch_blocker)
        return manifest
    pending_teardown_record = _string(launch.get("pending_teardown_record"))
    if pending_teardown_record:
        manifest["pending_teardown_record"] = pending_teardown_record
    if serve:
        # Warm pod: leave it RUNNING. Wait for the serve-ready marker, then hand the caller the inbox
        # PUT + output GET urls for its WarmPoolClient. NO watch_and_collect / teardown here — the warm
        # pod must stay alive across the caller's live job submissions; the caller tears it down.
        ready = _await_warm_serve_ready(job_dir, instance_id=launch["instance_id"],
                                        timeout_s=serve_ready_timeout)
        manifest["warm_serve"] = {
            "instance_id": launch["instance_id"],
            "ready": bool(ready.get("ready")),
            "broker_base_url_file": str(job_dir / "warm_broker_base_url.txt"),
            "broker_token_file": str(job_dir / "warm_broker_token.txt"),
            "transport": "durable_warm_render_broker",
            "single_object_transport_enabled": False,
            "ready_detail": ready,
        }
        if pending_teardown_record:
            manifest["warm_serve"]["pending_teardown_record"] = pending_teardown_record
        if ready.get("ready"):
            manifest["status"] = "serving"
        else:
            try:
                teardown = prov.terminate(launch["instance_id"])
            except Exception as exc:  # noqa: BLE001 - preserve cleanup failure in the manifest
                teardown = {
                    "status": "error",
                    "operation": "terminate_not_ready_warm_serve_pod",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            manifest["warm_serve"]["not_ready_teardown"] = teardown
            if pending_teardown_record:
                proof = _teardown_proof_from_attempt(
                    provider=prov,
                    instance_id=launch["instance_id"],
                    teardown=teardown if isinstance(teardown, Mapping) else {},
                    action="terminate",
                )
                manifest["warm_serve"]["not_ready_teardown_proof"] = proof
                closure = close_pending_teardown(pending_teardown_record, proof)
                manifest["warm_serve"]["pending_teardown_status"] = closure.get(
                    "status"
                )
            manifest["blockers"].append("warm_serve_not_ready")
        return manifest
    render_out = out_dir / "render_output"
    result = watch_and_collect(collect_job_dir, render_out, launch["instance_id"], provider=collect_provider,
                               max_seconds=max_seconds, preserve_instance=True,
                               progress_timeout_seconds=post_marker_progress_timeout)
    manifest["render"] = {
        "status": result.get("status"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "teardown": result.get("teardown"),
        "runner_result_source": result.get("runner_result_source"),
        "last_bootstrap": result.get("last_bootstrap"),
        "runner_console_tail": result.get("runner_console_tail"),
        "runner_timeout_observed": result.get("runner_timeout_observed"),
        "post_marker_progress_timeout_observed": result.get(
            "post_marker_progress_timeout_observed"
        ),
        "post_marker_progress_timeout": result.get("post_marker_progress_timeout"),
    }
    if pending_teardown_record:
        proof = _teardown_proof_from_watch_result(
            provider_name=getattr(collect_provider, "name", "unknown"),
            instance_id=launch["instance_id"],
            watch=result,
        )
        closure = close_pending_teardown(pending_teardown_record, proof)
        manifest["render"]["teardown_proof"] = proof
        manifest["render"]["pending_teardown_record"] = pending_teardown_record
        manifest["render"]["pending_teardown_status"] = closure.get("status")
    if render_noise_audit:
        audit_worker_result: dict = {}
        try:
            audit_worker_result = json.loads(
                (render_out / "render_noise_audit_result.json").read_text()
            )
        except Exception:  # noqa: BLE001
            pass
        manifest["render_noise_audit_worker_result"] = audit_worker_result
        analysis = None
        try:
            from .g1_render_noise_audit import AUDIT_MANIFEST_NAME, analyze_render_noise_audit_run
            analysis = analyze_render_noise_audit_run(render_out)
        except Exception as exc:  # noqa: BLE001
            manifest["blockers"].append("render_noise_audit_local_analysis_failed")
            manifest["render_noise_audit_analysis_error"] = repr(exc)
        if analysis is not None:
            manifest["render_noise_audit"] = {
                "status": analysis.get("status"),
                "primary_diagnosis": (analysis.get("interpretation") or {}).get("primary_diagnosis"),
                "variants_executed": analysis.get("variants_executed"),
                "audit_manifest": str(Path(str(analysis.get("audit_dir") or render_out)) / AUDIT_MANIFEST_NAME),
                "contact_sheet": (analysis.get("contact_sheet") or {}).get("path"),
                "analysis_blockers": analysis.get("blockers"),
            }
        worker_completed = str(audit_worker_result.get("status") or "").lower() == "completed"
        analysis_completed = bool(analysis) and analysis.get("status") == "completed"
        if worker_completed and analysis_completed:
            manifest["status"] = "completed"
        else:
            for blocker in audit_worker_result.get("blockers") or []:
                if blocker not in manifest["blockers"]:
                    manifest["blockers"].append(blocker)
            if "render_noise_audit_blocked" not in manifest["blockers"]:
                manifest["blockers"].append("render_noise_audit_blocked")
        return manifest
    parity_result = result.get("runner_result") or {}
    try:
        parity_result = json.loads((render_out / "isaac_g1_kitchen_parity_result.json").read_text())
    except Exception:  # noqa: BLE001
        pass
    manifest["parity_result"] = parity_result
    parity_status = str(parity_result.get("status") or "").strip().lower()
    runner_timeout_observed = bool(result.get("runner_timeout_observed"))
    runner_completed = (
        not bool(result.get("timed_out_without_runner_done"))
        and not runner_timeout_observed
    )
    manifest["runner_completed"] = runner_completed
    manifest["runner_timeout_observed"] = runner_timeout_observed
    manifest["parity_result_status"] = parity_status or None
    if parity_result:
        manifest["local_mp4_repair"] = _repair_collected_review_mp4s(
            render_out_dir=render_out,
            result=parity_result,
            fps=int(requested_render_settings.get("fps") or fps),
            optional_videos=(
                ("placement_topdown",) if not placement_topdown_capture else ()
            ),
            expected_frame_count=int(
                requested_render_settings.get("expected_frame_count_per_scenario") or 0
            )
            or None,
        )
    if parity_status == "completed" and image_startup_canary:
        manifest["image_startup_canary_result"] = parity_result
        manifest["status"] = "completed"
    elif parity_status == "completed":
        manifest["harness"] = build_harness_package(
            result=parity_result,
            render_out_dir=render_out,
            out_dir=out_dir,
            requested_render_settings=requested_render_settings,
        )
        manifest["status"] = "completed"
    elif runner_completed and parity_result:
        manifest["status"] = "blocked"
        for blocker in parity_result.get("blockers") or []:
            if blocker not in manifest["blockers"]:
                manifest["blockers"].append(blocker)
        if "isaac_parity_result_blocked" not in manifest["blockers"]:
            manifest["blockers"].append("isaac_parity_result_blocked")
    elif runner_timeout_observed:
        manifest["blockers"].append("isaac_runner_timeout")
    elif runner_completed:
        manifest["blockers"].append("isaac_runner_completed_without_result")
    else:
        manifest["blockers"].append("isaac_runtime_did_not_complete")
    return manifest


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Isaac G1 kitchen MuJoCo-parity eval job (GPU)")
    ap.add_argument("--scenarios", required=True, help="JSON file: list of {scenario_id, spawn_position_xyz, target_position_xyz}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kitchen-asset-dir", default=None, help="local Collected_KitchenRoom parent dir to ship in the bundle")
    ap.add_argument("--kitchen-url", default=None,
                    help="previously staged kitchen asset zip signed URL; skips the large asset upload")
    ap.add_argument("--g1-usd", default=DEFAULT_G1_USD_RELATIVE)
    ap.add_argument("--policy", default="blueprint_default_walk_to_target_smoke_policy",
                    choices=["blueprint_default_walk_to_target_smoke_policy", "groot_sonic"])
    ap.add_argument(
        "--groot-policy-command",
        default=(
            os.getenv(ISAAC_G1_GROOT_POLICY_COMMAND_ENV, "")
            or os.getenv(UNITREE_GROOT_POLICY_COMMAND_ENV, "")
        ),
        help=(
            "command the Isaac worker runs for GR00T/SONIC policy actions; required "
            "for --policy groot_sonic to pass paid preflight"
        ),
    )
    ap.add_argument(
        "--groot-policy-command-timeout-seconds",
        type=float,
        default=float(os.getenv(ISAAC_G1_GROOT_POLICY_COMMAND_TIMEOUT_ENV, "120") or 120),
        help="timeout for each GR00T/SONIC policy command call inside the Isaac worker",
    )
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument(
        "--no-dynamic-episode-termination",
        action="store_true",
        help=(
            "disable task-contract dynamic stop/extend behavior for manipulation review jobs; "
            "the default is enabled but inert for non-manipulation contracts"
        ),
    )
    ap.add_argument(
        "--episode-max-steps",
        type=int,
        default=0,
        help="max worker steps when dynamic episode termination is active; 0 uses the runner default",
    )
    ap.add_argument("--dynamic-episode-check-every", type=int, default=1)
    ap.add_argument("--capture-every", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--provider", default=DEFAULT_ISAAC_REVIEW_PROVIDER,
                    help=(
                        "provider name. DigitalOcean is the paid default for this high-reliability "
                        "Isaac review lane; pass runpod explicitly for cheaper compatibility runs. "
                        f"Vast is disabled for paid Isaac review renders unless "
                        f"{ALLOW_UNSTABLE_VAST_ISAAC_RENDER_ENV}=1"
                    ))
    ap.add_argument("--allow-paid", action="store_true")
    ap.add_argument(
        "--allow-dirty-paid-launch",
        action="store_true",
        help=(
            "override the dirty-worktree paid-launch block. Use only when the manifest's "
            "git_evidence is an intentional evidence boundary."
        ),
    )
    ap.add_argument("--cold", action="store_true")
    ap.add_argument(
        "--warm-candidate",
        action="append",
        default=[],
        help=(
            "RunPod stopped pod id to try as a warm restart before cold create. "
            "May be repeated; can also be supplied via BLUEPRINT_RUNPOD_WARM_CANDIDATES."
        ),
    )
    ap.add_argument(
        "--warm-only",
        action="store_true",
        help="try warm candidates only; block instead of creating a cold pod if warm restart fails",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help=(
            "launch one persistent warm Isaac render worker, wait for warm_serve_ready, "
            "and leave the pod running for WarmPoolClient job submissions"
        ),
    )
    ap.add_argument(
        "--serve-idle-timeout",
        type=float,
        default=1800.0,
        help="seconds the warm serve worker may sit idle before exiting",
    )
    ap.add_argument(
        "--serve-max-jobs",
        type=int,
        default=None,
        help="optional maximum number of warm jobs to serve before exiting",
    )
    ap.add_argument(
        "--serve-ready-timeout",
        type=int,
        default=1800,
        help="seconds to wait for warm_serve_ready.json before marking the serve launch blocked",
    )
    ap.add_argument(
        "--image-startup-canary",
        action="store_true",
        help=(
            "run only a provider image startup/upload canary. This proves user command "
            "execution in the selected image, not Isaac rendering, WAM quality, or task success."
        ),
    )
    ap.add_argument("--image", default=None)
    ap.add_argument("--max-seconds", type=int, default=1500)
    ap.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help=(
            "required positive spend cap for --allow-paid launches; may also be supplied "
            f"via {ISAAC_G1_MAX_SPEND_USD_ENV}"
        ),
    )
    ap.add_argument("--container-disk-gb", type=int, default=140,
                    help="RunPod container disk size. Must be >= existing pod size for warm update.")
    ap.add_argument("--volume-gb", type=int, default=80,
                    help="RunPod network volume size. Must be >= existing pod size for warm update.")
    ap.add_argument("--marker-timeout", type=int, default=900,
                    help="seconds to wait for a pod's boot marker before reaping it as a dud "
                         "(must exceed the worker image pull time on a slow node)")
    ap.add_argument(
        "--post-marker-progress-timeout",
        type=int,
        default=360,
        help=(
            "seconds to allow an early bootstrap phase to repeat without progress after the boot "
            "marker is visible before terminating the paid pod"
        ),
    )
    ap.add_argument(
        "--startup-no-runtime-timeout",
        type=int,
        default=DEFAULT_STARTUP_NO_RUNTIME_TIMEOUT_SECONDS,
        help=(
            "earlier RunPod dud guard: if provider inspection still shows no runtime/public IP "
            "and no bootstrap marker after this many seconds, terminate the pod before marker-timeout; "
            "0 disables"
        ),
    )
    ap.add_argument(
        "--cold-race-contenders",
        type=int,
        default=None,
        help=(
            "race N simultaneous cold creates on a single provider and keep the first boot marker "
            f"(default {DEFAULT_COLD_RACE_CONTENDERS}; 1 disables)"
        ),
    )
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="cold-launch attempts before giving up")
    ap.add_argument("--vast-max-hourly-rate", type=float, default=None,
                    help="maximum Vast offer hourly rate; defaults to env or $5/hr")
    ap.add_argument("--articulated", action="store_true")
    ap.add_argument("--physics-articulation-drive", action="store_true")
    ap.add_argument("--dynamic-standing-contact-steps", type=int, default=0)
    ap.add_argument("--cheap-collision", action="store_true")
    ap.add_argument("--settle-seconds", type=int, default=0)
    ap.add_argument("--focus-radius", type=float, default=0.0,
                    help="task-aware scene pruning radius in meters (0=full scene)")
    ap.add_argument("--keep-objects", default="",
                    help="comma substrings of object names to always keep during focus pruning")
    ap.add_argument("--per-scenario-seconds", type=int, default=420,
                    help="wall-clock cap per scenario inside the Isaac runner")
    ap.add_argument("--manipulation-cam", action="store_true",
                    help="egocentric at-sink POV (manipulation framing) instead of the navigation chase cam")
    ap.add_argument("--manipulation-look-at", default="",
                    help="fixed world 'x,y,z' the manipulation cam aims at (e.g. the faucet)")
    ap.add_argument("--render-subframes", type=int, default=0,
                    help="RTX subframes accumulated per captured frame to denoise grain (e.g. 16)")
    ap.add_argument("--manipulation-reach", action="store_true",
                    help="animate the arm reaching the faucet so the skeleton-video encodes the task")
    ap.add_argument("--manipulation-reach-arm", default="auto", choices=["auto", "right", "left", "both"])
    ap.add_argument("--fill-light-intensity", type=float, default=0.0,
                    help="sphere fill light over the faucet workspace to lift the dark basin (0=off)")
    ap.add_argument("--neutral-environment", action="store_true",
                    help="replace the kitchen's outdoor-HDRI dome with a neutral environment "
                         "(no cityscape through the windows + lifts shadows)")
    ap.add_argument("--robot-review-material-override", action="store_true",
                    help="bind neutral matte material over authored G1 materials/textures for a "
                         "clearer untextured manipulation seed image")
    ap.add_argument(
        "--robot-review-material-mode",
        default="",
        choices=["", "neutral_matte", "non_white_matte"],
        help=(
            "material mode forwarded to the Isaac worker when robot review material "
            "override is active"
        ),
    )
    ap.add_argument("--kinematic-arm-pose", action="store_true",
                    help="pose the rendered arm reaching the workspace (pure-USD, crash-safe)")
    ap.add_argument("--collision-approximation", default="",
                    choices=["", "boundingCube", "convexHull", "convexDecomposition"],
                    help="mesh collision shape (convexHull lets the robot stand centered + close)")
    ap.add_argument("--verify-cam", action="store_true",
                    help="render a 3rd-person verify_*.png that frames the whole robot at the workspace")
    ap.add_argument("--manipulation-stand", action="store_true",
                    help="place the robot AT the target facing the look-at (task start pose, no navigation)")
    ap.add_argument("--no-placement-topdown-capture", action="store_true")
    ap.add_argument("--render-noise-audit", action="store_true",
                    help="run the textured-robot render-noise audit variant matrix (A-G) instead of "
                         "the scenario eval: one raw PNG per material/render variant + material/"
                         "render/camera manifests + local gates/interpretation on collect")
    ap.add_argument("--audit-high-spp", type=int, default=0,
                    help="samples per pixel for the audit's high-budget variants (default 384)")
    ap.add_argument("--audit-warmup-frames", type=int, default=0,
                    help="shader warmup render steps before the first audit variant (default 8)")
    ap.add_argument("--audit-boost-light-intensity", type=float, default=0.0,
                    help="workspace boost light intensity for the audit's bright-lighting variant")
    args = ap.parse_args(argv)
    scenarios = json.loads(Path(args.scenarios).read_text())
    if isinstance(scenarios, dict):
        scenarios = scenarios.get("scenarios", [])
    m = run_isaac_g1_kitchen_parity_job(
        scenarios=scenarios, out_dir=args.out_dir, kitchen_asset_dir=args.kitchen_asset_dir,
        kitchen_url=args.kitchen_url,
        g1_usd=args.g1_usd, policy_id=args.policy, steps=args.steps, provider=args.provider,
        width=args.width, height=args.height, fps=args.fps,
        groot_policy_command=args.groot_policy_command,
        groot_policy_command_timeout_seconds=args.groot_policy_command_timeout_seconds,
        allow_paid=args.allow_paid, allow_dirty_paid_launch=args.allow_dirty_paid_launch,
        cold=args.cold, image=args.image, max_seconds=args.max_seconds,
        max_spend_usd=args.max_spend_usd,
        container_disk_gb=args.container_disk_gb, volume_gb=args.volume_gb,
        warm_candidates=tuple(args.warm_candidate or ()),
        warm_only=args.warm_only,
        serve=args.serve,
        serve_idle_timeout_s=args.serve_idle_timeout,
        serve_max_jobs=args.serve_max_jobs,
        serve_ready_timeout=args.serve_ready_timeout,
        image_startup_canary=args.image_startup_canary,
        marker_timeout=args.marker_timeout, max_attempts=args.max_attempts,
        post_marker_progress_timeout=args.post_marker_progress_timeout,
        startup_no_runtime_timeout=args.startup_no_runtime_timeout,
        cold_race_contenders=args.cold_race_contenders,
        vast_max_hourly_rate_usd=args.vast_max_hourly_rate,
        warmup=args.warmup,
        articulated=args.articulated, cheap_collision=args.cheap_collision,
        physics_articulation_drive=args.physics_articulation_drive,
        dynamic_standing_contact_steps=args.dynamic_standing_contact_steps,
        settle_seconds=args.settle_seconds, focus_radius=args.focus_radius,
        keep_objects=args.keep_objects, per_scenario_seconds=args.per_scenario_seconds,
        manipulation_cam=args.manipulation_cam,
        manipulation_look_at=args.manipulation_look_at, render_subframes=args.render_subframes,
        manipulation_reach=args.manipulation_reach, manipulation_reach_arm=args.manipulation_reach_arm,
        dynamic_episode_termination=not args.no_dynamic_episode_termination,
        episode_max_steps=args.episode_max_steps,
        dynamic_episode_check_every=args.dynamic_episode_check_every,
        capture_every=args.capture_every,
        fill_light_intensity=args.fill_light_intensity,
        neutral_environment=args.neutral_environment,
        robot_review_material_override=args.robot_review_material_override,
        robot_review_material_mode=args.robot_review_material_mode,
        kinematic_arm_pose=args.kinematic_arm_pose,
        collision_approximation=args.collision_approximation, verify_cam=args.verify_cam,
        manipulation_stand=args.manipulation_stand,
        placement_topdown_capture=not args.no_placement_topdown_capture,
        render_noise_audit=args.render_noise_audit,
        audit_high_spp=args.audit_high_spp,
        audit_warmup_frames=args.audit_warmup_frames,
        audit_boost_light_intensity=args.audit_boost_light_intensity)
    _write_job_manifest(args.out_dir, m)
    print(json.dumps(m, indent=2, default=str))
    return 0 if m.get("status") in ("completed", "prepared", "serving") else 1


if __name__ == "__main__":
    raise SystemExit(main())
