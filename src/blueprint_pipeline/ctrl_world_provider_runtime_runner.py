"""Standalone GPU runtime for the frozen Ctrl-World replay canary bundle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL_NAME = "Ctrl-World"
RESULT_NAME = "wam_runtime_result.json"
ACTION_ROLLOUT_MARKER = "action_conditioned_video_rollout_generated"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    return {
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "argv_redacted": ["<python>" if item == sys.executable else item for item in argv],
    }


def _phase(name: str, **details: Any) -> None:
    print(json.dumps({"phase": name, **details}, sort_keys=True), flush=True)


def _validate_packaged_inputs(*, bundle_dir: Path, manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    source_root = bundle_dir / "provider_runtime" / "ctrl_world_source"
    for row in manifest.get("source_files", []):
        if not isinstance(row, dict):
            blockers.append("ctrl_world_source_file_manifest_invalid")
            continue
        relative_path = str(row.get("relative_path") or "")
        path = source_root / relative_path
        if not relative_path or not path.is_file():
            blockers.append(f"ctrl_world_source_file_missing:{relative_path}")
            continue
        expected_size = row.get("size_bytes")
        if type(expected_size) is not int or expected_size < 0:
            blockers.append(f"ctrl_world_source_file_size_invalid:{relative_path}")
        elif path.stat().st_size != expected_size:
            blockers.append(f"ctrl_world_source_file_size_mismatch:{relative_path}")
        if _sha256_file(path) != str(row.get("sha256") or ""):
            blockers.append(f"ctrl_world_source_file_hash_mismatch:{relative_path}")
    return blockers


def _ensure_dependencies(manifest: dict[str, Any]) -> dict[str, Any]:
    required = manifest.get("python_dependencies")
    if not isinstance(required, list) or not required:
        return {"status": "blocked", "blockers": ["ctrl_world_dependency_freeze_missing"]}
    detail = _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *[str(item) for item in required],
        ],
        timeout=1800,
    )
    blockers: list[str] = []
    if detail["returncode"] != 0:
        blockers.append("ctrl_world_dependency_install_failed")
    observed: dict[str, str] = {}
    for item in required:
        package, _, expected = str(item).partition("==")
        if not package or not expected:
            blockers.append("ctrl_world_dependency_not_exactly_pinned")
            continue
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            blockers.append(f"ctrl_world_dependency_missing:{package}")
            continue
        if observed[package] != expected:
            blockers.append(f"ctrl_world_dependency_version_mismatch:{package}")
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = "missing"
    if torch_version != str(manifest.get("torch_version") or ""):
        blockers.append("ctrl_world_torch_version_mismatch")
    return {
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "install": detail,
        "observed_versions": observed,
        "torch_version": torch_version,
    }


def _download_models(
    *, work_dir: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    model_roots: dict[str, Path] = {}
    blockers: list[str] = []
    observed_blobs: list[dict[str, Any]] = []
    for model in manifest.get("models", []):
        if not isinstance(model, dict):
            blockers.append("ctrl_world_model_freeze_invalid")
            continue
        name = str(model.get("name") or "")
        repo_id = str(model.get("repository") or "")
        revision = str(model.get("revision") or "")
        target = work_dir / "models" / name
        allow_patterns = model.get("allow_patterns")
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=target,
                token=token,
                allow_patterns=(
                    [str(item) for item in allow_patterns]
                    if isinstance(allow_patterns, list)
                    else None
                ),
            )
        except Exception as exc:
            blockers.append(f"ctrl_world_model_download_failed:{name}:{type(exc).__name__}")
            continue
        model_roots[name] = target
        for blob in model.get("required_blobs", []):
            if not isinstance(blob, dict):
                blockers.append(f"ctrl_world_model_blob_freeze_invalid:{name}")
                continue
            relative_path = str(blob.get("relative_path") or "")
            path = target / relative_path
            observed = {
                "model": name,
                "relative_path": relative_path,
                "present": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            observed_blobs.append(observed)
            if not path.is_file():
                blockers.append(f"ctrl_world_model_blob_missing:{name}:{relative_path}")
                continue
            if observed["size_bytes"] != int(blob.get("size_bytes") or -1):
                blockers.append(f"ctrl_world_model_blob_size_mismatch:{name}:{relative_path}")
            if observed["sha256"] != str(blob.get("sha256") or ""):
                blockers.append(f"ctrl_world_model_blob_hash_mismatch:{name}:{relative_path}")
    return model_roots, {
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "observed_blobs": observed_blobs,
        "raw_hf_token_recorded": False,
        "hf_token_present": bool(token),
    }


def _cuda_probe() -> dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({'available':bool(torch.cuda.is_available()),"
        "'count':torch.cuda.device_count(),"
        "'memory_allocated':torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,"
        "'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    detail = _run([sys.executable, "-c", code], timeout=120)
    payload: dict[str, Any] = {}
    if detail["returncode"] == 0:
        try:
            payload = json.loads(str(detail["stdout_tail"]).splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {}
    blockers = [] if payload.get("available") is True else ["ctrl_world_cuda_unavailable"]
    return {
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "payload": payload,
    }


def _execute_public_replay(
    *,
    source_root: Path,
    work_dir: Path,
    model_roots: dict[str, Path],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    settings = manifest.get("canary_settings") or {}
    save_dir = work_dir / "public_script_output"
    wrapper = """
import os
import runpy
import sys
import config

_original = config.wm_args.__post_init__
def _blueprint_canary_post_init(self):
    _original(self)
    self.interact_num = int(os.environ['BLUEPRINT_CTRL_WORLD_INTERACTIONS'])
    self.val_id = [os.environ['BLUEPRINT_CTRL_WORLD_TRAJECTORY_ID']]
    self.start_idx = [int(os.environ['BLUEPRINT_CTRL_WORLD_START_INDEX'])]
    self.instruction = ['']
    self.save_dir = os.environ['BLUEPRINT_CTRL_WORLD_SAVE_DIR']

config.wm_args.__post_init__ = _blueprint_canary_post_init
sys.argv = [
    'scripts/rollout_replay_traj.py',
    '--svd_model_path', os.environ['BLUEPRINT_CTRL_WORLD_SVD_PATH'],
    '--clip_model_path', os.environ['BLUEPRINT_CTRL_WORLD_CLIP_PATH'],
    '--ckpt_path', os.environ['BLUEPRINT_CTRL_WORLD_CHECKPOINT_PATH'],
    '--dataset_root_path', 'dataset_example',
    '--dataset_meta_info_path', 'dataset_meta_info',
    '--dataset_names', 'droid_subset',
    '--task_type', 'replay',
]
runpy.run_path('scripts/rollout_replay_traj.py', run_name='__main__')
"""
    env = os.environ.copy()
    env.update(
        {
            "BLUEPRINT_CTRL_WORLD_INTERACTIONS": str(settings.get("interaction_count")),
            "BLUEPRINT_CTRL_WORLD_TRAJECTORY_ID": str(settings.get("trajectory_id")),
            "BLUEPRINT_CTRL_WORLD_START_INDEX": str(settings.get("start_index")),
            "BLUEPRINT_CTRL_WORLD_SAVE_DIR": str(save_dir),
            "BLUEPRINT_CTRL_WORLD_SVD_PATH": str(model_roots["stable_video_diffusion"]),
            "BLUEPRINT_CTRL_WORLD_CLIP_PATH": str(model_roots["clip"]),
            "BLUEPRINT_CTRL_WORLD_CHECKPOINT_PATH": str(
                model_roots["ctrl_world"] / "checkpoint-10000.pt"
            ),
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "SWANLAB_MODE": "disabled",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    detail = _run(
        [sys.executable, "-c", wrapper],
        cwd=source_root,
        env=env,
        timeout=int(settings.get("timeout_seconds") or 3600),
    )
    videos = sorted(save_dir.rglob("*.mp4"))
    blockers: list[str] = []
    if detail["returncode"] != 0:
        blockers.append("ctrl_world_public_replay_failed")
    if len(videos) != 1:
        blockers.append("ctrl_world_public_replay_output_count_invalid")
    return {
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "subprocess": detail,
        "comparison_video": str(videos[0]) if len(videos) == 1 else None,
        "exact_public_script_executed_unchanged": True,
        "blueprint_wrapper_only_reduced_trajectory_and_interaction_count": True,
    }


def _extract_generated_only_views(*, comparison_video: Path, output_dir: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(comparison_video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 4.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    blockers: list[str] = []
    if width <= 0 or height <= 0 or width % 3 or height % 2:
        blockers.append("ctrl_world_public_comparison_geometry_invalid")
    generated_height = height // 2
    view_width = width // 3
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "ctrl_world_generated_three_view.mp4"
    view_paths = [output_dir / f"ctrl_world_generated_view_{index}.mp4" for index in range(3)]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    combined_writer = cv2.VideoWriter(str(combined_path), fourcc, fps, (width, generated_height))
    view_writers = [
        cv2.VideoWriter(str(path), fourcc, fps, (view_width, generated_height))
        for path in view_paths
    ]
    if not combined_writer.isOpened() or not all(writer.isOpened() for writer in view_writers):
        blockers.append("ctrl_world_generated_only_video_writer_failed")
    frame_count = 0
    try:
        while not blockers:
            ok, frame = capture.read()
            if not ok:
                break
            generated = frame[generated_height:height, :width]
            combined_writer.write(generated)
            for index, writer in enumerate(view_writers):
                writer.write(generated[:, index * view_width : (index + 1) * view_width])
            frame_count += 1
    finally:
        capture.release()
        combined_writer.release()
        for writer in view_writers:
            writer.release()
    if frame_count <= 0:
        blockers.append("ctrl_world_generated_only_video_empty")
    comparison_video.unlink(missing_ok=True)
    media_paths = [combined_path, *view_paths]
    media = [
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in media_paths
        if path.is_file()
    ]
    if len(media) != 4:
        blockers.append("ctrl_world_generated_only_media_count_invalid")
    return {
        "status": "completed" if not blockers else "blocked",
        "blockers": blockers,
        "source_geometry": {"width": width, "height": height, "fps": fps},
        "generated_geometry": {
            "three_view_width": width,
            "single_view_width": view_width,
            "height": generated_height,
        },
        "frame_count": frame_count,
        "media": media,
        "physical_comparison_pixels_removed": True,
        "public_comparison_video_deleted_after_redaction": not comparison_video.exists(),
    }


def main() -> int:
    started = time.monotonic()
    bundle_dir = Path(os.environ.get("BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR", Path.cwd())).resolve()
    output_dir = Path(
        os.environ.get("BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR", bundle_dir / "runtime_output")
    ).resolve()
    work_dir = Path(
        os.environ.get("BLUEPRINT_WAM_PROVIDER_WORK_DIR", bundle_dir / "runtime_work")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / RESULT_NAME
    runtime_manifest_path = bundle_dir / "provider_runtime" / "wam_provider_runtime_manifest.json"
    manifest: dict[str, Any] = {}
    blockers: list[str] = []
    dependency: dict[str, Any] = {"status": "not_run"}
    cuda: dict[str, Any] = {"status": "not_run"}
    downloads: dict[str, Any] = {"status": "not_run"}
    replay: dict[str, Any] = {"status": "not_run"}
    redaction: dict[str, Any] = {"status": "not_run"}
    try:
        manifest = _read_json(runtime_manifest_path)
        _phase("inputs_loaded", runtime=manifest.get("runtime"), model=MODEL_NAME)
        blockers.extend(_validate_packaged_inputs(bundle_dir=bundle_dir, manifest=manifest))
        if not blockers:
            dependency = _ensure_dependencies(manifest)
            blockers.extend(dependency.get("blockers") or [])
        if not blockers:
            cuda = _cuda_probe()
            blockers.extend(cuda.get("blockers") or [])
        model_roots: dict[str, Path] = {}
        if not blockers:
            model_roots, downloads = _download_models(work_dir=work_dir, manifest=manifest)
            blockers.extend(downloads.get("blockers") or [])
        if not blockers:
            replay = _execute_public_replay(
                source_root=bundle_dir / "provider_runtime" / "ctrl_world_source",
                work_dir=work_dir,
                model_roots=model_roots,
                manifest=manifest,
            )
            blockers.extend(replay.get("blockers") or [])
        comparison = Path(str(replay.get("comparison_video") or ""))
        if not blockers and comparison.is_file():
            redaction = _extract_generated_only_views(
                comparison_video=comparison, output_dir=output_dir
            )
            blockers.extend(redaction.get("blockers") or [])
    except Exception as exc:
        blockers.append(f"ctrl_world_runtime_exception:{type(exc).__name__}")
    model_executed = replay.get("status") == "completed"
    completed = not blockers and len(redaction.get("media") or []) == 4
    result = {
        "schema_version": "ctrl_world_replay_runtime_result.v1",
        "status": "completed" if completed else "blocked",
        "model_name": MODEL_NAME,
        "arm_id": "ctrl_world_public_replay_reduced_canary",
        ACTION_ROLLOUT_MARKER: model_executed,
        "ctrl_world_model_executed": model_executed,
        "learned_world_model_ran": model_executed,
        "public_replay_mode": True,
        "candidate_policy_requeried": False,
        "closed_loop_policy_evaluation": False,
        "recorded_action_trace_used": True,
        "future_physical_rgb_used_for_generation": False,
        "future_physical_rgb_decoded_by_public_script_for_comparison": replay.get("status")
        == "completed",
        "physical_comparison_pixels_returned_to_blueprint": False,
        "physical_outcome_labels_accessed": False,
        "hard_coded_runtime_success_field_used_as_ground_truth": False,
        "blockers": sorted(set(blockers)),
        "duration_seconds": round(time.monotonic() - started, 6),
        "dependency": dependency,
        "cuda": cuda,
        "model_downloads": downloads,
        "public_replay": replay,
        "generated_only_redaction": redaction,
        "media": redaction.get("media") or [],
        "truth_boundary": {
            "technical_replay_canary_only": True,
            "no_policy_ranking_credit": True,
            "no_closed_loop_credit": True,
            "no_physical_success_credit": True,
            "no_thesis_credit": True,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    _write_json(result_path, result)
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
