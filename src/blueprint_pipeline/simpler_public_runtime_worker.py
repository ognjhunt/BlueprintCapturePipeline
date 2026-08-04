"""Self-contained worker for the pinned SIMPLER ADP public-reference run.

This module is copied into a Vast provider bundle.  It acquires only manifest-
bound public inputs, verifies every downloaded model object, runs the two RT-1
checkpoints closed loop against the same three-condition SAPIEN interface, and
emits raw trace artifacts plus one immutable execution package.  It never reads
the external physical-reference outcomes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "simpler_closed_loop_execution.v2"
RUNTIME_LOCK_SCHEMA_VERSION = "simpler_runtime_lock.v1"
FRAME_MANIFEST_SCHEMA_VERSION = "adp_observation_frame_manifest.v1"
VISUAL_EVIDENCE_SCHEMA_VERSION = "adp_episode_visual_evidence.v1"
PHASE_LABEL = "retrospective_external_reference"
CLAIM_CEILING = "development_only"
GCS_BUCKET = "gdm-robotics-open-x-embodiment"
PYTHON_REQUIREMENTS = (
    "numpy==1.24.4",
    "scipy==1.12.0",
    "gymnasium==0.29.1",
    "sapien==2.2.2",
    "h5py==3.10.0",
    "PyYAML==6.0.1",
    "tqdm==4.66.2",
    "GitPython==3.1.42",
    "tabulate==0.9.0",
    "gdown==5.1.0",
    "transforms3d==0.4.1",
    "opencv-python-headless==4.9.0.80",
    "imageio==2.34.0",
    "imageio-ffmpeg==0.4.9",
    "trimesh==4.2.0",
    "rtree==1.2.0",
    "ruckig==0.12.2",
    "tensorflow==2.15.1",
    "tensorflow-hub==0.16.0",
    "tf-keras==2.15.0",
    "tf-agents==0.19.0",
    "tensorflow-probability==0.23.0",
    "matplotlib==3.8.3",
    "mediapy==1.2.0",
    "Pillow==10.2.0",
)


def _phase(name: str, status: str = "running") -> None:
    print(f"BLUEPRINT_WAM_RUNTIME_PHASE:adp_simpler:{name}:{status}", flush=True)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Mapping[str, Any], *, digest_field: str | None = None) -> str:
    normalized = dict(value)
    if digest_field:
        normalized.pop(digest_field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(normalized).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_observation_png(
    image: Any,
    *,
    output_dir: Path,
    episode_id: str,
    frame_index: int,
    kind: str,
) -> dict[str, Any]:
    """Persist the exact RGB array used by the policy as a lossless PNG."""

    import numpy as np
    from PIL import Image

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"observation_frame_dtype_not_uint8:{array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"observation_frame_shape_not_rgb:{array.shape}")
    frame_dir = output_dir / "media" / episode_id / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    path = frame_dir / f"{frame_index:06d}-{kind}.png"
    Image.fromarray(array, mode="RGB").save(
        path,
        format="PNG",
        compress_level=6,
        optimize=False,
    )
    return {
        "frame_index": frame_index,
        "kind": kind,
        "relative_path": path.relative_to(output_dir).as_posix(),
        "raw_rgb_sha256": "sha256:" + hashlib.sha256(array.tobytes()).hexdigest(),
        "png_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "channels": 3,
        "dtype": "uint8",
    }


def _encode_episode_video(
    frame_paths: Sequence[Path],
    *,
    video_path: Path,
    frames_per_second: float,
) -> dict[str, Any]:
    """Encode a human-review MP4 derived from the authoritative PNG sequence."""

    import cv2

    if not frame_paths:
        raise ValueError("episode_video_requires_at_least_one_frame")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError("episode_video_first_frame_unreadable")
    height, width = first.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(frames_per_second),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("episode_video_encoder_unavailable")
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"episode_video_frame_unreadable:{path.name}")
            if frame.shape[:2] != (height, width):
                raise ValueError(f"episode_video_frame_shape_mismatch:{path.name}")
            writer.write(frame)
    finally:
        writer.release()
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise RuntimeError("episode_video_not_written")
    return {
        "relative_path": video_path.as_posix(),
        "sha256": _file_sha256(video_path),
        "size_bytes": video_path.stat().st_size,
        "container": "mp4",
        "codec": "mp4v",
        "frames_per_second": float(frames_per_second),
        "frame_count": len(frame_paths),
    }


def _finalize_visual_evidence(
    *,
    output_dir: Path,
    episode_id: str,
    identity: Mapping[str, Any],
    policy_input_frames: Sequence[Mapping[str, Any]],
    terminal_observation: Mapping[str, Any] | None,
    frames_per_second: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Write the authoritative frame manifest and derived review video."""

    ordered_frames = [dict(row) for row in policy_input_frames]
    if terminal_observation is not None:
        ordered_frames.append(dict(terminal_observation))
    manifest = {
        "schema_version": FRAME_MANIFEST_SCHEMA_VERSION,
        "episode_id": episode_id,
        "identity": dict(identity),
        "policy_input_frames": [dict(row) for row in policy_input_frames],
        "terminal_observation": dict(terminal_observation)
        if terminal_observation is not None
        else None,
        "policy_input_frame_count": len(policy_input_frames),
        "video_frame_order": [row["relative_path"] for row in ordered_frames],
        "lossless_policy_inputs_are_authoritative": True,
        "derived_video_is_human_review_convenience": True,
    }
    manifest["frame_manifest_digest"] = _canonical_digest(
        manifest, digest_field="frame_manifest_digest"
    )
    manifest_path = output_dir / "media" / episode_id / "frame_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts = [
        {
            "role": "observation_frame_manifest",
            "relative_path": manifest_path.relative_to(output_dir).as_posix(),
            "sha256": _file_sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        }
    ]
    for row in policy_input_frames:
        artifacts.append(
            {
                "role": "policy_input_frame",
                "relative_path": row["relative_path"],
                "sha256": row["png_sha256"],
                "size_bytes": row["size_bytes"],
                "raw_rgb_sha256": row["raw_rgb_sha256"],
                "frame_index": row["frame_index"],
            }
        )
    if terminal_observation is not None:
        artifacts.append(
            {
                "role": "terminal_observation_frame",
                "relative_path": terminal_observation["relative_path"],
                "sha256": terminal_observation["png_sha256"],
                "size_bytes": terminal_observation["size_bytes"],
                "raw_rgb_sha256": terminal_observation["raw_rgb_sha256"],
                "frame_index": terminal_observation["frame_index"],
            }
        )
    if not ordered_frames:
        return (
            {
                "schema_version": VISUAL_EVIDENCE_SCHEMA_VERSION,
                "status": "unavailable_before_first_observation",
                "human_review_available": False,
                "frame_manifest_digest": manifest["frame_manifest_digest"],
                "policy_input_frame_count": 0,
                "terminal_observation_frame_present": False,
                "vlm_grading_used": False,
            },
            artifacts,
        )
    video_path = output_dir / "media" / episode_id / "episode.mp4"
    video = _encode_episode_video(
        [output_dir / row["relative_path"] for row in ordered_frames],
        video_path=video_path,
        frames_per_second=frames_per_second,
    )
    video["relative_path"] = video_path.relative_to(output_dir).as_posix()
    video["derived_from_frame_manifest_digest"] = manifest["frame_manifest_digest"]
    artifacts.append(
        {
            "role": "episode_video",
            "relative_path": video["relative_path"],
            "sha256": video["sha256"],
            "size_bytes": video["size_bytes"],
            "media_type": "video/mp4",
        }
    )
    return (
        {
            "schema_version": VISUAL_EVIDENCE_SCHEMA_VERSION,
            "status": "complete",
            "human_review_available": True,
            "frame_manifest_digest": manifest["frame_manifest_digest"],
            "policy_input_frame_count": len(policy_input_frames),
            "terminal_observation_frame_present": terminal_observation is not None,
            "video": video,
            "vlm_grading_used": False,
        },
        artifacts,
    )


def _run(command: Sequence[str], *, cwd: Path | None = None, timeout: int = 3600) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_tail": (completed.stdout or "")[-12000:],
    }


def _cuda_toolkit_evidence(
    cuda_root: Path = Path("/usr/local/cuda"),
) -> dict[str, Any]:
    """Fail closed unless the pinned image supplies the XLA CUDA compiler inputs."""

    ptxas = shutil.which("ptxas")
    libdevice_files = sorted((cuda_root / "nvvm" / "libdevice").glob("libdevice*.bc"))
    if not ptxas:
        raise RuntimeError("cuda_toolkit_ptxas_missing")
    if not libdevice_files:
        raise RuntimeError("cuda_toolkit_libdevice_missing")
    ptxas_version = _run([ptxas, "--version"], timeout=60)
    if ptxas_version["returncode"] != 0:
        raise RuntimeError("cuda_toolkit_ptxas_version_failed")
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_root}"
    return {
        "cuda_root": str(cuda_root),
        "xla_flags": os.environ["XLA_FLAGS"],
        "ptxas_path": ptxas,
        "ptxas_version": ptxas_version["output_tail"].strip(),
        "libdevice_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in libdevice_files
        ],
    }


def _vulkan_runtime_evidence(
    icd_candidates: Sequence[Path] = (
        Path("/etc/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
        Path("/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json"),
    ),
) -> dict[str, Any]:
    """Bind the headless SAPIEN renderer to an observed NVIDIA Vulkan ICD."""

    icd = next((path for path in icd_candidates if path.is_file()), None)
    if icd is None:
        raise RuntimeError("nvidia_vulkan_icd_missing")
    vulkaninfo = shutil.which("vulkaninfo")
    if not vulkaninfo:
        raise RuntimeError("vulkaninfo_missing")
    os.environ["VK_DRIVER_FILES"] = str(icd)
    # Ubuntu 22.04's loader predates VK_DRIVER_FILES (Vulkan headers 1.3.207),
    # while newer loaders prefer it. Set both names to the same exact ICD so
    # either loader generation excludes Mesa/llvmpipe deterministically.
    os.environ["VK_ICD_FILENAMES"] = str(icd)
    probe = _run([vulkaninfo, "--summary"], timeout=60)
    output = probe["output_tail"].strip()
    if probe["returncode"] != 0:
        diagnostic = " ".join(output.split())[-800:] or "no_vulkaninfo_output"
        raise RuntimeError(f"nvidia_vulkan_probe_failed:{diagnostic}")
    lowered = output.lower()
    if "nvidia" not in lowered or "llvmpipe" in lowered:
        reason = "nvidia_marker_missing" if "nvidia" not in lowered else "llvmpipe_observed"
        raise RuntimeError(f"nvidia_vulkan_device_not_observed:{reason}")
    packages = _run(
        [
            "dpkg-query",
            "-W",
            "-f=${binary:Package}=${Version}\\n",
            "libegl1",
            "libxext6",
            "libvulkan1",
            "vulkan-tools",
        ],
        timeout=60,
    )
    if packages["returncode"] != 0:
        raise RuntimeError("vulkan_system_package_lock_failed")
    return {
        "driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
        "vk_driver_files": os.environ["VK_DRIVER_FILES"],
        "vk_icd_filenames": os.environ["VK_ICD_FILENAMES"],
        "icd_sha256": _file_sha256(icd),
        "vulkaninfo_path": vulkaninfo,
        "vulkaninfo_summary": output,
        "system_packages": sorted(
            line for line in packages["output_tail"].splitlines() if line.strip()
        ),
    }


def _download(url: str, path: Path, *, expected_size: int, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "BlueprintADPSimpler/1.0"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            handle.write(chunk)
    actual = "sha256:" + digest.hexdigest()
    if size != expected_size:
        raise ValueError(f"download_size_mismatch:{path.name}:{size}:{expected_size}")
    if actual != expected_sha256:
        raise ValueError(f"download_sha256_mismatch:{path.name}:{actual}:{expected_sha256}")


def _download_checkpoint_object(row: Mapping[str, Any], target: Path) -> None:
    name = str(row["name"])
    encoded_name = urllib.parse.quote(name, safe="")
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o/{encoded_name}"
        f"?alt=media&generation={row['generation']}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5()  # noqa: S324 - required to verify the upstream GCS object identity.
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "BlueprintADPSimpler/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            md5.update(chunk)
            handle.write(chunk)
    if size != row["size_bytes"]:
        raise ValueError(f"checkpoint_size_mismatch:{name}")
    if base64.b64encode(md5.digest()).decode() != row["md5_base64"]:
        raise ValueError(f"checkpoint_md5_mismatch:{name}")


def _checkpoint_identity_digest(candidate: Mapping[str, Any]) -> str:
    return _canonical_digest(
        {
            "candidate_id": candidate.get("candidate_id"),
            "checkpoint_prefix": candidate.get("checkpoint_prefix"),
            "checkpoint_objects": candidate.get("checkpoint_objects"),
        }
    )


def prepare_runtime(manifest: Mapping[str, Any], work_dir: Path) -> dict[str, Any]:
    _phase("cuda_toolkit_preflight")
    cuda_toolkit = _cuda_toolkit_evidence()
    _phase("cuda_toolkit_preflight", "completed")

    _phase("vulkan_runtime_preflight")
    vulkan_runtime = _vulkan_runtime_evidence()
    _phase("vulkan_runtime_preflight", "completed")

    _phase("source_checkout")
    source_dir = work_dir / "SimplerEnv"
    repository = manifest["source"]["repository"]
    if not source_dir.is_dir():
        clone = _run(
            ["git", "clone", "--filter=blob:none", repository["url"], str(source_dir)],
            timeout=900,
        )
        if clone["returncode"] != 0:
            raise RuntimeError("simpler_clone_failed:" + clone["output_tail"][-1000:])
    checkout = _run(
        ["git", "checkout", "--detach", repository["commit"]], cwd=source_dir, timeout=300
    )
    if checkout["returncode"] != 0:
        raise RuntimeError("simpler_checkout_failed:" + checkout["output_tail"][-1000:])
    submodule = _run(
        ["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir, timeout=900
    )
    if submodule["returncode"] != 0:
        raise RuntimeError("simpler_submodule_failed:" + submodule["output_tail"][-1000:])
    observed_source = _run(["git", "rev-parse", "HEAD"], cwd=source_dir)
    observed_submodule = _run(
        ["git", "-C", "ManiSkill2_real2sim", "rev-parse", "HEAD"], cwd=source_dir
    )
    if repository["commit"] not in observed_source["output_tail"]:
        raise RuntimeError("simpler_source_commit_mismatch")
    if repository["submodules"][0]["commit"] not in observed_submodule["output_tail"]:
        raise RuntimeError("simpler_submodule_commit_mismatch")
    _phase("source_checkout", "completed")

    _phase("dependency_install")
    install = _run(
        [sys.executable, "-m", "pip", "install", "--no-input", *PYTHON_REQUIREMENTS],
        timeout=2400,
    )
    if install["returncode"] != 0:
        raise RuntimeError("simpler_dependency_install_failed:" + install["output_tail"][-4000:])
    for editable in (source_dir / "ManiSkill2_real2sim", source_dir):
        installed = _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "--no-deps",
                "-e",
                str(editable),
            ],
            timeout=600,
        )
        if installed["returncode"] != 0:
            raise RuntimeError(
                "simpler_editable_install_failed:" + installed["output_tail"][-2000:]
            )
    _phase("dependency_install", "completed")

    _phase("language_encoder_acquisition")
    encoder = manifest["runtime"]["language_encoder"]
    encoder_archive = work_dir / "universal_sentence_encoder_large_v5.tar.gz"
    if not encoder_archive.is_file() or _file_sha256(encoder_archive) != encoder["archive_sha256"]:
        _download(
            encoder["download_url"],
            encoder_archive,
            expected_size=encoder["archive_size_bytes"],
            expected_sha256=encoder["archive_sha256"],
        )
    encoder_dir = work_dir / "universal_sentence_encoder_large_v5"
    if encoder_dir.exists():
        shutil.rmtree(encoder_dir)
    encoder_dir.mkdir()
    with tarfile.open(encoder_archive, "r:gz") as archive:
        # The pinned CUDA base image supplies Python 3.10, where tarfile's
        # ``filter=`` argument is unavailable.  The archive is already bound by
        # an exact SHA-256; still reject traversal and link members before
        # extracting the trusted immutable payload.
        root = encoder_dir.resolve()
        for member in archive.getmembers():
            target = (encoder_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError("language_encoder_archive_path_traversal")
            if member.issym() or member.islnk():
                raise ValueError("language_encoder_archive_link_member")
        archive.extractall(encoder_dir)  # noqa: S202 - digest-bound members validated above.
    _phase("language_encoder_acquisition", "completed")

    _phase("checkpoint_acquisition")
    checkpoints_root = work_dir / "checkpoints"
    for candidate in manifest["candidates"]:
        prefix = candidate["checkpoint_prefix"]
        destination = checkpoints_root / prefix.rsplit("/", 1)[-1]
        for row in candidate["checkpoint_objects"]:
            relative = row["name"][len(prefix) + 1 :]
            target = destination / relative
            if target.is_file() and target.stat().st_size == row["size_bytes"]:
                continue
            _download_checkpoint_object(row, target)
    _phase("checkpoint_acquisition", "completed")

    freeze = _run([sys.executable, "-m", "pip", "freeze", "--all"], timeout=120)
    if freeze["returncode"] != 0:
        raise RuntimeError("pip_freeze_failed")
    nvidia = _run(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader",
        ],
        timeout=60,
    )
    if nvidia["returncode"] != 0:
        raise RuntimeError("nvidia_smi_failed:" + nvidia["output_tail"][-1000:])
    lock = {
        "schema_version": RUNTIME_LOCK_SCHEMA_VERSION,
        "container_image": manifest["runtime"]["environment_lock"]["container_image"],
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pip_freeze": sorted(line for line in freeze["output_tail"].splitlines() if line.strip()),
        "declared_python_requirements": list(PYTHON_REQUIREMENTS),
        "source_commit": repository["commit"],
        "submodule_commits": {
            repository["submodules"][0]["path"]: repository["submodules"][0]["commit"]
        },
        "language_encoder_archive_sha256": encoder["archive_sha256"],
        "nvidia_smi": nvidia["output_tail"].strip(),
        "cuda_toolkit": cuda_toolkit,
        "vulkan_runtime": vulkan_runtime,
    }
    lock["runtime_lock_digest"] = _canonical_digest(lock, digest_field="runtime_lock_digest")
    _phase("runtime_lock", "completed")
    return {
        "source_dir": source_dir,
        "encoder_dir": encoder_dir,
        "checkpoints_root": checkpoints_root,
        "runtime_lock": lock,
        "install": install,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return repr(value)


def _activate_verified_source_roots(prepared: Mapping[str, Any]) -> list[str]:
    source_dir = Path(prepared["source_dir"]).resolve()
    roots = [source_dir / "ManiSkill2_real2sim", source_dir]
    for import_root in roots:
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    return [str(path) for path in roots]


def run_episodes(
    manifest: Mapping[str, Any], prepared: Mapping[str, Any], output_dir: Path
) -> list[dict[str, Any]]:
    # The editable installs are performed by child pip processes after this
    # interpreter starts, so their new .pth entries are not automatically
    # processed in the current process. Insert only the two commit-verified
    # roots returned by prepare_runtime before importing the public runtime.
    _activate_verified_source_roots(prepared)
    import numpy as np
    import tensorflow as tf

    from simpler_env.policies.rt1.rt1_model import RT1Inference
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    import simpler_env

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    np.random.seed(0)
    tf.random.set_seed(0)
    task_names = {
        "coke-can-horizontal-lr-switch": "google_robot_pick_horizontal_coke_can",
        "coke-can-vertical-laid": "google_robot_pick_vertical_coke_can",
        "coke-can-standing-upright": "google_robot_pick_standing_coke_can",
    }
    episodes: list[dict[str, Any]] = []
    runtime_lock = prepared["runtime_lock"]
    source_commit = manifest["source"]["repository"]["commit"]
    evaluator = manifest["task"]["evaluator"]
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        checkpoint_digest = _checkpoint_identity_digest(candidate)
        checkpoint_dir = (
            prepared["checkpoints_root"] / candidate["checkpoint_prefix"].rsplit("/", 1)[-1]
        )
        model = RT1Inference(
            saved_model_path=str(checkpoint_dir),
            lang_embed_model_path=str(prepared["encoder_dir"]),
            policy_setup="google_robot",
        )
        for condition in manifest["conditions"]:
            condition_id = condition["condition_id"]
            identity = {
                "candidate_id": candidate_id,
                "condition_id": condition_id,
                "seed": 0,
            }
            episode_id = (
                "adp-" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:24]
            )
            observations: list[str] = []
            actions: list[dict[str, Any]] = []
            metrics: list[dict[str, Any]] = []
            policy_input_frames: list[dict[str, Any]] = []
            failure: dict[str, Any] | None = None
            status = "failed"
            success: bool | None = None
            step_count = 0
            policy_queries = 0
            env = None
            latest_image: Any = None
            predicted_terminated = False
            truncated = False
            environment_terminated = False
            reset = {
                "condition": condition["reset_binding"],
                "robot_xy": manifest["task"]["reset"]["robot_xy"],
                "robot_quaternion_xyzw": manifest["task"]["reset"]["robot_quaternion_xyzw"],
                "object_xy": [-0.235, 0.2],
                "seed": 0,
                "representative_fixed_reset_not_published_25_trial_grid": True,
            }
            _phase(f"episode:{candidate_id}:{condition_id}")
            try:
                env = simpler_env.make(
                    task_names[condition_id],
                    control_freq=manifest["task"]["controller"]["control_frequency_hz"],
                    sim_freq=manifest["task"]["controller"]["simulation_frequency_hz"],
                    max_episode_steps=manifest["task"]["controller"]["max_control_steps"],
                )
                reset_options = {
                    "robot_init_options": {
                        "init_xy": np.array(reset["robot_xy"]),
                        "init_rot_quat": np.array(reset["robot_quaternion_xyzw"]),
                    },
                    "obj_init_options": {"init_xy": np.array(reset["object_xy"])},
                }
                obs, reset_info = env.reset(seed=0, options=reset_options)
                reset["environment_reset_info"] = _jsonable(reset_info)
                unwrapped = env.unwrapped
                instruction = unwrapped.get_language_instruction()
                model.reset(instruction)
                image = get_image_from_maniskill2_obs_dict(unwrapped, obs)
                latest_image = image
                while not (predicted_terminated or truncated):
                    frame = _write_observation_png(
                        image,
                        output_dir=output_dir,
                        episode_id=episode_id,
                        frame_index=policy_queries,
                        kind="policy-input",
                    )
                    policy_input_frames.append(frame)
                    observations.append(frame["raw_rgb_sha256"])
                    policy_queries += 1
                    raw_action, action = model.step(image, instruction)
                    actions.append(
                        {
                            "raw": _jsonable(raw_action),
                            "normalized": _jsonable(action),
                        }
                    )
                    predicted_terminated = bool(action["terminate_episode"][0] > 0)
                    if predicted_terminated and not unwrapped.is_final_subtask():
                        predicted_terminated = False
                        unwrapped.advance_to_next_subtask()
                    obs, _reward, done, truncated, info = env.step(
                        np.concatenate(
                            [action["world_vector"], action["rot_axangle"], action["gripper"]]
                        )
                    )
                    step_count += 1
                    environment_terminated = bool(done)
                    metric = _jsonable(info)
                    if not isinstance(metric, Mapping) or not isinstance(
                        metric.get("success"), bool
                    ):
                        raise RuntimeError("environment_success_metric_missing")
                    metrics.append(dict(metric))
                    success = bool(metric["success"])
                    new_instruction = unwrapped.get_language_instruction()
                    if new_instruction != instruction:
                        instruction = new_instruction
                    image = get_image_from_maniskill2_obs_dict(unwrapped, obs)
                    latest_image = image
                status = "completed"
            except Exception as exc:  # retain a typed terminal record for every cell.
                failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            finally:
                if env is not None:
                    env.close()
            terminal_observation: dict[str, Any] | None = None
            visual_artifacts: list[dict[str, Any]] = []
            try:
                if latest_image is not None and step_count > 0:
                    terminal_observation = _write_observation_png(
                        latest_image,
                        output_dir=output_dir,
                        episode_id=episode_id,
                        frame_index=len(policy_input_frames),
                        kind="terminal-observation",
                    )
                visual_evidence, visual_artifacts = _finalize_visual_evidence(
                    output_dir=output_dir,
                    episode_id=episode_id,
                    identity=identity,
                    policy_input_frames=policy_input_frames,
                    terminal_observation=terminal_observation,
                    frames_per_second=float(manifest["task"]["controller"]["control_frequency_hz"]),
                )
                if status == "completed" and visual_evidence["status"] != "complete":
                    raise RuntimeError("completed_episode_visual_evidence_incomplete")
            except Exception as exc:
                previous_failure = failure
                failure = {
                    "type": "VisualEvidenceCaptureError",
                    "message": f"{type(exc).__name__}:{str(exc)[:1000]}",
                    "prior_failure": previous_failure,
                }
                if status == "completed":
                    status = "invalid"
                visual_evidence = {
                    "schema_version": VISUAL_EVIDENCE_SCHEMA_VERSION,
                    "status": "capture_failed",
                    "human_review_available": False,
                    "policy_input_frame_count": len(policy_input_frames),
                    "terminal_observation_frame_present": terminal_observation is not None,
                    "vlm_grading_used": False,
                    "failure": failure,
                }
            success_evidence = {
                "grader_type": "deterministic_simulator_state",
                "source_field": "environment_step_info.success",
                "final_value": success,
                "vlm_used": False,
                "human_grade_used": False,
                "policy_self_report_used": False,
            }
            trace = {
                "schema_version": "simpler_episode_trace.v1",
                "episode_id": episode_id,
                "identity": identity,
                "reset": reset,
                "observation_sha256_trace": observations,
                "action_trace": actions,
                "environment_metric_trace": metrics,
                "success_evidence": success_evidence,
                "termination": {
                    "policy_predicted_terminated": predicted_terminated,
                    "simulator_truncated": bool(truncated),
                    "environment_terminated": environment_terminated,
                    "step_count": step_count,
                },
                "visual_evidence": visual_evidence,
                "status": status,
                "success": success,
                "failure": failure,
            }
            trace_path = output_dir / "traces" / f"{episode_id}.json"
            _write_json(trace_path, trace)
            episodes.append(
                {
                    "episode_id": episode_id,
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": 0,
                    "status": status,
                    "success": success,
                    "source_commit": source_commit,
                    "dependency_lock_digest": runtime_lock["runtime_lock_digest"],
                    "checkpoint_identity_digest": checkpoint_digest,
                    "reset_digest": _canonical_digest(reset),
                    "observation_trace_digest": _canonical_digest({"observations": observations}),
                    "action_trace_digest": _canonical_digest({"actions": actions}),
                    "metric_trace_digest": _canonical_digest({"metrics": metrics}),
                    "policy_query_count": policy_queries,
                    "simulator_step_count": step_count,
                    "evaluator": {
                        "owner": "environment_not_policy",
                        "policy_self_report_used": False,
                        "grader_type": "deterministic_simulator_state",
                        "success_source": "environment_step_info.success",
                        "vlm_used": False,
                        "human_grade_used": False,
                        "source_git_blob_sha1": evaluator["source_git_blob_sha1"],
                        "success_semantics": evaluator["success_semantics"],
                    },
                    "success_evidence": success_evidence,
                    "visual_evidence": visual_evidence,
                    "failure": failure,
                    "artifacts": [
                        {
                            "role": "normalized_episode_trace",
                            "relative_path": trace_path.relative_to(output_dir).as_posix(),
                            "sha256": _file_sha256(trace_path),
                            "size_bytes": trace_path.stat().st_size,
                        },
                        *visual_artifacts,
                    ],
                }
            )
            _phase(f"episode:{candidate_id}:{condition_id}", status)
    return episodes


def run(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    _phase("worker")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(os.environ.get("BLUEPRINT_ADP_SIMPLER_WORK_DIR", "/workspace/adp_simpler"))
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        prepared = prepare_runtime(manifest, work_dir)
        _write_json(output_dir / "simpler_runtime_lock.json", prepared["runtime_lock"])
        episodes = run_episodes(manifest, prepared, output_dir)
        candidates = [
            {
                "candidate_id": row["candidate_id"],
                "checkpoint_identity_digest": _checkpoint_identity_digest(row),
                "genuine_checkpoint_loaded": any(
                    episode["candidate_id"] == row["candidate_id"]
                    and episode["policy_query_count"] > 0
                    for episode in episodes
                ),
            }
            for row in manifest["candidates"]
        ]
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed"
            if all(episode["status"] == "completed" for episode in episodes)
            else "completed_with_retained_failures",
            "reference_id": manifest["reference_id"],
            "source_identity_digest": manifest["source_identity_digest"],
            "source_manifest_digest": manifest["manifest_digest"],
            "runtime_lock_digest": prepared["runtime_lock"]["runtime_lock_digest"],
            "candidates": candidates,
            "episodes": episodes,
            "physical_outcome_values_accessed": False,
            "phase_label": PHASE_LABEL,
            "claim_ceiling": CLAIM_CEILING,
            "blockers": [],
        }
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "reference_id": manifest.get("reference_id"),
            "source_identity_digest": manifest.get("source_identity_digest"),
            "source_manifest_digest": manifest.get("manifest_digest"),
            "runtime_lock_digest": None,
            "candidates": [],
            "episodes": [],
            "physical_outcome_values_accessed": False,
            "phase_label": PHASE_LABEL,
            "claim_ceiling": CLAIM_CEILING,
            "blockers": [f"adp_simpler_worker_failed:{type(exc).__name__}:{str(exc)[:2000]}"],
        }
    result["execution_digest"] = _canonical_digest(result, digest_field="execution_digest")
    _write_json(output_dir / "adp_simpler_closed_loop_execution.json", result)
    _phase("worker", result["status"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.manifest, args.output_dir)
    print(json.dumps({"status": result["status"], "execution_digest": result["execution_digest"]}))
    return 0 if result["status"] in {"completed", "completed_with_retained_failures"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
