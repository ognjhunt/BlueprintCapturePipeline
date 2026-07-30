"""Direct generated-only runtime for Blueprint Ctrl-World current-reference.

The runtime accepts an already-adapted 11x7 Cartesian conditioning request.
It never loads a policy, recorded future video, or outcome label.  Heavy
dependencies are imported only inside the GPU execution function so request
and result contracts remain hermetically testable.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REQUEST_SCHEMA_VERSION = "blueprint_ctrl_world_current_reference_staged_request.v1"
RESULT_SCHEMA_VERSION = "blueprint_ctrl_world_current_reference_runtime_result.v1"
ARM_ID = "blueprint_ctrl_world_current_reference"
ACTION_ROLLOUT_MARKER = "action_conditioned_video_rollout_generated"
VIEW_ORDER = (
    "observation/exterior_image_2_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)
MODEL_FREEZE = {
    "ctrl_world_source": {
        "repository": "https://github.com/Robert-gyj/Ctrl-World",
        "revision": "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d",
    },
    "ctrl_world_checkpoint": {
        "repository": "yjguo/Ctrl-World",
        "revision": "8cf814693f411962dc866a2ddb5b785afd17a93a",
    },
    "stable_video_diffusion": {
        "repository": "stabilityai/stable-video-diffusion-img2vid",
        "revision": "9cf024d5bfa8f56622af86c884f26a52f6676f2e",
    },
    "clip": {
        "repository": "openai/clip-vit-base-patch32",
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    },
}
EXPECTED_WORLD_MODEL_SHA256 = "ed17de48180d4e6f89fd33c53e9fb7a0196189c1a67d44c2c486a279a80ea8a8"
EXPECTED_STATE_STAT_SHA256 = "1e6fa202c87d6295f8b988dfd2764dec88796c910846cecdf684670fb818f208"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_generated_only_media(
    *,
    sequences: Mapping[str, Any],
    hashes: Mapping[str, Any],
    output_dir: Path,
    fps: float = 7.0,
) -> dict[str, Any]:
    """Encode retained generated PNGs into per-view and combined MP4 evidence."""

    import cv2
    import numpy as np

    if set(sequences) != set(VIEW_ORDER) or set(hashes) != set(VIEW_ORDER):
        raise ValueError("ctrl_world_runtime_generated_media_view_set_invalid")
    decoded: dict[str, list[np.ndarray]] = {}
    source_hashes: dict[str, list[str]] = {}
    for view_id in VIEW_ORDER:
        paths = sequences[view_id]
        digests = hashes[view_id]
        if not isinstance(paths, list) or len(paths) != 5:
            raise ValueError(f"ctrl_world_runtime_generated_media_frame_count_invalid:{view_id}")
        if not isinstance(digests, list) or len(digests) != 5:
            raise ValueError(f"ctrl_world_runtime_generated_media_hash_count_invalid:{view_id}")
        decoded[view_id] = []
        source_hashes[view_id] = []
        for path_value, expected_digest in zip(paths, digests, strict=True):
            path = Path(str(path_value)).expanduser().resolve()
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"ctrl_world_runtime_generated_media_frame_missing:{view_id}")
            digest = _sha256_file(path)
            if digest != expected_digest:
                raise ValueError(f"ctrl_world_runtime_generated_media_hash_mismatch:{view_id}")
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != (192, 320, 3):
                raise ValueError(f"ctrl_world_runtime_generated_media_geometry_invalid:{view_id}")
            decoded[view_id].append(frame)
            source_hashes[view_id].append(digest)

    video_dir = output_dir / "generated_video"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_specs = [
        (
            "combined_three_view",
            video_dir / "ctrl_world_generated_three_view.mp4",
            (960, 192),
        ),
        *[
            (
                f"view_{index}",
                video_dir / f"ctrl_world_generated_view_{index}.mp4",
                (320, 192),
            )
            for index in range(3)
        ],
    ]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writers = {
        name: cv2.VideoWriter(str(path), fourcc, fps, geometry)
        for name, path, geometry in video_specs
    }
    if not all(writer.isOpened() for writer in writers.values()):
        for writer in writers.values():
            writer.release()
        raise RuntimeError("ctrl_world_runtime_generated_media_writer_failed")
    try:
        for frame_index in range(5):
            per_view = [decoded[view_id][frame_index] for view_id in VIEW_ORDER]
            writers["combined_three_view"].write(np.concatenate(per_view, axis=1))
            for view_index, frame in enumerate(per_view):
                writers[f"view_{view_index}"].write(frame)
    finally:
        for writer in writers.values():
            writer.release()

    media: list[dict[str, Any]] = []
    for name, path, geometry in video_specs:
        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if (
            not path.is_file()
            or path.stat().st_size <= 0
            or frame_count != 5
            or (width, height) != geometry
        ):
            raise RuntimeError(f"ctrl_world_runtime_generated_media_validation_failed:{name}")
        media.append(
            {
                "role": name,
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
            }
        )
    return {
        "status": "completed",
        "media": media,
        "combined_three_view_path": media[0]["path"],
        "combined_three_view_sha256": media[0]["sha256"],
        "source_png_sha256_by_view": source_hashes,
        "physical_pixels_included": False,
        "generated_only": True,
    }


def _result_root_relative(path_value: Any, *, root: Path) -> str:
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError("ctrl_world_runtime_result_artifact_outside_output_root")
    return path.relative_to(root).as_posix()


def _portable_generated_artifacts(
    *,
    sequences: Mapping[str, Any],
    generated_media: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    portable_sequences = {
        view_id: [_result_root_relative(path, root=output_dir) for path in sequences[view_id]]
        for view_id in VIEW_ORDER
    }
    portable_media = dict(generated_media)
    portable_media["media"] = [
        {
            **dict(row),
            "path": _result_root_relative(row["path"], root=output_dir),
        }
        for row in generated_media["media"]
    ]
    portable_media["combined_three_view_path"] = _result_root_relative(
        generated_media["combined_three_view_path"], root=output_dir
    )
    portable_media["artifact_path_mode"] = "result_root_relative"
    return portable_sequences, portable_media


def _contained_file(root: Path, relative_value: Any, *, reason: str) -> Path:
    relative = Path(str(relative_value or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(reason)
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(reason)
    return path


def validate_staged_request(request_manifest_path: str | Path) -> dict[str, Any]:
    """Validate every staged request byte before model load or GPU mutation."""

    manifest_path = Path(request_manifest_path).expanduser().resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("ctrl_world_runtime_request_schema_invalid")
    recorded_digest = payload.get("request_sha256")
    digest_payload = dict(payload)
    digest_payload.pop("request_sha256", None)
    if recorded_digest != _canonical_sha256(digest_payload):
        raise ValueError("ctrl_world_runtime_request_digest_mismatch")
    if payload.get("model_freeze") != MODEL_FREEZE:
        raise ValueError("ctrl_world_runtime_request_model_freeze_mismatch")
    if tuple(payload.get("view_order") or ()) != VIEW_ORDER:
        raise ValueError("ctrl_world_runtime_request_view_order_invalid")
    if payload.get("selected_history_indices") != [0, 0, -12, -9, -6, -3]:
        raise ValueError("ctrl_world_runtime_request_history_indices_invalid")
    if payload.get("predicted_frame_count") != 5:
        raise ValueError("ctrl_world_runtime_request_frame_count_invalid")
    if payload.get("physical_future_observation_used") is not False:
        raise ValueError("ctrl_world_runtime_request_physical_future_not_false")
    if payload.get("physical_outcome_labels_accessed") is not False:
        raise ValueError("ctrl_world_runtime_request_outcome_access_not_false")
    if payload.get("policy_identity_in_provider_request") is not False:
        raise ValueError("ctrl_world_runtime_request_policy_identity_not_false")
    if payload.get("recorded_action_trace_used") is not False:
        raise ValueError("ctrl_world_runtime_request_recorded_trace_not_false")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("ctrl_world_runtime_request_seed_invalid")
    prompt = payload.get("task_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("ctrl_world_runtime_request_prompt_invalid")

    from PIL import Image

    histories = payload.get("selected_history_views")
    if not isinstance(histories, dict) or set(histories) != set(VIEW_ORDER):
        raise ValueError("ctrl_world_runtime_request_histories_invalid")
    history_paths: dict[str, list[Path]] = {}
    for view_id in VIEW_ORDER:
        rows = histories[view_id]
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError(f"ctrl_world_runtime_history_count_invalid:{view_id}")
        history_paths[view_id] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"ctrl_world_runtime_history_row_invalid:{view_id}")
            path = _contained_file(
                root,
                row.get("relative_path"),
                reason=f"ctrl_world_runtime_history_path_invalid:{view_id}",
            )
            if row.get("sha256") != _sha256_file(path):
                raise ValueError(f"ctrl_world_runtime_history_hash_mismatch:{view_id}")
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (320, 192):
                    raise ValueError(f"ctrl_world_runtime_history_geometry_invalid:{view_id}")
                image.verify()
            history_paths[view_id].append(path)

    current_rows = payload.get("current_views")
    if not isinstance(current_rows, dict) or set(current_rows) != set(VIEW_ORDER):
        raise ValueError("ctrl_world_runtime_request_current_views_invalid")
    current_paths: dict[str, Path] = {}
    for view_id in VIEW_ORDER:
        row = current_rows[view_id]
        if not isinstance(row, dict):
            raise ValueError(f"ctrl_world_runtime_current_view_row_invalid:{view_id}")
        path = _contained_file(
            root,
            row.get("relative_path"),
            reason=f"ctrl_world_runtime_current_view_path_invalid:{view_id}",
        )
        if row.get("sha256") != _sha256_file(path):
            raise ValueError(f"ctrl_world_runtime_current_view_hash_mismatch:{view_id}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (320, 192):
                raise ValueError(f"ctrl_world_runtime_current_view_geometry_invalid:{view_id}")
            image.verify()
        current_paths[view_id] = path

    action_row = payload.get("action_conditioning")
    if not isinstance(action_row, dict):
        raise ValueError("ctrl_world_runtime_action_record_invalid")
    action_path = _contained_file(
        root,
        action_row.get("relative_path"),
        reason="ctrl_world_runtime_action_path_invalid",
    )
    if action_row.get("sha256") != _sha256_file(action_path):
        raise ValueError("ctrl_world_runtime_action_hash_mismatch")
    import numpy as np

    action = np.load(action_path, allow_pickle=False)
    if (
        action.shape != (11, 7)
        or action.dtype != np.float64
        or not np.isfinite(action).all()
        or action_row.get("shape") != [11, 7]
        or action_row.get("dtype") != "float64"
    ):
        raise ValueError("ctrl_world_runtime_action_contract_invalid")
    return {
        "manifest": payload,
        "request_root": root,
        "request_sha256": recorded_digest,
        "seed": seed,
        "task_prompt": prompt.strip(),
        "history_paths": history_paths,
        "current_paths": current_paths,
        "action_conditioning_7d": action,
    }


def validate_runtime_assets(
    *,
    source_root: str | Path,
    source_manifest: Mapping[str, Any],
    world_model_checkpoint: str | Path,
    state_stat_path: str | Path,
) -> dict[str, Any]:
    """Bind direct execution to the exact public source and data-stat bytes."""

    root = Path(source_root).expanduser().resolve()
    if source_manifest.get("repository") != MODEL_FREEZE["ctrl_world_source"]["repository"]:
        raise ValueError("ctrl_world_runtime_source_repository_mismatch")
    if source_manifest.get("revision") != MODEL_FREEZE["ctrl_world_source"]["revision"]:
        raise ValueError("ctrl_world_runtime_source_revision_mismatch")
    rows = source_manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ctrl_world_runtime_source_manifest_files_missing")
    observed_source: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("ctrl_world_runtime_source_manifest_row_invalid")
        path = _contained_file(
            root, row.get("relative_path"), reason="ctrl_world_runtime_source_path_invalid"
        )
        digest = _sha256_file(path)
        if row.get("sha256") != digest or row.get("size_bytes") != path.stat().st_size:
            raise ValueError("ctrl_world_runtime_source_file_mismatch")
        observed_source.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    checkpoint = Path(world_model_checkpoint).expanduser().resolve()
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or _sha256_file(checkpoint) != EXPECTED_WORLD_MODEL_SHA256
    ):
        raise ValueError("ctrl_world_runtime_world_model_checkpoint_mismatch")
    stat_path = Path(state_stat_path).expanduser().resolve()
    if (
        not stat_path.is_file()
        or stat_path.is_symlink()
        or _sha256_file(stat_path) != EXPECTED_STATE_STAT_SHA256
    ):
        raise ValueError("ctrl_world_runtime_state_stat_mismatch")
    return {
        "source_manifest_sha256": _canonical_sha256(observed_source),
        "world_model_checkpoint_sha256": _sha256_file(checkpoint),
        "state_stat_sha256": _sha256_file(stat_path),
    }


def execute_generated_only_ctrl_world(
    *,
    validated_request: Mapping[str, Any],
    output_dir: Path,
    source_root: Path,
    world_model_checkpoint: Path,
    svd_model_root: Path,
    clip_model_root: Path,
    state_stat_path: Path,
) -> dict[str, Any]:
    """Run one direct Ctrl-World forward pass without future physical RGB."""

    import einops
    import numpy as np
    import torch
    from PIL import Image

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from config import wm_args
    from models.ctrl_world import CrtlWorld
    from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ctrl_world_runtime_exactly_one_cuda_device_required")
    seed = int(validated_request["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    args = wm_args(task_type="replay")
    args.svd_model_path = str(svd_model_root)
    args.clip_model_path = str(clip_model_root)
    args.ckpt_path = str(world_model_checkpoint)
    args.val_model_path = str(world_model_checkpoint)
    args.data_stat_path = str(state_stat_path)
    args.width = 320
    args.height = 192
    args.num_frames = 5
    args.num_history = 6
    args.action_dim = 7
    args.text_cond = True
    args.frame_level_cond = True
    args.dtype = torch.bfloat16

    model_started = time.monotonic()
    model = CrtlWorld(args)
    state_dict = torch.load(world_model_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    device = torch.device("cuda:0")
    model.to(device).to(args.dtype)
    model.eval()
    model_load_seconds = time.monotonic() - model_started
    stats = json.loads(state_stat_path.read_text(encoding="utf-8"))
    state_p01 = np.asarray(stats["state_01"], dtype=np.float64)[None, :]
    state_p99 = np.asarray(stats["state_99"], dtype=np.float64)[None, :]
    if state_p01.shape != (1, 7) or state_p99.shape != (1, 7):
        raise ValueError("ctrl_world_runtime_state_stat_shape_invalid")

    generator = torch.Generator(device=device).manual_seed(seed)
    history_latents = []
    encoded_view_cache: dict[str, Any] = {}
    encode_started = time.monotonic()

    def encode_view(path: Path) -> Any:
        cache_key = str(path)
        if cache_key in encoded_view_cache:
            return encoded_view_cache[cache_key]
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = (
            torch.from_numpy(array)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=args.dtype)
            / 255.0
            * 2.0
            - 1.0
        )
        latent = (
            model.pipeline.vae.encode(tensor)
            .latent_dist.sample(generator=generator)
            .mul_(model.pipeline.vae.config.scaling_factor)
        )
        if tuple(latent.shape) != (1, 4, 24, 40):
            raise RuntimeError("ctrl_world_runtime_encoded_view_shape_invalid")
        encoded_view_cache[cache_key] = latent[0]
        return latent[0]

    with torch.no_grad():
        for history_index in range(6):
            view_latents = []
            for view_id in VIEW_ORDER:
                view_latents.append(
                    encode_view(validated_request["history_paths"][view_id][history_index])
                )
            combined = torch.cat(view_latents, dim=1).unsqueeze(0)
            if tuple(combined.shape) != (1, 4, 72, 40):
                raise RuntimeError("ctrl_world_runtime_combined_history_shape_invalid")
            history_latents.append(combined)
        current_view_latents = [
            encode_view(validated_request["current_paths"][view_id]) for view_id in VIEW_ORDER
        ]
    history_encode_seconds = time.monotonic() - encode_started
    history = torch.cat(history_latents, dim=0).unsqueeze(0)
    current = torch.cat(current_view_latents, dim=1).unsqueeze(0)
    if tuple(current.shape) != (1, 4, 72, 40):
        raise RuntimeError("ctrl_world_runtime_current_latent_shape_invalid")

    action = np.asarray(validated_request["action_conditioning_7d"], dtype=np.float64)
    normalized = np.clip(
        2 * (action - state_p01) / (state_p99 - state_p01 + 1e-8) - 1,
        -1,
        1,
    )
    action_tensor = torch.tensor(normalized).unsqueeze(0).to(device).to(args.dtype)
    inference_started = time.monotonic()
    with torch.no_grad():
        text_token = model.action_encoder(
            action_tensor,
            validated_request["task_prompt"],
            model.tokenizer,
            model.text_encoder,
        )
        _, latents = CtrlWorldDiffusionPipeline.__call__(
            model.pipeline,
            image=current,
            text=text_token,
            width=320,
            height=576,
            num_frames=5,
            history=history,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
            generator=generator,
        )
        per_view_latents = einops.rearrange(
            latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1
        )
        if tuple(per_view_latents.shape) != (3, 5, 4, 24, 40):
            raise RuntimeError("ctrl_world_runtime_generated_latent_shape_invalid")
        flattened = per_view_latents.flatten(0, 1)
        decoded_chunks = []
        for index in range(0, flattened.shape[0], args.decode_chunk_size):
            chunk = flattened[index : index + args.decode_chunk_size]
            chunk = chunk / model.pipeline.vae.config.scaling_factor
            decoded_chunks.append(
                model.pipeline.vae.decode(chunk, num_frames=chunk.shape[0]).sample
            )
        videos = torch.cat(decoded_chunks, dim=0).reshape(3, 5, 3, 192, 320)
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        generated = videos.cpu().numpy().transpose(0, 1, 3, 4, 2)
    inference_seconds = time.monotonic() - inference_started

    sequences: dict[str, list[str]] = {}
    hashes: dict[str, list[str]] = {}
    for view_index, view_id in enumerate(VIEW_ORDER):
        sequences[view_id] = []
        hashes[view_id] = []
        for frame_index in range(5):
            path = output_dir / f"view_{view_index}" / f"frame_{frame_index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(generated[view_index, frame_index]).save(path)
            sequences[view_id].append(str(path))
            hashes[view_id].append(_sha256_file(path))
    return {
        "generated_view_frame_sequences": sequences,
        "generated_view_frame_sha256": hashes,
        "timing": {
            "model_load_seconds": round(model_load_seconds, 6),
            "history_encode_seconds": round(history_encode_seconds, 6),
            "inference_and_decode_seconds": round(inference_seconds, 6),
        },
        "cuda": {
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
        },
        "randomness": {
            "seed": seed,
            "torch_generator_bound": True,
            "bitwise_cross_hardware_determinism_claimed": False,
        },
    }


def run_ctrl_world_current_reference_runtime(
    *,
    request_manifest_path: str | Path,
    output_dir: str | Path,
    source_root: str | Path,
    source_manifest: Mapping[str, Any],
    world_model_checkpoint: str | Path,
    svd_model_root: str | Path,
    clip_model_root: str | Path,
    state_stat_path: str | Path,
    executor: Callable[..., Mapping[str, Any]] = execute_generated_only_ctrl_world,
) -> dict[str, Any]:
    """Validate, execute, and write one generated-only result receipt."""

    started = time.monotonic()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    validated = validate_staged_request(request_manifest_path)
    assets = validate_runtime_assets(
        source_root=source_root,
        source_manifest=source_manifest,
        world_model_checkpoint=world_model_checkpoint,
        state_stat_path=state_stat_path,
    )
    generated = executor(
        validated_request=validated,
        output_dir=output,
        source_root=Path(source_root).expanduser().resolve(),
        world_model_checkpoint=Path(world_model_checkpoint).expanduser().resolve(),
        svd_model_root=Path(svd_model_root).expanduser().resolve(),
        clip_model_root=Path(clip_model_root).expanduser().resolve(),
        state_stat_path=Path(state_stat_path).expanduser().resolve(),
    )
    generated_media = _write_generated_only_media(
        sequences=generated["generated_view_frame_sequences"],
        hashes=generated["generated_view_frame_sha256"],
        output_dir=output,
    )
    portable_sequences, portable_media = _portable_generated_artifacts(
        sequences=generated["generated_view_frame_sequences"],
        generated_media=generated_media,
        output_dir=output,
    )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "arm_id": ARM_ID,
        "request_sha256": validated["request_sha256"],
        "seed": validated["seed"],
        "model_freeze": MODEL_FREEZE,
        "runtime_assets": assets,
        "artifact_path_mode": "result_root_relative",
        "generated_view_frame_sequences": portable_sequences,
        "generated_view_frame_sha256": generated["generated_view_frame_sha256"],
        "generated_media": portable_media,
        "generated_rollout_video_path": portable_media["combined_three_view_path"],
        "same_frozen_wam_generated_all_views": True,
        ACTION_ROLLOUT_MARKER: True,
        "physical_future_observation_used": False,
        "physical_outcome_labels_accessed": False,
        "recorded_action_trace_used": False,
        "wam_to_wam_chaining": False,
        "candidate_policy_loaded_by_wam_runtime": False,
        "current_reference_not_exact_paper_reproduction": True,
        "timing": generated.get("timing", {}),
        "cuda": generated.get("cuda", {}),
        "randomness": generated.get("randomness", {}),
        "total_runtime_seconds": round(time.monotonic() - started, 6),
    }
    result["result_sha256"] = _canonical_sha256(result)
    _write_json(output / "wam_runtime_result.json", result)
    return result


def main() -> int:
    bundle_root = Path(os.environ["BLUEPRINT_WAM_PROVIDER_BUNDLE_DIR"]).expanduser().resolve()
    runtime_root = bundle_root / "provider_runtime"
    output_dir = Path(os.environ["BLUEPRINT_WAM_PROVIDER_OUTPUT_DIR"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_manifest = json.loads(
        (runtime_root / "wam_provider_runtime_manifest.json").read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    dependency: dict[str, Any] = {"status": "not_run"}
    cuda: dict[str, Any] = {"status": "not_run"}
    downloads: dict[str, Any] = {"status": "not_run"}
    try:
        from ctrl_world_provider_runtime_support import (
            _cuda_probe,
            _download_models,
            _ensure_dependencies,
            _validate_packaged_inputs,
        )

        blockers.extend(
            _validate_packaged_inputs(bundle_dir=bundle_root, manifest=runtime_manifest)
        )
        if not blockers:
            dependency = _ensure_dependencies(runtime_manifest)
            blockers.extend(dependency.get("blockers") or [])
        if not blockers:
            cuda = _cuda_probe()
            blockers.extend(cuda.get("blockers") or [])
        model_roots: dict[str, Path] = {}
        if not blockers:
            work_dir = (
                Path(
                    os.environ.get("BLUEPRINT_WAM_PROVIDER_WORK_DIR", bundle_root / "runtime_work")
                )
                .expanduser()
                .resolve()
            )
            work_dir.mkdir(parents=True, exist_ok=True)
            model_roots, downloads = _download_models(work_dir=work_dir, manifest=runtime_manifest)
            blockers.extend(downloads.get("blockers") or [])
        if not blockers:
            result = run_ctrl_world_current_reference_runtime(
                request_manifest_path=runtime_root
                / "ctrl_world_request"
                / "ctrl_world_current_reference_request.json",
                output_dir=output_dir,
                source_root=runtime_root / "ctrl_world_source",
                source_manifest=runtime_manifest["source_manifest"],
                world_model_checkpoint=model_roots["ctrl_world"] / "checkpoint-10000.pt",
                svd_model_root=model_roots["stable_video_diffusion"],
                clip_model_root=model_roots["clip"],
                state_stat_path=runtime_root
                / "ctrl_world_source"
                / "dataset_meta_info/droid/stat.json",
            )
            result["dependency"] = dependency
            result["model_downloads"] = downloads
            result.pop("result_sha256", None)
            result["result_sha256"] = _canonical_sha256(result)
            _write_json(output_dir / "wam_runtime_result.json", result)
            return 0
    except Exception as exc:
        blockers.append(f"ctrl_world_current_reference_runtime_exception:{type(exc).__name__}")
    blocked_result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "arm_id": ARM_ID,
        "blockers": sorted(set(blockers)),
        "model_freeze": MODEL_FREEZE,
        "dependency": dependency,
        "cuda": cuda,
        "model_downloads": downloads,
        "same_frozen_wam_generated_all_views": False,
        ACTION_ROLLOUT_MARKER: False,
        "physical_future_observation_used": False,
        "physical_outcome_labels_accessed": False,
        "recorded_action_trace_used": False,
        "wam_to_wam_chaining": False,
        "candidate_policy_loaded_by_wam_runtime": False,
    }
    _write_json(output_dir / "wam_runtime_result.json", blocked_result)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
