"""Direct generated-only runtime for the joint-position Ctrl-World arm.

The runtime accepts Blueprint's already-adapted 11x7 Cartesian request. It
does not load a candidate policy, recorded future video, outcome label, score,
or ranking. Heavy model imports occur only inside the single-GPU executor.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .ctrl_world_joint_position_reference_wam import (
    ACTION_CONDITIONING_SHAPE,
    ARM_ID,
    MODEL_FREEZE,
    PREDICTED_FRAME_COUNT,
    RUNTIME_RESULT_SCHEMA_VERSION,
    STAGED_REQUEST_SCHEMA_VERSION,
)
from .droid_ctrl_world_joint_position_closed_loop_adapter import (
    CTRL_WORLD_RELEASED_VIEW_ORDER,
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


HISTORY_FRAME_COUNT = 6
ENGINEERING_PROVENANCE = {
    "generic_mechanics_reference_commit": "82c3d14a519569104f6974445bfa0995810c3aed",
    "adaptation": "independent joint-position request and result contract",
    "confirmation_session_or_result_reused": False,
}


def _read_object(path: Path, *, reason: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(reason)
    return dict(value)


def _contained_file(root: Path, relative_value: Any, *, reason: str) -> Path:
    relative = Path(str(relative_value or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(reason)
    candidate = root.joinpath(*relative.parts)
    current = candidate
    while current != root:
        if current.is_symlink():
            raise ValueError(reason)
        current = current.parent
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(reason)
    return path


def _validated_image(path: Path, *, reason: str) -> Path:
    try:
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (320, 192):
                raise ValueError(reason)
            image.verify()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(reason) from exc
    return path


def validate_staged_joint_position_request(
    request_manifest_path: str | Path, *, expected_seed: int
) -> dict[str, Any]:
    """Validate every staged byte before model import or GPU mutation."""

    manifest_candidate = Path(request_manifest_path).expanduser()
    if manifest_candidate.is_symlink():
        raise ValueError("ctrl_world_joint_position_runtime_request_missing_or_unsafe")
    manifest_path = manifest_candidate.resolve()
    if not manifest_path.is_file():
        raise ValueError("ctrl_world_joint_position_runtime_request_missing_or_unsafe")
    root = manifest_path.parent
    payload = _read_object(
        manifest_path, reason="ctrl_world_joint_position_runtime_request_not_object"
    )
    if payload.get("schema_version") != STAGED_REQUEST_SCHEMA_VERSION:
        raise ValueError("ctrl_world_joint_position_runtime_request_schema_invalid")
    recorded_digest = payload.get("request_sha256")
    digest_payload = dict(payload)
    digest_payload.pop("request_sha256", None)
    if recorded_digest != canonical_sha256(digest_payload):
        raise ValueError("ctrl_world_joint_position_runtime_request_digest_mismatch")
    if payload.get("model_freeze") != MODEL_FREEZE:
        raise ValueError("ctrl_world_joint_position_runtime_model_freeze_mismatch")
    if tuple(payload.get("view_order") or ()) != CTRL_WORLD_RELEASED_VIEW_ORDER:
        raise ValueError("ctrl_world_joint_position_runtime_view_order_invalid")
    if tuple(payload.get("selected_history_indices") or ()) != (
        CTRL_WORLD_SELECTED_HISTORY_INDICES
    ):
        raise ValueError("ctrl_world_joint_position_runtime_history_indices_invalid")
    if payload.get("predicted_frame_count") != PREDICTED_FRAME_COUNT:
        raise ValueError("ctrl_world_joint_position_runtime_frame_count_invalid")
    if payload.get("seed") != expected_seed:
        raise ValueError("ctrl_world_joint_position_runtime_seed_mismatch")
    prompt = payload.get("task_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("ctrl_world_joint_position_runtime_prompt_invalid")
    for key in (
        "physical_future_observation_used",
        "physical_outcome_labels_accessed",
        "policy_identity_in_provider_request",
        "recorded_action_trace_used",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"ctrl_world_joint_position_runtime_request_{key}_not_false")

    histories = payload.get("selected_history_views")
    if not isinstance(histories, Mapping) or set(histories) != set(CTRL_WORLD_RELEASED_VIEW_ORDER):
        raise ValueError("ctrl_world_joint_position_runtime_histories_invalid")
    history_paths: dict[str, list[Path]] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        rows = histories[view_id]
        if not isinstance(rows, list) or len(rows) != HISTORY_FRAME_COUNT:
            raise ValueError(f"ctrl_world_joint_position_runtime_history_count_invalid:{view_id}")
        history_paths[view_id] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"ctrl_world_joint_position_runtime_history_row_invalid:{view_id}")
            path = _contained_file(
                root,
                row.get("relative_path"),
                reason=f"ctrl_world_joint_position_runtime_history_path_invalid:{view_id}",
            )
            if row.get("sha256") != file_sha256(path):
                raise ValueError(
                    f"ctrl_world_joint_position_runtime_history_hash_mismatch:{view_id}"
                )
            history_paths[view_id].append(
                _validated_image(
                    path,
                    reason=f"ctrl_world_joint_position_runtime_history_image_invalid:{view_id}",
                )
            )

    current_rows = payload.get("current_views")
    if not isinstance(current_rows, Mapping) or set(current_rows) != set(
        CTRL_WORLD_RELEASED_VIEW_ORDER
    ):
        raise ValueError("ctrl_world_joint_position_runtime_current_views_invalid")
    current_paths: dict[str, Path] = {}
    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER:
        row = current_rows[view_id]
        if not isinstance(row, Mapping):
            raise ValueError(
                f"ctrl_world_joint_position_runtime_current_view_row_invalid:{view_id}"
            )
        path = _contained_file(
            root,
            row.get("relative_path"),
            reason=f"ctrl_world_joint_position_runtime_current_view_path_invalid:{view_id}",
        )
        if row.get("sha256") != file_sha256(path):
            raise ValueError(
                f"ctrl_world_joint_position_runtime_current_view_hash_mismatch:{view_id}"
            )
        current_paths[view_id] = _validated_image(
            path,
            reason=f"ctrl_world_joint_position_runtime_current_view_image_invalid:{view_id}",
        )

    action_row = payload.get("action_conditioning")
    if not isinstance(action_row, Mapping):
        raise ValueError("ctrl_world_joint_position_runtime_action_record_invalid")
    action_path = _contained_file(
        root,
        action_row.get("relative_path"),
        reason="ctrl_world_joint_position_runtime_action_path_invalid",
    )
    if action_row.get("sha256") != file_sha256(action_path):
        raise ValueError("ctrl_world_joint_position_runtime_action_hash_mismatch")
    import numpy as np

    action = np.load(action_path, allow_pickle=False)
    if (
        action.shape != ACTION_CONDITIONING_SHAPE
        or action.dtype != np.float64
        or not np.isfinite(action).all()
        or action_row.get("shape") != list(ACTION_CONDITIONING_SHAPE)
        or action_row.get("dtype") != "float64"
    ):
        raise ValueError("ctrl_world_joint_position_runtime_action_contract_invalid")
    return {
        "manifest": payload,
        "request_sha256": recorded_digest,
        "seed": expected_seed,
        "task_prompt": prompt.strip(),
        "history_paths": history_paths,
        "current_paths": current_paths,
        "action_conditioning_7d": action,
    }


def _validate_source_manifest(source_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _read_object(
        manifest_path, reason="ctrl_world_joint_position_runtime_source_manifest_not_object"
    )
    freeze = MODEL_FREEZE["ctrl_world_source"]
    if manifest.get("repository") != freeze["repository"]:
        raise ValueError("ctrl_world_joint_position_runtime_source_repository_mismatch")
    if manifest.get("revision") != freeze["revision"]:
        raise ValueError("ctrl_world_joint_position_runtime_source_revision_mismatch")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("ctrl_world_joint_position_runtime_source_files_missing")
    observed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("ctrl_world_joint_position_runtime_source_row_invalid")
        path = _contained_file(
            source_root,
            row.get("relative_path"),
            reason="ctrl_world_joint_position_runtime_source_path_invalid",
        )
        digest = file_sha256(path)
        if row.get("sha256") != digest or row.get("size_bytes") != path.stat().st_size:
            raise ValueError("ctrl_world_joint_position_runtime_source_file_mismatch")
        observed.append(
            {
                "relative_path": path.relative_to(source_root).as_posix(),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "repository": freeze["repository"],
        "revision": freeze["revision"],
        "files_sha256": canonical_sha256(observed),
        "file_count": len(observed),
    }


def _validate_snapshot_root(
    path_value: str | Path, *, repository: str, revision: str, reason: str
) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_symlink():
        raise ValueError(reason)
    path = candidate.resolve()
    if not path.is_dir() or path.name != revision:
        raise ValueError(reason)
    marker = path / ".blueprint_snapshot_identity.json"
    identity = _read_object(marker, reason=reason)
    if identity != {"repository": repository, "revision": revision}:
        raise ValueError(reason)
    return path


def validate_joint_position_runtime_assets(
    *,
    source_root: str | Path,
    source_manifest_path: str | Path,
    world_model_checkpoint: str | Path,
    svd_model_root: str | Path,
    clip_model_root: str | Path,
    state_stat_path: str | Path,
) -> dict[str, Any]:
    """Bind direct execution to the exact source, checkpoint, stats, and snapshots."""

    source_candidate = Path(source_root).expanduser()
    if source_candidate.is_symlink():
        raise ValueError("ctrl_world_joint_position_runtime_source_root_invalid")
    source = source_candidate.resolve()
    if not source.is_dir():
        raise ValueError("ctrl_world_joint_position_runtime_source_root_invalid")
    source_manifest = _validate_source_manifest(
        source, Path(source_manifest_path).expanduser().resolve()
    )
    checkpoint_candidate = Path(world_model_checkpoint).expanduser()
    checkpoint = checkpoint_candidate.resolve()
    checkpoint_freeze = MODEL_FREEZE["ctrl_world_checkpoint"]
    if (
        checkpoint_candidate.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != checkpoint_freeze["size_bytes"]
        or file_sha256(checkpoint) != checkpoint_freeze["sha256"]
    ):
        raise ValueError("ctrl_world_joint_position_runtime_checkpoint_mismatch")
    stats_candidate = Path(state_stat_path).expanduser()
    stats = stats_candidate.resolve()
    stats_freeze = MODEL_FREEZE["ctrl_world_state_stats"]
    if (
        stats_candidate.is_symlink()
        or not stats.is_file()
        or file_sha256(stats) != stats_freeze["sha256"]
    ):
        raise ValueError("ctrl_world_joint_position_runtime_state_stats_mismatch")
    svd_freeze = MODEL_FREEZE["stable_video_diffusion"]
    clip_freeze = MODEL_FREEZE["clip"]
    svd = _validate_snapshot_root(
        svd_model_root,
        repository=svd_freeze["repository"],
        revision=svd_freeze["revision"],
        reason="ctrl_world_joint_position_runtime_svd_snapshot_mismatch",
    )
    clip = _validate_snapshot_root(
        clip_model_root,
        repository=clip_freeze["repository"],
        revision=clip_freeze["revision"],
        reason="ctrl_world_joint_position_runtime_clip_snapshot_mismatch",
    )
    return {
        "source": source_manifest,
        "world_model_checkpoint_sha256": file_sha256(checkpoint),
        "world_model_checkpoint_size_bytes": checkpoint.stat().st_size,
        "state_stats_sha256": file_sha256(stats),
        "stable_video_diffusion_snapshot": {
            "repository": svd_freeze["repository"],
            "revision": svd.name,
        },
        "clip_snapshot": {
            "repository": clip_freeze["repository"],
            "revision": clip.name,
        },
    }


def execute_generated_only_ctrl_world_joint_position(
    *,
    validated_request: Mapping[str, Any],
    output_dir: Path,
    source_root: Path,
    world_model_checkpoint: Path,
    svd_model_root: Path,
    clip_model_root: Path,
    state_stat_path: Path,
) -> dict[str, Any]:  # pragma: no cover - exact pinned CUDA runtime
    """Run one direct Ctrl-World forward pass without future physical RGB."""

    import einops
    import numpy as np
    import torch

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from config import wm_args
    from models.ctrl_world import CrtlWorld
    from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ctrl_world_joint_position_runtime_exactly_one_cuda_required")
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
    args.num_frames = PREDICTED_FRAME_COUNT
    args.num_history = HISTORY_FRAME_COUNT
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
    stats = _read_object(state_stat_path, reason="ctrl_world_joint_position_state_stats_invalid")
    state_p01 = np.asarray(stats["state_01"], dtype=np.float64)[None, :]
    state_p99 = np.asarray(stats["state_99"], dtype=np.float64)[None, :]
    if state_p01.shape != (1, 7) or state_p99.shape != (1, 7):
        raise ValueError("ctrl_world_joint_position_state_stats_shape_invalid")

    generator = torch.Generator(device=device).manual_seed(seed)
    encoded_view_cache: dict[str, Any] = {}

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
            raise RuntimeError("ctrl_world_joint_position_encoded_view_shape_invalid")
        encoded_view_cache[cache_key] = latent[0]
        return latent[0]

    encode_started = time.monotonic()
    history_latents = []
    with torch.no_grad():
        for history_index in range(HISTORY_FRAME_COUNT):
            combined = torch.cat(
                [
                    encode_view(validated_request["history_paths"][view_id][history_index])
                    for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER
                ],
                dim=1,
            ).unsqueeze(0)
            if tuple(combined.shape) != (1, 4, 72, 40):
                raise RuntimeError("ctrl_world_joint_position_history_latent_shape_invalid")
            history_latents.append(combined)
        current = torch.cat(
            [
                encode_view(validated_request["current_paths"][view_id])
                for view_id in CTRL_WORLD_RELEASED_VIEW_ORDER
            ],
            dim=1,
        ).unsqueeze(0)
    history_encode_seconds = time.monotonic() - encode_started
    history = torch.cat(history_latents, dim=0).unsqueeze(0)
    if tuple(current.shape) != (1, 4, 72, 40):
        raise RuntimeError("ctrl_world_joint_position_current_latent_shape_invalid")

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
            num_frames=PREDICTED_FRAME_COUNT,
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
        if tuple(per_view_latents.shape) != (3, PREDICTED_FRAME_COUNT, 4, 24, 40):
            raise RuntimeError("ctrl_world_joint_position_generated_latent_shape_invalid")
        flattened = per_view_latents.flatten(0, 1)
        decoded_chunks = []
        for index in range(0, flattened.shape[0], args.decode_chunk_size):
            chunk = flattened[index : index + args.decode_chunk_size]
            decoded_chunks.append(
                model.pipeline.vae.decode(
                    chunk / model.pipeline.vae.config.scaling_factor,
                    num_frames=chunk.shape[0],
                ).sample
            )
        videos = torch.cat(decoded_chunks, dim=0).reshape(3, PREDICTED_FRAME_COUNT, 3, 192, 320)
        generated = (
            ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
            .transpose(0, 1, 3, 4, 2)
        )
    inference_seconds = time.monotonic() - inference_started

    sequences: dict[str, list[str]] = {}
    hashes: dict[str, list[str]] = {}
    for view_index, view_id in enumerate(CTRL_WORLD_RELEASED_VIEW_ORDER):
        sequences[view_id] = []
        hashes[view_id] = []
        for frame_index in range(PREDICTED_FRAME_COUNT):
            path = output_dir / f"view_{view_index}" / f"frame_{frame_index:02d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(generated[view_index, frame_index]).save(path)
            sequences[view_id].append(str(path))
            hashes[view_id].append(file_sha256(path))
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


@dataclass(frozen=True)
class CtrlWorldJointPositionReferenceRuntime:
    """Configured direct callable accepted by the canary WAM arm."""

    source_root: Path
    source_manifest_path: Path
    world_model_checkpoint: Path
    svd_model_root: Path
    clip_model_root: Path
    state_stat_path: Path

    def __call__(
        self, *, request_manifest_path: Path, output_dir: Path, seed: int
    ) -> dict[str, Any]:
        started = time.monotonic()
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        validated = validate_staged_joint_position_request(
            request_manifest_path, expected_seed=seed
        )
        assets = validate_joint_position_runtime_assets(
            source_root=self.source_root,
            source_manifest_path=self.source_manifest_path,
            world_model_checkpoint=self.world_model_checkpoint,
            svd_model_root=self.svd_model_root,
            clip_model_root=self.clip_model_root,
            state_stat_path=self.state_stat_path,
        )
        generated = execute_generated_only_ctrl_world_joint_position(
            validated_request=validated,
            output_dir=output,
            source_root=self.source_root.expanduser().resolve(),
            world_model_checkpoint=self.world_model_checkpoint.expanduser().resolve(),
            svd_model_root=self.svd_model_root.expanduser().resolve(),
            clip_model_root=self.clip_model_root.expanduser().resolve(),
            state_stat_path=self.state_stat_path.expanduser().resolve(),
        )
        result: dict[str, Any] = {
            "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
            "status": "completed",
            "arm_id": ARM_ID,
            "request_sha256": validated["request_sha256"],
            "seed": seed,
            "model_freeze": MODEL_FREEZE,
            "runtime_asset_admission_passed": True,
            "runtime_assets": assets,
            "generated_view_frame_sequences": generated["generated_view_frame_sequences"],
            "generated_view_frame_sha256": generated["generated_view_frame_sha256"],
            "same_frozen_wam_generated_all_views": True,
            "physical_future_observation_used": False,
            "physical_outcome_labels_accessed": False,
            "recorded_action_trace_used": False,
            "wam_to_wam_chaining": False,
            "candidate_policy_loaded_by_wam_runtime": False,
            "blueprint_joint_position_reference_not_exact_paper_reproduction": True,
            "engineering_provenance": ENGINEERING_PROVENANCE,
            "timing": generated.get("timing", {}),
            "cuda": generated.get("cuda", {}),
            "randomness": generated.get("randomness", {}),
            "total_runtime_seconds": round(time.monotonic() - started, 6),
        }
        result["result_sha256"] = canonical_sha256(result)
        from .common import write_json

        write_json(output / "ctrl_world_joint_position_runtime_result.json", result)
        return result


__all__ = [
    "CtrlWorldJointPositionReferenceRuntime",
    "ENGINEERING_PROVENANCE",
    "execute_generated_only_ctrl_world_joint_position",
    "validate_joint_position_runtime_assets",
    "validate_staged_joint_position_request",
]
