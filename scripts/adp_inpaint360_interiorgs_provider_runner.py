#!/usr/bin/env python3
"""Execute one exact InteriorGS edit with unchanged Inpaint360GS source."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


SCHEMA_VERSION = "adp_inpaint360_interiorgs_result.v1"
METHOD_NATIVE_MAX_WIDTH = 1600
MASK_ASSOCIATION_MODE = "pre_registered_single_target_method_native_resolution"
COMMAND_TIMEOUT_SECONDS = 10_800


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return dict(value) if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepend_pythonpath(env: dict[str, str], root: Path) -> dict[str, str]:
    """Put the checked-out method ahead of similarly named installed packages."""

    updated = dict(env)
    existing = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(root.resolve()), existing) if part
    )
    return updated


def _run(
    command: Sequence[str], *, cwd: Path, log_path: Path, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("wb") as log_stream:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
        returncode, timed_out = completed.returncode, False
    except subprocess.TimeoutExpired:
        returncode, timed_out = 124, True
    finished = dt.datetime.now(dt.timezone.utc)
    return {
        "command": [str(item) for item in command],
        "cwd": str(cwd),
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_seconds": (finished - started).total_seconds(),
        "stdout_stderr_sha256": _sha256(log_path),
        "log": log_path.name,
    }


def _source_identity(source: Path, spec: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for record in spec["source"]["files"]:
        path = source / str(record["path"])
        if record.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != record.get("target"):
                changed.append(str(record["path"]))
            continue
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {"matches": not changed, "changed_files": changed[:100]}


def _packet_identity(packet: Path, spec: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for record in spec["adapter"]["files"]:
        path = packet / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {"matches": not changed, "changed_files": changed[:100]}


def _artifact(path: Path, output: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    destination = output / "artifacts" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {
        "relative_path": destination.relative_to(output).as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _materialize_pre_registered_mask_binding(
    *, source_data: Path, target_method_instance_id: int, output: Path
) -> dict[str, Any]:
    """Bind one preregistered target across views without heuristic ID discovery."""

    raw_masks = sorted((source_data / "raw_hqsam").glob("*.png"))
    source_images = sorted((source_data / "images").glob("*.png"))
    source_images_by_name = {path.name: path for path in source_images}
    associated_dir = source_data / "associated_hqsam"
    existing_masks = sorted(associated_dir.glob("*.png"))
    target_pixel_counts: dict[str, int] = {}
    associated_target_pixel_counts: dict[str, int] = {}
    image_mask_dimensions: dict[str, dict[str, list[int]]] = {}
    source_image_sha256: dict[str, str] = {}
    raw_mask_sha256: dict[str, str] = {}
    associated_mask_sha256: dict[str, str] = {}
    derived_masks: dict[str, Image.Image] = {}
    invalid_masks: list[str] = []
    missing_source_images: list[str] = []
    for path in raw_masks:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            histogram = grayscale.histogram()
            mask_size = list(grayscale.size)
        source_image = source_images_by_name.get(path.name)
        if source_image is None:
            missing_source_images.append(path.name)
            invalid_masks.append(path.name)
            source_size: list[int] = []
        else:
            with Image.open(source_image) as image:
                source_size = list(image.size)
            source_image_sha256[path.name] = _sha256(source_image)
            if source_size != mask_size:
                invalid_masks.append(path.name)
            width, height = source_size
            if width > METHOD_NATIVE_MAX_WIDTH:
                scale = width / METHOD_NATIVE_MAX_WIDTH
                method_size = [int(width / scale), int(height / scale)]
            else:
                method_size = source_size
            derived = grayscale.resize(tuple(method_size), resample=Image.Resampling.NEAREST)
            derived_histogram = derived.histogram()
            derived_values = {
                index for index, count in enumerate(derived_histogram) if count
            }
            derived_target_count = int(derived_histogram[target_method_instance_id])
            associated_target_pixel_counts[path.name] = derived_target_count
            if (
                derived_values - {0, target_method_instance_id}
                or derived_target_count <= 0
            ):
                invalid_masks.append(path.name)
            derived_masks[path.name] = derived
        image_mask_dimensions[path.name] = {
            "source_image": source_size,
            "raw_mask": mask_size,
            "method_image_and_associated_mask": method_size if source_image else [],
        }
        populated_values = {index for index, count in enumerate(histogram) if count}
        target_count = (
            int(histogram[target_method_instance_id])
            if target_method_instance_id < len(histogram)
            else 0
        )
        target_pixel_counts[path.name] = target_count
        raw_mask_sha256[path.name] = _sha256(path)
        if populated_values - {0, target_method_instance_id} or target_count <= 0:
            invalid_masks.append(path.name)

    raw_mask_names = {path.name for path in raw_masks}
    unpaired_source_images = sorted(set(source_images_by_name) - raw_mask_names)
    invalid_masks = sorted(set(invalid_masks))
    valid = bool(raw_masks) and not existing_masks and not invalid_masks
    valid = valid and not unpaired_source_images and len(source_images) == len(raw_masks)
    if valid:
        associated_dir.mkdir(parents=True, exist_ok=True)
        for path in raw_masks:
            associated_path = associated_dir / path.name
            derived_masks[path.name].save(associated_path, format="PNG")
            associated_mask_sha256[path.name] = _sha256(associated_path)
            with Image.open(associated_path) as image:
                valid = valid and list(image.size) == image_mask_dimensions[path.name][
                    "method_image_and_associated_mask"
                ]
        _write_json(
            associated_dir / "scene.json",
            {
                "association_mode": MASK_ASSOCIATION_MODE,
                "num_classes": target_method_instance_id + 1,
                "raw_mask_folder": str(source_data / "raw_hqsam"),
                "associated_mask_folder": str(associated_dir),
                "target_method_instance_id": target_method_instance_id,
                "method_native_max_width": METHOD_NATIVE_MAX_WIDTH,
                "categorical_resize_filter": "nearest",
            },
        )

    receipt = {
        "schema_version": "adp_inpaint360_pre_registered_mask_binding.v1",
        "status": "accepted" if valid else "blocked",
        "association_mode": MASK_ASSOCIATION_MODE,
        "target_method_instance_id": target_method_instance_id,
        "raw_mask_count": len(raw_masks),
        "source_image_count": len(source_images),
        "existing_associated_mask_count": len(existing_masks),
        "full_resolution_source_preserved": valid,
        "method_native_max_width": METHOD_NATIVE_MAX_WIDTH,
        "categorical_resize_filter": "nearest",
        "image_mask_dimensions": image_mask_dimensions,
        "target_pixel_counts": target_pixel_counts,
        "associated_target_pixel_counts": associated_target_pixel_counts,
        "source_image_sha256": source_image_sha256,
        "raw_mask_sha256": raw_mask_sha256,
        "associated_mask_sha256": associated_mask_sha256,
        "invalid_masks": invalid_masks,
        "missing_source_images": sorted(missing_source_images),
        "unpaired_source_images": unpaired_source_images,
        "blockers": [] if valid else ["inpaint360_pre_registered_mask_binding_invalid"],
    }
    _write_json(output / "pre_registered_mask_binding.json", receipt)
    return receipt


def _validate_method_native_resolution_commands(
    commands: list[tuple[str, list[str], Path, dict[str, str]]], *, output: Path
) -> dict[str, Any]:
    """Bind model stages to 1.6K native loading and LaMa handoff to its folder contract."""

    required_stage_resolutions = {
        "distillation": "-1",
        "removal": "-1",
        "virtual_views": "-1",
        "lama_prepare": "1",
        "lama_collect": "1",
        "ply_fusion": "-1",
        "inpaint_3d": "-1",
    }
    observed: dict[str, str | None] = {}
    violations: list[str] = []
    for stage, command, _cwd, _env in commands:
        if stage not in required_stage_resolutions:
            continue
        positions = [index for index, value in enumerate(command) if value == "--resolution"]
        value = (
            command[positions[0] + 1]
            if len(positions) == 1 and positions[0] + 1 < len(command)
            else None
        )
        observed[stage] = value
        if len(positions) != 1 or value != required_stage_resolutions[stage]:
            violations.append(stage)
    missing_stages = sorted(set(required_stage_resolutions) - set(observed))
    violations.extend(missing_stages)
    valid = not violations
    receipt = {
        "schema_version": "adp_inpaint360_method_native_resolution_command_contract.v1",
        "status": "accepted" if valid else "blocked",
        "method_native_max_width": METHOD_NATIVE_MAX_WIDTH,
        "required_stage_resolutions": required_stage_resolutions,
        "observed_stage_resolutions": observed,
        "violating_or_missing_stages": sorted(set(violations)),
        "blockers": (
            [] if valid else ["inpaint360_method_native_resolution_command_contract_invalid"]
        ),
    }
    _write_json(output / "method_native_resolution_command_contract.json", receipt)
    return receipt


def _retain_review_frames(
    render_dir: Path, output: Path, *, count: int = 8
) -> list[dict[str, Any]]:
    frames = sorted(render_dir.glob("[0-9][0-9][0-9][0-9][0-9].png"))
    if not frames:
        return []
    selected_indices = sorted(
        {round(index * (len(frames) - 1) / max(1, count - 1)) for index in range(count)}
    )
    retained: list[dict[str, Any]] = []
    review_dir = output / "review_frames"
    review_dir.mkdir(parents=True, exist_ok=True)
    for source_index in selected_indices:
        source = frames[source_index]
        with Image.open(source) as image:
            comparison = image.convert("RGB")
            comparison_size = comparison.size
            rgb = comparison.crop((0, 0, comparison.width // 2, comparison.height))
            rgb_name = f"frame_{source.stem}_rgb.png"
            comparison_name = f"frame_{source.stem}_rgb_and_mask.png"
            rgb_path = review_dir / rgb_name
            comparison_path = review_dir / comparison_name
            rgb.save(rgb_path, format="PNG")
            comparison.save(comparison_path, format="PNG")
        retained.append(
            {
                "source_frame": source.name,
                "source_frame_index": source_index,
                "rgb": {
                    "relative_path": rgb_path.relative_to(output).as_posix(),
                    "width": rgb.width,
                    "height": rgb.height,
                    "size_bytes": rgb_path.stat().st_size,
                    "sha256": _sha256(rgb_path),
                },
                "rgb_and_mask": {
                    "relative_path": comparison_path.relative_to(output).as_posix(),
                    "width": comparison_size[0],
                    "height": comparison_size[1],
                    "size_bytes": comparison_path.stat().st_size,
                    "sha256": _sha256(comparison_path),
                },
            }
        )
    return retained


def main() -> int:
    runtime = Path(__file__).resolve().parent
    source = runtime / "Inpaint360GS"
    packet = runtime / "interiorgs_adapter"
    output = Path(
        os.environ.get("BLUEPRINT_ADP_INPAINT360_OUTPUT_DIR", runtime.parent / "runtime_output")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _read_json(runtime / "execution_spec.json")
    main_python = str(source / ".venv/bin/python")
    lama_python = str(source / "LaMa/.venv/bin/python")
    source_before = _source_identity(source, spec)
    packet_before = _packet_identity(packet, spec)
    hardware = _run(["nvidia-smi", "-q"], cwd=source, log_path=output / "nvidia-smi.log")
    main_env = _prepend_pythonpath(dict(os.environ), source)
    main_env.update({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1"})
    lama_env = dict(main_env)
    lama_env["TORCH_HOME"] = str(source / "LaMa")
    lama_env["PYTHONPATH"] = str(source / "LaMa")
    config_removal = packet / f"config/object_removal/blueprint/{spec['scene_id']}.json"
    config_inpaint = packet / f"config/object_inpaint/blueprint/{spec['scene_id']}.json"
    source_data = packet / "source"
    model = packet / "inpaint360_model"
    vanilla = packet / "vanilla_3dgs"
    commands: list[tuple[str, list[str], Path, dict[str, str]]] = [
        (
            "distillation",
            [
                main_python,
                "seg/distillation.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--vanilla_3dgs_path",
                str(vanilla),
                "--images",
                "images",
                "--object_path",
                "associated_hqsam",
                "--eval",
                "--config_file",
                str(packet / "config/distill.json"),
                "--resolution",
                "-1",
                "--save_iterations",
                "2000",
            ],
            source,
            main_env,
        ),
        (
            "removal",
            [
                main_python,
                "edit_object_removal.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--images",
                "images",
                "--iteration",
                "2000",
                "--config_file",
                str(config_removal),
                "--resolution",
                "-1",
                "--skip_test",
            ],
            source,
            main_env,
        ),
        (
            "virtual_views",
            [
                main_python,
                "tools/virtual_pose.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--images",
                "images",
                "--iteration",
                "2000",
                "--config_file",
                str(config_removal),
                "--resolution",
                "-1",
            ],
            source,
            main_env,
        ),
        (
            "virtual_masks",
            [
                main_python,
                str(runtime / "materialize_inpaint360_virtual_masks.py"),
                "--runtime-root",
                str(runtime),
                "--predicted-mask-dir",
                str(model / "virtual/ours_2000/objects_pred"),
                "--output-dir",
                str(source / "Segment-and-Track-Anything/tracking_results/images/images_masks"),
                "--receipt",
                str(output / "virtual_mask_handoff.json"),
                "--target-instance-id",
                str(spec["target_method_instance_id"]),
                "--expected-count",
                str(spec["runtime"]["virtual_view_count"]),
            ],
            source,
            main_env,
        ),
        (
            "lama_prepare",
            [
                main_python,
                "tools/prepare_lama_data.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--resolution",
                "1",
                "--inpaint2lama",
            ],
            source,
            main_env,
        ),
        (
            "lama_color",
            [lama_python, "bin/predict_color.py", "--data_name", "360_source_virtual"],
            source / "LaMa",
            lama_env,
        ),
        (
            "lama_depth",
            [lama_python, "bin/predict_depth.py", "--data_name", "360_source_virtual"],
            source / "LaMa",
            lama_env,
        ),
        (
            "lama_collect",
            [
                main_python,
                "tools/prepare_lama_data.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--resolution",
                "1",
            ],
            source,
            main_env,
        ),
        (
            "ply_fusion",
            [
                main_python,
                "edit_object_removal_plyfusion.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--config_file",
                str(config_removal),
                "--resolution",
                "-1",
            ],
            source,
            main_env,
        ),
        (
            "inpaint_3d",
            [
                main_python,
                "edit_object_inpaint.py",
                "--source_path",
                str(source_data),
                "--model_path",
                str(model),
                "--config_file",
                str(config_inpaint),
                "--resolution",
                "-1",
                "--render_video",
            ],
            source,
            main_env,
        ),
    ]
    workflow: list[dict[str, Any]] = []
    method_native_resolution_validation = _validate_method_native_resolution_commands(
        commands, output=output
    )
    resolution_accepted = method_native_resolution_validation["status"] == "accepted"
    workflow.append(
        {
            "stage": "method_native_resolution_contract",
            "operation": "bind_author_native_1_6k_loading_and_lama_folder_resolution",
            "cwd": str(packet),
            "returncode": 0 if resolution_accepted else 44,
            "timed_out": False,
            "receipt": "method_native_resolution_command_contract.json",
        }
    )
    mask_association_validation = _materialize_pre_registered_mask_binding(
        source_data=source_data,
        target_method_instance_id=int(spec["target_method_instance_id"]),
        output=output,
    )
    binding_accepted = mask_association_validation["status"] == "accepted"
    workflow.append(
        {
            "stage": "pre_registered_mask_binding",
            "operation": "copy_digest_bound_binary_target_masks",
            "cwd": str(packet),
            "returncode": 0 if binding_accepted else 43,
            "timed_out": False,
            "receipt": "pre_registered_mask_binding.json",
        }
    )
    if resolution_accepted and binding_accepted:
        for stage, command, cwd, env in commands:
            observed = _run(command, cwd=cwd, env=env, log_path=output / f"{stage}.log")
            observed["stage"] = stage
            workflow.append(observed)
            if observed["returncode"] != 0:
                break
    completed = {row["stage"] for row in workflow if row["returncode"] == 0}
    final_ply = model / "point_cloud_object_inpaint_virtual/iteration_5000/point_cloud.ply"
    final_video = model / "video/ours__object_inpaint_virtual/iteration_5000/final_video.mp4"
    final_render_dir = final_video.parent
    source_after = _source_identity(source, spec)
    blockers: list[str] = []
    required = [
        "method_native_resolution_contract",
        "pre_registered_mask_binding",
        *[stage for stage, _, _, _ in commands],
    ]
    for stage in required:
        if stage not in completed:
            blockers.append(f"inpaint360_{stage}_failed_or_not_executed")
            break
    if hardware["returncode"] != 0:
        blockers.append("inpaint360_nvidia_hardware_probe_failed")
    if not source_before["matches"] or not source_after["matches"]:
        blockers.append("inpaint360_author_source_modified")
    if not packet_before["matches"]:
        blockers.append("inpaint360_adapter_input_changed_before_execution")
    if not final_ply.is_file() or final_ply.stat().st_size == 0:
        blockers.append("inpaint360_final_point_cloud_missing")
    review_frames = _retain_review_frames(final_render_dir, output)
    if "inpaint_3d" in completed and len(review_frames) != 8:
        blockers.append("inpaint360_review_frames_missing")
    main_freeze = output / "main-pip-freeze.txt"
    lama_freeze = output / "lama-pip-freeze.txt"
    for freeze, blocker in (
        (main_freeze, "inpaint360_main_environment_receipt_missing"),
        (lama_freeze, "inpaint360_lama_environment_receipt_missing"),
    ):
        if not freeze.is_file() or freeze.stat().st_size == 0:
            blockers.append(blocker)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "scene_id": spec["scene_id"],
        "target_instance_id": spec["target_instance_id"],
        "target_method_instance_id": spec["target_method_instance_id"],
        "source_commit": spec["source"]["commit"],
        "source_tree": spec["source"]["tree"],
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "source_modified": not source_after["matches"],
        "adapter_identity_before": packet_before,
        "hardware_probe": hardware,
        "workflow": workflow,
        "mask_association_validation": mask_association_validation,
        "method_native_resolution_validation": method_native_resolution_validation,
        "method_native_resolution_execution": resolution_accepted,
        "mask_association_executed": False,
        "mask_association_mode": MASK_ASSOCIATION_MODE,
        "pre_registered_mask_binding_materialized": (
            "pre_registered_mask_binding" in completed
        ),
        "virtual_masks_materialized": "virtual_masks" in completed,
        "lama_color_executed": "lama_color" in completed,
        "lama_depth_executed": "lama_depth" in completed,
        "inpaint_3d_executed": "inpaint_3d" in completed and final_ply.is_file(),
        "final_point_cloud": _artifact(final_ply, output),
        "final_review_video": _artifact(final_video, output),
        "final_review_frames": review_frames,
        "rendered_frames_have_no_hidden_background_truth": True,
        "publisher_splat_is_not_metric_surface_truth": True,
        "replacement_or_physics_result": False,
        "retry_cap": 0,
        "blockers": sorted(set(blockers)),
        "raw_secret_values_recorded": False,
    }
    _write_json(output / "adp_inpaint360_interiorgs_result.json", result)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
