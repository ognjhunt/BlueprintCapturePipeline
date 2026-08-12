#!/usr/bin/env python3
"""Execute one sealed object-free ArtiFixer/3D/3D+ candidate packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "public_scene_artifixer3d_bundle.v1"
REQUEST_SCHEMA = "public_scene_artifixer3d_runtime_request.v1"
RESULT_SCHEMA = "public_scene_artifixer3d_runtime_result.v1"
INPUT_SCHEMA = "public_scene_artifixer3d_candidate_inputs.v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_digest(value: Mapping[str, Any], field: str) -> str:
    payload = json.loads(json.dumps(dict(value)))
    payload.pop(field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(code) from exc
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(value) + "\n", encoding="utf-8")


def _bound(root: Path, record: Any, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    relative = Path(str(record.get("relative_path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(code)
    path = root / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or root.resolve() not in path.resolve().parents
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _validate_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = root / "provider_runtime"
    manifest = _read(runtime / "artifixer3d_bundle_manifest.json", "artifixer3d_manifest_unreadable")
    request = _read(runtime / "artifixer3d_runtime_request.json", "artifixer3d_request_unreadable")
    candidate = _read(
        runtime / "input" / "public_scene_artifixer3d_candidate_inputs.v3.json",
        "artifixer3d_candidate_unreadable",
    )
    attestation = _read(
        runtime / "artifixer3d_use_attestation.json",
        "artifixer3d_use_attestation_unreadable",
    )
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("manifest_digest") != _canonical_digest(manifest, "manifest_digest")
        or request.get("schema_version") != REQUEST_SCHEMA
        or request.get("runtime_request_digest") != _canonical_digest(request, "runtime_request_digest")
        or candidate.get("schema_version") != INPUT_SCHEMA
        or candidate.get("receipt_digest") != _canonical_digest(candidate, "receipt_digest")
        or manifest.get("runtime_request", {}).get("runtime_request_digest")
        != request["runtime_request_digest"]
        or manifest.get("candidate_input_receipt", {}).get("receipt_digest")
        != candidate["receipt_digest"]
        or request.get("candidate_input_receipt_digest") != candidate["receipt_digest"]
        or manifest.get("contains_raw_dataset_bytes") is not False
        or manifest.get("contains_model_weights") is not False
        or request.get("source_object_restoration_permitted") is not False
        or request.get("outside_exact_support_changed_pixels_permitted") != 0
        or manifest.get("blueprint_source_identity")
        != request.get("blueprint_source_identity")
        or attestation.get("attestation_digest")
        != _canonical_digest(attestation, "attestation_digest")
        or manifest.get("use_attestation", {}).get("attestation_digest")
        != attestation.get("attestation_digest")
        or request.get("use_attestation", {}).get("attestation_digest")
        != attestation.get("attestation_digest")
        or attestation.get("internal_noncommercial_research_and_development_only")
        is not True
        or attestation.get("private_derived_input_upload_authorized") is not True
        or attestation.get("raw_dataset_bytes_upload_authorized") is not False
        or attestation.get("provider_training_authorized") is not False
        or attestation.get("commercial_use_authorized") is not False
        or attestation.get("redistribution_authorized") is not False
        or attestation.get("publication_authorized") is not False
    ):
        raise ValueError("artifixer3d_bundle_binding_invalid")
    for row in manifest.get("candidate_files") or []:
        _bound(runtime / "input", row, "artifixer3d_candidate_file_invalid")
    for row in manifest.get("source_files") or []:
        _bound(runtime / "ArtiFixer_official", row, "artifixer3d_source_file_invalid")
    if _bound(
        root,
        manifest.get("use_attestation"),
        "artifixer3d_use_attestation_unbound",
    ) != runtime / "artifixer3d_use_attestation.json":
        raise ValueError("artifixer3d_use_attestation_unbound")
    return manifest, request, candidate


def _verify_inventory(root: Path, rows: Sequence[Mapping[str, Any]], code: str) -> None:
    for row in rows:
        relative = Path(str(row.get("path") or ""))
        path = root / relative
        if (
            not relative.parts
            or relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ValueError(code + ":" + relative.as_posix())


def _download_models(request: Mapping[str, Any], cache: Path) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download, snapshot_download

    model = request["model"]
    wan = request["wan_base"]
    checkpoint_dir = cache / "artifixer"
    wan_dir = cache / "wan"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    wan_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(
        hf_hub_download(
            repo_id=model["repository"],
            revision=model["revision"],
            filename=model["files"][0]["path"],
            local_dir=checkpoint_dir,
        )
    )
    snapshot_download(
        repo_id=wan["repository"],
        revision=wan["revision"],
        allow_patterns=[row["path"] for row in wan["files"]],
        local_dir=wan_dir,
    )
    _verify_inventory(checkpoint_dir, model["files"], "artifixer3d_checkpoint_invalid")
    _verify_inventory(wan_dir, wan["files"], "artifixer3d_wan_runtime_invalid")
    return checkpoint, wan_dir


def _zero_prompt(path: Path) -> None:
    import h5py
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        dataset = output.create_dataset("unconditioned", data=np.zeros((1, 4096), dtype=np.uint16))
        dataset.attrs["caption"] = ""


def _run(command: Sequence[str], *, cwd: Path, log: Path, timeout: int) -> None:
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "COMMAND " + " ".join(command) + "\n"
        + f"DURATION_SECONDS {time.monotonic() - started:.6f}\n"
        + (completed.stdout or "")
        + (completed.stderr or ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise ValueError(f"artifixer3d_command_failed:{Path(command[1]).name if len(command) > 1 else 'command'}")


def _materialize_split(template_path: Path, output_path: Path) -> dict[str, Any]:
    template = _read(template_path, "artifixer3d_split_template_invalid")
    split = template.get("upstream_split")
    if not isinstance(split, Mapping) or set(split) != {"test"}:
        raise ValueError("artifixer3d_split_template_invalid")
    _write(output_path, split)
    return dict(split)


def _prediction_dir(save_root: Path, task_id: str) -> Path:
    matches = sorted(save_root.glob(f"**/{task_id}/frames/batch_0000/pred"))
    if len(matches) != 1 or not matches[0].is_dir():
        raise ValueError("artifixer3d_prediction_directory_invalid")
    return matches[0]


def _exact_composite(
    *, retained: Path, mask: Path, prediction: Path, output: Path
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    with Image.open(retained) as image:
        before = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(mask) as image:
        support = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    with Image.open(prediction) as image:
        generated = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if before.shape != generated.shape or before.shape[:2] != support.shape:
        raise ValueError("artifixer3d_composite_shape_invalid")
    composite = before.copy()
    composite[support] = generated[support]
    outside = ~support
    outside_changes = int(
        np.count_nonzero(np.any(composite[outside] != before[outside], axis=1))
    )
    if outside_changes != 0:
        raise ValueError("artifixer3d_outside_support_change")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite, mode="RGB").save(output)
    return {
        "path": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "repair_pixel_count": int(np.count_nonzero(support)),
        "outside_support_changed_pixels": outside_changes,
    }


def _copy_scene(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("artifixer3d_repaired_scene_exists")
    shutil.copytree(source, destination, symlinks=False)


def _task_runtime(
    *,
    task: Mapping[str, Any],
    input_root: Path,
    source_root: Path,
    output_root: Path,
    checkpoint: Path,
    wan_root: Path,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(task["task_id"])
    staged_task = input_root / task_id
    task_output = output_root / "tasks" / task_id
    logs = task_output / "logs"
    python = sys.executable
    prompt = staged_task / "captions" / "unconditioned_zero_prompt.h5"
    _zero_prompt(prompt)
    fold_predictions: dict[int, Path] = {}
    direct_rows: list[dict[str, Any]] = []
    for fold in task["direct_inference_folds"]:
        fold_id = str(fold["fold_id"])
        template = staged_task / Path(fold["split_template"]["path"]).name
        split_path = staged_task / f"split.direct_{fold_id}.json"
        _materialize_split(template, split_path)
        save_dir = task_output / "direct" / fold_id
        command = [
            python,
            "-m",
            "model_eval.run_inference",
            "--evalset",
            "reconstructed_colmap",
            "--checkpoint_pt",
            str(checkpoint),
            "--model_id",
            str(wan_root),
            "--save_dir",
            str(save_dir),
            "--split_path",
            str(split_path),
            "--render_trajectory",
            "trajectory",
            "--num_views",
            str(len(fold["selected_indices"])),
            "--neighbor_selection_mode",
            request["direct_inference"]["neighbor_selection_mode"],
            "--num_inference_steps",
            str(request["direct_inference"]["num_inference_steps"]),
            "--frames_per_block",
            str(request["direct_inference"]["frames_per_block"]),
            "--max_neighbors_per_encode",
            "1",
            "--save_frame_outputs_only",
            "--log_with",
            "none",
            "--seed",
            str(request["random_seed"]),
        ]
        _run(command, cwd=source_root, log=logs / f"direct_{fold_id}.log", timeout=3600)
        predictions = _prediction_dir(save_dir, task_id)
        for index in fold["target_indices"]:
            prediction = predictions / f"{int(index):05d}.png"
            if not prediction.is_file() or int(index) in fold_predictions:
                raise ValueError("artifixer3d_direct_prediction_coverage_invalid")
            fold_predictions[int(index)] = prediction
        direct_rows.append(
            {
                "fold_id": fold_id,
                "target_indices": fold["target_indices"],
                "prediction_directory": str(predictions),
                "log_sha256": _sha256(logs / f"direct_{fold_id}.log"),
            }
        )
    expected = set(task["direct_prediction_coverage_indices"])
    if set(fold_predictions) != expected:
        raise ValueError("artifixer3d_direct_prediction_coverage_invalid")

    repaired_scene = task_output / "repaired_scene"
    _copy_scene(staged_task, repaired_scene)
    composite_rows: list[dict[str, Any]] = []
    for frame in task["frames"]:
        index = int(frame["frame_index"])
        retained = staged_task / frame["rendered_rgb"]["relative_path"]
        mask = staged_task / frame["exact_repair_mask"]["relative_path"]
        output = repaired_scene / "images" / f"{index:05d}.png"
        row = _exact_composite(
            retained=retained,
            mask=mask,
            prediction=fold_predictions[index],
            output=output,
        )
        row.update(frame_index=index, camera_id=frame["camera_id"])
        composite_rows.append(row)

    distill_split = repaired_scene / "split.distill.json"
    _write(
        distill_split,
        {
            "test": {
                task_id: {
                    "transforms_path": "transforms.json",
                    "image_root": ".",
                    "render_dir": "renders",
                    "opacity_dir": "opacity",
                    "selected_indices_path": "selected_indices.json",
                    "prompt_path": "captions/unconditioned_zero_prompt.h5",
                    "camera_scale": 1.0,
                    "has_gt": False,
                }
            }
        },
    )
    artifixer3d_root = task_output / "artifixer3d"
    command = [
        python,
        "-m",
        "data_processing.run_artifixer3d",
        "--scene_root",
        str(repaired_scene),
        "--artifixer_frames_dir",
        str(repaired_scene / "images"),
        "--split_path",
        str(distill_split),
        "--output_root",
        str(artifixer3d_root),
        "--artifixer3d_steps",
        str(request["artifixer3d"]["steps"]),
        "--config_name",
        request["artifixer3d"]["config_name"],
        "--phases",
        "distill,render,prepare_artifixer3d_plus",
        "--no-use_wandb",
    ]
    _run(command, cwd=source_root, log=logs / "artifixer3d.log", timeout=10_800)
    plus_split = repaired_scene / "split_artifixer3d_plus.json"
    if not plus_split.is_file():
        raise ValueError("artifixer3d_plus_split_missing")
    plus_save = task_output / "artifixer3d_plus"
    command = [
        python,
        "-m",
        "model_eval.run_inference",
        "--evalset",
        "reconstructed_colmap",
        "--checkpoint_pt",
        str(checkpoint),
        "--model_id",
        str(wan_root),
        "--save_dir",
        str(plus_save),
        "--split_path",
        str(plus_split),
        "--render_trajectory",
        "all_frames",
        "--num_views",
        str(len(task["artifixer3d_distillation"]["selected_anchor_indices"])),
        "--neighbor_selection_mode",
        request["direct_inference"]["neighbor_selection_mode"],
        "--num_inference_steps",
        str(request["direct_inference"]["num_inference_steps"]),
        "--frames_per_block",
        str(request["direct_inference"]["frames_per_block"]),
        "--max_neighbors_per_encode",
        "1",
        "--save_frame_outputs_only",
        "--log_with",
        "none",
        "--seed",
        str(request["random_seed"]),
    ]
    _run(command, cwd=source_root, log=logs / "artifixer3d_plus.log", timeout=3600)
    plus_predictions = _prediction_dir(plus_save, task_id)
    final_root = task_output / "final_candidate_frames"
    final_rows: list[dict[str, Any]] = []
    for frame in task["frames"]:
        index = int(frame["frame_index"])
        prediction = plus_predictions / f"{index:05d}.png"
        if not prediction.is_file():
            raise ValueError("artifixer3d_plus_prediction_missing")
        row = _exact_composite(
            retained=staged_task / frame["rendered_rgb"]["relative_path"],
            mask=staged_task / frame["exact_repair_mask"]["relative_path"],
            prediction=prediction,
            output=final_root / f"{index:05d}.png",
        )
        row.update(frame_index=index, camera_id=frame["camera_id"])
        final_rows.append(row)
    checkpoints = sorted(artifixer3d_root.glob("**/ckpt_*.pt"))
    if len(checkpoints) != 1:
        raise ValueError("artifixer3d_checkpoint_missing_or_ambiguous")
    return {
        "task_id": task_id,
        "direct_folds": direct_rows,
        "direct_exact_composite_frames": composite_rows,
        "artifixer3d_checkpoint": {
            "path": str(checkpoints[0]),
            "size_bytes": checkpoints[0].stat().st_size,
            "sha256": _sha256(checkpoints[0]),
        },
        "artifixer3d_log_sha256": _sha256(logs / "artifixer3d.log"),
        "artifixer3d_plus_log_sha256": _sha256(logs / "artifixer3d_plus.log"),
        "final_candidate_frames": final_rows,
        "outside_support_changed_pixels_total": sum(
            row["outside_support_changed_pixels"] for row in final_rows
        ),
        "semantic_object_free_review_passed": False,
        "multiview_consistency_review_passed": False,
    }


def execute(*, bundle_root: Path, output_root: Path, rehearsal: bool) -> dict[str, Any]:
    manifest, request, candidate = _validate_bundle(bundle_root)
    base: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "runtime_request_digest": request["runtime_request_digest"],
        "manifest_digest": manifest["manifest_digest"],
        "candidate_input_receipt_digest": candidate["receipt_digest"],
        "replacement_object_count": candidate["replacement_object_count"],
        "task_ids": request["task_ids"],
        "source_object_restoration_permitted": False,
        "outside_exact_support_changed_pixels_permitted": 0,
        "provider_zero_required_after_return": True,
        "physical_or_deployment_evidence": False,
    }
    if rehearsal:
        return {
            "schema_version": "provider_bundle_rehearsal.v1",
            "status": "passed",
            "bundle_manifest_digest": manifest["manifest_digest"],
            "runtime_request_digest": request["runtime_request_digest"],
            "candidate_input_receipt_digest": candidate["receipt_digest"],
            "replacement_object_count": candidate["replacement_object_count"],
            "task_ids": request["task_ids"],
            "paid_inference_performed": False,
            "gpu_runtime_started": False,
            "provider_mutations_performed": 0,
            "blockers": [],
        }
    cache = bundle_root.parent / "artifixer3d_model_cache"
    checkpoint, wan_root = _download_models(request, cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    source_root = bundle_root / "provider_runtime" / "ArtiFixer_official"
    input_root = bundle_root / "provider_runtime" / "input"
    tasks = [
        _task_runtime(
            task=task,
            input_root=input_root,
            source_root=source_root,
            output_root=output_root,
            checkpoint=checkpoint,
            wan_root=wan_root,
            request=request,
        )
        for task in candidate["tasks"]
    ]
    if any(task["outside_support_changed_pixels_total"] != 0 for task in tasks):
        raise ValueError("artifixer3d_outside_support_change")
    return {
        **base,
        "status": "candidate_completed_requires_visual_and_multiview_review",
        "tasks": tasks,
        "model_loaded": True,
        "artifixer_direct_inference_executed": True,
        "artifixer3d_distillation_executed": True,
        "artifixer3d_plus_inference_executed": True,
        "provider_mutations_performed": 1,
        "blockers": [
            "semantic_object_free_visual_review_required",
            "multiview_consistency_review_required",
            "appearance_repair_not_yet_qualified",
        ],
        "claim_boundary": "generated_candidate_appearance_not_capture_or_physical_evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / (
        "provider_bundle_rehearsal.json"
        if args.rehearsal
        else "public_scene_artifixer3d_runtime_result.json"
    )
    try:
        result = execute(
            bundle_root=args.bundle_root.resolve(),
            output_root=output,
            rehearsal=args.rehearsal,
        )
    except Exception as exc:  # preserve the typed terminal runtime failure
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "blocked",
            "tasks": [],
            "model_loaded": False,
            "artifixer_direct_inference_executed": False,
            "artifixer3d_distillation_executed": False,
            "artifixer3d_plus_inference_executed": False,
            "provider_mutations_performed": 0 if args.rehearsal else 1,
            "blockers": [f"artifixer3d_runtime_exception:{type(exc).__name__}", str(exc)],
            "provider_zero_required_after_return": True,
            "physical_or_deployment_evidence": False,
            "claim_boundary": "runtime_failure_only",
        }
    _write(result_path, result)
    print(_canonical_json(result), flush=True)
    return 0 if result["status"] != "blocked" else 2


if __name__ == "__main__":
    raise SystemExit(main())
