#!/usr/bin/env python3
"""Execute AuraFusion360's unchanged sunflower inpaint-init author command once."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "adp_aura_author_smoke_result.v1"
COMMAND_TIMEOUT_SECONDS = 7200
QUALITY_FRAME_INDICES = (0, 34, 68, 102, 136, 170, 204, 239)


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


def _run(command: Sequence[str], *, cwd: Path, log_path: Path) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("wb") as log_stream:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                timeout=COMMAND_TIMEOUT_SECONDS,
                check=False,
            )
            returncode = completed.returncode
            timed_out = False
    except subprocess.TimeoutExpired:
        returncode = 124
        timed_out = True
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


def _ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                try:
                    return int(line.rsplit(" ", 1)[-1])
                except ValueError:
                    return None
            if line == "end_header":
                return None
    return None


def _compare_quality_frames(
    *,
    produced_render_dir: Path,
    reference_render_dir: Path,
    retained_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    retained_root.mkdir(parents=True, exist_ok=True)
    comparisons: list[dict[str, Any]] = []
    for index in QUALITY_FRAME_INDICES:
        name = f"{index:05d}.png"
        produced = produced_render_dir / name
        reference = reference_render_dir / name
        if not produced.is_file() or not reference.is_file():
            raise ValueError("aurafusion360_quality_frame_missing")
        produced_image = Image.open(produced).convert("RGB")
        reference_image = Image.open(reference).convert("RGB")
        if produced_image.size != reference_image.size:
            raise ValueError("aurafusion360_quality_frame_shape_mismatch")
        produced_array = np.asarray(produced_image, dtype=np.float64)
        reference_array = np.asarray(reference_image, dtype=np.float64)
        difference = produced_array - reference_array
        mean_absolute_error = float(np.mean(np.abs(difference)))
        mean_squared_error = float(np.mean(np.square(difference)))
        psnr_db = (
            None
            if mean_squared_error == 0.0
            else float(10.0 * math.log10((255.0**2) / mean_squared_error))
        )
        produced_retained = retained_root / f"produced_{name}"
        reference_retained = retained_root / f"publisher_reference_{name}"
        shutil.copy2(produced, produced_retained)
        shutil.copy2(reference, reference_retained)
        comparisons.append(
            {
                "frame_index": index,
                "width": produced_image.width,
                "height": produced_image.height,
                "produced": {
                    "relative_path": produced_retained.relative_to(
                        retained_root.parent.parent
                    ).as_posix(),
                    "size_bytes": produced_retained.stat().st_size,
                    "sha256": _sha256(produced_retained),
                },
                "publisher_reference": {
                    "relative_path": reference_retained.relative_to(
                        retained_root.parent.parent
                    ).as_posix(),
                    "size_bytes": reference_retained.stat().st_size,
                    "sha256": _sha256(reference_retained),
                },
                "mean_absolute_error_8bit": mean_absolute_error,
                "mean_squared_error_8bit": mean_squared_error,
                "psnr_db": psnr_db,
            }
        )
    return {
        "claim_ceiling": "same_camera_similarity_to_publisher_expected_point_cloud",
        "physical_or_hidden_surface_truth": False,
        "frame_indices": list(QUALITY_FRAME_INDICES),
        "frame_comparisons": comparisons,
        "mean_absolute_error_8bit": float(
            sum(item["mean_absolute_error_8bit"] for item in comparisons)
            / len(comparisons)
        ),
        "mean_psnr_db": (
            None
            if any(item["psnr_db"] is None for item in comparisons)
            else float(sum(item["psnr_db"] for item in comparisons) / len(comparisons))
        ),
    }


def _prepare_quality_reference_model(
    *,
    runtime: Path,
    working_output: Path,
    expected_point_cloud: Path,
) -> Path:
    reference_model = runtime / "quality_reference_model"
    if reference_model.exists():
        shutil.rmtree(reference_model)
    point_cloud_dir = (
        reference_model / "point_cloud/iteration_object_inpaint_init"
    )
    point_cloud_dir.mkdir(parents=True)
    os.link(expected_point_cloud.resolve(), point_cloud_dir / "point_cloud.ply")
    cfg_args = working_output / "cfg_args"
    if not cfg_args.is_file():
        raise ValueError("aurafusion360_quality_reference_cfg_args_missing")
    shutil.copy2(cfg_args, reference_model / "cfg_args")
    return reference_model


def _source_identity(source: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    changed: list[str] = []
    for record in manifest.get("source_files") or []:
        path = source / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["size_bytes"]
            or _sha256(path) != record["sha256"]
        ):
            changed.append(str(record["path"]))
    return {"matches": not changed, "changed_files": changed[:100]}


def _pin_hf_ref(cache: Path, repository: str, revision: str) -> None:
    repo_dir = cache / ("models--" + repository.replace("/", "--"))
    refs = repo_dir / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")


def _extract_author_data(runtime: Path, source: Path, spec: dict[str, Any]) -> None:
    data = spec["author_data"]
    archive = runtime / data["archive"]
    if not archive.is_file() or _sha256(archive) != data["archive_sha256"]:
        raise ValueError("aurafusion360_author_data_archive_changed")
    destination = source / "data"
    selected = destination / data["path_prefix"]
    if selected.is_dir():
        shutil.rmtree(selected)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source_archive:
        for member in source_archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError("aurafusion360_author_data_archive_path_traversal")
        source_archive.extractall(destination)
    expected_paths: set[str] = set()
    for item in data["files"]:
        relative = str(item["path"])
        path = destination / relative
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError("aurafusion360_author_data_materialized_bytes_changed")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in selected.rglob("*")
        if path.is_file()
    }
    expected_selected = {
        path for path in expected_paths if path.startswith(data["path_prefix"])
    }
    if actual_paths != expected_selected:
        raise ValueError("aurafusion360_author_data_materialized_file_set_changed")


def _extract_runtime_dependency(runtime: Path, spec: dict[str, Any]) -> Path:
    dependency = spec["wonderworld_marigold_runtime"]
    archive = runtime / dependency["archive"]
    if not archive.is_file() or _sha256(archive) != dependency["archive_sha256"]:
        raise ValueError("aurafusion360_wonderworld_runtime_archive_changed")
    destination = runtime / "runtime_dependencies"
    if destination.is_dir():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source_archive:
        for member in source_archive.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError("aurafusion360_wonderworld_runtime_archive_path_traversal")
        source_archive.extractall(destination)
    for item in dependency["source_files"]:
        path = destination / str(item["path"])
        if (
            not path.is_file()
            or path.stat().st_size != item["size_bytes"]
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError("aurafusion360_wonderworld_runtime_bytes_changed")
    return destination


def _verify_runtime_model_snapshot(snapshot: Path, model: dict[str, Any]) -> None:
    expected_files = model.get("materialized_files")
    if not expected_files:
        return
    expected_paths: set[str] = set()
    total_size_bytes = 0
    for item in expected_files:
        relative = str(item["path"])
        path = snapshot / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(item["size_bytes"])
            or _sha256(path) != item["sha256"]
        ):
            raise ValueError("aurafusion360_runtime_model_file_changed")
        expected_paths.add(relative)
        total_size_bytes += path.stat().st_size
    actual_paths = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("aurafusion360_runtime_model_file_set_changed")
    if total_size_bytes != int(model["materialized_total_size_bytes"]):
        raise ValueError("aurafusion360_runtime_model_total_size_changed")


def _materialize_cache_alias(
    *,
    cache: Path,
    source_snapshot: Path,
    model: dict[str, Any],
) -> Path:
    alias_of = str(model.get("cache_alias_of") or "")
    if not alias_of:
        raise ValueError("aurafusion360_runtime_model_cache_alias_missing")
    destination = (
        cache
        / ("models--" + str(model["repository"]).replace("/", "--"))
        / "snapshots"
        / str(model["revision"])
    )
    if destination.exists():
        raise ValueError("aurafusion360_runtime_model_cache_alias_destination_exists")
    for item in model.get("materialized_files") or []:
        relative = str(item["path"])
        source = source_snapshot / relative
        target = destination / relative
        if not source.is_file():
            raise ValueError("aurafusion360_runtime_model_cache_alias_source_missing")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source.resolve(), target)
    _verify_runtime_model_snapshot(destination, model)
    _pin_hf_ref(cache, str(model["repository"]), str(model["revision"]))
    return destination


def _prepare(runtime: Path, source: Path, spec: dict[str, Any]) -> int:
    from huggingface_hub import hf_hub_download, snapshot_download

    _extract_author_data(runtime, source, spec)
    _extract_runtime_dependency(runtime, spec)
    expected = spec["expected_output"]
    reference = runtime / expected["bundled_path"]
    if (
        not reference.is_file()
        or reference.stat().st_size != expected["expected_ply_size_bytes"]
        or _sha256(reference) != expected["expected_ply_sha256"]
    ):
        raise ValueError("aurafusion360_published_expected_ply_changed")

    cache = Path(os.environ["HF_HUB_CACHE"])
    resolved_snapshots: dict[str, Path] = {}
    for model in spec["runtime_models"]:
        materialized_files = model.get("materialized_files")
        alias_of = str(model.get("cache_alias_of") or "")
        if alias_of:
            source_snapshot = resolved_snapshots.get(alias_of)
            if source_snapshot is None:
                raise ValueError("aurafusion360_runtime_model_cache_alias_source_unresolved")
            snapshot = _materialize_cache_alias(
                cache=cache,
                source_snapshot=source_snapshot,
                model=model,
            )
        else:
            snapshot = Path(
                snapshot_download(
                    repo_id=model["repository"],
                    revision=model["revision"],
                    allow_patterns=(
                        [str(item["path"]) for item in materialized_files]
                        if materialized_files
                        else None
                    ),
                    max_workers=1,
                )
            )
            _pin_hf_ref(cache, model["repository"], model["revision"])
        _verify_runtime_model_snapshot(snapshot, model)
        resolved_snapshots[str(model["repository"])] = snapshot
    sd2 = spec["sd2_checkpoint"]
    checkpoint = hf_hub_download(
        repo_id=sd2["repository"],
        revision=sd2["revision"],
        filename=sd2["path"],
    )
    destination = source / "utils/LeftRefill/pretrained_models/512-inpainting-ema.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination)
    if (
        destination.stat().st_size != sd2["size_bytes"]
        or _sha256(destination) != sd2["sha256"]
    ):
        raise ValueError("aurafusion360_sd2_checkpoint_changed")

    working_output = source / "output/360-USID/sunflower"
    for relative in (
        "point_cloud/iteration_object_inpaint_init",
        "train/ours_object_inpaint_init",
        "test/ours_object_inpaint_init",
        "traj/ours_object_inpaint_init",
    ):
        path = working_output / relative
        if path.is_dir():
            shutil.rmtree(path)
    _write_json(runtime / "prepare_receipt.json", {"status": "prepared"})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    runtime = Path(__file__).resolve().parent
    source = runtime / "AuraFusion360_official"
    output = Path(
        os.environ.get("BLUEPRINT_ADP_AURA_OUTPUT_DIR", runtime.parent / "runtime_output")
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _read_json(runtime / "smoke_spec.json")
    if args.prepare_only:
        return _prepare(runtime, source, spec)

    runtime_dependencies = runtime / "runtime_dependencies"
    if not runtime_dependencies.is_dir():
        raise ValueError("aurafusion360_runtime_dependencies_missing")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(runtime_dependencies) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    source_before = _source_identity(source, spec)
    hardware = _run(
        ["nvidia-smi", "-q"],
        cwd=source,
        log_path=output / "nvidia-smi.log",
    )
    python = str(source / ".venv/bin/python")
    command_specs = [
        (
            "train",
            [python, "train.py", "--config", "configs/360-USID/sunflower/train.config"],
            "aura-train.log",
        ),
        (
            "render",
            [
                python, "render.py", "-s", "data/360-USID/sunflower", "-m",
                "output/360-USID/sunflower", "--skip_mesh", "--render_path",
                "--iteration", "30000",
            ],
            "aura-render.log",
        ),
        (
            "remove",
            [python, "remove.py", "--config", "configs/360-USID/sunflower/remove.config"],
            "aura-remove.log",
        ),
        (
            "sam2_masks",
            [python, "utils/sam2_utils.py", "--dataset", "360-USID", "--scene", "sunflower"],
            "aura-sam2-masks.log",
        ),
        (
            "inpaint_init",
            [python, "inpaint.py", "--config", "configs/360-USID/sunflower/inpaint.config"],
            "aura-inpaint-init.log",
        ),
    ]
    workflow: list[dict[str, Any]] = []
    for stage, command, log_name in command_specs:
        observed = _run(command, cwd=source, log_path=output / log_name)
        observed["stage"] = stage
        workflow.append(observed)
        if observed["returncode"] != 0:
            break
    execution = next(
        (row for row in workflow if row["stage"] == "inpaint_init"),
        {
            "stage": "inpaint_init",
            "command": command_specs[-1][1],
            "cwd": str(source),
            "returncode": None,
            "timed_out": False,
            "runtime_seconds": 0.0,
            "stdout_stderr_sha256": None,
            "log": "aura-inpaint-init.log",
            "executed": False,
        },
    )
    working_output = source / "output/360-USID/sunflower"
    produced = working_output / "point_cloud/iteration_object_inpaint_init/point_cloud.ply"
    expected = runtime / "published_expected_point_cloud.ply"
    quality_command: dict[str, Any] = {
        "command": None,
        "returncode": None,
        "timed_out": False,
        "runtime_seconds": 0.0,
        "stdout_stderr_sha256": None,
        "log": "aura-publisher-reference-render.log",
        "executed": False,
    }
    quality_comparison: dict[str, Any] | None = None
    if execution.get("returncode") == 0 and produced.is_file():
        try:
            reference_model = _prepare_quality_reference_model(
                runtime=runtime,
                working_output=working_output,
                expected_point_cloud=expected,
            )
            quality_command = _run(
                [
                    python,
                    "render.py",
                    "-s",
                    "data/360-USID/sunflower",
                    "-m",
                    str(reference_model),
                    "--skip_train",
                    "--skip_test",
                    "--skip_mesh",
                    "--render_path",
                    "--iteration",
                    "object_inpaint_init",
                ],
                cwd=source,
                log_path=output / "aura-publisher-reference-render.log",
            )
            quality_command["executed"] = True
            if quality_command["returncode"] == 0:
                quality_comparison = _compare_quality_frames(
                    produced_render_dir=(
                        working_output / "traj/ours_object_inpaint_init/renders"
                    ),
                    reference_render_dir=(
                        reference_model / "traj/ours_object_inpaint_init/renders"
                    ),
                    retained_root=output / "artifacts/quality_frames",
                )
        except (OSError, ValueError) as error:
            quality_command["error"] = str(error)
    source_after = _source_identity(source, spec)
    blockers: list[str] = []
    if not source_before["matches"] or not source_after["matches"]:
        blockers.append("aurafusion360_author_source_modified")
    completed_stages = {row["stage"] for row in workflow if row["returncode"] == 0}
    required_stages = ["train", "render", "remove", "sam2_masks", "inpaint_init"]
    for stage in required_stages:
        if stage not in completed_stages:
            blockers.append(f"aurafusion360_{stage}_command_failed_or_not_executed")
            break
    if hardware["returncode"] != 0:
        blockers.append("aurafusion360_nvidia_hardware_probe_failed")
    if not produced.is_file() or produced.stat().st_size == 0:
        blockers.append("aurafusion360_inpaint_init_point_cloud_missing")
    if "inpaint_init" in completed_stages and produced.is_file():
        if quality_command.get("returncode") != 0:
            blockers.append("aurafusion360_quality_reference_render_failed")
        elif quality_comparison is None:
            blockers.append("aurafusion360_quality_frame_comparison_missing")
    retained_produced = output / "artifacts/aurafusion360_inpaint_init_point_cloud.ply"
    if produced.is_file():
        retained_produced.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, retained_produced)
    freeze = output / "pip-freeze.txt"
    if not freeze.is_file() or freeze.stat().st_size == 0:
        blockers.append("aurafusion360_python_environment_receipt_missing")
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "completed" if not blockers else "blocked",
        "source_commit": spec["source_commit"],
        "source_tree": spec["source_tree"],
        "source_identity_before": source_before,
        "source_identity_after": source_after,
        "author_source_modified": not source_after["matches"],
        "author_workflow_commands": workflow,
        "author_command": execution,
        "hardware_probe": hardware,
        "training_executed": "train" in completed_stages,
        "render_executed": "render" in completed_stages,
        "removal_executed": "remove" in completed_stages,
        "sam2_masks_executed": "sam2_masks" in completed_stages,
        "inpaint_init_executed": "inpaint_init" in completed_stages and produced.is_file(),
        "published_expected_output_bound": expected.is_file()
        and _sha256(expected) == spec["expected_output"]["expected_ply_sha256"],
        "produced_point_cloud": (
            {
                "size_bytes": produced.stat().st_size,
                "sha256": _sha256(produced),
                "vertex_count": _ply_vertex_count(produced),
                "retained_relative_path": retained_produced.relative_to(output).as_posix(),
            }
            if produced.is_file()
            else None
        ),
        "published_expected_point_cloud": {
            "size_bytes": expected.stat().st_size,
            "sha256": _sha256(expected),
            "vertex_count": _ply_vertex_count(expected),
        },
        "quality_validation_command": quality_command,
        "quality_comparison": quality_comparison,
        "python_environment": (
            {"path": freeze.name, "sha256": _sha256(freeze)} if freeze.is_file() else None
        ),
        "depth_model": "prs-eth/marigold-depth-v1-0",
        "wonderworld_marigold_runtime": spec["wonderworld_marigold_runtime"],
        "depth_anything3_used": False,
        "retry_cap": 0,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
    }
    _write_json(output / "adp_aura_author_smoke_result.json", result)
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
