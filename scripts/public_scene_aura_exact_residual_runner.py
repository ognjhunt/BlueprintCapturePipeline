#!/usr/bin/env python3
"""Execute no-finetune Aura residual initialization inside a sealed bundle.

The shell installs the released software boundary.  This runner verifies the
bundle request again, materializes the exact masks, calls only train/remove/
inpaint from the released Aura tree, and retains raw per-task native frames.
It intentionally leaves provider allocation and provider-zero evidence to the
outer paid-resource adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping

RESULT_NAME = "public_scene_aura_exact_residual_runtime_result.json"
REQUEST_NAME = "aura_exact_residual_runtime_request.json"
REQUEST_SCHEMA = "public_scene_aura_exact_residual_runtime_request.v1"
RUNTIME_SCHEMA = "public_scene_aura_exact_residual_runtime_result.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bound(runtime: Path, record: Any, *, code: str) -> Path:
    if not isinstance(record, Mapping):
        raise ValueError(code)
    relative = str(record.get("relative_path") or "")
    if not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError(code)
    path = (runtime / relative).resolve()
    if (
        runtime not in path.parents
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise ValueError(code)
    return path


def _load_request(runtime: Path) -> dict[str, Any]:
    path = runtime / "aura_exact_residual_runtime_request.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("aura_exact_residual_runtime_request_invalid")
    expected = value.get("request_digest")
    copy = dict(value)
    copy.pop("request_digest", None)
    canonical = json.dumps(copy, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if expected != digest:
        raise ValueError("aura_exact_residual_runtime_request_digest_invalid")
    if (
        value.get("private_derived_upload_only") is not True
        or value.get("raw_dataset_bytes_included") is not False
        or value.get("provider_training_authorized") is not False
        or value.get("scene_fitting_is_provider_training") is not False
        or value.get("automatic_paid_retry_allowed") is not False
        or value.get("learned_policy_outcomes_accessed") is not False
    ):
        raise ValueError("aura_exact_residual_runtime_authority_invalid")
    _bound(runtime, value.get("preflight"), code="aura_exact_residual_runtime_preflight_changed")
    _bound(runtime, value.get("backend_admission"), code="aura_exact_residual_runtime_backend_changed")
    _bound(runtime, value.get("shared_retained_scene"), code="aura_exact_residual_runtime_ply_changed")
    _bound(runtime, value.get("big_lama_checkpoint"), code="aura_exact_residual_runtime_big_lama_changed")
    return value


def _run(
    command: list[str], *, cwd: Path, output: Path, stage: str, environment: Mapping[str, str]
) -> dict[str, Any]:
    log = output / "logs" / f"{stage}.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=21600,
        )
    return {"stage": stage, "command": command, "returncode": completed.returncode, "log": _record(log, output)}


def _ply_vertex_count(path: Path) -> int:
    """Read only the declared PLY vertex count; do not reinterpret Aura geometry."""

    try:
        with path.open("rb") as stream:
            for raw in stream:
                line = raw.decode("ascii", errors="strict").strip()
                if line.startswith("element vertex "):
                    count = int(line.rsplit(" ", 1)[-1])
                    if count > 0:
                        return count
                if line == "end_header":
                    break
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("aura_exact_residual_runtime_native_point_cloud_invalid") from exc
    raise ValueError("aura_exact_residual_runtime_native_point_cloud_invalid")


def _pin_hf_ref(cache: Path, repository: str, revision: str) -> None:
    refs = cache / ("models--" + repository.replace("/", "--")) / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(revision, encoding="utf-8")


def _verify_model_snapshot(snapshot: Path, model: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = model.get("materialized_files")
    if not isinstance(expected, list) or not expected:
        raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
    records: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    total = 0
    for row in expected:
        if not isinstance(row, Mapping):
            raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
        name = str(row.get("path") or "")
        if not name or name.startswith("/") or ".." in Path(name).parts or name in expected_names:
            raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
        path = snapshot / name
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or _sha256(path) != row.get("sha256")
        ):
            raise ValueError("aura_exact_residual_runtime_marigold_model_bytes_changed")
        expected_names.add(name)
        total += path.stat().st_size
        records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        # Hugging Face snapshots normally use links into its content-addressed
        # blob store.  Hash the resolved bytes; link identity is not evidence.
        if path.is_file()
    }
    if actual != expected_names or total != model.get("materialized_total_size_bytes"):
        raise ValueError("aura_exact_residual_runtime_marigold_model_file_set_changed")
    return records


def _materialize_cache_alias(
    *, cache: Path, source_snapshot: Path, model: Mapping[str, Any]
) -> Path:
    alias_of = str(model.get("cache_alias_of") or "")
    repository = str(model.get("repository") or "")
    revision = str(model.get("revision") or "")
    if not alias_of or not repository or not revision:
        raise ValueError("aura_exact_residual_runtime_marigold_cache_alias_invalid")
    destination = cache / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    if destination.exists():
        raise ValueError("aura_exact_residual_runtime_marigold_cache_alias_conflict")
    for row in model.get("materialized_files") or []:
        name = str(row.get("path") or "")
        source = source_snapshot / name
        target = destination / name
        if not source.is_file():
            raise ValueError("aura_exact_residual_runtime_marigold_cache_alias_source_missing")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source.resolve(), target)
    _verify_model_snapshot(destination, model)
    _pin_hf_ref(cache, repository, revision)
    return destination


def _prepare_marigold_runtime(runtime: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    """Download one pinned released model then run Aura fully offline.

    This is intentionally provider-side released-model acquisition, not an
    upload of scene material.  The exact files are checked before the runtime
    environment is switched to the offline flags consumed by diffusers.
    """

    models = request.get("marigold_runtime_models")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
    cache = runtime / ".hf_home" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError("aura_exact_residual_runtime_huggingface_client_missing") from exc
    for model in models:
        if not isinstance(model, Mapping):
            raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
        repository = str(model.get("repository") or "")
        revision = str(model.get("revision") or "")
        if repository not in {"prs-eth/marigold-depth-v1-0", "prs-eth/marigold-v1-0"} or len(revision) != 40:
            raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
        alias_of = str(model.get("cache_alias_of") or "")
        if alias_of:
            source_snapshot = resolved.get(alias_of)
            if source_snapshot is None:
                raise ValueError("aura_exact_residual_runtime_marigold_cache_alias_unresolved")
            snapshot = _materialize_cache_alias(
                cache=cache, source_snapshot=source_snapshot, model=model
            )
        else:
            names = [str(row.get("path") or "") for row in model.get("materialized_files") or []]
            if not names or any(not name for name in names):
                raise ValueError("aura_exact_residual_runtime_marigold_model_spec_invalid")
            snapshot = Path(
                snapshot_download(
                    repo_id=repository,
                    revision=revision,
                    cache_dir=str(cache),
                    allow_patterns=names,
                    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                )
            )
            _pin_hf_ref(cache, repository, revision)
        files = _verify_model_snapshot(snapshot, model)
        resolved[repository] = snapshot
        records.append(
            {
                "repository": repository,
                "revision": revision,
                "snapshot_digest": model.get("snapshot_digest"),
                "cache_alias_of": alias_of or None,
                "files": files,
                "verified_before_offline_execution": True,
            }
        )
    return {"cache_root": str(cache.relative_to(runtime)), "models": records}


def _runtime_environment(runtime: Path, request: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    dependency = request.get("wonderworld_marigold_runtime")
    if not isinstance(dependency, Mapping) or dependency.get("license") != "Apache-2.0":
        raise ValueError("aura_exact_residual_runtime_wonderworld_binding_invalid")
    files = dependency.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("aura_exact_residual_runtime_wonderworld_binding_invalid")
    for record in files:
        _bound(runtime, record, code="aura_exact_residual_runtime_wonderworld_bytes_changed")
    prepared = _prepare_marigold_runtime(runtime, request)
    environment = dict(os.environ)
    dependencies = runtime / "runtime_dependencies"
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(dependencies) + (os.pathsep + inherited if inherited else "")
    environment["HF_HOME"] = str(runtime / ".hf_home")
    environment["HF_HUB_CACHE"] = str(runtime / ".hf_home" / "hub")
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return environment, {"wonderworld_helper_files_verified": len(files), "marigold": prepared}


def _composite_exact(*, before: Path, raw: Path, mask: Path, destination: Path) -> dict[str, Any]:
    # Kept inside the paid/GPU branch so the no-cost shell rehearsal needs only
    # the standard library available in the sealed base image.
    import numpy as np
    from PIL import Image

    with Image.open(before) as value:
        before_pixels = np.asarray(value.convert("RGB"), dtype=np.uint8)
    with Image.open(raw) as value:
        raw_pixels = np.asarray(value.convert("RGB"), dtype=np.uint8)
    with Image.open(mask) as value:
        mask_pixels = np.asarray(value.convert("L"), dtype=np.uint8)
    if before_pixels.shape != raw_pixels.shape or mask_pixels.shape != before_pixels.shape[:2]:
        raise ValueError("aura_exact_residual_runtime_reference_dimensions_invalid")
    result = before_pixels.copy()
    result[mask_pixels > 0] = raw_pixels[mask_pixels > 0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(destination)
    outside = np.any(result != before_pixels, axis=2) & ~(mask_pixels > 0)
    if bool(outside.any()):
        raise ValueError("aura_exact_residual_runtime_reference_outside_mask_changed")
    return {"outside_mask_changed_pixels": int(outside.sum())}


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _sha256(source) != _sha256(destination):
        raise ValueError("aura_exact_residual_runtime_copy_digest_mismatch")


def _bind_shared_removal_checkpoint(
    *, runtime: Path, task_id: str, expected_sha256: str
) -> dict[str, Any]:
    """Prove a task reads the one shared Aura removal output without copying it."""

    source = (
        runtime
        / "work"
        / "model"
        / "point_cloud"
        / "iteration_30000_object_removal"
        / "point_cloud.ply"
    )
    if (
        not source.is_file()
        or source.is_symlink()
        or _sha256(source) != expected_sha256
    ):
        raise ValueError("aura_exact_residual_runtime_shared_removal_checkpoint_invalid")
    return {
        "task_id": task_id,
        "shared_removal_point_cloud": _record(source, runtime),
        "task_reads_one_shared_removal_output_without_task_specific_removal": True,
    }


def _extract_big_lama(*, runtime: Path, archive: Path) -> Path:
    """Extract the sealed released checkpoint without accepting ZIP traversal."""

    destination = runtime / "big-lama"
    if destination.exists():
        raise ValueError("aura_exact_residual_runtime_big_lama_destination_exists")
    try:
        with zipfile.ZipFile(archive) as payload:
            for member in payload.infolist():
                member_path = Path(member.filename)
                if (
                    not member.filename
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                    or not member.filename.startswith("big-lama/")
                    or (member.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("aura_exact_residual_runtime_big_lama_archive_invalid")
            payload.extractall(runtime)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("aura_exact_residual_runtime_big_lama_archive_invalid") from exc
    if (
        not destination.is_dir()
        or destination.is_symlink()
        or not (destination / "config.yaml").is_file()
        or not (destination / "models" / "best.ckpt").is_file()
    ):
        raise ValueError("aura_exact_residual_runtime_big_lama_archive_invalid")
    return destination


def _workspaces(runtime: Path, request: Mapping[str, Any]) -> list[dict[str, Any]]:
    scene_root = runtime / "data" / "Other-360" / "shared_retained_scene"
    plans = request.get("task_plans")
    cameras = request.get("camera_inputs")
    if not isinstance(plans, list) or not isinstance(cameras, list) or not plans or not cameras:
        raise ValueError("aura_exact_residual_runtime_task_plan_invalid")
    staged = {str(row.get("staged_name")): row for row in cameras if isinstance(row, Mapping)}
    if len(staged) != len(cameras):
        raise ValueError("aura_exact_residual_runtime_camera_plan_invalid")
    result = []
    for plan in plans:
        if not isinstance(plan, Mapping):
            raise ValueError("aura_exact_residual_runtime_task_plan_invalid")
        task_id = str(plan.get("task_id") or "")
        workspace_text = str(plan.get("workspace") or "")
        if not task_id or not workspace_text or workspace_text.startswith("/") or ".." in Path(workspace_text).parts:
            raise ValueError("aura_exact_residual_runtime_task_plan_invalid")
        workspace = runtime / workspace_text
        task_scene = runtime / "data" / "Other-360" / f"shared_retained_scene--{task_id}"
        for source_name in ("sparse", "images"):
            source = scene_root / source_name
            target = task_scene / source_name
            if target.exists():
                raise ValueError("aura_exact_residual_runtime_task_scene_conflict")
            shutil.copytree(source, target)
        for name in ("unseen_masks", "reference"):
            (task_scene / name).mkdir(parents=True, exist_ok=True)
        for mask in sorted((workspace / "unseen_masks").glob("*.png")):
            _copy_exact(mask, task_scene / "unseen_masks" / mask.name)
        staged_name = str(plan.get("reference_staged_name") or "")
        reference_row = staged.get(staged_name)
        if reference_row is None:
            raise ValueError("aura_exact_residual_runtime_reference_missing")
        input_dir = workspace / "reference_lama_input"
        result.append({"plan": dict(plan), "task_scene": task_scene, "workspace": workspace, "reference": reference_row, "input_dir": input_dir})
    return result


def run(*, runtime: Path, output: Path, rehearsal: bool) -> dict[str, Any]:
    request = _load_request(runtime)
    if rehearsal:
        return {
            "schema_version": "provider_bundle_rehearsal.v1",
            "status": "passed",
            "aura_released_code_executed": False,
            "gpu_runtime_started": False,
            "paid_inference_performed": False,
            "provider_mutations_performed": 0,
            "verified_camera_count": len(request["camera_inputs"]),
            "verified_task_count": len(request["task_plans"]),
            "blockers": [],
        }
    source = runtime / "AuraFusion360_official"
    lama = runtime / "LaMa"
    for path in (source / "train.py", source / "remove.py", source / "inpaint.py", lama / "bin" / "predict.py"):
        if not path.is_file() or path.is_symlink():
            raise ValueError("aura_exact_residual_runtime_released_entrypoint_missing")
    aura_python = Path(sys.executable)
    lama_python = runtime / ".lama-venv" / "bin" / "python"
    if not lama_python.is_file():
        raise ValueError("aura_exact_residual_runtime_lama_python_missing")
    big_lama = _extract_big_lama(
        runtime=runtime,
        archive=_bound(
            runtime,
            request.get("big_lama_checkpoint"),
            code="aura_exact_residual_runtime_big_lama_changed",
        ),
    )
    environment, runtime_dependencies = _runtime_environment(runtime, request)
    lama_environment = dict(environment)
    lama_environment["PYTHONPATH"] = str(lama) + os.pathsep + environment["PYTHONPATH"]
    workflows: list[dict[str, Any]] = []
    train = _run([str(aura_python), "train.py", "--config", "../configs/train.config"], cwd=source, output=output, stage="train_shared_retained_scene", environment=environment)
    workflows.append(train)
    if train["returncode"] != 0:
        raise ValueError("aura_exact_residual_runtime_train_failed")
    remove = _run([str(aura_python), "remove.py", "--config", "../configs/remove.config"], cwd=source, output=output, stage="remove_shared_exact_mask_association", environment=environment)
    workflows.append(remove)
    if remove["returncode"] != 0:
        raise ValueError("aura_exact_residual_runtime_remove_failed")
    shared_removal = runtime / "work" / "model" / "point_cloud" / "iteration_30000_object_removal" / "point_cloud.ply"
    if not shared_removal.is_file() or shared_removal.is_symlink():
        raise ValueError("aura_exact_residual_runtime_shared_removal_checkpoint_missing")
    shared_removal_record = _record(shared_removal, runtime)
    task_outputs: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    task_removal_bindings: list[dict[str, Any]] = []
    for state in _workspaces(runtime, request):
        plan = state["plan"]
        task_id = str(plan["task_id"])
        shared_removal_binding = _bind_shared_removal_checkpoint(
            runtime=runtime,
            task_id=task_id,
            expected_sha256=shared_removal_record["sha256"],
        )
        reference = state["reference"]
        reference_name = str(plan["reference_staged_name"])
        raw_reference_dir = output / "big_lama_raw" / task_id
        lama_command = [
            str(lama_python),
            str(lama / "bin" / "predict.py"),
            f"model.path={big_lama}",
            f"indir={state['input_dir']}",
            f"outdir={raw_reference_dir}",
        ]
        completed = _run(lama_command, cwd=lama, output=output, stage=f"big_lama_{task_id}", environment=lama_environment)
        workflows.append(completed)
        if completed["returncode"] != 0:
            raise ValueError("aura_exact_residual_runtime_big_lama_failed")
        raw_reference = raw_reference_dir / f"{reference_name}_mask.png"
        before = _bound(runtime, reference.get("retained_scene_before"), code="aura_exact_residual_runtime_reference_before_changed")
        mask = _bound(runtime, reference.get("exact_residual_mask"), code="aura_exact_residual_runtime_reference_mask_changed")
        composite_reference = state["task_scene"] / "reference" / f"{reference_name}.png"
        _composite_exact(before=before, raw=raw_reference, mask=mask, destination=composite_reference)
        config = runtime / "configs" / f"inpaint_{task_id}.config"
        inpaint = _run([str(aura_python), "inpaint.py", "--config", str(config)], cwd=source, output=output, stage=f"inpaint_init_{task_id}", environment=environment)
        workflows.append(inpaint)
        if inpaint["returncode"] != 0:
            raise ValueError("aura_exact_residual_runtime_inpaint_failed")
        # The released entrypoint must leave the common removal output intact;
        # its per-task native candidate is written only under its experiment.
        if _sha256(shared_removal) != shared_removal_record["sha256"]:
            raise ValueError("aura_exact_residual_runtime_shared_removal_checkpoint_changed")
        shared_removal_binding["shared_removal_unchanged_after_task_initialization"] = True
        task_removal_bindings.append(shared_removal_binding)
        ply = runtime / "work" / str(plan["output_experiment"]) / "point_cloud" / "iteration_object_inpaint_init" / "point_cloud.ply"
        if not ply.is_file() or ply.is_symlink() or ply.stat().st_size <= 0:
            raise ValueError("aura_exact_residual_runtime_native_point_cloud_missing")
        native = output / "native_task_outputs" / task_id / "point_cloud.ply"
        _copy_exact(ply, native)
        rows = [row for row in request["camera_inputs"] if row["task_id"] == task_id]
        native_count = _ply_vertex_count(native)
        rendered = runtime / "work" / str(plan["output_experiment"]) / "train" / "ours_object_inpaint_init" / "renders"
        candidates = sorted(rendered.glob("*.png"))
        sorted_rows = sorted(request["camera_inputs"], key=lambda item: item["staged_name"])
        if len(candidates) != len(sorted_rows):
            raise ValueError("aura_exact_residual_runtime_native_frame_set_invalid")
        for row in rows:
            # Aura's own inpaint entrypoint exported its full camera set before this point.
            # The raw frame mapping is retained from that exact release output by staged name.
            index = [item["staged_name"] for item in sorted_rows].index(row["staged_name"])
            if index >= len(candidates):
                raise ValueError("aura_exact_residual_runtime_native_frame_missing")
            frame = output / "native_task_outputs" / task_id / "frames" / f"{row['camera_id']}.png"
            _copy_exact(candidates[index], frame)
            frames.append({"task_id": task_id, "camera_id": row["camera_id"], "native_aura_frame": _record(frame, output), "native_aura_point_cloud_sha256": _sha256(native)})
        task_outputs.append({
            "task_id": task_id,
            "native_aura_point_cloud": _record(native, output),
            "native_aura_point_cloud_sha256": _sha256(native),
            "native_aura_gaussian_count": native_count,
            "native_aura_representation": "aura_2d_gaussian_surfels_scale_0_scale_1",
            "render_camera_ids": list(plan["review_camera_ids"]),
        })
    return {
        "schema_version": RUNTIME_SCHEMA,
        "status": "completed",
        "preflight_digest": request["preflight"]["preflight_digest"],
        "aura_inpainting_executed": True,
        "provider_mutations_performed": 0,
        "learned_policy_outcomes_accessed": False,
        "released_entrypoints": ["train.py", "remove.py", "inpaint.py"],
        "excluded_stock_entrypoints": ["utils/sam2_utils.py", "utils/LeftRefill/sdedit_utils.py"],
        "workflows": workflows,
        "shared_removal_point_cloud": shared_removal_record,
        "task_shared_removal_bindings": task_removal_bindings,
        "task_outputs": task_outputs,
        "frames": frames,
        "runtime_dependencies": runtime_dependencies,
        "provider_zero_required_after_return": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rehearsal", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()
    output = args.output_dir.resolve()
    try:
        result = run(runtime=runtime, output=output, rehearsal=args.rehearsal)
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema_version": RUNTIME_SCHEMA,
            "status": "blocked",
            "blockers": [f"aura_exact_residual_runtime_exception:{type(exc).__name__}", str(exc)],
            "aura_inpainting_executed": False,
            "provider_mutations_performed": 0,
            "learned_policy_outcomes_accessed": False,
            "provider_zero_required_after_return": True,
        }
    name = "provider_bundle_rehearsal.json" if args.rehearsal else RESULT_NAME
    _write(output / name, result)
    return 0 if result.get("status") in {"passed", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
