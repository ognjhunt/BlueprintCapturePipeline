"""Production-host CPU adapters for the typed SAM source-preparation stages.

The dispatcher owns jobs, roots, profiles and stage order. This module performs
no allocation, upload, model invocation, robot qualification or policy execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

from .decision_evidence_contracts import canonical_json
from .public_scene_removal_selection import (
    ADAPTER, materialize_public_scene_removal_selections,
    validate_removal_scene_selection, validate_removal_task_selection,
)
from .public_scene_inpainting_inputs import (
    build_public_scene_inpainting_input_request, materialize_public_scene_inpainting_inputs,
    prepare_public_scene_inpainting_inputs,
)
from .public_scene_sam31_task_inputs import materialize_public_scene_sam31_task_inputs
from .standard_splat_conversion import (
    build_standard_splat_conversion_request, materialize_standard_splat_conversion,
)
from .task_evaluation_scene_configuration_submission_inputs import beneath, checked_file, read, sha

STAGES = {"source_selections", "standard_splat_conversion", "calibrated_views", "sam31_inputs"}
RAW_ROLES = {
    "appearance_3dgs": "source_appearance", "semantic_metadata": "source_labels",
    "scene_structure": "source_structure", "collision_usd": "source_collision",
}


class Sam31PreparationCPUStageError(ValueError):
    """A server-derived CPU stage cannot be executed with its exact inputs."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise Sam31PreparationCPUStageError("sam31_preparation_cpu_" + code)


def _resident(path: str | Path, root: Path) -> Path:
    unresolved = Path(path)
    _require(not any(item.is_symlink() for item in (unresolved, *unresolved.parents)), "symlink_forbidden")
    result = unresolved.resolve()
    _require(result.is_relative_to(root), "path_outside_server_data_root")
    return result


def _input(records: Mapping[str, Any], name: str, root: Path) -> Path:
    row = records.get(name)
    _require(isinstance(row, Mapping), "input_missing:" + name)
    path = _resident(str(row.get("path") or ""), root)
    return checked_file(path, dict(row))


def _record(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "producer_artifact_missing")
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(dict(value)) + "\n")


def execute_cpu_stage(job: Mapping[str, Any], *, prepare_hardware_render: bool = False) -> dict[str, Any]:
    """Execute exactly one server-owned, no-spend source-preparation stage."""
    stage_id = job.get("stage_id")
    _require(stage_id in STAGES, "stage_invalid")
    plan, request = job.get("plan"), job.get("request")
    _require(isinstance(plan, Mapping) and isinstance(request, Mapping), "job_invalid")
    root = Path(job.get("server_data_root") or "/var/lib/blueprint/task-evaluation-inputs")
    _require(root.is_absolute() and root.is_dir(), "server_data_root_invalid")
    root = _resident(root, root.resolve())
    output = _resident(str(job.get("output_root") or ""), root)
    _require(output != root and not output.exists(), "output_not_fresh")
    repo = Path(str(job.get("repo_root") or ""))
    _require(repo.is_absolute() and repo.is_dir(), "repo_root_invalid")
    commit = str(request.get("expected_production_commit") or plan.get("source_commit") or "")
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "commit_invalid")
    _require(not plan.get("source_commit") or plan["source_commit"] == commit, "commit_mismatch")
    host = plan.get("host_inputs")
    inputs = job.get("inputs") or {}
    _require(isinstance(host, Mapping) and isinstance(inputs, Mapping), "input_inventory_invalid")
    artifacts: dict[str, Any] = {}

    if stage_id == "source_selections":
        paths = {key: _input(host, key, root) for key in (
            "task_request", "installation_receipt", "publisher_intake", "source_preparation_receipt",
        )}
        result = materialize_public_scene_removal_selections(
            task_request_path=paths["task_request"],
            installation_receipt_path=paths["installation_receipt"],
            publisher_intake_path=paths["publisher_intake"],
            source_preparation_receipt_path=paths["source_preparation_receipt"],
            expected_production_commit=commit, output_root=output,
        )
        artifacts.update({key: result[key] for key in ("scene_selection", "task_selection", "registered_frame")})
        installation = read(paths["installation_receipt"], digest_field="receipt_digest")
        for row in installation["files"]:
            if row.get("role") in RAW_ROLES:
                path = _resident(beneath(Path(installation["destination_root"]), row["relative_path"]), root)
                checked_file(path, row)
                artifacts[RAW_ROLES[row["role"]]] = _record(path)
        _require(set(RAW_ROLES.values()).issubset(artifacts), "installed_sources_missing")
    elif stage_id == "standard_splat_conversion":
        source = _input(inputs, "source_appearance", root)
        scene = validate_removal_scene_selection(read(_input(inputs, "scene_selection", root)))
        terms = _input(host, "interiorgs_terms", root)
        expected = scene["source_components"]["interiorgs"]
        _require(sha(source) == expected["sha256"] and source.stat().st_size == expected["size_bytes"],
                 "conversion_source_mismatch")
        runtime = _resident(str(job.get("runtime_root") or ""), root)
        _require(runtime.is_dir(), "runtime_root_invalid")
        conversion_request = build_standard_splat_conversion_request({
            "schema_version": "standard_splat_conversion_request.v1",
            "program_id": "arm-decision-proof-v1", "frozen_before_conversion": True,
            "learned_policy_outcomes_observed": False,
            "source": {"relative_path": source.relative_to(root).as_posix(),
                       "dataset": expected["repository"], "revision": expected["revision"],
                       "license": "retained_publisher_terms_bound_by_digest",
                       "sha256": expected["sha256"], "size_bytes": expected["size_bytes"]},
            "rights": {"conversion_execution_location": "local_only",
                       "raw_private_upload_authorized": False, "training_authorized": False,
                       "terms_digest": sha(terms)},
            "output_filename": "source_standard.ply",
        })
        output.mkdir(parents=True)
        request_path = output / "standard_splat_conversion_request.v1.json"
        _write(request_path, conversion_request)
        produced = output / "converted"
        receipt = materialize_standard_splat_conversion(
            request_path=request_path, repo_root=repo, data_root=root, output_root=produced,
            production_runtime_root=runtime,
        )
        _require(receipt.get("status") == "standard_splat_conversion_materialized"
                 and receipt.get("raw_source_uploaded") is False, "conversion_not_completed")
        standard = beneath(produced, receipt["output"]["relative_path"])
        checked_file(standard, receipt["output"])
        artifacts = {
            "conversion_request": _record(request_path),
            "standard_splat_conversion_receipt": _record(produced / "standard_splat_conversion_receipt.v1.json"),
            "standard_splat": _record(standard),
        }
    elif stage_id == "calibrated_views":
        names = ("scene_selection", "task_selection", "standard_splat_conversion_receipt",
                 "standard_splat", "source_labels", "source_structure", "registered_frame")
        paths = {key: _input(inputs, key, root) for key in names}
        validate_removal_task_selection(read(paths["task_selection"]))
        runtime = _resident(str(job.get("runtime_root") or ""), root)
        _require(runtime.is_dir(), "runtime_root_invalid")
        cameras = plan.get("camera_policy")
        _require(isinstance(cameras, Mapping) and len(cameras.get("views") or []) == 16,
                 "sixteen_distinct_calibrated_views_required")
        positions = [tuple(row.get("position_offset_m") or ()) for row in cameras["views"]]
        _require(len(set(positions)) == 16, "duplicate_calibrated_view_position")
        rendering = {
            "renderer": "reference_spark_renderer_exact_camera", "graphics_backend": "swiftshader",
            "width": 1280, "height": 1280, "vertical_fov_deg": 55.0,
            "warmup_ms": 2500, "settle_frames": 6, "settle_ms": 100, "timeout_seconds": 3600,
            **dict(plan.get("rendering") or {}),
        }
        render_request = build_public_scene_inpainting_input_request({
            "schema_version": "public_scene_interiorgs_edit_input_request.v2",
            "program_id": "arm-decision-proof-v1", "adp_item": "ADP-009D",
            "frozen_before_render": True, "method_outcomes_observed_before_freeze": False,
            "scene": {
                "source_adapter": ADAPTER, "scene_freeze_path": str(paths["scene_selection"]),
                "task_freeze_path": str(paths["task_selection"]),
                "standard_splat_conversion_receipt_path": str(paths["standard_splat_conversion_receipt"]),
                "standard_splat_path": str(paths["standard_splat"]),
                "labels_path": str(paths["source_labels"]), "structure_path": str(paths["source_structure"]),
                "registered_frame_receipt_path": str(paths["registered_frame"]),
            },
            "rendering": rendering, "camera_policy": dict(cameras),
            # These masks are calibration reconnaissance only. The final
            # cutout is selected later from reviewed SAM masks/contributions.
            "mask_policy": {"authority": "publisher_target_obb_plus_contained_gaussians",
                            "minimum_contained_gaussians": 16, "dilation_pixels": 0,
                            "support_threshold_8bit": 24, "minimum_support_inside_final_fraction": 0.99},
        })
        output.mkdir(parents=True)
        request_path = output / "public_scene_interiorgs_edit_input_request.v2.json"
        _write(request_path, render_request)
        produced = output / "views"
        if prepare_hardware_render:
            prepared = prepare_public_scene_inpainting_inputs(request_path=request_path, repo_root=repo,
                data_root=root, output_root=produced, production_runtime_root=runtime)
            return {"status": "prepared_for_hardware_render", "stage_id": stage_id,
                    "prepared_inputs": _record(Path(prepared["preparation_path"])),
                    "calibrated_view_request": _record(request_path), "source_commit": commit,
                    "provider_mutation_performed": False, "candidate_policy_queried": False}
        receipt = materialize_public_scene_inpainting_inputs(
            request_path=request_path, repo_root=repo, data_root=root, output_root=produced,
            production_runtime_root=runtime,
        )
        _require(receipt.get("status") == "render_derived_input_packet_materialized",
                 "calibrated_views_not_completed")
        artifacts = {
            "calibrated_view_request": _record(request_path),
            "calibrated_view_receipt": _record(produced / "public_scene_interiorgs_edit_input_receipt.v2.json"),
            "camera_contract": _record(beneath(produced, receipt["derived_artifacts"]["cameras"]["relative_path"])),
        }
    else:
        task = read(_input(host, "task_request", root))
        selection = _input(inputs, "task_selection", root)
        validated = validate_removal_task_selection(read(selection))
        _require(validated["task_id"] == task["task_identity"]["id"], "task_selection_mismatch")
        label = str(task["subject"].get("review_label") or "").replace("_", " ").strip()
        _require(bool(label), "subject_prompt_missing")
        profile = _input(inputs, "sam31_provider_profile", root)
        calibrated = _input(inputs, "calibrated_view_receipt", root)
        output.mkdir(parents=True)
        prompts = output / "source_subject_prompts.json"
        _write_prompts = [{"prompt_id": "selected-source-subject", "text": label, "output_label": label}]
        with prompts.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(_write_prompts) + "\n")
        produced = output / "sam31-inputs"
        receipt = materialize_public_scene_sam31_task_inputs(
            calibrated_view_receipt_path=calibrated, task_freeze_path=selection,
            provider_profile_path=profile, prompts_path=prompts, output_root=produced,
            ffmpeg_executable=job.get("ffmpeg_executable"),
        )
        _require(receipt.get("status") == "prepared_no_upload_no_execution"
                 and receipt.get("paid_execution_started") is False, "sam31_inputs_not_completed")
        artifacts = {
            "sam31_prompts": _record(prompts),
            "sam31_task_input_packet": _record(produced / "public_scene_sam31_task_input_packet.v1.json"),
            "sam31_run_request": _record(beneath(produced, receipt["run_request"]["relative_path"])),
        }
    return {"status": "completed", "stage_id": stage_id, "source_commit": commit,
            "artifacts": artifacts, "provider_mutation_performed": False,
            "candidate_policy_queried": False, "evaluation_authorized": False}
