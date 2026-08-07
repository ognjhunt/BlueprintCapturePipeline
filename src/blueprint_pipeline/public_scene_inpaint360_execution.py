"""Seal the observed Inpaint360GS InteriorGS v3 run without self-admission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_inpaint360_adapter import SCHEMA_VERSION as ADAPTER_SCHEMA_VERSION


SCHEMA_VERSION = "adp009b_inpaint360_interiorgs_execution_receipt.v1"
RUNTIME_SCHEMA_VERSION = "adp_inpaint360_interiorgs_result.v1"
RUN_SCHEMA_VERSION = "adp_inpaint360_interiorgs_vast_run.v1"
SELECTION_SCHEMA_VERSION = "inpaint360_supplemental_fusion_view_selection.v1"
GAUSSIAN_BUDGET_SCHEMA_VERSION = "inpaint360_added_gaussian_budget_validation.v1"
RENDER_SCHEMA_VERSION = "sealed_camera_render_manifest.v1"
LOCALITY_SCHEMA_VERSION = "public_scene_inpainting_locality_measurement.v1"
QUALITY_BLOCKER = "inpaint360_interiorgs_visual_artifact_rejection"
REQUIRED_STAGES = (
    "method_resolution_contract",
    "pre_registered_mask_binding",
    "distillation",
    "baseline_render",
    "removal",
    "virtual_views",
    "virtual_masks",
    "lama_color",
    "lama_depth",
    "ply_fusion",
    "inpaint_3d",
)


class Inpaint360ExecutionReceiptError(ValueError):
    """A deterministic Inpaint360 execution-evidence validation failure."""

    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: str | Path, roots: Sequence[Path], code: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise Inpaint360ExecutionReceiptError([code])
    return resolved


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Inpaint360ExecutionReceiptError([code]) from exc
    if not isinstance(value, dict):
        raise Inpaint360ExecutionReceiptError([code])
    return value


def _file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if root is None:
        record["path"] = str(path)
    else:
        record["relative_path"] = path.relative_to(root).as_posix()
    return record


def _verify_runtime_record(
    record: Mapping[str, Any], *, root: Path, code: str
) -> tuple[Path, dict[str, Any]]:
    relative = str(record.get("relative_path") or "")
    path = (root / relative).resolve()
    if not relative or (path != root and root not in path.parents):
        raise Inpaint360ExecutionReceiptError([code])
    if (
        not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256(path) != record.get("sha256")
    ):
        raise Inpaint360ExecutionReceiptError([code])
    return path, _file_record(path, root)


def _verify_teardown(run: Mapping[str, Any], evidence: Path) -> tuple[Path, dict[str, Any]]:
    teardown_path = _under(
        str(run.get("teardown_manifest_path") or ""),
        (evidence,),
        "inpaint360_teardown_manifest_outside_evidence_root",
    )
    teardown = _read_object(teardown_path, "inpaint360_teardown_manifest_invalid")
    instance_ids = teardown.get("vast_instance_ids")
    actions = teardown.get("teardown_actions_performed")
    destroyed_ids = {
        row.get("instance_id")
        for row in actions or []
        if isinstance(row, Mapping)
        and row.get("action") == "destroy_instance"
        and row.get("status") == "completed"
        and isinstance(row.get("http_status_code"), int)
        and 200 <= row["http_status_code"] < 300
    }
    if (
        teardown.get("schema_version") != "vast_teardown_manifest.v1"
        or teardown.get("status") != "completed"
        or teardown.get("continuing_spend_from_this_run") is not False
        or teardown.get("runner_gpu_teardown_completed") is not True
        or not isinstance(instance_ids, list)
        or not instance_ids
        or set(instance_ids) != destroyed_ids
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_provider_zero_not_proven"]
        )
    return teardown_path, teardown


def _verify_render_manifest(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_camera_ids: Sequence[str],
    final_ply_sha256: str,
) -> list[dict[str, Any]]:
    if (
        manifest.get("schema_version") != RENDER_SCHEMA_VERSION
        or manifest.get("status") != "rendered_exact_cameras"
        or canonical_digest(
            manifest, digest_field="sealed_camera_render_manifest_digest"
        )
        != manifest.get("sealed_camera_render_manifest_digest")
        or manifest.get("rendered_by") != "reference_spark_renderer_exact_camera"
        or (manifest.get("renderer_identity") or {}).get("graphics_backend") != "metal"
        or manifest.get("hidden_pixels_read_by_renderer") is not False
        or manifest.get("rendered_by_isaac_rtx") is not False
        or manifest.get("splat_digest") != final_ply_sha256
        or manifest.get("provider_splat_import_receipt_digest") != final_ply_sha256
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_exact_metal_render_manifest_invalid"]
        )
    rows = manifest.get("renders")
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected_camera_ids)
        or manifest.get("render_count") != len(rows)
        or [str(row.get("camera_id") or "") for row in rows]
        != list(expected_camera_ids)
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_exact_metal_render_camera_set_mismatch"]
        )
    root = manifest_path.parent
    verified: list[dict[str, Any]] = []
    for row in rows:
        relative = str(row.get("relative_path") or "")
        frame = (root / relative).resolve()
        if (
            not relative
            or (frame != root and root not in frame.parents)
            or not frame.is_file()
            or _sha256(frame) != row.get("digest")
        ):
            raise Inpaint360ExecutionReceiptError(
                ["inpaint360_exact_metal_render_frame_changed"]
            )
        with Image.open(frame) as image:
            width, height = image.size
        if width != row.get("width") or height != row.get("height"):
            raise Inpaint360ExecutionReceiptError(
                ["inpaint360_exact_metal_render_frame_dimensions_changed"]
            )
        verified.append(
            {
                "camera_id": row["camera_id"],
                "relative_path": relative,
                "size_bytes": frame.stat().st_size,
                "sha256": _sha256(frame),
                "width": width,
                "height": height,
            }
        )
    return verified


def _verify_locality(
    *,
    locality: Mapping[str, Any],
    locality_path: Path,
    render_manifest: Mapping[str, Any],
    render_manifest_path: Path,
) -> None:
    render_rows = render_manifest.get("renders") or []
    locality_rows = locality.get("rows") or []
    render_by_camera = {
        str(row.get("camera_id") or ""): row.get("digest") for row in render_rows
    }
    locality_by_camera = {
        str(row.get("camera_id") or ""): row.get("after_sha256")
        for row in locality_rows
        if isinstance(row, Mapping)
    }
    aggregate = locality.get("aggregate") or {}
    if (
        locality.get("schema_version") != LOCALITY_SCHEMA_VERSION
        or locality.get("status") != "measured_no_admission_effect"
        or canonical_digest(locality, digest_field="locality_measurement_digest")
        != locality.get("locality_measurement_digest")
        or locality.get("after_render_manifest_sha256") != _sha256(render_manifest_path)
        or locality.get("after_render_manifest_digest")
        != render_manifest.get("sealed_camera_render_manifest_digest")
        or locality.get("quality_pass_claimed") is not False
        or locality.get("admission_effect") != "none"
        or locality.get("thresholds_frozen_before_evaluation") is not False
        or render_by_camera != locality_by_camera
        or aggregate.get("view_count") != len(render_rows)
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_outside_mask_locality_measurement_invalid"]
        )
    if not locality_path.is_file():
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_outside_mask_locality_measurement_invalid"]
        )


def materialize_inpaint360_execution_receipt(
    *,
    adapter_receipt_path: str | Path,
    runtime_result_path: str | Path,
    run_result_path: str | Path,
    selected_view_receipt_path: str | Path,
    gaussian_budget_receipt_path: str | Path,
    metal_render_manifest_path: str | Path,
    locality_measurement_path: str | Path,
    evidence_root: str | Path,
    repo_root: str | Path,
    receipt_output: str | Path | None = None,
) -> dict[str, Any]:
    """Derive one rejected-quality receipt from exact observed v3 evidence."""

    evidence = Path(evidence_root).expanduser().resolve()
    repo = Path(repo_root).expanduser().resolve()
    if not evidence.is_dir() or not repo.is_dir():
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_execution_allowlisted_root_missing"]
        )
    roots = (evidence, repo)
    adapter_path = _under(
        adapter_receipt_path,
        roots,
        "inpaint360_adapter_receipt_outside_allowlisted_roots",
    )
    runtime_path = _under(
        runtime_result_path,
        (evidence,),
        "inpaint360_runtime_result_outside_evidence_root",
    )
    run_path = _under(
        run_result_path, (evidence,), "inpaint360_run_result_outside_evidence_root"
    )
    selection_path = _under(
        selected_view_receipt_path,
        (evidence,),
        "inpaint360_selected_view_receipt_outside_evidence_root",
    )
    budget_path = _under(
        gaussian_budget_receipt_path,
        (evidence,),
        "inpaint360_gaussian_budget_receipt_outside_evidence_root",
    )
    render_path = _under(
        metal_render_manifest_path,
        (evidence,),
        "inpaint360_metal_render_manifest_outside_evidence_root",
    )
    locality_path = _under(
        locality_measurement_path,
        (evidence,),
        "inpaint360_locality_measurement_outside_evidence_root",
    )
    adapter = _read_object(adapter_path, "inpaint360_adapter_receipt_invalid")
    runtime = _read_object(runtime_path, "inpaint360_runtime_result_invalid")
    run = _read_object(run_path, "inpaint360_run_result_invalid")
    selection = _read_object(selection_path, "inpaint360_selected_view_receipt_invalid")
    budget = _read_object(budget_path, "inpaint360_gaussian_budget_receipt_invalid")
    render = _read_object(render_path, "inpaint360_metal_render_manifest_invalid")
    locality = _read_object(locality_path, "inpaint360_locality_measurement_invalid")

    if (
        adapter.get("schema_version") != ADAPTER_SCHEMA_VERSION
        or adapter.get("status") != "prepared_unexecuted"
        or canonical_digest(adapter, digest_field="receipt_digest")
        != adapter.get("receipt_digest")
    ):
        raise Inpaint360ExecutionReceiptError(["inpaint360_adapter_receipt_invalid"])
    if (
        runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION
        or runtime.get("status") != "completed"
        or runtime.get("blockers")
        or runtime.get("inpaint_3d_executed") is not True
        or runtime.get("retry_cap") != 0
    ):
        raise Inpaint360ExecutionReceiptError(["inpaint360_runtime_not_completed"])
    if (
        run.get("schema_version") != RUN_SCHEMA_VERSION
        or run.get("status") != "completed"
        or run.get("blockers")
        or run.get("continuing_spend_from_this_run") is not False
        or run.get("all_staged_objects_absent") is not True
        or run.get("retry_cap") != 0
    ):
        raise Inpaint360ExecutionReceiptError(["inpaint360_provider_run_not_completed"])
    if Path(str(run.get("execution_result_path") or "")).resolve() != runtime_path:
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_runtime_result_binding_mismatch"]
        )
    teardown_path, _ = _verify_teardown(run, evidence)

    scene = adapter.get("scene") or {}
    source = adapter.get("source") or {}
    if (
        runtime.get("scene_id") != scene.get("publisher_scene_id")
        or runtime.get("target_instance_id") != scene.get("target_instance_id")
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_execution_scene_or_target_mismatch"]
        )
    source_before = runtime.get("source_identity_before") or {}
    source_after = runtime.get("source_identity_after") or {}
    nested_before = runtime.get("nested_dependency_identity_before") or {}
    nested_after = runtime.get("nested_dependency_identity_after") or {}
    if (
        runtime.get("source_commit") != source.get("commit")
        or runtime.get("source_tree") != source.get("tree")
        or runtime.get("source_modified") is not False
        or source_before.get("matches") is not True
        or source_after.get("matches") is not True
        or source_before.get("changed_files") != []
        or source_after.get("changed_files") != []
        or nested_before.get("matches") is not True
        or nested_after.get("matches") is not True
        or nested_before != nested_after
        or runtime.get("adapter_identity_before", {}).get("matches") is not True
        or runtime.get("execution_source_class")
        != "released_source_with_digest_bound_blueprint_obb_and_input_validity_adapters"
        or runtime.get("unchanged_source_execution_claimed") is not False
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_execution_source_identity_invalid"]
        )

    workflow = runtime.get("workflow") or []
    completed_stages = {
        str(row.get("stage") or "")
        for row in workflow
        if isinstance(row, Mapping)
        and row.get("returncode") == 0
        and row.get("timed_out") is False
    }
    if (
        not set(REQUIRED_STAGES).issubset(completed_stages)
        or any(
            row.get("returncode") != 0 or row.get("timed_out") is not False
            for row in workflow
            if isinstance(row, Mapping)
        )
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_execution_workflow_incomplete"]
        )

    runtime_root = runtime_path.parent
    _, final_ply = _verify_runtime_record(
        runtime.get("final_point_cloud") or {},
        root=runtime_root,
        code="inpaint360_final_point_cloud_changed",
    )
    if (
        selection.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection.get("status") != "accepted"
        or selection.get("blockers") != []
        or not (selection.get("selected_view") or {}).get("view_id")
        or selection != runtime.get("supplemental_fusion_view_selection")
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_selected_view_receipt_not_accepted"]
        )
    if (
        budget.get("schema_version") != GAUSSIAN_BUDGET_SCHEMA_VERSION
        or budget.get("status") != "accepted"
        or budget.get("blockers") != []
        or not isinstance(budget.get("final_vertex_count"), int)
        or budget["final_vertex_count"] <= 0
        or budget.get("added_vertex_count", 0) > budget.get("maximum_added_vertex_count", -1)
        or budget != runtime.get("added_gaussian_budget_validation")
    ):
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_gaussian_budget_receipt_not_accepted"]
        )
    final_ply["vertex_count"] = budget["final_vertex_count"]

    expected_camera_ids = [
        Path(str(row.get("relative_path") or "")).stem
        for row in (adapter.get("adapter") or {}).get("staged_artifacts") or []
        if str(row.get("relative_path") or "").startswith("source/images/")
        and str(row.get("relative_path") or "").endswith(".png")
    ]
    if not expected_camera_ids:
        raise Inpaint360ExecutionReceiptError(
            ["inpaint360_adapter_camera_set_missing"]
        )
    verified_renders = _verify_render_manifest(
        manifest_path=render_path,
        manifest=render,
        expected_camera_ids=expected_camera_ids,
        final_ply_sha256=final_ply["sha256"],
    )
    _verify_locality(
        locality=locality,
        locality_path=locality_path,
        render_manifest=render,
        render_manifest_path=render_path,
    )

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "adp_item": "ADP-009B",
        "status": "executed_rejected_quality",
        "scene": {
            "publisher_scene_id": scene.get("publisher_scene_id"),
            "target_instance_id": scene.get("target_instance_id"),
            "target_semantic_label": scene.get("target_semantic_label"),
            "camera_ids": expected_camera_ids,
        },
        "source": {
            **source,
            "identity_before": source_before,
            "identity_after": source_after,
            "nested_dependency_identity": nested_after,
            "execution_source_class": runtime.get("execution_source_class"),
            "unchanged_source_execution_claimed": runtime.get(
                "unchanged_source_execution_claimed"
            ),
        },
        "prepared_adapter": {
            **_file_record(adapter_path),
            "receipt_digest": adapter.get("receipt_digest"),
        },
        "provider_run": {
            **_file_record(run_path),
            "estimated_cost_usd": run.get("estimated_cost_usd"),
            "hard_cap_usd": run.get("hard_cap_usd"),
            "hard_ttl_seconds": run.get("hard_ttl_seconds"),
            "retry_cap": 0,
            "continuing_spend_from_this_run": False,
            "all_staged_objects_absent": True,
            "teardown_manifest": _file_record(teardown_path),
        },
        "execution": {
            "runtime_result": _file_record(runtime_path),
            "required_stages": list(REQUIRED_STAGES),
            "final_point_cloud": final_ply,
            "selected_view_receipt": {
                **_file_record(selection_path),
                "selected_view": selection["selected_view"],
            },
            "gaussian_budget_receipt": {
                **_file_record(budget_path),
                "baseline_vertex_count": budget.get("baseline_vertex_count"),
                "post_removal_vertex_count": budget.get("post_removal_vertex_count"),
                "final_vertex_count": budget.get("final_vertex_count"),
                "added_vertex_count": budget.get("added_vertex_count"),
                "maximum_added_vertex_count": budget.get(
                    "maximum_added_vertex_count"
                ),
            },
        },
        "independent_render": {
            "manifest": {
                **_file_record(render_path),
                "manifest_digest": render.get(
                    "sealed_camera_render_manifest_digest"
                ),
                "graphics_backend": "metal",
            },
            "renders": verified_renders,
        },
        "quality": {
            "status": "rejected_visual_artifacts",
            "locality_measurement": {
                **_file_record(locality_path),
                "measurement_digest": locality.get("locality_measurement_digest"),
                "aggregate": locality.get("aggregate"),
                "dilation_pixels": locality.get("dilation_pixels"),
            },
            "thresholds_frozen_before_evaluation": False,
            "quality_pass_claimed": False,
            "admission_effect": "none",
        },
        "claim_boundary": {
            "released_method_execution_completed": True,
            "successful_inpainting_admitted": False,
            "public_scene_suite_component_admitted": False,
            "publisher_splat_edited_in_place": False,
            "source_collider_removed": False,
            "simready_replacement_inserted": False,
            "output_claim_ceiling": "rejected_visual_candidate_only",
        },
        "blockers": [QUALITY_BLOCKER],
        "raw_secret_values_recorded": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if receipt_output is not None:
        output = _under(
            receipt_output,
            (repo,),
            "inpaint360_execution_receipt_output_outside_repo",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-receipt", required=True)
    parser.add_argument("--runtime-result", required=True)
    parser.add_argument("--run-result", required=True)
    parser.add_argument("--selected-view-receipt", required=True)
    parser.add_argument("--gaussian-budget-receipt", required=True)
    parser.add_argument("--metal-render-manifest", required=True)
    parser.add_argument("--locality-measurement", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--receipt-output", required=True)
    args = parser.parse_args(argv)
    receipt = materialize_inpaint360_execution_receipt(
        adapter_receipt_path=args.adapter_receipt,
        runtime_result_path=args.runtime_result,
        run_result_path=args.run_result,
        selected_view_receipt_path=args.selected_view_receipt,
        gaussian_budget_receipt_path=args.gaussian_budget_receipt,
        metal_render_manifest_path=args.metal_render_manifest,
        locality_measurement_path=args.locality_measurement,
        evidence_root=args.evidence_root,
        repo_root=args.repo_root,
        receipt_output=args.receipt_output,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
