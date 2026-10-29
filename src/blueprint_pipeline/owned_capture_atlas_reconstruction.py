"""Adopt a retained owned Atlas draft without disclosure, allocation or qualification.

The original video remains the capture. Provider images/poses and generated
geometry remain derived appearance, including when a mesh is present.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .owned_capture_atlas_context import validate_owned_capture_atlas_context
from .reconstruction_capability import (
    aggregate_reconstruction_costs, build_reconstruction_method_profile,
    normalize_reconstruction_result, plan_reconstruction_methods,
)

PROJECTION_FILE = "owned_atlas_reconstruction_projection.json"


def _verified(value: Mapping[str, Any], field: str) -> None:
    if value.get(field) != canonical_digest(value, digest_field=field):
        raise ValueError("owned_atlas_reconstruction_digest_invalid")


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("owned_atlas_reconstruction_asset_path_invalid")
    path = PurePosixPath(value)
    if (not value or "\\" in value or path.is_absolute() or any(p in {".", ".."} for p in path.parts)
            or ":" in path.parts[0]):
        raise ValueError("owned_atlas_reconstruction_asset_path_invalid")
    return path


def project_owned_atlas_reconstruction(*, owned_plan: Mapping[str, Any], job: Mapping[str, Any],
                                      result: Mapping[str, Any], manifest: Mapping[str, Any],
                                      implementation_digest: str) -> dict[str, Any]:
    """Pure deterministic projection; the adoption loader verifies actual files/source."""
    context = validate_owned_capture_atlas_context(owned_plan["source_context"])
    if (owned_plan.get("schema_version") != "owned_capture_atlas_preparation_plan.v1"
            or owned_plan.get("plan_digest") != cross_runtime_canonical_digest(owned_plan, digest_field="plan_digest")):
        raise ValueError("owned_atlas_reconstruction_plan_invalid")
    _verified(result, "digest")
    _verified(manifest, "digest")
    archive = job.get("archive") or {}
    download = result.get("download") or {}
    if (result.get("schema_version") != "website_atlas_pipeline_result.v1"
            or job.get("state") != "ready" or job.get("plan_id") != owned_plan["plan_id"]
            or context["request_id"] != owned_plan["plan_id"] or job.get("plan_digest") != owned_plan["plan_digest"]
            or job.get("capture_id") != context["capture_id"] or job.get("intake_id") != context["intake_id"]
            or job.get("task_context_digest") != context["context_digest"]
            or result.get("task_context_digest") != context["context_digest"]
            or any(v != context["source_capture"]["capture_digest"] for v in
                   (owned_plan.get("source_video_digest"), job.get("source_video_digest"),
                    result.get("source_video_digest"), manifest.get("source_video_digest")))
            or job.get("result_digest") != result["digest"]
            or archive.get("digest") != download.get("zip_sha256")
            or archive.get("manifest_digest") != manifest["digest"]
            or download.get("manifest_digest") != manifest["digest"]
            or manifest.get("schema_version") != "atlas_download_bundle.v1"
            or manifest.get("claim_ceiling") != "development_only"
            or manifest.get("metric_measurement_proven") is not False
            or manifest.get("simulation_qualified") is not False
            or manifest.get("scale_status") != "unscaled_provider_world_units"
            or not manifest.get("splat_available")
            or (result.get("scene_completion") or {}).get("status") not in (None, "completed")):
        raise ValueError("owned_atlas_reconstruction_binding_invalid")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("owned_atlas_reconstruction_assets_required")
    references = {}
    paths = set()
    for index, asset in enumerate(assets):
        if (not isinstance(asset, Mapping) or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(asset.get("sha256")))
                or type(asset.get("bytes")) is not int or asset["bytes"] < 0):
            raise ValueError("owned_atlas_reconstruction_asset_invalid")
        relative = _relative(asset.get("path"))
        if str(relative) in paths:
            raise ValueError("owned_atlas_reconstruction_duplicate_asset")
        paths.add(str(relative))
        references[f"retained_asset_{index}"] = {"uri": f"local-reconstruction://{asset['sha256'][7:]}",
            "digest": asset["sha256"], "member_path": str(relative), "size_bytes": asset["bytes"],
            "bundle_digest": archive["digest"], "bundle_manifest_digest": manifest["digest"]}
    if not any(path.startswith("splat/") and path.endswith(".ply") for path in paths):
        raise ValueError("owned_atlas_reconstruction_splat_required")
    profile = build_reconstruction_method_profile({"method_id": "owned_atlas_retained_draft", "version": "1",
        "method_kind": "gaussian_splat_3d", "provider_identity": "world_labs_atlas",
        "implementation_digest": implementation_digest, "execution_mode": "retained_artifact_adoption",
        "outputs": ["appearance_layer"], "execution_authorized": True, "qualification_status": "debug_only",
        "expected_cost_usd": 0.0, "cost_scope": "local_adoption_only"})
    plan = plan_reconstruction_methods(intake_id=context["intake_id"],
        capture_digest=context["source_capture"]["capture_digest"],
        capture_authority_profile=context["source_capture"]["capture_authority_profile"], claim_ceiling={},
        requested_claim_types=owned_plan["requested_claim_types"], permitted_provider_identities=["world_labs_atlas"],
        method_profiles=[profile])
    # Adoption attaches an already-existing diagnostic even when the requested
    # claim needs a metric/physics representation it cannot provide.
    plan["selected_methods"] = [{"method_id": profile["method_id"], "method_version": "1",
        "method_profile_digest": profile["method_profile_digest"], "provider_identity": "world_labs_atlas",
        "representations": ["appearance_layer"], "expected_cost_usd": 0.0}]
    plan["cost_scope"] = "local_adoption_only"
    plan["reconstruction_plan_digest"] = canonical_digest(plan, digest_field="reconstruction_plan_digest")
    provenance = {"owned_plan_digest": owned_plan["plan_digest"], "owned_result_digest": result["digest"],
        "bundle_digest": archive["digest"], "bundle_manifest_digest": manifest["digest"],
        "authorization_digest": job["authorization_digest"], "source_context_digest": context["context_digest"],
        "owned_plan_id": owned_plan["plan_id"], "owner_user_id": job["owner_user_id"]}
    runtime_identity = "blueprint-owned-atlas-retained-projection-v1"
    normalized = normalize_reconstruction_result({"result_id": "owned-atlas-" + canonical_digest(provenance)[7:31],
        "intake_id": context["intake_id"], "capture_digest": context["source_capture"]["capture_digest"],
        "method_id": profile["method_id"], "method_version": "1", "method_profile_digest": profile["method_profile_digest"],
        "implementation_digest": implementation_digest, "provider_identity": "world_labs_atlas",
        "runtime_identity": runtime_identity, "runtime_digest": canonical_digest({"runtime": runtime_identity}),
        "outputs": ["appearance_layer"], "source_frames": {"source_capture": context["source_capture"],
            "provider_views_are_raw_capture": False},
        "camera_solution": {"status": "provider_estimated", "calibrated": False,
                            "pose_result_digest": manifest.get("pose_result_digest")},
        "coordinate_system": {"scale_status": "unscaled_provider_world_units", "metric_scale_established": False},
        "asset_references": references, "coverage_map": {"status": "not_independently_qualified"},
        "observed_regions": [], "generated_regions": [],
        "invalid_regions": [{"region_id": "unverified_asset_extent",
                             "reason": "observed_generated_partition_not_qualified_for_physics"}],
        "uncertainty_map": {"generated_extent": "unknown", "metric_scale": "unverified", "collision": "unverified"},
        "validation_metrics": {"retained_bundle_binding_verified": True, "qualified_metric_validation": False},
        "cost_usd": None, "cost_status": "unknown", "cost_reason": "provider_usage_unreported",
        "known_cost_subtotal_usd": 0.0,
        "duration_seconds": max(0, job["revised_at_epoch"] - job["enqueued_at_epoch"]),
        "duration_scope": "enqueue_to_archive_wall_time", "provider_receipt": provenance,
        "rights_and_retention": deepcopy(context["governance"]), "deletion_evidence": {"status": "see_current_owned_job_cleanup_receipts",
            "owned_plan_id": owned_plan["plan_id"]},
        "claim_ceiling": {"appearance_review": True, "metric_geometry": False, "metric_scale": False,
            "calibrated_camera_poses": False, "collision_geometry": False, "physics": False,
            "physical_task_success": False, "deployment_readiness": False, "safety_certification": False}})
    normal_context = {"schema_version": "reconstruction_control_plane_context.v1",
        "capture_session_id": context["capture_id"], "intake_id": context["intake_id"],
        **context["source_capture"], "owned_atlas_plan_id": owned_plan["plan_id"],
        "owned_source_context_digest": context["context_digest"]}
    normal_context["context_digest"] = canonical_digest(normal_context, digest_field="context_digest")
    execution = {"schema_version": "reconstruction_control_plane_execution_result.v1", "plan_id": owned_plan["plan_id"],
        "state": "partial", "reconstruction_plan_digest": plan["reconstruction_plan_digest"],
        "authorization_digest": job["authorization_digest"], "context_digest": normal_context["context_digest"],
        "results": [normalized], "errors": [],
        "missing_representations": [row["representation"] for row in plan["missing_representations"]],
        "next_cheapest_experiments": [], **aggregate_reconstruction_costs([normalized]),
        "proof_boundary": {"execution_was_local_and_explicitly_authorized": True,
            "retained_artifact_adoption_only": True, "provider_dispatch_performed": False,
            "provider_reconstruction_was_remote": True,
            "derived_reconstruction_upgrades_raw_capture": False, "physical_task_success_established": False,
            "deployment_or_safety_approved": False, "comparative_policy_ranking_verdict": "thesis_not_supported"},
        "already_exists": False}
    execution["execution_result_digest"] = cross_runtime_canonical_digest(execution, digest_field="execution_result_digest")
    return {"context": normal_context, "reconstruction_plan": plan, "reconstruction_results": [normalized],
            "execution_result": execution}


def adopt_owned_atlas_reconstruction(*, state_root: Path, capture_store_root: Path, plan_id: str,
                                    expected_execution_result_digest: str | None = None) -> dict[str, Any]:
    """Verify local retained bytes and immutable source; never dispatch a provider."""
    from .owned_capture_atlas_runtime import bind_owned_atlas_source, owned_atlas_source_checkpoint
    from .owned_capture_atlas_worker import _job_path, _check_job_source, read_job
    from .reconstruction_control_plane import _source_binding, _write_immutable
    from .website_atlas_preview import preparation_job_lock
    path = _job_path(state_root, plan_id)

    def read(file):
        if file.is_symlink():
            raise ValueError("owned_atlas_reconstruction_symlink")
        return json.loads(file.read_text())

    def inside(value):
        file = Path(value)
        if file.is_symlink() or not file.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError("owned_atlas_reconstruction_path_unsafe")
        return file

    with preparation_job_lock(path.parent), bind_owned_atlas_source(state_root=state_root,
            capture_store_root=capture_store_root, plan_id=plan_id) as context:
        checkpoint = owned_atlas_source_checkpoint(context)
        checkpoint("owned_atlas_result_adoption")
        job = read_job(path)
        _check_job_source(job, context=context, path=path)
        owned_plan = read(path.parent.parent / "artifacts/owned_atlas_preparation_plan.json")
        binding = read(path.parent / "execution_binding.json")
        if binding.get("schema_version") != "owned_capture_atlas_queue_binding.v1" or any(binding.get(key) != job.get(key) for key in (
                "plan_id", "capture_id", "intake_id", "owner_user_id", "task_context_digest",
                "source_video_digest", "authorization_digest")):
            raise ValueError("owned_atlas_reconstruction_execution_binding_invalid")
        result = read(path.parent / "result.json")
        download = result["download"]
        directory, archive = inside(download["directory"]), inside(download["zip_path"])
        if _sha256_file(archive) != download["zip_sha256"] or archive.stat().st_size != job["archive"]["size_bytes"]:
            raise ValueError("owned_atlas_reconstruction_archive_changed")
        manifest = read(directory / "manifest.json")
        for asset in manifest["assets"]:
            file = inside(directory / str(_relative(asset["path"])))
            if _sha256_file(file) != asset["sha256"] or file.stat().st_size != asset["bytes"]:
                raise ValueError("owned_atlas_reconstruction_asset_changed")
        projection_path = path.parent.parent / "artifacts" / PROJECTION_FILE
        prior = read(projection_path) if projection_path.exists() else None
        implementation = (prior["reconstruction_results"][0]["implementation_digest"] if prior is not None
                          else _sha256_file(Path(__file__)))
        projection = project_owned_atlas_reconstruction(owned_plan=owned_plan, job=job, result=result,
            manifest=manifest, implementation_digest=implementation)
        # Bind the local view to the actual private downloadable ZIP, without extraction.
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            expected = {"manifest.json", *(asset["path"] for asset in manifest["assets"])}
            if len(names) != len(set(names)) or set(names) != expected:
                raise ValueError("owned_atlas_reconstruction_archive_members_invalid")
            if json.loads(bundle.read("manifest.json")) != manifest:
                raise ValueError("owned_atlas_reconstruction_archive_manifest_changed")
            for asset in manifest["assets"]:
                info = bundle.getinfo(asset["path"])
                if info.file_size != asset["bytes"] or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("owned_atlas_reconstruction_archive_asset_changed")
                digest = hashlib.sha256()
                with bundle.open(info) as member:
                    for chunk in iter(lambda: member.read(1024 * 1024), b""):
                        digest.update(chunk)
                if "sha256:" + digest.hexdigest() != asset["sha256"]:
                    raise ValueError("owned_atlas_reconstruction_archive_asset_changed")
        if (expected_execution_result_digest is not None
                and projection["execution_result"]["execution_result_digest"] != expected_execution_result_digest):
            raise ValueError("reconstruction_execution_digest_mismatch")
        checkpoint("owned_atlas_result_adoption_commit")
        retained = _write_immutable(path.parent.parent / "artifacts" / PROJECTION_FILE, projection)
        source = _source_binding(capture_store_root=capture_store_root,
            capture_session_id=context["capture_id"], intake_id=context["intake_id"])
        return {**retained, "capture_intake_envelope": source["envelope"], "capture_qa_report": source["qa"]}
