"""Read-only scientific joins for the two admitted SAM prefix lengths."""
from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import zipfile

from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, sha

SOURCE_CODE = (
    "src/blueprint_pipeline/public_scene_inpainting_inputs.py",
    "src/blueprint_pipeline/public_scene_inpainting_preparation.py",
    "src/blueprint_pipeline/sam31_camera_geometry.py",
    "src/blueprint_pipeline/source_calibration_camera_resolution.py",
    "src/blueprint_pipeline/sam31_source_calibration_stage.py",
    "tools/splat_render/render_splat.mjs", "tools/splat_render/src/render_entry.mjs",
    "scripts/adp_retained_scene_render_provider_runner.mjs",
    "scripts/source_calibration_camera_recovery.mjs",
)
SAM_CODE = (
    "src/blueprint_pipeline/public_scene_sam31_task_inputs.py",
    "src/blueprint_pipeline/sam31_source_track_canary_worker.py",
    "src/blueprint_pipeline/sam31_source_track_provider_stage.py",
    "src/blueprint_pipeline/scene_placement/sam31_source_track_provider.py",
)


def _load(name):
    # These existing validators own provider evidence but perform no allocation.
    return importlib.import_module("blueprint_pipeline." + name)


def record_identity(value):
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256", "size_bytes"}:
            return {k: record_identity(v) for k, v in value.items() if k != "path"}
        return {k: record_identity(v) for k, v in value.items()}
    if isinstance(value, list):
        return [record_identity(v) for v in value]
    return value


def task_science(task):
    """Only explicitly named release/namespace fields may differ on adoption."""
    value = deepcopy(task)
    for name in ("expected_production_commit", "run_prefix", "output_identity", "request_digest"):
        value.pop(name, None)
    value.get("configuration_provenance", {}).pop("execution_release_rebinding", None)
    value.get("scene_intent_authority", {}).pop("attempt", None)
    # These are independently reopened through the source and rights validators.
    references = value.get("source_input_references", {})
    for name in ("installation_receipt", "source_preparation_receipt", "standard_splat_conversion_receipt"):
        references.pop(name, None)
    authority = value.get("human_authority", {})
    for name in ("full_source_provider_disclosure_authority", "full_source_provider_disclosure_authorities"):
        authority.pop(name, None)
    return record_identity(value)


def camera_science(policy):
    value = deepcopy(policy)
    screen = value.get("geometry_screen", {})
    # Source bytes, collision bounds and shared frame are independently joined.
    screen.pop("source_files", None)
    screen.pop("screen_digest", None)
    return value


def source_science(host, commit):
    from .public_scene_removal_selection import _source_context, EVIDENCE_FIELDS
    task, context = _source_context({k: host[k] for k in EVIDENCE_FIELDS}, commit)
    frame = read(context["registered_frame"]["path"], digest_field="receipt_digest")
    correspondences = deepcopy(frame.get("correspondences"))
    for row in correspondences:
        row.pop("identity_receipt_digest", None)
    science = {
        "scene_id": context["scene_id"],
        "raw": {role: {k: row[k] for k in ("sha256", "size_bytes", "publisher_revision", "publisher_url")}
                for role, row in context["raw"].items()},
        "identities": {role: {"target": row["receipt"]["target"], "match": row["match"],
                              "coordinate_frame": row["receipt"]["coordinate_frame"]}
                       for role, row in context["identities"].items()},
        "registered_frame": {k: v for k, v in frame.items() if k not in
                             {"receipt_digest", "source_commit", "source_files", "correspondences"}},
        "correspondences": correspondences,
    }
    return task, context, science


def validate_current_rights(task, context, host, commit, roots):
    from .sam31_contribution_disclosure import validate_full_source_disclosure
    ref = task["source_input_references"]["standard_splat_conversion_receipt"]
    conversion_path = checked_file(ref["path"], ref)
    conversion = read(conversion_path, digest_field="receipt_digest")
    standard = checked_file(conversion_path.parent / conversion["output"]["relative_path"], conversion["output"])
    require(conversion["rights"]["terms_digest"] == host["interiorgs_terms"]["sha256"], "sam31_adoption_terms_changed")
    for purpose in ("exact_source_calibration_gpu_render", "released_code_segment_contribution_sweep",
                    "configured_scene_partitioned_source_processing"):
        validate_full_source_disclosure(task_authority=task["human_authority"], conversion_path=conversion_path,
            standard_splat_path=standard, original_source_path=context["raw"]["appearance_3dgs"]["path"],
            expected_source_commit=commit, publisher_scene_id=context["scene_id"], approved_roots=roots, purpose=purpose)
    return {"conversion": {"path": str(conversion_path), "sha256": sha(conversion_path), "size_bytes": conversion_path.stat().st_size},
            "standard": {"path": str(standard), "sha256": sha(standard), "size_bytes": standard.stat().st_size}}


def require_producer_files_present(code, old_root, current_root):
    """Producer files must still exist under both trees; return their retained (prefix-time) shas.

    Content identity, not commit identity (piece 1).  A completed paid prefix is trusted
    because its retained OUTPUTS are independently re-validated with the current code (the
    calibration outcome via validate_retained_source_calibration_stage, the finalized render
    recomputed to equal the retained receipt, scene/task/camera/rights re-derived, tracking
    identity re-joined).  A producer-code diff that leaves those outputs valid must NOT
    invalidate the prefix on every deploy, so we no longer require byte-identical producer
    files -- only that each still exists.  The retained sha is pinned for provenance.
    """

    old_root, current_root = Path(old_root), Path(current_root)
    files = []
    for relative in code:
        before, after = old_root / relative, current_root / relative
        require(before.is_file() and after.is_file(), "sam31_adoption_producer_code_missing:" + relative)
        files.append({"relative_path": relative, "sha256": sha(before), "size_bytes": before.stat().st_size})
    return files


def validate_render(outcome, artifacts, old_plan, current_repo, through_phase):
    from .public_scene_inpainting_preparation import validate_prepared_inputs, adopt_finalized_public_scene_inpainting_inputs
    from .public_scene_removal_selection import validate_removal_scene_selection, validate_removal_task_selection
    prepared_ref = artifacts["source_calibration_prepared_inputs"]
    prepared = validate_prepared_inputs(checked_file(prepared_ref["path"], prepared_ref))
    _load("sam31_source_calibration_stage").validate_retained_source_calibration_stage(outcome)
    # Recompute final masks and the complete receipt from all retained pixels.
    adopted = adopt_finalized_public_scene_inpainting_inputs(
        preparation_path=prepared_ref["path"], returned_group_path=artifacts["source_calibration_return"]["path"])
    require(adopted == read(artifacts["calibrated_view_receipt"]["path"]), "sam31_adoption_finalized_render_changed")
    scene = validate_removal_scene_selection(read(artifacts["scene_selection"]["path"]))
    task = validate_removal_task_selection(read(artifacts["task_selection"]["path"]))
    request = read(artifacts["calibrated_view_request"]["path"])
    require(request["scene"]["scene_freeze_path"] == artifacts["scene_selection"]["path"]
            and request["scene"]["task_freeze_path"] == artifacts["task_selection"]["path"]
            and request["scene"]["standard_splat_path"] == artifacts["standard_splat"]["path"]
            and task["scene_freeze_digest"] == scene["scene_freeze_digest"], "sam31_adoption_render_source_changed")
    require(camera_science(request["camera_policy"]) == camera_science(old_plan["camera_policy"]),
            "sam31_adoption_camera_changed")
    old_repo = Path(prepared["context"]["paths"]["repo"])
    code = SOURCE_CODE + (SAM_CODE if through_phase == "sam31_tracking" else ())
    files = require_producer_files_present(code, old_repo, current_repo)
    return {"path": str(old_repo), "source_commit": prepared["repository"]["commit"],
            "tree": prepared["repository"]["tree"], "unchanged_producer_files": files}


def provider_science(value):
    value = deepcopy(value)
    for key in ("source_commit_sha", "profile_digest", "execution_authorization_digest"):
        value.pop(key, None)
    value.get("authorization_sources", {}).pop("execution", None)
    return record_identity(value)


def validate_tracking(outcome, artifacts, old_profile, current_profile_path, old_commit, billing_source_path):
    from .scene_placement.semantic_gaussian_lifting import canonical_json_digest
    from .public_scene_calibrated_object_masks import _verified_source_tracks
    from .sam31_provider_launch_packet import validate_sam31_provider_profile_sources
    _load("task_evaluation_sam31_preparation_paid_stages").validate_retained_paid_stage(outcome, stage_id="sam31_tracking")
    old_provider = read(old_profile["artifact_references"]["sam31_provider_profile"]["path"])
    current_provider = read(current_profile_path)
    validate_sam31_provider_profile_sources(old_provider, source_commit_sha=old_commit)
    validate_sam31_provider_profile_sources(current_provider, source_commit_sha=current_provider["source_commit_sha"])
    require(provider_science(old_provider) == provider_science(current_provider), "sam31_adoption_model_changed")
    execution_path = Path(artifacts["sam31_allocator_result"]["path"])
    execution = read(execution_path, digest_field="execution_result_digest")
    root = execution_path.parent
    bound = read(root / "bound-request.json", digest_field="bound_request_digest")
    runtime_path = root / "sam31_vast_source_track_canary" / "provider_runtime_result.json"
    runtime = _load("sam31_vast_source_track_canary").validate_sam31_runtime_result(read(runtime_path), bound_request=bound)
    tracks = _verified_source_tracks(Path(artifacts["sam31_source_tracks"]["path"]))
    require(runtime["normalized_source_tracks"] == tracks and execution["source_commit_sha"] == old_commit
            and execution["status"] == "completed" and execution["provider_zero_verified"] is True
            and execution["all_staged_objects_absent"] is True
            and execution["continuing_spend_from_this_run"] is False
            and execution["retry_cap"] == 0 and execution["blockers"] == []
            and execution["source_track_import_result_digest"] == tracks["result_digest"]
            and execution["provider_runtime_result_digest"] == runtime["runtime_result_digest"], "sam31_adoption_tracking_changed")
    for name in ("request_digest", "bound_request_digest", "input_bundle_digest", "source_track_run_request_digest", "worker_image_digest"):
        require(execution[name] == bound[name] == runtime[name], "sam31_adoption_runtime_binding_changed:" + name)
    zero = read(artifacts["sam31_provider_zero"]["path"], digest_field="provider_zero_digest")
    require(zero.get("schema_version") == "semantic_sam31_vast_provider_zero.v1" and zero.get("status") == "PASS"
            and zero.get("api_confirmed") is True and zero.get("provider") == "vast"
            and zero.get("scoped_live_resource_count") == zero.get("global_live_resource_count") == 0
            and zero.get("provider_zero_digest") == execution.get("provider_zero_digest")
            and zero.get("request_digest") == execution["request_digest"]
            and zero.get("bound_request_digest") == execution["bound_request_digest"], "sam31_adoption_provider_zero_invalid")
    from .public_scene_sam31_track_selection_review import _resolve_prepared_task
    packet_path = Path(artifacts["sam31_task_input_packet"]["path"])
    packet = read(packet_path, digest_field="receipt_digest")
    require(all(packet[key][field] == artifacts[name][field] for key, name in
                (("task_freeze", "task_selection"), ("calibrated_view_receipt", "calibrated_view_receipt"))
                for field in ("path", "sha256", "size_bytes")), "sam31_adoption_task_packet_changed")
    _resolve_prepared_task(task_input_packet_path=packet_path,
        source_track_result_path=artifacts["sam31_source_tracks"]["path"],
        selected_track_ids=[row["track_id"] for row in tracks["track_registry"]])
    request = read(artifacts["sam31_run_request"]["path"])
    require(request["provider_profile"] == old_provider, "sam31_adoption_request_model_changed")
    # Reopen the actual portable request and every JPEG, not merely the top ZIP hash.
    bundle_dir = execution_path.parents[1] / "prepared"
    receipts = list(bundle_dir.glob("*bundle*receipt*.json"))
    require(len(receipts) == 1, "sam31_adoption_bundle_receipt_missing")
    bundle_receipt = read(receipts[0], digest_field="receipt_digest")
    bundle = checked_file(bundle_dir / bundle_receipt["bundle"]["filename"], bundle_receipt["bundle"])
    require(sha(bundle) == execution["input_bundle_digest"], "sam31_adoption_bundle_changed")
    with zipfile.ZipFile(bundle) as archive:
        import json
        portable = json.loads(archive.read("request.json"))
        require(canonical_json_digest(portable) == execution["source_track_run_request_digest"], "sam31_adoption_portable_request_changed")
        before, after = deepcopy(request), deepcopy(portable)
        source_frames, portable_frames = before.pop("frame_artifacts"), after.pop("frame_artifacts")
        require(before == after and len(source_frames) == len(portable_frames), "sam31_adoption_request_changed")
        for source, portable_frame in zip(source_frames, portable_frames, strict=True):
            original = checked_file(source["path"], source)
            require(archive.read(portable_frame["path"]) == original.read_bytes()
                    and {k:v for k,v in source.items() if k != "path"} == {k:v for k,v in portable_frame.items() if k != "path"},
                    "sam31_adoption_frame_changed")
    from .vast_official_billing_extractor import (
        _validate_source_receipt, _load_vast_responses, extract_vast_official_instance_charge,
    )
    source_path, billing, _ = _validate_source_receipt(billing_source_path)
    labels = []
    for _, _, _, response in _load_vast_responses(source_receipt_path=source_path, source_receipt=billing):
        labels.extend(row.get("metadata", {}).get("label") for row in response["results"]
                      if row.get("source") == "instance-" + str(execution["instance_id"]))
    require(len(labels) == 1 and isinstance(labels[0], str)
            and labels[0].startswith("blueprint-sam31-source-tracks-")
            and labels[0].endswith(execution["request_digest"].removeprefix("sha256:")[:12]),
            "sam31_adoption_official_billing_instance_invalid")
    charge = extract_vast_official_instance_charge(provider_billing_source_receipt_path=billing_source_path,
        instance_id=int(execution["instance_id"]), launch_label=labels[0])
    require(0 <= charge["official_charge_usd"] <= 1., "sam31_adoption_official_charge_invalid")
    return {"raw_runtime_result": {"path": str(runtime_path), "sha256": sha(runtime_path), "size_bytes": runtime_path.stat().st_size},
            "provider_instance_id": execution["instance_id"], "checkpoint_digest": execution["checkpoint_digest"],
            "official_charge": charge}
