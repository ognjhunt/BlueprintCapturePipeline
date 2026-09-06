"""Materialize a publication-ready attempt from an owner-provided completed asset.

The source resolver binds the owner's finished 3DGS bytes, its companion
collision mesh, the exact subject/support object identities, and the declared
coordinate frame.  This factory turns that immutable binding plus the reserved
attempt and the exact release into a ``publication_ready`` submission, calling
only CPU record builders.  It never uploads the owner's raw bytes, queries a
model, allocates a provider, or asserts a physical measurement: every
un-measured property is carried as a run-produced plan or a typed blocker.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_intake as intake
from .task_evaluation_public_scene_attempt_factory import RELEASE_SCHEMA, record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha
from .task_evaluation_scene_progression_state import require, safe_path

FACTORY_SCHEMA = "task_evaluation_completed_scene_attempt_factory.v1"
BINDING_SCHEMA = "task_evaluation_completed_scene_source.v1"
MACHINERY_SCHEMA = "task_evaluation_completed_scene_machinery.v1"
APPEARANCE_KIND = {"gaussian_splat": "gaussian_splat", "mesh": "other_observed"}
_SUCCESS_POSITIVE = ("control_frequency_hz", "maximum_episode_seconds", "minimum_lift_m",
                     "pregrasp_clearance_m", "minimum_planar_displacement_m",
                     "maximum_final_planar_target_error_m")


def _positive(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _vector(value: Any, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            return None
        out.append(float(item))
    return out


def _grasp_axis(minimum: list[float], maximum: list[float]) -> int:
    extents = [maximum[i] - minimum[i] for i in range(2)]
    return 0 if extents[0] <= extents[1] else 1


def _task_and_blockers(*, intent: dict, binding: dict, commit: str) -> tuple[dict | None, list[str]]:
    request = intent["request"]
    owner_task = request["task"]
    bindings = binding["object_bindings"]
    frame = binding.get("coordinate_frame") or {}
    blockers: list[str] = []
    mpu, axis = frame.get("meters_per_unit"), frame.get("up_axis")
    if not _positive(mpu) or axis not in {"Y", "Z"}:
        blockers.append("source_metric_scale_declaration_required")
    destination = owner_task.get("destination") if isinstance(owner_task.get("destination"), dict) else {}
    position = _vector(destination.get("position_world_m"), 3)
    orientation = _vector(destination.get("orientation_xyzw"), 4)
    if position is None or orientation is None or not math.isclose(
            sum(v * v for v in orientation), 1.0, rel_tol=0.0, abs_tol=1e-6):
        blockers.append("task_destination_pose_required")
    success = owner_task.get("success") if isinstance(owner_task.get("success"), dict) else {}
    valid_success = (all(_positive(success.get(field)) for field in _SUCCESS_POSITIVE)
                     and success.get("maximum_retries") == 0 and success.get("maximum_regrasps") == 0)
    if valid_success:
        steps = float(success["control_frequency_hz"]) * float(success["maximum_episode_seconds"])
        valid_success = math.isclose(round(steps), steps, rel_tol=0.0, abs_tol=1e-9)
    if not valid_success:
        blockers.append("task_success_criteria_required")
    if blockers:
        return None, blockers

    version = commit[:8]
    scene_key = binding["source_content_digest"][7:27]
    subject = bindings["subject"]
    support = bindings["support"]
    lower = [float(v) for v in subject["world_aabb_min_m"]]
    upper = [float(v) for v in subject["world_aabb_max_m"]]
    review_label = str((subject.get("owner_request") or {}).get("description")
                       or str(subject["source_object_id"]).rsplit("/", 1)[-1])
    support_label = str((support.get("owner_request") or {}).get("description")
                        or str(support["source_object_id"]).rsplit("/", 1)[-1])
    accepted_on = datetime.fromtimestamp(request["consent"]["accepted_at_epoch"], timezone.utc).isoformat()
    human_authority = {"accepted_by": request["owner"]["user_id"], "accepted_on": accepted_on,
        "authority_reference": "scene-intent:" + intent["intent_digest"],
        "private_derived_frame_disclosure_authorized": True, "provider_retention_terms_accepted": True,
        "provider_training_terms_accepted": True, "provider_training_authorized": False,
        "owner_provided_asset": True, "captured_observation_supplied": False}
    companion = request["source"].get("collision_mesh") if isinstance(request["source"], dict) else None
    task = {"schema_version": "task_evaluation_completed_scene_task_request.v1",
        "expected_production_commit": commit,
        "team_namespace": "scene-" + canonical_digest({"intent_digest": intent["intent_digest"],
            "attempt_binding": binding["binding_digest"]})[7:55],
        "run_prefix": "scene-" + intent["intent_id"].removeprefix("scene-")[:20] + "-completed",
        "scene_identity": {"id": "completed-scene-" + scene_key, "version": version},


        # The owner's declared task_id is carried unchanged as the task identity.
        # Controls autoprovision binds the owner authorization to the configured
        # scene by requiring the preparation link's task_id to equal BOTH the
        # owner intent's task_id and the preparation request's task identity id
        # (task_evaluation_controls_autoprovision.provision_link). A synthesized
        # "completed-task-<digest>" id would satisfy the second equality but never
        # the first, stranding every completed-asset run at the controls seam.
        # Intake already constrains task_id to a safe identifier (a subset of the
        # recipe schema's identifier pattern), so it is safe to use verbatim.
        "task_identity": {"id": owner_task["task_id"], "version": version},
        "output_identity": {"id": "completed-scene-" + scene_key + "-configured", "version": version},
        "appearance_kind": APPEARANCE_KIND[binding["source_kind"]],
        "coordinate_frame": {"declared_meters_per_unit": mpu, "declared_up_axis": axis,
                             "physical_scale_measured": False},
        "collision_rights_reference": companion.get("rights_reference") if isinstance(companion, dict) else None,
        "subject": {"identity": {"id": "completed-subject-" + scene_key, "version": version},
            "source_object_id": str(subject["source_object_id"]), "review_label": review_label,
            "aabb_min_xyz_m": lower, "aabb_max_xyz_m": upper,
            "point_count": int(subject["point_count"]), "face_count": int(subject["face_count"])},
        "support": {"source_object_id": str(support["source_object_id"]), "label": support_label,
            "aabb_min_xyz_m": [float(v) for v in support["world_aabb_min_m"]],
            "aabb_max_xyz_m": [float(v) for v in support["world_aabb_max_m"]],
            "point_count": int(support["point_count"]), "face_count": int(support["face_count"])},
        "destination": {"identity": {"id": "completed-destination-" + scene_key, "version": version},
            "relation": destination.get("relation") if destination.get("relation") in {"inside", "on"} else "on",
            "visible_label": str(destination.get("visible_label") or support_label or "destination"),
            "position_world_m": position, "orientation_xyzw": orientation},
        "grasp": {"axis": _grasp_axis(lower, upper), "sign": 1.0},
        "success": {field: success[field] for field in (*_SUCCESS_POSITIVE, "maximum_retries",
                    "maximum_regrasps")},
        "human_authority": human_authority}
    return task, []


def materialize_completed_scene_attempt(*, intent_path, source_binding_path, machinery_path,
                                        release_binding_path, output_root, attempt_id, now=None):
    """Build publication-ready inputs from an already reserved immutable attempt."""
    from .task_evaluation_scene_owner_authority import reopen_scene_intent
    from .task_evaluation_completed_scene_submission import materialize_completed_scene_submission

    intent = reopen_scene_intent(record(intent_path), now=now)
    binding = read(source_binding_path, digest_field="binding_digest")
    machinery = read(machinery_path, digest_field="machinery_digest")
    release = read(release_binding_path, digest_field="release_digest")
    require(binding.get("schema_version") == BINDING_SCHEMA
            and machinery.get("schema_version") == MACHINERY_SCHEMA
            and release.get("schema_version") == RELEASE_SCHEMA, "completed_factory_schema_invalid")
    request = intent["request"]
    require(binding.get("owner") == request["owner"]
            and binding.get("intent_digest") == intent["intent_digest"]
            and binding.get("source_content_digest") == request["source"]["content_digest"]
            and binding.get("task_digest") == intent["task_content_digest"]
            and binding.get("status") == "source_task_objects_bound", "completed_factory_binding_mismatch")
    commit = release["source_commit"]
    from .task_evaluation_scene_preparation_attempts import preparation_attempt_path
    attempt_path = preparation_attempt_path(Path(intent_path).parent, attempt_id)
    attempt = intake._read(attempt_path, "attempt_digest")
    require(attempt.get("intent_digest") == intent["intent_digest"]
            and attempt.get("source_commit") == commit
            and attempt.get("input_digest") == binding["binding_digest"], "completed_factory_attempt_mismatch")

    task, blockers = _task_and_blockers(intent=intent, binding=binding, commit=commit)
    if blockers:
        return {"schema_version": FACTORY_SCHEMA, "status": "needs_input", "blockers": blockers,
                "provider_mutation_performed": False}

    task["scene_intent_authority"] = record(intent_path)
    task["source_binding"] = record(source_binding_path)
    from . import task_evaluation_completed_scene_geometry as geometry
    source_ref = binding["references"]["collision"]
    cache = safe_path(output_root).parents[1] / "normalized-sources"
    normalized_root = cache / (
        source_ref["sha256"][7:] + "-" + sha(Path(geometry.__file__))[7:19] + "-normalized")
    normalized = geometry.normalize_completed_mesh(
        source=checked_file(source_ref["path"], source_ref),
        original_filename=binding["source_filenames"]["collision"],
        coordinate_frame=binding["coordinate_frame"], output_root=normalized_root)
    task["geometry_normalization"] = record(normalized_root / "mesh_normalization.v1.json")
    if binding["source_kind"] == "gaussian_splat":
        from . import task_evaluation_completed_scene_splat as splat_normalization
        primary = binding["references"]["primary"]
        splat_root = cache / (primary["sha256"][7:] + "-" + sha(Path(splat_normalization.__file__))[7:19] + "-splat")
        converted = splat_normalization.normalize_completed_splat(
            source=checked_file(primary["path"], primary), coordinate_frame=binding["coordinate_frame"],
            output_root=splat_root)
        if converted is not None:
            task["splat_normalization"] = record(splat_root / "splat_normalization.v1.json")
    for role in ("subject", "support"):
        task[role]["runtime_prim_path"] = normalized["object_mapping"][task[role]["source_object_id"]]

    catalog = machinery.get("destination_catalog")
    require(isinstance(catalog, list) and 1 <= len(catalog) <= 64,
            "completed_factory_destination_catalog_missing")
    destination = request["task"]["destination"]
    requested_asset = destination.get("asset_binding_id")
    description = str(destination.get("description") or destination.get("visible_label") or "").strip().casefold()
    candidates = [row for row in catalog if (
        row.get("binding_id") == requested_asset if requested_asset else
        description in [str(label).strip().casefold() for label in row.get("owner_description_aliases", [])])]
    if len(candidates) != 1:
        return {"schema_version": FACTORY_SCHEMA, "status": "needs_input",
                "blockers": ["task_destination_asset_selection_required"], "provider_mutation_performed": False}
    asset = candidates[0]
    task["destination"]["simready_result"] = record(checked_file(
        asset["simready_result"]["path"], asset["simready_result"]))
    task["destination"]["catalog_binding_id"] = asset["binding_id"]
    physics = machinery.get("simulation_physics_bounds")
    require(isinstance(physics, dict), "completed_factory_simulation_physics_bounds_missing")
    task["subject"]["physics_bounds"] = physics

    output = safe_path(Path(output_root))
    require(output.is_absolute() and not any(p.is_symlink() for p in (output, *output.parents))
            and not output.is_relative_to(Path(release["repo_root"])), "completed_factory_output_root_invalid")
    output.mkdir(parents=True, exist_ok=True, mode=0o750)
    receipt_path = output / "factory_receipt.json"
    if receipt_path.exists():
        receipt = read(receipt_path, digest_field="factory_digest")
        require(receipt.get("intent_digest") == intent["intent_digest"]
                and receipt.get("attempt_digest") == attempt["attempt_digest"],
                "completed_factory_immutable_conflict")
        return receipt

    task_path = output / "task_request.json"
    if not task_path.exists():
        intake.write_exclusive(task_path, task)
    require(read(task_path) == task, "completed_factory_task_conflict")
    submission_root = output / "submission"
    manifest_path = submission_root / "bundle_manifest.v1.json"
    materialize_completed_scene_submission(binding=binding, task=task, task_request_path=task_path,
            deploy_receipt_path=release["deploy_receipt"]["path"],
            release_provenance_path=release["release_provenance"]["path"],
            release_environment_path=release["release_environment"]["path"],
            runtime_publication_root=release["runtime_publication_root"], expected_production_commit=commit,
            namespace_timestamp=release["namespace_timestamp"],
            release_admission_mode=release["release_admission_mode"], staging_root=submission_root,
            scene_intent_digest=intent["intent_digest"])
    manifest = read(manifest_path, digest_field="manifest_digest")
    for row in manifest["files"]:
        checked_file(submission_root / row["relative_path"],
                     {"sha256": row["digest"], "size_bytes": row["size_bytes"]})
    receipt = {"schema_version": FACTORY_SCHEMA, "status": "publication_ready",
        "intent_digest": intent["intent_digest"], "attempt_digest": attempt["attempt_digest"],
        "source_commit": commit, "source_kind": binding["source_kind"],
        "task_request": record(task_path), "submission_manifest": record(manifest_path),
        "submission_request": record(submission_root / "scene_configuration_preparation_request.v1.json"),
        "frozen_policy_candidates": request["execution"]["policy_candidates"],
        "coordinate_frame": task["coordinate_frame"], "physical_scale_measured": False,
        "physical_registration_proven": False, "provider_reconstruction_started": False,
        "original_source_reinstalled": False, "task_model_queried": False, "source_uploaded": False,
        "provider_mutation_performed": False, "claim_scope": "development_only"}
    receipt["factory_digest"] = canonical_digest(receipt, digest_field="factory_digest")
    intake.write_exclusive(receipt_path, receipt)
    return receipt


__all__ = ["materialize_completed_scene_attempt", "FACTORY_SCHEMA"]
