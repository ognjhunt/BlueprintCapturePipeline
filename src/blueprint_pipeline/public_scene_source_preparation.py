"""Derive source context from installed scene bytes and selected task objects.

No customer-supplied registration, geometry qualification, robot packet, model,
or policy is consumed. This is source preparation, not native qualification.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_host_input_intake import (
    DEFAULT_DESTINATION_ROOT, RECEIPT_SCHEMA, _sha256_file, _verified_checkout_head,
)
from .public_scene_viewpoint_survey import build_room_viewpoint_survey
from .sage_collision_identity import (
    SageCollisionIdentityError, build_interiorgs_sage_shared_frame_candidate,
    inspect_sage_collision_identity,
)

from .scene_placement.interiorgs_index import (
    load_interiorgs_labels, supporting_fixtures_for,
)

SCHEMA_VERSION = "public_scene_source_preparation.v1"
SOURCE_ROLES = {"movable_subject", "source_support"}
REQUIRED_FILES = {"appearance_3dgs", "semantic_metadata", "scene_structure", "collision_usd"}


class PublicSceneSourcePreparationError(ValueError):
    """Source context could not be derived without inventing evidence."""


def _resident(value: str | Path, roots: Sequence[Path]) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise PublicSceneSourcePreparationError("source_preparation_symlink_forbidden")
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PublicSceneSourcePreparationError("source_preparation_path_outside_admitted_roots")
    return resolved


def _record(path: Path, root: Path) -> dict[str, Any]:
    return {"relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
    path.chmod(0o440)


def _cached_result(
    output: Path, *, source_commit: str, installation_digest: str, task_digest: str,
) -> dict[str, Any]:
    try:
        path = _resident(output / (SCHEMA_VERSION + ".json"), (output,))
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("schema_version") != SCHEMA_VERSION
                or value.get("source_commit") != source_commit
                or value.get("source_installation_digest") != installation_digest
                or value.get("task_objects_digest") != task_digest
                or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")):
            raise ValueError("binding")
        for record in value["artifacts"]:
            artifact = _resident(output / record["relative_path"], (output,))
            if _record(artifact, output) != record:
                raise ValueError("artifact")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PublicSceneSourcePreparationError("source_preparation_output_conflict") from exc
    return dict(value)


def materialize_public_scene_source_preparation(
    *,
    installation_receipt_path: str | Path,
    task_objects: Sequence[Mapping[str, Any]],
    expected_source_commit: str,
    output_root: str | Path | None = None,
    approved_roots: Sequence[str | Path] = (DEFAULT_DESTINATION_ROOT,),
) -> dict[str, Any]:
    roots = tuple(Path(root).expanduser().resolve() for root in approved_roots)
    receipt_path = _resident(installation_receipt_path, roots)
    installation = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        not isinstance(installation, Mapping)
        or installation.get("schema_version") != RECEIPT_SCHEMA
        or installation.get("status") != "installed"
        or installation.get("service_readable") is not True
        or installation.get("receipt_digest") != canonical_digest(installation, digest_field="receipt_digest")
        or Path(str(installation.get("destination_root") or "")).resolve() != receipt_path.parent
        or expected_source_commit != _verified_checkout_head()
    ):
        raise PublicSceneSourcePreparationError("source_preparation_installation_or_execution_invalid")
    inputs = receipt_path.parent
    files: dict[str, Path] = {}
    for row in installation.get("files") or []:
        if not isinstance(row, Mapping):
            raise PublicSceneSourcePreparationError("source_preparation_inventory_invalid")
        relative = str(row.get("relative_path") or "")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts or "\\" in relative:
            raise PublicSceneSourcePreparationError("source_preparation_inventory_invalid")
        path = _resident(inputs / relative, (inputs,))
        size = row.get("size_bytes")
        if (not path.is_file() or isinstance(size, bool) or not isinstance(size, int)
                or size <= 0 or path.stat().st_size != size or _sha256_file(path) != row.get("sha256")):
            raise PublicSceneSourcePreparationError("source_preparation_input_bytes_changed")
        role = str(row.get("role") or "")
        if role in REQUIRED_FILES:
            if role in files:
                raise PublicSceneSourcePreparationError("source_preparation_source_role_duplicate")
            files[role] = path
    if set(files) != REQUIRED_FILES:
        raise PublicSceneSourcePreparationError("source_preparation_raw_scene_components_missing")
    # Inspect the layer without composing references outside the admitted packet.
    from pxr import Sdf

    collision_layer = Sdf.Layer.FindOrOpen(str(files["collision_usd"]))
    if collision_layer is None or collision_layer.subLayerPaths or collision_layer.GetExternalReferences():
        raise PublicSceneSourcePreparationError("source_preparation_external_usd_dependency_forbidden")
    if (not isinstance(task_objects, (list, tuple)) or not 1 <= len(task_objects) <= 3
            or any(not isinstance(row, Mapping) for row in task_objects)):
        raise PublicSceneSourcePreparationError("source_preparation_task_objects_invalid")
    if sum(row.get("role") == "supplemental_destination" for row in task_objects) > 1:
        raise PublicSceneSourcePreparationError("source_preparation_task_objects_invalid")
    task_digest = canonical_digest({"task_objects": list(task_objects)})
    selected = []
    for obj in task_objects:
        role = obj.get("role")
        if role == "supplemental_destination":
            if set(obj) != {"role", "description"} or not str(obj.get("description") or "").strip():
                raise PublicSceneSourcePreparationError("source_preparation_fake_destination_source_forbidden")
            continue
        identity = str(obj.get("source_instance_id") or "")
        if set(obj) != {"role", "source_instance_id"} or role not in SOURCE_ROLES or not identity:
            raise PublicSceneSourcePreparationError("source_preparation_task_objects_invalid")
        selected.append({"role": role, "source_instance_id": identity})
    objects = load_interiorgs_labels(files["semantic_metadata"])
    if not any(row["role"] == "source_support" for row in selected):
        subjects = [row for row in selected if row["role"] == "movable_subject"]
        candidates = [obj for obj in objects if len(subjects) == 1
                      and obj.id == subjects[0]["source_instance_id"]]
        supports = (supporting_fixtures_for(candidates[0], objects, top_tolerance_m=0.02)
                    if len(candidates) == 1 else [])
        if candidates:
            subject = candidates[0]
            supports = [obj for obj in supports if all(
                obj.bbox_min[i] <= subject.bbox_min[i]
                and obj.bbox_max[i] >= subject.bbox_max[i] for i in (0, 1)
            )]
        if len(supports) != 1:
            raise PublicSceneSourcePreparationError("source_preparation_support_selection_ambiguous")
        selected.append({"role": "source_support", "source_instance_id": supports[0].id})
    if (len(selected) != 2 or {row["role"] for row in selected} != SOURCE_ROLES
            or len({row["source_instance_id"] for row in selected}) != len(selected)):
        raise PublicSceneSourcePreparationError("source_preparation_subject_support_identity_invalid")
    output = _resident(
        output_root if output_root is not None else inputs.with_name(
            inputs.name + "." + task_digest.removeprefix("sha256:")[:12]
            + "." + expected_source_commit[:8] + ".source-preparation"
        ),
        roots,
    )
    if output == inputs or inputs in output.parents:
        raise PublicSceneSourcePreparationError("source_preparation_output_not_fresh")
    if output.exists():
        return _cached_result(
            output, source_commit=expected_source_commit,
            installation_digest=installation["receipt_digest"], task_digest=task_digest,
        )
    output.mkdir(parents=True, mode=0o750)
    survey = build_room_viewpoint_survey(
        structure_path=files["scene_structure"], labels_path=files["semantic_metadata"],
        scene_id=str(installation["scene_id"]), approved_roots=(inputs,),
    )
    survey_path = output / "room_topology_survey.json"
    _write(survey_path, survey)
    identities, observed_objects, artifacts, blockers = [], [], [_record(survey_path, output)], []
    for index, obj in enumerate(selected):
        try:
            result = inspect_sage_collision_identity(
                labels_path=files["semantic_metadata"],
                target_instance_id=obj["source_instance_id"],
                sage_collision_usd_path=files["collision_usd"],
            )
        except (SageCollisionIdentityError, ValueError, RuntimeError) as exc:
            code = "source_preparation_identity_measurement_failed:" + obj["source_instance_id"]
            failure_path = output / f"source_identity_{index:02d}_failure.json"
            _write(failure_path, {"status": "blocked", "blocker": code, "exception_type": type(exc).__name__})
            artifacts.append(_record(failure_path, output))
            blockers.append(code)
            continue
        artifact = output / f"source_identity_{index:02d}.json"
        _write(artifact, result)
        artifacts.append(_record(artifact, output))
        identities.append(result)
        observed_objects.append(obj)
        if result["whole_object_collision_identity_passed"] is not True:
            blockers.append("source_preparation_whole_object_match_not_unique:" + obj["source_instance_id"])
    if not blockers and len({
        row["whole_object_matches"][0]["prim_path"] for row in identities
    }) != len(identities):
        blockers.append("source_preparation_source_colliders_not_distinct")
    frame = None
    if not blockers:
        frame = build_interiorgs_sage_shared_frame_candidate(identities)
        frame_path = output / "shared_frame_candidate.json"
        _write(frame_path, frame)
        artifacts.append(_record(frame_path, output))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "source_context_prepared_pending_calibrated_views",
        "source_commit": expected_source_commit,
        "scene_id": installation["scene_id"],
        "source_installation_digest": installation["receipt_digest"],
        "task_objects": [dict(row) for row in task_objects],
        "task_objects_digest": task_digest,
        "artifacts": artifacts,
        "source_identities": [
            {**obj, "identity_receipt_digest": value["receipt_digest"], "target": value["target"]}
            for obj, value in zip(observed_objects, identities, strict=True)
        ],
        "shared_frame_receipt_digest": frame["receipt_digest"] if frame else None,
        "blockers": blockers,
        "provider_mutation_performed": False, "paid_resource_used": False,
        "candidate_policy_queried": False,
        "claim_boundary": {
            "source_correspondence_only": True,
            "metric_registration_independently_qualified": False,
            "method_input": False, "evaluation_authorized": False,
            "native_import_qualified": False, "robot_reachability_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write(output / (SCHEMA_VERSION + ".json"), receipt)
    return receipt
