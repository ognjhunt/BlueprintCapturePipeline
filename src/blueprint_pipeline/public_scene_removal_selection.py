"""Source-removal selections, deliberately separate from robot/task qualification.

These host-resident, digest-bound selections authorize no evaluation. The legacy
dual-task validators remain unchanged; only source-preparation consumers use
the explicit compatibility wrappers below.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .decision_evidence_contracts import canonical_digest, canonical_json
from .dual_task_rehearsal_contract import (
    DualTaskRehearsalContractError, validate_scene_freeze, validate_task_freeze,
    validate_task_freeze_set,
)
from .public_scene_host_input_intake import _APPROVED_RIGHTS_STATES, _rights_state
from .task_evaluation_scene_configuration_submission_inputs import (
    beneath, checked_file, read, sha, slug, source_inputs,
)

SCENE_SCHEMA = "public_scene_removal_scene_selection.v1"
TASK_SCHEMA = "public_scene_removal_task_selection.v1"
ADAPTER = "source_removal_selection_and_standard_splat_v1"
BOUNDARIES = {
    "evaluation_authorized": False,
    "robot_reachability_established": False,
    "candidate_policy_queried": False,
    "native_import_qualified": False,
    "raw_source_uploaded": False,
}
EVIDENCE_FIELDS = ("task_request", "installation_receipt", "publisher_intake",
                   "source_preparation_receipt")


class PublicSceneRemovalSelectionError(DualTaskRehearsalContractError):
    """A source-only selection is not justified by the retained evidence."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise PublicSceneRemovalSelectionError(["public_scene_removal_selection_" + code])


def _record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def _bound(record: Mapping[str, Any]) -> Path:
    _require(isinstance(record, Mapping), "evidence_invalid")
    return checked_file(str(record.get("path") or ""), dict(record))


def _rights(installation: dict[str, Any]) -> None:
    receipts = {}
    source_rows = []
    for row in installation["files"]:
        # The canonical installer preserves receipt_id but omits archive-only
        # kind markers. Never let a source-role row masquerade as authority.
        if "receipt_id" not in row and row.get("kind") != "rights_receipt":
            source_rows.append(row)
            continue
        _require(not row.get("role") and "rights_receipt_ids" not in row
                 and row.get("kind") in {None, "rights_receipt"}, "rights_invalid")
        identifier = row.get("receipt_id")
        _require(isinstance(identifier, str) and bool(identifier)
                 and identifier not in receipts, "rights_invalid")
        path = checked_file(beneath(Path(installation["destination_root"]),
                                    row["relative_path"]), row)
        value = read(path)
        _require(value.get("schema_version") == "public_scene_rights_authority.v1"
                 and _rights_state(value) in _APPROVED_RIGHTS_STATES
                 and value.get("agent_accepted_terms") is not True
                 and isinstance(value.get("authorized_source_sha256"), list),
                 "rights_invalid")
        receipts[identifier] = value
    _require(bool(receipts), "rights_missing")
    for row in source_rows:
        identifiers = row.get("rights_receipt_ids")
        _require(isinstance(identifiers, list) and bool(identifiers)
                 and all(identifier in receipts for identifier in identifiers)
                 and any(row["sha256"] in receipts[identifier]["authorized_source_sha256"]
                         for identifier in identifiers), "rights_source_join_invalid")


def _source_context(evidence: Mapping[str, Any], commit: str) -> tuple[dict, dict]:
    _require(set(evidence) == set(EVIDENCE_FIELDS), "evidence_invalid")
    paths = {key: _bound(evidence[key]) for key in EVIDENCE_FIELDS}
    task = read(paths["task_request"])
    installation = read(paths["installation_receipt"], digest_field="receipt_digest")
    _rights(installation)
    context = source_inputs(
        installation_path=paths["installation_receipt"],
        publisher_path=paths["publisher_intake"],
        preparation_path=paths["source_preparation_receipt"],
        task=task, commit=commit,
    )
    frames = [(path, value) for path, value in context["artifacts"]
              if value.get("schema_version") == "interiorgs_sage_shared_frame_candidate.v1"]
    _require(len(frames) == 1, "shared_frame_missing")
    path, frame = frames[0]
    _require(frame.get("receipt_digest") == canonical_digest(frame, digest_field="receipt_digest")
             and frame.get("source_digests") == {
                 "interiorgs_labels": context["raw"]["semantic_metadata"]["sha256"],
                 "sage_collision_usd": context["raw"]["collision_usd"]["sha256"],
             }, "shared_frame_source_mismatch")
    for key in ("subject", "support"):
        identity = context["identities"][key]
        matches = [row for row in frame.get("correspondences", [])
                   if str(row.get("interiorgs_instance_id")) ==
                   str(identity["receipt"]["target"]["interiorgs_instance_id"])]
        _require(len(matches) == 1
                 and matches[0].get("identity_receipt_digest") == identity["receipt"]["receipt_digest"]
                 and matches[0].get("sage_prim_path") == identity["match"]["prim_path"],
                 "shared_frame_identity_mismatch")
    context["registered_frame"] = {**_record(path), "receipt_digest": frame["receipt_digest"]}
    return task, context


def _components(context: dict) -> dict[str, Any]:
    def component(role: str) -> dict[str, Any]:
        row = context["raw"][role]
        parts = urlparse(row["publisher_url"]).path.split("/")
        return {"repository": "/".join(parts[2:4]), "revision": row["publisher_revision"],
                "sha256": row["sha256"], "size_bytes": row["size_bytes"]}
    interiorgs = component("appearance_3dgs")
    interiorgs["supporting_files"] = {
        name: {"sha256": context["raw"][role]["sha256"],
               "size_bytes": context["raw"][role]["size_bytes"]}
        for name, role in (("labels", "semantic_metadata"), ("structure", "scene_structure"))
    }
    return {"interiorgs": interiorgs, "sage_collision": component("collision_usd")}


def _task_fields(task: dict, context: dict) -> dict[str, Any]:
    subject = context["identities"]["subject"]
    support = context["identities"]["support"]
    target = subject["receipt"]["target"]
    task_id = slug(task["task_identity"]["id"])
    replacement = slug(task["subject"]["replacement_identity"]["id"])
    return {
        "task_id": task_id,
        "source_object": {
            "instance_id": str(target["interiorgs_instance_id"]),
            "semantic_label": target["semantic_label"],
            "observed_bounds_world_m": {
                "minimum": target["world_aabb_min_m"], "maximum": target["world_aabb_max_m"],
            },
            "support_or_attachment_id": str(support["receipt"]["target"]["interiorgs_instance_id"]),
            "collision_identity_receipt_digest": subject["receipt"]["receipt_digest"],
            "support_receipt_digest": support["receipt"]["receipt_digest"],
        },
        "removal_plan": {
            "removal_id": task_id + "-removal",
            "mask_set_id": task_id + "-source-masks",
            "source_collider_prim_path": subject["match"]["prim_path"],
            "collider_deletion_id": task_id + "-collider-deletion",
            "replacement_asset_id": replacement,
            "replacement_qualification_id": replacement + "-pending-qualification",
        },
    }


def validate_removal_scene_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    scene = dict(value)
    _require(scene.get("schema_version") == SCENE_SCHEMA
             and scene.get("scene_freeze_digest") ==
             canonical_digest(scene, digest_field="scene_freeze_digest")
             and all(scene.get(key) is val for key, val in BOUNDARIES.items()), "scene_invalid")
    _task, context = _source_context(scene["source_evidence"], scene["source_commit"])
    _require(scene.get("selected_scene_id") == context["scene_id"]
             and scene.get("source_components") == _components(context)
             and scene.get("registered_frame") == context["registered_frame"], "scene_source_mismatch")
    return scene


def validate_removal_task_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    task = dict(value)
    _require(task.get("schema_version") == TASK_SCHEMA
             and task.get("task_freeze_digest") == canonical_digest(task, digest_field="task_freeze_digest")
             and all(task.get(key) is val for key, val in BOUNDARIES.items()), "task_invalid")
    scene_path = _bound(task["scene_selection"])
    scene = validate_removal_scene_selection(read(scene_path))
    owner_task, context = _source_context(scene["source_evidence"], scene["source_commit"])
    fields = _task_fields(owner_task, context)
    _require(task.get("scene_freeze_digest") == scene["scene_freeze_digest"]
             and all(task.get(key) == item for key, item in fields.items()), "task_source_mismatch")
    forbidden = {"franka_placement_packet_digest", "visibility_receipt_digest"}
    _require(not forbidden.intersection(task["source_object"]), "robot_evidence_forbidden")
    return task


def validate_source_preparation_scene_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") == SCENE_SCHEMA:
        return validate_removal_scene_selection(value)
    return validate_scene_freeze(value)


def validate_source_preparation_task_selection(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") == TASK_SCHEMA:
        return validate_removal_task_selection(value)
    return validate_task_freeze(value)


def validate_source_preparation_task_selection_set(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if all(value.get("schema_version") != TASK_SCHEMA for value in values):
        return validate_task_freeze_set(values)
    _require(1 <= len(values) <= 5 and all(value.get("schema_version") == TASK_SCHEMA for value in values),
             "mixed_or_invalid_selection_set")
    tasks = [validate_removal_task_selection(value) for value in values]
    _require(len({task["scene_freeze_digest"] for task in tasks}) == 1
             and len({task["task_id"] for task in tasks}) == len(tasks)
             and len({task["source_object"]["instance_id"] for task in tasks}) == len(tasks),
             "selection_set_identity_mismatch")
    for key in ("removal_id", "mask_set_id", "source_collider_prim_path", "collider_deletion_id",
                "replacement_asset_id", "replacement_qualification_id"):
        _require(len({task["removal_plan"][key] for task in tasks}) == len(tasks),
                 "selection_set_shared_removal")
    result = {
        "schema_version": "public_scene_removal_selection_set.v1",
        "scene_freeze_digest": tasks[0]["scene_freeze_digest"],
        "task_count": len(tasks), "maximum_task_count": 5,
        "task_freeze_digests": sorted(task["task_freeze_digest"] for task in tasks),
        "task_ids": sorted(task["task_id"] for task in tasks), **BOUNDARIES, "set_digest": "",
    }
    result["set_digest"] = canonical_digest(result, digest_field="set_digest")
    return result


def materialize_public_scene_removal_selections(
    *, task_request_path: str | Path, installation_receipt_path: str | Path,
    publisher_intake_path: str | Path, source_preparation_receipt_path: str | Path,
    expected_production_commit: str, output_root: str | Path,
) -> dict[str, Any]:
    """Derive source selections from real installed bytes, never robot evidence."""
    evidence = {name: _record(Path(path)) for name, path in (
        ("task_request", task_request_path), ("installation_receipt", installation_receipt_path),
        ("publisher_intake", publisher_intake_path),
        ("source_preparation_receipt", source_preparation_receipt_path),
    )}
    task, context = _source_context(evidence, expected_production_commit)
    output = Path(output_root)
    _require(not output.exists() and not any(path.is_symlink() for path in (output, *output.parents)),
             "output_exists_or_unsafe")
    output.mkdir(parents=True)
    scene = {
        "schema_version": SCENE_SCHEMA, "source_commit": expected_production_commit,
        "selected_scene_id": context["scene_id"], "source_evidence": evidence,
        "source_components": _components(context), "registered_frame": context["registered_frame"],
        **BOUNDARIES, "scene_freeze_digest": "",
    }
    scene["scene_freeze_digest"] = canonical_digest(scene, digest_field="scene_freeze_digest")
    validate_removal_scene_selection(scene)
    scene_path = output / (SCENE_SCHEMA + ".json")
    with scene_path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(scene) + "\n")
    selection = {"schema_version": TASK_SCHEMA, **_task_fields(task, context),
                 "scene_selection": _record(scene_path), "scene_freeze_digest": scene["scene_freeze_digest"],
                 **BOUNDARIES, "task_freeze_digest": ""}
    selection["task_freeze_digest"] = canonical_digest(selection, digest_field="task_freeze_digest")
    validate_removal_task_selection(selection)
    task_path = output / (TASK_SCHEMA + ".json")
    with task_path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(selection) + "\n")
    return {"scene_selection": _record(scene_path), "task_selection": _record(task_path),
            "registered_frame": context["registered_frame"],
            "scene_freeze_digest": scene["scene_freeze_digest"],
            "task_freeze_digest": selection["task_freeze_digest"], **BOUNDARIES}


def main(argv: Sequence[str] | None = None) -> int:
    """Derive removal-only source selections from installed publisher bytes."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-request", required=True)
    parser.add_argument("--installation-receipt", required=True)
    parser.add_argument("--publisher-intake", required=True)
    parser.add_argument("--source-preparation-receipt", required=True)
    parser.add_argument("--expected-production-commit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = materialize_public_scene_removal_selections(
        task_request_path=args.task_request,
        installation_receipt_path=args.installation_receipt,
        publisher_intake_path=args.publisher_intake,
        source_preparation_receipt_path=args.source_preparation_receipt,
        expected_production_commit=args.expected_production_commit,
        output_root=args.output_root,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
