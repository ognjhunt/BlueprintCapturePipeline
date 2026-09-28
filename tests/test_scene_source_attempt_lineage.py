# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_source_attempt_lineage.py
#   src/blueprint_pipeline/task_evaluation_scene_preparation_lineage.py
"""ADP-009D/day-28: supplied source lineage is never eviction authority."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

ROOTS = {"intent_root": "/retained/intents", "factory_output_root": "/retained/factories"}
INTENT, ATTEMPT, COMMIT = "intent-1", "historical-recovery-7", "a" * 40
DIGEST = "sha256:" + "b" * 64
FAMILIES = {
    "completed": ("task_evaluation_completed_scene_source.v1", "task_evaluation_completed_scene_machinery.v1",
                  "task_evaluation_completed_scene_attempt_factory.v1"),
    "website": ("website_scene_source_binding.v1", "task_evaluation_website_scene_machinery.v1",
                "website_scene_attempt_factory.v1"),
    "public": ("task_evaluation_public_source_binding.v1", "task_evaluation_public_scene_machinery.v1",
               "task_evaluation_public_scene_attempt_factory.v1"),
}


def module():
    return importlib.import_module("blueprint_pipeline.task_evaluation_scene_source_attempt_lineage")


def pair(path, value):
    return path, json.dumps(value, sort_keys=True, allow_nan=False).encode()


def ref(record):
    path, raw = record
    return {"path": path, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def fixture(family="completed", *, attempt_id=ATTEMPT, commit=COMMIT, alias="preparation-attempts", edits=None):
    """Reseal every descendant/raw reference so failures exercise the intended edge."""
    edits = edits or {}

    def edit(role, value, seal=None, cross=False):
        value = copy.deepcopy(value)
        if role in edits:
            edits[role](value)
        if seal:
            digest = cross_runtime_canonical_digest if cross else canonical_digest
            value[seal] = digest(value, digest_field=seal)
        return value

    base = ROOTS["intent_root"] + "/" + INTENT
    workspace = ROOTS["factory_output_root"] + "/" + INTENT + "/" + attempt_id
    task = {"task_id": "task-1", "historical_number": 1.0}
    owner = {"user_id": "owner@site", "organization_id": "different:org"}
    intent = edit("intent", {"schema_version": "task_evaluation_scene_intent.v1", "intent_id": INTENT,
        "request": {"schema_version": "task_evaluation_scene_intake_request.v1", "submission_id": "submission-1",
                    "owner": owner, "task": task, "source": {"kind": "public_scene" if family == "public" else "mesh",
                        "binding_id": "binding-1", "content_digest": DIGEST}},
        "source_content_digest": DIGEST, "task_content_digest": cross_runtime_canonical_digest(task)},
        "intent_digest", True)
    binding = {"schema_version": FAMILIES[family][0], "binding_id": intent["request"]["source"]["binding_id"],
        "source_content_digest": intent["request"]["source"]["content_digest"], "owner": intent["request"]["owner"]}
    if family == "public":
        binding.update(intent_task_digest=intent["task_content_digest"], status="admitted_for_private_processing")
    else:
        binding.update(intent_digest=intent["intent_digest"], task_digest=intent["task_content_digest"])
    if family == "completed":
        binding.update(source_kind=intent["request"]["source"]["kind"], status="source_task_objects_bound")
    binding = edit("binding", binding, "binding_digest")
    machinery = edit("machinery", {"schema_version": FAMILIES[family][1],
        **({"retained_prefix_only_binding_ids": [binding["binding_id"]]} if family == "public" else {})}, "machinery_digest")
    release = edit("release", {"schema_version": "task_evaluation_public_scene_release_binding.v1",
        "source_commit": commit, "runtime_digest": DIGEST}, "release_digest")
    attempt = edit("attempt", {"schema_version": "task_evaluation_scene_preparation_attempt.v1",
        "intent_id": INTENT, "intent_digest": intent["intent_digest"], "attempt_id": attempt_id,
        "source_commit": commit, "runtime_digest": DIGEST, "input_digest": binding["binding_digest"],
        "provider": "control_plane", "maximum_spend_usd": 0, "status": "preparation_only",
        "paid_authority_granted": False, "provider_allocation_permitted": False}, "attempt_digest", True)
    request = edit("request", {"schema_version": "task_evaluation_launch_preparation_request.v1",
        "run_mode": "scene_configuration", "preparation_id": "prep-1", "scene_intent_digest": intent["intent_digest"],
        "expected_production_commit": commit, "team_namespace": "independent-team",
        "scene": {"identity": {"id": "scene-1"}}, "task": {"identity": {"id": "task-1"}},
        "publication": {"input_namespace": "retained-namespace"}})
    manifest = edit("manifest", {"schema_version": "task_evaluation_scene_configuration_submission_manifest.v1",
        "status": "validated_pending_production_publication_and_submission", "source_commit": commit,
        "request_digest": canonical_digest(request), "input_namespace": request["publication"]["input_namespace"]}, "manifest_digest")
    intent_pair = pair(base + "/intent.json", intent)
    attempt_pair = pair(base + "/" + alias + "/" + attempt_id + ".json", attempt)
    snapshots = [pair(workspace + "/" + name, value) for name, value in (
        ("source_binding.json", binding), ("machinery.json", machinery), ("release_binding.json", release))]
    submissions = [pair(workspace + "/materialized/submission/" + name, value) for name, value in (
        ("scene_configuration_preparation_request.v1.json", request), ("bundle_manifest.v1.json", manifest))]
    factory = {"schema_version": FAMILIES[family][2], "status": "publication_ready",
        "intent_digest": intent["intent_digest"], "attempt_digest": attempt["attempt_digest"], "source_commit": commit,
        "provider_mutation_performed": False, "submission_request": ref(submissions[0]), "submission_manifest": ref(submissions[1])}
    if family == "completed":
        factory["source_kind"] = binding["source_kind"]
    if family == "public":
        factory["identity"] = {"intent": ref(intent_pair), "attempt": ref(attempt_pair),
            **{role: ref(row) for role, row in zip(("source_binding", "machinery", "release"), snapshots)},
            "factory_started_at_epoch": 12.0}
    factory = edit("factory", factory, "factory_digest")
    return {"intent_id": INTENT, "intent_record": intent_pair, "attempt_records": [attempt_pair],
        "snapshot_records": snapshots, "factory_records": [pair(workspace + "/factory.json", factory)],
        "submission_records": submissions, "roots": dict(ROOTS)}


def refuses(args, code=None):
    api = module()
    with pytest.raises(api.SceneSourceLineageError) as raised:
        api.join_scene_source_attempt_lineage(**args)
    assert str(raised.value).startswith("scene_source_lineage_")
    assert len(str(raised.value)) < 100
    if code:
        assert str(raised.value) == "scene_source_lineage_" + code


@pytest.mark.parametrize("family", ["completed", "website"])
def test_retained_administrative_factory_joins_exact_workspace_without_authority(family):
    args = fixture(family)
    result = module().join_scene_source_attempt_lineage(**args)
    row = result["attempts"][0]
    assert result["attempt_count"] == result["bound_workspace_count"] == 1
    assert row["workspace_path"] == ROOTS["factory_output_root"] + "/" + INTENT + "/" + ATTEMPT
    assert row["source_family"] == family
    assert row["workspace_membership_bound"] is True
    assert row["snapshot_binding_strength"] == "sealed_snapshots_at_expected_paths"
    assert row["preparation_identity"]["team_namespace"] == "independent-team"
    assert len(row["source_provenance"]) == 7
    assert result["mutations"] == 0 and result["execution_authorized"] is False


@pytest.mark.parametrize(("role", "field", "value"), [
    ("attempt", "provider", "vast"), ("attempt", "maximum_spend_usd", True),
    ("attempt", "paid_authority_granted", 0), ("attempt", "provider_allocation_permitted", True),
    ("attempt", "input_digest", "sha256:" + "c" * 64), ("attempt", "runtime_digest", "sha256:" + "c" * 64),
    ("attempt", "source_commit", "c" * 40), ("attempt", "intent_id", "foreign"),
    ("binding", "owner", {"user_id": "foreign", "organization_id": "different:org"}),
    ("binding", "binding_id", "other-binding"), ("binding", "task_digest", "sha256:" + "c" * 64),
    ("binding", "source_content_digest", "sha256:" + "c" * 64), ("binding", "source_kind", "gaussian_splat"),
    ("machinery", "schema_version", FAMILIES["public"][1]), ("release", "runtime_digest", "sha256:" + "c" * 64),
    ("release", "source_commit", "c" * 40), ("factory", "source_kind", "gaussian_splat"),
    ("factory", "provider_mutation_performed", 0), ("factory", "attempt_digest", "sha256:" + "c" * 64),
    ("request", "expected_production_commit", "c" * 40), ("request", "scene_intent_digest", "sha256:" + "c" * 64),
    ("manifest", "request_digest", "sha256:" + "c" * 64), ("manifest", "input_namespace", "other/namespace"),
])
def test_resealed_foreign_edges_refuse(role, field, value):
    refuses(fixture(edits={role: lambda obj: obj.update({field: value})}))


@pytest.mark.parametrize("extra", [{"extra": False}, {"maximum_spend_usd": 1}, {"status": "reserved"}])
def test_administrative_exact_fields_and_zero_contract(extra):
    refuses(fixture(edits={"attempt": lambda obj: obj.update(extra)}))


def test_namespace_requires_historical_identifier_not_relative_path():
    refuses(fixture(edits={"request": lambda obj: obj["publication"].update(input_namespace="not/a/namespace")}))
