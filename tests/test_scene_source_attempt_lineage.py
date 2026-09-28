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


@pytest.mark.parametrize("family", list(FAMILIES))
def test_missing_each_available_proof_is_kept_not_absent(family):
    for group in ("snapshot_records", "factory_records", "submission_records"):
        for index in range(len(fixture(family)[group])):
            args = fixture(family)
            removed = args[group].pop(index)
            result = module().join_scene_source_attempt_lineage(**args)
            row = result["attempts"][0]
            assert row["status"] == "kept_unresolved"
            assert row["workspace_membership_bound"] is False
            assert row["preparation_identity"] is None
            assert result["bound_workspace_count"] == 0
            assert removed[0] not in {p["path"] for p in row["source_provenance"]}
            assert len(row["source_provenance"]) == 6


def test_missing_earlier_proof_does_not_hide_available_bad_record_or_edge():
    for role, group in (("release", "snapshot_records"), ("factory", "factory_records"),
                        ("manifest", "submission_records")):
        args = fixture(edits={role: lambda obj: obj.update(schema_version="foreign")})
        args["snapshot_records"] = args["snapshot_records"][1:]
        assert args[group]
        refuses(args)
    args = fixture(edits={"request": lambda obj: obj.update(expected_production_commit="c" * 40)})
    args["factory_records"] = []
    refuses(args)


def test_all_missing_proof_lists_three_kept_reasons():
    args = fixture()
    for group in ("snapshot_records", "factory_records", "submission_records"):
        args[group] = []
    row = module().join_scene_source_attempt_lineage(**args)["attempts"][0]
    assert row["reasons"] == ["source_factory_missing", "source_snapshot_missing", "source_submission_missing"]


def test_public_retained_prefix_uses_exact_raw_snapshot_references():
    result = module().join_scene_source_attempt_lineage(**fixture("public"))
    row = result["attempts"][0]
    assert row["source_family"] == "public" and row["workspace_membership_bound"] is True
    assert row["snapshot_binding_strength"] == "factory_raw_references"
    for role in ("intent", "attempt", "source_binding", "machinery", "release"):
        args = fixture("public", edits={"factory": lambda obj: obj["identity"][role].update(sha256="sha256:" + "c" * 64)})
        refuses(args)


@pytest.mark.parametrize(("role", "edit"), [
    ("binding", lambda obj: obj.update(intent_task_digest="sha256:" + "c" * 64)),
    ("machinery", lambda obj: obj.update(retained_prefix_only_binding_ids=[])),
    ("machinery", lambda obj: obj.update(retained_prefix_only_binding_ids="binding-1")),
    ("factory", lambda obj: obj["identity"].update(factory_started_at_epoch=True)),
    ("intent", lambda obj: obj["request"]["source"].update(kind="mesh")),
])
def test_public_administrative_family_requires_retained_mode_and_owner_task(role, edit):
    refuses(fixture("public", edits={role: edit}))


def test_all_historical_attempts_survive_and_two_aliases_refuse():
    args = fixture(alias="attempts")
    next_args = fixture(attempt_id="source-new", commit="c" * 40)
    for group in ("attempt_records", "snapshot_records", "factory_records", "submission_records"):
        args[group] += next_args[group]
    result = module().join_scene_source_attempt_lineage(**args)
    assert result["attempt_count"] == result["bound_workspace_count"] == 2
    assert [row["attempt_id"] for row in result["attempts"]] == [ATTEMPT, "source-new"]
    assert result["attempts"][0]["attempt_alias"] == "attempts"
    args = fixture()
    path, raw = args["attempt_records"][0]
    args["attempt_records"].append((path.replace("/preparation-attempts/", "/attempts/"), raw))
    refuses(args, "attempt_ambiguous")


@pytest.mark.parametrize("family", list(FAMILIES))
def test_factory_aliases_are_family_specific_and_preserve_both_raw_proofs(family):
    args = fixture(family)
    path, raw = args["factory_records"][0]
    local = path.removesuffix("/factory.json") + "/materialized/factory_receipt.json"
    args["factory_records"].append((local, json.dumps(json.loads(raw), indent=1).encode()))
    if family == "website":
        refuses(args, "factory_path_invalid")
        return
    row = module().join_scene_source_attempt_lineage(**args)["attempts"][0]
    proofs = [p for p in row["source_provenance"] if p["role"] == "factory"]
    assert len(proofs) == 2 and proofs[0]["sha256"] != proofs[1]["sha256"]
    args["factory_records"] = args["factory_records"][1:]
    assert module().join_scene_source_attempt_lineage(**args)["bound_workspace_count"] == 1
    value = json.loads(raw)
    value["claim_scope"] = "changed"
    value["factory_digest"] = canonical_digest(value, digest_field="factory_digest")
    args["factory_records"].append(pair(path, value))
    refuses(args, "factory_ambiguous")


def test_paid_attempt_is_explicitly_out_of_scope_not_guessed_source_workspace():
    args = fixture(edits={"attempt": lambda obj: obj.update(schema_version="task_evaluation_scene_attempt.v1",
        provider="vast", maximum_spend_usd=50, status="reserved")})
    path, raw = args["attempt_records"][0]
    args["attempt_records"] = [(path.replace("/preparation-attempts/", "/attempts/"), raw)]
    for group in ("snapshot_records", "factory_records", "submission_records"):
        args[group] = []
    row = module().join_scene_source_attempt_lineage(**args)["attempts"][0]
    assert row["status"] == "kept_out_of_scope"
    assert row["workspace_path"] is None and row["preparation_identity"] is None
    assert row["reasons"] == ["paid_attempt_out_of_scope"]
    assert "maximum_spend_usd" not in row and "provider" not in row
    args["snapshot_records"] = fixture()["snapshot_records"]
    refuses(args, "record_unmatched")


def test_orphan_foreign_records_are_never_dropped():
    for group in ("snapshot_records", "factory_records", "submission_records"):
        args = fixture()
        path, raw = args[group][0]
        args[group].append((path.replace("/" + ATTEMPT + "/", "/unrelated/"), raw))
        refuses(args, "record_unmatched")


@pytest.mark.parametrize("message", ["scene_lineage_secretvaluetoken", "secretvaluetoken"])
def test_shared_error_translation_keeps_only_known_fixed_codes(monkeypatch, message):
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as shared

    def malformed_error(*_args):
        raise shared.SceneLineageError(message)

    monkeypatch.setattr(shared, "_record", malformed_error)
    refuses(fixture(), "input_invalid")


@pytest.mark.parametrize("target", ["root", "record"])
def test_path_character_cap_precedes_shared_utf8_encoding(monkeypatch, target):
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as shared

    args = fixture()
    oversized = "/" + "a" * shared.MAX_PATH_BYTES
    if target == "root":
        args["roots"]["intent_root"] = oversized
    else:
        args["submission_records"][0] = (oversized, args["submission_records"][0][1])
    original = shared._path

    def encoding_guard(value):
        assert not (isinstance(value, str) and len(value) > shared.MAX_PATH_BYTES), "oversize path reached encoder"
        return original(value)

    monkeypatch.setattr(shared, "_path", encoding_guard)
    refuses(args, "path_invalid")


@pytest.mark.parametrize("raw", [b'{}', b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}',
                                b'{"x":' + b'9' * 400 + b'}', b'{"x":"\\ud800"}', b'\xff', b'[]', b'{'])
def test_malformed_supplied_raw_evidence_refuses_without_text(raw):
    args = fixture()
    args["submission_records"][0] = (args["submission_records"][0][0], raw)
    refuses(args)


@pytest.mark.parametrize("raw", ["{}", bytearray(b"{}"), None, b""])
def test_nonbytes_and_empty_records_refuse_before_parse(raw):
    args = fixture()
    args["submission_records"][0] = (args["submission_records"][0][0], raw)
    refuses(args, "record_invalid")


@pytest.mark.parametrize("field", ["MAX_RECORD_BYTES", "MAX_TOTAL_BYTES", "MAX_RECORDS", "MAX_OUTPUT_BYTES",
                                  "MAX_PATH_BYTES", "MAX_PATH_COMPONENTS"])
def test_shared_defining_module_caps_refuse_without_truncation(monkeypatch, field):
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as shared

    monkeypatch.setattr(shared, field, 1)
    refuses(fixture())


def test_exact_shared_resource_boundaries_are_inclusive(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as shared

    args = fixture("public")
    result = module().join_scene_source_attempt_lineage(**args)
    pairs = [args["intent_record"], *(record for group in ("attempt_records", "snapshot_records",
        "factory_records", "submission_records") for record in args[group])]
    paths = [path for path, _ in pairs] + list(args["roots"].values())
    for field, value in {
        "MAX_RECORD_BYTES": max(len(raw) for _, raw in pairs), "MAX_TOTAL_BYTES": sum(len(raw) for _, raw in pairs),
        "MAX_RECORDS": len(pairs), "MAX_OUTPUT_BYTES": len(shared._encoded(result)),
        "MAX_PATH_BYTES": max(len(path.encode()) for path in paths),
        "MAX_PATH_COMPONENTS": max(len(path[1:].split("/")) for path in paths),
    }.items():
        monkeypatch.setattr(shared, field, value)
    assert module().join_scene_source_attempt_lineage(**args) == result


def test_groups_counts_and_aggregate_preflight_precede_parse_hash(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as shared

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preflight bypassed")

    args = fixture()
    monkeypatch.setattr(shared, "_record", forbidden)
    args["attempt_records"] = iter(args["attempt_records"])
    refuses(args, "records_limit")
    args = fixture()
    monkeypatch.setattr(shared, "MAX_RECORDS", 1)
    refuses(args, "records_limit")
    monkeypatch.setattr(shared, "MAX_RECORDS", 10_000)
    monkeypatch.setattr(shared, "MAX_TOTAL_BYTES", 1)
    refuses(args, "bytes_limit")


@pytest.mark.parametrize("group", ["intent_record", "attempt_records", "snapshot_records", "factory_records", "submission_records"])
@pytest.mark.parametrize("path", ["/retained/../foreign", "/retained//foreign", "/retained/<redacted>",
                                 "/retained/foreign\\path", "/retained/\x00secret", "/retained/\ud800"])
def test_all_role_paths_obey_canonical_lexical_policy(group, path):
    args = fixture()
    if group == "intent_record":
        args[group] = (path, args[group][1])
    else:
        args[group][0] = (path, args[group][0][1])
    refuses(args, "path_invalid")


@pytest.mark.parametrize("role", ["intent_root", "factory_output_root"])
def test_installed_root_keys_and_paths_are_explicit(role):
    args = fixture()
    args["roots"][role] = "/retained/../foreign"
    refuses(args, "path_invalid")
    args = fixture()
    del args["roots"][role]
    refuses(args, "parameters_invalid")


def test_duplicate_paths_and_empty_supplied_scope_are_explicit():
    args = fixture()
    args["snapshot_records"].append(args["snapshot_records"][0])
    refuses(args, "record_duplicate")
    args = fixture()
    for group in ("attempt_records", "snapshot_records", "factory_records", "submission_records"):
        args[group] = []
    result = module().join_scene_source_attempt_lineage(**args)
    assert result["attempt_count"] == result["bound_workspace_count"] == 0
    assert result["historical_attempt_inventory_complete"] is False


def test_exact_output_and_reordered_inputs_have_no_presence_or_retirement_claims():
    from blueprint_pipeline import task_evaluation_scene_preparation_lineage as shared

    args = fixture("public")
    other = fixture("public", attempt_id="source-next", commit="c" * 40)
    for group in ("attempt_records", "snapshot_records", "factory_records", "submission_records"):
        args[group] += other[group]
    result = module().join_scene_source_attempt_lineage(**args)
    assert set(result) == {"schema_version", "status", "scope", "intent_id", "intent_digest", "intent_provenance",
        "attempt_count", "bound_workspace_count", "attempts", "mutations", "execution_authorized",
        "complete_scene_inventory", "historical_attempt_inventory_complete", "payload_presence_checked",
        "payload_members_verified", "publication_readback_checked", "remote_availability_checked", "finished_state_checked",
        "references_checked", "consumer_fence_checked", "requires_fresh_reference_check"}
    for flag in ("execution_authorized", "complete_scene_inventory", "historical_attempt_inventory_complete",
                 "payload_presence_checked", "payload_members_verified", "publication_readback_checked",
                 "remote_availability_checked", "finished_state_checked", "references_checked", "consumer_fence_checked"):
        assert result[flag] is False
    assert result["requires_fresh_reference_check"] is True and result["mutations"] == 0
    for row in result["attempts"]:
        assert set(row) == {"attempt_id", "attempt_digest", "attempt_schema", "attempt_alias", "source_commit",
            "runtime_digest", "input_digest", "source_family", "status", "reasons", "workspace_path",
            "workspace_membership_bound", "snapshot_binding_strength", "preparation_identity", "source_provenance"}
        assert set(row["preparation_identity"]) == {"preparation_id", "request_digest", "team_namespace", "scene_id",
                                                    "task_id", "expected_production_commit"}
        for proof in [result["intent_provenance"], *row["source_provenance"]]:
            assert set(proof) == {"role", "path", "sha256", "size_bytes", "seal_field", "seal_digest"}
            if proof["role"] == "submission_request":
                assert proof["seal_field"] is proof["seal_digest"] is None
    for group in ("attempt_records", "snapshot_records", "factory_records", "submission_records"):
        args[group].reverse()
    assert shared._encoded(module().join_scene_source_attempt_lineage(**args)) == shared._encoded(result)


@pytest.mark.parametrize("family", ["completed", "public"])
@pytest.mark.slow
def test_first_source_join_in_fresh_process_avoids_runtime_imports_and_filesystem(family):
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = '''
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("fixtures", sys.argv[1])
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
args = fixtures.fixture(sys.argv[2])
def forbidden(*args, **kwargs):
    raise AssertionError("pure retained source join invoked filesystem API")
for name in ("resolve", "stat", "lstat", "read_bytes", "read_text", "write_bytes", "write_text"):
    setattr(Path, name, forbidden)
result = fixtures.module().join_scene_source_attempt_lineage(**args)
assert result["bound_workspace_count"] == 1
for name in ("task_evaluation_scene_intake", "task_evaluation_scene_progression", "task_evaluation_controls_autoprovision",
             "task_evaluation_public_scene_attempt_factory", "task_evaluation_launch_preparation_contract"):
    assert "blueprint_pipeline." + name not in sys.modules, name
'''
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).absolute()), family],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_join_never_reads_publishes_repairs_or_starts_processes(monkeypatch):
    import builtins
    import os
    from pathlib import Path
    import subprocess

    args, api = fixture("public"), module()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure retained source join invoked an effect")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        for name in ("open", "stat", "lstat", "mkdir", "unlink", "remove", "rename", "replace", "listdir", "scandir"):
            guard.setattr(os, name, forbidden)
        for name in ("resolve", "stat", "lstat", "read_bytes", "read_text", "write_bytes", "write_text"):
            guard.setattr(Path, name, forbidden)
        guard.setattr(subprocess, "run", forbidden)
        assert api.join_scene_source_attempt_lineage(**args)["bound_workspace_count"] == 1
