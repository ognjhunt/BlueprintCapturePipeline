# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_preparation_activation_references.py
"""Supplied identities and canonical seals; no queue or payload observation."""
import hashlib
import json
from dataclasses import replace

import pytest

from blueprint_pipeline import control_plane_preparation_activation_references as subject

D = "sha256:" + "1" * 64
C = "a" * 40


def digest(value, field=None):
    value = {k: v for k, v in value.items() if k != field}
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sealed(value, field):
    return {**value, field: digest(value, field)}


def reference(uri="gs://bucket/object", size=1):
    return {"uri": uri, "digest": D, "size_bytes": size}


def preparation_request(**changes):
    return {"schema_version": "task_evaluation_launch_preparation_request.v1",
            "preparation_id": "prep", "team_namespace": "team", "run_id": "run",
            "expected_production_commit": C, "run_mode": "scene_configuration",
            "scene": {"mode": "reuse_configured_revision", "configured_revision": reference()},
            "construction": {"mode": "reuse_configured_scene"},
            "task": {"binding_mode": "reuse_configuration_template",
                     "subject": {"mode": "configured_scene_object"}},
            "sensors": {"configuration": reference()},
            "runtime": {"health_protocol": reference(), "mounts": []},
            "execution_adapter": {"runtime_source_bundle": reference()},
            "publication": {}, "spend": {}, **changes}


def activation_request(**changes):
    return {"schema_version": "task_evaluation_launch_activation_request.v1",
            "activation_id": "activate", "team_namespace": "team", "expected_production_commit": C,
            "lane": "task_evaluation_scene_configuration",
            "preparation": {"preparation_id": "prep", "request_digest": D, "result_digest": D},
            "release_window": reference(), "lineage": {"mode": "initial_project",
                "project_spend_reconciliation": reference(), "initial_provider_zero": reference()},
            "authorization": {"reference": "owner-reviewed"}, "requested_mutations": {}, **changes}


def record(family="preparation", role="envelope", request=None, state="materialized", **changes):
    root = "/queues/" + family
    request = request or (preparation_request() if family == "preparation" else activation_request())
    identifier = request[family + "_id"]
    request_digest = digest(request)
    stem = identifier + "-" + request_digest[7:]
    if family == "activation" and len((stem + ".json").encode()) > 255:
        stem = "activation-" + hashlib.sha256(identifier.encode()).hexdigest() + "-" + request_digest[7:]
    if role == "identity":
        value = {"schema_version": "task_evaluation_launch_" + family + "_identity.v1",
                 family + "_id": identifier, "request_digest": request_digest}
        path = root + "/identities/" + identifier + ".json"
        field = "identity_digest"
    elif role == "envelope":
        value = {"schema_version": "task_evaluation_launch_" + family + "_envelope.v1",
                 "request": request, "request_digest": request_digest,
                 "submitted_by": "owner", "submitted_at_iso": "2026-09-28T00:00:00Z",
                 "provider_mutation_performed_inside_intake": False,
                 "catalog_mutation_performed_inside_intake": False}
        if family == "activation":
            value.update(standing_authorization_published_inside_intake=False, paid_execution_requested=False)
        path = root + "/" + state + "/" + stem + ".json"
        field = "envelope_digest"
    else:
        value = {"schema_version": "task_evaluation_launch_" + family + "_result.v1",
                 family + "_id": identifier, "status": "blocked", "blockers": ["waiting"]}
        field = "result_digest"
        path = root + "/results/" + stem + ".json"
    value.update(changes)
    value = sealed(value, field)
    if role == "result_conflict":
        path = root + "/results/conflicts/" + stem + "-" + value[field][7:] + ".json"
    return subject.RetainedReferenceRecord(family, root, role, path, json.dumps(value).encode())


def observe(*rows, **kwargs):
    contracts = [subject.ReferenceFamilyContract(family, "/queues/" + family)
                 for family in sorted({row.family for row in rows})] or [subject.ReferenceFamilyContract("preparation", "/queues/preparation")]
    return subject.interpret_preparation_activation_references(contracts, list(rows), **kwargs)


@pytest.mark.parametrize("family,role", [(f, r) for f in ("preparation", "activation")
                                        for r in ("envelope", "identity", "result", "result_conflict")])
def test_each_source_role_retains_exact_raw_identity(family, role):
    row = record(family, role, state="pending")
    result = observe(row)
    proof = result.records[0].source
    assert (proof.row_path, proof.raw_sha256, proof.raw_size_bytes) == (
        row.row_path, "sha256:" + hashlib.sha256(row.raw_bytes).hexdigest(), len(row.raw_bytes))
    assert result.mutations == 0 and not result.references_clear and not result.execution_authorized
    assert result.records[0].disposition != "invalid"


@pytest.mark.parametrize("family", ["preparation", "activation"])
def test_canonical_seals_are_not_raw_format_identity(family):
    row = record(family, "result")
    formatted = replace(row, raw_bytes=(json.dumps(json.loads(row.raw_bytes), indent=2) + "\n").encode())
    result = observe(row, formatted)
    assert len(result.records) == 2
    assert len({r.source.raw_sha256 for r in result.records}) == 2
    assert len({r.canonical_digest for r in result.records}) == 1
    if family == "activation":
        assert "immutable_record_conflict" in result.blockers
    else:
        assert "immutable_record_conflict" not in result.blockers


def test_exact_duplicate_raw_identity_is_api_refusal():
    row = record()
    with pytest.raises(subject.PreparationActivationReferenceError, match="reference_duplicate_input"):
        observe(row, row)


@pytest.mark.parametrize("change", [
    {"row_path": "/queues/preparation/pending/../x.json"}, {"raw_bytes": "{}"},
    {"raw_bytes": b""}, {"family": "construction"}, {"role": "progress"},
    {"observed_identity": (1, 2, True, 4, 5)}, {"observed_identity": (1, 2, 999, 4, 5)},
])
def test_malformed_api_inputs_are_fixed_refusals(change):
    with pytest.raises(subject.PreparationActivationReferenceError, match="reference_parameters_invalid"):
        observe(replace(record(), **change))


@pytest.mark.parametrize("raw", [b"{}", b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b'\xff', b'{"a":"\\ud800"}'])
def test_invalid_supplied_json_is_retained_with_fixed_disposition(raw):
    result = observe(replace(record(), raw_bytes=raw))
    assert result.records[0].disposition == "invalid"
    assert not result.complete_supplied_supported_projection


@pytest.mark.parametrize("field,value", [("schema_version", "unknown"), ("envelope_digest", D),
                                         ("request_digest", D), ("provider_mutation_performed_inside_intake", True)])
def test_wrong_schema_seal_request_or_intake_declaration_cannot_join(field, value):
    row = record()
    data = json.loads(row.raw_bytes)
    data[field] = value
    if field != "envelope_digest":
        data = sealed(data, "envelope_digest")
    result = observe(replace(row, raw_bytes=json.dumps(data).encode()))
    assert result.records[0].disposition == "invalid"


def test_hashed_activation_filename_supports_actual_192_character_id():
    result = observe(record("activation", request=activation_request(activation_id="a" * 192), state="prepared"))
    assert result.records[0].disposition != "invalid"


def test_preparation_does_not_invent_activation_filename_fallback():
    with pytest.raises(subject.PreparationActivationReferenceError):
        observe(record(request=preparation_request(preparation_id="p" * 192)))


def test_versions_and_immutable_state_conflicts_are_order_independent():
    first = record()
    second = replace(first, row_path=first.row_path.replace("materialized", "completed"))
    assert observe(first, second) == observe(second, first)
    assert "immutable_record_conflict" in observe(first, second).blockers


def test_empty_supplied_set_never_claims_observed_empty_queue():
    result = observe()
    assert not result.general_reference_inventory_complete
    assert not result.full_request_schema_verified and not result.payload_bytes_verified
    assert result.scope == "preparation_activation_supplied_reference_contracts_only"
