from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
import json
import time

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.owned_capture_atlas_context import build_owned_capture_atlas_context
from blueprint_pipeline.owned_capture_atlas_project import (
    bind_owned_atlas_project,
    owned_atlas_project_checkpoint,
    validate_owned_atlas_configuration,
)
from blueprint_pipeline.website_atlas_pose import AtlasClient


@pytest.fixture
def context():
    return build_owned_capture_atlas_context(plan_id="owned-plan", source={
        "receipt": {"capture_session_id": "owned-session", "intake_id": "owned-intake",
                    "capture_digest": "sha256:" + "1" * 64, "envelope_digest": "sha256:" + "2" * 64},
        "qa": {"qa_report_digest": "sha256:" + "3" * 64},
        "object_manifest": {"manifest_digest": "sha256:" + "4" * 64},
        "envelope": {
            "capture_authority_profile": "monocular_video",
            "known_task_specification": {"description": "Place the block and close the box.",
                                         "success_parameters_status": "proposed_unconfirmed"},
            "governance": {"rights": "accepted", "consent": "not_required", "privacy": "cleared"},
            "coordinate_frame_declaration": {"metric_scale": "unverified"},
        },
    })


def _seal(configuration):
    authority = configuration["authority"]
    authority["authorization_digest"] = cross_runtime_canonical_digest(authority, digest_field="authorization_digest")
    return configuration


@pytest.fixture
def configuration(context):
    return _seal({
        "schema_version": "owned_capture_atlas_configuration.v1",
        "task_context_digest": context["context_digest"],
        "source_video_digest": context["source_capture"]["capture_digest"],
        "dispatch_authorized": False, "grant_creation_allowed": False,
        "authority": {
            "schema_version": "owned_capture_atlas_authorization.v1",
            "authorization_basis": "explicit_company_source_approval",
            "request_id": context["request_id"], "capture_id": context["capture_id"],
            "owner_user_id": "owner", "accepted_by": "owner",
            "task_context_digest": context["context_digest"], "source_capture": deepcopy(context["source_capture"]),
            "source_video_digest": context["source_capture"]["capture_digest"],
            "provider_identity": {"account_id": "account-approved", "project_id": "project-approved"},
            "company_authorization_digest": "sha256:" + "5" * 64,
            "original_allowance_digest": "sha256:" + "6" * 64,
            "approval_reference": "hermetic-human-approval",
            "private_processing_authorized": True, "provider_training_authorized": False,
            "provider_terms_reference": "sha256:" + "7" * 64,
            "provider_retention": "account_does_not_retain",
            "planner": {"provider": "openai", "private_processing_authorized": True,
                        "provider_training_authorized": False, "provider_terms_reference": "sha256:" + "8" * 64,
                        "approval_reference": "hermetic-planner-approval"},
            "maximum_cost_usd": 30, "maximum_requests": 3, "expires_at_epoch": time.time() + 3600,
            "claim_scope": "development_only",
        },
    })


class FakeAtlas:
    """Store paths only; fake keys are inspected in memory, never retained."""

    def __init__(self):
        self.sent = []
        self.evidence = {"operations": [{"name": "accounts/account-approved/projects/project-approved/operations/op"}]}

    def __call__(self, method, url, headers, data):
        path = url.split("/api/v2")[-1]
        self.sent.append((method, path))
        if path == "/operations?origin=DEVELOPER_API&pageSize=1":
            if headers["WLT-Api-Key"] == "rotated-hermetic-key":
                return 200, b'{"operations":[{"name":"accounts/account-approved/projects/other-project/operations/op"}]}'
            if isinstance(self.evidence, Exception):
                raise self.evidence
            return 200, json.dumps(self.evidence).encode()
        return 200, b'{}'


def _client(context, transport, **kwargs):
    return AtlasClient(base_url="https://atlas.example.test/api/v2", api_key="hermetic-key", transport=transport,
                       project_checkpoint=owned_atlas_project_checkpoint(context), **kwargs)


def test_validated_authority_is_detached_and_owner_bound(context, configuration):
    value = validate_owned_atlas_configuration(context, configuration, expected_owner_uid="owner")
    assert value == configuration and value is not configuration
    value["authority"]["provider_identity"]["project_id"] = "different"
    assert configuration["authority"]["provider_identity"]["project_id"] == "project-approved"
    with pytest.raises(ValueError, match="authority_binding_invalid"):
        validate_owned_atlas_configuration(context, configuration, expected_owner_uid="someone-else")


@pytest.mark.parametrize("field,value", [
    ("schema_version", "website_atlas_authorization.v1"), ("request_id", "other-plan"),
    ("capture_id", "other-capture"), ("task_context_digest", "sha256:" + "f" * 64),
    ("source_video_digest", "sha256:" + "f" * 64), ("accepted_by", "different-owner"),
    ("private_processing_authorized", False), ("provider_training_authorized", True),
    ("claim_scope", "production"), ("provider_retention", "unverified"), ("provider_retention", []),
    ("company_authorization_digest", "missing"), ("original_allowance_digest", None),
    ("maximum_cost_usd", 0), ("maximum_cost_usd", 1001), ("maximum_cost_usd", True),
    ("maximum_requests", 1.5), ("maximum_requests", True), ("maximum_requests", 0),
    ("provider_identity", {"account_id": "account-approved", "project_id": "../project"}),
    ("provider_identity", {"account_id": "account-approved"}),
])
def test_rehashed_incompatible_authority_refuses(context, configuration, field, value):
    configuration["authority"][field] = value
    _seal(configuration)
    with pytest.raises(ValueError, match="authority_binding_invalid"):
        validate_owned_atlas_configuration(context, configuration)


@pytest.mark.parametrize("scope", ["source_capture", "planner"])
def test_nested_authority_binding_and_planner_terms_are_checked(context, configuration, scope):
    if scope == "source_capture":
        configuration["authority"][scope]["qa_report_digest"] = "sha256:" + "f" * 64
    else:
        configuration["authority"][scope]["provider_training_authorized"] = True
    _seal(configuration)
    with pytest.raises(ValueError, match="authority_binding_invalid"):
        validate_owned_atlas_configuration(context, configuration)


@pytest.mark.parametrize("field,value", [("dispatch_authorized", True), ("grant_creation_allowed", True),
                                        ("source_video_digest", "sha256:" + "f" * 64)])
def test_configuration_never_substitutes_a_source_or_executable_grant(context, configuration, field, value):
    configuration[field] = value
    with pytest.raises(ValueError, match="configuration_invalid"):
        validate_owned_atlas_configuration(context, configuration)


def test_expiry_digest_and_nonfinite_values_refuse(context, configuration):
    with pytest.raises(ValueError, match="expired"):
        validate_owned_atlas_configuration(context, configuration, now=configuration["authority"]["expires_at_epoch"])
    configuration["authority"]["maximum_requests"] = 4
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_owned_atlas_configuration(context, configuration)
    configuration["authority"]["maximum_cost_usd"] = float("inf")
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_owned_atlas_configuration(context, configuration)


def test_each_upload_and_mutation_checks_actual_key_and_rotations(context, configuration):
    fake = FakeAtlas()
    with bind_owned_atlas_project(task_context=context, configuration=configuration, expected_owner_uid="owner"):
        client = _client(context, fake)
    client.call("POST", "/assets", {})
    client.send_bytes("https://upload.example.test/frame", method="PUT", headers={}, data=b"frame")
    assert [method for method, _ in fake.sent] == ["GET", "POST", "GET", "PUT"]
    client._key = "rotated-hermetic-key"
    with pytest.raises(ValueError, match="identity_mismatch"):
        client.call("POST", "/tasks:atlasGenerate", {})
    assert [method for method, _ in fake.sent] == ["GET", "POST", "GET", "PUT", "GET"]


@pytest.mark.parametrize("evidence,error", [
    ({}, "unavailable"), ({"operations": []}, "unavailable"),
    ({"operations": [{"name": "accounts/other/projects/project-approved/operations/op"}]}, "mismatch"),
    ({"operations": [{"name": "accounts/account-approved/projects/other/operations/op"}]}, "mismatch"),
    ({"operations": [{}]}, "invalid"),
    ({"operations": [{"name": "accounts/account-approved/projects/project-approved/operations/op/extra"}]}, "invalid"),
    (RuntimeError("hermetic-key-must-not-leak"), "unavailable"),
])
def test_unknown_or_wrong_identity_never_reaches_mutation(context, configuration, evidence, error):
    fake = FakeAtlas()
    fake.evidence = evidence
    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        client = _client(context, fake)
    with pytest.raises(ValueError, match="identity_" + error) as caught:
        client.call("POST", "/tasks:atlasGenerate", {})
    assert "must-not-leak" not in str(caught.value)
    assert len(fake.sent) == 1 and fake.sent[0][0] == "GET"


def test_header_override_cannot_use_an_unchecked_key(context, configuration):
    fake = FakeAtlas()
    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        client = _client(context, fake)
    with pytest.raises(ValueError, match="project_key_changed"):
        client.call("POST", "/assets", {}, headers={"WLT-Api-Key": "rotated-hermetic-key"})
    assert len(fake.sent) == 1


def test_private_fence_cannot_be_retargeted_and_survives_sdk_threads(context, configuration):
    fake = FakeAtlas()
    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        client = _client(context, fake)
        configuration["authority"]["provider_identity"]["project_id"] = "other-project"
        context["description"] = context["task_specification"]["description"] = "An altered task"
        context["context_digest"] = cross_runtime_canonical_digest(context, digest_field="context_digest")
        other = _client(context, fake)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(client.call, "POST", "/assets", {}).result()
        with pytest.raises(ValueError, match="project_fence_required"):
            executor.submit(other.call, "POST", "/assets", {}).result()
    assert [method for method, _ in fake.sent] == ["GET", "POST"]


def test_missing_or_expired_authority_allows_reads_cleanup_but_no_mutations(context, configuration, monkeypatch):
    fake = FakeAtlas()
    missing = _client(context, fake)
    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        expired = _client(context, fake)
    monkeypatch.setattr("blueprint_pipeline.owned_capture_atlas_project.time.time",
                        lambda: configuration["authority"]["expires_at_epoch"] + 1)
    for client, refusal in [(missing, "fence_required"), (expired, "authority_expired")]:
        client.call("GET", "/operations/retained")
        client.call("HEAD", "/assets/retained")
        client.call("DELETE", "/assets/retained")
        client.call("POST", "/assets/retained:createReadUrl")
        with pytest.raises(ValueError, match=refusal):
            client.call("POST", "/assets", {})
    assert len(fake.sent) == 8


def test_source_revocation_refuses_mutation_and_website_behavior_is_unchanged(context, configuration):
    fake = FakeAtlas()

    def revoked(_stage):
        raise ValueError("source_revoked")

    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        client = _client(context, fake, source_checkpoint=revoked)
    with pytest.raises(ValueError, match="source_revoked"):
        client.call("POST", "/assets", {})
    assert fake.sent == [("GET", "/operations?origin=DEVELOPER_API&pageSize=1")]
    fake.sent.clear()
    assert owned_atlas_project_checkpoint({"schema_version": "website_task_context.v1"}) is None
    website = _client({"schema_version": "website_task_context.v1"}, fake)
    website.call("POST", "/assets", {})
    assert fake.sent == [("POST", "/assets")]


def test_revocation_during_metadata_is_seen_before_upload(context, configuration):
    fake = FakeAtlas()
    checked = []

    def source_checkpoint(_stage):
        checked.append(len(fake.sent))
        raise ValueError("source_revoked_during_metadata")

    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        client = _client(context, fake, source_checkpoint=source_checkpoint)
    with pytest.raises(ValueError, match="source_revoked_during_metadata"):
        client.send_bytes("https://upload.example.test/frame", method="PUT", headers={}, data=b"frame")
    assert checked == [1]
    assert len(fake.sent) == 1


def test_expiry_during_metadata_refuses_before_mutation(context, configuration, monkeypatch):
    fake = FakeAtlas()
    clock = [configuration["authority"]["expires_at_epoch"] - 1]
    monkeypatch.setattr("blueprint_pipeline.owned_capture_atlas_project.time.time", lambda: clock[0])

    def transport(*args):
        answer = fake(*args)
        clock[0] += 2
        return answer

    with bind_owned_atlas_project(task_context=context, configuration=configuration):
        client = _client(context, transport)
    with pytest.raises(ValueError, match="authority_expired"):
        client.call("POST", "/assets", {})
    assert len(fake.sent) == 1


@pytest.mark.parametrize("factory", ["pose", "task"])
@pytest.mark.parametrize("installed", [False, True])
def test_shared_adapter_factories_install_the_owned_project_fence(context, configuration, tmp_path, monkeypatch,
                                                                 factory, installed):
    """Stop after construction: exercise factory wiring without admission or spend."""
    from blueprint_pipeline import atlas_preparation_context as preparation
    from blueprint_pipeline import website_atlas_pose as pose
    from blueprint_pipeline import website_atlas_tasks as tasks

    class Constructed(Exception):
        pass

    fake = FakeAtlas()

    def construct(**kwargs):
        client = AtlasClient(**kwargs)
        assert callable(kwargs["project_checkpoint"])
        client.call("POST", "/assets", {})
        raise Constructed

    monkeypatch.setattr(pose, "AtlasClient", construct)
    monkeypatch.setattr(pose, "atlas_api_config", lambda: ("https://atlas.example.test/api/v2", "hermetic-key"))
    monkeypatch.setattr(preparation, "preparation_source_checkpoint", lambda _context: lambda _stage: None)
    monkeypatch.setattr(tasks, "validate_task_request", lambda *_: None)
    monkeypatch.setattr(tasks, "_price", lambda *_: .01)
    monkeypatch.setattr(pose, "atlas_provider_retention", lambda: "account_does_not_retain")
    binding = bind_owned_atlas_project(task_context=context, configuration=configuration) if installed else nullcontext()
    with binding, pytest.raises(Constructed if installed else ValueError,
                                match=None if installed else "project_fence_required"):
        if factory == "pose":
            pose.run_admitted_atlas_pose(output_root=tmp_path, capture_id=context["capture_id"],
                                        task_context=context, release_blockers=lambda: ("fixture", []),
                                        transport=fake)
        else:
            tasks._run_admitted_atlas_task_attempt(task="atlasGenerate", request={}, output_root=tmp_path,
                source_video_digest=context["source_capture"]["capture_digest"], task_context=context,
                price_review={}, release_blockers=lambda: ("fixture", []), transport=fake)
    assert [method for method, _ in fake.sent] == (["GET", "POST"] if installed else [])
    assert not list(tmp_path.iterdir())
