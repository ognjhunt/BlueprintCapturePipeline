"""Dream traverses real staging and sealed admission; no provider is allocated."""
from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from blueprint_pipeline import agent_run_executor as executor
from blueprint_pipeline import team_candidate_quick10_launch as launch
from blueprint_pipeline.agent_execution_offer import _evaluation_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dreamlabs_policy_client import (
    ENDPOINT_URL, GATEWAY_ORIGIN, INTERFACE_ID, METADATA_URL, DreamLabsPolicySpec, validate_policy_config,
)
from blueprint_pipeline.team_candidate_quick10_admission import (
    TeamCandidateAdmissionError, _candidate_blockers, validate_admission,
)
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_dreamlabs_policy_client import MODEL, Service
from tests.test_site_task_simulation_qualification import NOW, fixture
from tests.test_team_arm_kinematics import named_inputs
from tests.test_team_candidate_quick10_launch import _Broker, _world


def dream_world(tmp_path, monkeypatch):
    model, manifest = named_inputs(tmp_path)
    manifest["action_chunk_rows"] = 48
    (tmp_path / "world").mkdir()
    world = _world(tmp_path / "world", monkeypatch, robot={"manifest": manifest, "model": model})
    # The real60Hz TTL is4x the15Hz budget. A deliberately cheap approved
    # fixture profile fits the unchanged $20 gate; no production cap is raised.
    registry_path = world["operator"] / "registry.json"
    registry = json.loads(registry_path.read_text())
    registry["profiles"][0]["max_hourly_rate_usd"] = 0.2
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    registry_path.write_text(json.dumps(registry))
    row = world["row"]
    request = row["execution_admission"]["canonical_execution_request"]
    payload = request["policy_package"]["policy_api_endpoint"]
    payload.update(policy_interface_id=INTERFACE_ID, endpoint_url=ENDPOINT_URL,
                   policy_config={"model_id": MODEL["id"], "camera_roles": {
                       "overhead": "scene_rgb", "wrist_l": "wrist_rgb", "wrist_r": "right_wrist_rgb"}})
    # Synthetic, explicit authority for this hermetic fixture only. Production
    # approvals must come from the operator's actual team_rights record.
    rights_path = world["operator"] / "rights.json"
    rights = json.loads(rights_path.read_text())
    disclosure = {"schema_version": "blueprint.dreamlabs_provider_disclosure.v1",
                  **{key: rights[key] for key in ("team_id", "checkpoint_id", "candidate_id")},
                  "model_id": MODEL["id"], "scene_revision_digest": world["probe_receipt"]["configured_scene_revision_digest"],
                  "inference_origin": GATEWAY_ORIGIN, "metadata_origin": "https://platform.dreamlabs.ai",
                  "approved_by": "synthetic-test-operator", "approval_reference": "hermetic-test-only",
                  "accepted_at_iso": "2026-10-01T00:00:00Z",
                  "disclosure": {"rgb": True, "robot_state": True, "task_instruction": True},
                  "provider_terms": {"retention_terms_reference": "fixture-provider-retention-review",
                                     "training_terms_reference": "fixture-provider-training-review", "accepted": True},
                  "hosted_billing": {"payer": "team_account", "authorized_by": "synthetic-team-account-owner",
                                     "approval_reference": "hermetic-hosted-usage-only",
                                     "policy_inference_usage_authorized": True, "blueprint_billing_permitted": False},
                  "sdk_extra_recording": False, "disclosure_digest": ""}
    disclosure["disclosure_digest"] = canonical_digest(disclosure, digest_field="disclosure_digest")
    rights["provider_disclosure"] = disclosure
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    rights_path.write_text(json.dumps(rights))

    class Broker(_Broker):
        def release_bearer_lease(self, *, lease_id):
            result = super().release_bearer_lease(lease_id=lease_id)
            result["credential"]["endpoint_url"] = ENDPOINT_URL
            return result

    service = Service()
    world["hooks"].update(credential_client=Broker, metadata_transport=service)
    world.update(request=request, dream_service=service)
    seal_request(world)
    return world


def seal_request(world):
    frozen = json.dumps(world["row"]["execution_admission"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    world["row"].update(execution_admission_canonical_json=frozen,
                        execution_admission_digest="sha256:" + hashlib.sha256(frozen.encode()).hexdigest())


def test_dream_account_discovery_seals_requested_interface_and_exact_config_without_inference(tmp_path, monkeypatch):
    world = dream_world(tmp_path, monkeypatch)
    result = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert result["staged"] == 1, (result, world["client"].blocked)
    assert len(_Broker.leases) == 1 and not world["allocator_calls"] and not world["provider"].calls, world["client"].blocked
    assert [(call["url"], call["method"]) for call in world["dream_service"].calls] == [(METADATA_URL, "GET")]
    job = world["job_dir"]
    admission = validate_admission(json.loads((job / "team_candidate_admission.json").read_text()))
    assert admission["policy_interface_id"] == INTERFACE_ID and admission["embodiment_preset_id"] == "team_embodiment"
    assert admission["delivery"]["endpoint_url"] == ENDPOINT_URL
    assert admission["delivery"]["metadata_url"] == METADATA_URL
    assert admission["delivery"]["provider_model_metadata"] == MODEL
    assert admission["delivery"]["policy_config"]["model_id"] == MODEL["id"]
    assert admission["delivery"]["policy_config"]["record"] is False
    assert admission["provider_disclosure"]["sdk_extra_recording"] is False
    assert admission["credential"]["origin"] == GATEWAY_ORIGIN  # Lease is not widened to arbitrary metadata origins.
    assert admission["delivery"]["open_loop_horizon"] == 30
    spec = json.loads(next(job.glob("shards/00/stage-00/runtime_inputs/policy_execution_spec.*.json")).read_text())
    assert spec["policy_interface_id"] == INTERFACE_ID
    assert spec["checkpoint_identity_basis"] == "provider_metadata" and spec["weights_identity_verified"] is False
    assert spec["provider_disclosure"] == admission["provider_disclosure"]
    DreamLabsPolicySpec(**spec["policy_spec"]).validate()
    assert spec["policy_spec"]["action_hz"] == 20 and spec["policy_spec"]["action_space"] == "named_arm_pose"
    assert spec["max_policy_queries"] == 11  # 240 canonical15Hz steps →320 Dream20Hz rows /30, rounded up.
    with zipfile.ZipFile(next(job.glob("shards/00/stage-00/bundle/*.zip"))) as bundle:
        assert "provider_runtime/blueprint_pipeline/dreamlabs_policy_client.py" in bundle.namelist()
        assert all(b"team-bearer-" not in bundle.read(member) for member in bundle.namelist())


@pytest.mark.parametrize("fault", ["extra_config", "record", "wrong_role", "wrong_endpoint", "wrong_metadata", "hz", "no_arms", "rows"])
def test_bad_dream_contract_fails_before_credential_release_or_provider_allocation(tmp_path, monkeypatch, fault):
    world = dream_world(tmp_path, monkeypatch)
    payload = world["request"]["policy_package"]["policy_api_endpoint"]
    if fault == "extra_config":
        payload["policy_config"]["api_key"] = "must-not-be-accepted"
    elif fault == "record":
        payload["policy_config"]["record"] = True
    elif fault == "wrong_role":
        payload["policy_config"]["camera_roles"]["wrist_r"] = "wrist_rgb"
    elif fault == "wrong_endpoint":
        payload["endpoint_url"] = ENDPOINT_URL + "?access=secret"
    elif fault == "wrong_metadata":
        payload["metadata_url"] = "https://attacker.example/v1/models"
    else:
        manifest = payload["robot_embodiment"]
        manifest.pop("embodiment_digest", None)
        if fault == "hz":
            manifest["control_hz"] = 30
        elif fault == "rows":
            manifest["action_chunk_rows"] = 49
        else:
            manifest.pop("arms")
        payload["robot_embodiment"] = validate_embodiment_manifest(manifest)
    seal_request(world)
    blockers = launch.team_request_blockers(world["request"], world["row"])
    assert blockers
    result = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert result["staged"] == 0 and not _Broker.leases
    assert not world["dream_service"].calls and not world["allocator_calls"] and not world["provider"].calls


def test_discovery_refuses_an_unlisted_model_before_bundle_and_does_not_alias(tmp_path, monkeypatch):
    world = dream_world(tmp_path, monkeypatch)
    world["request"]["policy_package"]["policy_api_endpoint"]["policy_config"]["model_id"] = "dreamlabs/dream-1"
    update_rights(world, lambda rights: rights["provider_disclosure"].update(model_id="dreamlabs/dream-1"))
    seal_request(world)
    result = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert len(_Broker.leases) == 1, world["client"].blocked
    assert result["pending"] == 1
    for _ in range(launch.MAX_PRELAUNCH_ATTEMPTS - 1):
        executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert world["client"].blocked == ["team_candidate_dream_model_discovery_failed"]
    assert not list(world["job_dir"].glob("shards/**/*.zip")) and not world["allocator_calls"]
    assert all(call["method"] == "GET" for call in world["dream_service"].calls)


def update_rights(world, update):
    path = world["operator"] / "rights.json"
    rights = json.loads(path.read_text())
    update(rights)
    if "provider_disclosure" in rights:
        receipt = rights["provider_disclosure"]
        receipt["disclosure_digest"] = canonical_digest(receipt, digest_field="disclosure_digest")
    rights["rights_digest"] = canonical_digest(rights, digest_field="rights_digest")
    path.write_text(json.dumps(rights))


@pytest.mark.parametrize("fault", ["license_only", "terms", "scene", "model", "origin", "sdk_recording", "future", "rgb"])
def test_dream_disclosure_requires_exact_explicit_operator_terms_before_credential_or_data_transfer(tmp_path, monkeypatch, fault):
    world = dream_world(tmp_path, monkeypatch)

    def change(rights):
        receipt = rights["provider_disclosure"]
        if fault == "license_only":
            rights.pop("provider_disclosure")
        elif fault == "terms":
            receipt["provider_terms"]["accepted"] = False
        elif fault == "scene":
            receipt["scene_revision_digest"] = "sha256:" + "1" * 64
        elif fault == "model":
            receipt["model_id"] = "dreamlabs/different-model"
        elif fault == "origin":
            receipt["metadata_origin"] = "https://other.example"
        elif fault == "sdk_recording":
            receipt["sdk_extra_recording"] = True
        elif fault == "future":
            receipt["accepted_at_iso"] = "2099-01-01T00:00:00Z"
        else:
            receipt["disclosure"]["rgb"] = 1

    update_rights(world, change)
    executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert world["client"].blocked == ["team_candidate_provider_disclosure_required" if fault == "license_only"
                                         else "team_candidate_provider_disclosure_terms_invalid"]
    assert not _Broker.leases and not world["dream_service"].calls and not world["allocator_calls"]
    assert not (world["job_dir"] / ".team-credential").exists()


@pytest.mark.parametrize("fault", ["missing", "blueprint_payer", "blueprint_allowed", "no_usage_authority", "no_owner"])
def test_sponsored_gpu_funding_does_not_authorize_hosted_api_account_charges(tmp_path, monkeypatch, fault):
    world = dream_world(tmp_path, monkeypatch)

    def change(rights):
        receipt = rights["provider_disclosure"]
        billing = receipt["hosted_billing"]
        if fault == "missing":
            receipt.pop("hosted_billing")
        elif fault == "blueprint_payer":
            billing["payer"] = "blueprint"
        elif fault == "blueprint_allowed":
            billing["blueprint_billing_permitted"] = True
        elif fault == "no_usage_authority":
            billing["policy_inference_usage_authorized"] = False
        else:
            billing["authorized_by"] = ""

    update_rights(world, change)
    executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert world["client"].blocked == ["team_candidate_hosted_policy_billing_authority_required"]
    assert not _Broker.leases and not world["dream_service"].calls and not world["allocator_calls"]


def test_dream_config_unknowns_and_protocol_changes_cannot_be_resealed():
    for config in ({"model_id": MODEL["id"], "camera_roles": {}, "timeout": 50},
                   {"model_id": MODEL["id"], "camera_roles": {}, "record": True}):
        with pytest.raises(ValueError):
            validate_policy_config(config, camera_names=["one", "two", "three"])


def test_sealed_dream_admission_refuses_provider_record_drift_even_with_new_self_digest(tmp_path, monkeypatch):
    world = dream_world(tmp_path, monkeypatch)
    result = executor.poll_once(client=world["client"], capture_root=world["capture"], team_launch_hooks=world["hooks"])
    assert result["staged"] == 1
    admission = json.loads((world["job_dir"] / "team_candidate_admission.json").read_text())
    admission["delivery"]["provider_model_metadata"]["base_model"] = "silently-different"
    admission["admission_digest"] = canonical_digest(admission, digest_field="admission_digest")
    with pytest.raises(TeamCandidateAdmissionError, match="dream_contract_invalid"):
        validate_admission(admission)
    candidate = {**admission, "schema_version": "blueprint.team_policy_candidate.v1"}
    assert "team_candidate_dream_contract_invalid" in _candidate_blockers(candidate)


def test_dream_qualification_retains_existing_simulation_only_evidence_gates(tmp_path):
    from blueprint_pipeline.site_task_simulation_qualification import qualify_from_profile
    profile = fixture(tmp_path, binding={"policy_interface_id": INTERFACE_ID})
    receipt = qualify_from_profile(profile, now=NOW)
    assert receipt["binding"]["policy_interface_id"] == INTERFACE_ID
    assert receipt["evidence_scope"] == "simulation" and receipt["physical_performance_claimed"] is False


def test_operator_contract_documents_provider_protocol_and_unverified_weights(tmp_path, monkeypatch):
    from blueprint_pipeline.team_candidate_quick10_registry import offer_eligibility, profile_for_request
    world = dream_world(tmp_path, monkeypatch)
    profile = profile_for_request(world["request"])
    facts, blockers = offer_eligibility(profile, "scored")
    assert not blockers
    contract = _evaluation_contract(facts, profile)
    descriptor = next(row for row in contract["policy_interfaces"] if row["policy_interface_id"] == INTERFACE_ID)
    assert descriptor["metadata_url"] == METADATA_URL
    assert descriptor["weights_identity_verified"] is False and descriptor["sdk_recording_enabled"] is False
    assert descriptor["action"]["action_hz"] == 20 and descriptor["action"]["control_hz"] == 60
    assert descriptor["observation"]["history_frames_max"] == 9
