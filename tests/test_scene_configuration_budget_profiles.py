"""Backend quote parity and unchanged owner authorization for fresh scene construction."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_contract import (
    TaskEvaluationLaunchPreparationContractError, validate_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    scene_configuration_budget_profile, scene_configuration_budget_profile_contract,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import spend_block
from blueprint_pipeline.task_evaluation_scene_configuration_paid_authority import _required_external_stage_minima
from blueprint_pipeline.task_evaluation_scene_intake import SceneIntakeError
from tests.test_task_evaluation_scene_intake import request as owner_request, stage, attempt

ROOT = Path(__file__).resolve().parents[1]


def request():
    return json.loads((ROOT / "tests/fixtures/scene_configuration/astra_preparation_request.v1.json").read_text())


def test_shared_profile_quote_and_schema_contract_match():
    contract = json.loads((ROOT / "docs/schemas/task_evaluation_scene_configuration_budget_profiles.v1.json").read_text())
    assert contract == scene_configuration_budget_profile_contract()
    assert contract["creates_spend_authority"] is False
    legacy = spend_block()
    assert legacy["hard_cap_usd"] == 12 and legacy["external_service_caps"]["openai"]["maximum_cost_usd"] == 6
    quote = spend_block("astra_cad_blender_v1")
    assert quote["hard_cap_usd"] == 16.76
    assert quote["external_service_caps"]["openai"]["maximum_cost_usd"] == 10.76
    maxima = spend_block("astra_cad_blender_v1", authoring_max_cost_usd=15)
    assert maxima["hard_cap_usd"] == 26.76
    assert maxima["external_service_caps"]["openai"]["maximum_cost_usd"] == 20.76
    assert _required_external_stage_minima(diagnostic_only=False, diagnostic_bootstrap_mode=None,
        carried_stage_count=0, authoring_backend="astra_cad_blender_v1") == quote["external_service_caps"]["openai"]["stage_max_cost_usd"]
    with pytest.raises(ValueError, match="authoring_spend_invalid"):
        spend_block("astra_cad_blender_v1", authoring_max_cost_usd=15.01)


@pytest.mark.parametrize("author_cap", [5, 10, 15])
def test_explicit_astra_requests_are_admitted_without_rewriting_proposed_caps(author_cap):
    value = request()
    value["spend"] = spend_block("astra_cad_blender_v1", authoring_max_cost_usd=author_cap)
    before = deepcopy(value)
    assert validate_launch_preparation_request(value) == before
    assert value == before


def test_legacy_and_underfunded_astra_cannot_take_the_higher_profile():
    value = request()
    value.pop("replacement_authoring_backend")
    with pytest.raises(TaskEvaluationLaunchPreparationContractError):
        validate_launch_preparation_request(value)
    value["spend"] = spend_block()
    assert validate_launch_preparation_request(value) == value
    value["replacement_authoring_backend"] = "astra_cad_blender_v1"
    with pytest.raises(TaskEvaluationLaunchPreparationContractError, match="external_spend_invalid"):
        validate_launch_preparation_request(value)
    with pytest.raises(ValueError, match="authoring_backend_invalid"):
        scene_configuration_budget_profile("unrecognized")


def test_existing_owner_cap_cannot_be_raised_by_selecting_the_astra_profile(tmp_path):
    value = owner_request()
    value["execution"]["max_total_spend_usd"] = 12
    intent = stage(tmp_path, value)
    proposed = spend_block("astra_cad_blender_v1")["hard_cap_usd"]
    # Cumulative grants cannot increase the original per-action ceiling.
    with pytest.raises(SceneIntakeError, match="attempt_spend_exceeds_original_limit"):
        attempt(tmp_path, intent, cost=proposed)
    stored = json.loads((tmp_path / intent["intent_id"] / "intent.json").read_text())
    assert stored["request"]["execution"]["max_total_spend_usd"] == 12
    assert not list((tmp_path / intent["intent_id"] / "attempts").glob("*.json"))


def test_quote_can_reserve_only_when_the_owner_cap_covers_it(tmp_path):
    value = owner_request()
    value["execution"]["max_total_spend_usd"] = 20
    intent = stage(tmp_path, value)
    result = attempt(tmp_path, intent, cost=spend_block("astra_cad_blender_v1")["hard_cap_usd"])
    assert result["maximum_spend_usd"] == 16.76
    assert result["attempt_digest"] == canonical_digest(result, digest_field="attempt_digest")


@pytest.mark.parametrize("backend,author_cap", [("content_agents", 0.24), ("astra_cad_blender_v1", 5), ("astra_cad_blender_v1", 15)])
def test_paid_authority_materialization_and_reopen_use_the_bound_backend(tmp_path, monkeypatch, backend, author_cap):
    from blueprint_pipeline import task_evaluation_scene_configuration_paid_authority as module
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import REQUIRED_PARENT_TTL_SECONDS

    receipt = {"replacement_authoring_backend": backend, "source_commit": "a" * 40,
        "bundle_sha256": "sha256:" + "b" * 64, "portable_construction_envelope_digest": "sha256:" + "c" * 64,
        "toolchain_digest": "sha256:" + "d" * 64, "run_id": "fixture-astra-budget"}
    receipt_path = tmp_path / "bundle.json"
    receipt_path.write_text(json.dumps(receipt))
    project_path = tmp_path / "project.json"
    project_path.write_text('{"fixture":true}')
    zero_path = tmp_path / "zero.json"
    zero_path.write_text('{"fixture":true}')
    monkeypatch.setattr(module, "load_scene_configuration_provider_bundle_receipt", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(module, "validate_project_spend_reconciliation", lambda path, **kwargs: ({"total_cost_usd": 100}, module._record(Path(path))))
    monkeypatch.setattr(module, "_provider_zero", lambda path: {"observed_at_utc": "2026-09-10T12:00:00Z", "provider_zero_digest": "sha256:" + "e" * 64})
    quote = spend_block(backend, authoring_max_cost_usd=author_cap)
    external = quote["external_service_caps"]["openai"]
    args = dict(bundle_receipt_path=receipt_path, project_spend_reconciliation_path=project_path,
        initial_provider_zero_path=zero_path, authorization_reference="fixture-owner-authorization",
        authorized_by="fixture-owner", authorized_on="2026-09-10T12:05:00Z", source_commit="a" * 40,
        container_image=module.SCENE_CONFIGURATION_PROVIDER_IMAGE, resource_name="adp-fixture-astra-budget-20260910",
        max_hourly_rate_usd=0.5, hard_cap_usd=quote["hard_cap_usd"], hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        output_path=tmp_path / "authority.json", provider_compute_spend_cap_usd=6,
        openai_max_cost_usd=external["maximum_cost_usd"], openai_max_requests=32,
        openai_artifixer_semantic_teacher_max_cost_usd=4.8, openai_artifixer_visual_review_max_cost_usd=0.96,
        openai_content_agents_max_cost_usd=author_cap)
    result = module.materialize_scene_configuration_paid_authority(**args)
    assert module.validate_scene_configuration_paid_authority(result, bundle_receipt=receipt) == result
    assert result["hard_attempt_spend_cap_usd"] == quote["hard_cap_usd"]
    assert result["maximum_paid_attempts"] == 1 and result["retry_cap"] == 0
    if backend == "astra_cad_blender_v1":
        for overrides in ({"hard_cap_usd": 12}, {"openai_max_cost_usd": 6},
                {"openai_content_agents_max_cost_usd": 4.99}, {"openai_content_agents_max_cost_usd": 15.01}):
            with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="configuration_invalid"):
                module.materialize_scene_configuration_paid_authority(**{**args, **overrides, "output_path": tmp_path / "refused.json"})
        legacy_receipt = {**receipt, "replacement_authoring_backend": "content_agents"}
        with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="authority_contract_invalid"):
            module.validate_scene_configuration_paid_authority(result, bundle_receipt=legacy_receipt)
