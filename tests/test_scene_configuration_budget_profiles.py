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
    maxima = spend_block("astra_cad_blender_v1", authoring_max_cost_usd=25)
    assert maxima["hard_cap_usd"] == 36.76
    assert maxima["external_service_caps"]["openai"]["maximum_cost_usd"] == 30.76
    profile = scene_configuration_budget_profile("astra_cad_blender_v1")
    assert (profile.content_agents_minimum, profile.content_agents_maximum) == (5.0, 25.0)
    assert (profile.external_maximum, profile.attempt_maximum) == (30.76, 36.76)
    assert _required_external_stage_minima(diagnostic_only=False, diagnostic_bootstrap_mode=None,
        carried_stage_count=0, authoring_backend="astra_cad_blender_v1") == quote["external_service_caps"]["openai"]["stage_max_cost_usd"]
    with pytest.raises(ValueError, match="authoring_spend_invalid"):
        spend_block("astra_cad_blender_v1", authoring_max_cost_usd=25.01)


def test_content_only_astra_quote_preserves_the_separate_default() -> None:
    sol = spend_block("astra_cad_blender_v1", authoring_max_cost_usd=7,
                      requires_artifixer=False)
    assert sol["external_service_caps"]["openai"] == {
        "maximum_cost_usd": 7.0, "maximum_requests": 32,
        "stage_max_cost_usd": {"artifixer_semantic_teacher": 0.0,
                               "artifixer_visual_review": 0.0, "content_agents": 7.0}}
    assert sol["hard_cap_usd"] == 13.0
    assert spend_block("astra_cad_blender_v1", authoring_max_cost_usd=15)["hard_cap_usd"] == 26.76
    assert spend_block("astra_cad_blender_v1", authoring_max_cost_usd=25)["hard_cap_usd"] == 36.76
    # A website stage without ArtiFixer: provider compute plus the whole shared authoring pool.
    assert spend_block("astra_cad_blender_v1", authoring_max_cost_usd=25,
                       requires_artifixer=False)["hard_cap_usd"] == 31.0
    assert spend_block("astra_cad_blender_v1", authoring_max_cost_usd=7,
                       requires_artifixer=False, authoring_provider="anthropic")["hard_cap_usd"] == 13.0


@pytest.mark.parametrize("ceiling", [True, 0, -1, float("nan"), float("inf"), "20"])
def test_invalid_total_ceiling_cannot_change_the_quote(ceiling):
    with pytest.raises(ValueError, match="scene_configuration_total_spend_invalid"):
        spend_block("artisan_v1", authoring_max_cost_usd=15,
                    requires_artifixer=False, maximum_total_spend_usd=ceiling)


def test_large_finite_total_ceiling_does_not_expand_any_stage():
    args = {"authoring_max_cost_usd": 15, "requires_artifixer": False}
    assert spend_block("artisan_v1", **args, maximum_total_spend_usd=1e308) == spend_block("artisan_v1", **args)


@pytest.mark.parametrize("author_cap", [5, 10, 15, 25])
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
    # Without an owner grant, the original ceiling bounds a single action.
    with pytest.raises(SceneIntakeError, match="attempt_spend_exceeds_authorized_limit"):
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


def test_artisan_quote_lets_its_dollar_cap_bind_not_a_call_count() -> None:
    """2026-11-01, capture-447ae4f4: ARTISAN was issued 32 OpenAI requests.

    Its agents are built for up to 150 coding and 150 review turns per
    invocation and five repair rounds, each turn a billed call, and the meter
    refuses the next call once the count is spent. 32 would stop it mid-coding
    with most of its $15 unspent. The quote now issues the validator maximum for
    ARTISAN and changes nothing else: no dollar cap, TTL or retry cap moves.
    """
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
        DEFAULT_OPENAI_REQUESTS, MAX_OPENAI_REQUESTS,
    )

    assert (DEFAULT_OPENAI_REQUESTS, MAX_OPENAI_REQUESTS) == (32, 100)
    schema = json.loads((ROOT / "docs/schemas/task_evaluation_launch_preparation_request.v1.schema.json").read_text())

    def openai_request_maxima(node, inside_openai=False):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "maximum_requests" and inside_openai and isinstance(value, dict):
                    yield value["maximum"]
                yield from openai_request_maxima(value, inside_openai or key == "openai")
        elif isinstance(node, list):
            for value in node:
                yield from openai_request_maxima(value, inside_openai)

    # The preparation schema (and the WebApp contract) admit exactly this ceiling.
    assert set(openai_request_maxima(schema)) == {MAX_OPENAI_REQUESTS}
    website = dict(authoring_max_cost_usd=15.0, requires_artifixer=False, maximum_total_spend_usd=20.0)
    artisan = spend_block("artisan_v1", **website)
    assert artisan["external_service_caps"]["openai"]["maximum_requests"] == MAX_OPENAI_REQUESTS
    # Everything else is exactly what a 32-request quote would carry.
    assert artisan["external_service_caps"]["openai"]["maximum_cost_usd"] == 15.0
    assert artisan["external_service_caps"]["openai"]["stage_max_cost_usd"]["content_agents"] == 15.0
    assert artisan["hard_cap_usd"] == 20.0 and artisan["provider_compute_spend_cap_usd"] == 5.0
    assert artisan["retry_cap"] == 0
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import REQUIRED_PARENT_TTL_SECONDS
    assert artisan["hard_ttl_seconds"] == REQUIRED_PARENT_TTL_SECONDS == 27_000
    for backend, cap in (("astra_cad_blender_v1", 7.0), ("content_agents", None)):
        assert spend_block(backend, authoring_max_cost_usd=cap)["external_service_caps"]["openai"][
            "maximum_requests"] == DEFAULT_OPENAI_REQUESTS


@pytest.mark.parametrize("compute_cap", [6, 5])
@pytest.mark.parametrize("backend,author_cap", [("content_agents", 0.24), ("astra_cad_blender_v1", 5),
                                               ("astra_cad_blender_v1", 15), ("astra_cad_blender_v1", 25),
                                               ("artisan_v1", 15), ("artisan_v1", 80)])
def test_paid_authority_materialization_and_reopen_use_the_bound_backend(tmp_path, monkeypatch, backend, author_cap, compute_cap):
    from blueprint_pipeline import task_evaluation_scene_configuration_paid_authority as module
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import REQUIRED_PARENT_TTL_SECONDS

    receipt = {"replacement_authoring_backend": backend, "source_commit": "a" * 40,
        "bundle_sha256": "sha256:" + "b" * 64, "portable_construction_envelope_digest": "sha256:" + "c" * 64,
        "toolchain_digest": "sha256:" + "d" * 64, "run_id": "fixture-astra-budget"}
    if backend == "artisan_v1":
        receipt["authoring_max_cost_usd"] = author_cap
    receipt_path = tmp_path / "bundle.json"
    receipt_path.write_text(json.dumps(receipt))
    project_path = tmp_path / "project.json"
    project_path.write_text('{"fixture":true}')
    zero_path = tmp_path / "zero.json"
    zero_path.write_text('{"fixture":true}')
    monkeypatch.setattr(module, "load_scene_configuration_provider_bundle_receipt", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(module, "bundle_requires_artifixer", lambda _receipt: True)
    monkeypatch.setattr(module, "validate_project_spend_reconciliation", lambda path, **kwargs: ({"total_cost_usd": 100}, module._record(Path(path))))
    monkeypatch.setattr(module, "_provider_zero", lambda path: {"observed_at_utc": "2026-09-10T12:00:00Z", "provider_zero_digest": "sha256:" + "e" * 64})
    quote = spend_block(backend, authoring_max_cost_usd=author_cap)
    quote = spend_block(backend, authoring_max_cost_usd=author_cap,
                        maximum_total_spend_usd=quote["hard_cap_usd"] - (6 - compute_cap))
    assert quote["provider_compute_spend_cap_usd"] == compute_cap
    external = quote["external_service_caps"]["openai"]
    args = dict(bundle_receipt_path=receipt_path, project_spend_reconciliation_path=project_path,
        initial_provider_zero_path=zero_path, authorization_reference="fixture-owner-authorization",
        authorized_by="fixture-owner", authorized_on="2026-09-10T12:05:00Z", source_commit="a" * 40,
        container_image=module.SCENE_CONFIGURATION_PROVIDER_IMAGE, resource_name="adp-fixture-astra-budget-20260910",
        max_hourly_rate_usd=quote["maximum_hourly_rate_usd"], hard_cap_usd=quote["hard_cap_usd"], hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        output_path=tmp_path / "authority.json", provider_compute_spend_cap_usd=compute_cap,
        openai_max_cost_usd=external["maximum_cost_usd"], openai_max_requests=external["maximum_requests"],
        openai_artifixer_semantic_teacher_max_cost_usd=4.8, openai_artifixer_visual_review_max_cost_usd=0.96,
        openai_content_agents_max_cost_usd=author_cap)
    result = module.materialize_scene_configuration_paid_authority(**args)
    assert module.validate_scene_configuration_paid_authority(result, bundle_receipt=receipt) == result
    assert result["hard_attempt_spend_cap_usd"] == quote["hard_cap_usd"]
    assert result["maximum_paid_attempts"] == 1 and result["retry_cap"] == 0
    assert result["external_service_spend_caps"]["openai"]["maximum_requests"] == (
        100 if backend == "artisan_v1" else 32)
    # A tighter ceiling never admits underfunded lifetime, extra compute or a
    # component sum above the total. Reopening independently checks each bound.
    for overrides in ({"provider_compute_spend_cap_usd": 0},
                      {"provider_compute_spend_cap_usd": 6.01},
                      {"hard_cap_usd": quote["hard_cap_usd"] - 0.01},
                      {"max_hourly_rate_usd": compute_cap * 3600 / REQUIRED_PARENT_TTL_SECONDS + 0.000001}):
        with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="configuration_invalid"):
            module.materialize_scene_configuration_paid_authority(**{**args, **overrides,
                "output_path": tmp_path / "refused.json"})
    changed = deepcopy(result)
    changed["maximum_hourly_rate_usd"] = compute_cap * 3600 / REQUIRED_PARENT_TTL_SECONDS + 0.000001
    changed["authority_digest"] = canonical_digest(changed, digest_field="authority_digest")
    with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="authority_contract_invalid"):
        module.validate_scene_configuration_paid_authority(changed, bundle_receipt=receipt)
    if backend == "astra_cad_blender_v1":
        for overrides in ({"hard_cap_usd": 12}, {"openai_max_cost_usd": 6},
                {"openai_content_agents_max_cost_usd": 4.99}, {"openai_content_agents_max_cost_usd": 25.01}):
            with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="configuration_invalid"):
                module.materialize_scene_configuration_paid_authority(**{**args, **overrides, "output_path": tmp_path / "refused.json"})
        legacy_receipt = {**receipt, "replacement_authoring_backend": "content_agents"}
        with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="authority_contract_invalid"):
            module.validate_scene_configuration_paid_authority(result, bundle_receipt=legacy_receipt)


def test_future_anthropic_authority_is_separate_from_openai_cap(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_configuration_paid_authority as module
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import REQUIRED_PARENT_TTL_SECONDS

    receipt = {"replacement_authoring_backend": "astra_cad_blender_v1",
        "replacement_authoring_model_provider": "anthropic", "source_commit": "a" * 40,
        "bundle_sha256": "sha256:" + "b" * 64,
        "portable_construction_envelope_digest": "sha256:" + "c" * 64,
        "toolchain_digest": "sha256:" + "d" * 64, "run_id": "future-anthropic"}
    receipt_path = tmp_path / "bundle.json"
    receipt_path.write_text(json.dumps(receipt))
    project_path = tmp_path / "project.json"
    project_path.write_text('{"fixture":true}')
    zero_path = tmp_path / "zero.json"
    zero_path.write_text('{"fixture":true}')
    monkeypatch.setattr(module, "load_scene_configuration_provider_bundle_receipt", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(module, "bundle_requires_artifixer", lambda _receipt: False)
    monkeypatch.setattr(module, "validate_project_spend_reconciliation",
        lambda path, **kwargs: ({"total_cost_usd": 100}, module._record(Path(path))))
    monkeypatch.setattr(module, "_provider_zero",
        lambda path: {"observed_at_utc": "2026-09-10T12:00:00Z",
                      "provider_zero_digest": "sha256:" + "e" * 64})
    args = dict(bundle_receipt_path=receipt_path,
        project_spend_reconciliation_path=project_path,
        initial_provider_zero_path=zero_path,
        authorization_reference="fixture-owner-authorization", authorized_by="fixture-owner",
        authorized_on="2026-09-10T12:05:00Z", source_commit="a" * 40,
        container_image=module.SCENE_CONFIGURATION_PROVIDER_IMAGE,
        resource_name="adp-future-anthropic-20260910", max_hourly_rate_usd=0.5,
        hard_cap_usd=13, hard_ttl_seconds=REQUIRED_PARENT_TTL_SECONDS,
        output_path=tmp_path / "authority.json", provider_compute_spend_cap_usd=6,
        anthropic_max_cost_usd=7, anthropic_max_requests=32)
    result = module.materialize_scene_configuration_paid_authority(**args)
    assert module.validate_scene_configuration_paid_authority(result, bundle_receipt=receipt) == result
    assert result["external_service_spend_caps"]["openai"]["maximum_cost_usd"] == 0
    assert result["external_service_spend_caps"]["anthropic"]["maximum_cost_usd"] == 7
    with pytest.raises(module.TaskEvaluationSceneConfigurationAuthorityError, match="configuration_invalid"):
        module.materialize_scene_configuration_paid_authority(**{**args, "anthropic_max_cost_usd": 7.01,
            "output_path": tmp_path / "refused.json"})


def test_scene_configuration_submission_requests_the_astra_authoring_ceiling():
    """The astra stage must not be quoted its floor; its reservations are worst case.

    Scene 840938 object 219, 2026-09-15: CAD produced a valid STEP/STL passing
    dimensional readback, then the stage died on
    `agents_sdk_inference_budget_ceiling_exceeded` inside independent_visual_review,
    projecting $11.47 against the quoted $5.00 while its observed cost was $1.744056.
    """

    from blueprint_pipeline.task_evaluation_scene_configuration_submission import (
        _authoring_spend_request,
    )

    profile = scene_configuration_budget_profile("astra_cad_blender_v1")
    requested = _authoring_spend_request("astra_cad_blender_v1")

    assert requested == profile.content_agents_maximum == 25.0
    # Comfortably above the projection that actually blocked the run.
    assert requested > 11.475
    quote = spend_block("astra_cad_blender_v1", authoring_max_cost_usd=requested)
    assert quote["external_service_caps"]["openai"]["stage_max_cost_usd"]["content_agents"] == 25.0
    assert quote["hard_cap_usd"] == 36.76

    # Other backends keep their own quote; only astra asks for more.
    assert _authoring_spend_request("content_agents") is None
    assert spend_block("content_agents")["hard_cap_usd"] == 12

    # The shared default quote is deliberately unchanged, so the pinned schema holds.
    assert spend_block("astra_cad_blender_v1")["hard_cap_usd"] == 16.76
