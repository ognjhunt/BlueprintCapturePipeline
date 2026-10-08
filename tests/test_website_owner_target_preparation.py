"""ADP-030/day 28: owner targets survive scene-only preparation, never become scored truth.

Owned synthetic geometry, hermetic native submission and real authority contracts;
no model/provider calls or physical-outcome claim.
"""
import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
from tests.test_website_task_preparation import _arguments
from tests.test_website_native_submission import setup

UNKNOWN = {"successDefinition": None, "successRate": None,
           "cycleTimeSeconds": None, "unknown": True}
EXPLICIT = {"successDefinition": "Arrives intact", "successRate": 95,
            "cycleTimeSeconds": 30, "unknown": False}


@pytest.mark.parametrize("criteria", [UNKNOWN, EXPLICIT], ids=["unknown-null", "explicit-95pct-30sec"])
def test_scene_only_preparation_retains_owner_targets_without_evaluating_them(tmp_path, criteria):
    args = _arguments(tmp_path)
    context = args["task_context"]
    context["success_criteria"] = copy.deepcopy(criteria)
    context["context_digest"] = canonical_digest(context, digest_field="context_digest")
    original = copy.deepcopy(context)
    value = compile_website_scene_preparation(**args)
    assert value["status"] == "intake_ready", value["blockers"]
    assert context == original
    assert value["owner_success_criteria"]["targets"] == criteria
    assert value["owner_success_criteria"]["scorer_translation_verified"] is False
    assert value["owner_success_criteria"]["basis"] == "owner_stated_target"
    assert value["intake_request"]["execution"]["purpose"] == "scene_preparation"
    assert value["intake_request"]["execution"]["policy_candidates"] == []
    assert value["claim_ceiling"] == "development_only"
    assert value["provider_mutation_performed"] is False
    assert value["authoring_inputs"]["configuration"]["construction_constraints"]["owner_success_criteria"] == criteria
    persisted = json.loads((tmp_path / "out/preparation.json").read_text())
    assert persisted["owner_success_criteria"] == value["owner_success_criteria"]


@pytest.mark.parametrize("criteria", [UNKNOWN, EXPLICIT], ids=["unknown-null", "explicit-95pct-30sec"])
def test_staged_native_owner_authority_cannot_confirm_fixed_controls_as_the_owner_bar(tmp_path, monkeypatch, criteria):
    from blueprint_pipeline.website_native_submission import materialize_website_submission
    from blueprint_pipeline.task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract
    from blueprint_pipeline.adp_task_scoring import TaskNeutralScoringError

    kwargs, _ = setup(tmp_path, monkeypatch, success_criteria=criteria)
    original_context_bytes = Path(kwargs["task"]["task_context"]["path"]).read_bytes()
    materialize_website_submission(**kwargs)
    root = kwargs["staging_root"]
    assert (root / "website/task_context.json").read_bytes() == original_context_bytes
    template = json.loads((root / "configuration/task.json").read_text())
    targets = template["owner_success_criteria"]
    assert targets["targets"] == criteria
    assert targets["scorer_translation_verified"] is False
    authority = template["owner_success_contract_authority"]
    assert authority["confirmation_status"] == "proposal_only"
    assert authority["basis"] == "fixed_development_control_not_translated_owner_target"
    assert authority["scorer_translation_verified"] is False
    assert authority["owner_success_criteria"] == targets
    # This is the actual performance-owner sealing gate, before any scorer.
    with pytest.raises(TaskNeutralScoringError, match="configured_owner_success_contract_authority_missing"):
        _derive_configured_owner_success_contract({
            "configured_success_criteria": {"owner_success_contract_required": True},
            "configured_owner_authority": authority,
        }, site_id="synthetic-site", task_id="synthetic-task")
    manifest = json.loads((root / "bundle_manifest.v1.json").read_text())
    assert manifest["native_qualification_claimed"] is False
    assert manifest["provider_allocated"] is False

    # Reuse the retained native-adapter fixture only to check lossless authority
    # transport; its synthetic qualification records are not fresh native proof.
    from tests.test_task_evaluation_rigid_relocation_native_adapter import _case, _rewrite, DEFINITION
    from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import adapt_rigid_relocation_task_template
    adapter_root = tmp_path / "authority-transport"
    adapter_root.mkdir()
    launch, configured, references, docs = _case(adapter_root)
    definition = {**docs[DEFINITION], "owner_success_contract_authority": authority}
    _rewrite(tmp_path=adapter_root, configured=configured, references=references,
             contract_path=DEFINITION, document=definition)
    launch["task"]["configured_scene_revision_digest"] = configured["revision_digest"]
    native = adapt_rigid_relocation_task_template(request=launch, configured_revision=configured,
                                                materialized_references=references)
    assert native["native_task_definition"]["task_spec"]["configured_owner_authority"] == authority


@pytest.mark.parametrize("criteria", [UNKNOWN, EXPLICIT], ids=["unknown-null", "explicit-95pct-30sec"])
def test_performance_request_keeps_owner_translation_refusal(criteria):
    from blueprint_pipeline.website_task_preparation import _owner_targets_require_translation
    from blueprint_pipeline.website_task_evidence import owner_success_criteria
    from tests.test_website_scene_preparation_scope import preparation_request
    request = preparation_request()
    targets = owner_success_criteria({"success_criteria": criteria})
    assert not _owner_targets_require_translation(targets, request)
    del request["execution"]["purpose"]  # The existing evaluation discriminator.
    assert _owner_targets_require_translation(targets, request)
    assert not _owner_targets_require_translation(owner_success_criteria({}), request)
