"""Protect private caller compatibility and stage-chain claim boundaries after extraction."""
from copy import deepcopy

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_artifacts as artifacts
from blueprint_pipeline import task_evaluation_scene_configuration_artifixer_driver as driver
from blueprint_pipeline import task_evaluation_scene_configuration_provider_stage_chain as chains
from blueprint_pipeline import task_evaluation_scene_configuration_vast as vast
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def test_existing_artifixer_private_imports_still_resolve_to_same_implementations():
    for name in ("TaskEvaluationSceneConfigurationArtifixerError", "_sha256", "_read", "_record",
                 "_component_record", "_diagnostic_rejection_permitted", "_materialize_selected_task_thumbnail",
                 "_materialize_ungraded_task_thumbnail_and_receipt", "_materialize_diagnostic_rejected_artifixer_artifacts"):
        assert getattr(driver, name) is getattr(artifacts, name)
    assert vast._completed_stage_chain_valid is chains._completed_stage_chain_valid


def _chain(diagnostic):
    rows = []
    for index in range(6):
        row = {"schema_version": "task_evaluation_scene_configuration_stage_result.v1",
               "stage_id": f"stage-{index}", "status": "completed", "canonical_allocator": None,
               "provider_mutations_performed": 0, "paid_execution_requested": False,
               "executed_inside_parent_configuration_run": True, "raw_secret_values_recorded": False,
               "output_artifacts": [], "diagnostic_only": diagnostic, "qualification_eligible": not diagnostic,
               "executed_inside_one_parent_provider_run": not diagnostic,
               "configured_revision_publication_permitted": not diagnostic,
               "offering_publication_permitted": not diagnostic,
               "terminal_e2e_completion_permitted": not diagnostic}
        row["stage_result_digest"] = canonical_digest(row, digest_field="stage_result_digest")
        rows.append(row)
    result = {"schema_version": "task_evaluation_scene_configuration_" +
              ("diagnostic" if diagnostic else "provider") + "_stage_chain.v1",
              "status": "completed_diagnostic_only_not_qualification_eligible" if diagnostic else "completed",
              "run_id": "run", "stage_count": 6, "stage_results": rows,
              "stage_result_digests": [row["stage_result_digest"] for row in rows],
              "executed_inside_one_parent_provider_run": not diagnostic,
              "nested_provider_mutations_performed": 0, "nested_paid_execution_requested": False,
              "evaluation_episode_executed": False, "retry_cap": 0}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return result


@pytest.mark.parametrize("diagnostic", [False, True])
def test_stage_chain_preserves_parent_execution_and_rejects_rehashed_mutation(diagnostic):
    chain = _chain(diagnostic)
    assert vast._completed_stage_chain_valid(chain, provider_result={"run_id": "run"}, diagnostic_only=diagnostic)
    changed = deepcopy(chain)
    changed["nested_paid_execution_requested"] = True
    changed["result_digest"] = canonical_digest(changed, digest_field="result_digest")
    assert not vast._completed_stage_chain_valid(changed, provider_result={"run_id": "run"}, diagnostic_only=diagnostic)
    assert not vast._completed_stage_chain_valid(chain, provider_result={"run_id": "run"}, diagnostic_only=not diagnostic)


def test_rehashed_qualification_downgrade_is_not_accepted_as_production():
    chain = _chain(False)
    row = chain["stage_results"][0]
    row["qualification_eligible"] = False
    row["stage_result_digest"] = canonical_digest(row, digest_field="stage_result_digest")
    chain["stage_result_digests"][0] = row["stage_result_digest"]
    chain["result_digest"] = canonical_digest(chain, digest_field="result_digest")
    assert not vast._completed_stage_chain_valid(chain, provider_result={"run_id": "run"}, diagnostic_only=False)
