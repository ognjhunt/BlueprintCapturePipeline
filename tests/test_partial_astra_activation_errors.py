"""Keep partial-reuse refusal predicates visible at the activation boundary."""
import pytest

from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
from blueprint_pipeline import task_evaluation_partial_astra_transport as transport
from tests.test_task_evaluation_launch_activation_worker import (
    test_activation_builds_robot_neutral_scene_configuration_context as build_fixture,
)


def test_partial_reuse_refusal_is_a_reportable_activation_error(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_owner_attempt", lambda operations, *_a: operations.update(
        scene_owner_attempt=str(tmp_path / "owner.json")))

    def refuse(**_kwargs):
        transport._require(False, "construction_owner_binding_changed")

    monkeypatch.setattr(transport, "select_partial_astra_source", refuse)
    with pytest.raises(worker.TaskEvaluationLaunchActivationWorkerError,
                       match="^partial_astra_transport_construction_owner_binding_changed$"):
        build_fixture(tmp_path)


def test_unknown_value_error_is_not_promoted_to_public_predicate(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_owner_attempt", lambda operations, *_a: operations.update(
        scene_owner_attempt=str(tmp_path / "owner.json")))

    def refuse(**_kwargs):
        raise ValueError("unknown private artifact content")

    monkeypatch.setattr(transport, "select_partial_astra_source", refuse)
    with pytest.raises(ValueError) as failure:
        build_fixture(tmp_path)
    assert not isinstance(failure.value, worker.TaskEvaluationLaunchActivationWorkerError)
