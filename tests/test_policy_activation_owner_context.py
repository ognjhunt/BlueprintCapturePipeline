"""Policy context uses the policy reservation, not a construction-phase hold."""
import json

import pytest

from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
from blueprint_pipeline.task_evaluation_launch_preparation_contract import launch_preparation_request_digest
from blueprint_pipeline.task_evaluation_scene_owner_attempt_profiles import SceneOwnerAttemptProfileError
from tests.test_task_evaluation_launch_activation_contract import request as activation_request
from tests.test_task_evaluation_launch_activation_worker import _stage_verified_preparation
from tests.test_task_evaluation_policy_run_contract import configuration, setup
from blueprint_pipeline.task_evaluation_policy_run_contract import build_policy_run_plan


@pytest.mark.parametrize("lane", ["native_task_arena_policy_evaluation", "native_task_arena_controls"])
def test_owner_bound_policy_context_does_not_require_a_nonpolicy_reservation(tmp_path, lane):
    preparation, result, _, queue, inputs = _stage_verified_preparation(tmp_path)
    activation = activation_request(lane="native_task_arena_controls")
    activation["preparation"] = {
        "preparation_id": preparation["preparation_id"],
        "request_digest": launch_preparation_request_digest(preparation),
        "result_digest": result["result_digest"],
    }
    preparation, _, adapter, materialized = worker._load_verified_preparation(
        activation_request=activation, preparation_queue_root=queue,
        preparation_input_root=inputs,
    )
    activation["lane"] = lane
    # Website policy admission already reserves a policy attempt. It carries
    # owner identity without the construction/controls-only attempt record.
    preparation["scene_intent_digest"] = "sha256:" + "a" * 64
    activation["authorization"]["reference"] = "scene-intent:" + preparation["scene_intent_digest"]
    policy_setup = setup()
    policy_configuration = configuration(policy_setup)
    preparation["policy_run_configuration"] = policy_configuration
    lineage = {}
    for name in activation["lineage"]:
        if name != "kind":
            path = tmp_path / (name + ".json")
            path.write_text(json.dumps({"kind": name}))
            lineage["lineage." + name] = path

    def build():
        return worker._build_native_context(
            activation_request=activation, preparation_request=preparation,
            policy_run_plan=build_policy_run_plan(policy_configuration, setup=policy_setup),
            adapter=adapter, preparation_materialized=materialized,
            activation_materialized=lineage, activation_root=tmp_path / "activation",
            repository_root=tmp_path, destination_prefix="s3://blueprint/activation",
            profile_dir=tmp_path / "profiles", webapp_catalog=tmp_path / "catalog.json",
            standing_authorization_dir=tmp_path / "authorizations",
            service_account="blueprint", service_group="blueprint",
        )

    if lane == "native_task_arena_controls":
        with pytest.raises(SceneOwnerAttemptProfileError, match="native_owner_attempt_missing"):
            build()
    else:
        context = build()
        assert context["references"]["policy_run"]["configuration_digest"] == policy_configuration["configuration_digest"]
        assert "scene_owner_attempt" not in context["operations"]
        assert not (tmp_path / "activation" / "scene_owner_attempt.json").exists()
