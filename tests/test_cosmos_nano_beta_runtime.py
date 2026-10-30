"""Nano exercises the actual DROID bridge, with a clearly injected policy wire."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from blueprint_pipeline import cosmos_nano_beta_runtime as runtime
from blueprint_pipeline.cosmos_nano_droid_policy_runtime import CosmosNanoDroidPolicySpec
from blueprint_pipeline.cosmos_edge_droid_policy_runtime import CosmosEdgeDroidPolicySpec
from blueprint_pipeline.adp009d_droid_action_execution import plan_chunk_execution
from blueprint_pipeline.adp009d_droid_observation import build_droid_observation
from blueprint_pipeline.cosmos_droid_bundle import runtime_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.droid_policy_bridge import DROID_ROBOARENA_CONCAT_VIEWS
from blueprint_pipeline.policy_interface_binding import observation_interface
from tests.test_cosmos_edge_droid_policy_runtime import _metadata


def execution_spec():
    spec = {
        "runtime_identity": {"provisioning_identity": runtime_contract(runtime.CANDIDATE_ID)},
        "candidate_id": runtime.CANDIDATE_ID,
        "policy_interface_id": runtime.INTERFACE_ID,
        "evaluation_backend": "robolab.v1",
        "checkpoint_digest": runtime.CHECKPOINT_DIGEST,
        "policy_endpoint": {"host": "127.0.0.1", "port": 8000},
        "policy_spec": asdict(CosmosNanoDroidPolicySpec()),
        "open_loop_horizon": 8,
    }
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec


def test_exact_nano_contract_cannot_be_replaced_with_edge_or_json_prompt():
    policy = CosmosNanoDroidPolicySpec()
    runtime.validate_execution_spec(json.loads(json.dumps(execution_spec())))
    assert observation_interface(execution_spec()) == runtime.CANDIDATE_ID
    assert policy.server_metadata()["format_prompt_as_json"] is False
    for altered in [
        replace(policy, format_prompt_as_json=True),
        replace(policy, action_chunk_rows=16),
        replace(policy, model_revision=CosmosEdgeDroidPolicySpec().model_revision),
    ]:
        with pytest.raises(ValueError, match="exact_contract"):
            altered.validate()
    spec = execution_spec()
    spec["runtime_identity"]["provisioning_identity"]["server_python_version"] = "3.12.12"
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    with pytest.raises(ValueError, match="execution_binding_invalid"):
        runtime.validate_execution_spec(spec)


def test_nano_retains_all_native_actions_and_executes_only_eight_observed_state_steps():
    policy = CosmosNanoDroidPolicySpec()
    requests = []

    class Transport:
        def __init__(self, **kwargs):
            pass

        def get_server_metadata(self):
            return _metadata(policy)

        def infer(self, observation):
            requests.append(observation)
            actions = np.zeros((32, 8))
            actions[:, :7] = observation["observation/joint_position"]
            actions[:, 7] = 0.75
            return {"action": actions}

    raw = {
        key: np.full((720, 1280, 3), index * 80, np.uint8)
        for index, key in enumerate(DROID_ROBOARENA_CONCAT_VIEWS)
    }
    observation = build_droid_observation(
        candidate_id=runtime.CANDIDATE_ID,
        camera_rgb=raw,
        joint_position=[0] * 7,
        gripper_position=0.25,
        prompt="Move the cup.",
    )
    observation["privileged_task_state"] = {"success": True}
    client = runtime.CosmosNanoSimulatorClient(
        policy_spec=asdict(policy), host="127.0.0.1", port=8000, client_factory=Transport
    )
    assert client.preflight_readiness()["candidate_policy_queried"] is False
    actions = client.infer(observation)
    assert actions.shape == (32, 8) and len(requests) == 1
    assert client.last_inference_evidence()["action_chunk_shape"] == [32, 8]
    assert set(DROID_ROBOARENA_CONCAT_VIEWS).issubset(requests[0])
    assert "privileged_task_state" not in requests[0]
    plan = plan_chunk_execution(
        actions, horizon=8, action_space=client.action_space, candidate_id=runtime.CANDIDATE_ID
    )
    assert plan["executed_rows"] == 8 and plan["discarded_rows"] == 24
    assert (
        plan["source_action_space"] == "cosmos_nano_absolute_joint_position_plus_absolute_gripper"
    )


def test_nano_requires_actual_third_camera_and_verifies_publisher_inventory():
    manifest = runtime.verified_snapshot_manifest(runtime.snapshot_manifest_path())
    assert len(manifest["required_files"]) == 39
    assert sum(row["size_bytes"] for row in manifest["required_files"]) == 32_936_198_308
    with pytest.raises(ValueError, match="required_policy_camera_unavailable"):
        runtime.candidate_execution_spec(
            scene_plan={
                "cameras": [
                    {"role": "external", "policy_input": True},
                    {"role": "wrist", "policy_input": True},
                    {"role": "exterior_right", "policy_input": False},
                ]
            },
            contract={},
            source_commit="a" * 40,
        )


@pytest.mark.parametrize("candidate,prompt_json", [("cosmos3_edge_policy_droid", True), ("cosmos3_nano_policy_droid", False)])
def test_server_selects_the_publisher_specific_prompt_without_changing_actions(tmp_path, monkeypatch, candidate, prompt_json):
    import sys
    from types import ModuleType, SimpleNamespace
    from blueprint_pipeline import cosmos_edge_droid_policy_server as server
    from blueprint_pipeline import cosmos_nano_droid_policy_runtime as nano_policy
    from blueprint_pipeline.cosmos_droid_bundle import profile

    calls = []
    official = ModuleType("cosmos_framework.scripts.action_policy_server_robolab")
    official.RobolabServerArgs = lambda **kwargs: SimpleNamespace(**kwargs)

    class Service:
        def __init__(self, args):
            calls.append(args)

        def infer(self, observation):
            pytest.fail("server startup must not run model inference")

    class Server:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def serve_forever(self):
            pass

    official.RobolabPolicyService = Service
    official._load_openpi_websocket_policy_server = lambda: Server
    monkeypatch.setitem(sys.modules, official.__name__, official)
    # This argument-routing test does not claim to verify or load model weights.
    def verifier(**kwargs):
        return {"local_snapshot_verified": True}
    monkeypatch.setattr(server, "verify_local_policy_snapshot", verifier)
    monkeypatch.setattr(nano_policy, "verify_local_policy_snapshot", verifier)
    selected = profile(candidate)
    result = server.serve_identity_bound_policy(checkpoint_path=tmp_path,
        snapshot_manifest_path=selected.snapshot_manifest_path(), host="127.0.0.1", port=8000,
        output_dir=tmp_path / "output", candidate_id=candidate)
    args = calls[0]
    assert args.format_prompt_as_json is prompt_json
    assert (args.action_chunk_size, args.action_dim, args.conditioning_fps, args.history_length) == (32, 8, 15, 1)
    assert args.action_space == "joint_pos" and args.use_state is True
    assert result["metadata"]["policy_id"] == selected.MODEL_ID
