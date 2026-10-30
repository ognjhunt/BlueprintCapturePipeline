from __future__ import annotations

from dataclasses import asdict
import json
import subprocess

import numpy as np
import pytest

from blueprint_pipeline import cosmos_edge_beta_runtime as runtime
from blueprint_pipeline import adp009d_policy_server_worker as server_worker
from blueprint_pipeline.adp009d_droid_action_execution import plan_chunk_execution
from blueprint_pipeline.adp009d_droid_observation import build_droid_observation
from blueprint_pipeline.cosmos_edge_beta_provisioning import build_provisioning_script, DEPENDENCY_LOCK_SHA256
from blueprint_pipeline.cosmos_edge_droid_policy_runtime import CosmosEdgeDroidPolicySpec
from blueprint_pipeline.cosmos_droid_bundle import runtime_contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.droid_policy_bridge import DROID_ROBOARENA_CONCAT_VIEWS
from blueprint_pipeline.policy_interface_binding import observation_interface
from tests.test_cosmos_edge_droid_policy_runtime import _metadata
from tests.test_beta_robolab_runtime import stock_robolab  # noqa: F401 - shared pinned upstream loop fixture


def _policy():
    return CosmosEdgeDroidPolicySpec(snapshot_manifest_sha256=runtime.SNAPSHOT_MANIFEST_SHA256)


def _execution():
    spec = {"runtime_identity": {"provisioning_identity": runtime_contract(runtime.CANDIDATE_ID)},
            "candidate_id": runtime.CANDIDATE_ID, "policy_interface_id": runtime.INTERFACE_ID,
            "evaluation_backend": "robolab.v1", "checkpoint_digest": runtime.CHECKPOINT_DIGEST,
            "policy_endpoint": {"host": "127.0.0.1", "port": 8000},
            "policy_spec": asdict(_policy()), "open_loop_horizon": 8}
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    return spec


def test_simulator_uses_real_three_views_and_all_native_rows_with_position_provenance():
    observed = []
    native_actions = np.arange(32 * 8, dtype=float).reshape(32, 8) / 1000

    class Transport:
        def __init__(self, **kwargs):
            assert kwargs == {"host": "127.0.0.1", "port": 8000, "api_key": None}

        def get_server_metadata(self):
            return _metadata(_policy())

        def infer(self, observation):
            observed.append(observation)
            return {"action": native_actions}

    raw = {key: np.full((720, 1280, 3), index * 80, dtype=np.uint8)
           for index, key in enumerate(DROID_ROBOARENA_CONCAT_VIEWS)}
    observation = build_droid_observation(candidate_id=runtime.CANDIDATE_ID, camera_rgb=raw,
        joint_position=[0] * 7, gripper_position=0, prompt="Move the task object to the destination.")
    observation["privileged_task_state"] = {"success": True}
    client = runtime.CosmosEdgeSimulatorClient(policy_spec=asdict(_policy()), host="127.0.0.1",
                                               port=8000, client_factory=Transport)
    response = client.infer(observation)
    assert response.shape == (32, 8)
    np.testing.assert_array_equal(response, native_actions)
    assert "privileged_task_state" not in observed[0]
    for key in DROID_ROBOARENA_CONCAT_VIEWS:
        assert observed[0][key].shape == (224, 224, 3)
    assert len({observed[0][key].tobytes() for key in DROID_ROBOARENA_CONCAT_VIEWS}) == 3
    plan = plan_chunk_execution(response, horizon=8,
                               action_space=client.action_space, candidate_id=runtime.CANDIDATE_ID)
    assert plan["executed_rows"] == 8 and plan["discarded_rows"] == 24
    assert plan["source_action_space"] == "cosmos_edge_absolute_joint_position_plus_absolute_gripper"
    assert observation_interface(_execution()) == runtime.CANDIDATE_ID


@pytest.mark.parametrize("backend", [None, "robolab.v1"])
@pytest.mark.usefixtures("stock_robolab")
def test_actual_native_episode_consumes_cosmos_arrays_and_retains_three_view_requests(backend):
    from tests.test_adp009d_policy_episode import _Environment, _run
    from blueprint_pipeline.droid_policy_bridge import DROID_EXTERIOR_VIEW_2

    calls, retained = [], []

    class Environment(_Environment):
        def read_policy_inputs(self):
            inputs = super().read_policy_inputs()
            inputs[DROID_EXTERIOR_VIEW_2] = np.full((720, 1280, 3), 120, dtype=np.uint8)
            return inputs

    class Transport:
        def __init__(self, **kwargs):
            pass

        def get_server_metadata(self):
            return _metadata(_policy())

        def infer(self, observation):
            calls.append(observation)
            actions = np.zeros((32, 8))
            actions[:, 0] = 0.25
            actions[:, 7] = 0.9
            return {"action": actions}

    client = runtime.CosmosEdgeSimulatorClient(policy_spec=asdict(_policy()), host="127.0.0.1",
                                               port=8000, client_factory=Transport)
    client.bind_request_evidence_sink(retained.append)
    environment = Environment()
    result = _run(environment, client, candidate_id=runtime.CANDIDATE_ID, evaluation_backend=backend)
    assert len(calls) == 4 and len(environment.steps) == 38
    assert len(result["commanded_actions"]) == 32
    assert result["candidate_policy_queried"] is True
    assert client.last_inference_evidence()["action_chunk_shape"] == [32, 8]
    assert client.last_request_evidence()["credential_material_retained"] is False
    assert all(DROID_EXTERIOR_VIEW_2 in row for row in calls)


@pytest.mark.parametrize("value", [np.zeros((31, 8)), np.full((32, 8), np.nan)])
def test_bad_returned_actions_still_record_a_completed_candidate_query(value):
    from blueprint_pipeline.native_task_arena_policy_worker import _PolicyQueryTracker
    from blueprint_pipeline.adp009d_droid_action_execution import DroidActionExecutionError
    from blueprint_pipeline.adp009d_policy_episode_evidence import prevalidation_vendor_action_evidence

    class Transport:
        def __init__(self, **kwargs):
            pass

        def get_server_metadata(self):
            return _metadata(_policy())

        def infer(self, observation):
            return {"action": value}

    client = runtime.CosmosEdgeSimulatorClient(policy_spec=asdict(_policy()), host="127.0.0.1",
                                               port=8000, client_factory=Transport)
    tracked = _PolicyQueryTracker(client)
    observation = {key: np.zeros((224, 224, 3), dtype=np.uint8) for key in DROID_ROBOARENA_CONCAT_VIEWS}
    observation.update({"observation/joint_position": np.zeros(7),
                        "observation/gripper_position": np.zeros(1), "prompt": "Move the task object."})
    with pytest.raises(DroidActionExecutionError):
        tracked.infer(observation)
    assert tracked.candidate_policy_queried is True
    evidence = client.last_inference_evidence()
    assert evidence["server_response_received"] is True and evidence["actions_extracted"] is False
    assert prevalidation_vendor_action_evidence(evidence, query_index=0)["action_payload_returned"] is True
    assert client.preflight_readiness()["candidate_policy_queried"] is False
    with pytest.raises(ValueError, match="inference_evidence_missing"):
        client.last_inference_evidence()


def test_snapshot_inventory_and_execution_are_exact_and_secret_free(tmp_path):
    manifest = runtime.verified_snapshot_manifest(runtime.snapshot_manifest_path())
    assert len(manifest["required_files"]) == 21
    assert sum(row["size_bytes"] for row in manifest["required_files"]) == 9_154_401_203
    runtime.validate_execution_spec(json.loads(json.dumps(_execution())))
    bad = _execution()
    bad["policy_endpoint"]["host"] = "policy.example"
    bad["execution_spec_digest"] = canonical_digest(bad, digest_field="execution_spec_digest")
    with pytest.raises(ValueError, match="execution_binding_invalid"):
        runtime.validate_execution_spec(bad)
    manifest["required_files"].pop()
    altered = tmp_path / "manifest.json"
    altered.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="snapshot_manifest_mismatch"):
        runtime.verified_snapshot_manifest(altered)


def test_bootstrap_uses_cosmos_locked_environment_and_does_not_install_model_into_isaac(tmp_path):
    script = build_provisioning_script()
    path = tmp_path / "provision.sh"
    path.write_text(script)
    checked = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    from blueprint_pipeline.cosmos_droid_dependencies import verified_dependency_manifest
    dependencies = verified_dependency_manifest()
    assert dependencies["upstream_uv_lock_sha256"] == DEPENDENCY_LOCK_SHA256
    assert dependencies["interpreter"]["version"] == "3.13.9"
    assert "blueprint_pipeline.cosmos_droid_provisioning" in script
    assert "uv sync" not in script and "snapshot_download" not in script
    command = server_worker.build_serve_command(candidate_id=runtime.CANDIDATE_ID,
        python="/separate/venv/bin/python", source_root="/source", checkpoint_root="/checkpoint",
        port=8000, policy_spec_path="/execution.json", checkpoint_inventory_path="/manifest.json")
    assert command[:3] == ["/separate/venv/bin/python", "-m", "blueprint_pipeline.cosmos_edge_droid_policy_server"]
    assert command[command.index("--snapshot-manifest") + 1] == "/manifest.json"
    assert "openpi_droid_policy_runtime.py" not in " ".join(command)


def test_readiness_checks_cosmos_identity_without_requesting_an_action(monkeypatch):
    events = []

    class Client:
        def __init__(self, **kwargs):
            events.append(kwargs)

        def evidence_summary(self):
            return {"identity_verified": True, "request_count": 0, "server_metadata": {"model": "exact"}}

        def infer(self, observation):
            pytest.fail("readiness must not advance model state")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(runtime, "CosmosEdgeSimulatorClient", Client)
    result = server_worker.attempt_handshake("127.0.0.1", 8000, candidate_id=runtime.CANDIDATE_ID,
                                            policy_spec=asdict(_policy()))
    assert result["handshake_completed"] is True
    assert result["candidate_inference_performed"] is False
    assert events[-1] == "closed"


def test_beta_bundle_requires_the_actual_cosmos_bootstrap_and_manifest():
    from blueprint_pipeline.policy_canary_session_bundle_identity import run_kind_required_entries
    required = run_kind_required_entries({"execution_mode": "blueprint_beta_screening_session",
        "run_kind": "blueprint_beta_screening", "episodes_per_policy": 10,
        "candidate_ids": [runtime.CANDIDATE_ID], "evaluation_backend": "robolab.v1",
        "cosmos_runtime": runtime_contract(runtime.CANDIDATE_ID)})
    assert "provider_runtime/" + runtime.SNAPSHOT_STAGED_NAME in required
    assert "provider_runtime/blueprint_pipeline/cosmos_edge_beta_runtime.py" in required
