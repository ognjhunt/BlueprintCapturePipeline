"""Real relay/JSONL child with the existing scored G1 fixture, never GPU proof."""

import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import pytest

from blueprint_pipeline import native_g1_team_relay_runtime_session as remote
from blueprint_pipeline.native_g1_team_policy_relay import RelayBinding, G1PolicyRelayServer
from blueprint_pipeline.native_g1_team_policy_conformance import run_g1_team_policy_synthetic_conformance
from blueprint_pipeline.native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
from blueprint_pipeline.native_g1_team_runtime_session import NativeG1TeamRuntimeSession
from blueprint_pipeline.native_g1_team_supervised_episode import run_g1_team_supervised_episode
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_g1_team_policy_jsonl_client import _process, _close
from tests.test_native_g1_team_runtime_session import _action
from tests.test_native_g1_team_policy_approval import _approval
from tests.test_native_g1_team_scored_scene_episode import _inputs
from tests.test_native_g1_shared_scene_episode import _Bridge
from tests.test_native_task_episode_environment import _RigidNativeReadback
from tests.test_team_policy_delivery_profile import OWNER, _profile

pytestmark = pytest.mark.slow
SECRET = "e" * 64
PACKET = "sha256:" + "f" * 64


@pytest.fixture
def paired(tmp_path, monkeypatch, request):
    setup, profile, scene = _inputs(tmp_path, monkeypatch)
    mode = getattr(request, "param", "container")
    if mode == "noncontainer_artifact":
        profile = _profile(setup, {"mode": mode, "artifact_uri": "https://files.example.org/policy.tar.gz",
                                  "artifact_sha256": "sha256:" + "b" * 64, "entrypoint": "policy/run.py",
                                  "protocol": "jsonl_observation_action_v1"})
    binding = RelayBinding(PACKET, profile["profile_digest"], setup["setup_digest"], mode)
    process = _process("{'ok': True, 'action_chunk': " + repr([_action()]) + "}")
    policy = NativeG1TeamPolicyJsonlClient(process, timeout_seconds=2, profile_digest=profile["profile_digest"])
    conformance = run_g1_team_policy_synthetic_conformance(profile=profile, trusted_setup=setup,
                                                          authenticated_owner=OWNER, policy_client=policy)
    host_root = tmp_path / "host"
    host_root.mkdir(mode=0o700)
    class Lease:
        def close(self):
            _close(process)
            value = {"status": "process_exited", "exit_code": process.poll()}
            value["receipt_digest"] = canonical_digest(value)
            return value
    session = NativeG1TeamRuntimeSession(client=policy, conformance=conformance, profile_digest=binding.profile_digest,
                                        delivery_mode=mode, output_dir=host_root, lease=Lease())
    with tempfile.TemporaryDirectory(prefix="g1-remote-", dir=str(Path("/tmp").resolve())) as root:
        root = Path(root)
        server = G1PolicyRelayServer(path=root / "wire", binding=binding, secret=SECRET)
        config = root / "private.json"
        remote.write_g1_team_relay_config(path=config, binding=binding, socket_path=server.path, secret=SECRET)
        results = []
        thread = threading.Thread(target=lambda: results.append(server.serve_one(session_factory=lambda: session,
                                                                                 timeout_seconds=2)))
        thread.start()
        yield setup, profile, scene, binding, config, session, process, results, thread
        server.close()
        thread.join(timeout=3)
        assert not thread.is_alive()
        session.close()


def _open(paired, output, **changes):
    setup, profile, _, _, config, *_ = paired
    args = {"config_path": config, "execution_packet_digest": PACKET,
            "profile": profile, "trusted_setup": setup, "authenticated_owner": OWNER,
            "output_dir": output, "timeout_seconds": 2}
    args.update(changes)
    return remote.open_g1_team_relay_runtime_session(**args)


@pytest.mark.parametrize("paired", ["container", "noncontainer_artifact"], indirect=True)
def test_existing_scored_episode_uses_original_remote_mode_and_actual_close(paired, tmp_path, monkeypatch):
    from blueprint_pipeline import native_g1_joint_episode_environment as environment
    from blueprint_pipeline import native_task_arena_readback
    setup, profile, scene, _, config, host, process, results, thread = paired
    from tests.test_adp_task_scoring import _rigid_v2_spec
    # The shared scene fixture originally supplies only telemetry fields.
    # Add the real scorer's frozen predicates and its native fixture start pose.
    scene.plan["task_spec"] = {**_rigid_v2_spec(), **scene.plan["task_spec"],
                               "start_pose_world": [1.1, 2.1, 0.8, 0.0, 0.0, 0.0, 1.0]}
    scene.plan["plan_digest"] = canonical_digest(scene.plan, digest_field="plan_digest")
    monkeypatch.setattr(environment, "NativeG1JointEpisodeEnvironment", lambda **kw: scene)
    monkeypatch.setattr(native_task_arena_readback, "NativeRigidTaskArenaReadback", lambda built: _RigidNativeReadback(
        finger_separation_m=0.08, grasp_frame_position_world_m=[1.1, 2.1, 0.9],
        destination_scene_forbidden_contact_peak_force_n=0.0))
    mode = profile["delivery"]["mode"]
    binding = ({"mode": mode, "profile_digest": profile["profile_digest"], "image_ref": profile["delivery"]["image_ref"],
                "gpu_device": 0} if mode == "container" else
               {"mode": mode, "profile_digest": profile["profile_digest"], "artifact_sha256": profile["delivery"]["artifact_sha256"],
                "staged_artifact_path": str(tmp_path / "operator-staged.tar")})
    approval = _approval(setup, profile, binding)
    approval["expires_at_epoch"] = time.time() + 3600
    approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
    output = tmp_path / "guest"
    result = run_g1_team_supervised_episode(built=type("Built", (), {"plan": scene.plan})(),
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER, operator_approval=approval,
        sonic_bridge=_Bridge(), objective_id="task_success", max_steps=2, output_dir=output,
        to_tensor=lambda value: value, make_action_tensor=lambda value, **kw: value,
        policy_relay_config_path=config, execution_packet_digest=PACKET)
    assert result["status"] == "completed_development_only", (result["phase_reached"], result["blocker_type"], results)
    assert result["delivery_mode"] == mode and result["policy_query_count"] > 0
    thread.join(timeout=3)
    assert not thread.is_alive() and process.poll() is not None
    host_close = json.loads((host.output_dir / "native_g1_team_runtime_session.v1.json").read_text())
    guest_close = json.loads((output / "runtime/native_g1_team_runtime_session.v1.json").read_text())
    assert guest_close == host_close
    assert result["runtime_teardown_digest"] == host_close["receipt_digest"]
    assert host_close["linked_scored_episode_result_digest"] == result["scored_episode_result_digest"]
    assert results[0]["inference_query_count"] == result["policy_query_count"]
    assert host_close["child_teardown_required"] is True
    assert host_close["provider_teardown_verified"] is False
    scored = json.loads((output / "episode/native_g1_team_scored_scene_episode.v1.json").read_text())
    assert scored["score"]["status"] == "scored"
    assert scored["score"]["outcome"] == "never_moved"
    assert scored["score"]["task_succeeded"] is False
    assert SECRET not in "".join(path.read_text() for path in output.rglob("*.json"))


@pytest.mark.parametrize("fault", ["packet", "secret_mode", "extra", "symlink", "absent"])
def test_wrong_private_configuration_never_sends_site_input(paired, tmp_path, fault):
    setup, profile, _, _, config, host, process, results, thread = paired
    args = {}
    if fault == "packet":
        args["execution_packet_digest"] = "sha256:" + "d" * 64
    elif fault == "secret_mode":
        config.chmod(0o644)
    elif fault == "extra":
        value = json.loads(config.read_text())
        value["command"] = "do-not-run"
        config.write_text(json.dumps(value))
    elif fault == "symlink":
        alias = config.parent / "alias.json"
        alias.symlink_to(config)
        args["config_path"] = alias
    else:
        args["config_path"] = config.parent / "absent.json"
    with pytest.raises(ValueError):
        _open(paired, tmp_path / "invalid", **args)
    assert host.client._request_index == 2  # only actual synthetic reset/infer


@pytest.mark.parametrize("fault", ["queries", "zero", "profile", "setup", "mode", "digest"])
def test_invalid_episode_cannot_link_or_fabricate_runtime_close(paired, tmp_path, fault):
    setup, profile, _, _, _, host, process, results, thread = paired
    session = _open(paired, tmp_path / "guest")
    session.client.reset(seed=3)
    session.client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8), observation_state=[0.0] * 64,
                               task="Place the book.")
    from blueprint_pipeline.native_g1_shared_scene_episode import team_policy_candidate_id
    episode = {"schema_version": "native_g1_team_scored_scene_episode.v1", "status": "development_only_scored_episode",
               "profile_digest": profile["profile_digest"], "candidate_id": team_policy_candidate_id(profile["profile_digest"]),
               "source_setup_digest": setup["setup_digest"], "delivery_mode": profile["delivery"]["mode"],
               "policy_query_count": 1}
    if fault == "queries":
        episode["policy_query_count"] = 2
    elif fault == "zero":
        episode["policy_query_count"] = 0
    elif fault in {"profile", "setup"}:
        episode[{"profile": "profile_digest", "setup": "source_setup_digest"}[fault]] = "sha256:" + "a" * 64
    elif fault == "mode":
        episode["delivery_mode"] = "noncontainer_artifact"
    episode["result_digest"] = canonical_digest(episode)
    if fault == "digest":
        episode["result_digest"] = "sha256:" + "a" * 64
    with pytest.raises((EOFError, ValueError)):
        # Bypass the local validator to exercise the host's independent check.
        session.client._exchange("link_episode", {"episode": episode})
    thread.join(timeout=3)
    assert not thread.is_alive() and process.poll() is not None
    assert host._linked_episode_digest is None
    assert not (tmp_path / "guest/native_g1_team_runtime_session.v1.json").exists()


@pytest.mark.parametrize("fault", ["digest", "child_missing", "extra", "blocked", "linked"])
def test_altered_child_close_receipt_blocks_guest_completion(paired, tmp_path, monkeypatch, fault):
    *_, host, process, results, thread = paired
    original = host.close
    def broken():
        value = dict(original())
        if fault == "digest":
            value["child_teardown_digest"] = "sha256:" + "a" * 64
            return value  # deliberately keep original digest
        if fault == "child_missing":
            value["child_teardown_digest"] = None
        elif fault == "extra":
            value["untrusted_payload"] = SECRET
        elif fault == "blocked":
            value["status"] = "teardown_blocked"
        else:
            value["linked_scored_episode_result_digest"] = "sha256:" + "a" * 64
        value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
        return value
    monkeypatch.setattr(host, "close", broken)
    session = _open(paired, tmp_path / "guest")
    with pytest.raises((ValueError, EOFError)):
        session.close()
    assert not (tmp_path / "guest/native_g1_team_runtime_session.v1.json").exists()


def test_repeated_host_episode_link_never_reaches_policy_child(paired, tmp_path):
    from blueprint_pipeline.native_g1_shared_scene_episode import team_policy_candidate_id
    setup, profile, _, _, _, host, process, results, thread = paired
    session = _open(paired, tmp_path / "guest")
    session.client.reset(seed=3)
    session.client.infer_chunk(front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                               observation_state=[0.0] * 64, task="Place the book.")
    episode = {"schema_version": "native_g1_team_scored_scene_episode.v1", "status": "development_only_scored_episode",
               "profile_digest": profile["profile_digest"], "candidate_id": team_policy_candidate_id(profile["profile_digest"]),
               "source_setup_digest": setup["setup_digest"], "delivery_mode": profile["delivery"]["mode"],
               "policy_query_count": 1}
    episode["result_digest"] = canonical_digest(episode)
    session.link_scored_episode(episode)
    requests = host.client._request_index
    with pytest.raises(EOFError):
        session.client._exchange("link_episode", {"episode": episode})
    thread.join(timeout=3)
    assert not thread.is_alive() and process.poll() is not None
    assert host.client._request_index == requests == 4
    assert host._linked_episode_digest == episode["result_digest"]
    assert results[0]["forwarded_request_count"] == 2
    assert not (tmp_path / "guest/native_g1_team_runtime_session.v1.json").exists()


@pytest.mark.parametrize("fault", ["extra", "site_input", "robot", "digest"])
def test_foreign_conformance_closes_child_before_guest_receipt(paired, tmp_path, fault):
    *_, host, process, results, thread = paired
    if fault == "extra":
        host.conformance["untrusted_payload"] = SECRET
    elif fault == "site_input":
        host.conformance["site_observation_sent"] = True
    elif fault == "robot":
        host.conformance["robot_preset_id"] = "franka"
    if fault != "digest":
        host.conformance["receipt_digest"] = canonical_digest(host.conformance, digest_field="receipt_digest")
    else:
        host.conformance["receipt_digest"] = "sha256:" + "a" * 64
    with pytest.raises((ValueError, EOFError)):
        _open(paired, tmp_path / "guest")
    thread.join(timeout=3)
    assert not thread.is_alive() and process.poll() is not None
    assert host.client._request_index == 2
    assert not (tmp_path / "guest").exists()
    assert SECRET not in json.dumps(results)
