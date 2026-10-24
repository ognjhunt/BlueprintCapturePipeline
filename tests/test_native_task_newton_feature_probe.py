import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_isaaclab_ga import IMAGE
from blueprint_pipeline import native_task_newton_feature_probe as probe


def request():
    comparison = {"variant": "isaaclab_3_ga", "runtime_identity": {"container_image": IMAGE}}
    comparison["request_digest"] = canonical_digest(comparison, digest_field="request_digest")
    plan = {"plan_digest": "sha256:" + "a" * 64, "scenario": {"seed": 17}}
    return probe.build_feature_request(comparison, plan), comparison, plan


@pytest.mark.parametrize("mutation", ["renderer", "physics_backend", "control_steps", "scene_plan_digest",
                                      "qualification_effect", "candidate_policy_queried"])
def test_feature_cannot_change_inputs_or_borrow_qualification(mutation):
    value, comparison, plan = request()
    value[mutation] = "changed"
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    with pytest.raises(ValueError, match="feature_request_invalid"):
        probe.validate_feature_request(value, comparison=comparison, plan=plan)


def test_feature_refuses_non_ga_or_mutated_parent():
    value, comparison, plan = request()
    for key, change in [("variant", "baseline"), ("request_digest", "sha256:wrong")]:
        other = deepcopy(comparison)
        other[key] = change
        with pytest.raises(ValueError, match="feature_request_invalid"):
            probe.validate_feature_request(value, comparison=other, plan=plan)


def test_feature_requires_exact_ga_sources_before_bundle_transport():
    from blueprint_pipeline import native_task_isaaclab_ga as ga
    source = {"runtime_profile": ga.PROFILE, "container_image": ga.IMAGE,
              "repositories": [{"commit": ga.COMMIT, "tree": ga.TREE},
                               {"commit": ga.ARENA_COMMIT, "tree": ga.ARENA_TREE}]}
    probe.validate_feature_source(source)
    for key, change in [("runtime_profile", "baseline"), ("container_image", "latest"), ("repositories", [])]:
        with pytest.raises(ValueError, match="ga_sources_required"):
            probe.validate_feature_source({**source, key: change})


class Tensor(np.ndarray):
    def clone(self):
        return self.copy()

    def abs(self):
        return np.abs(self)


class FakeEnv:
    def __init__(self, *, drift=0.0, nonfinite=False, wrong_backend=False):
        manager = type("NewtonMJWarpManager", (), {})
        manager.__module__ = "isaaclab_newton.physics.mjwarp_manager" if not wrong_backend else "physx"
        solver = type("SolverMuJoCo", (), {})
        manager._solver = solver()
        manager.get_device = lambda: "cuda:0"
        self.sim = SimpleNamespace(physics_manager=manager)
        self.unwrapped = self
        self.data = SimpleNamespace(joint_pos=np.zeros((1, 13)).view(Tensor),
                                    joint_vel=np.zeros((1, 13)).view(Tensor))
        self.scene = {"robot": SimpleNamespace(data=self.data,
            joint_names=[*[f"panda_joint{i}" for i in range(1, 8)], "finger_joint", *[f"mimic{i}" for i in range(5)]])}
        self.steps = 0
        self.resets = []
        self.drift = drift
        self.nonfinite = nonfinite

    def reset(self, *, seed):
        self.resets.append(seed)
        self.data.joint_pos[:] = 0

    def step(self, action):
        assert action.shape == (1, 8)
        assert np.all(action == 0)
        self.steps += 1
        self.data.joint_pos[:] += self.drift
        if self.nonfinite:
            self.data.joint_pos[0, 0] = np.nan


def fake_torch():
    return SimpleNamespace(as_tensor=lambda value, **kwargs: np.asarray(value).view(Tensor),
                           isfinite=np.isfinite, cuda=SimpleNamespace(synchronize=lambda: None))


def test_hold_rehearsal_runs_real_interface_and_reports_drift_without_scoring(tmp_path):
    value, _, _ = request()
    env = FakeEnv(drift=.0001)
    captures = []
    def snapshot(**kwargs):
        captures.append(kwargs["snapshot_id"])
        return {"cameras": [{"observability": {"render_passed": True}} for _ in range(3)]}
    result = probe.run_hold_probe(built=SimpleNamespace(env=env, camera_scene_names={}),
        request=value, torch=fake_torch(), snapshot=snapshot, output_root=tmp_path)
    assert env.steps == 64
    assert env.resets == [17, 17]
    assert captures == ["newton_reset", "newton_after_hold"]
    assert result["max_joint_drift_rad"] == pytest.approx(.0064)
    assert result["reset_max_joint_error_rad"] == 0
    assert "score" not in result


@pytest.mark.parametrize("problem", ["nonfinite", "wrong_backend", "bad_camera"])
def test_native_smoke_refuses_silent_fallback_and_invalid_observations(tmp_path, problem):
    value, _, _ = request()
    env = FakeEnv(nonfinite=problem == "nonfinite", wrong_backend=problem == "wrong_backend")
    def snapshot(**kwargs):
        return {"cameras": [{"observability": {"render_passed": problem != "bad_camera"}}
                            for _ in range(3)]}
    with pytest.raises(ValueError):
        probe.run_hold_probe(built=SimpleNamespace(env=env, camera_scene_names={}),
            request=value, torch=fake_torch(), snapshot=snapshot, output_root=tmp_path)


def test_launch_failure_retains_typed_media_gap_and_no_qualification(tmp_path, monkeypatch):
    from blueprint_pipeline import native_task_isaaclab_launch as launch
    def fail(*args, **kwargs):
        raise RuntimeError("retained native launch refusal")
    monkeypatch.setattr(launch, "launch_native_task_isaaclab", fail)
    monkeypatch.setattr("blueprint_pipeline.native_task_nurec_render_setup.appearance_render_path_from_plan",
                        lambda plan: "particlefield")
    value, _, plan = request()
    result, app = probe.run_feature_probe(request=value, plan=plan, runtime=tmp_path,
                                         output_root=tmp_path, snapshot=None, implementation_commit="a" * 40)
    assert app is None
    retained = json.loads((tmp_path / probe.RESULT_FILENAME).read_text())
    assert result == retained
    assert result["status"] == "blocked"
    assert result["media_gap"]["type"] == "before_first_observation"
    assert result["construction_gate_qualified"] is False
    assert result["runtime_promotion_authorized"] is False
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")


def test_feature_bundle_seals_request_and_cannot_omit_reference(tmp_path, monkeypatch):
    import zipfile
    from tests.test_native_task_arena_bundle import _articulated_packet, _runtime_source_packet
    from blueprint_pipeline.native_task_runtime_comparison_cli import main
    from blueprint_pipeline.native_task_arena_construction_bundle import build_native_task_arena_construction_bundle
    packet, plan = _articulated_packet(tmp_path)
    # This transport fixture predates camera/assets comparison bindings.
    plan.update(objects=[], cameras=[])
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path = packet / "native_task_arena_scene_plan.v1.json"
    plan_path.write_text(json.dumps(plan))
    receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["arena_scene_plan_digest"] = plan["plan_digest"]
    import hashlib
    row = next(row for row in receipt["artifacts"] if row["role"] == "arena_scene_plan")
    row.update(size_bytes=plan_path.stat().st_size, sha256="sha256:" + hashlib.sha256(plan_path.read_bytes()).hexdigest())
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path.write_text(json.dumps(receipt))
    comparison_path = tmp_path / "ga-request.json"
    assert main(["prepare-construction", "--packet-dir", str(packet), "--variant", "isaaclab_3_ga",
                 "--output", str(comparison_path)]) == 0
    feature_path = tmp_path / probe.REQUEST_FILENAME
    assert main(["prepare-newton-smoke", "--packet-dir", str(packet),
                 "--reference-construction-request", str(comparison_path), "--output", str(feature_path)]) == 0
    # Asset-byte preservation has its own exact digest tests; this fixture
    # exercises request transport using the existing synthetic scene packet.
    monkeypatch.setattr("blueprint_pipeline.native_task_droid_runtime_compat.comparison_robot_asset_inputs", lambda path: {})
    kwargs = dict(job_dir=tmp_path / "bundle", packet_dir=packet,
                  runtime_source_packet_receipt=_runtime_source_packet(tmp_path),
                  implementation_commit="a" * 40, container_image=IMAGE,
                  native_feature_request_path=feature_path)
    with pytest.raises(ValueError, match="requires_retained_ga_reference"):
        build_native_task_arena_construction_bundle(**kwargs)
    with pytest.raises(ValueError, match="ga_sources_required"):
        build_native_task_arena_construction_bundle(**kwargs, runtime_comparison_request_path=comparison_path)
    # The generic archive fixture has synthetic baseline source bytes. Exact
    # GA source admission is exercised above; this lane tests sealed transport.
    monkeypatch.setattr(probe, "validate_feature_source", lambda source: None)
    receipt = build_native_task_arena_construction_bundle(**kwargs, runtime_comparison_request_path=comparison_path)
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        assert json.loads(archive.read("provider_runtime/runtime_inputs/" + probe.REQUEST_FILENAME)) == json.loads(feature_path.read_text())
        assert "provider_runtime/blueprint_pipeline/native_task_newton_feature_probe.py" in archive.namelist()
        archive.extractall(tmp_path / "extracted")
    assert receipt["execution_mode"] == "construction_canary"
    assert receipt["candidate_policy_queried"] is False
    from blueprint_pipeline import native_task_arena_construction_worker as worker
    output = tmp_path / "native-output"
    closed = []
    class App:
        def close(self):
            retained = json.loads((output / worker.RESULT_FILENAME).read_text())
            assert retained["status"] == "completed"
            assert retained["construction_gate_qualified"] is False
            assert retained["native_feature_test_only"] is True
            assert retained["qualification_effect"] == "none"
            assert retained["result_digest"] == canonical_digest(retained, digest_field="result_digest")
            closed.append(True)
    def native_feature(**kwargs):
        return {"status": "completed", "blockers": [], "native_isaac_executed": True}, App()
    monkeypatch.setattr(worker, "__file__", str(tmp_path / "extracted/provider_runtime/adp_arena_provider_runner.py"))
    monkeypatch.setenv("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", str(output))
    monkeypatch.setattr(probe, "run_feature_probe", native_feature)
    assert worker.main([]) == 0
    assert closed == [True]
