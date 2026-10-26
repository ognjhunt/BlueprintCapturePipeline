"""Run the unchanged pinned RoboLab loop; fake only simulator/library boundaries."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import beta_robolab_runtime as runtime
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_episode_events import NativeStep, PolicyQuery
from tests.test_adp009d_policy_episode import _Environment, _Policy, _run
from tests.test_native_task_arena_policy_canary_bundle import _spec


@pytest.fixture
def stock_robolab(tmp_path, monkeypatch):
    runtime.stage_robolab_source(tmp_path)
    old = {name: module for name, module in sys.modules.items() if name == "robolab" or name.startswith("robolab.")}
    for name in old:
        del sys.modules[name]
    root = ModuleType("robolab")
    root.__file__ = str(tmp_path / "robolab/__init__.py")
    root.__path__ = [str(tmp_path / "robolab")]
    monkeypatch.setitem(sys.modules, "robolab", root)

    def module(name, **members):
        value = ModuleType(name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    # The real upstream evaluation and client modules are imported from the
    # source archive. These stand in only for unavailable Isaac/PyTorch I/O.
    module("torch", zeros=lambda *shape, device=None: np.zeros(shape, dtype=np.float32),
           tensor=lambda value, device=None: np.asarray(value, dtype=np.float32))
    module("robolab.core.logging.results", get_all_env_subtask_infos=lambda env: None)
    module("robolab.core.observations.observation_utils", unpack_image_obs=lambda *a, **k: {},
           unpack_viewport_cams=lambda *a, **k: {})
    module("robolab.core.utils.video_utils", VideoWriter=object)
    module("robolab.core.world.world_state", get_world=lambda env: None)
    omni = module("omni")
    omni.kit = module("omni.kit")
    omni.kit.app = module("omni.kit.app", get_app=lambda: SimpleNamespace(update=lambda: None))
    omni.timeline = module("omni.timeline", get_timeline_interface=lambda: SimpleNamespace(is_playing=lambda: True))
    yield tmp_path
    for name in list(sys.modules):
        if name == "robolab" or name.startswith("robolab."):
            sys.modules.pop(name, None)
    sys.modules.update(old)


def test_stock_episode_drives_native_steps_and_preserves_scoring(stock_robolab):
    native_env, robolab_env = _Environment(), _Environment()
    native_policy, robolab_policy = _Policy(), _Policy()
    legacy = _run(native_env, native_policy)
    result = _run(robolab_env, robolab_policy, evaluation_backend="robolab.v1")
    assert len(robolab_env.steps) == len(native_env.steps) == 38
    assert robolab_env.steps == native_env.steps
    assert result["score"] == legacy["score"]
    assert result["commanded_actions"] == legacy["commanded_actions"]
    assert result["task_object_trajectory"] == legacy["task_object_trajectory"]
    assert len(robolab_policy.observations) == 4
    assert all("gt_state" not in observation for observation in robolab_policy.observations)
    assert result["evaluation_backend"]["native_action_steps"] == 38
    assert result["evaluation_backend"]["results"][0]["success"] is True
    assert result["evaluation_backend"]["episode_loop"] == "robolab.eval.run_episode"
    assert result["receipt_digest"] == canonical_digest(result, digest_field="receipt_digest")
    assert Path(importlib.import_module("robolab.eval.episode").__file__).is_relative_to(stock_robolab)


def test_inference_failure_returns_through_native_evidence_handler(stock_robolab):
    observed = []
    def events():
        try:
            yield PolicyQuery({"pixels": "retained"})
        except RuntimeError as error:
            observed.append(str(error))
            raise
    policy = SimpleNamespace(infer=lambda observation: (_ for _ in ()).throw(RuntimeError("provider_failed")))
    with pytest.raises(RuntimeError, match="provider_failed"):
        runtime.run_robolab_episode(events(), policy=policy, prompt="Actual job", maximum_steps=5,
                                    control_hz=15, action_dimension=1)
    assert observed == ["provider_failed"]


def test_native_step_mismatch_is_refused_before_actuation():
    applied = []
    def events():
        yield NativeStep([1.0, 2.0])
        applied.append(True)
        return {"score": {"task_succeeded": True}}
    environment = runtime.NativeEpisodeEnvironment(events(), policy=None, maximum_steps=1, action_dimension=2)
    environment.reset()
    environment.reset()
    environment.command()
    with pytest.raises(ValueError, match="action_changed"):
        environment.step(np.array([[9.0, 2.0]]))
    assert applied == []
    with pytest.raises(ValueError, match="reset_after_start"):
        environment.reset()


def test_source_tampering_refuses_before_inference(stock_robolab):
    source = stock_robolab / "robolab/eval/episode.py"
    source.write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="loaded_source_digest"):
        runtime.verify_loaded_robolab()


def test_sealed_spec_cannot_change_task_or_interface():
    spec = _spec("pi05_droid")
    spec["evaluation_backend"] = "robolab.v1"
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    assert runtime.validate_robolab_policy_execution_spec(spec)["candidate_id"] == "pi05_droid"
    for changed in ({**spec, "candidate_id": "other"}, {**spec, "evaluation_backend": "custom"}):
        with pytest.raises(ValueError):
            runtime.validate_robolab_policy_execution_spec(changed)
    spec["task_success_contract"]["criteria"]["untrusted_success"] = True
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    with pytest.raises(ValueError):
        runtime.validate_robolab_policy_execution_spec(spec)


def test_source_archive_contains_unmodified_entrypoints_and_license(tmp_path):
    receipt = runtime.stage_robolab_source(tmp_path)
    assert receipt["source_commit"] == runtime.ROBOLAB_COMMIT
    assert (tmp_path / "robolab/eval/episode.py").is_file()
    assert (tmp_path / "robolab/eval/base_client.py").is_file()
    assert "Apache License" in (tmp_path / "robolab-LICENSE").read_text()
    assert json.loads((tmp_path / "robolab-source.v1.json").read_text())["commit"] == runtime.ROBOLAB_COMMIT


def test_stock_loop_keeps_custom_articulated_job_and_single_reset(stock_robolab):
    from tests.test_adp009d_policy_episode import _ArticulatedEnvironment, _articulated_task_spec

    environment = _ArticulatedEnvironment()
    result = _run(environment, evaluation_backend="robolab.v1", task_spec=_articulated_task_spec(),
                  destination_position_world_m=None,
                  prompt="Open the upper refrigerator door, release it, and retreat.")
    assert result["score"]["outcome"] == "opened_and_settled"
    assert result["score"]["judgement_source"] == "deterministic_native_simulator_joint_state"
    assert environment.reset_count == 1
    assert result["evaluation_backend"]["results"][0]["success"] is True


def test_stock_loop_drives_declared_team_robot_and_its_control_cadence(stock_robolab, tmp_path, monkeypatch):
    from blueprint_pipeline import adp_task_scoring, native_task_episode_environment
    from blueprint_pipeline.team_embodiment_runtime import run_team_episode
    from tests.test_team_embodiment import _Policy as TeamPolicy, _Scoring, _team_loop

    built, plan, _spec = _team_loop(tmp_path, monkeypatch)
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    samples = []
    monkeypatch.setattr(adp_task_scoring, "score_task_episode_from_spec",
                        lambda **kwargs: samples.extend(kwargs["samples"]) or
                        {"status": "scored", "task_succeeded": False})
    plan["cadence"].update(maximum_action_steps=8, settle_window_samples=2)
    policy = TeamPolicy([[[0.1, 0.2, 0.3, 1.0]] * 4])
    result = run_team_episode(built=built, task_readback=None, policy=policy, plan=plan,
                              prompt="Put the mug away", max_policy_queries=2, open_loop_horizon=4,
                              media_root=tmp_path / "media", episode_id="team-scene", progress={},
                              to_action=lambda targets: [targets], evaluation_backend="robolab.v1")
    assert len(built.env.steps) == result["evaluation_backend"]["native_action_steps"] == 10
    assert built.env.steps[0] == [0.1, 0.2, 0.3, 0.0]
    assert len(policy.observations) == 2
    assert result["evaluation_backend"]["results"][0]["success"] is False
    assert samples[-1]["simulation_time_s"] == pytest.approx(10 / 30)


def test_remote_configuration_uses_explicit_supported_interface_and_private_file():
    spec = _spec("customer_policy_v7")
    spec.update(evaluation_backend="robolab.v1", policy_interface_id="openpi_droid_https.v1",
                policy_endpoint={"credential_env": "BLUEPRINT_POLICY_TOKEN_FILE"})
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    assert runtime.validate_robolab_policy_execution_spec(spec)["candidate_id"] == "customer_policy_v7"
    for endpoint in ({"credential_env": "invalid;command_FILE"},
                     {"credential_env": "BLUEPRINT_POLICY_TOKEN_FILE", "token": "forbidden-in-bundle"}):
        refused = {**spec, "policy_endpoint": endpoint}
        refused["execution_spec_digest"] = canonical_digest(refused, digest_field="execution_spec_digest")
        with pytest.raises(ValueError, match="credential_binding"):
            runtime.validate_robolab_policy_execution_spec(refused)


def test_remote_entrypoint_requires_private_regular_file_without_reading_secret(tmp_path):
    import subprocess
    from blueprint_pipeline.native_task_arena_policy_canary_bundle import _entrypoint

    script = _entrypoint(local_candidates=(), run_kind="blueprint_beta_screening",
                         credential_env="BLUEPRINT_POLICY_TOKEN_FILE")
    assert "adp009d_policy_provisioning.pi05_droid.sh" not in script
    assert "adp009d_policy_provisioning.groot_n17_droid.sh" not in script
    check = script.split("<<'PYCREDENTIAL'\n", 1)[1].split("\nPYCREDENTIAL", 1)[0]
    path = tmp_path / "credential"
    path.write_text("test-token-never-output")
    path.chmod(0o600)
    def result(candidate_path):
        return subprocess.run([sys.executable, "-", "BLUEPRINT_POLICY_TOKEN_FILE"], input=check,
                              text=True, capture_output=True,
                              env={"BLUEPRINT_POLICY_TOKEN_FILE": str(candidate_path)}, check=False)
    valid = result(path)
    assert valid.returncode == 0 and not valid.stdout and not valid.stderr
    path.chmod(0o644)
    assert result(path).returncode == 86
    path.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    assert result(alias).returncode == 86
    assert result(tmp_path / "missing").returncode == 86
