from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blueprint_pipeline import native_g1_runtime_assembly as assembly


SCENE_DIGEST = "sha256:" + "a" * 64
CANDIDATE = "humanoidarena_dp_g1_dex3_sonic"


def _built():
    return SimpleNamespace(
        plan={
            "plan_digest": SCENE_DIGEST,
            "robot": {"robot_id": "unitree_g1", "joint_position_limits_rad": {}},
        },
        env=SimpleNamespace(scene={}, device="cuda:0"),
    )


def _inputs():
    return {
        "candidate_id": CANDIDATE,
        "sonic_provider_source": "/staged/action_provider/action_provider_sonic.py",
        "sonic_encoder": "/staged/encoder.onnx",
        "sonic_encoder_sha256": "sha256:" + "b" * 64,
        "sonic_decoder": "/staged/decoder.onnx",
        "sonic_decoder_sha256": "sha256:" + "c" * 64,
    }


def test_sonic_bridge_primes_real_root_before_provider_construction() -> None:
    data = SimpleNamespace(
        root_state_w=torch.empty((1, 0)),
        root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
        root_quat_w=torch.tensor([[0.1, 0.2, 0.3, 0.9]]),
        root_lin_vel_w=torch.zeros((1, 3)),
        root_ang_vel_w=torch.zeros((1, 3)),
    )
    robot = SimpleNamespace(data=data)
    calls: list[int] = []

    class _Env:
        scene = {"robot": robot}

        def reset(self, *, seed: int):
            calls.append(seed)
            return None

    env = _Env()
    built = SimpleNamespace(env=env)
    view = assembly._prime_sonic_native_root(built, env, {"scenario": {"seed": 7}})
    assert calls == [7]
    assert view.scene["robot"].data.root_state_w.shape == (1, 13)


def test_sonic_bridge_primes_beta2_proxy_root() -> None:
    class _Proxy:
        def __init__(self, tensor):
            self.shape = (tensor.shape[0],)
            self.torch = tensor

    data = SimpleNamespace(
        root_state_w=_Proxy(torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]])),
    )
    robot = SimpleNamespace(data=data)

    class _Env:
        scene = {"robot": robot}

        def reset(self, *, seed: int):
            assert seed == 7

    env = _Env()
    view = assembly._prime_sonic_native_root(
        SimpleNamespace(env=env), env, {"scenario": {"seed": 7}}
    )
    assert view.scene["robot"].data.root_state_w[0].tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3]
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:bad",
        "http://127.0.0.1:99999",
        "http://0.0.0.0:8443",
        "https://127.0.0.1:8443",
    ],
)
def test_controller_rejects_invalid_server_address_before_import(url: str) -> None:
    with pytest.raises(ValueError, match="g1_sonic_runtime_configuration_invalid"):
        assembly.build_pinned_g1_sonic_bridge(
            built=_built(),
            source_path=Path("/missing"),
            encoder_path=Path("/missing"),
            encoder_sha256="x",
            decoder_path=Path("/missing"),
            decoder_sha256="x",
            server_url=url,
        )


class _Lease:
    def __init__(self):
        self.client = SimpleNamespace(base_url="http://127.0.0.1:8443")
        self.receipt = {"candidate_id": CANDIDATE, "scene_plan_digest": SCENE_DIGEST}
        self.closed = 0

    def close(self):
        self.closed += 1
        return {"status": "child_exited", "pid": 123}


def test_supervised_episode_binds_scene_and_retains_teardown(tmp_path: Path, monkeypatch) -> None:
    lease = _Lease()
    monkeypatch.setattr(assembly, "start_g1_policy_server", lambda **kwargs: lease)
    monkeypatch.setattr(assembly, "build_pinned_g1_sonic_bridge", lambda **kwargs: object())
    monkeypatch.setattr(
        assembly,
        "run_g1_built_scene_policy_episode",
        lambda **kwargs: {"result_digest": "sha256:" + "d" * 64},
    )
    output_dir = tmp_path / "attempt"
    result = assembly.run_g1_supervised_built_scene_episode(
        built=_built(),
        candidate_id=CANDIDATE,
        preflight_inputs=_inputs(),
        python_executable=Path("/python"),
        port=8443,
        device="cuda:0",
        max_steps=1,
        output_dir=output_dir,
        to_tensor=lambda value: value,
        make_action_tensor=lambda value: value,
    )
    assert result["status"] == "completed_development_only"
    assert result["episode_result_digest"] == "sha256:" + "d" * 64
    assert result["ranking_eligible"] is False
    assert lease.closed == 1
    saved = json.loads(
        (output_dir / "native_g1_supervised_built_scene_episode.v1.json").read_text()
    )
    assert saved == result
    with pytest.raises(ValueError, match="scene_or_candidate_invalid"):
        assembly.run_g1_supervised_built_scene_episode(
            built=_built(),
            candidate_id=CANDIDATE,
            preflight_inputs=_inputs(),
            python_executable=Path("/python"),
            port=8443,
            device="cuda:0",
            max_steps=1,
            output_dir=output_dir,
            to_tensor=lambda value: value,
            make_action_tensor=lambda value: value,
        )


def test_supervised_episode_failure_still_closes_child_and_records_blocker(
    tmp_path: Path, monkeypatch
) -> None:
    lease = _Lease()
    monkeypatch.setattr(assembly, "start_g1_policy_server", lambda **kwargs: lease)
    monkeypatch.setattr(assembly, "build_pinned_g1_sonic_bridge", lambda **kwargs: object())

    def fail(**kwargs):
        raise RuntimeError("inference_failed")

    monkeypatch.setattr(assembly, "run_g1_built_scene_policy_episode", fail)
    output_dir = tmp_path / "failed-attempt"
    with pytest.raises(RuntimeError, match="inference_failed"):
        assembly.run_g1_supervised_built_scene_episode(
            built=_built(),
            candidate_id=CANDIDATE,
            preflight_inputs=_inputs(),
            python_executable=Path("/python"),
            port=8443,
            device="cuda:0",
            max_steps=1,
            output_dir=output_dir,
            to_tensor=lambda value: value,
            make_action_tensor=lambda value: value,
        )
    assert lease.closed == 1
    saved = json.loads(
        (output_dir / "native_g1_supervised_built_scene_episode.v1.json").read_text()
    )
    assert saved["status"] == "blocked"
    assert saved["blocker"] == "RuntimeError"
    assert saved["server_teardown"]["status"] == "child_exited"
