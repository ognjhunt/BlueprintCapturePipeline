import hashlib
from pathlib import Path

import pytest
import torch

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline.native_g1_humanoidarena_interface import CANONICAL_BODY_JOINT_NAMES_29
from blueprint_pipeline import native_g1_official_sonic_target_bridge as bridge_module


class _Session:
    def run(self, *_args, **_kwargs):
        return [0]


class _Beta2ProxyArray:
    """Isaac Lab's Warp shape hides vec/state columns until .torch is read."""

    def __init__(self, tensor):
        self.shape = (tensor.shape[0],)
        self.torch = tensor


class _Scene:
    def __init__(self):
        self.robot = type("Robot", (), {})()
        self.robot.data = type("Data", (), {})()
        self.robot.data.root_state_w = torch.tensor(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]], dtype=torch.float32
        )
        self.robot.data.root_pos_w = torch.tensor([[1.0, 2.0, 3.0]])
        self.robot.data.root_quat_w = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
        self.robot.data.root_lin_vel_w = torch.tensor([[4.0, 5.0, 6.0]])
        self.robot.data.root_ang_vel_w = torch.tensor([[7.0, 8.0, 9.0]])

    def __getitem__(self, key):
        assert key == "robot"
        return self.robot


class _Environment:
    def __init__(self):
        self.scene = _Scene()
        self.step_count = 0

    def step(self, *_args):
        self.step_count += 1


class SonicActionProvider:
    def __init__(self, *, encoder_path: Path, decoder_path: Path):
        self.env = bridge_module.SonicWxyzEnvironmentView(_Environment())
        self._sonic_joint29_mode = True
        self._use_vla_latent64 = False
        self._dex3_dds = None
        self._encoder = _Session()
        self._decoder = _Session()
        self.encoder_path = str(encoder_path)
        self.decoder_path = str(decoder_path)
        self._smpl_data_valid = False
        self._latest_consumed_new_this_step = False
        self._left_hand_target = [0.1] * 7
        self._right_hand_target = [0.2] * 7
        self._perf_encoder_ms = []
        self._perf_decoder_ms = []
        self.skip_decoder = False
        self.body_target = 0.0
        self.consume_reference = True
        self.reference_history_valid = True

    def _apply_lerobot_semantic_action(self, _action):
        # The pinned joint29 provider sets validity in _run_gear_sonic after
        # inspecting its history, rather than during action application.
        self._latest_consumed_new_this_step = self.consume_reference

    def _run_gear_sonic(self):
        if self.reference_history_valid:
            self._smpl_data_valid = True
        if not self._smpl_data_valid:
            return [0.0] * 29
        self._encoder.run(None, {})
        if not self.skip_decoder:
            self._decoder.run(None, {})
            self._latest_decoder_raw_action = [self.body_target] * 29
            self._latest_decoder_target = [self.body_target] * 29
            self._perf_encoder_ms.append(1.0)
            self._perf_decoder_ms.append(1.0)
        return [self.body_target] * 29


def _action():
    return [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 29 + [0.0, 1.0]


def _adapter(tmp_path: Path, monkeypatch, provider):
    source = Path(__file__)
    monkeypatch.setattr(
        bridge_module,
        "PINNED_ACTION_PROVIDER_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    return bridge_module.NativeG1OfficialSonicTargetBridge(
        provider=provider,
        source_path=source,
        encoder_sha256="sha256:"
        + hashlib.sha256(Path(provider.encoder_path).read_bytes()).hexdigest(),
        decoder_sha256="sha256:"
        + hashlib.sha256(Path(provider.decoder_path).read_bytes()).hexdigest(),
        joint_limits={name: [-1.0, 1.0] for name in PROTOCOL_V4_FULL_JOINT_ORDER},
    )


def _provider(tmp_path: Path):
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    encoder.write_bytes(b"encoder fixture")
    decoder.write_bytes(b"decoder fixture")
    return SonicActionProvider(encoder_path=encoder, decoder_path=decoder)


def test_native_xyzw_is_shown_as_wxyz_without_mutation():
    native = _Environment()
    view = bridge_module.SonicWxyzEnvironmentView(native)
    assert view.scene["robot"].data.root_state_w[0, 3:7].tolist() == pytest.approx(
        [0.9, 0.1, 0.2, 0.3]
    )
    assert native.scene["robot"].data.root_state_w[0, 3:7].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.9]
    )


def test_beta2_proxy_root_state_uses_expanded_torch_view_without_mutation():
    native = _Environment()
    data = native.scene.robot.data
    measured = torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9, 4, 5, 6, 7, 8, 9]])
    data.root_state_w = _Beta2ProxyArray(measured)
    converted = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w
    assert converted.shape == (1, 13)
    assert converted[0, :7].tolist() == pytest.approx([1, 2, 3, 0.9, 0.1, 0.2, 0.3])
    assert measured[0, 3:7].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.9])


def test_beta2_proxy_root_refuses_nonfinite_measured_state():
    native = _Environment()
    native.scene.robot.data.root_state_w = _Beta2ProxyArray(
        torch.tensor([[1.0, 2.0, 3.0, 0.1, 0.2, float("nan"), 0.9]])
    )
    with pytest.raises(ValueError, match="nonfinite_combined_state"):
        _ = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w


def test_beta2_proxy_measured_pose_fallback_uses_expanded_torch_fields():
    native = _Environment()
    data = native.scene.robot.data
    data.root_state_w = _Beta2ProxyArray(torch.empty((1, 0)))
    data.root_pos_w = _Beta2ProxyArray(torch.tensor([[1.0, 2.0, 3.0]]))
    data.root_quat_w = _Beta2ProxyArray(torch.tensor([[0.1, 0.2, 0.3, 0.9]]))
    data.root_lin_vel_w = _Beta2ProxyArray(torch.tensor([[4.0, 5.0, 6.0]]))
    data.root_ang_vel_w = _Beta2ProxyArray(torch.tensor([[7.0, 8.0, 9.0]]))
    root = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w
    assert root[0].tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )


def test_incomplete_combined_root_uses_measured_components_without_fabrication():
    native = _Environment()
    native.scene.robot.data.root_state_w = torch.empty((1, 0))
    root = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w
    assert root.shape == (1, 13)
    assert root[0].tolist() == pytest.approx(
        [1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )
    native.scene.robot.data.root_ang_vel_w = torch.empty((1, 0))
    pose = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w
    assert pose[0].tolist() == pytest.approx([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])


def test_incomplete_combined_root_uses_measured_pose_when_velocity_is_unavailable():
    native = _Environment()
    native.scene.robot.data.root_state_w = torch.empty((1, 0))
    native.scene.robot.data.root_lin_vel_w = torch.empty((1, 0))
    native.scene.robot.data.root_ang_vel_w = torch.empty((1, 0))
    root = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w
    assert root.shape == (1, 7)
    assert root[0].tolist() == pytest.approx([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3])
    assert native.scene.robot.data.root_state_w.shape == (1, 0)


def test_incomplete_combined_root_rejects_unmeasured_pose():
    native = _Environment()
    native.scene.robot.data.root_state_w = torch.empty((1, 0))
    native.scene.robot.data.root_quat_w = torch.empty((1, 0))
    with pytest.raises(
        ValueError, match=r"g1_sonic_native_root_state_invalid:.*quaternion_shape=\(1, 0\)"
    ):
        _ = bridge_module.SonicWxyzEnvironmentView(native).scene["robot"].data.root_state_w


def test_exact_semantic_action_runs_both_onnx_sessions_and_maps_names(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, _provider(tmp_path))
    targets = adapter.targets_for_action(_action())
    assert set(targets) == set(PROTOCOL_V4_FULL_JOINT_ORDER)
    assert targets["left_hand_middle_0_joint"] == pytest.approx(0.1)
    assert targets["right_hand_index_0_joint"] == pytest.approx(0.2)
    assert (adapter.encoder.calls, adapter.decoder.calls) == (1, 1)
    assert adapter.provider.env._native_environment.step_count == 0


def test_new_joint29_reference_is_validated_by_sonic_during_inference(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    assert provider._smpl_data_valid is False
    adapter = _adapter(tmp_path, monkeypatch, provider)
    adapter.targets_for_action(_action())
    assert provider._smpl_data_valid is True
    assert (adapter.encoder.calls, adapter.decoder.calls) == (1, 1)


def test_missing_new_reference_refuses_inference(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.consume_reference = False
    adapter = _adapter(tmp_path, monkeypatch, provider)
    with pytest.raises(RuntimeError, match="reference_not_consumed"):
        adapter.targets_for_action(_action())
    assert (adapter.encoder.calls, adapter.decoder.calls) == (0, 0)


def test_invalid_joint29_history_cannot_be_counted_as_inference(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.reference_history_valid = False
    adapter = _adapter(tmp_path, monkeypatch, provider)
    with pytest.raises(RuntimeError, match="reference_not_consumed"):
        adapter.targets_for_action(_action())
    assert (adapter.encoder.calls, adapter.decoder.calls) == (0, 0)


def test_upstream_default_pose_fallback_cannot_count_as_controller_inference(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.skip_decoder = True
    adapter = _adapter(tmp_path, monkeypatch, provider)
    with pytest.raises(RuntimeError, match="inference_unmeasured"):
        adapter.targets_for_action(_action())


def test_finite_out_of_limit_target_is_projected_and_retained(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.body_target = 2.0
    adapter = _adapter(tmp_path, monkeypatch, provider)
    targets = adapter.targets_for_action(_action())
    assert all(targets[name] == 1.0 for name in CANONICAL_BODY_JOINT_NAMES_29)
    assert provider._latest_decoder_target == [2.0] * 29
    assert adapter.last_target_projection[0] == {
        "joint_name": CANONICAL_BODY_JOINT_NAMES_29[0],
        "requested_target_rad": 2.0,
        "applied_target_rad": 1.0,
        "lower_rad": -1.0,
        "upper_rad": 1.0,
        "excess_rad": 1.0,
        "raw_decoder_action": 2.0,
    }
    assert len(adapter.last_target_projection) == 29
    provider.body_target = 0.0
    adapter.targets_for_action(_action())
    assert adapter.last_target_projection == []


def test_nonfinite_target_still_fails_closed(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider._left_hand_target[0] = float("nan")
    adapter = _adapter(tmp_path, monkeypatch, provider)
    with pytest.raises(bridge_module.G1SonicTargetLimitError, match="target_out_of_limits"):
        adapter.targets_for_action(_action())
    assert adapter.last_target_projection == []


def test_inverted_sealed_limit_still_fails_closed(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, _provider(tmp_path))
    adapter.limits[CANONICAL_BODY_JOINT_NAMES_29[0]] = [1.0, -1.0]
    with pytest.raises(bridge_module.G1SonicTargetLimitError, match="target_out_of_limits"):
        adapter.targets_for_action(_action())
    assert adapter.last_target_projection == []


def test_unpinned_source_is_rejected(tmp_path):
    source = tmp_path / "action_provider_sonic.py"
    source.write_bytes(b"untrusted")
    with pytest.raises(ValueError, match="revision_mismatch"):
        bridge_module.require_pinned_sonic_source(source)


def test_controller_model_bytes_must_match_supplied_digest(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    source = Path(__file__)
    monkeypatch.setattr(
        bridge_module,
        "PINNED_ACTION_PROVIDER_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="model_artifact_identity_mismatch"):
        bridge_module.NativeG1OfficialSonicTargetBridge(
            provider=provider,
            source_path=source,
            encoder_sha256="sha256:" + "0" * 64,
            decoder_sha256="sha256:" + "0" * 64,
            joint_limits={name: [-1.0, 1.0] for name in PROTOCOL_V4_FULL_JOINT_ORDER},
        )
