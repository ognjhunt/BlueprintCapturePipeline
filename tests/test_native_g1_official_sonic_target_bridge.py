import hashlib
from pathlib import Path

import pytest
import torch

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline import native_g1_official_sonic_target_bridge as bridge_module


class _Session:
    def run(self, *_args, **_kwargs):
        return [0]


class _Scene:
    def __init__(self):
        self.robot = type("Robot", (), {})()
        self.robot.data = type("Data", (), {})()
        self.robot.data.root_state_w = torch.tensor(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]], dtype=torch.float32
        )

    def __getitem__(self, key):
        assert key == "robot"
        return self.robot


class _Environment:
    def __init__(self):
        self.scene = _Scene()


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

    def _apply_lerobot_semantic_action(self, _action):
        self._smpl_data_valid = True
        self._latest_consumed_new_this_step = True

    def _run_gear_sonic(self):
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
        encoder_sha256="sha256:" + hashlib.sha256(Path(provider.encoder_path).read_bytes()).hexdigest(),
        decoder_sha256="sha256:" + hashlib.sha256(Path(provider.decoder_path).read_bytes()).hexdigest(),
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


def test_exact_semantic_action_runs_both_onnx_sessions_and_maps_names(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, monkeypatch, _provider(tmp_path))
    targets = adapter.targets_for_action(_action())
    assert set(targets) == set(PROTOCOL_V4_FULL_JOINT_ORDER)
    assert targets["left_hand_middle_0_joint"] == pytest.approx(0.1)
    assert targets["right_hand_index_0_joint"] == pytest.approx(0.2)
    assert (adapter.encoder.calls, adapter.decoder.calls) == (1, 1)


def test_upstream_default_pose_fallback_cannot_count_as_controller_inference(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.skip_decoder = True
    adapter = _adapter(tmp_path, monkeypatch, provider)
    with pytest.raises(RuntimeError, match="inference_unmeasured"):
        adapter.targets_for_action(_action())


def test_out_of_limit_target_is_rejected_not_clipped(tmp_path, monkeypatch):
    provider = _provider(tmp_path)
    provider.body_target = 2.0
    adapter = _adapter(tmp_path, monkeypatch, provider)
    with pytest.raises(ValueError, match="target_out_of_limits"):
        adapter.targets_for_action(_action())


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
