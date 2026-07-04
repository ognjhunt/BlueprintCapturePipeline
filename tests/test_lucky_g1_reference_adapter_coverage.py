from __future__ import annotations

import json
import sys
import types
import subprocess
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import lucky_g1_reference_adapter as lucky


pytestmark = pytest.mark.slow


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_lucky_root(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in lucky.REQUIRED_FILES:
        (root / name).write_text("{}" if name.endswith(".json") else "asset", encoding="utf-8")
    (root / "assets").mkdir(exist_ok=True)
    _write_json(
        root / "model_config.json",
        {"joint_names": ["hip", "knee"], "default_joint_pos": {"hip": 0.1, "missing": 9.0}},
    )
    (root / "run.py").write_text("VALUE = 42\n", encoding="utf-8")
    return root


def test_lucky_root_resolution_fetch_and_module_loading(monkeypatch, tmp_path: Path) -> None:
    assert lucky._string(None) == ""
    assert lucky._mapping({"a": 1}) == {"a": 1}
    assert lucky._mapping([]) == {}

    valid_root = _make_lucky_root(tmp_path / "valid")
    monkeypatch.setenv("BLUEPRINT_LUCKY_G1_CACHE_DIR", str(tmp_path / "cache"))

    def fake_subprocess_run(command, **kwargs):
        assert command == ["git", "status"]
        assert kwargs["cwd"] == str(valid_root)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0, stdout=" clean\n", stderr="")

    monkeypatch.setattr(lucky.subprocess, "run", fake_subprocess_run)
    assert lucky._run_git(["status"], cwd=valid_root) == "clean"

    monkeypatch.setattr(lucky, "_run_git", lambda args, cwd=None: "commit123")
    root, blockers, commit = lucky.resolve_lucky_g1_reference_root(lucky_root=valid_root)
    assert root == valid_root.resolve()
    assert blockers == []
    assert commit == "commit123"

    monkeypatch.setattr(lucky, "_run_git", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not git")))
    assert lucky.resolve_lucky_g1_reference_root(lucky_root=valid_root)[2] is None

    missing_root = tmp_path / "missing"
    root, blockers, commit = lucky.resolve_lucky_g1_reference_root(lucky_root=missing_root, fetch_if_missing=False)
    assert root is None
    assert blockers == [
        f"missing_lucky_g1_challenge_root_checked:{missing_root.resolve()}",
        f"missing_lucky_g1_challenge_root_checked:{(tmp_path / 'cache' / 'g1-manipulation-challenge').resolve()}",
    ]
    assert commit is None

    cache_root = tmp_path / "cache" / "g1-manipulation-challenge"
    cache_root.mkdir(parents=True)
    root, blockers, _ = lucky.resolve_lucky_g1_reference_root(fetch_if_missing=True)
    assert root is None
    assert blockers == [f"cache_path_exists_but_is_not_git_repo:{cache_root.resolve()}"]

    def fake_git(args, cwd=None):
        if args[0] == "clone":
            _make_lucky_root(Path(args[-1]))
            (Path(args[-1]) / ".git").mkdir()
            return ""
        if args[0] in {"fetch", "checkout"}:
            return ""
        if args[:2] == ["rev-parse", "HEAD"]:
            return "fetched-commit"
        raise AssertionError(args)

    cache_root.rmdir()
    monkeypatch.setattr(lucky, "_run_git", fake_git)
    root, blockers, commit = lucky.resolve_lucky_g1_reference_root(fetch_if_missing=True, repo_url="https://example/repo.git")
    assert root == cache_root.resolve()
    assert blockers == []
    assert commit == "fetched-commit"

    (cache_root / "walker.onnx").unlink()
    root, blockers, commit = lucky.resolve_lucky_g1_reference_root(fetch_if_missing=True)
    assert root is None
    assert "missing_after_fetch:walker.onnx" in blockers
    assert commit is None

    module = lucky._load_lucky_run_module(valid_root)
    assert module.VALUE == 42

    class RemovingLoader:
        def exec_module(self, module) -> None:
            sys.path.remove(str(valid_root))
            module.REMOVED_PATH_DURING_LOAD = True

    monkeypatch.setattr(
        lucky.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: types.SimpleNamespace(loader=RemovingLoader()),
    )
    monkeypatch.setattr(
        lucky.importlib.util,
        "module_from_spec",
        lambda _spec: types.SimpleNamespace(),
    )
    assert lucky._load_lucky_run_module(valid_root).REMOVED_PATH_DURING_LOAD is True

    monkeypatch.setattr(lucky.importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="unable_to_load_lucky_run_py"):
        lucky._load_lucky_run_module(valid_root)


def test_lucky_contact_records_and_trace_video(tmp_path: Path) -> None:
    class Contact:
        geom = [1, 2]
        dist = 0.1234567899

    data = types.SimpleNamespace(ncon=1, contact=[Contact()])
    model = object()
    fake_mujoco = types.SimpleNamespace(
        mjtObj=types.SimpleNamespace(mjOBJ_GEOM=1),
        mj_id2name=lambda _model, _kind, geom_id: {1: "right_hand_tip", 2: "red_cylinder"}.get(geom_id),
    )
    records = lucky._contact_records(model, data, fake_mujoco)
    assert records == [
        {
            "contact_index": 0,
            "geom_ids": [1, 2],
            "geom_names": ["right_hand_tip", "red_cylinder"],
            "distance": 0.12345679,
            "right_hand_object_contact": True,
        }
    ]

    empty_manifest = lucky._write_trace_video(out_dir=tmp_path / "empty", rows=[], generated_at="2026-06-20T00:00:00Z")
    assert empty_manifest["status"] == "blocked_no_frames"

    rows = [
        {"phase": "walk", "walker_policy_executed": True, "right_reacher_policy_executed": False},
        {
            "phase": "reach",
            "walker_policy_executed": True,
            "right_reacher_policy_executed": True,
            "red_block_pose_xyz": [0.1, 0.2, 0.0],
            "right_palm_pose_xyz": [0.2, 0.1, 0.0],
        },
    ]
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    manifest = lucky._write_trace_video(out_dir=video_dir, rows=rows, generated_at="2026-06-20T00:00:00Z")
    assert manifest["status"] == "complete"
    assert manifest["frame_count"] == 2
    assert Path(manifest["videos"][0]["path"]).is_file()

    blocked = lucky._blocked_manifest(
        out_dir=tmp_path / "blocked",
        generated_at="2026-06-20T00:00:00Z",
        blockers=["missing"],
        repo_url="https://example/repo.git",
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["missing"]
    assert Path(blocked["output_path"]).is_file()


def test_run_lucky_g1_reference_adapter_with_fake_runtime(monkeypatch, tmp_path: Path) -> None:
    lucky_root = _make_lucky_root(tmp_path / "lucky")
    monkeypatch.setattr(lucky, "resolve_lucky_g1_reference_root", lambda **_kwargs: (lucky_root, [], "commit"))

    class FakeModel:
        def __init__(self) -> None:
            self.opt = types.SimpleNamespace(timestep=0.0)
            self.nu = 12
            self.nq = 20

        @classmethod
        def from_xml_path(cls, _path):
            return cls()

    class FakeData:
        def __init__(self, _model) -> None:
            self.qpos = np.zeros(20, dtype=float)
            self.time = 0.0
            self.site_xpos = np.array([[0.2, -0.2, 0.4]], dtype=float)
            self.xpos = np.array([[0.3, -0.2, 0.05]], dtype=float)
            self.ncon = 1
            self.contact = [types.SimpleNamespace(geom=[1, 2], dist=0.01)]

    def fake_mj_step(_model, data):
        data.time += 0.005
        data.xpos[0, 2] += 0.0005

    fake_mujoco = types.SimpleNamespace(
        MjModel=FakeModel,
        MjData=FakeData,
        mjtObj=types.SimpleNamespace(mjOBJ_GEOM=1, mjOBJ_BODY=2, mjOBJ_SITE=3, mjOBJ_CAMERA=4),
        mj_forward=lambda _model, _data: None,
        mj_step=fake_mj_step,
        mj_name2id=lambda _model, _kind, name: {"red_block": 0, "right_palm": 0, "head_cam": 1, "wrist_cam": 2}.get(name, -1),
        mj_id2name=lambda _model, _kind, geom_id: {1: "right_wrist", 2: "red_cylinder"}.get(geom_id),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    class FakePolicy:
        def __init__(self, path: str) -> None:
            self.path = path

        def __call__(self, values):
            return np.ones(1, dtype=np.float32) * float(len(values))

    class FakeController:
        def __init__(self, _model, _data, _walker, _croucher, _rotator, _config, *, right_reacher) -> None:
            self.right_finger_actuators = [1, 2]
            self.right_arm_indices = [3, 4, 5]
            self.last_action = np.zeros(2, dtype=np.float32)
            self.last_arm_action = np.zeros(2, dtype=np.float32)
            self.reach_target = np.zeros(3, dtype=np.float32)
            self.grip_closed = False
            self.reach_active = False
            self.input_mode = "walk"
            self.lin_vel_x = 0.0
            self.lin_vel_y = 0.0
            self.ang_vel_z = 0.0

        def step(self):
            self.last_action = np.array([self.lin_vel_x, self.ang_vel_z], dtype=np.float32)
            self.last_arm_action = np.ones(2, dtype=np.float32) if self.reach_active else np.zeros(2, dtype=np.float32)
            return np.zeros(12, dtype=np.float32)

        def apply_pd_control(self, _target_pos) -> None:
            return None

    fake_module = types.SimpleNamespace(
        ONNXPolicy=FakePolicy,
        G1Controller=FakeController,
        set_armature=lambda _model, _joint_names: None,
    )
    monkeypatch.setattr(lucky, "_load_lucky_run_module", lambda _root: fake_module)

    result = lucky.run_lucky_g1_reference_adapter(
        capture_root=tmp_path / "capture",
        output_dir=tmp_path / "out",
        lucky_root=lucky_root,
    )

    assert result["status"] == "complete"
    assert result["repo_commit"] == "commit"
    assert result["official_lucky_walker_reacher_policy_assets_executed"] is True
    assert result["model"]["right_arm_joint_count"] == 3
    assert result["model"]["head_camera_available"] is True
    assert result["policies"]["walker"]["executed"] is True
    assert result["policies"]["right_reacher"]["executed"] is True
    assert result["metrics"]["trace_sample_count"] > 0
    assert result["metrics"]["hand_object_contact_count"] > 0
    assert Path(result["artifacts"]["lucky_g1_reference_trace"]).is_file()
    assert Path(result["output_path"]).is_file()


def test_lucky_g1_reference_adapter_blocked_and_cli(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(lucky, "resolve_lucky_g1_reference_root", lambda **_kwargs: (None, ["missing_root"], None))
    blocked = lucky.run_lucky_g1_reference_adapter(capture_root=tmp_path / "capture", output_dir=tmp_path / "out")
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["missing_root"]

    monkeypatch.setattr(
        lucky,
        "run_lucky_g1_reference_adapter",
        lambda **_kwargs: {"status": "complete", "output_path": str(tmp_path / "manifest.json")},
    )
    assert lucky.main(["--capture-root", str(tmp_path / "capture"), "--output-dir", str(tmp_path / "out"), "--steps", "5"]) == 0
    output = capsys.readouterr().out
    assert str(tmp_path / "manifest.json") in output
    assert "complete" in output

    monkeypatch.setattr(
        lucky,
        "run_lucky_g1_reference_adapter",
        lambda **_kwargs: {"status": "blocked", "output_path": str(tmp_path / "manifest.json")},
    )
    assert lucky.main(["--capture-root", str(tmp_path / "capture")]) == 1
