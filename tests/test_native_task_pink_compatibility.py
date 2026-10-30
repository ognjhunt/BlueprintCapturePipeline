"""Real pinned metadata with development-only SDK payloads; no native proof."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import sys
import tomllib

import pytest

from blueprint_pipeline import native_task_pink_compatibility as pink
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_task_isaaclab_launch import _receipt


def blob(data, path):
    return {"path": path, "size_bytes": len(data),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    source = pink._manifest()
    sdk = tmp_path / "image"
    for name, row in source["extensions"].items():
        root = sdk / "exts" / name
        (root / "config").mkdir(parents=True)
        (root / "config/extension.toml").write_text(row["config"])
        relative = name.replace(".", "/") + "/__init__.py"
        payload = b"# development-only SDK substitute\n"
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)
        row["files"] = [blob(payload, relative)]
    urdf = b'<robot name="development-only"/>'
    source["franka_urdf"] = blob(urdf, "robot_configurations/franka/robot.urdf")
    path = sdk / "exts/isaacsim.robot_motion.pink" / source["franka_urdf"]["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(urdf)
    monkeypatch.setattr(pink, "_manifest", lambda: source)
    receipt = _receipt(tmp_path)
    value = json.loads(receipt.read_text())
    apps = Path(value["runtime_experience"]["path"]).parent
    fixtures = Path(__file__).parent / "fixtures"
    (apps / "isaaclab.python.headless.kit").write_bytes((fixtures / "isaaclab_headless_ffff603e.kit").read_bytes())
    (apps / "isaaclab.python.headless.rendering.kit").write_bytes((fixtures / "isaaclab_headless_rendering_ffff603e.kit").read_bytes())
    value["runtime_experience"]["sha256"] = pink._sha((apps / "isaaclab.python.headless.rendering.kit").read_bytes())
    target = Path(value["extraction_dir"]) / "runtime_python_dependencies"
    target.mkdir()
    value["runtime_dependency_target"] = str(target)
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    receipt.write_text(json.dumps(value))
    return SimpleNamespace(source=source, sdk=sdk, receipt=receipt, apps=apps)


@pytest.mark.parametrize("cameras", [False, True])
def test_only_four_warp_edges_change_and_original_sdk_bytes_are_unchanged(fixture, cameras):
    result = pink.prepare_pink_runtime(fixture.receipt, enable_cameras=cameras, sdk_root=fixture.sdk)
    changed = 0
    for row in result["extensions"]:
        original = Path(row["source_path"]) / "config/extension.toml"
        assert original.read_text() == fixture.source["extensions"][row["extension_id"]]["config"]
        before = tomllib.loads(original.read_text())
        after = tomllib.loads((Path(row["overlay_path"]) / "config/extension.toml").read_text())
        if "omni.warp.core" in before["dependencies"]:
            before["dependencies"][pink.PROVIDER] = before["dependencies"].pop("omni.warp.core")
            changed += 1
        assert before == after
        assert "--ext-path " + row["overlay_path"] in result["kit_args"]
        for record in fixture.source["extensions"][row["extension_id"]]["files"]:
            assert (Path(row["overlay_path"]) / record["path"]).read_bytes() == (Path(row["source_path"]) / record["path"]).read_bytes()
    assert changed == 4
    for experience in result["experiences"]:
        text = Path(experience["path"]).read_text()
        assert 'app.extensions.excluded = ["omni.isaac.ml_archive", "isaacsim.pip.newton", "omni.warp.core"]' in text
        assert "${app}" not in text
        assert pink._sha(Path(experience["source"]).read_bytes()) == experience["source_sha256"]


@pytest.mark.parametrize("mutation", ["config", "python", "robot", "experience", "config_crlf", "experience_crlf"])
def test_source_changes_are_not_accepted_as_pinned_sdk(fixture, mutation):
    root = fixture.sdk / "exts/isaacsim.robot_motion.pink"
    paths = {"config": root / "config/extension.toml", "python": root / "isaacsim/robot_motion/pink/__init__.py",
             "robot": root / "robot_configurations/franka/robot.urdf", "experience": fixture.apps / "isaaclab.python.headless.kit"}
    path = paths[mutation.removesuffix("_crlf")]
    if mutation.endswith("_crlf"):
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    else:
        path.write_text(path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError):
        pink.prepare_pink_runtime(fixture.receipt, enable_cameras=True, sdk_root=fixture.sdk)


def test_wrong_kit_path_precedence_cannot_claim_required_controller(fixture):
    pink.prepare_pink_runtime(fixture.receipt, enable_cameras=True, sdk_root=fixture.sdk)
    manager = SimpleNamespace(get_enabled_extension_id=lambda name: name,
                              get_extension_path=lambda name: str(fixture.sdk / "exts" / name))
    with pytest.raises(ValueError, match="pink_runtime_extension_precedence_mismatch"):
        pink.readback_pink_runtime(manager)


def test_shadowed_python_module_cannot_hide_behind_correct_extension_path(fixture, monkeypatch):
    result = pink.prepare_pink_runtime(fixture.receipt, enable_cameras=True, sdk_root=fixture.sdk)
    paths = {row["extension_id"]: row["overlay_path"] for row in result["extensions"]}
    manager = SimpleNamespace(get_enabled_extension_id=lambda name: name, get_extension_path=paths.__getitem__)
    monkeypatch.setattr(pink.importlib, "import_module", lambda name: SimpleNamespace(__file__="/unbound/module.py"))
    with pytest.raises(ValueError, match="pink_runtime_python_origin_changed"):
        pink.readback_pink_runtime(manager)


def test_readback_constructs_required_controller_with_exact_module_origins(fixture, monkeypatch):
    """Exercise orchestration with explicit SDK doubles, not a native solve."""
    result = pink.prepare_pink_runtime(fixture.receipt, enable_cameras=True, sdk_root=fixture.sdk)
    paths = {row["extension_id"]: row["overlay_path"] for row in result["extensions"]}
    manager = SimpleNamespace(get_enabled_extension_id=lambda name: name, get_extension_path=paths.__getitem__,
                              is_extension_enabled=lambda name: name == pink.PROVIDER)
    modules = {}
    for row in result["extensions"]:
        name = row["extension_id"]
        module = ModuleType(name)
        module.__file__ = str(Path(row["source_path"]) / name.replace(".", "/") / "__init__.py")
        modules[name] = module
    sdk_module = modules["isaacsim.robot_motion.pink"]
    controller_path = Path(sdk_module.__file__).parent / "impl/pink_ik_controller.py"
    code = "class PinkIKController:\n def __init__(self, **kwargs): self.arguments=kwargs\n def get_frame_task(self): return self.arguments['tool_frame']\n"
    exec(compile(code, str(controller_path), "exec"), sdk_module.__dict__)
    robot = SimpleNamespace(controlled_joint_names=[f"joint_{i}" for i in range(7)])
    requested = []
    sdk_module.load_pink_supported_robot = lambda name: requested.append(name) or robot
    # inspect.getfile on a class resolves the defining module; use a separate
    # source-bound class namespace, just as the real SDK's exported class does.
    controller_module = ModuleType("isaacsim.robot_motion.pink.impl.pink_ik_controller")
    controller_module.__file__ = str(controller_path)
    sdk_module.PinkIKController.__module__ = controller_module.__name__
    monkeypatch.setitem(sys.modules, controller_module.__name__, controller_module)
    for name in ("pink", "pinocchio", "qpsolvers", "osqp"):
        modules[name] = SimpleNamespace(__file__=str(Path(result["dependency_target"]) / name / "__init__.py"),
                                        __version__="development-only")
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(pink.importlib, "import_module", modules.__getitem__)
    proof = pink.readback_pink_runtime(manager)
    assert requested == ["franka"]
    assert proof["controller_constructed"] is True
    assert proof["solver_step_executed"] is proof["task_executed"] is False
    assert len(proof["extensions"]) == 6
