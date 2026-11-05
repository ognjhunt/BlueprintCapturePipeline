"""Our Isaac Lab and Arena calls exist in the sources production installs.

The first GPU run of the rigid robot-free probe (2026-11-03) crashed on
``set_use_tiled_camera``, a method only a test fake had. This checks every such
call against an index of the pinned Isaac Lab and Arena sources, without Kit.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import isaac_api_contract as contract
from blueprint_pipeline import native_task_runtime_source_packet as packet


ROOT = Path(__file__).resolve().parents[1]
REBUILD = "python -m blueprint_pipeline.isaac_api_contract index --fetch-into <empty dir>"


@pytest.fixture(scope="module")
def index() -> dict:
    return contract.load_index(ROOT)


@pytest.fixture(scope="module")
def report(index):
    return contract.check_repository(ROOT, index)


def test_the_index_is_read_from_the_exact_sources_production_installs(index) -> None:
    assert index["schema_version"] == contract.INDEX_SCHEMA_VERSION
    assert index["sources"] == {
        "isaaclab": {"repository": packet.ISAACLAB_REPOSITORY, "commit": packet.ISAACLAB_COMMIT,
                     "tree": packet.ISAACLAB_TREE},
        "arena": {"repository": packet.ARENA_REPOSITORY, "commit": packet.ARENA_COMMIT,
                  "tree": packet.ARENA_TREE},
    }, f"the runtime source packet moved its pins; rebuild the index: {REBUILD}"
    assert set(index["packages"]) == set(contract.PINNED_PACKAGES)
    assert {"isaaclab_arena.utils.cameras.ArenaCameraCfg",
            "isaaclab_arena.embodiments.droid.droid.DroidCameraCfg"} <= set(index["classes"])


def test_every_isaac_call_we_make_exists_in_the_pinned_sources(report) -> None:
    """The ratchet: a new finding fails; a fixed one must leave the known list."""
    known = contract.load_known_findings(ROOT)
    found = {finding.key: finding for finding in report.findings}
    new = [found[key] for key in sorted(found.keys() - known.keys())]
    assert not new, "calls the pinned Isaac Lab/Arena sources do not have:\n" + "\n".join(
        f"  {row.path}:{row.line} {row.kind} {row.detail}" for row in new)
    fixed = sorted(known.keys() - found.keys())
    assert not fixed, (
        f"fixed, so delete from {contract.KNOWN_FINDINGS_PATH}:\n" + "\n".join(f"  {key}" for key in fixed))
    assert not report.unindexed, (
        f"our code now reaches pinned names the committed index left out; rebuild it: {REBUILD}\n"
        + "\n".join(f"  {name}" for name in sorted(report.unindexed)))


def test_the_scan_still_reaches_the_arena_runtime(report) -> None:
    """A scan that silently stops typing objects would pass everything."""
    assert report.unused_bindings == []
    assert {
        "src/blueprint_pipeline/native_task_arena_runtime.py",
        "src/blueprint_pipeline/native_task_robot_registry.py",
        "src/blueprint_pipeline/native_g1_embodiment.py",
        "src/blueprint_pipeline/team_embodiment_runtime.py",
    } <= set(report.files)
    assert report.checked >= 550


def test_known_findings_are_explained() -> None:
    payload = json.loads((ROOT / contract.KNOWN_FINDINGS_PATH).read_text(encoding="utf-8"))
    assert payload["schema_version"] == contract.FINDINGS_SCHEMA_VERSION
    assert all(row["note"].strip() for row in payload["findings"])


def _repo(tmp_path: Path, source: str, *, test: str | None = None) -> Path:
    module = tmp_path / contract.SOURCE_ROOT / "probe.py"
    module.parent.mkdir(parents=True)
    module.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "tests").mkdir()
    if test is not None:
        (tmp_path / "tests/test_probe.py").write_text(textwrap.dedent(test), encoding="utf-8")
    return tmp_path


def _flags(tmp_path: Path, index: dict, source: str, *, bindings=(), test: str | None = None) -> set[str]:
    report = contract.check_repository(_repo(tmp_path, source, test=test), index, bindings)
    assert not report.unindexed, report.unindexed
    return {f"{finding.kind}:{finding.detail}" for finding in report.findings}


SLOT = """
from dataclasses import fields


def add_slot(embodiment):
    from copy import deepcopy

    from isaaclab.sensors.camera.camera_cfg import CameraCfg
    from isaaclab.utils import configclass

    rig = embodiment.camera_config
    slot = deepcopy(rig.external_camera_2)
    slot.prim_path = "{ENV_REGEX_NS}/exterior_right_camera"
    extended_type = configclass(type("Extended", (type(rig),),
                                     {"__annotations__": {"exterior_right_camera": CameraCfg},
                                      "exterior_right_camera": slot}))
    extended = extended_type(**{field.name: getattr(rig, field.name) for field in fields(rig)})
    TILED
    embodiment.camera_config = extended
"""
DROID = "isaaclab_arena.embodiments.droid.droid.DroidAbsoluteJointPositionEmbodiment"


@pytest.mark.parametrize(("tiled", "expected"), [
    ("extended.set_use_tiled_camera(rig.use_tiled_camera)",
     {"missing_attribute:type(..., (DroidCameraCfg,)).set_use_tiled_camera"}),
    ("extended.use_tiled_camera = rig.use_tiled_camera", set()),
    ("extended.exterior_right_camera.prim_path = slot.prim_path", set()),
])
def test_the_launch_crash_of_2026_11_03_is_caught_and_its_fix_passes(tmp_path, index, tiled, expected) -> None:
    source = SLOT.replace("TILED", tiled)
    bindings = ((f"{contract.SOURCE_ROOT}/probe.py", "add_slot", "embodiment", DROID),)
    assert _flags(tmp_path, index, source, bindings=bindings) == expected


def test_without_a_binding_an_untyped_receiver_is_not_guessed(tmp_path, index) -> None:
    assert _flags(tmp_path, index, SLOT.replace("TILED", "extended.set_use_tiled_camera(True)")) == set()


def test_names_modules_keywords_and_config_fields_are_checked(tmp_path, index) -> None:
    source = """
    import importlib

    import isaaclab.sim as sim_utils
    import isaaclab.simz
    from isaaclab.sensors import CameraCfg, CameraCfgz
    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment


    def build(pose):
        importlib.import_module("isaaclab_arena.embodiments.droid.droid")
        importlib.import_module("isaaclab_arena.embodiments.droidz")
        camera = CameraCfg(prim_path="/World/camera", heigth=720)
        camera.update_latest_camera_pose = True
        camera.colorize_semantic_segmentaton = False
        camera.offset.convention = "world"
        camera.offset.rotation = (0.0, 0.0, 0.0, 1.0)
        spawn = sim_utils.PinholeCameraCfg(focal_length=2.1, focus=1.0)
        sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        robot = DroidAbsoluteJointPositionEmbodiment(enable_cameras=True, initial_pose=pose)
        robot.scene_config.robot.spawn.usd_path = "robot.usd"
        robot.camera_config.use_tiled_camera = False
        robot.camera_config.set_use_tiled_camera(False)
        return spawn, robot
    """
    assert _flags(tmp_path, index, source) == {
        "missing_module:isaaclab.simz",
        "missing_name:isaaclab.sensors.CameraCfgz",
        "missing_module:isaaclab_arena.embodiments.droidz",
        "unknown_keyword:CameraCfg(heigth=)",
        "missing_field:CameraCfg.colorize_semantic_segmentaton",
        "missing_field:OffsetCfg.rotation",
        "unknown_keyword:PinholeCameraCfg(focus=)",
        "missing_attribute:DroidCameraCfg.set_use_tiled_camera",
    }


def test_a_name_assigned_two_ways_is_not_typed(tmp_path, index) -> None:
    source = """
    from isaaclab.sensors import CameraCfg, ContactSensorCfg


    def build(flag):
        sensor = CameraCfg(prim_path="/a")
        if flag:
            sensor = ContactSensorCfg(prim_path="/a")
        sensor.data_types = ["rgb"]
        return sensor
    """
    assert _flags(tmp_path, index, source) == set()


def test_a_fake_standing_in_for_a_pinned_class_may_only_declare_its_members(tmp_path, index) -> None:
    test = """
    import sys
    import types

    import isaaclab  # noqa: F401


    class CameraCfg:
        def __init__(self, prim_path):
            self.prim_path = prim_path
            self.exposure_bias = 0.0


    class DroidCameraCfg:
        use_tiled_camera = False

        def set_use_tiled_camera(self, value):
            self.use_tiled_camera = value


    def test_rig(monkeypatch):
        camera_cfg = types.ModuleType("isaaclab.sensors.camera.camera_cfg")
        camera_cfg.CameraCfg = CameraCfg
        camera_cfg.CameraRig = object
        monkeypatch.setitem(sys.modules, camera_cfg.__name__, camera_cfg)
    """
    assert _flags(tmp_path, index, "x = 1\n", test=test) == {
        "fake_member_not_pinned:CameraCfg.exposure_bias",
        "fake_member_not_pinned:DroidCameraCfg.set_use_tiled_camera",
        "fake_name_not_pinned:isaaclab.sensors.camera.camera_cfg.CameraRig",
    }


def test_a_rebuilt_index_is_written_one_record_per_line(index) -> None:
    text = contract.dump_index(index)
    assert json.loads(text) == index
    assert text == (ROOT / contract.INDEX_PATH).read_text(encoding="utf-8")
    assert len(text.splitlines()) == len(index["classes"]) + len(index["modules"]) + len(index) + 4


@pytest.mark.slow
@pytest.mark.external_data
def test_the_committed_index_says_only_what_the_pinned_sources_say(tmp_path, index) -> None:
    """Fetches the pinned sources (about 25 MB, 10 s), so neither CI lane selects it.

    Locally: ``python -m blueprint_pipeline.isaac_api_contract index --fetch-into DIR --verify``."""
    roots = contract.fetch_pinned_sources(tmp_path)
    full = contract.build_index(contract.pinned_package_dirs(roots["isaaclab"], roots["arena"]),
                                sources=contract.pinned_sources())
    assert contract.index_drift(index, full) == []
