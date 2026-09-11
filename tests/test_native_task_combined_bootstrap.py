"""Real worker mains: retained V28d bootstrap, fake native edges, real bodies.

BLUEPRINT_V28D_SAVED_BUNDLE optionally replays all exact retained runtime inputs;
only candidate module hashes are resealed in the scratch copy. Provider/model
operations are never part of this CPU rehearsal.
"""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
import zipfile

import pytest

from blueprint_pipeline import native_task_combined_diagnostic_worker as parent
from blueprint_pipeline import native_task_composition_worker as composition
from blueprint_pipeline import native_task_isaaclab_launch as launcher
from blueprint_pipeline.native_task_composition_diagnostic import file_record, seal
from tests import test_native_task_composition_diagnostic as composition_fixture
from tests.test_native_task_composition_diagnostic import request as composition_request
from tests.test_native_task_retained_command_replay import request as replay_request

native = composition_fixture.native

FIXTURE = Path(__file__).parent / "fixtures/native_task_combined_bootstrap_v28d"
RECEIPT_NAME = "native_task_runtime_source_provisioning.v1.json"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return file_record(path)


def _runtime(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    saved = os.environ.get("BLUEPRINT_V28D_SAVED_BUNDLE")
    if saved:
        with zipfile.ZipFile(saved) as archive:
            archive.extractall(root)
        root = root / "provider_runtime"
        manifest = json.loads((root / "adp_arena_provider_manifest.json").read_text())
    else:
        plan = {"plan_digest": "sha256:" + "d" * 64, "scenario": {"seed": 1881096655},
                "robot": {"robot_id": "franka"}}
        plan_record = _write(root / "runtime_inputs/composition_scene_plan.json", plan)
        req = composition_request()
        req.update(resolved_scene_plan=plan_record, resolved_scene_plan_digest=plan["plan_digest"],
                   seed=1881096655, packet_receipt_digest="sha256:" + "e" * 64)
        _write(root / "runtime_inputs/composition_request.json", seal(req, "request_digest"))
        req = replay_request()
        req["source_episode_json_pointer"] = "/episodes/0"
        req["scene_plan_digest"] = plan["plan_digest"]
        req["source_result"] = _write(root / "runtime_inputs/retained_cell_result.json", {
            "episodes": [{"episode": {"commanded_actions": req["commands"],
                                      "scientific_reset": req["expected_reset"]}}]})
        req["retained_adapter_reset"] = _write(root / "runtime_inputs/retained_adapter_reset.json", {
            "adapter_binding": {"gripper_command_mapping": {"closed_command": 1., "open_command": 0.,
                "closed_finger_separation_m": 0., "open_finger_separation_m": .08}}})
        _write(root / "runtime_inputs/replay_request.json", seal(req, "request_digest"))
        _write(root / "runtime_inputs/replay_scene_plan.json", plan)
        manifest = {"schema_version": "native_task_arena_provider_bundle.v1", "execution_mode": "runtime_preflight",
                    "implementation_commit": "a" * 40, "container_image": "cpu-fixture",
                    "packet_receipt_digest": "sha256:" + "e" * 64, "packet_files": [],
                    "bound_runtime_inputs": [{"relative_path": str(p.relative_to(root)), **file_record(p)}
                                             for p in sorted((root / "runtime_inputs").glob("*.json"))],
                    "runtime_modules": []}
    from blueprint_pipeline.native_task_composition_bundle import composition_runtime_sources
    package = root / "blueprint_pipeline"
    package.mkdir(exist_ok=True)
    sources = composition_runtime_sources(include_replay=True)
    for source in sources:
        shutil.copyfile(source, package / source.name)
    manifest["runtime_modules"] = [{"relative_path": "blueprint_pipeline/" + p.name, **file_record(p)}
                                   for p in sorted(package.glob("*.py"))]
    _write(root / "adp_arena_provider_manifest.json", seal(manifest, "input_digest"))
    return root


def _relocate_retained_experience(monkeypatch):
    """Redirect source filesystem reads only; never rewrite/reseal real receipt."""
    original = Path.resolve
    old_root = json.loads((FIXTURE / RECEIPT_NAME).read_text())["extraction_dir"]

    def resolve(path, *args, **kwargs):
        if str(path) == old_root or str(path).startswith(old_root + "/"):
            path = FIXTURE / str(path)[len(old_root):].lstrip("/")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)


def test_real_children_reach_render_and_174_command_bodies(tmp_path, monkeypatch, native):
    runtime = _runtime(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    shutil.copyfile(FIXTURE / RECEIPT_NAME, output / RECEIPT_NAME)
    _relocate_retained_experience(monkeypatch)
    # The actual saved receipt passes the original verifier against its retained
    # experience bytes. No bypass and no invented provisioning fields.
    assert launcher.verify_native_task_isaaclab_launch_contract(output / RECEIPT_NAME)["status"] == "qualified"
    req = json.loads((runtime / "runtime_inputs/replay_request.json").read_text())
    events, actions = [], []
    initial = req["expected_reset"]["observed"]["robot"]
    robot = NS(joint_names=[f"joint{i}" for i in range(9)], data=NS(
        joint_pos=deepcopy(initial["joint_pos"]), joint_limits=deepcopy(initial["joint_limits"]), body_pose_w=[]))
    sim = NS(current_time=0., current_time_step_index=0)
    env = NS(scene={"robot": robot}, sim=sim, reset=lambda **kw: events.append("env_reset"),
             close=lambda: events.append("env_close"))
    env.unwrapped = env
    built = NS(env=env)
    adapter = NS(_arm_joint_indices=list(range(7)), reset=lambda: events.append("adapter_reset"),
        joint_limits=lambda: initial["joint_limits"][0][:7],
        read_arm_joint_positions=lambda: list(actions[-1][:7]) if actions else initial["joint_pos"][0][:7],
        _arm_vector=lambda name: [0.] * 7, step=lambda action: actions.append(list(action)))
    from blueprint_pipeline import native_task_arena_construction_worker as construction
    from blueprint_pipeline import native_task_arena_preconstruction as preconstruction
    from blueprint_pipeline import native_task_arena_runtime as arena
    from blueprint_pipeline import native_task_nurec_render_setup as render
    from blueprint_pipeline import native_franka_pose_servo as servo
    from blueprint_pipeline import native_task_episode_environment as episode
    from blueprint_pipeline import native_task_arena_readback as readback
    from blueprint_pipeline import policy_scientific_reset as reset
    original_launch = launcher.launch_native_task_isaaclab

    def launch(path, **kwargs):
        assert Path(path).read_bytes() == (output / RECEIPT_NAME).read_bytes()
        events.append("launch:" + Path(path).parent.name)
        return original_launch(path, **kwargs,
            app_launcher_factory=lambda **kw: NS(app=NS(close=lambda: events.append("app_close"))),
            nurec_renderer_probe_factory=lambda: {"extension_enabled": True, "nurec_utils_extension_enabled": True,
                "renderer_hints": 3, "ujitso_geometry_enabled": True, "multi_gpu_enabled": False,
                "gaussian_skip_tonemapping_enabled": True, "schema_registered": True})

    monkeypatch.setattr(launcher, "launch_native_task_isaaclab", launch)
    monkeypatch.setattr(construction, "preflight_native_dependency_matrix", lambda **kw: {"all_required_available": True})
    monkeypatch.setattr(preconstruction, "prepare_native_task_arena_preconstruction", lambda **kw: {"passed": True})
    monkeypatch.setattr(arena, "build_native_task_arena_environment", lambda *a, **kw: built)
    monkeypatch.setattr(render, "setup_and_warm_native_nurec_renderer", lambda *a, **kw: {"passed": True})
    monkeypatch.setattr(servo, "NativeFrankaDifferentialIkServo", lambda **kw: object())
    monkeypatch.setattr(readback, "NativeRigidTaskArenaReadback", lambda *a: object())
    monkeypatch.setattr(episode, "build_native_task_episode_environment", lambda **kw: (adapter, {"cpu_fake": True}))
    monkeypatch.setattr(reset, "read_native_reset_channels", lambda *a: {
        key: deepcopy(req["expected_reset"][key]) for key in ("observed", "sources", "gaps")})
    monkeypatch.setattr(composition, "make_native_adapters", lambda **kw: native.adapters)
    monkeypatch.setitem(sys.modules, "omni", NS(usd=NS(get_context=lambda: NS(get_stage=lambda: native.stage))))
    monkeypatch.setitem(sys.modules, "omni.usd", sys.modules["omni"].usd)
    monkeypatch.setenv("PYTHONPATH", "/retained/isaac/wrapper/search/path")

    def runner(command, **kwargs):
        assert kwargs["env"]["PYTHONPATH"] == str(runtime) + os.pathsep + "/retained/isaac/wrapper/search/path"
        module_path = runtime / "blueprint_pipeline" / (command[2].rsplit(".", 1)[1] + ".py")
        spec = importlib.util.spec_from_file_location("rehearsed_" + module_path.stem, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module_path.name == "native_task_composition_worker.py":
            monkeypatch.setattr(module, "make_native_adapters", lambda **kw: native.adapters)
        with monkeypatch.context() as child_patch:
            child_patch.setenv("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", kwargs["env"]["BLUEPRINT_ADP_ARENA_OUTPUT_DIR"])
            code = module.main(command[3:])
        return NS(returncode=code)

    result = parent.run_diagnostic_children(runtime_root=runtime, output_root=output, runner=runner)
    documents = [json.loads((output / name / filename).read_text()) for name, _, filename in parent.CHILDREN]
    assert result["status"] == "completed", documents
    assert [row["pass"] for row in documents[0]["composition_diagnostic"]["passes"]] == ["full", "appearance_only", "native_meshes_only"]
    assert documents[1]["replay"]["executed_commands"] == 174
    assert actions == [row["isaac_action"] for row in req["commands"]]
    assert events.count("env_close") == events.count("app_close") == 2
    assert events.index("app_close") < events.index("launch:command_replay")
    assert all(row["runtime_source_provisioning"]["sha256"] == file_record(output / RECEIPT_NAME)["sha256"] for row in result["children"])
    report = os.environ.get("BLUEPRINT_V28D_REPLAY_REPORT")
    if report:
        _write(Path(report), {"cpu_only": True, "saved_bundle": os.environ.get("BLUEPRINT_V28D_SAVED_BUNDLE"),
            "provisioning_receipt": file_record(output / RECEIPT_NAME), "result": result,
            "events": events, "executed_fake_native_commands": len(actions),
            "native_render_or_physics_evidence_generated": False})


def test_missing_parent_receipt_stops_before_child_start(tmp_path):
    runtime = _runtime(tmp_path)
    result = parent.run_diagnostic_children(runtime_root=runtime, output_root=tmp_path / "out",
        runner=lambda *a, **kw: pytest.fail("missing receipt must not launch a child"))
    assert result["status"] == "blocked"
    assert len(result["children"]) == 1


def test_corrupt_retained_receipt_still_fails_original_validator(tmp_path, monkeypatch):
    _relocate_retained_experience(monkeypatch)
    raw = json.loads((FIXTURE / RECEIPT_NAME).read_text())
    raw["status"] = "blocked"
    path = tmp_path / RECEIPT_NAME
    _write(path, raw)
    with pytest.raises(launcher.NativeTaskIsaacLabLaunchError, match="provisioning_receipt_invalid"):
        launcher.verify_native_task_isaaclab_launch_contract(path)
