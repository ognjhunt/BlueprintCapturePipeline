from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_native_import_driver as driver
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import (
    ADAPTER_ID,
    RUNTIME_RESULT_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationNativeImportDriverError,
    _one_native_settle,
    _subscribe_body_contact_reports,
    execute_native_import_component,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _environment(tmp_path: Path) -> dict[str, str]:
    output = tmp_path / "output"
    output.mkdir()
    stage_input = {
        "schema_version": ("task_evaluation_scene_configuration_stage_production_input.v1"),
        "stage": {
            "stage_id": "stage-5",
            "adapter": {"id": ADAPTER_ID, "version": "v1"},
        },
        "configuration": {
            "schema_version": ("replacement_native_import_qualification_configuration.v1"),
            "replacement_identity": {"id": "rigid-object", "version": "v1"},
            "required_checks": {
                "stage_import": True,
                "rigid_body_enabled": True,
                "collider_enabled": True,
                "gravity_settle_seconds": 3.0,
                "maximum_settle_translation_m": 0.01,
                "maximum_settle_rotation_rad": 0.08,
                "support_contact_required": True,
                "explosion_or_tunneling_forbidden": True,
                "deterministic_reset_required": True,
                "state_digest_repeat_count": 3,
            },
        },
    }
    stage_input_path = output / "stage-input.json"
    stage_input_path.write_text(json.dumps(stage_input), encoding="utf-8")
    asset = output / "qualified.usda"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    static = output / "static.json"
    static.write_text('{"qualified":true}\n', encoding="utf-8")
    dependencies = [
        {
            "output_artifacts": [
                {
                    "role": "statically_qualified_replacement_asset",
                    "path": str(asset),
                    "digest": _sha256(asset),
                    "size_bytes": asset.stat().st_size,
                },
                {
                    "role": "static_qualification_receipt",
                    "path": str(static),
                    "digest": _sha256(static),
                    "size_bytes": static.stat().st_size,
                },
            ]
        }
    ]
    dependencies_path = output / "dependencies.json"
    dependencies_path.write_text(json.dumps(dependencies), encoding="utf-8")
    return {
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT": str(stage_input_path),
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(dependencies_path),
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output),
        "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(output / "component-result.json"),
    }


def _observed(*, mismatch: bool = False) -> dict:
    repeats = []
    for index in range(3):
        state = {"position_m": [0.0, 0.0, 0.05], "orientation_xyzw": [0, 0, 0, 1]}
        repeats.append(
            {
                "asset_imported": True,
                "rigid_body_paths": ["/World/Placement/Replacement/links/root"],
                "collision_paths": ["/World/Placement/Replacement/links/root/geometry/collision"],
                "support_contact_observed": True,
                "contact_report_event_count": 5,
                "settle_translation_m": 0.005,
                "settle_rotation_rad": 0.01,
                "final_state": state,
                "final_state_digest": (
                    canonical_digest(state) if not mismatch or index < 2 else "sha256:" + "f" * 64
                ),
            }
        )
    return {
        "runtime_identity": {"engine_version": "6.0.1"},
        "repeats": repeats,
    }


def _native_runner(observed: dict):
    def _run(*, observation_consumer, **_kwargs):
        return observation_consumer(observed)

    return _run


def test_native_settle_uses_contact_callback_instead_of_unsafe_polling() -> None:
    class Interface:
        callback = None

        def subscribe_contact_report_events(self, callback):
            self.callback = callback
            return object()

        def get_contact_report(self):
            raise AssertionError("contact reports must not be polled")

    class OmniPhysx:
        interface = Interface()

        @classmethod
        def get_physx_simulation_interface(cls):
            return cls.interface

    class PhysicsSchemaTools:
        @staticmethod
        def intToSdfPath(value: int) -> str:
            return {
                1: "/World/Placement/Replacement/Body/Collider",
                2: "/World/Ground",
                3: "/World/Other",
                4: "/World/Placement/Replacement/Drawer",
            }.get(value, "")

    event_count = [0]
    subscription = _subscribe_body_contact_reports(
        omni_physx=OmniPhysx,
        physics_schema_tools=PhysicsSchemaTools,
        body_path="/World/Placement/Replacement/Body",
        support_path="/World/Ground",
        event_count=event_count,
    )
    assert subscription is not None
    assert OmniPhysx.interface.callback is not None
    OmniPhysx.interface.callback(
        [SimpleNamespace(actor0=1, actor1=2, collider0=0, collider1=0)],
        [],
    )
    OmniPhysx.interface.callback(
        [SimpleNamespace(actor0=3, actor1=2, collider0=0, collider1=0)],
        [],
    )
    OmniPhysx.interface.callback(
        [SimpleNamespace(actor0=1, actor1=4, collider0=0, collider1=0)],
        [],
    )
    assert event_count == [1]

    settle_source = inspect.getsource(_one_native_settle)
    assert "_subscribe_body_contact_reports(" in settle_source
    assert "_contact_count(" not in settle_source


def test_native_driver_seals_only_three_matching_contact_settles(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    result = execute_native_import_component(
        environment=environment,
        native_runner=_native_runner(_observed()),
    )

    assert result["status"] == "completed"
    assert result["provider_mutations_performed"] == 0
    artifact = result["artifacts"][0]
    runtime = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    assert runtime["schema_version"] == RUNTIME_RESULT_SCHEMA_VERSION
    assert runtime["native_isaac_executed"] is True
    assert runtime["support_contact_observed"] is True
    assert runtime["deterministic_reset_state_digest_repeat_count"] == 3
    assert runtime["qualification_limits"] == {
        "gravity_settle_seconds": 3.0,
        "maximum_settle_rotation_rad": 0.08,
        "maximum_settle_translation_m": 0.01,
        "state_digest_repeat_count": 3,
    }


def test_native_driver_rejects_nondeterministic_reset(tmp_path: Path) -> None:
    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_qualification_failed",
    ):
        execute_native_import_component(
            environment=_environment(tmp_path),
            native_runner=_native_runner(_observed(mismatch=True)),
        )


def test_native_driver_rejects_unexpected_isaac_runtime_identity(
    tmp_path: Path,
) -> None:
    observed = _observed()
    observed["runtime_identity"] = {"engine_version": "6.0.0"}

    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_runtime_identity_invalid",
    ):
        execute_native_import_component(
            environment=_environment(tmp_path),
            native_runner=_native_runner(observed),
        )


def test_native_driver_seals_result_before_simulation_app_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Isaac shutdown may terminate Python, so durable proof must exist first."""

    closed = False

    class FakeSimulationApp:
        def __init__(self, _config: dict) -> None:
            pass

        def close(self) -> None:
            nonlocal closed
            closed = True
            raise SystemExit(0)

    monkeypatch.setattr(driver, "_bind_isaac_runtime_environment", lambda: None)
    monkeypatch.setattr(driver, "_import_simulation_app", lambda: FakeSimulationApp)
    monkeypatch.setattr(
        driver,
        "_observe_isaac_runtime_identity",
        lambda _app: {"engine_version": "6.0.1"},
    )
    observations = iter(_observed()["repeats"])
    monkeypatch.setattr(driver, "_one_native_settle", lambda **_kwargs: next(observations))
    environment = _environment(tmp_path)

    with pytest.raises(SystemExit):
        execute_native_import_component(environment=environment)

    assert closed is True
    component_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT"])
    component = json.loads(component_path.read_text(encoding="utf-8"))
    artifact = component["artifacts"][0]
    assert component["status"] == "completed"
    assert Path(artifact["path"]).is_file()


def test_native_driver_rejects_invalid_bounds_before_runtime(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    stage_input = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    value = json.loads(stage_input.read_text(encoding="utf-8"))
    value["configuration"]["required_checks"]["maximum_settle_translation_m"] = float("inf")
    stage_input.write_text(json.dumps(value), encoding="utf-8")
    executed = False

    def native_runner(**_kwargs):
        nonlocal executed
        executed = True
        return _kwargs["observation_consumer"](_observed())

    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_input_invalid",
    ):
        execute_native_import_component(
            environment=environment,
            native_runner=native_runner,
        )
    assert executed is False


# --- supplemental passive destination imported in the same Isaac session -----

DESTINATION_IDENTITY = {"id": "document-tray", "version": "v1"}


def _destination_environment(tmp_path: Path, *, dependency_roles: bool = True) -> dict[str, str]:
    environment = _environment(tmp_path)
    output = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
    stage_input_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    stage_input = json.loads(stage_input_path.read_text(encoding="utf-8"))
    stage_input["construction_envelope"] = {
        "recipe": {
            "subject_identity": {"id": "rigid-object", "version": "v1"},
            "supplemental_destination": {
                "identity": DESTINATION_IDENTITY,
                "relation": "inside",
            },
        }
    }
    stage_input_path.write_text(json.dumps(stage_input), encoding="utf-8")
    if dependency_roles:
        asset = output / "tray.usdz"
        asset.write_bytes(b"PK-tray")
        static = output / "tray-static.json"
        static.write_text('{"replacement_identity":"document-tray"}\n', encoding="utf-8")
        dependencies_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"])
        dependencies = json.loads(dependencies_path.read_text(encoding="utf-8"))
        dependencies[0]["output_artifacts"].extend(
            [
                {
                    "role": "statically_qualified_destination_asset",
                    "path": str(asset),
                    "digest": _sha256(asset),
                    "size_bytes": asset.stat().st_size,
                },
                {
                    "role": "destination_static_qualification_receipt",
                    "path": str(static),
                    "digest": _sha256(static),
                    "size_bytes": static.stat().st_size,
                },
            ]
        )
        dependencies_path.write_text(json.dumps(dependencies), encoding="utf-8")
    return environment


def _destination_observed() -> dict:
    observed = _observed()
    repeats = []
    for _index in range(3):
        state = {"position_m": [0.0, 0.0, 0.0145], "orientation_xyzw": [0, 0, 0, 1]}
        repeats.append(
            {
                "asset_imported": True,
                "rigid_body_paths": ["/World/Placement/Replacement"],
                "collision_paths": ["/World/Placement/Replacement/Colliders/Bottom"],
                "support_contact_observed": True,
                "contact_report_event_count": 4,
                "settle_translation_m": 0.001,
                "settle_rotation_rad": 0.002,
                "final_state": state,
                "final_state_digest": canonical_digest(state),
            }
        )
    observed["destination_repeats"] = repeats
    return observed


def test_native_driver_imports_the_supplemental_destination_in_the_same_session(
    tmp_path: Path,
) -> None:
    environment = _destination_environment(tmp_path)
    seen: dict = {}

    def runner(*, asset_path, required_checks, observation_consumer, destination_asset_path=None):
        seen["asset_path"] = asset_path
        seen["destination_asset_path"] = destination_asset_path
        return observation_consumer(_destination_observed())

    result = execute_native_import_component(environment=environment, native_runner=runner)

    assert seen["destination_asset_path"] is not None
    assert seen["destination_asset_path"].name == "tray.usdz"
    artifacts = {row["role"]: row for row in result["artifacts"]}
    assert set(artifacts) == {
        "native_import_runtime_result",
        "destination_native_import_runtime_result",
    }
    destination = json.loads(
        Path(artifacts["destination_native_import_runtime_result"]["path"]).read_text()
    )
    assert destination["schema_version"] == RUNTIME_RESULT_SCHEMA_VERSION
    assert destination["status"] == "qualified"
    assert destination["replacement_identity"] == DESTINATION_IDENTITY
    output = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
    assert destination["asset_digest"] == _sha256(output / "tray.usdz")
    assert destination["static_qualification_digest"] == _sha256(output / "tray-static.json")
    assert destination["deterministic_reset_state_digest_repeat_count"] == 3
    assert destination["result_digest"] == canonical_digest(
        destination, digest_field="result_digest"
    )
    subject = json.loads(Path(artifacts["native_import_runtime_result"]["path"]).read_text())
    assert subject["replacement_identity"] == {"id": "rigid-object", "version": "v1"}


def test_native_driver_refuses_a_declared_destination_without_its_settle_observations(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_destination_execution_invalid",
    ):
        execute_native_import_component(
            environment=_destination_environment(tmp_path),
            native_runner=_native_runner(_observed()),
        )


def test_native_driver_refuses_a_declared_destination_without_stage4_artifacts(
    tmp_path: Path,
) -> None:
    executed = False

    def runner(**_kwargs):
        nonlocal executed
        executed = True

    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_destination_dependency_invalid",
    ):
        execute_native_import_component(
            environment=_destination_environment(tmp_path, dependency_roles=False),
            native_runner=runner,
        )
    assert executed is False


def _drawer_graph() -> dict:
    """The qualified graph a real drawer-cabinet receipt carries (carcass root, one slide)."""
    drive = {"drive_type": "none", "stiffness": 0.0, "damping": 0.0, "maximum_force": 0.0}
    links = ("carcass", "drawer_0", "drawer_1", "drawer_2")
    return {
        "schema_version": "adp_articulation_graph.v1",
        "links": [{"link_id": name, "is_root": name == "carcass", "semantic_role": name} for name in links],
        "joints": [
            {"joint_id": "task_part_joint", "role": "target", "joint_type": "prismatic",
             "parent_link_id": "carcass", "child_link_id": "drawer_1", "axis": [1.0, 0.0, 0.0],
             "limits": [0.0, 0.32], "reset_position": 0.0, "reset_tolerance": 0.005, "drive": dict(drive)},
            *[{"joint_id": f"{name}_fixed", "role": "locked", "joint_type": "fixed",
               "parent_link_id": "carcass", "child_link_id": name, "axis": [0.0, 0.0, 0.0],
               "limits": [0.0, 0.0], "reset_position": 0.0, "reset_tolerance": 0.005, "drive": dict(drive)}
              for name in ("drawer_0", "drawer_2")],
        ],
        "collision_pairs": [],
        "success_predicate": {"combination": "all", "joint_intervals": {"task_part_joint": [0.192, 0.32]}},
    }


def _articulated_environment(
    tmp_path: Path, *, with_destination: bool = False
) -> dict[str, str]:
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_records import (
        stage_five_configuration,
    )
    environment = (
        _destination_environment(tmp_path) if with_destination else _environment(tmp_path)
    )
    stage_input_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    stage_input = json.loads(stage_input_path.read_text())
    stage_input["configuration"] = {
        **stage_five_configuration(replacement_identity={"id": "cabinet", "version": "v1"},
                                   articulated=True),
        "schema_version": "replacement_native_import_qualification_configuration.v1",
    }
    if with_destination:
        stage_input["construction_envelope"]["recipe"]["subject_identity"] = {
            "id": "cabinet", "version": "v1"
        }
    stage_input_path.write_text(json.dumps(stage_input), encoding="utf-8")
    output = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"])
    asset = output / "qualified.usda"
    links = {
        name: {
            "prim_path": f"/Asset/links/{name}", "mass_kg": 1.0,
            "center_of_mass_m": [0.0, 0.0, 0.0],
            "diagonal_inertia_kg_m2": [0.1, 0.1, 0.1],
            "collision_prim_paths": [f"/Asset/links/{name}/collision/shape"],
        }
        for name in ("carcass", "drawer_0", "drawer_1", "drawer_2")
    }
    links["drawer_1"]["collision_prim_paths"].append("/Asset/links/drawer_1/collision/handle")
    static = {
        "schema_version": "task_evaluation_articulated_replacement_static_qualification.v1",
        "status": "authored_structure_statically_qualified",
        "asset_kind": "articulated_assembly",
        "replacement_identity": {"id": "cabinet", "version": "v1"},
        "replacement_usd": {"sha256": _sha256(asset), "size_bytes": asset.stat().st_size},
        "structural_findings": [],
        "observed_structure": {
            "links": links,
            "joint_prim_paths": ["/Asset/joints/task_part_joint", "/Asset/joints/drawer_0_fixed", "/Asset/joints/drawer_2_fixed"],
        },
        "task_joint": {"joint_id": "task_part_joint", "prim_path": "/Asset/joints/task_part_joint", "joint_type": "prismatic", "limits": [0.0, 0.32], "reset_position": 0.0},
        "task_contact": {"contact_link_id": "drawer_1", "handle_prim_paths": ["/Asset/links/drawer_1/collision/handle"]},
        "articulation_graph": _drawer_graph(),
        "result_digest": "",
    }
    static["result_digest"] = canonical_digest(static, digest_field="result_digest")
    static_path = output / "static.json"
    static_path.write_text(json.dumps(static), encoding="utf-8")
    dependencies_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"])
    dependencies = json.loads(dependencies_path.read_text())
    static_artifact = dependencies[0]["output_artifacts"][1]
    static_artifact.update(digest=_sha256(static_path), size_bytes=static_path.stat().st_size)
    dependencies_path.write_text(json.dumps(dependencies), encoding="utf-8")
    return environment


def _articulated_observed(**overrides) -> dict:
    joint = {
        "task_joint_prim_path": "/World/Placement/Replacement/joints/task_part_joint",
        "task_joint_name": "task_part_joint", "task_joint_type": "prismatic",
        "task_joint_limits": [0.0, 0.32], "task_joint_reset_position": 0.0,
        "moving_link_prim_path": "/World/Placement/Replacement/links/drawer_1",
        "fixed_joint_prim_paths": ["/World/Placement/Replacement/joints/drawer_0_fixed", "/World/Placement/Replacement/joints/drawer_2_fixed"],
        "task_joint_axis": "X", "initial_task_joint_position": 0.0,
        "task_joint_drive_forbidden_verified": True,
        "settled_task_joint_position": 0.0, "task_joint_returned_to_reset": True,
        **overrides,
    }
    repeats = []
    for _ in range(3):
        state = {"position_m": [0.0, 0.0, 0.31], "orientation_xyzw": [0, 0, 0, 1], "task_joint_position_m": round(float(joint["settled_task_joint_position"]), 7)}
        links = {
            name: {"prim_path": f"/World/Placement/Replacement/links/{name}", "mass_kg": 1.0,
                   "center_of_mass_m": [0.0, 0.0, 0.0], "diagonal_inertia_kg_m2": [0.1, 0.1, 0.1],
                   "collision_prim_paths": [f"/World/Placement/Replacement/links/{name}/collision/shape"]}
            for name in ("carcass", "drawer_0", "drawer_1", "drawer_2")
        }
        links["drawer_1"]["collision_prim_paths"].append(
            "/World/Placement/Replacement/links/drawer_1/collision/handle"
        )
        repeats.append({
            "asset_imported": True,
            "rigid_body_paths": [row["prim_path"] for row in links.values()],
            "articulation_root_paths": ["/World/Placement/Replacement"],
            "settle_measured_body_prim_path": "/World/Placement/Replacement/links/carcass",
            "collision_paths": ["/World/Placement/Replacement/links/carcass/collision/FinalVisualShape"],
            "fixed_joint_prim_paths": joint["fixed_joint_prim_paths"],
            "link_physics_readback": links,
            "handle_prim_paths": ["/World/Placement/Replacement/links/drawer_1/collision/handle"],
            "task_joint": dict(joint),
            "support_contact_observed": True, "contact_report_event_count": 7,
            "settle_translation_m": 0.001, "settle_rotation_rad": 0.002,
            "final_state": state, "final_state_digest": canonical_digest(state),
            "task_joint_trace_digest": "sha256:" + "a" * 64,
        })
    return {"runtime_identity": {"engine_version": "6.0.1"}, "repeats": repeats}


def test_articulated_assembly_qualifies_on_links_and_its_one_task_joint(tmp_path: Path) -> None:
    environment = _articulated_environment(tmp_path)
    captured: dict = {}

    def runner(*, observation_consumer, **kwargs):
        captured.update(kwargs)
        return observation_consumer(_articulated_observed())

    result = execute_native_import_component(environment=environment, native_runner=runner)
    assert captured["articulated"] is True
    artifact = next(row for row in result["artifacts"] if row["role"] == "native_import_runtime_result")
    runtime = json.loads(Path(artifact["path"]).read_text())
    assert runtime["status"] == "qualified" and runtime["asset_kind"] == "articulated_assembly"
    assert runtime["native_simulator_import_qualified"] is True
    assert runtime["task_joint_readback"]["task_joint_name"] == "task_part_joint"
    assert runtime["task_joint_readback"]["task_joint_limits"] == [0.0, 0.32]
    assert runtime["task_joint_drive_forbidden_verified"] is True
    assert runtime["task_joint_closed_at_reset_verified"] is True
    assert runtime["task_joint_travel_is_measured"] is False
    assert runtime["evaluation_episode_executed"] is False


def test_articulated_subject_with_rigid_destination_seals_both_results(
    tmp_path: Path,
) -> None:
    environment = _articulated_environment(tmp_path, with_destination=True)
    observed = _articulated_observed()
    observed["destination_repeats"] = _destination_observed()["destination_repeats"]
    result = execute_native_import_component(
        environment=environment, native_runner=_native_runner(observed)
    )
    artifacts = {row["role"]: row for row in result["artifacts"]}
    subject = json.loads(Path(artifacts["native_import_runtime_result"]["path"]).read_text())
    destination = json.loads(
        Path(artifacts["destination_native_import_runtime_result"]["path"]).read_text()
    )
    assert subject["asset_kind"] == "articulated_assembly"
    assert destination["replacement_identity"] == DESTINATION_IDENTITY
    assert "link_physics_readback" not in destination
    assert destination["result_digest"] == canonical_digest(
        destination, digest_field="result_digest"
    )


def test_articulated_native_rejects_matching_but_wrong_fixed_joint_paths(
    tmp_path: Path,
) -> None:
    observed = _articulated_observed()
    for repeat in observed["repeats"]:
        wrong = [
            "/World/Placement/Replacement/joints/wrong_0",
            "/World/Placement/Replacement/joints/wrong_2",
        ]
        repeat["fixed_joint_prim_paths"] = wrong
        repeat["task_joint"]["fixed_joint_prim_paths"] = wrong
    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_qualification_failed",
    ):
        execute_native_import_component(
            environment=_articulated_environment(tmp_path),
            native_runner=_native_runner(observed),
        )


@pytest.mark.parametrize("mutation", [
    {"task_joint_returned_to_reset": False},
    {"task_joint_drive_forbidden_verified": False},
    {"task_joint_limits": [0.0, 0.0]},
])
def test_articulated_qualification_refuses_a_crept_driven_or_unlimited_joint(
    tmp_path: Path, mutation: dict
) -> None:
    environment = _articulated_environment(tmp_path)
    runner = _native_runner(_articulated_observed(**mutation))
    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError,
                       match="native_import_qualification_failed"):
        execute_native_import_component(environment=environment, native_runner=runner)


def test_a_single_rigid_body_cannot_satisfy_the_articulated_gate(tmp_path: Path) -> None:
    """A rigid solid presented for an articulated stage has no joint to read."""
    environment = _articulated_environment(tmp_path)
    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError,
                       match="native_import_qualification_failed"):
        execute_native_import_component(environment=environment,
                                        native_runner=_native_runner(_observed()))


def test_rigid_stage_still_refuses_an_articulated_observation(tmp_path: Path) -> None:
    """The rigid gate keeps its exactly-one-body rule."""
    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError,
                       match="native_import_qualification_failed"):
        execute_native_import_component(environment=_environment(tmp_path),
                                        native_runner=_native_runner(_articulated_observed()))


@pytest.mark.parametrize("damage", ["missing_drawer", "mass", "handle", "joint_state_digest"])
def test_articulated_native_readback_must_match_the_qualified_parts(
    tmp_path: Path, damage: str
) -> None:
    environment = _articulated_environment(tmp_path)
    observation = _articulated_observed()
    for repeat in observation["repeats"]:
        if damage == "missing_drawer":
            del repeat["link_physics_readback"]["drawer_2"]
            repeat["rigid_body_paths"].remove("/World/Placement/Replacement/links/drawer_2")
        elif damage == "mass":
            repeat["link_physics_readback"]["drawer_1"]["mass_kg"] = 0.01
        elif damage == "handle":
            repeat["handle_prim_paths"] = ["/World/Placement/Replacement/links/carcass/collision/shape"]
        else:
            repeat["final_state"]["task_joint_position_m"] = 0.02
    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError,
                       match="native_import_qualification_failed"):
        execute_native_import_component(environment=environment,
                                        native_runner=_native_runner(observation))


def test_articulated_native_import_refuses_unbound_static_receipt_before_runtime(tmp_path: Path) -> None:
    environment = _articulated_environment(tmp_path)
    static_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"]) / "static.json"
    receipt = json.loads(static_path.read_text())
    receipt["replacement_usd"]["sha256"] = "sha256:" + "0" * 64
    receipt["result_digest"] = canonical_digest(receipt, digest_field="result_digest")
    static_path.write_text(json.dumps(receipt), encoding="utf-8")
    dependencies_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES"])
    dependencies = json.loads(dependencies_path.read_text())
    dependencies[0]["output_artifacts"][1].update(digest=_sha256(static_path), size_bytes=static_path.stat().st_size)
    dependencies_path.write_text(json.dumps(dependencies), encoding="utf-8")
    invoked = False

    def runner(**_kwargs):
        nonlocal invoked
        invoked = True

    with pytest.raises(TaskEvaluationSceneConfigurationNativeImportDriverError,
                       match="native_import_static_receipt_invalid"):
        execute_native_import_component(environment=environment, native_runner=runner)
    assert invoked is False


def test_native_settle_reads_the_task_joint_and_refuses_a_driven_one() -> None:
    """The in-Isaac joint read is exercised here against a stage, not on a rented GPU."""
    from pxr import Usd, UsdGeom, UsdPhysics
    from blueprint_pipeline.task_evaluation_scene_configuration_native_import_driver import (
        _articulated_joint_observation,
    )
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Placement/Replacement")
    for name in ("carcass", "drawer_1"):
        UsdPhysics.RigidBodyAPI.Apply(
            UsdGeom.Xform.Define(stage, f"/World/Placement/Replacement/links/{name}").GetPrim())
    task = UsdPhysics.PrismaticJoint.Define(stage, "/World/Placement/Replacement/joints/task_part_joint")
    task.CreateAxisAttr("X")
    task.CreateLowerLimitAttr(0.0)
    task.CreateUpperLimitAttr(0.32)
    task.GetPrim().SetCustomDataByKey("blueprint:resetPosition", 0.0)
    UsdPhysics.FixedJoint.Define(stage, "/World/Placement/Replacement/joints/drawer_0_fixed")
    observed = _articulated_joint_observation(
        stage=stage, usd_physics=UsdPhysics, root_path="/World/Placement/Replacement")
    assert observed["task_joint_name"] == "task_part_joint"
    assert observed["task_joint_type"] == "prismatic"
    # USD stores the limit as float32; the readback keeps the stage's own value.
    assert observed["task_joint_limits"][0] == 0.0
    assert observed["task_joint_limits"][1] == pytest.approx(0.32, abs=1e-6)
    assert observed["fixed_joint_prim_paths"] == ["/World/Placement/Replacement/joints/drawer_0_fixed"]
    assert observed["task_joint_drive_forbidden_verified"] is True
    UsdPhysics.DriveAPI.Apply(task.GetPrim(), "linear").CreateStiffnessAttr().Set(250.0)
    with pytest.raises(RuntimeError, match="task_joint_drive_forbidden"):
        _articulated_joint_observation(stage=stage, usd_physics=UsdPhysics,
                                       root_path="/World/Placement/Replacement")


def test_imported_articulation_reads_exact_four_link_physics_and_handle(tmp_path: Path) -> None:
    """Exercise the composed USD readback before any Isaac allocation."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    environment = _articulated_environment(tmp_path)
    static_path = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT"]) / "static.json"
    receipt = json.loads(static_path.read_text())
    root = "/World/Placement/Replacement"
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, root).GetPrim())
    for name, row in receipt["observed_structure"]["links"].items():
        path = f"{root}/links/{name}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(row["mass_kg"])
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*row["center_of_mass_m"]))
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*row["diagonal_inertia_kg_m2"]))
        for collider in row["collision_prim_paths"]:
            imported = root + collider[len("/Asset"):]
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, imported).GetPrim())
    task = UsdPhysics.PrismaticJoint.Define(stage, root + "/joints/task_part_joint")
    task.CreateAxisAttr("X")
    task.CreateLowerLimitAttr(0.0)
    task.CreateUpperLimitAttr(0.32)
    task.CreateBody0Rel().SetTargets([root + "/links/carcass"])
    task.CreateBody1Rel().SetTargets([root + "/links/drawer_1"])
    task.GetPrim().SetCustomDataByKey("blueprint:resetPosition", 0.0)
    task.GetPrim().SetCustomDataByKey("blueprint:graphAxis", Gf.Vec3d(1.0, 0.0, 0.0))
    for name in ("drawer_0", "drawer_2"):
        fixed = UsdPhysics.FixedJoint.Define(stage, root + f"/joints/{name}_fixed")
        fixed.CreateBody0Rel().SetTargets([root + "/links/carcass"])
        fixed.CreateBody1Rel().SetTargets([root + f"/links/{name}"])
    observed = driver._articulated_structure_observation(
        stage=stage, usd_physics=UsdPhysics, static_receipt=receipt)
    assert set(observed["link_physics_readback"]) == {"carcass", "drawer_0", "drawer_1", "drawer_2"}
    assert observed["task_joint"]["task_joint_axis"] == "X"
    assert observed["handle_prim_paths"] == [root + "/links/drawer_1/collision/handle"]
    UsdPhysics.MassAPI(stage.GetPrimAtPath(root + "/links/drawer_1")).GetMassAttr().Set(0.5)
    with pytest.raises(RuntimeError, match="link_physics_mismatch"):
        driver._articulated_structure_observation(
            stage=stage, usd_physics=UsdPhysics, static_receipt=receipt)


def test_joint_numeric_readback_uses_initialized_isaac_articulation() -> None:
    class FakeArticulation:
        dof_names = ["task_part_joint"]

        def get_dof_index(self, name):
            assert name == "task_part_joint"
            return 0

        def get_joint_positions(self):
            return [0.003]

    assert driver._live_joint_position(FakeArticulation(), "task_part_joint") == 0.003
    with pytest.raises(RuntimeError, match="joint_dof_unresolved"):
        driver._live_joint_position(FakeArticulation(), "other")


def test_native_structure_readback_matches_actual_packaged_and_statically_qualified_asset(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics
    from tests.test_task_evaluation_scene_configuration_articulated_static_qualification import (
        IDENTITY, _sealed,
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_articulated_static_qualification import (
        qualify_scene_configuration_articulated_asset_static,
    )

    asset, graph, authoring = _sealed(tmp_path)
    receipt = qualify_scene_configuration_articulated_asset_static(
        asset_path=asset, graph_spec=graph, authoring_receipt=authoring,
        replacement_identity=IDENTITY, output_path=tmp_path / "static.json",
    )
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Placement")
    replacement = stage.DefinePrim("/World/Placement/Replacement", "Xform")
    replacement.GetReferences().AddReference(str(asset), "/Asset")
    stage.Load()
    observed = driver._articulated_structure_observation(
        stage=stage, usd_physics=UsdPhysics, static_receipt=receipt,
    )
    assert set(observed["link_physics_readback"]) == {"carcass", "drawer_0", "drawer_1", "drawer_2"}
    assert len(observed["fixed_joint_prim_paths"]) == 2
    assert observed["handle_prim_paths"] == [
        "/World/Placement/Replacement/links/drawer_1/collision/handle"
    ]
