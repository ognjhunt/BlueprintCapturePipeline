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
            }.get(value, "")

    event_count = [0]
    subscription = _subscribe_body_contact_reports(
        omni_physx=OmniPhysx,
        physics_schema_tools=PhysicsSchemaTools,
        body_path="/World/Placement/Replacement/Body",
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
