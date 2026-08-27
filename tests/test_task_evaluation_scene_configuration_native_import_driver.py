from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        "schema_version": (
            "task_evaluation_scene_configuration_stage_production_input.v1"
        ),
        "stage": {
            "stage_id": "stage-5",
            "adapter": {"id": ADAPTER_ID, "version": "v1"},
        },
        "configuration": {
            "schema_version": (
                "replacement_native_import_qualification_configuration.v1"
            ),
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
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_DEPENDENCIES": str(
            dependencies_path
        ),
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_OUTPUT_ROOT": str(output),
        "BLUEPRINT_SCENE_CONFIGURATION_COMPONENT_RESULT": str(
            output / "component-result.json"
        ),
    }


def _observed(*, mismatch: bool = False) -> dict:
    repeats = []
    for index in range(3):
        state = {"position_m": [0.0, 0.0, 0.05], "orientation_xyzw": [0, 0, 0, 1]}
        repeats.append(
            {
                "asset_imported": True,
                "rigid_body_paths": ["/World/Placement/Replacement/links/root"],
                "collision_paths": [
                    "/World/Placement/Replacement/links/root/geometry/collision"
                ],
                "support_contact_observed": True,
                "contact_report_event_count": 5,
                "settle_translation_m": 0.005,
                "settle_rotation_rad": 0.01,
                "final_state": state,
                "final_state_digest": (
                    canonical_digest(state)
                    if not mismatch or index < 2
                    else "sha256:" + "f" * 64
                ),
            }
        )
    return {
        "runtime_identity": {"engine_version": "6.0.1"},
        "repeats": repeats,
    }


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
        native_runner=lambda **_kwargs: _observed(),
    )

    assert result["status"] == "completed"
    assert result["provider_mutations_performed"] == 0
    artifact = result["artifacts"][0]
    runtime = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
    assert runtime["schema_version"] == RUNTIME_RESULT_SCHEMA_VERSION
    assert runtime["native_isaac_executed"] is True
    assert runtime["support_contact_observed"] is True
    assert runtime["deterministic_reset_state_digest_repeat_count"] == 3


def test_native_driver_rejects_nondeterministic_reset(tmp_path: Path) -> None:
    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_qualification_failed",
    ):
        execute_native_import_component(
            environment=_environment(tmp_path),
            native_runner=lambda **_kwargs: _observed(mismatch=True),
        )


def test_native_driver_rejects_invalid_bounds_before_runtime(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    stage_input = Path(environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_INPUT"])
    value = json.loads(stage_input.read_text(encoding="utf-8"))
    value["configuration"]["required_checks"][
        "maximum_settle_translation_m"
    ] = float("inf")
    stage_input.write_text(json.dumps(value), encoding="utf-8")
    executed = False

    def native_runner(**_kwargs):
        nonlocal executed
        executed = True
        return _observed()

    with pytest.raises(
        TaskEvaluationSceneConfigurationNativeImportDriverError,
        match="native_import_input_invalid",
    ):
        execute_native_import_component(
            environment=environment,
            native_runner=native_runner,
        )
    assert executed is False
