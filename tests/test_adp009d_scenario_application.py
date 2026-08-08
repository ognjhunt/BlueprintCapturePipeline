from __future__ import annotations

import copy
import hashlib
import json
import math
from types import SimpleNamespace
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_scenario_application import (
    APPLIED_PARAMETER_IDS,
    PARAMETER_RUNTIME_TARGETS,
    ScenarioApplicationError,
    apply_scenario_instance,
    build_scenario_application_plan,
    verify_scenario_application_receipt,
)
from blueprint_pipeline.adp009d_isaac_runtime import (
    _IsaacNativeScenarioBackend,
    _environment_flag_enabled,
    _scenario_object_center_z,
    _scenario_object_top_z,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


SUITE_DIGEST = "sha256:" + "1" * 64
HARNESS_DIGEST = "sha256:" + "2" * 64
INSTANCE_DIGEST_FIELD = "instance_digest"


def _instance(*, factor_id: str | None = None, cousin_id: str = "approved_can") -> dict:
    parameters = {
        "camera_blur_sigma": 0.0,
        "camera_exposure_scale": 1.0,
        "camera_latency_frames": 0,
        "camera_noise_std": 0.0,
        "external_camera_extrinsic_dx_m": 0.0,
        "external_camera_extrinsic_dy_m": 0.0,
        "external_camera_extrinsic_dz_m": 0.0,
        "external_camera_focal_scale": 1.0,
        "light_color_temperature_kelvin": 5000.0,
        "light_direction_yaw_degrees": 0.0,
        "light_intensity_scale": 1.0,
        "object_contact_offset_m": 0.005,
        "object_dynamic_friction": 0.4,
        "object_height_m": 0.1694279937744141,
        "object_mass_kg": 0.355,
        "object_radius_m": 0.031094726014345042,
        "object_rest_offset_m": 0.0,
        "object_restitution": 0.1,
        "object_start_x_m": 3.4681748,
        "object_start_y_m": -3.3100837,
        "object_start_z_m": 0.5264650138348479,
        "object_static_friction": 0.5,
        "object_yaw_degrees": 0.0,
        "robot_base_x_m": 3.4681748,
        "robot_base_y_m": -2.8100837,
        "robot_base_yaw_degrees": -90.0,
        "target_x_m": 3.750152333333333,
        "target_y_m": -3.4074919,
        "target_z_m": 0.5264650138348479,
        "wrist_camera_extrinsic_dx_m": 0.0,
        "wrist_camera_extrinsic_dy_m": 0.0,
        "wrist_camera_extrinsic_dz_m": 0.0,
    }
    values = {
        "object_start_y_m": -3.3000837,
        "object_yaw_degrees": 10.0,
        "light_intensity_scale": 1.25,
        "light_color_temperature_kelvin": 6000.0,
        "external_camera_extrinsic_dx_m": 0.005,
        "wrist_camera_extrinsic_dx_m": -0.005,
        "object_mass_kg": 0.39,
        "object_dynamic_friction": 0.5,
    }
    records = []
    if factor_id is not None:
        parameters[factor_id] = values.get(factor_id, 7.0)
        records.append(
            {
                "parameter_id": factor_id,
                "resolved_value": parameters[factor_id],
                "runtime_target": PARAMETER_RUNTIME_TARGETS.get(
                    factor_id, "unsupported.native.target"
                ),
                "validity": {
                    "invalid_behavior": "reject_instance_fail_closed",
                    "native_probe_required": True,
                },
            }
        )
    cousin = _bindings()[cousin_id]
    value = {
        "schema_version": "adp009d_scenario_instance.v1",
        "program_id": "arm-decision-proof-v1",
        "suite_digest": SUITE_DIGEST,
        "harness_digest": HARNESS_DIGEST,
        "cell_id": "cell_1",
        "template_id": "template_1",
        "family": (
            "visual_material_cousin"
            if cousin_id == "adp009d_visual_material_cousin"
            else "geometric_cousin"
            if cousin_id == "adp009d_geometric_cousin"
            else {
                "object_start_y_m": "placement_approach",
                "object_yaw_degrees": "placement_approach",
                "light_intensity_scale": "illumination",
                "light_color_temperature_kelvin": "illumination",
                "external_camera_extrinsic_dx_m": "camera_sensor",
                "wrist_camera_extrinsic_dx_m": "camera_sensor",
                "object_mass_kg": "physics",
                "object_dynamic_friction": "physics",
            }.get(factor_id, "canonical")
        ),
        "partition": "qualification",
        "scored": True,
        "seed": 2026080600,
        "cell_seed_digest": "sha256:" + "3" * 64,
        "cousin_id": cousin_id,
        "cousin_digest": cousin["asset_digest"],
        "cousin_static_validation_receipt_digest": cousin.get(
            "static_validation_receipt_digest"
        ),
        "resolved_parameters": parameters,
        "factor_records": records,
        "required_controls": [
            "deterministic_scripted_positive",
            "zero_action_negative",
        ],
        "policy_neutral": True,
        "caller_asserted_success": False,
        INSTANCE_DIGEST_FIELD: "",
    }
    value[INSTANCE_DIGEST_FIELD] = canonical_digest(
        value, digest_field=INSTANCE_DIGEST_FIELD
    )
    return value


def _bindings() -> dict[str, dict]:
    return {
        "approved_can": {
            "cousin_id": "approved_can",
            "cousin_type": "canonical",
            "admission_status": "canonical_anchor",
            "asset_digest": "sha256:" + "a" * 64,
            "native_asset_path": "/runtime/assets/approved_can.usda",
            "native_asset_sha256": "sha256:" + "1" * 64,
            "static_validation_receipt_digest": None,
        },
        "adp009d_visual_material_cousin": {
            "cousin_id": "adp009d_visual_material_cousin",
            "cousin_type": "visual_material",
            "admission_status": "admitted_for_control_execution",
            "asset_digest": "sha256:" + "b" * 64,
            "native_asset_path": "/runtime/assets/visual.usda",
            "native_asset_sha256": "sha256:" + "2" * 64,
            "static_validation_receipt_digest": "sha256:" + "c" * 64,
        },
        "adp009d_geometric_cousin": {
            "cousin_id": "adp009d_geometric_cousin",
            "cousin_type": "geometric",
            "admission_status": "admitted_for_control_execution",
            "asset_digest": "sha256:" + "d" * 64,
            "native_asset_path": "/runtime/assets/geometric.usda",
            "native_asset_sha256": "sha256:" + "3" * 64,
            "static_validation_receipt_digest": "sha256:" + "e" * 64,
        },
    }


def _expected(instance: dict) -> dict[str, str]:
    return {instance["cell_id"]: instance["instance_digest"]}


class FakeNativeBackend:
    def __init__(self) -> None:
        self.parameters: dict[str, float] = {}
        self.cousin: dict | None = None
        self.readback_overrides: dict[str, float] = {}
        self.supported = set(APPLIED_PARAMETER_IDS)

    def runtime_identity(self) -> dict:
        return {
            "backend": "hermetic_native_simulator_fake",
            "version": "1",
            "configuration_digest": "sha256:" + "f" * 64,
        }

    def supported_parameter_ids(self) -> set[str]:
        return set(self.supported)

    def apply_parameter(self, parameter_id: str, value: float) -> None:
        self.parameters[parameter_id] = value

    def apply_cousin(self, binding: dict) -> None:
        self.cousin = copy.deepcopy(binding)

    def commit_application(self) -> None:
        return None

    def read_parameter(self, parameter_id: str) -> dict:
        return {
            "value": self.readback_overrides.get(
                parameter_id, self.parameters[parameter_id]
            ),
            "source": (
                "native_simulator_state"
                if parameter_id.startswith("object_start_")
                or parameter_id == "object_yaw_degrees"
                else "native_simulator_configuration"
            ),
            "native_path": f"native/{parameter_id}",
        }

    def read_cousin(self) -> dict:
        assert self.cousin is not None
        return {
            "cousin_id": self.cousin["cousin_id"],
            "asset_digest": self.cousin["asset_digest"],
            "native_asset_sha256": self.cousin["native_asset_sha256"],
            "source": "native_simulator_configuration",
            "native_path": "native/object_asset",
        }


@pytest.mark.parametrize(
    "factor_id",
    [
        "object_start_y_m",
        "object_yaw_degrees",
        "light_intensity_scale",
        "light_color_temperature_kelvin",
        "external_camera_extrinsic_dx_m",
        "wrist_camera_extrinsic_dx_m",
        "object_mass_kg",
        "object_dynamic_friction",
    ],
)
def test_applies_each_frozen_factor_and_receipts_native_readback(
    tmp_path: Path, factor_id: str
) -> None:
    instance = _instance(factor_id=factor_id)
    backend = FakeNativeBackend()

    receipt = apply_scenario_instance(
        instance,
        backend=backend,
        admitted_cousins=_bindings(),
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests=_expected(instance),
        receipt_path=tmp_path / "receipt.json",
    )

    rows = {row["parameter_id"]: row for row in receipt["applied_parameters"]}
    assert rows[factor_id]["requested_value"] == instance["resolved_parameters"][factor_id]
    assert rows[factor_id]["native_readback"]["value"] == rows[factor_id]["requested_value"]
    assert rows[factor_id]["runtime_target"] == PARAMETER_RUNTIME_TARGETS[factor_id]
    assert receipt["status"] == "applied_and_native_readback_verified"
    assert receipt["instance_digest"] == instance["instance_digest"]
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    assert (tmp_path / "receipt.json").is_file()


@pytest.mark.parametrize(
    "cousin_id,cousin_type",
    [
        ("adp009d_visual_material_cousin", "visual_material"),
        ("adp009d_geometric_cousin", "geometric"),
    ],
)
def test_applies_each_admitted_cousin_and_reads_native_asset_identity(
    cousin_id: str, cousin_type: str
) -> None:
    instance = _instance(cousin_id=cousin_id)
    if cousin_type == "geometric":
        instance["resolved_parameters"].update(
            {
                "object_height_m": 0.1795936694,
                "object_radius_m": 0.0590799794 / 2.0,
                "object_mass_kg": 0.33961075,
            }
        )
        instance["instance_digest"] = canonical_digest(
            instance, digest_field="instance_digest"
        )

    receipt = apply_scenario_instance(
        instance,
        backend=FakeNativeBackend(),
        admitted_cousins=_bindings(),
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests=_expected(instance),
    )

    assert receipt["object_asset"]["requested_binding"]["cousin_type"] == cousin_type
    assert receipt["object_asset"]["native_readback"]["cousin_id"] == cousin_id
    assert receipt["object_asset"]["native_readback"]["asset_digest"] == instance["cousin_digest"]


def test_fails_closed_when_instance_requests_unsupported_parameter() -> None:
    instance = _instance(factor_id="unregistered_parameter")

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(instance),
        )

    assert "scenario_application_factor_unsupported:unregistered_parameter" in exc.value.errors


def test_fails_closed_when_native_backend_lacks_a_required_parameter() -> None:
    backend = FakeNativeBackend()
    backend.supported.remove("object_mass_kg")

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(),
            backend=backend,
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(_instance()),
        )

    assert "scenario_application_native_parameter_unsupported:object_mass_kg" in exc.value.errors


def test_fails_closed_when_native_readback_differs_from_requested_value() -> None:
    backend = FakeNativeBackend()
    backend.readback_overrides["object_dynamic_friction"] = 0.401

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(),
            backend=backend,
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(_instance()),
        )

    assert "scenario_application_native_readback_mismatch:object_dynamic_friction" in exc.value.errors


def test_fails_closed_on_unadmitted_or_digest_mismatched_cousin() -> None:
    bindings = _bindings()
    bindings["adp009d_visual_material_cousin"]["admission_status"] = "static_candidate"

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(cousin_id="adp009d_visual_material_cousin"),
            backend=FakeNativeBackend(),
            admitted_cousins=bindings,
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(
                _instance(cousin_id="adp009d_visual_material_cousin")
            ),
        )

    assert "scenario_application_cousin_not_admitted" in exc.value.errors

    instance = _instance(cousin_id="adp009d_geometric_cousin")
    instance["cousin_digest"] = "sha256:" + "0" * 64
    instance[INSTANCE_DIGEST_FIELD] = canonical_digest(
        instance, digest_field=INSTANCE_DIGEST_FIELD
    )
    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(instance),
        )
    assert "scenario_application_cousin_digest_mismatch" in exc.value.errors


def test_fails_closed_on_instance_or_frozen_identity_mismatch() -> None:
    instance = _instance()
    instance["resolved_parameters"]["object_mass_kg"] = 0.39

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(instance),
        )
    assert "scenario_application_instance_digest_mismatch" in exc.value.errors

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            _instance(),
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest="sha256:" + "9" * 64,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=_expected(_instance()),
        )
    assert "scenario_application_suite_digest_mismatch" in exc.value.errors


def test_rejects_self_consistent_forgery_not_bound_by_materialization() -> None:
    instance = _instance(factor_id="object_mass_kg")
    frozen_bindings = _expected(instance)
    instance["resolved_parameters"]["object_mass_kg"] = 0.32
    instance["factor_records"][0]["resolved_value"] = 0.32
    instance["instance_digest"] = canonical_digest(
        instance, digest_field="instance_digest"
    )

    with pytest.raises(ScenarioApplicationError) as exc:
        apply_scenario_instance(
            instance,
            backend=FakeNativeBackend(),
            admitted_cousins=_bindings(),
            expected_suite_digest=SUITE_DIGEST,
            expected_harness_digest=HARNESS_DIGEST,
            expected_instance_digests=frozen_bindings,
        )

    assert "scenario_application_materialization_binding_mismatch" in exc.value.errors


def test_applies_immutable_canonical_anchor() -> None:
    instance = _instance()

    receipt = apply_scenario_instance(
        instance,
        backend=FakeNativeBackend(),
        admitted_cousins=_bindings(),
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests=_expected(instance),
    )

    assert receipt["status"] == "applied_and_native_readback_verified"
    assert len(receipt["applied_parameters"]) == len(APPLIED_PARAMETER_IDS)
    assert receipt["object_asset"]["native_readback"]["cousin_id"] == "approved_can"


@pytest.mark.parametrize(
    "first_factor,second_factor",
    [
        ("light_intensity_scale", "object_mass_kg"),
        ("external_camera_extrinsic_dx_m", "object_dynamic_friction"),
    ],
)
def test_applies_each_held_out_composition_atomically(
    first_factor: str, second_factor: str
) -> None:
    instance = _instance(factor_id=first_factor)
    second = _instance(factor_id=second_factor)
    instance["family"] = "held_out_composed"
    instance["factor_records"].append(second["factor_records"][0])
    instance["resolved_parameters"][second_factor] = second["resolved_parameters"][
        second_factor
    ]
    instance["instance_digest"] = canonical_digest(
        instance, digest_field="instance_digest"
    )

    receipt = apply_scenario_instance(
        instance,
        backend=FakeNativeBackend(),
        admitted_cousins=_bindings(),
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests=_expected(instance),
    )

    rows = {row["parameter_id"]: row for row in receipt["applied_parameters"]}
    assert rows[first_factor]["native_readback"]["value"] == rows[first_factor][
        "requested_value"
    ]
    assert rows[second_factor]["native_readback"]["value"] == rows[second_factor][
        "requested_value"
    ]
    assert receipt["instance_digest"] == instance["instance_digest"]


def test_checked_in_frozen_suite_has_no_unsupported_runtime_target() -> None:
    suite = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/arm_decision_proof_v1/manifests/adp009d_scenario_suite.v1.json"
        ).read_text(encoding="utf-8")
    )

    factors = {row["parameter_id"]: row for row in suite["factors"]}
    assert set(factors) == set(PARAMETER_RUNTIME_TARGETS) & set(factors)
    assert set(factors) == {
        "object_start_y_m",
        "object_yaw_degrees",
        "light_intensity_scale",
        "light_color_temperature_kelvin",
        "external_camera_extrinsic_dx_m",
        "wrist_camera_extrinsic_dx_m",
        "object_mass_kg",
        "object_dynamic_friction",
    }
    for parameter_id, factor in factors.items():
        assert factor["runtime_target"] == PARAMETER_RUNTIME_TARGETS[parameter_id]
        assert factor["validity"]["native_probe_required"] is True


class _FakePhysxRootView:
    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self.masses = torch_module.tensor([[0.355]], dtype=torch_module.float32)
        self.materials = torch_module.tensor(
            [[[0.5, 0.4, 0.1]]], dtype=torch_module.float32
        )

    def get_masses(self):
        return self.masses

    def get_material_properties(self):
        return self.materials

    def set_material_properties(self, materials, env_ids) -> None:
        import warp as wp

        assert wp.to_torch(env_ids).tolist() == [0]
        self.materials = wp.to_torch(materials).clone()


class _FakeIsaacCan:
    def __init__(self, torch_module) -> None:
        self.data = SimpleNamespace(
            root_pose_w=torch_module.tensor(
                [[3.4681748, -3.3100837, 0.526465, 1.0, 0.0, 0.0, 0.0]],
                dtype=torch_module.float32,
            )
        )
        self.root_view = _FakePhysxRootView(torch_module)

    def write_root_pose_to_sim_index(self, *, root_pose) -> None:
        self.data.root_pose_w = root_pose.clone()

    def set_masses_index(self, *, masses) -> None:
        self.root_view.masses = masses.clone()


class _FakeIsaacEnvironment:
    def __init__(self, scene: dict) -> None:
        self.unwrapped = SimpleNamespace(scene=scene, device="cpu")
        self.reset_seeds: list[int] = []

    def reset(self, *, seed: int):
        self.reset_seeds.append(seed)
        return {}, {}


def test_isaac_backend_applies_and_reads_all_native_families(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    wp = pytest.importorskip("warp")
    wp.init()
    factors = [
        "object_start_y_m",
        "object_yaw_degrees",
        "light_intensity_scale",
        "light_color_temperature_kelvin",
        "external_camera_extrinsic_dx_m",
        "wrist_camera_extrinsic_dx_m",
        "object_mass_kg",
        "object_dynamic_friction",
    ]
    instance = _instance(factor_id=factors[0])
    instance["family"] = "held_out_composed"
    for factor_id in factors[1:]:
        other = _instance(factor_id=factor_id)
        instance["factor_records"].append(other["factor_records"][0])
        instance["resolved_parameters"][factor_id] = other["resolved_parameters"][
            factor_id
        ]
    native_asset = tmp_path / "native.usda"
    native_asset.write_text('#usda 1.0\ndef Xform "canned_beverage" {}\n')
    native_sha256 = "sha256:" + hashlib.sha256(native_asset.read_bytes()).hexdigest()
    bindings = _bindings()
    bindings["approved_can"]["native_asset_path"] = native_asset.name
    bindings["approved_can"]["native_asset_sha256"] = native_sha256
    instance["instance_digest"] = canonical_digest(
        instance, digest_field="instance_digest"
    )
    plan = build_scenario_application_plan(
        instance,
        admitted_cousins=bindings,
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests=_expected(instance),
    )
    fake_can = _FakeIsaacCan(torch)
    external_cfg = SimpleNamespace(offset=SimpleNamespace(pos=(1.0, 2.0, 3.0)))
    wrist_cfg = SimpleNamespace(offset=SimpleNamespace(pos=(0.1, 0.2, 0.3)))
    light_cfg = SimpleNamespace(
        intensity=1500.0,
        enable_color_temperature=False,
        color_temperature=5000.0,
    )
    object_definition = SimpleNamespace(
        object_cfg=SimpleNamespace(
            spawn=SimpleNamespace(usd_path=str(native_asset))
        )
    )
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdLux = pytest.importorskip("pxr.UsdLux")
    stage = Usd.Stage.CreateInMemory()
    object_prim = stage.DefinePrim("/World/envs/env_0/approved_can", "Xform")
    object_prim.GetReferences().AddReference(str(native_asset), "/canned_beverage")
    fake_can.root_view.prim_paths = ["/World/envs/env_0/approved_can"]
    light_prim = UsdLux.DomeLight.Define(stage, "/World/Light")
    light_prim.CreateIntensityAttr(1875.0)
    light_prim.CreateEnableColorTemperatureAttr(True)
    light_prim.CreateColorTemperatureAttr(6000.0)
    external_prim = UsdGeom.Camera.Define(stage, "/World/ExternalCamera")
    wrist_prim = UsdGeom.Camera.Define(stage, "/World/WristCamera")
    UsdGeom.XformCommonAPI(external_prim).SetTranslate((1.0, 2.005, 3.0))
    UsdGeom.XformCommonAPI(wrist_prim).SetTranslate((0.095, 0.2, 0.3))
    env = _FakeIsaacEnvironment(
        {
            "approved_can": fake_can,
            "external_camera": SimpleNamespace(_sensor_prims=[external_prim]),
            "wrist_camera": SimpleNamespace(_sensor_prims=[wrist_prim]),
        }
    )
    backend = _IsaacNativeScenarioBackend(
        runtime=tmp_path,
        env=env,
        torch=torch,
        native_configuration={
            "approved_can_definition": object_definition,
            "light_definition": SimpleNamespace(spawner_cfg=light_cfg),
            "external_camera_cfg": external_cfg,
            "wrist_camera_cfg": wrist_cfg,
            "canonical_external_offset_robot": external_cfg.offset.pos,
            "canonical_wrist_offset_robot": wrist_cfg.offset.pos,
            "selected_object_path": native_asset,
            "live_stage": stage,
        },
        plan=plan,
    )

    receipt = apply_scenario_instance(
        instance,
        backend=backend,
        admitted_cousins=bindings,
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests=_expected(instance),
    )

    assert receipt["status"] == "applied_and_native_readback_verified"
    assert float(fake_can.root_view.masses[0, 0]) == pytest.approx(0.39)
    assert float(fake_can.root_view.materials[0, 0, 1]) == pytest.approx(0.5)
    assert light_prim.GetIntensityAttr().Get() == pytest.approx(1875.0)
    assert light_prim.GetColorTemperatureAttr().Get() == pytest.approx(6000.0)
    assert env.reset_seeds == [2026080600]
    assert receipt["object_asset"]["native_readback"][
        "native_asset_sha256"
    ] == native_sha256
    assert str(native_asset.resolve()) in receipt["object_asset"]["native_readback"][
        "composition_layers"
    ]


@pytest.mark.parametrize("yaw_degrees", [0.0, 10.0, -10.0])
def test_isaac_backend_writes_and_reads_pinned_xyzw_yaw(yaw_degrees: float) -> None:
    torch = pytest.importorskip("torch")
    fake_can = _FakeIsaacCan(torch)
    backend = object.__new__(_IsaacNativeScenarioBackend)
    backend._torch = torch
    backend._env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu"))
    backend._can = fake_can
    backend._values = {
        "object_start_x_m": 3.4681748,
        "object_start_y_m": -3.3100837,
        "object_start_z_m": 0.526465,
        "object_yaw_degrees": yaw_degrees,
    }

    backend._write_object_pose()
    pose = [float(value) for value in fake_can.data.root_pose_w[0]]

    half = math.radians(yaw_degrees) / 2.0
    assert pose[3:7] == pytest.approx(
        [0.0, 0.0, math.sin(half), math.cos(half)], abs=1e-7
    )
    assert backend.read_parameter("object_yaw_degrees")["value"] == pytest.approx(
        yaw_degrees, abs=1e-5
    )


def test_post_reset_verification_rejects_native_state_drift(tmp_path: Path) -> None:
    instance = _instance(factor_id="object_mass_kg")
    admitted = _bindings()
    backend = FakeNativeBackend()
    receipt = apply_scenario_instance(
        instance,
        backend=backend,
        admitted_cousins=admitted,
        expected_suite_digest=SUITE_DIGEST,
        expected_harness_digest=HARNESS_DIGEST,
        expected_instance_digests={instance["cell_id"]: instance["instance_digest"]},
        receipt_path=tmp_path / "receipt.json",
    )
    verify_scenario_application_receipt(receipt, backend=backend)

    backend.parameters["object_mass_kg"] = 9.0
    with pytest.raises(
        ScenarioApplicationError,
        match="scenario_application_native_readback_mismatch:object_mass_kg",
    ):
        verify_scenario_application_receipt(receipt, backend=backend)


def test_geometric_cousin_height_drives_live_visibility_bounds() -> None:
    canonical = {"object_start_z_m": 0.526465, "object_height_m": 0.169335}
    geometric = {"object_start_z_m": 0.526465, "object_height_m": 0.17959}

    assert _scenario_object_center_z(canonical) == pytest.approx(0.6111325)
    assert _scenario_object_top_z(canonical) == pytest.approx(0.6958)
    assert _scenario_object_center_z(geometric) == pytest.approx(0.61626)
    assert _scenario_object_top_z(geometric) == pytest.approx(0.706055)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes "])
def test_controls_environment_true_spellings_share_one_parser(value: str) -> None:
    assert _environment_flag_enabled(value) is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_controls_environment_false_spellings_share_one_parser(
    value: str | None,
) -> None:
    assert _environment_flag_enabled(value) is False
