"""Production factory invocation, byte qualification and native cleanup seams.

The installed test SDK below is a hermetic driver double, never physics or
runtime qualification evidence. No constructed runtime is injected at the
production builder seam.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import task_family_native_runtime as factory
from blueprint_pipeline.task_family_native_controls import qualify_task_family_site
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_task_family_native_adapter import execute_task_family_episode
from tests.test_task_evaluation_task_family_native_adapter import Policy, _runtime_contract

SDK_SOURCE = '''
import numpy as np
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_task_family_native_adapter import native_runtime_provenance

instances = []

class Environment:
    def __init__(self, render_mode):
        assert render_mode == "rgb_array"
        self.unwrapped = self
        self.close_count = 0
        self.time = 0
        self.position = 0
        self.actions = []
        self.robot_present = True
        self.robot_action_applied = False
        instances.append(self)

    def apply_native_task_contract(self, *, contract, asset_paths):
        self.contract = contract
        self.asset_paths = asset_paths
        self.runtime_identity_digest = contract["runtime_identity_digest"]
        self.engine_id = contract["task_state_binding"]["engine_id"]
        self.native_runtime_contract_digest = contract["contract_digest"]
        return {**native_runtime_provenance(contract),
            "runtime_identity_digest": self.runtime_identity_digest,
            "engine_id": self.engine_id, "native_contract_applied": True,
            "asset_digests": {a["asset_id"]: a["content_digest"] for a in contract["assets"]},
            "task_state_binding_digest": canonical_digest(contract["task_state_binding"]),
            "robot_configuration": contract["robot_configuration"]}

    def reset(self, *, seed):
        assert seed == self.contract["seed"]
        self.time = 0
        self.position = 0
        self.robot_present = True
        self.robot_action_applied = False

    def native_state(self):
        return {"source": "native_simulator_state", "simulation_time_seconds": self.time,
            "native_provenance": native_runtime_provenance(self.contract), "safety_violation": False,
            "robot_present": self.robot_present, "robot_action_applied": self.robot_action_applied,
            "asset_states": {"button1": {"joint_position_m": self.position,
                "native_button_sensor_active": self.position >= .01, "contact_forces_n": [[0, 0, 0]]}}}

    def policy_observation(self):
        return {"rgb": np.full((16, 16, 3), int(self.time), dtype=np.uint8), "proprioception": [self.position]}

    def step_native_action(self, action, *, duration_seconds):
        self.robot_action_applied = True
        self.actions.append(action)
        self.time += duration_seconds
        self.position += .02 * action[0] * duration_seconds

    def set_robot_enabled(self, *, enabled):
        self.robot_present = enabled

    def step_without_robot_action(self, *, duration_seconds):
        self.time += duration_seconds
        self.robot_action_applied = False

    def scripted_control_action(self, observation, *, step_index):
        return [1]

    def native_scene_readiness(self):
        return {"source": "native_simulator_state", "camera_measurement_source": "native_renderer",
            "native_provenance": native_runtime_provenance(self.contract), "robot_present": self.robot_present,
            "asset_digests": {a["asset_id"]: a["content_digest"] for a in self.contract["assets"]},
            "cameras": {camera["camera_id"]: {"pose_world": camera["pose_world"],
                "intrinsics": camera["intrinsics"], "resolution": camera["resolution"], "rendered": True,
                "rendered_scene_asset_ids": [a["asset_id"] for a in self.contract["assets"]
                    if a.get("semantic_role") == "scene_appearance"]} for camera in self.contract["cameras"]}}

    def close(self):
        self.close_count += 1
'''


def _sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_catalog(args, registration, *, qualification_changes=None):
    catalog_path = args["runtime_catalog_path"]
    qualification = {"schema_version": factory.QUALIFICATION_SCHEMA,
        "runtime_registration_digest": canonical_digest({k: v for k, v in registration.items() if k != "qualification"}),
        "decision": "qualified_for_bounded_simulation", "reviewed_by": "test-operator",
        "expires_at_epoch": time.time() + 3600,
        "checks": {key: True for key in factory._CHECKS}, "physical_validation": False,
        **(qualification_changes or {})}
    qualification["qualification_digest"] = canonical_digest(qualification, digest_field="qualification_digest")
    qualification_path = catalog_path.parent / "qualification.json"
    qualification_path.write_text(json.dumps(qualification))
    registration["qualification"] = {"path": "qualification.json", "sha256": _sha(qualification_path)}
    catalog = {"schema_version": factory.CATALOG_SCHEMA, "runtimes": [registration]}
    catalog["catalog_digest"] = canonical_digest(catalog, digest_field="catalog_digest")
    catalog_path.write_text(json.dumps(catalog))
    args["qualified_runtime_selection"] = {"runtime_id": registration["runtime_id"],
        "runtime_catalog_digest": catalog["catalog_digest"]}


@pytest.fixture
def installed_sdk(tmp_path, monkeypatch):
    module_name = "qualified_family_test_engine"
    source = tmp_path / (module_name + ".py")
    source.write_text(SDK_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    spec = SimpleNamespace(entry_point=module_name + ":Environment", kwargs={}, additional_wrappers=())
    calls = []

    def make(environment_id, **kwargs):
        calls.append((environment_id, kwargs))
        module = importlib.import_module(module_name)
        return module.Environment(**kwargs)

    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(spec=lambda _: spec, make=make))
    asset_root = tmp_path / "packet"
    asset_root.mkdir()
    asset = asset_root / "button.usd"
    asset.write_bytes(b"hermetic SDK asset bytes; not a physics qualification")
    native = _runtime_contract()
    native["task_spec"].update(maximum_action_steps=2, control_frequency_hz=1)
    native["assets"][0].update(content_digest=_sha(asset), bundle_relative_path="button.usd")
    native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")
    registration = {"runtime_id": "installed-test-engine", "factory_id": factory.GYMNASIUM_FACTORY,
        "runtime_identity_digest": native["runtime_identity_digest"], "engine_id": "engine1",
        "task_families": ["button_press"], "robot_interfaces": [native["robot_configuration"]],
        "policy_interfaces": ["task_family_policy_https.v1"], "gymnasium_environment_id": "TestNativeFamily-v1",
        "registration_module": module_name, "entry_point": spec.entry_point,
        "runtime_sources": [{"module": module_name, "sha256": _sha(source)}]}
    args = {"native_contract": native, "asset_root": asset_root, "runtime_catalog_path": tmp_path / "catalog.json"}
    _seal_catalog(args, registration)
    yield SimpleNamespace(args=args, registration=registration, source=source, asset=asset, calls=calls, spec=spec,
                          module=lambda: importlib.import_module(module_name))
    sys.modules.pop(module_name, None)


def test_registered_factory_constructs_applies_runs_scores_and_closes(installed_sdk, tmp_path):
    sdk = installed_sdk
    runtime = factory.build_task_family_native_runtime(**sdk.args)
    assert sdk.calls == [("TestNativeFamily-v1", {"render_mode": "rgb_array"})]
    engine = sdk.module().instances[0]
    assert engine.contract == sdk.args["native_contract"]
    assert engine.asset_paths == {"button1": str(sdk.asset)}
    receipt = execute_task_family_episode(runtime=runtime, policy=Policy(),
        contract=engine.contract["task_spec"]["task_success_contract"], maximum_steps=2, control_frequency_hz=1,
        output_dir=tmp_path / "episode", episode_id="registered-family", seed=0)
    assert receipt["status"] == "completed" and receipt["task_succeeded"]
    assert receipt["scoring_source"] == "deterministic_native_state"
    assert receipt["visual_evidence"]["status"] == "complete"
    assert receipt["physical_validation"] is False and engine.actions == [[1.0], [1.0]]
    assert engine.close_count == 1
    runtime.close()
    assert engine.close_count == 1


@pytest.mark.parametrize("change,reason", [
    ("catalog", "catalog_binding_mismatch"), ("asset", "asset_bytes_changed"),
    ("source", "sdk_source_changed"), ("entrypoint", "sdk_factory_identity_mismatch"),
    ("wrappers", "sdk_factory_identity_mismatch"), ("kwargs", "sdk_factory_arguments_mismatch"),
    ("selection_code", "selection_invalid"), ("asset_escape", "asset_path_invalid"),
])
def test_refuses_changed_or_untrusted_inputs_before_construction(installed_sdk, change, reason):
    sdk = installed_sdk
    if change == "catalog":
        sdk.args["qualified_runtime_selection"]["runtime_catalog_digest"] = "sha256:" + "0" * 64
    elif change == "asset":
        sdk.asset.write_bytes(b"changed")
    elif change == "source":
        sdk.source.write_text(SDK_SOURCE + "\nchanged = True\n")
    elif change == "entrypoint":
        sdk.spec.entry_point = "other_module:Environment"
    elif change == "wrappers":
        sdk.spec.additional_wrappers = ("unqualified-wrapper",)
    elif change == "kwargs":
        sdk.spec.kwargs = {"unqualified": True}
    elif change == "selection_code":
        sdk.args["qualified_runtime_selection"]["module"] = "untrusted"
    elif change == "asset_escape":
        native = sdk.args["native_contract"]
        native["assets"][0]["bundle_relative_path"] = "../button.usd"
        native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")
    with pytest.raises(ValueError, match=reason):
        factory.build_task_family_native_runtime(**sdk.args)
    assert sdk.calls == []


@pytest.mark.parametrize("change,reason", [
    ("expired", "qualification_invalid"), ("qualification_bytes", "qualification_bytes_changed"),
    ("qualification_registration", "qualification_invalid"), ("family", "family_not_qualified"),
    ("robot", "robot_interface_not_qualified"), ("identity", "identity_mismatch"),
    ("factory", "factory_unregistered"),
])
def test_requires_exact_operator_qualification(installed_sdk, change, reason):
    sdk = installed_sdk
    if change == "expired":
        _seal_catalog(sdk.args, sdk.registration, qualification_changes={"expires_at_epoch": 0})
    elif change == "qualification_bytes":
        path = sdk.args["runtime_catalog_path"].parent / "qualification.json"
        path.write_text(path.read_text() + " ")
    elif change == "qualification_registration":
        _seal_catalog(sdk.args, sdk.registration,
                      qualification_changes={"runtime_registration_digest": "sha256:" + "0" * 64})
    else:
        field, value = {"family": ("task_families", ["pouring"]), "robot": ("robot_interfaces", []),
            "identity": ("engine_id", "other-engine"), "factory": ("factory_id", "arbitrary_python")}[change]
        sdk.registration[field] = value
        _seal_catalog(sdk.args, sdk.registration)
    with pytest.raises(ValueError, match=reason):
        factory.build_task_family_native_runtime(**sdk.args)
    assert sdk.calls == []


def test_native_application_failure_closes_constructed_sdk(installed_sdk, monkeypatch):
    sdk = installed_sdk
    engine_type = sdk.module().Environment
    original = engine_type.apply_native_task_contract

    def wrong_application(self, **kwargs):
        receipt = original(self, **kwargs)
        receipt["task_state_binding_digest"] = "sha256:" + "0" * 64
        return receipt

    monkeypatch.setattr(engine_type, "apply_native_task_contract", wrong_application)
    with pytest.raises(ValueError, match="native_application_readback_mismatch"):
        factory.build_task_family_native_runtime(**sdk.args)
    assert sdk.module().instances[0].close_count == 1


@pytest.mark.parametrize("record", ["catalog", "qualification"])
@pytest.mark.parametrize("mode", [0o664, 0o646])
def test_runtime_factory_refuses_non_operator_writeable_trust_records(installed_sdk, record, mode):
    sdk = installed_sdk
    path = sdk.args["runtime_catalog_path"]
    if record == "qualification":
        path = path.parent / "qualification.json"
    path.chmod(mode)
    with pytest.raises(ValueError, match="operator_record_untrusted"):
        factory.build_task_family_native_runtime(**sdk.args)
    assert sdk.calls == []


def test_runtime_factory_refuses_foreign_owned_operator_catalog(installed_sdk, monkeypatch):
    sdk = installed_sdk
    original = factory.os.fstat

    def foreign_owner(descriptor):
        status = original(descriptor)
        return SimpleNamespace(st_mode=status.st_mode, st_size=status.st_size,
            st_uid=factory.os.getuid() + 1)

    monkeypatch.setattr(factory.os, "fstat", foreign_owner)
    with pytest.raises(ValueError, match="operator_record_untrusted"):
        factory.build_task_family_native_runtime(**sdk.args)
    assert sdk.calls == []


def test_native_provenance_failure_is_invalid_and_closes_sdk(installed_sdk, monkeypatch, tmp_path):
    sdk = installed_sdk
    runtime = factory.build_task_family_native_runtime(**sdk.args)
    engine = sdk.module().instances[0]
    original = engine.native_state

    def wrong_state():
        state = copy.deepcopy(original())
        state["native_provenance"]["scenario_cell_id"] = "another-cell"
        return state

    monkeypatch.setattr(engine, "native_state", wrong_state)
    receipt = execute_task_family_episode(runtime=runtime, policy=Policy(),
        contract=engine.contract["task_spec"]["task_success_contract"], maximum_steps=2,
        control_frequency_hz=1, output_dir=tmp_path / "episode", episode_id="bad-provenance", seed=0)
    assert receipt["status"] == "invalid" and not receipt["task_succeeded"]
    assert "readback_scene_binding_mismatch" in receipt["failure"]["reason"]
    assert engine.close_count == 1


@pytest.mark.parametrize("field,expected", [("native_runtime_contract_digest", "applied_runtime_contract_mismatch"),
                                           ("engine_id", "native_identity_readback_mismatch")])
def test_mid_episode_identity_change_is_invalid_and_closes_sdk(installed_sdk, monkeypatch, tmp_path, field, expected):
    sdk = installed_sdk
    runtime = factory.build_task_family_native_runtime(**sdk.args)
    engine = sdk.module().instances[0]
    original = engine.step_native_action

    def changed_contract(action, **kwargs):
        original(action, **kwargs)
        setattr(engine, field, "sha256:" + "0" * 64)

    monkeypatch.setattr(engine, "step_native_action", changed_contract)
    receipt = execute_task_family_episode(runtime=runtime, policy=Policy(),
        contract=engine.contract["task_spec"]["task_success_contract"], maximum_steps=2,
        control_frequency_hz=1, output_dir=tmp_path / "episode", episode_id="changed-contract", seed=0)
    assert receipt["status"] == "invalid"
    assert expected in receipt["failure"]["reason"]
    assert engine.close_count == 1


def test_site_qualification_constructs_fresh_native_probe_and_controls(installed_sdk, tmp_path):
    sdk = installed_sdk
    receipt = qualify_task_family_site(**sdk.args, output_dir=tmp_path / "site-qualification")
    assert receipt["status"] == "passed" and receipt["candidate_policy_queried"] is False
    assert set(receipt["controls"]) == {"robot_free_probe", "zero_action_negative", "deterministic_scripted_positive"}
    assert len(sdk.calls) == 3 and all(engine.close_count == 1 for engine in sdk.module().instances)
    probe = receipt["controls"]["robot_free_probe"]
    assert all(state["robot_present"] is False for state in probe["native_state_trace"])
    assert all(state["robot_action_applied"] is False for state in probe["native_state_trace"])
    assert not probe["episode"]["task_succeeded"]
    assert receipt["controls"]["deterministic_scripted_positive"]["episode"]["task_succeeded"]
    assert all(row["episode"]["visual_evidence"]["status"] == "complete" for row in receipt["controls"].values())
    assert json.loads((tmp_path / "site-qualification" / "task_family_native_site_qualification.v1.json").read_text()) == receipt


def test_site_qualification_consumes_sealed_execution_bounds_without_mutating_native_contract(installed_sdk, tmp_path):
    sdk = installed_sdk
    native = sdk.args["native_contract"]
    native["task_spec"].pop("maximum_action_steps")
    native["task_spec"].pop("control_frequency_hz")
    native["contract_digest"] = canonical_digest(native, digest_field="contract_digest")
    original = copy.deepcopy(native)
    receipt = qualify_task_family_site(**sdk.args, output_dir=tmp_path / "site-qualification",
        maximum_steps=2, control_frequency_hz=1)
    assert receipt["status"] == "passed" and native == original
    assert all(engine.contract == original for engine in sdk.module().instances)


def test_site_qualification_refuses_conflicting_execution_bounds(installed_sdk, tmp_path):
    sdk = installed_sdk
    with pytest.raises(ValueError, match="episode_bounds_mismatch"):
        qualify_task_family_site(**sdk.args, output_dir=tmp_path / "site-qualification",
            maximum_steps=3, control_frequency_hz=1)
    assert sdk.calls == []


def test_site_qualification_preserves_previous_retained_evidence(installed_sdk, tmp_path):
    sdk = installed_sdk
    output = tmp_path / "site-qualification"
    receipt = qualify_task_family_site(**sdk.args, output_dir=output)
    retained = (output / "task_family_native_site_qualification.v1.json").read_bytes()
    with pytest.raises(ValueError, match="output_exists"):
        qualify_task_family_site(**sdk.args, output_dir=output)
    assert len(sdk.calls) == 3 and receipt["status"] == "passed"
    assert (output / "task_family_native_site_qualification.v1.json").read_bytes() == retained


@pytest.mark.parametrize("change", ["robot_not_removed", "unactuated_step_applies_action", "camera", "safety", "positive"])
def test_site_qualification_fails_native_probe_controls_or_readiness(installed_sdk, monkeypatch, tmp_path, change):
    sdk = installed_sdk
    engine_type = sdk.module().Environment
    if change == "robot_not_removed":
        monkeypatch.setattr(engine_type, "set_robot_enabled", lambda self, **_: None)
    elif change == "unactuated_step_applies_action":
        monkeypatch.setattr(engine_type, "step_without_robot_action",
            lambda self, **kwargs: self.step_native_action([1], **kwargs))
    elif change == "camera":
        original = engine_type.native_scene_readiness

        def no_camera(self):
            row = original(self)
            row["cameras"]["camera1"]["rendered"] = False
            return row

        monkeypatch.setattr(engine_type, "native_scene_readiness", no_camera)
    elif change == "safety":
        original = engine_type.native_state

        def unsafe(self):
            row = original(self)
            row["safety_violation"] = True
            return row

        monkeypatch.setattr(engine_type, "native_state", unsafe)
    elif change == "positive":
        monkeypatch.setattr(engine_type, "scripted_control_action", lambda self, *_, **__: [0])
    receipt = qualify_task_family_site(**sdk.args, output_dir=tmp_path / "site-qualification")
    assert receipt["status"] == "failed" and receipt["failure"]
    assert receipt["candidate_policy_queried"] is False
    assert all(engine.close_count == 1 for engine in sdk.module().instances)
