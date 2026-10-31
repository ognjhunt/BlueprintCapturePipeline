"""Real retained source/USD bindings, fake native stepper: no GPU/policy claims."""
from __future__ import annotations

from copy import deepcopy
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_native_smoke as smoke
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime
from tests import test_deployment_embodiment_runtime_configuration as runtime_fixtures

configured = runtime_fixtures.configured


@pytest.fixture
def inputs(configured, monkeypatch, tmp_path):
    case = configured()
    runtime_fixtures._fake_sdk_from_urdf(monkeypatch)
    prepared = tmp_path / "prepared"
    importer.import_source_asset(case.output, prepared, fix_base=True,
                                 station_parameters=case.record["station_parameters"])
    binding = runtime.bind_prepared_configuration(case.output, prepared, "https://assets.example/robot.usda")
    return SimpleNamespace(binding=binding, model=prepared / "robot.usda", configuration=case.output,
                           output=tmp_path / "diagnostic")


class Stepper:
    def __init__(self, robot_row, *, fault=None, clock=None):
        self.row, self.fault, self.clock = robot_row, fault, clock
        self.joint_names = list(robot_row["joint_reset_positions_rad"])
        if fault == "missing_joint":
            self.joint_names.pop()
        self.q = dict(robot_row["joint_reset_positions_rad"])
        self.commands, self.closed = [], False
        self.runtime_readback = {"articulation_class": "test.FakeNativeStepper", "device": "cpu-test-fixture"}

    def reset(self, positions):
        self.q = dict(positions)

    def step(self, targets):
        self.commands.append(dict(targets))
        if self.fault != "stuck":
            self.q = dict(targets)
        if self.fault == "coupled_grippers" and targets["left_finger"] != self.row["joint_reset_positions_rad"]["left_finger"]:
            self.q["right_finger"] -= .01
        if self.fault == "coupled_arms" and targets["left_joint"] != 0:
            self.q["right_joint"] += .01
        if self.fault == "overshoot" and targets["left_joint"] != 0:
            self.q["left_joint"] *= 10
        if self.fault == "transient_overshoot" and len(self.commands) == 31:
            self.q["left_joint"] *= 10
        if self.fault == "transient_runaway" and len(self.commands) == 1:
            self.q["left_joint"] = 2.0
        if self.clock is not None:
            self.clock.value += .01

    def positions(self):
        if self.fault == "nonfinite":
            return {**self.q, self.joint_names[0]: float("nan")}
        if self.fault == "reset_error":
            return {name: value + .01 for name, value in self.q.items()}
        return dict(self.q)

    def close(self):
        self.closed = True


def execute(inputs, *, fault=None, clock=None, **kwargs):
    made = []

    def factory(row):
        made.append(Stepper(row, fault=fault, clock=clock))
        return made[-1]

    result = smoke.run_native_controller_smoke(
        inputs.binding, inputs.model, inputs.output, configuration_dir=inputs.configuration,
        backend_factory=factory, **({"now": lambda: clock.value} if clock is not None else {}), **kwargs)
    assert json.loads((inputs.output / smoke.FILENAME).read_text()) == result
    assert result["candidate_policy_queried"] is False
    assert result["task_executed"] is False
    assert result["physical_success_proven"] is False
    assert result["collision_behavior_qualified"] is False
    assert result["claim_ceiling"] == "native_controller_diagnostic_only"
    assert inputs.model.is_file()  # Failed diagnostics retain the imported asset.
    if made:
        assert made[0].closed
    return result, made


def test_reset_both_arm_channels_and_independent_grippers_use_retained_binding(inputs):
    result, made = execute(inputs)
    assert result["status"] == "passed", result["reasons"]
    assert result["model_sha256"] == inputs.binding["model_record"]["sha256"]
    assert result["embodiment_digest"] == inputs.binding["robot_embodiment"]["embodiment_digest"]
    assert result["source_bundle_digest"] == inputs.binding["model_record"]["source_import_provenance"]["source_bundle_digest"]
    assert result["reset"]["passed"] and result["final_reset"]["passed"]
    assert [(row["kind"], row["name"]) for row in result["probes"]] == [
        ("arm", "left"), ("arm", "right"), ("gripper", "left_gripper"), ("gripper", "right_gripper")]
    assert all(row["status"] == "passed" for row in result["probes"])
    assert 0 < result["physics_ticks"] <= smoke.MAX_TICKS
    row = made[0].row
    assert row["base_pose_world"]["orientation_xyzw"] == [0., 0., 0., 1.]
    assert row["enabled_self_collisions"] is False
    assert "task_spec" not in row and "policy" not in row
    assert result["gravity_m_s2"] == [0., 0., 0.]


@pytest.mark.parametrize("fault", ["stuck", "missing_joint", "nonfinite", "reset_error", "coupled_grippers",
                                  "coupled_arms", "overshoot", "transient_overshoot", "transient_runaway"])
def test_native_readback_failures_are_separate_diagnostics_and_keep_model(inputs, fault):
    result, _ = execute(inputs, fault=fault)
    assert result["status"] == "failed"
    assert result["reasons"]
    if fault == "coupled_grippers":
        probes = {row["name"]: row for row in result["probes"]}
        assert probes["left_gripper"]["status"] == "failed"
        assert probes["right_gripper"]["status"] == "passed"


@pytest.mark.parametrize("change", ["model", "import_identity", "manifest", "configuration_identity"])
def test_native_work_never_starts_for_changed_model_source_or_manifest(inputs, change):
    if change == "model":
        inputs.model.write_text("#usda 1.0\nchanged")
    elif change == "import_identity":
        inputs.binding["model_record"]["source_import_provenance"]["import_digest"] = "sha256:" + "0" * 64
    elif change == "configuration_identity":
        inputs.binding["configuration_digest"] = "sha256:" + "0" * 64
    else:
        manifest = deepcopy(inputs.binding["robot_embodiment"])
        manifest.pop("embodiment_digest")
        manifest["actuator"]["stiffness"] += 1
        inputs.binding["robot_embodiment"] = smoke.validate_embodiment_manifest(manifest)
    result, made = execute(inputs)
    assert result["status"] == "failed"
    assert made == []


@pytest.mark.parametrize("limit", ["ticks", "wall", "expired"])
def test_diagnostic_budget_stops_stepping_and_closes_native_handles(inputs, monkeypatch, limit):
    clock = SimpleNamespace(value=0.)
    kwargs = {}
    if limit == "ticks":
        monkeypatch.setattr(smoke, "MAX_TICKS", 5)
    else:
        kwargs["wall_budget_seconds"] = .025 if limit == "wall" else 0
    result, made = execute(inputs, clock=clock, **kwargs)
    assert result["status"] == "failed"
    assert result["physics_ticks"] <= (5 if limit == "ticks" else 3)
    assert any("budget_exhausted" in reason for reason in result["reasons"])
    if limit == "expired":
        assert made == []


def test_real_hung_smoke_process_is_killed_with_import_left_intact(inputs):
    inputs.output.mkdir()

    def hang(_command, **options):
        assert options["timeout"] <= 1.0
        return subprocess.run([sys.executable, "-c", "import time; time.sleep(30)"], **options)

    start = time.monotonic()
    result = smoke.run_bounded_smoke_process(inputs.output, wall_seconds=1., run=hang)
    assert time.monotonic() - start < 5
    assert result["status"] == "failed"
    assert result["reasons"] == ["native_controller_process_timeout"]
    assert result["timeout_includes_sdk_startup"] is True
    assert inputs.model.is_file()


def test_bounded_process_reopens_and_binds_actual_diagnostic_receipt(inputs):
    native, _ = execute(inputs)
    (inputs.output / "binding-inputs.json").write_text(json.dumps(inputs.binding))
    result = smoke.run_bounded_smoke_process(inputs.output, wall_seconds=90.,
                                           run=lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    assert result["status"] == "passed"
    assert result["smoke_digest"] == native["smoke_digest"]


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "corrupt_checkpoint"])
def test_batch_recovers_only_verified_import_when_process_dies_after_checkpoint(inputs, monkeypatch, tmp_path, failure):
    import shutil
    from pathlib import Path
    from blueprint_pipeline import deployment_embodiment_preparation_worker as worker
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.deployment_embodiment_preparation_contract import sealed

    binding = inputs.binding
    identity = binding["robot_embodiment"]["embodiment_id"]
    provenance = binding["model_record"]["source_import_provenance"]
    row = {"embodiment_id": identity, "status": "imported", "native_import_executed": True,
        "configuration_digest": binding["configuration_digest"], "import_digest": provenance["import_digest"],
        "source_bundle_digest": provenance["source_bundle_digest"], "model_sha256": binding["model_record"]["sha256"],
        "task_executed": False, "policy_executed": False}
    checkpoint = sealed({"schema_version": "embodiment_import_checkpoint.v1", "body_result": row,
                         "binding_digest": canonical_digest(binding)}, "checkpoint_digest")
    monkeypatch.setattr(worker, "verify_inputs", lambda _path: {
        "inputs_digest": "sha256:" + "1" * 64,
        "request": {"embodiment_ids": [identity], "request_digest": "sha256:" + "2" * 64, "expires_at_epoch": 10000},
        "sources": [{"embodiment_id": identity, "status": "source_ready"}]})
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "10000")

    def interrupted(command, **options):
        body = Path(command[command.index("--output") + 1])
        body.mkdir()
        shutil.copytree(inputs.configuration, body / "configuration")
        shutil.copytree(inputs.model.parent, body / "prepared")
        (body / "binding-inputs.json").write_text(json.dumps(binding))
        (body / "import-checkpoint.json").write_text(json.dumps(checkpoint))
        worker.seal_verified_import(body, identity)  # the Kit child seals before its fast shutdown
        if failure == "corrupt_checkpoint":
            (body / "prepared" / "robot.usda").write_text("#usda 1.0\nchanged import")
        if failure == "nonzero":
            return SimpleNamespace(returncode=23)
        raise subprocess.TimeoutExpired(command, options["timeout"])

    output = tmp_path / "batch-output"
    result = worker.run_batch(tmp_path / "runtime", output, run=interrupted, now=lambda: 0., wall=lambda: 1000.)
    actual = result["results"][0]
    if failure == "corrupt_checkpoint":
        assert actual["status"] == "import_failed"
        assert actual["checkpoint_recovery"].startswith("unverified:")
        assert not (output / "bodies" / identity).exists()
    else:
        assert actual["status"] == "imported"
        assert actual["native_controller_smoke"]["status"] == "failed"
        assert actual["native_controller_smoke"]["checkpoint_recovered"] is True
        assert (output / "bodies" / identity / "prepared" / "robot.usda").is_file()
        assert json.loads((output / "bodies" / identity / "body-result.json").read_text()) == actual
