from dataclasses import dataclass
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_task_runtime_comparison import (
    FIXED_FIELDS, GA_ARENA_COMMIT, GA_COMMIT, GA_IMAGE, RuntimeMeasurement,
    BASE_ARENA_COMMIT, BASE_COMMIT, BASE_IMAGE,
    compare_runtime_measurements, compare_runtime_measurement_sets, comparison_binding,
    gpu_observation,
)


def test_gpu_memory_accepts_identical_driver_rows_only(monkeypatch):
    monkeypatch.setattr('blueprint_pipeline.native_task_runtime_comparison.os.getpid', lambda: 3817)
    raw = SimpleNamespace(stdout='3817, GPU-retained, 692\n3817, GPU-retained, 692\n')
    monkeypatch.setattr('blueprint_pipeline.native_task_runtime_comparison.subprocess.run', lambda *a, **k: raw)
    assert gpu_observation() == {'gpu_uuid': 'GPU-retained', 'used_mib': 692.0}
    for conflicting in ('3817, GPU-retained, 693\n', '3817, GPU-other, 692\n'):
        raw.stdout = '3817, GPU-retained, 692\n' + conflicting
        with pytest.raises(ValueError, match='process_gpu_not_unique'):
            gpu_observation()


@dataclass(frozen=True)
class Runtime:
    launch_isaac: object
    build_environment: object
    prepolicy_camera_gate: object
    run_policy_episode: object


class Tensor:
    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [[0.25, 0.5]]


def measured(tmp_path, variant, *, identity_overrides=None, input_overrides=None, gpu="GPU-test"):
    binding = comparison_binding(**{**{key: {"fixture": key} for key in FIXED_FIELDS}, **(input_overrides or {})})
    identity = {"physics_backend": "isaacsim_physx", "renderer": "isaac_rtx",
                "isaaclab_commit": GA_COMMIT if variant == "isaaclab_3_ga" else BASE_COMMIT,
                "arena_commit": GA_ARENA_COMMIT if variant == "isaaclab_3_ga" else BASE_ARENA_COMMIT,
                "container_image": GA_IMAGE if variant == "isaaclab_3_ga" else BASE_IMAGE,
                **(identity_overrides or {})}
    ticks = iter(range(100))
    path = tmp_path / (variant + ".json")
    measurement = RuntimeMeasurement(output_path=path, variant=variant, runtime_identity=identity,
        binding=binding, clock=lambda: float(next(ticks)), synchronize=lambda: None,
        gpu_reader=lambda: {"gpu_uuid": gpu, "used_mib": 4000.})
    action = object()
    observation = object()
    calls = []
    class Env:
        unwrapped = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=Tensor()))})

        def step(self, value):
            assert value is action
            calls.append("step")
            return observation

        def reset(self, *, seed):
            calls.append(("reset", seed))
            return observation

    class App:
        def close(self):
            assert json.loads(path.read_text())["status"] == "measured"
            raise SystemExit(0)

    episode = {"episode_id": "fixture", "score": {"success": False},
               "contact_force_evidence": {"measured": True},
               "candidate_exact_policy_input_manifest_digest": "sha256:" + "a"*64}
    runtime = Runtime(lambda: (App(), {"launched": True}), lambda: SimpleNamespace(env=Env()),
                      lambda: {"passed": True}, lambda: episode)
    wrapped = measurement.wrap(runtime)
    app, _ = wrapped.launch_isaac()
    built = wrapped.build_environment()
    assert built.env.reset(seed=123) is observation
    assert built.env.step(action) is observation
    assert wrapped.run_policy_episode() is episode
    assert wrapped.prepolicy_camera_gate() == {"passed": True}
    with pytest.raises(SystemExit):
        app.close()
    assert calls == [("reset", 123), "step"]
    return json.loads(path.read_text())


def test_measurement_preserves_runner_values_and_seals_before_kit_exit(tmp_path):
    a = measured(tmp_path, "baseline")
    b = measured(tmp_path, "isaaclab_3_ga")
    report = compare_runtime_measurements(a, b)
    assert report["status"] == "measured_pending_correctness_review"
    assert report["improvement_qualified"] is False
    assert report["metrics"]["step"]["candidate_change_percent"] == 0
    assert a["episodes"][0]["score"]["success"] is False
    assert a["timings"]["reset"]["count"] == 1
    assert a["resets"][0]["robot_joint_positions"] == [[.25, .5]]


def test_different_task_or_gpu_cannot_be_called_runtime_comparison(tmp_path):
    a = measured(tmp_path, "baseline")
    b = measured(tmp_path, "isaaclab_3_ga", input_overrides={"assets": {"different": True}}, gpu="other")
    report = compare_runtime_measurements(a, b)
    assert set(report["blockers"]) == {"fixed_workcell_inputs_changed", "same_gpu_required"}


def _cell_measurement(value, index, *, gpu=None):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    value = deepcopy(value)
    cell = {'cell_id': f'cell-{index}', 'seed': 100 + index}
    value.update(measurement_scope='retained_policy_diagnostic', selected_cell_index=index,
                 expected_cell_count=2, selected_cell=cell,
                 selected_cell_digest=canonical_digest({'value': cell}))
    if gpu is not None:
        for row in value['gpu_samples']:
            row['gpu_uuid'] = gpu
    value['receipt_digest'] = canonical_digest(value, digest_field='receipt_digest')
    return value


def test_policy_set_refuses_wrong_missing_duplicate_or_cross_gpu_scenarios(tmp_path):
    a, b = measured(tmp_path, 'baseline'), measured(tmp_path, 'isaaclab_3_ga')
    baseline = [_cell_measurement(a, i) for i in range(2)]
    candidate = [_cell_measurement(b, i) for i in range(2)]
    assert compare_runtime_measurement_sets(baseline, candidate)['blockers'] == []
    assert 'runtime_comparison_scenario_mismatch' in compare_runtime_measurements(
        baseline[0], candidate[1])['blockers']
    assert 'candidate_scenario_set_incomplete' in compare_runtime_measurement_sets(
        baseline, candidate[:1])['blockers']
    assert 'candidate_scenario_identity_invalid_or_duplicated' in compare_runtime_measurement_sets(
        baseline, [*candidate, candidate[0]])['blockers']
    # Matching each pair independently cannot authorize different hardware
    # for different cells in this one-workcell comparison.
    baseline[1] = _cell_measurement(a, 1, gpu='GPU-other')
    candidate[1] = _cell_measurement(b, 1, gpu='GPU-other')
    assert 'same_gpu_required_across_scenarios' in compare_runtime_measurement_sets(
        baseline, candidate)['blockers']


def test_comparison_cli_reads_complete_policy_cell_directories(tmp_path):
    from blueprint_pipeline.native_task_runtime_comparison_cli import main
    for variant in ('baseline', 'isaaclab_3_ga'):
        value = measured(tmp_path, variant)
        for index in range(2):
            folder = tmp_path / variant / str(index)
            folder.mkdir(parents=True)
            (folder / 'native_task_runtime_measurement.v1.json').write_text(
                json.dumps(_cell_measurement(value, index)))
    output = tmp_path / 'comparison.json'
    assert main(['compare-set', '--baseline-root', str(tmp_path / 'baseline'),
                 '--candidate-root', str(tmp_path / 'isaaclab_3_ga'), '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert len(result['scenario_comparisons']) == 2
    assert result['improvement_qualified'] is False


@pytest.mark.parametrize("overrides", [{"renderer": "ovrtx"}, {"physics_backend": "newton"}])
def test_other_backends_require_separate_experiment(tmp_path, overrides):
    with pytest.raises(ValueError, match="separate_experiment"):
        measured(tmp_path, "isaaclab_3_ga", identity_overrides=overrides)


def test_ga_identity_cannot_be_a_floating_image(tmp_path):
    with pytest.raises(ValueError, match="ga_identity_invalid"):
        measured(tmp_path, "isaaclab_3_ga", identity_overrides={"container_image": "nvcr.io/nvidia/isaac-lab:latest"})


def test_construction_freeze_refuses_any_changed_task_camera_or_controller():
    from copy import deepcopy
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.native_task_runtime_comparison import (
        construction_comparison_binding, validate_construction_comparison_request,
    )
    plan = {"plan_digest": "fixed", "task_spec": {"joint": "door"},
            "objects": [{"sha256": "asset"}], "robot": {"id": "droid"},
            "cameras": [{"width": 256}], "scenario": {"seed": 42}}
    phases = {"phases": [{"action": "approach"}], "steps": 100}
    value = {"schema_version": "native_task_runtime_comparison_request.v1",
             "scope": "retained_construction_replay", "variant": "baseline",
             "binding": construction_comparison_binding(plan, phases),
             "runtime_identity": {"container_image": BASE_IMAGE, "isaaclab_commit": BASE_COMMIT,
                                  "arena_commit": BASE_ARENA_COMMIT, "renderer": "isaac_rtx",
                                  "physics_backend": "isaacsim_physx"}}
    value["request_digest"] = canonical_digest(value, digest_field="request_digest")
    assert validate_construction_comparison_request(value, plan=plan, phase_plan=phases) == value
    for field in ("task_spec", "cameras", "objects", "robot", "scenario"):
        changed = deepcopy(plan)
        changed[field] = {"changed": True}
        with pytest.raises(ValueError, match="input_mismatch"):
            validate_construction_comparison_request(value, plan=changed, phase_plan=phases)
    with pytest.raises(ValueError, match="input_mismatch"):
        validate_construction_comparison_request(value, plan=plan, phase_plan={"steps": 101})
