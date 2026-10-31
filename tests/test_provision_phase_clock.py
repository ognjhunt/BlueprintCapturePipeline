"""A deploy receipt says where runtime provisioning spent its time, and timing changes nothing else."""
import json

from blueprint_pipeline import provision_phase_clock as clock
from blueprint_pipeline.provision_phase_clock import mark_phase, phase_clock
from scripts import provision_task_evaluation_scene_configuration_release as provision


def test_each_mark_records_the_time_since_the_previous_one():
    ticks = iter([100.0, 103.0, 103.5])
    original = clock.time.monotonic
    clock.time.monotonic = lambda: next(ticks)
    try:
        with phase_clock() as phases:
            mark_phase("first")
            mark_phase("second")
    finally:
        clock.time.monotonic = original
    assert phases == {"first": 3.0, "second": 0.5}


def test_a_mark_outside_a_clock_does_nothing():
    mark_phase("nobody is timing this")


def test_provisioning_prints_its_phases_beside_the_unchanged_result(monkeypatch, capsys, tmp_path):
    result = {"status": "ready", "source_commit": "a" * 40, "environment": {}}

    def provisioned(_args, _readback):
        mark_phase("splat_render_runtime")
        return result

    class Reader:
        closed = False

        def close(self):
            Reader.closed = True

    monkeypatch.setattr(provision, "_provision_from_arguments", provisioned)
    monkeypatch.setattr(provision, "service_account_readback", lambda _user: Reader())
    monkeypatch.setattr(provision, "sweep_stale_staging", lambda **_kwargs: [])
    monkeypatch.setattr(provision.signal, "signal", lambda *_args: None)
    monkeypatch.setattr("sys.argv", ["provision", "--repository-root", str(tmp_path), "--source-commit", "a" * 40,
                                     "--runtime-root", str(tmp_path), "--node-executable", "n", "--browser-root", "b",
                                     "--browser-executable", "e", "--node-modules-root", "m", "--artifixer-root", "f",
                                     "--readback-user", "blueprint"])

    assert provision.main() == 0

    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert set(printed.pop("phase_seconds")) == {"splat_render_runtime"}
    assert printed == result
    assert Reader.closed is True
