"""A Vast host that cannot isolate generated CAD programs fails fast and stays avoided.

Two September scene-configuration runs paid for stages 1-2 before ARTISAN's
own sandbox preflight refused the host (``asset_sandbox_unavailable``). The
refusal never reached the machine avoidlist, and each launch's avoidlist lived
only in its own job directory, so the next launch could rent the same host.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline import vast_provider_adapter as adapter
from blueprint_pipeline.asset_authoring_sandbox import AssetSandboxError, SandboxedAssetRunner
from blueprint_pipeline.provider_machine_avoidlist import (
    SCENE_CONFIGURATION_MACHINE_AVOIDLIST_FILENAME,
    avoidlist_machine_ids,
    scene_configuration_excluded_machine_ids,
)


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts/task_evaluation_scene_configuration_provider_runner.py"
    spec = importlib.util.spec_from_file_location("scene_configuration_provider_runner_isolation_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_blender(runtime: Path, *, script: str = 'echo "Blender 5.0.0"') -> Path:
    blender = runtime / "toolchain/components/content_agents_rigid_replacement/package/runtime/blender/blender"
    blender.parent.mkdir(parents=True, exist_ok=True)
    blender.write_text("#!/bin/sh\n" + script + "\n")
    blender.chmod(0o755)
    return blender


def _prepare_runner(tmp_path, monkeypatch, *, backend="artisan_v1"):
    runner = _runner()
    runtime, output = tmp_path / "runtime", tmp_path / "output"
    runtime.mkdir()
    _fake_blender(runtime)
    result_path = output / "result.json"
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT", str(runtime))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT", str(output))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_PROVIDER_RESULT", str(result_path))
    monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH", str(time.time() + 27_000))
    configurations = {"stage-1.json": {}, "stage-3.json": {"authoring_backend": backend}}
    monkeypatch.setattr(runner, "_read", lambda p: configurations.get(p.name, {"envelope_digest": "sha256:" + "a" * 64}))
    monkeypatch.setattr(runner, "_hydrate_envelope", lambda *_: {
        "run_id": "run", "expected_production_commit": "b" * 40,
        "recipe": {"stage_sequence": [{"stage_id": "stage-1"}, {"stage_id": "stage-3"}]},
        "stage_configuration_references": [
            {"stage_id": stage, "materialized_path": str(tmp_path / f"{stage}.json")}
            for stage in ("stage-1", "stage-3")]})
    chain_calls = []

    def stages(**kwargs):
        chain_calls.append(kwargs)
        raise RuntimeError("test_stops_before_stage_execution")

    monkeypatch.setattr(runner, "execute_scene_configuration_stage_chain", stages)
    return runner, result_path, chain_calls


def test_host_without_isolation_blocks_before_stage_one_with_avoidlist_marker(tmp_path, monkeypatch, capsys):
    runner, result_path, chain_calls = _prepare_runner(tmp_path, monkeypatch)

    def refused(self):
        raise AssetSandboxError("asset_sandbox_unavailable")

    monkeypatch.setattr(SandboxedAssetRunner, "preflight", refused)
    assert runner.main() == 2
    assert chain_calls == []
    logged = capsys.readouterr().out
    assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED:asset_sandbox_unavailable" in logged
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "blocked"
    assert "asset_sandbox_unavailable" in result["blockers"][0]
    assert result["first_stage_started"] is False


def test_isolated_host_reaches_the_stage_chain(tmp_path, monkeypatch, capsys):
    runner, _result_path, chain_calls = _prepare_runner(tmp_path, monkeypatch)
    probes = []
    monkeypatch.setattr(SandboxedAssetRunner, "preflight", lambda self: probes.append(self.write_root))
    assert runner.main() == 2  # the fixture chain stops itself
    assert len(chain_calls) == 1
    assert probes == [(tmp_path / "output/asset_isolation_host_preflight").resolve()]
    logged = capsys.readouterr().out
    assert "BLUEPRINT_SCENE_CONFIGURATION_ASSET_ISOLATION_PREFLIGHT_PASSED" in logged
    assert "BLUEPRINT_SCENE_CONFIGURATION_ARTISAN_BLENDER_STARTUP_PASSED" in logged


def test_blender_that_cannot_start_on_the_image_blocks_before_stage_one(tmp_path, monkeypatch, capsys):
    runner, result_path, chain_calls = _prepare_runner(tmp_path, monkeypatch)
    _fake_blender(tmp_path / "runtime", script=(
        'echo "blender: error while loading shared libraries: libXi.so.6: cannot open shared object file" >&2; exit 127'))
    monkeypatch.setattr(SandboxedAssetRunner, "preflight", lambda self: None)
    assert runner.main() == 2
    assert chain_calls == []
    blocker = json.loads(result_path.read_text(encoding="utf-8"))["blockers"][0]
    assert "artisan_blender_runtime_unavailable:127:" in blocker and "libXi.so.6" in blocker
    # An image-level fault is not a machine fault: nothing to avoidlist.
    assert "BLUEPRINT_VAST_PROVIDER_BUNDLE_BLOCKED" not in capsys.readouterr().out


@pytest.mark.parametrize(("backend", "stage_limit"), [("astra_cad_blender_v1", None), ("artisan_v1", "stage-1")])
def test_runs_that_never_reach_artisan_skip_the_probe(tmp_path, monkeypatch, backend, stage_limit):
    runner, _result_path, chain_calls = _prepare_runner(tmp_path, monkeypatch, backend=backend)
    if stage_limit:
        monkeypatch.setenv("BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT", stage_limit)
    if backend == "astra_cad_blender_v1":
        from blueprint_pipeline import task_evaluation_scene_configuration_astra_driver as astra
        from blueprint_pipeline import task_evaluation_scene_configuration_builtin_producers as producers
        monkeypatch.setattr(producers, "_validate_toolchain", lambda **_: ({"stages": {
            "content_agents_rigid_replacement": {"component_entrypoint": "component/run.sh"}}}, []))
        monkeypatch.setattr(astra, "preflight_astra_execution_runtime", lambda **_: None)
    monkeypatch.setattr(SandboxedAssetRunner, "preflight", lambda self: pytest.fail("probe not needed"))
    assert runner.main() == 2
    assert len(chain_calls) == 1


def test_sandbox_refusal_marker_avoidlists_the_machine():
    blockers = ["provider_remote_blocker:asset_sandbox_unavailable", "provider_runtime_result_missing_from_output_zip"]
    assert adapter._machine_avoidlist_reason(blockers) == "vast_host_asset_sandbox_unavailable"
    assert adapter._machine_avoidlist_reason(["provider_remote_blocker:download_failed:6"]) is None


def _retained(runs: Path, launch: str, machine_ids: list[int]) -> None:
    job = runs / launch / "allocator/scene-configuration-job"
    job.mkdir(parents=True)
    (job / "vast_machine_avoidlist.json").write_text(json.dumps({
        "schema_version": "vast_machine_avoidlist.v1", "status": "completed",
        "machine_ids": machine_ids, "entries": [{"machine_id": machine_ids[0], "reason": "fixture"}]}))


def test_scene_configuration_launch_excludes_machines_its_siblings_proved_bad(tmp_path):
    runs = tmp_path / "task-evaluation-launch-runs"
    _retained(runs, "launch-a", [140607])
    _retained(runs, "launch-b", [138964, 140607])
    job = runs / "launch-c/allocator/scene-configuration-job"
    job.mkdir(parents=True)

    assert scene_configuration_excluded_machine_ids(job_dir=job) == {138964, 140607}
    shared = runs / "provider-machine-avoidlists" / SCENE_CONFIGURATION_MACHINE_AVOIDLIST_FILENAME
    assert avoidlist_machine_ids(shared) == {138964, 140607}
    # The launch's own avoidlist path stays untouched for its adapter to write.
    assert not (job / "vast_machine_avoidlist.json").exists()


def test_no_history_or_foreign_layout_excludes_nothing(tmp_path):
    job = tmp_path / "runs/launch/allocator/scene-configuration-job"
    job.mkdir(parents=True)
    assert scene_configuration_excluded_machine_ids(job_dir=job) == set()
    assert scene_configuration_excluded_machine_ids(job_dir=tmp_path / "job") == set()


def test_unreadable_shared_list_is_reported_and_excludes_nothing(tmp_path, capsys):
    runs = tmp_path / "runs"
    shared = runs / "provider-machine-avoidlists" / SCENE_CONFIGURATION_MACHINE_AVOIDLIST_FILENAME
    shared.parent.mkdir(parents=True)
    shared.write_text(json.dumps({"schema_version": "vast_machine_avoidlist.v1", "machine_ids": ["bad"]}))
    job = runs / "launch/allocator/scene-configuration-job"
    job.mkdir(parents=True)
    assert scene_configuration_excluded_machine_ids(job_dir=job) == set()
    assert "scene_configuration_machine_avoidlist_unavailable:ValueError" in capsys.readouterr().err


def test_onstart_installs_the_packaged_blender_system_libraries():
    """ARTISAN's Blender ships its own libraries except these X ones.

    ``readelf -d`` of the sealed toolchain's ``runtime/blender/blender``
    (d63579c5) lists libX11.so.6, libXrender.so.1, libXfixes.so.3, libXi.so.6
    and libxkbcommon.so.0 beyond glibc and libstdc++, with RUNPATH $ORIGIN/lib
    for everything else. The Isaac image has to supply them; naming them keeps
    that from depending on what other packages happen to pull in.
    """

    blender_packages = {"libx11-6", "libxrender1", "libxfixes3", "libxi6", "libxkbcommon0"}
    assert blender_packages <= set(adapter.SCENE_CONFIGURATION_PROVISIONED_COMMANDS)
    assert all(adapter.SCENE_CONFIGURATION_PROVISIONED_COMMANDS[name] == () for name in blender_packages)
