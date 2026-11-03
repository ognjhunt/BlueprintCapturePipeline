# Covers (for impacted-test selection):
#   src/blueprint_pipeline/team_scene_packet_remote.py
#   src/blueprint_pipeline/team_scene_packet_worker.py
#   src/blueprint_pipeline/team_scene_packet.py
#   src/blueprint_pipeline/remote_cpu_environment.py
"""The robot-team scene half compiles on a remote CPU worker exactly as on the host (plan 14's lane, team stage)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.team_scene_packet import RECORD_FILENAME
from tests.remote_team_scene_packet_support import (
    COMMIT,
    EXECUTION,
    HOST_RECORD,
    TeamHost,
    TeamRemoteWorld,
    team_config,
    tree_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.parametrize("bodies", [False, True], ids=["franka_only", "registered_bodies"])
def test_a_remote_compile_is_byte_identical_to_the_host_compile(tmp_path: Path, monkeypatch, bodies: bool) -> None:
    """The host's own compile of a request (``cloud_run_shadow``), then the worker's compile of the same envelope
    at the same paths, from the extracted recipe-v2 archive in a spawned stage child, landed whole: every path,
    byte and mode agree, and the worker's result names the host's own team scene inputs record."""

    host = TeamHost(tmp_path / "host", monkeypatch, mode="cloud_run_shadow", bodies=bodies)
    [status] = host.process()["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled"), status
    assert status["remote_compile"]["shadowed"] is True and host.calls == []
    _, marker = host.marker("shadow")
    plan = lane.RemotePlan.from_record(marker["plan"])
    output = host.local(plan.output_root)
    assert str(output / RECORD_FILENAME) == status["team_scene_inputs_path"]
    snapshot = tree_snapshot(output)
    record = json.loads((output / RECORD_FILENAME).read_text(encoding="utf-8"))
    assert ("embodiment_stations" in record) is bodies
    # Every file the compile reads ships, at its host path: bundle, documents, profile, receipt, models.
    shipped = {row["contract_path"] for row in plan.inputs}
    assert {"queue_envelope", "configured_scene_bundle_path", "droid_profile_path",
            "runtime_source_receipt_path"} <= shipped
    assert any(name.startswith("documents.") for name in shipped)
    assert any(name.startswith("embodiment_model.") for name in shipped) is bodies

    world = TeamRemoteWorld(tmp_path / "remote", config=host.config)
    descriptor = world.descriptor(plan, mode="shadow")
    assert descriptor["limits"]["phase_seconds"] == {"fetch": 300, "stage": 2700, "seal_upload": 420}
    assert descriptor["limits"]["max_output_bytes"] == 4 * 1024**3
    environ = world.environ(descriptor)
    # The worker's /var/lib/blueprint starts empty: every host path moves aside while it runs.
    aside = tmp_path / "host" / "aside"
    os.rename(host.fs, aside)
    try:
        assert world.execute(environ, filesystem_root=host.fs, measure=lambda: HOST_RECORD) == 0, world.logs
    finally:
        os.rename(host.fs, tmp_path / "worker-tree")
        os.rename(aside, host.fs)
    receipt = json.loads(world.staged(descriptor, "receipt.json"))
    assert receipt["status"] == "succeeded", (receipt["infrastructure_failures"], receipt["release_path_misses"],
                                              receipt["result"])
    verdict = contract.validate_receipt(receipt, descriptor=descriptor, execution_name=EXECUTION)
    assert (verdict["outcome"], verdict["terminal"]) == ("succeeded", True)
    result = receipt["result"]
    assert lane.validate_result(result, request_id=host.world["key"], envelope_digest=plan.queue_row["envelope_digest"],
                                source_commit=COMMIT, output_root=plan.output_root) == []
    assert result["compiled"]["inputs_digest"] == record["inputs_digest"] == status["team_scene_inputs_digest"]

    landed = tmp_path / "landed" / "scene"
    landed.parent.mkdir()
    index = world.land(descriptor, plan, landed)
    assert tree_snapshot(landed) == snapshot
    # Only new bytes left the worker: the shipped inputs it copied travelled as references.
    assert index["paths_total"] == len(snapshot["files"])


def test_an_unset_mode_is_auto_and_a_set_one_keeps_its_name(tmp_path: Path) -> None:
    config_path = tmp_path / "remote-cpu-workers.json"
    with_config = {lane.CONFIG_ENV: str(config_path)}
    assert lane.resolve_execution_mode(with_config) == {"requested": None, "effective": "host",
                                                        "reason": "auto_without_config", "findings": []}
    config_path.write_text(json.dumps(team_config()), encoding="utf-8")
    config_path.chmod(0o640)
    assert lane.resolve_execution_mode(with_config)["effective"] == "cloud_run"
    assert lane.resolve_execution_mode(with_config)["reason"] == "auto_with_config"
    # The production config today names only episode compilation: auto is host, byte for byte.
    episode_only = team_config()
    episode_only["stages"].pop("team_scene_packet")
    from tests.remote_cpu_allocator_fakes import seal

    config_path.write_text(json.dumps(seal(episode_only, "config_digest")), encoding="utf-8")
    assert lane.resolve_execution_mode(with_config)["effective"] == "host"
    for mode in lane.MODES:
        assert lane.resolve_execution_mode({**with_config, lane.EXECUTION_ENV: mode})["effective"] == mode
    invalid = lane.resolve_execution_mode({**with_config, lane.EXECUTION_ENV: "cloudrun"})
    assert invalid == {"requested": "invalid", "effective": "host", "reason": "explicit",
                       "findings": ["team_scene_packet_execution_mode_invalid"]}
    # A config the allocator would refuse (group-writable) is no config at all.
    config_path.write_text(json.dumps(team_config()), encoding="utf-8")
    config_path.chmod(0o660)
    assert lane.resolve_execution_mode(with_config)["effective"] == "host"


def test_host_mode_is_todays_path_byte_for_byte(tmp_path: Path, monkeypatch) -> None:
    """``host`` compiles through ``prepare_team_scene`` exactly as before: no plan, no marker, no envelope, and
    the same status a run without any routing writes."""

    from blueprint_pipeline import team_scene_packet

    host = TeamHost(tmp_path, monkeypatch, mode="host")
    calls = []
    original = team_scene_packet.prepare_team_scene
    monkeypatch.setattr(team_scene_packet, "prepare_team_scene", lambda **kwargs: calls.append(kwargs) or original(
        **kwargs))
    monkeypatch.setattr(lane, "plan_remote_team_scene", lambda **_: pytest.fail("planned in host mode"))
    run = host.process()
    [status] = run["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled") and len(calls) == 1
    assert "remote_compile" not in status and "execution_mode" not in run
    assert not any((host.jobs / directory).exists() for directory in ("handoffs", "shadow", "envelopes"))


def test_cloud_run_hands_an_eligible_request_off_and_keeps_its_claim(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import team_scene_packet

    host = TeamHost(tmp_path, monkeypatch)
    monkeypatch.setattr(team_scene_packet, "materialize_team_scene", lambda **_: pytest.fail("compiled on the host"))
    run = host.process()
    [status] = run["results"]
    assert status["status"] == "processing" and host.state() == "processing"
    assert status["remote_compile"]["mode"] == "cloud_run" and host.status() == status
    assert run["execution_mode"]["effective"] == "cloud_run"
    _, marker = host.marker("authoritative")
    plan = lane.RemotePlan.from_record(marker["plan"])
    assert marker["request_id"] == host.world["key"] and plan.attempt_root == status["attempt_root"]
    assert plan.output_root == "/" + Path(status["attempt_root"]).relative_to(host.fs).as_posix() + "/scene"
    assert lane.envelope_path(host.jobs, plan.queue_row["name"]).is_file()
    # The documents the host fetched stay at the attempt; nothing of the scene half exists on the host yet.
    attempt = Path(status["attempt_root"])
    assert (attempt / "documents").is_dir() and not (attempt / "scene").exists()
    # A later run neither recovers nor recompiles a request the remote lane holds.
    again = host.process()
    assert again["recovered"] == [] and again["results"] == [] and host.state() == "processing"
    monkeypatch.setenv(lane.JOBS_ROOT_ENV, str(host.jobs))
    assert worker.requeue_transient_blocked(queue_root=host.world["queue"])["recovered"] == []
    assert host.state() == "processing" and (attempt / "documents").is_dir()


@pytest.mark.parametrize("change, reason", [
    ("no_environment", "remote_ineligible:environment_unrecorded"),
    ("environment_mismatch", "remote_ineligible:environment_mismatch:distributions"),
    ("stage_distributions", "remote_ineligible:environment_mismatch:stage_distributions"),
    ("config_without_stage", "remote_ineligible:config_invalid"),
    ("unknown_release", "remote_ineligible:release_unknown"),
    ("small_ephemeral", "remote_ineligible:ephemeral_budget_exceeded"),
])
def test_an_ineligible_request_compiles_on_the_host(tmp_path: Path, monkeypatch, change: str, reason: str) -> None:
    from dataclasses import replace

    from tests.remote_cpu_allocator_fakes import environment, seal

    host = TeamHost(tmp_path, monkeypatch)
    routing = host.routing
    if change == "no_environment":
        (host.jobs / "environment" / "team_scene_packet.json").unlink()
    elif change == "environment_mismatch":
        host.record_worker_environment(environment(distributions=[{"name": "numpy", "version": "9.9.9"}]))
    elif change == "stage_distributions":
        host.record_worker_environment(environment(stage_distributions={"team_scene_packet": [
            {"name": "rfc8785", "version": "0.0.1"}]}))
    elif change == "config_without_stage":
        config = team_config()
        config["stages"].pop("team_scene_packet")
        routing = replace(routing, config=seal(config, "config_digest"))
    elif change == "unknown_release":
        routing = replace(routing, source_commit=None)
    else:
        from tests.remote_team_scene_packet_support import team_stage

        config = team_config()
        config["stages"]["team_scene_packet"] = team_stage(ephemeral_bytes=1024)
        routing = replace(routing, config=seal(config, "config_digest"))
    [status] = host.process(routing)["results"]
    assert (status["status"], status["outcome"]) == ("completed", "compiled"), status
    assert status["remote_compile"] == {"mode": "cloud_run", "host_decision": reason}
    envelopes = host.jobs / "envelopes" / "team_scene_packet"
    assert lane.markers(host.jobs, "authoritative") == [] and not (envelopes.is_dir() and any(envelopes.iterdir()))


def test_a_path_outside_the_permitted_roots_keeps_the_compile_on_the_host(tmp_path: Path, monkeypatch) -> None:
    host = TeamHost(tmp_path, monkeypatch)
    # The operator's DROID profile moves outside /var/lib/blueprint: the worker could not materialize it there.
    outside = tmp_path / "etc-blueprint" / "droid_profile.json"
    outside.parent.mkdir()
    outside.write_bytes(Path(host.world["inputs"]["droid_profile_path"]).read_bytes())
    from tests.test_team_scene_packet_worker import _registry

    _registry(host.world["operator"], packet_inputs={**host.world["inputs"], "droid_profile_path": str(outside)})
    routing = host.dispatch("cloud_run")
    [status] = host.process(routing)["results"]
    assert status["status"] == "completed"
    assert status["remote_compile"]["host_decision"] == "remote_ineligible:path_outside_permitted_roots"


def test_the_worker_stage_is_registered_and_imports_no_authority() -> None:
    from blueprint_pipeline import remote_cpu_worker

    assert remote_cpu_worker.STAGE_HANDLERS[lane.STAGE] == (
        "blueprint_pipeline.team_scene_packet_remote:run_team_scene_packet_in_worker")
    source = Path(lane.__file__).read_text(encoding="utf-8")
    for authority in ("remote_cpu_job_allocator", "paid_resource_allocator", "cloud_run_jobs_client",
                      "remote_cpu_transport", "task_evaluation_configured_scene_object_store", "boto3",
                      "google.oauth2"):
        assert authority not in source, authority


_CLOSURE = '''
import importlib.metadata, json, sys
from pathlib import Path
# What site loaded at start-up (.pth hooks: setuptools' distutils shim, cmeel, CUDA path finders) is the
# interpreter's, never the compile's: only what this script loads after it counts, as in the episode stage's proof.
baseline = set(sys.modules)
inputs = json.loads(Path(sys.argv[1]).read_text())
inputs["documents"] = {name: Path(path) for name, path in inputs["documents"].items()}
# What the registered handler imports before it compiles: its own module and the host worker's blocker typing.
import blueprint_pipeline.team_scene_packet_remote
from blueprint_pipeline.team_scene_packet_worker import _blocker
from blueprint_pipeline.team_scene_packet import materialize_team_scene
materialize_team_scene(**inputs, output_root=Path(sys.argv[2]))
repository = Path(sys.argv[4]).resolve()


def ours(name):
    path = getattr(sys.modules.get(name), "__file__", None)
    return bool(path) and Path(path).resolve().is_relative_to(repository)


mapping = importlib.metadata.packages_distributions()
loaded = {name.split(".")[0] for name in set(sys.modules) - baseline if not ours(name)}
seen = {dist for top in loaded for dist in mapping.get(top, [])}
Path(sys.argv[3]).write_text(json.dumps({"distributions": sorted(seen), "loaded": sorted(sys.modules)}))
'''


@pytest.mark.slow
def test_parity_set_covers_every_distribution_a_team_compile_loads(tmp_path: Path) -> None:
    """A real team compile, in a fresh interpreter, loads only distributions the census measures (its digested
    set, or this stage's own row), and no allocation or staging authority: the runtime proof the worker's static
    closure test defers to for this compiler stage (``tests/test_remote_cpu_worker.py::COMPILER_STAGES``)."""

    from blueprint_pipeline import remote_cpu_environment as census
    from tests.test_team_scene_packet import _scene

    scene = _scene(tmp_path)
    inputs = {key: str(value) if isinstance(value, Path) else value for key, value in scene.items()
              if key not in ("profile", "store", "revision_reference")}
    inputs["documents"] = {name: str(path) for name, path in scene["documents"].items()}
    (tmp_path / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    run = subprocess.run(
        [sys.executable, "-c", _CLOSURE, str(tmp_path / "inputs.json"), str(tmp_path / "scene"), str(tmp_path / "out.json"),
         str(ROOT / "src")],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, run.stderr[-4000:]
    report = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    normal = {name.replace("_", "-").lower() for name in report["distributions"]} - {"blueprint-capture-pipeline"}
    measured = {name.replace("_", "-").lower() for name in (
        *census.COMPILE_DISTRIBUTIONS, *census.STAGE_DISTRIBUTIONS[lane.STAGE])}
    assert {"usd-core", "numpy", "rfc8785"} <= normal
    assert normal <= measured, sorted(normal - measured)
    authority = {"blueprint_pipeline.paid_resource_allocator", "blueprint_pipeline.remote_cpu_job_allocator",
                 "blueprint_pipeline.cloud_run_jobs_client", "blueprint_pipeline.remote_cpu_transport",
                 "blueprint_pipeline.task_evaluation_configured_scene_object_store"}
    assert not authority & set(report["loaded"])
