# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_episode_compilation_remote.py
#   src/blueprint_pipeline/remote_cpu_environment.py
#   tests/remote_episode_compilation_support.py
"""ADP-009D/day-28, plan 14 PR 4: which episode compilations run remotely, decided without credentials (§13)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_environment as census
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import task_evaluation_episode_compilation_remote as remote
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.remote_cpu_allocator_fakes import CAS, IMAGE, OBJECT_PREFIX, environment, remote_cpu_config
from tests.remote_episode_compilation_support import (
    CACHE,
    COMPILE_CASES,
    HOST_RECORD,
    INPUTS,
    OUTPUTS,
    QUEUE,
    Host,
    RemoteWorld,
    digest_of,
    nurec_usdz,
    stage_compile,
    tree_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40


def _plan(host: Host, claimed: Path, *, config: dict | None = None, gate: bool = False,
          host_environment: dict | None = None) -> remote.RemotePlan | remote.HostDecision:
    return remote.plan_remote_compilation(
        claimed, inputs=host.inputs, outputs=host.outputs, source_commit=COMMIT, config=config or remote_cpu_config(),
        jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache,
        host_environment=HOST_RECORD if host_environment is None else host_environment, require_shadow_gate=gate)


def _descriptor(plan: remote.RemotePlan, config: dict) -> dict:
    """The descriptor the paid unit seals from a plan once its inputs are staged (plan 14 §3)."""

    from blueprint_pipeline import remote_cpu_job_allocator as allocator

    limits = contract.stage_limits(config, "episode_compilation", allowed_cpu_classes=plan.allowed_cpu_classes)
    return contract.build_descriptor(
        config=config, stage="episode_compilation", mode="authoritative", attempt=1, queue_row=plan.queue_row,
        code={"source_commit": plan.source_commit, "image": plan.image, "environment_digest": plan.environment_digest,
              "source_archive": {"digest": "sha256:" + "5" * 64, "size_bytes": 1234,
                                 "uri": f"{CAS}/remote-cpu-source/sha256/{'5' * 64}/source.tar"}},
        environment=plan.environment,
        inputs=[{**{key: row[key] for key in ("role", "contract_path", "digest", "size_bytes", "mode",
                                              "materialize_at")},
                 "uri": f"{CAS}/remote-cpu-input/sha256/{row['digest'][7:]}/input.bin"} for row in plan.inputs],
        outputs={"output_root": plan.output_root, "declared_scratch": list(plan.declared_scratch),
                 "object_prefix": OBJECT_PREFIX},
        limits=limits, closure=plan.closure,
        spend={"worst_case_usd": allocator.worst_case_usd(limits=limits, rate_table=config["rate_table"]),
               "rate_table_digest": canonical_digest(config["rate_table"])})


def test_nurec_over_inline_limit_without_cache_stays_on_host(tmp_path: Path, monkeypatch) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    monkeypatch.setattr(remote, "MAX_INLINE_NUREC_BYTES", 8192)
    small, large = nurec_usdz(4096), nurec_usdz(16384)
    _, name = host.stage(label="small", appearance=small, appearance_name="appearance.usdz")
    plan = _plan(host, host.claim(name))
    # A NuRec appearance at or under the inline limit converts inside the worker: its float32 math needs a
    # qualified CPU class, the one the preflight probe measured.
    assert isinstance(plan, remote.RemotePlan)
    assert plan.closure == {"class": "absent_inline_only", "source_appearance_digest": digest_of(small)}
    assert plan.allowed_cpu_classes == (HOST_RECORD["cpu_class"],) and plan.environment == {}
    # Over the limit with no cache entry, the host runs its transcoder, which writes into the host cache.
    _, name = host.stage(label="large", appearance=large, appearance_name="appearance.usdz")
    assert _plan(host, host.claim(name)) == remote.HostDecision("remote_ineligible:particlefield_transcode_required")
    # An appearance that is not NuRec needs no transcode however large it is, and neither does a policy
    # observation setup, which replaces the configured appearance.
    _, name = host.stage(label="usd", appearance=b"#usda 1.0\n" + b" " * 20000)
    assert _plan(host, host.claim(name)).closure == {"class": "not_applicable", "source_appearance_digest": None}
    observed = Host(tmp_path / "observed")
    observed.record_worker_environment()
    envelope, name = stage_compile(observed, policy_observation_override=True, appearance=large,
                                   appearance_name="appearance.usdz")
    plan = remote.plan_remote_compilation(
        observed.claim(name), inputs=observed.inputs, outputs=observed.outputs,
        source_commit=envelope["expected_production_commit"], config=remote_cpu_config(), jobs_root=observed.jobs,
        filesystem_root=observed.fs, cache_root=observed.cache, host_environment=HOST_RECORD,
        require_shadow_gate=False)
    assert plan.closure == {"class": "not_applicable", "source_appearance_digest": None}


def test_cached_particlefield_entry_ships_with_the_cache_root_environment(tmp_path: Path, monkeypatch) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    monkeypatch.setattr(remote, "MAX_INLINE_NUREC_BYTES", 8192)
    appearance = nurec_usdz(16384)
    envelope, name = host.stage(appearance=appearance, appearance_name="appearance.usdz")
    entry = host.cache_entry(digest_of(appearance))
    claimed = host.claim(name)
    plan = _plan(host, claimed)
    assert isinstance(plan, remote.RemotePlan)
    assert plan.closure == {"class": "shipped", "source_appearance_digest": digest_of(appearance)}
    # The worker imports the cache root at import time, so the descriptor sets it before execute starts.
    assert plan.environment == {contract.CACHE_ROOT_VARIABLE: CACHE}
    shipped = {row["materialize_at"]: row for row in plan.inputs if row["role"] == "particlefield_cache_member"}
    entry_root = f"{CACHE}/{digest_of(appearance)[7:]}"
    assert set(shipped) == {f"{entry_root}/{path.name}" for path in entry.values()}
    for row in shipped.values():
        local = Path(row["host_path"])
        assert local == host.local(row["materialize_at"]) and digest_of(local.read_bytes()) == row["digest"]
        assert row["mode"] == "0440" and row["size_bytes"] == local.stat().st_size
    # Every input is at the host's own path, relative to ``/``; the row's envelope is at its claimed path.
    by_role = {row["role"] for row in plan.inputs}
    assert by_role == {"queue_envelope", "materialized_reference", "particlefield_cache_member"}
    envelope_row = next(row for row in plan.inputs if row["role"] == "queue_envelope")
    assert envelope_row["materialize_at"] == f"{QUEUE}/processing/{name}"
    assert all(row["materialize_at"].startswith(INPUTS + "/") for row in plan.inputs
               if row["role"] == "materialized_reference")
    assert plan.output_root == f"{OUTPUTS}/{envelope['compilation_id']}"
    assert plan.declared_scratch == (f"{OUTPUTS}/content-addressed/",)
    assert plan.queue_row == {"queue": "task-evaluation-episode-compilations", "name": name,
                              "envelope_digest": envelope["envelope_digest"]}
    # The plan is exactly what a sealed descriptor accepts.
    descriptor = _descriptor(plan, remote_cpu_config())
    assert descriptor["closure"]["class"] == "shipped" and descriptor["environment"] == plan.environment
    # A cache entry whose manifest no longer matches its files is the host's to refuse, not the worker's.
    entry["asset"].chmod(0o640)
    entry["asset"].write_bytes(b"replaced")
    assert _plan(host, claimed) == remote.HostDecision("remote_ineligible:particlefield_cache_invalid")


_CLOSURE = textwrap.dedent('''
    import importlib.metadata, json, sys
    from pathlib import Path

    claimed, inputs, outputs, commit, out = sys.argv[1:6]
    baseline = set(sys.modules)
    import tests.remote_cpu_worker_stages as stages
    from blueprint_pipeline.task_evaluation_episode_compilation_worker import compile_claimed_envelope

    compiler = stages.install_compile_stand_ins()
    if sys.argv[6] == "real_appearance":  # the real NuRec conversion loads pxr and msgpack lazily
        from blueprint_pipeline.task_evaluation_native_arena_episode_compiler import compile_native_arena_episode
        compiler = compile_native_arena_episode
    state, result = compile_claimed_envelope(Path(claimed), source_name=Path(claimed).name, inputs=Path(inputs),
                                             outputs=Path(outputs), source_commit=commit, episode_compiler=compiler,
                                             disk_reservation_root=None, storage_pins_root=None)
    owners = importlib.metadata.packages_distributions()
    loaded = {name.split(".")[0] for name in set(sys.modules) - baseline}
    distributions = sorted({owner for top in loaded for owner in owners.get(top, ())} - {"blueprint-capture-pipeline"})
    Path(out).write_text(json.dumps({"status": result["status"], "blockers": result["blockers"],
                                     "distributions": distributions}), encoding="utf-8")
''')


def _real_nurec(path: Path) -> bytes:
    """A small real NuRec USDZ, so the fixture compile converts it inline as a NuRec appearance does."""

    import numpy as np

    from blueprint_pipeline.aura_nurec_usdz import write_aura_nurec_usdz
    from blueprint_pipeline.nurec_volume_codec import build_state_dict

    rng, count = np.random.default_rng(7), 256
    arrays = {"positions": rng.normal(size=(count, 3)).astype(np.float32),
              "rotations": np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
              "scales": np.full((count, 3), -3.0, dtype=np.float32),
              "densities": np.full((count, 1), 2.0, dtype=np.float32),
              "features_albedo": rng.uniform(size=(count, 3)).astype(np.float32),
              "features_specular": np.zeros((count, 45), dtype=np.float32)}
    document = {"version": "0.2.576", "model": "nre", "config": {"layers": {"gaussians": {
        "precision": 32, "density_activation": "sigmoid", "scale_activation": "exp",
        "rotation_activation": "normalize",
        "particle": {"density_kernel_planar": False, "radiance_sph_degree": 3}}}, "renderer": {"name": "3dgut-nrend"}},
        "state_dict": build_state_dict(arrays, precision=32)}
    write_aura_nurec_usdz(document, path)
    return path.read_bytes()


@pytest.mark.slow
def test_parity_set_covers_every_distribution_a_fixture_compile_loads(tmp_path: Path) -> None:
    """A real fixture compile, in a fresh interpreter, loads only distributions the parity census measures.

    pxr and msgpack load lazily, so only a compile that reaches them names them (plan 14 Facts): one case
    converts a real NuRec appearance inline, and the destination-qualification case writes its probe.
    """

    measured = {name.replace("_", "-").lower() for name in census.COMPILE_DISTRIBUTIONS}
    seen: set[str] = set()
    for label, stage, appearance in (("nurec", "real_appearance", _real_nurec(tmp_path / "appearance.usdz")), ("probe", "stand_in", None)):
        host = Host(tmp_path / label)
        envelope, name = stage_compile(host, destination_support=label == "probe", qualification_only=label == "probe",
                                       appearance=appearance,
                                       appearance_name="appearance.usdz" if appearance else "appearance.usda")
        claimed, out = host.claim(name), tmp_path / f"{label}.json"
        run = subprocess.run(
            [sys.executable, "-c", _CLOSURE, str(claimed), str(host.inputs.resolve()), str(host.outputs.resolve()),
             envelope["expected_production_commit"], str(out), stage],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}", "PYTHONDONTWRITEBYTECODE": "1"})
        assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-4000:]
        report = json.loads(out.read_text(encoding="utf-8"))
        # The compile ran to the end, so every lazy import on its path happened.
        assert (report["status"], report["blockers"]) == ("compiled_for_production_launch", []), report
        seen |= {name.replace("_", "-").lower() for name in report["distributions"]}
    assert {"usd-core", "msgpack", "numpy", "jsonschema"} <= seen
    assert seen <= measured, sorted(seen - measured)


def test_dependency_zlib_or_python_mismatch_stays_on_host(tmp_path: Path) -> None:
    host = Host(tmp_path)
    _, name = host.stage()
    claimed = host.claim(name)
    # No probe has recorded the worker environment for this image: nothing to compare, so the host compiles.
    assert _plan(host, claimed) == remote.HostDecision("remote_ineligible:environment_unrecorded")
    mismatches = {
        "distributions": environment(distributions=[{"name": "numpy", "version": "1.26.4"}]),
        "golden_deflate": environment(golden_deflate={**HOST_RECORD["golden_deflate"],
                                                      "raw_deflate_level6_digest": "sha256:" + "e" * 64}),
        "python_version_info": environment(python_version_info=[3, 11, 9, "final", 0]),
        "golden_simd": environment(golden_simd={**HOST_RECORD["golden_simd"], "output_digest": "sha256:" + "f" * 64}),
    }
    for field, worker in mismatches.items():
        host.record_worker_environment(worker)
        assert _plan(host, claimed) == remote.HostDecision(f"remote_ineligible:environment_mismatch:{field}"), field
    # Only the CPU class may differ; the descriptor then names the worker's own environment digest.
    elsewhere = environment(cpu_class="sha256:" + "d" * 64)
    host.record_worker_environment(elsewhere)
    plan = _plan(host, claimed)
    assert isinstance(plan, remote.RemotePlan) and plan.environment_digest == elsewhere["environment_digest"]
    # A probe recorded for another image is drift until the config is re-pinned and re-probed (plan 14 §15).
    host.record_worker_environment(elsewhere, image=IMAGE.replace("d" * 64, "e" * 64))
    assert _plan(host, claimed) == remote.HostDecision("remote_ineligible:image_drift")
    host.record_worker_environment(elsewhere)
    remote.record_job_image(host.jobs, job_image=IMAGE.replace("d" * 64, "e" * 64), config_image=IMAGE, now=1.0)
    assert _plan(host, claimed) == remote.HostDecision("remote_ineligible:image_drift")
    remote.record_job_image(host.jobs, job_image=IMAGE, config_image=IMAGE, now=2.0)
    assert isinstance(_plan(host, claimed), remote.RemotePlan)


def _parity(host: Host, klass: str, passed: bool, *, index: int, image: str = IMAGE,
            host_digest: str | None = None, cpu_class: str | None = None) -> None:
    remote.record_shadow_parity(host.jobs, {
        "closure_class": klass, "attempt_id": f"rcj-ec-{'0' * 24}-a1-{index:032x}",
        "queue_row": {"queue": "task-evaluation-episode-compilations", "name": f"row-{index}.json",
                      "envelope_digest": "sha256:" + "1" * 64},
        "image": image, "host_environment_digest": host_digest or HOST_RECORD["environment_digest"],
        "worker_environment_digest": HOST_RECORD["environment_digest"],
        "cpu_class": cpu_class or HOST_RECORD["cpu_class"], "parity": "passed" if passed else "failed",
        "mismatches": [] if passed else ["native-task-arena-bundle.zip"], "compared_at_epoch": float(index)})


def test_closure_class_without_three_shadow_passes_stays_on_host(tmp_path: Path) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    _, name = host.stage()
    claimed = host.claim(name)
    unproven = remote.HostDecision("remote_ineligible:shadow_parity_unproven:not_applicable")
    assert _plan(host, claimed, gate=True) == unproven
    # Shadow mode is how passes accumulate: it never applies the gate.
    assert isinstance(_plan(host, claimed, gate=False), remote.RemotePlan)
    for index in range(2):
        _parity(host, "not_applicable", True, index=index + 1)
    # Passes for another class, image, host environment or CPU class do not count toward this one.
    _parity(host, "shipped", True, index=3)
    _parity(host, "not_applicable", True, index=4, image=IMAGE.replace("d" * 64, "e" * 64))
    _parity(host, "not_applicable", True, index=5, host_digest="sha256:" + "7" * 64)
    _parity(host, "not_applicable", True, index=6, cpu_class="sha256:" + "8" * 64)
    assert _plan(host, claimed, gate=True) == unproven
    _parity(host, "not_applicable", True, index=7)
    assert isinstance(_plan(host, claimed, gate=True), remote.RemotePlan)
    # A failed comparison restarts the count: three consecutive passes again.
    _parity(host, "not_applicable", False, index=8)
    assert _plan(host, claimed, gate=True) == unproven
    for index in (9, 10, 11):
        _parity(host, "not_applicable", True, index=index)
    assert isinstance(_plan(host, claimed, gate=True), remote.RemotePlan)
    # A failure compared in the same instant as a pass counts as the later one.
    identity = {"image": IMAGE, "host_environment_digest": HOST_RECORD["environment_digest"],
                "cpu_class": HOST_RECORD["cpu_class"]}
    _parity(host, "shipped", True, index=20)
    remote.record_shadow_parity(host.jobs, {**json.loads((host.jobs / "parity" / "episode_compilation" / (
        f"rcj-ec-{'0' * 24}-a1-{20:032x}.json")).read_text(encoding="utf-8")), "attempt_id": f"rcj-ec-{'0' * 24}-a1-{21:032x}",
        "parity": "failed", "record_digest": ""} | {"compared_at_epoch": 20.0})
    assert remote.shadow_passes(host.jobs, closure_class="shipped", **identity) == 0


def test_ephemeral_budget_overflow_stays_on_host(tmp_path: Path) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    _, name = host.stage(runtime_member_bytes=64 * 1024)
    claimed = host.claim(name)
    plan = _plan(host, claimed)
    assert isinstance(plan, remote.RemotePlan)
    inputs = sum(row["size_bytes"] for row in plan.inputs)
    # Inputs, three times the configured-scene assets (bundle, extraction, packet) and the runtime members.
    assert plan.ephemeral_bytes_required > inputs + 64 * 1024
    tight = remote_cpu_config(stages={"episode_compilation": {
        **remote_cpu_config()["stages"]["episode_compilation"], "ephemeral_bytes": plan.ephemeral_bytes_required - 1}})
    assert _plan(host, claimed, config=tight) == remote.HostDecision("remote_ineligible:ephemeral_budget_exceeded")
    roomy = remote_cpu_config(stages={"episode_compilation": {
        **remote_cpu_config()["stages"]["episode_compilation"], "ephemeral_bytes": plan.ephemeral_bytes_required}})
    assert isinstance(_plan(host, claimed, config=roomy), remote.RemotePlan)


def test_an_invalid_envelope_or_config_compiles_on_the_host(tmp_path: Path) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    _, name = host.stage()
    claimed = host.claim(name)
    assert _plan(host, claimed, config={"schema_version": "remote_cpu_workers_config.v1"}) == remote.HostDecision(
        "remote_ineligible:config_invalid")
    other = remote.plan_remote_compilation(
        claimed, inputs=host.inputs, outputs=host.outputs, source_commit="b" * 40, config=remote_cpu_config(),
        jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache, host_environment=HOST_RECORD)
    assert other == remote.HostDecision("remote_ineligible:envelope_invalid")
    reference = json.loads(claimed.read_text(encoding="utf-8"))["materialized_references"][2]
    Path(reference["materialized_path"]).chmod(0o640)
    Path(reference["materialized_path"]).write_bytes(b"changed after readback")
    assert _plan(host, claimed) == remote.HostDecision("remote_ineligible:envelope_invalid")


EXECUTION = "blueprint-remote-cpu-episode-compilation-x7k2p"


@pytest.mark.slow
@pytest.mark.parametrize(("policy_observation_override", "destination_support", "qualification_only"), COMPILE_CASES)
def test_remote_compile_is_byte_identical_to_host_compile(tmp_path: Path, monkeypatch, policy_observation_override: bool,
                                                          destination_support: bool, qualification_only: bool) -> None:
    """Plan 14 §5, acceptance 2: the host compile at R, then the worker's compile at the same R from the
    extracted recipe-v2 archive in a spawned stage child, landed whole: every path, byte and mode agree."""

    import io

    from blueprint_pipeline.remote_cpu_output_archive import land_subset
    from blueprint_pipeline.task_evaluation_episode_compilation_worker import process_episode_compilation_queue
    from blueprint_pipeline.task_evaluation_scene_construction_queue import _canonical_bytes
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    host = Host(tmp_path / "host")
    host.record_worker_environment()
    compiler = install_compile_stand_ins(monkeypatch.setattr)
    envelope, name = stage_compile(host, policy_observation_override=policy_observation_override,
                                   destination_support=destination_support, qualification_only=qualification_only)
    commit = envelope["expected_production_commit"]
    world = RemoteWorld(tmp_path / "remote")
    plan = remote.plan_remote_compilation(
        host.claim(name), inputs=host.inputs, outputs=host.outputs, source_commit=commit, config=world.config,
        jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache, host_environment=HOST_RECORD,
        require_shadow_gate=False)
    assert isinstance(plan, remote.RemotePlan), plan
    os.replace(host.queue / "processing" / name, host.queue / "pending" / name)

    # (1) The host compile at R, exactly as the no-spend unit runs it today.
    run = process_episode_compilation_queue(queue_root=host.queue, input_root=host.inputs, output_root=host.outputs,
                                            source_commit=commit, episode_compiler=compiler)
    assert run["results"][0]["status"] == "compiled_for_production_launch", run["results"][0]["blockers"]
    host_result = (host.queue / "results" / name).read_bytes()
    output_root = host.outputs / envelope["compilation_id"]
    snapshot = tree_snapshot(output_root)
    descriptor = world.descriptor(plan, queue_root=host.queue)
    environ = world.environ(descriptor, execution=EXECUTION)

    # (2) R, and every other host path, moves aside: the worker's /var/lib/blueprint starts empty.
    aside = tmp_path / "host" / "aside"
    os.rename(host.fs, aside)

    # (3) At the same R, the worker executes; its stage child spawns from the extracted release archive with
    # PYTHONPATH set only to that tree.
    assert world.execute(environ, filesystem_root=host.fs, measure=lambda: HOST_RECORD) == 0
    receipt = json.loads(world.staged(descriptor, "receipt.json"))
    assert receipt["status"] == "succeeded", (receipt["infrastructure_failures"], receipt["release_path_misses"])
    verdict = contract.validate_receipt(receipt, descriptor=descriptor, execution_name=EXECUTION)
    assert (verdict["outcome"], verdict["terminal"]) == ("succeeded", True)
    assert _canonical_bytes(receipt["result"]) == host_result

    # (4) Land all of it, from the archive and the host's own copies of host-known blobs, and compare.
    index, blobs = json.loads(world.staged(descriptor, "index.json")), world.staged(descriptor, "blobs.tar")
    landed = tmp_path / "landed" / envelope["compilation_id"]
    landed.parent.mkdir()
    sources = {row["digest"]: aside / Path(row["host_path"]).relative_to(host.fs) for row in plan.inputs}
    members = aside / OUTPUTS.lstrip("/") / "content-addressed" / "adapter-members" / "sha256"
    land_subset(index=index, reader=lambda offset, length: io.BytesIO(blobs[offset:offset + length]),
                host_sources=sources, destination_root=landed, selectors=["**"], member_store=members)
    assert tree_snapshot(landed) == snapshot
    # Only new bytes left the worker: input files and runtime members travelled as references.
    assert index["host_known"]["count"] > 0 and len(blobs) < sum(entry["size_bytes"] for entry in index["entries"])


_RUNTIME_CLOSURE = textwrap.dedent('''
    import json, sys
    from pathlib import Path

    claimed, inputs, outputs, commit, out = sys.argv[1:6]
    import tests.remote_cpu_worker_stages as stages
    from blueprint_pipeline import remote_cpu_worker

    module, _, name = remote_cpu_worker.STAGE_HANDLERS["episode_compilation"].partition(":")
    handler = getattr(__import__(module, fromlist=[name]), name)
    from blueprint_pipeline.task_evaluation_episode_compilation_worker import compile_claimed_envelope

    state, result = compile_claimed_envelope(Path(claimed), source_name=Path(claimed).name, inputs=Path(inputs),
                                             outputs=Path(outputs), source_commit=commit,
                                             episode_compiler=stages.install_compile_stand_ins(),
                                             disk_reservation_root=None, storage_pins_root=None)
    Path(out).write_text(json.dumps({"status": result["status"], "handler": handler.__name__,
                                     "loaded": sorted(sys.modules)}), encoding="utf-8")
''')


def test_episode_compilation_is_a_registered_worker_stage() -> None:
    from blueprint_pipeline import remote_cpu_worker as worker

    assert worker.STAGE_HANDLERS[remote.STAGE] == (
        "blueprint_pipeline.task_evaluation_episode_compilation_remote:run_episode_compilation_in_worker")
    source = Path(remote.__file__).read_text(encoding="utf-8")
    for authority in ("remote_cpu_job_allocator", "paid_resource_allocator", "cloud_run_jobs_client",
                      "remote_cpu_transport", "task_evaluation_configured_scene_object_store", "boto3",
                      "google.oauth2"):
        assert authority not in source, authority


@pytest.mark.slow
def test_a_worker_compile_loads_no_allocation_or_staging_authority(tmp_path: Path) -> None:
    """The handler runs inside the paid execution: a real compile through it loads no allocator, Cloud Run
    client, host staging, boto3 or google.oauth2.  (The compiler itself imports the admission chokepoint and
    the object-store module's key constants; neither holds or reaches a credential in the worker.)"""

    host = Host(tmp_path)
    envelope, name = stage_compile(host, destination_support=True, qualification_only=True)
    claimed, out = host.claim(name), tmp_path / "loaded.json"
    run = subprocess.run(
        [sys.executable, "-c", _RUNTIME_CLOSURE, str(claimed), str(host.inputs.resolve()), str(host.outputs.resolve()),
         envelope["expected_production_commit"], str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}", "PYTHONDONTWRITEBYTECODE": "1"})
    assert run.returncode == 0, run.stderr[-4000:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert (report["status"], report["handler"]) == ("compiled_for_production_launch", "run_episode_compilation_in_worker")
    loaded = set(report["loaded"])
    assert not {"blueprint_pipeline.paid_resource_allocator", "blueprint_pipeline.remote_cpu_job_allocator",
                "blueprint_pipeline.cloud_run_jobs_client", "blueprint_pipeline.remote_cpu_transport",
                "blueprint_pipeline.task_evaluation_episode_compilation_collector"} & loaded
    assert not {name for name in loaded if name == "boto3" or name.startswith(("boto3.", "google.oauth2"))}
