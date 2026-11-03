"""Test support for the remote team-scene stage: one control-plane host's team paths under a fake ``/``.

``TeamHost`` builds the team packet worker's real fixture world (``tests/test_team_scene_packet_worker._world``:
the rigid fixture revision, its configured bundle, the registry with its packet inputs, the capture partition
and one queued request) under ``tmp_path/fs/var/lib/blueprint/test``, so every path the compile reads maps to a
worker path under ``/var/lib/blueprint``, exactly as a Cloud Run execution's in-memory ``/var/lib/blueprint``
differs from the host's only by that root. ``TeamRemoteWorld`` stages a plan's inputs and the release source
in the B2 fake, seals the descriptor with the paid unit's own ``plan_descriptor``, mints the transport with the
allocator's ``mint_transport`` and runs the real worker, whose stage child spawns from the extracted release.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from blueprint_pipeline import team_scene_packet_remote as lane
from blueprint_pipeline import team_scene_packet_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.remote_cpu_environment import DIGESTED_FIELDS
from tests.remote_cpu_allocator_fakes import (
    GIB,
    IMAGE,
    JOB_SHORT,
    PROJECT,
    REGION,
    environment,
    remote_cpu_config,
    standing_authority,
)

TEAM_JOB_SHORT = "blueprint-remote-cpu-team-scene-packet"
TEAM_JOB = f"projects/{PROJECT}/locations/{REGION}/jobs/{TEAM_JOB_SHORT}"
JOBS = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
BASE = "/var/lib/blueprint/test"
CACHE = "/var/lib/blueprint/task-evaluation-inputs/particlefield-runtime-assets"
COMMIT = "a" * 40
EXECUTION = f"{TEAM_JOB_SHORT}-x7k2p"
HOST_RECORD = environment()
# 3600 s x (4 x $0.000018 + 16 x $0.000002) + 4 GiB x $0.12 (the stage's own output bound).
TEAM_WORST_CASE_USD = 0.8544


def team_stage(**changes: Any) -> dict[str, Any]:
    return {"job": TEAM_JOB_SHORT, "image": IMAGE, "vcpu": 4, "memory_bytes": 16 * GIB, "ephemeral_bytes": 10 * GIB,
            "task_timeout_seconds": 3600, **changes}


def team_config(**changes: Any) -> dict[str, Any]:
    """The host config with both stages: episode compilation as in production, and the proposed team stage."""

    episode = {"job": JOB_SHORT, "image": IMAGE, "vcpu": 4, "memory_bytes": 16 * GIB, "ephemeral_bytes": 10 * GIB,
               "task_timeout_seconds": 1800}
    return remote_cpu_config(**{"stages": {"episode_compilation": episode, "team_scene_packet": team_stage()},
                                **changes})


def team_authority(**changes: Any) -> dict[str, Any]:
    """The standing authority proposed for the team stage: $1 per attempt, $5 a day, $25 in total."""

    return standing_authority(**{"stages": ["team_scene_packet"], **changes})


def tree_snapshot(root: Path) -> dict[str, Any]:
    """Every file's ``(sha256, mode)`` and every directory's mode under ``root``, by relative path."""

    import hashlib
    import stat as stat_module

    files, folders = {}, {}
    for path in sorted(root.rglob("*")):
        relative, info = path.relative_to(root).as_posix(), path.lstat()
        if stat_module.S_ISDIR(info.st_mode):
            folders[relative] = oct(stat_module.S_IMODE(info.st_mode))
        else:
            assert not stat_module.S_ISLNK(info.st_mode), relative
            files[relative] = ("sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                               oct(stat_module.S_IMODE(info.st_mode)))
    return {"root_mode": oct(stat_module.S_IMODE(root.lstat().st_mode)), "files": files, "directories": folders}


class TeamHost:
    """The team worker's fixture world on one host, rooted at ``tmp_path/fs``, routed to the remote lane."""

    def __init__(self, tmp_path: Path, monkeypatch: Any, *, mode: str = "cloud_run", config: dict[str, Any] | None = None,
                 bodies: bool = False, clock: Any = None) -> None:
        from tests.test_team_scene_packet_worker import _world

        self.fs = tmp_path / "fs"
        self.base = self.local(BASE)
        self.base.mkdir(parents=True)
        self.world = _world(self.base, monkeypatch)
        self.jobs = self.local(JOBS)
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.config = config or team_config()
        self.calls: list[dict[str, Any]] = []
        self.clock = clock or (lambda: 2_000_000_000.0)
        self.record_worker_environment()
        if bodies:
            self.register_bodies()
        self.routing = self.dispatch(mode)

    def local(self, path: str) -> Path:
        return self.fs / path.lstrip("/")

    def dispatch(self, mode: str) -> lane.Dispatch:
        return lane.Dispatch(execution={"requested": mode, "effective": mode, "reason": "explicit", "findings": []},
                             jobs_root=self.jobs, config=self.config, source_commit=COMMIT, filesystem_root=self.fs,
                             cache_root=self.local(CACHE), host_environment=HOST_RECORD, clock=self.clock)

    def register_bodies(self) -> dict[str, Any]:
        """Two registered team bodies, one with its own robot model, as the worker test registers them."""

        from blueprint_pipeline import beta_policy_screening_launch as launch
        from tests.test_team_scene_packet import bodies
        from tests.test_team_scene_packet_worker import _registry

        body = bodies(self.base)
        records = {name: launch._record(launch._write(self.base / f"{name}.manifest.json", value))
                   for name, value in body.items()}
        arm = launch._record(self.base / "robot.usda")
        _registry(self.world["operator"], packet_inputs=self.world["inputs"], defaults={"screening_candidates": {
            "team_long": {"robot_policy": {"manifest": records["long"], "model": arm}},
            "team_rover": {"robot_policy": {"manifest": records["wheeled"]}}}})
        return body

    def transport(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"status": "accepted"}

    def process(self, routing: lane.Dispatch | None = None) -> dict[str, Any]:
        """One no-spend worker run."""

        store = self.world["scene"]["store"]
        return worker.process_team_scene_queue(queue_root=self.world["queue"],
                                               fetcher=lambda reference: store[reference["uri"]],
                                               offer_transport=self.transport, remote=routing or self.routing)

    def record_worker_environment(self, worker_record: dict[str, Any] | None = None, *, image: str = IMAGE) -> None:
        """What the allocator's preflight probe records for this stage's image."""

        record_value = HOST_RECORD if worker_record is None else worker_record
        record = {"schema_version": "remote_cpu_worker_environment.v1", "stage": lane.STAGE, "job": TEAM_JOB_SHORT,
                  "image": image, "environment_digest": record_value["environment_digest"],
                  "cpu_class": record_value["cpu_class"], "host_environment_digest": HOST_RECORD["environment_digest"],
                  "parity": {name: record_value.get(name) == HOST_RECORD.get(name) for name in DIGESTED_FIELDS},
                  "worker_environment": dict(record_value), "probe_attempt_id": "rcj-ep-" + "0" * 24 + "-a1-" + "0" * 32,
                  "receipt_digest": "sha256:" + "0" * 64, "recorded_at_epoch": 1.0}
        path = self.jobs / "environment" / f"{lane.STAGE}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        path.chmod(0o640)

    def status(self) -> dict[str, Any]:
        return json.loads((self.world["queue"] / "status" / f"{self.world['key']}.json").read_text(encoding="utf-8"))

    def state(self) -> str | None:
        found = [state for state in ("pending", "processing", "processed")
                 if (self.world["queue"] / state / f"{self.world['key']}.json").exists()]
        return found[0] if len(found) == 1 else None

    def marker(self, kind: str) -> tuple[Path, dict[str, Any]]:
        [(path, marker)] = lane.markers(self.jobs, kind)
        assert marker is not None
        return path, marker


class TeamRemoteWorld:
    """One B2 account, transport bucket and Cloud Run execution environment for a remote team compile."""

    def __init__(self, tmp_path: Path, *, config: dict[str, Any], clock: Any = None) -> None:
        from tests.remote_cpu_allocator_fakes import B2_BUCKET, T0, TRANSPORT_BUCKET
        from tests.remote_cpu_fakes import FakeArtifactStore, FakeClock, FakeTransportBucket
        from tests.remote_cpu_worker_support import RecordingHttp

        self.tmp_path, self.clock, self.config = tmp_path, clock or FakeClock(T0), config
        self.store = FakeArtifactStore(clock=self.clock, bucket=B2_BUCKET)
        self.bucket = FakeTransportBucket(TRANSPORT_BUCKET, clock=self.clock)
        self.http, self.bucket_name, self.logs = RecordingHttp(self.store), B2_BUCKET, []

    def release(self, commit: str) -> dict[str, Any]:
        """The recipe-v2 release of this working tree, staged in CAS as the paid unit stages it."""

        from blueprint_pipeline.remote_cpu_transport import SOURCE_FILENAME, SOURCE_KIND, cas_key
        from tests.remote_cpu_worker_support import digest_of, worker_release_archive

        archive = worker_release_archive(commit, entries=(
            "blueprint_pipeline.remote_cpu_worker", "blueprint_pipeline.team_scene_packet_remote"))
        key = cas_key(SOURCE_KIND, digest_of(archive), SOURCE_FILENAME)
        self.store.put_object(Bucket=self.bucket_name, Key=key, Body=archive,
                              Metadata={"sha256": digest_of(archive)[7:]})
        return {"source_commit": commit, "digest": digest_of(archive), "size_bytes": len(archive),
                "uri": f"s3://{self.bucket_name}/{key}"}

    def descriptor(self, plan: lane.RemotePlan, *, mode: str = "authoritative", attempt: int = 1) -> dict[str, Any]:
        from blueprint_pipeline.remote_cpu_transport import stage_inputs
        from blueprint_pipeline.team_scene_packet_collector import plan_descriptor
        from tests.remote_cpu_allocator_fakes import OBJECT_PREFIX

        staged = stage_inputs(lane.input_sources(plan), client=self.store, bucket=self.bucket_name)
        return plan_descriptor(plan, config=self.config, mode=mode, attempt=attempt, staged=staged,
                               source=self.release(plan.source_commit), object_prefix=OBJECT_PREFIX)

    def environ(self, descriptor: dict[str, Any], *, execution: str = EXECUTION) -> dict[str, str]:
        """Mint the attempt's transport as the allocator does; the execution's environment variables."""

        from blueprint_pipeline import remote_cpu_job_allocator as allocator
        from tests.remote_cpu_allocator_fakes import B2_REGION, TRANSPORT_BUCKET
        from tests.remote_cpu_worker_support import PREFIX_URL

        admission, grant = allocator.admit_remote_cpu_job(blockers=[], binding={"attempt_id": descriptor["attempt_id"]},
                                                          execute=True)
        name = f"gs://{TRANSPORT_BUCKET}/transport/{descriptor['job_id']}/{descriptor['attempt_id']}-{'a' * 32}.json"
        minted = allocator.mint_transport(
            grant=grant, binding_digest=admission["allocation_binding_digest"], descriptor=descriptor,
            bucket=self.bucket, object_store=(self.store, self.bucket_name, B2_REGION), clock=self.clock,
            object_uri=name)
        return {"CLOUD_RUN_JOB": TEAM_JOB_SHORT, "CLOUD_RUN_EXECUTION": execution, "CLOUD_RUN_TASK_INDEX": "0",
                "CLOUD_RUN_TASK_ATTEMPT": "0", "CLOUD_RUN_TASK_COUNT": "1", "PATH": os.environ.get("PATH", ""),
                "BLUEPRINT_REMOTE_CPU_STAGE": "team-scene-packet", "BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX": PREFIX_URL,
                "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID": descriptor["attempt_id"],
                "BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256": descriptor["descriptor_digest"],
                "BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT": name,
                "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION": str(minted["transport_generation"])}

    def execute(self, environ: dict[str, str], *, filesystem_root: Path, measure: Any, run_stage: Any = None) -> int:
        """Bootstrap, then execute in this process; the stage child is a real spawn from the release running the
        registered handler."""

        from blueprint_pipeline import remote_cpu_worker as worker_module

        common = {"environ": environ, "http": self.http, "filesystem_root": filesystem_root, "clock": self.clock,
                  "measure": measure, "log": self.logs.append}
        execute = worker_module.WorkerRuntime(**common, run_stage=run_stage)
        return worker_module.bootstrap(["bootstrap"], worker_module.WorkerRuntime(
            **common, reader=self.bucket.reader(),
            launch=lambda handoff: worker_module.execute_attempt(handoff, execute)))

    def staged(self, descriptor: dict[str, Any], name: str) -> bytes | None:
        from botocore.exceptions import ClientError

        key = descriptor["outputs"]["staging_prefix"].removeprefix(f"s3://{self.bucket_name}/") + name
        try:
            return self.store.get_object(Bucket=self.bucket_name, Key=key)["Body"].read()
        except ClientError:
            return None

    def land(self, descriptor: dict[str, Any], plan: lane.RemotePlan, destination: Path) -> dict[str, Any]:
        """Land the attempt's whole staged output at ``destination``, as the paid unit lands it."""

        from blueprint_pipeline.remote_cpu_output_archive import land_subset

        index, blobs = json.loads(self.staged(descriptor, "index.json")), self.staged(descriptor, "blobs.tar")
        sources = {row["digest"]: row["host_path"] for row in plan.inputs}
        land_subset(index=index, reader=lambda offset, length: io.BytesIO(blobs[offset:offset + length]),
                    host_sources=sources, destination_root=destination, selectors=["**"], member_store=None)
        return index


def sealed(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    value[digest_field] = canonical_digest(value, digest_field=digest_field)
    return value


TEAM_QUEUE = "team-scene-packets"
HALVES = "/var/lib/blueprint/task-evaluation-inputs/team-scene-halves/" + "c" * 32


def team_descriptor(config: dict[str, Any], *, attempt_name: str = "0123456789abcdef-01234567", attempt: int = 1,
                    nonce: str | None = None, environment_digest: str | None = None,
                    output_root: str | None = None, mode: str = "authoritative") -> dict[str, Any]:
    """A team-scene descriptor as the paid unit seals one, with synthetic content-addressed inputs."""

    import hashlib

    from blueprint_pipeline import remote_cpu_job_allocator as allocator
    from blueprint_pipeline import remote_cpu_job_contract as contract
    from tests.remote_cpu_allocator_fakes import CAS, HOST_ENVIRONMENT, OBJECT_PREFIX

    envelope = hashlib.sha256(attempt_name.encode()).hexdigest()
    name = f"{attempt_name}-{envelope}.json"
    limits = contract.stage_limits(config, "team_scene_packet")
    source = "5" * 64

    def row(role: str, contract_path: str, digit: str, path: str) -> dict[str, Any]:
        return {"role": role, "contract_path": contract_path, "digest": "sha256:" + digit * 64, "size_bytes": 4096,
                "mode": "0640", "materialize_at": path, "uri": f"{CAS}/remote-cpu-input/sha256/{digit * 64}/input.bin"}

    return contract.build_descriptor(
        config=config, stage="team_scene_packet", mode=mode, attempt=attempt,
        queue_row={"queue": TEAM_QUEUE, "name": name, "envelope_digest": "sha256:" + envelope},
        code={"source_commit": COMMIT, "image": IMAGE, "environment_digest": environment_digest or HOST_ENVIRONMENT,
              "source_archive": {"digest": "sha256:" + source, "size_bytes": 1234,
                                 "uri": f"{CAS}/remote-cpu-source/sha256/{source}/source.tar"}},
        environment={},
        inputs=[row("queue_envelope", "queue_envelope", "1", f"{JOBS}/envelopes/team_scene_packet/{name}"),
                row("materialized_reference", "configured_scene_bundle_path", "2",
                    "/var/lib/blueprint/launch-runs/run-1/configured_scene_publication/configured_scene_bundle.v1.zip"),
                row("materialized_reference", "documents.definition.json", "3",
                    f"{HALVES}/{attempt_name}/documents/definition.json")],
        outputs={"output_root": output_root or f"{HALVES}/{attempt_name}/scene", "declared_scratch": [],
                 "object_prefix": OBJECT_PREFIX},
        limits=limits, closure={"class": "not_applicable", "source_appearance_digest": None},
        spend={"worst_case_usd": allocator.worst_case_usd(limits=limits, rate_table=config["rate_table"]),
               "rate_table_digest": canonical_digest(config["rate_table"])}, nonce=nonce)


class Crash(BaseException):
    """A process death at a chosen point: nothing in the collector catches a ``BaseException``."""


class TeamCollectorWorld:
    """A host with one queued team request, a project of fakes (``RemoteCpuWorld``) and the team paid unit's
    collector, whose Cloud Run executions run the real worker at the host's own paths with the registered
    handler in-process. While a worker runs, the host tree moves aside, exactly as a Cloud Run execution's
    in-memory ``/var/lib/blueprint`` starts empty. ``compiles`` records where each compile ran."""

    def __init__(self, tmp_path: Path, monkeypatch: Any, *, mode: str = "cloud_run", authority: Any = None,
                 with_authority: bool = True, bodies: bool = False) -> None:
        from blueprint_pipeline import team_scene_packet
        from tests.remote_cpu_allocator_fakes import RemoteCpuWorld, add_bootstrap_job

        self.tmp_path, self.mode = tmp_path, mode
        self.remote = RemoteCpuWorld(tmp_path / "remote", monkeypatch, config=team_config(),
                                     authority=authority or team_authority(), with_authority=with_authority)
        add_bootstrap_job(self.remote.jobs, name=TEAM_JOB, timeout_seconds=3600)
        self.clock = self.remote.clock
        self.host = TeamHost(tmp_path / "host", monkeypatch, mode=mode, config=self.remote.config, bodies=bodies,
                             clock=lambda: self.clock.now)
        self.remote.root = self.host.jobs
        self.store = self.remote.store
        self.in_worker, self.compiles, self.executed, self.results = False, [], [], []
        self.stage_override: Any = None
        original = team_scene_packet.materialize_team_scene

        def recorded(**kwargs: Any) -> Any:
            self.compiles.append("worker" if self.in_worker else "host")
            return original(**kwargs)

        monkeypatch.setattr(team_scene_packet, "materialize_team_scene", recorded)

    # ------------------------------------------------------------------ the paid unit
    def allocate(self, argv: list[str]) -> dict[str, Any]:
        import argparse

        from blueprint_pipeline.remote_cpu_job_allocator import run_remote_cpu_job

        values: dict[str, Any] = {}
        index = 1
        while index < len(argv):
            if argv[index] == "--execute":
                values["--execute"] = True
                index += 1
            else:
                values[argv[index]] = argv[index + 1]
                index += 2
        Path(values["--out"]).parent.mkdir(parents=True, exist_ok=True)
        args = argparse.Namespace(action=values["--action"], stage=values["--stage"],
                                  descriptor=values.get("--descriptor"), lease=values["--lease"], out=values["--out"],
                                  execute=bool(values.get("--execute")))
        return run_remote_cpu_job(args, runtime=self.remote.runtime)

    def stage_release(self, object_store: Any, commit: str) -> dict[str, Any]:
        from blueprint_pipeline.remote_cpu_transport import SOURCE_FILENAME, SOURCE_KIND, cas_key
        from tests.remote_cpu_worker_support import digest_of, release_archive

        archive = release_archive(commit)
        key = cas_key(SOURCE_KIND, digest_of(archive), SOURCE_FILENAME)
        object_store[0].put_object(Bucket=object_store[1], Key=key, Body=archive,
                                   Metadata={"sha256": digest_of(archive)[7:]})
        return {"source_commit": commit, "digest": digest_of(archive), "size_bytes": len(archive),
                "uri": f"s3://{object_store[1]}/{key}"}

    def collector(self, **changes: Any) -> Any:
        from blueprint_pipeline import team_scene_packet_collector as collector

        values = {"runtime": self.remote.runtime, "config": self.remote.config, "jobs_root": self.host.jobs,
                  "queue_root": self.host.world["queue"], "source_commit": COMMIT, "mode": self.mode,
                  "allocate": self.allocate, "stage_release": self.stage_release, "filesystem_root": self.host.fs,
                  "execution": {"requested": self.mode, "effective": self.mode, "reason": "explicit", "findings": []},
                  **changes}
        return collector.Collector(**values)

    def collect(self, **changes: Any) -> dict[str, Any]:
        from blueprint_pipeline import team_scene_packet_collector as collector

        result = collector.run_collector(self.collector(**changes))
        self.results.append(result)
        return result

    # ------------------------------------------------------------------ Cloud Run
    def executions(self) -> list[dict[str, Any]]:
        return [self.remote.jobs.get_execution(row["name"]) for row in self.remote.jobs.executions[TEAM_JOB]]

    def _stage(self, *, runtime: Any, descriptor: dict[str, Any], release: Path, handler: str,
               seconds: float) -> dict[str, Any]:
        from blueprint_pipeline import remote_cpu_worker as worker_module

        roots = worker_module.StageRoots(runtime.filesystem_root, release)
        if self.stage_override is not None:
            return self.stage_override(descriptor, roots)
        return {"result": lane.run_team_scene_packet_in_worker(descriptor, roots), "release_path_misses": [],
                "failures": []}

    def run_workers(self) -> None:
        """Start the worker once in each running execution whose container would run it."""

        from blueprint_pipeline import remote_cpu_worker as worker_module
        from tests.remote_cpu_worker_support import PREFIX_URL, RecordingHttp

        for view in self.executions():
            name = view["name"].rsplit("/", 1)[1]
            behaviour = next(row["behaviour"] for row in self.remote.jobs.executions[TEAM_JOB]
                             if row["name"] == view["name"])
            if view["runningCount"] != 1 or name in self.executed or behaviour not in {"succeed", "duplicate"}:
                continue
            self.executed.append(name)
            container = view["template"]["containers"][0]
            environ = {**{row["name"]: row["value"] for row in container["env"]}, "CLOUD_RUN_JOB": view["job"],
                       "CLOUD_RUN_EXECUTION": name, "CLOUD_RUN_TASK_COUNT": "1",
                       "BLUEPRINT_REMOTE_CPU_STAGE": "team-scene-packet", worker_module.PREFIX_VARIABLE: PREFIX_URL}
            aside, tree = self.tmp_path / "host-aside", self.tmp_path / "workers" / name
            tree.parent.mkdir(exist_ok=True)
            os.rename(self.host.fs, aside)
            self.in_worker = True
            try:
                common = {"environ": environ, "http": RecordingHttp(self.store), "filesystem_root": self.host.fs,
                          "clock": self.clock, "measure": lambda: HOST_RECORD, "log": lambda line: None}
                execute = worker_module.WorkerRuntime(**common, run_stage=self._stage)
                assert worker_module.bootstrap(["bootstrap"], worker_module.WorkerRuntime(
                    **common, reader=self.remote.bucket.reader(),
                    launch=lambda handoff: worker_module.execute_attempt(handoff, execute))) == 0
            finally:
                self.in_worker = False
                if self.host.fs.exists():
                    os.rename(self.host.fs, tree)
                os.rename(aside, self.host.fs)

    def advance(self, seconds: float) -> None:
        remaining = float(seconds)
        while remaining > 0:
            self.clock.advance(min(30.0, remaining))
            remaining -= 30.0
            self.run_workers()

    def drive(self, *, until: Any, step: float = 60.0, limit: int = 200) -> None:
        """Run the collector, then let time pass and executions run, until ``until()`` holds."""

        for _ in range(limit):
            self.collect()
            if until():
                return
            self.advance(step)
        raise AssertionError(f"not reached after {limit} steps: {self.results[-1]}")

    # ------------------------------------------------------------------ observations
    def lease(self) -> dict[str, Any] | None:
        from blueprint_pipeline.remote_cpu_job_contract import job_id_for

        markers = [marker for kind in ("authoritative", "shadow", "fallback", "collected")
                   for _, marker in lane.markers(self.host.jobs, kind) if marker]
        names = {marker["queue_row"]["name"] for marker in markers} or set(self.names)
        self.names = sorted(names)
        leases = [self.host.jobs / "leases" / f"{job_id_for('team_scene_packet', name)}.json" for name in self.names]
        found = [json.loads(path.read_text(encoding="utf-8")) for path in leases if path.exists()]
        return found[-1] if found else None

    names: list[str] = []

    def terminal(self) -> bool:
        from blueprint_pipeline.remote_cpu_job_lease import TERMINAL_STATES

        lease = self.lease()
        return lease is not None and lease["state"] in TERMINAL_STATES
