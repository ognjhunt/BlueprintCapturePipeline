"""Plan 14 PR 2 test support: drive the real Cloud Run jobs client against ``FakeCloudRunJobs``.

``CloudRunRest`` stands where ``safe_outbound_http`` would: it receives exactly the REST calls the
client makes to ``run.googleapis.com`` and routes them into the PR 1 fake, so the client's URL
building, overrides and error handling are exercised rather than bypassed.  A lost response
becomes a transport error after the fake has already acted, which is what makes it ambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.remote_cpu_fakes import (
    FakeArtifactStore,
    FakeClock,
    FakeCloudRunError,
    FakeCloudRunJobs,
    FakeTransportBucket,
)

GIB = 1024**3
T0 = 2_000_000_000.0
PROJECT = "blueprint-8c1ca"
REGION = "us-central1"
JOB_SHORT = "blueprint-remote-cpu-episode-compilation"
JOB = f"projects/{PROJECT}/locations/{REGION}/jobs/{JOB_SHORT}"
IMAGE = "gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:" + "d" * 64
BOOTSTRAP = ["python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap"]
TRANSPORT_BUCKET = "blueprint-8c1ca-remote-cpu-transport"
B2_BUCKET = "b2-bucket"
B2_REGION = "us-west-004"
OBJECT_PREFIX = f"s3://{B2_BUCKET}/blueprint/arm-decision-proof-v1/configured-scenes"
CAS = f"{OBJECT_PREFIX}/artifacts"
HOST_ENVIRONMENT = "sha256:" + "4" * 64
QUEUE = "task-evaluation-episode-compilations"
ROWS = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations/processing"
COMPILED = "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes"
WORST_CASE_USD = 0.6672  # 1800 s x (4 x 0.000018 + 16 x 0.000002) + 4 GiB x 0.12


def seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = canonical_digest(value, digest_field=field)
    return value


def remote_cpu_config(**changes: Any) -> dict[str, Any]:
    stage = {"job": JOB_SHORT, "image": IMAGE, "vcpu": 4, "memory_bytes": 16 * GIB, "ephemeral_bytes": 10 * GIB,
             "task_timeout_seconds": 1800}
    config = {
        "schema_version": "remote_cpu_workers_config.v1", "project": PROJECT, "region": REGION,
        "transport_bucket": TRANSPORT_BUCKET, "stages": {"episode_compilation": stage},
        "rate_table": {"source": "cloud-run-jobs-and-premium-egress-list-prices", "observed_on": "2026-09-28",
                       "usd_per_vcpu_second": 0.000018, "usd_per_gib_second": 0.000002, "usd_per_egress_gib": 0.12},
        "max_live_executions": 2, "max_attempts": 2, "config_digest": "",
    }
    config.update(changes)
    return seal(config, "config_digest")


def standing_authority(**changes: Any) -> dict[str, Any]:
    authority = {
        "schema_version": "remote_cpu_standing_authorization.v1", "stages": ["episode_compilation"],
        "max_executions": 300, "max_attempt_usd": 1.0, "max_daily_usd": 5.0, "max_total_usd": 25.0,
        "expires_at_epoch": T0 + 30 * 86400, "authorized_by": "owner", "authorized_on": "2026-09-28",
        "authorization_reference": "plan-14-owner-decisions-2026-09-28", "authorization_digest": "",
    }
    authority.update(changes)
    return seal(authority, "authorization_digest")


def environment(**changes: Any) -> dict[str, Any]:
    """A ``remote_cpu_environment.v1`` record, sealed by PR 1's ``environment_digest``."""

    from blueprint_pipeline.remote_cpu_environment import environment_digest

    record: dict[str, Any] = {
        "schema_version": "remote_cpu_environment.v1", "python_version_info": [3, 12, 11, "final", 0],
        "golden_deflate": {"corpus_digest": "sha256:" + "1" * 64, "zlib_level6_digest": "sha256:" + "2" * 64,
                           "raw_deflate_level6_digest": "sha256:" + "3" * 64},
        "golden_simd": {"input_digest": "sha256:" + "5" * 64, "output_digest": "sha256:" + "6" * 64},
        "distributions": [{"name": "numpy", "version": "2.3.0"}], "cpu_class": "sha256:" + "c" * 64,
        "informational": {"python_version": "3.12.11", "machine": "x86_64", "cpu_flags_source": "/proc/cpuinfo"},
    }
    record.update(changes)
    record["environment_digest"] = environment_digest(record)
    return record


HOST_RECORD = environment()


def queue_name(label: str = "prep-1") -> str:
    return f"{label}-{hashlib.sha256(label.encode()).hexdigest()}.json"


def stage_descriptor(config: dict[str, Any], *, label: str = "prep-1", attempt: int = 1, nonce: str | None = None,
                     environment_digest: str = HOST_ENVIRONMENT) -> dict[str, Any]:
    """An episode-compilation descriptor as PR 4's eligibility will seal it."""

    from blueprint_pipeline import remote_cpu_job_allocator as allocator

    name = queue_name(label)
    envelope = name.removesuffix(".json").rsplit("-", 1)[1]
    limits = allocator.stage_limits(config, "episode_compilation")
    source = "5" * 64

    def cas(kind: str, digit: str, filename: str) -> str:
        return f"{CAS}/{kind}/sha256/{digit * 64}/{filename}"

    return contract.build_descriptor(
        config=config, stage="episode_compilation", mode="shadow", attempt=attempt,
        queue_row={"queue": QUEUE, "name": name, "envelope_digest": "sha256:" + envelope},
        code={"source_commit": "a" * 40, "image": IMAGE, "environment_digest": environment_digest,
              "source_archive": {"digest": "sha256:" + source, "size_bytes": 1234,
                                 "uri": f"{CAS}/remote-cpu-source/sha256/{source}/source.tar"}},
        environment={},
        inputs=[
            {"role": "queue_envelope", "contract_path": "queue_envelope", "digest": "sha256:" + envelope,
             "size_bytes": 2048, "mode": "0440", "materialize_at": f"{ROWS}/{name}",
             "uri": f"{CAS}/remote-cpu-input/sha256/{envelope}/{name}"},
            {"role": "materialized_reference", "contract_path": "execution_adapter.runtime_source_bundle",
             "digest": "sha256:" + "2" * 64, "size_bytes": 4096, "mode": "0440",
             "materialize_at": f"/var/lib/blueprint/task-evaluation-inputs/prepared-references/{label}/runtime.zip",
             "uri": cas("remote-cpu-input", "2", "runtime.zip")},
        ],
        outputs={"output_root": f"{COMPILED}/{label}", "declared_scratch": [f"{COMPILED}/content-addressed/"],
                 "object_prefix": OBJECT_PREFIX},
        limits=limits, closure={"class": "not_applicable", "source_appearance_digest": None},
        spend={"worst_case_usd": allocator.worst_case_usd(limits=limits, rate_table=config["rate_table"]),
               "rate_table_digest": canonical_digest(config["rate_table"])},
        nonce=nonce,
    )


class RecordingArtifactStore(FakeArtifactStore):
    """The B2 fake, also recording every presign it is asked for (method, key, lifetime)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.presigned: list[tuple[str, str, int]] = []

    def generate_presigned_url(self, ClientMethod: str, Params: Mapping[str, Any] | None = None,
                               ExpiresIn: int = 3600, HttpMethod: str | None = None) -> str:
        url = super().generate_presigned_url(ClientMethod, Params=Params, ExpiresIn=ExpiresIn, HttpMethod=HttpMethod)
        self.presigned.append((ClientMethod, str((Params or {}).get("Key")), ExpiresIn))
        return url


class RemoteCpuWorld:
    """One host, one project and one B2 account of fakes, wired into the allocator's runtime."""

    def __init__(self, tmp_path: Path, monkeypatch: Any, *, config: dict[str, Any] | None = None,
                 authority: dict[str, Any] | None = None, with_authority: bool = True) -> None:
        from blueprint_pipeline.cloud_run_jobs_client import CloudRunJobsClient
        from blueprint_pipeline.remote_cpu_job_allocator import RemoteCpuRuntime

        self.tmp_path, self.clock = tmp_path, FakeClock(T0)
        self.jobs = FakeCloudRunJobs(clock=self.clock)
        add_bootstrap_job(self.jobs)
        self.rest = CloudRunRest(self.jobs)
        self.bucket = FakeTransportBucket(TRANSPORT_BUCKET, clock=self.clock)
        self.store = RecordingArtifactStore(clock=self.clock, bucket=B2_BUCKET)
        self.root = tmp_path / "remote-cpu-jobs"
        self.spend = tmp_path / "spend-authority"
        monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(self.spend))
        self.config_path = tmp_path / "etc" / "remote-cpu-workers.json"
        self.config = config or remote_cpu_config()
        self.write_config(self.config)
        if with_authority:
            self.write_authority(authority or standing_authority())
        self.runtime = RemoteCpuRuntime(
            config_path=str(self.config_path), clock=self.clock, sleep=self.sleep,
            cloud_run=CloudRunJobsClient(credentials=FakeCredentials(), transport=self.rest),
            transport_bucket=self.bucket, object_store=(self.store, B2_BUCKET, B2_REGION),
            stage_release_source=source_reference, host_environment=lambda: HOST_RECORD,
            presigned_put=lambda url, data: self.store.request("PUT", url, body=data).status, poll_seconds=10.0)
        self.runs = 0
        self.on_sleep: Any = None

    def sleep(self, seconds: float) -> None:
        self.clock.advance(seconds)
        if self.on_sleep is not None:
            self.on_sleep()

    def write_config(self, config: dict[str, Any], *, mode: int = 0o640) -> None:
        self.config = config
        _write_private(self.config_path, config, mode)

    def write_authority(self, authority: dict[str, Any], *, mode: int = 0o600) -> None:
        _write_private(self.spend / "authorizations" / "remote-cpu-standing-authorization.v1.json", authority, mode)

    def record_environment(self, *, stage: str = "episode_compilation", digest: str = HOST_ENVIRONMENT,
                           image: str = IMAGE) -> None:
        _write_private(self.root / "environment" / f"{stage}.json", {
            "schema_version": "remote_cpu_worker_environment.v1", "stage": stage, "job": JOB_SHORT, "image": image,
            "environment_digest": digest, "cpu_class": None, "parity": True}, 0o640)

    def descriptor(self, **changes: Any) -> dict[str, Any]:
        return stage_descriptor(self.config, **changes)

    def run(self, action: str, *, descriptor: dict[str, Any] | None = None, execute: bool = True,
            stage: str = "episode_compilation") -> dict[str, Any]:
        from blueprint_pipeline.remote_cpu_job_allocator import run_remote_cpu_job

        self.runs += 1
        path = None
        if descriptor is not None:
            path = self.tmp_path / "descriptors" / f"{descriptor['attempt_id']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(descriptor), encoding="utf-8")
        out = self.tmp_path / "out" / f"{self.runs:03d}-{action}.json"
        args = argparse.Namespace(action=action, stage=stage, descriptor=None if path is None else str(path),
                                  lease=str(self.root), out=str(out), execute=execute)
        result = run_remote_cpu_job(args, runtime=self.runtime)
        assert json.loads(out.read_text(encoding="utf-8")) == result
        return result

    def consumed(self) -> list[Path]:
        return sorted((self.spend / "consumed").glob("remote-cpu-*.json"))

    def lease(self, descriptor: dict[str, Any]) -> dict[str, Any] | None:
        path = self.root / "leases" / f"{descriptor['job_id']}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def assert_untouched(self) -> None:
        """No execution, Cloud Run mutation, transport object, B2 operation or presign, and nothing consumed."""

        assert self.rest.mutations == [] and all(not rows for rows in self.jobs.executions.values())
        assert self.bucket._objects == {} and self.store.operations == [] and self.store.presigned == []
        assert self.consumed() == []


def source_reference(_object_store: Any = None) -> dict[str, Any]:
    """What PR 3's release-source staging returns: the commit and its content-addressed archive."""

    digest = hashlib.sha256(b"release-source.tar").hexdigest()
    return {"source_commit": "a" * 40, "digest": "sha256:" + digest, "size_bytes": 18,
            "uri": f"{CAS}/remote-cpu-source/sha256/{digest}/source.tar"}


class FakeWorker:
    """What the host can observe of PR 3's worker: it reads its transport at the pinned generation,
    PUTs heartbeats through its presigned URLs, and uploads the receipt last.

    ``environment`` is the record the worker reports; ``receipt=False`` never uploads one.
    """

    def __init__(self, world: RemoteCpuWorld, *, environment: Mapping[str, Any], receipt: bool = True) -> None:
        self.world, self.environment, self.receipt = world, dict(environment), receipt
        self.heartbeats: dict[str, int] = {}
        self.receipted: set[str] = set()
        world.on_sleep = self.step

    def step(self) -> None:
        for rows in list(self.world.jobs.executions.values()):
            for execution in rows:
                view = self.world.jobs.get_execution(execution["name"])
                if view["runningCount"] == 1:
                    self._act(view)

    def _env(self, view: Mapping[str, Any], name: str) -> str | None:
        from tests.remote_cpu_fakes import env_value

        return env_value(view, name)

    def _act(self, view: Mapping[str, Any]) -> None:
        name = view["name"].rsplit("/", 1)[1]
        transport = json.loads(self.world.bucket.reader().get(
            self._env(view, "BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT").split("/", 3)[3],
            generation=int(self._env(view, "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION"))))
        descriptor = transport["descriptor"]
        assert descriptor["descriptor_digest"] == self._env(view, "BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256")
        sequence = self.heartbeats[name] = self.heartbeats.get(name, 0) + 1
        heartbeat = {"schema_version": "remote_cpu_job_heartbeat.v1", "attempt_id": descriptor["attempt_id"],
                     "execution_name": name, "sequence": sequence, "phase": "stage",
                     "elapsed_seconds": 10.0 * sequence, "bytes_fetched": 100, "bytes_uploaded": 0}
        self._put(transport["outputs"]["heartbeat.json"], heartbeat)
        if not self.receipt or name in self.receipted:
            return
        self.receipted.add(name)
        result = seal({"schema_version": "remote_cpu_environment_probe_result.v1", "status": "environment_recorded",
                       "blockers": [], "source_commit": descriptor["code"]["source_commit"],
                       "environment": self.environment, "result_digest": ""}, "result_digest")
        index = json.dumps({"environment.json": {"blob": "sha256:" + "9" * 64}}).encode()
        self._put(transport["outputs"]["index.json"], index)
        self._put(transport["outputs"]["blobs.tar"], b"\0" * 1024)
        receipt = seal({
            "schema_version": "remote_cpu_job_receipt.v1", "job_id": descriptor["job_id"], "attempt": descriptor["attempt"],
            "attempt_id": descriptor["attempt_id"], "stage": descriptor["stage"],
            "descriptor_digest": descriptor["descriptor_digest"], "execution_name": name, "status": "succeeded",
            "result": result,
            "output": {"format": "remote_cpu_output.v1", "paths_total": 1, "bytes_total": 1024,
                       "index": {"digest": "sha256:" + hashlib.sha256(index).hexdigest(), "size_bytes": len(index)},
                       "archive": {"digest": "sha256:" + hashlib.sha256(b"\0" * 1024).hexdigest(), "size_bytes": 1024},
                       "host_known": {"count": 0, "bytes": 0}},
            "infrastructure_failures": [], "release_path_misses": [],
            "environment": {"environment_digest": self.environment["environment_digest"],
                            "cpu_class": self.environment["cpu_class"]},
            "phases": {"bootstrap": 1.0, "fetch": 2.0, "stage": 3.0, "seal_upload": 1.0},
            "bytes_fetched": 100, "bytes_uploaded": 2048, "private_url_recorded": False, "receipt_digest": "",
        }, "receipt_digest")
        self._put(transport["outputs"]["receipt.json"], receipt)

    def _put(self, url: str, value: Any) -> None:
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        assert self.world.store.request("PUT", url, body=body).status == 200


def _write_private(path: Path, value: Mapping[str, Any], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(mode)


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None

    def json(self) -> dict[str, Any]:
        return json.loads(self.body) if self.body else {}


class FakeCredentials:
    """What ``google.oauth2.service_account.Credentials`` offers the client: a token and refresh."""

    def __init__(self, token: str = "fake-dispatcher-token", *, valid: bool = True) -> None:
        self.token, self.valid, self.refreshes = token, valid, 0

    def refresh(self, _request: Any) -> None:
        self.refreshes += 1
        self.valid = True


class CloudRunRest:
    """The Cloud Run Admin API v2 REST surface, backed by a ``FakeCloudRunJobs``."""

    def __init__(self, jobs: FakeCloudRunJobs) -> None:
        self.jobs = jobs
        self.requests: list[RecordedRequest] = []

    @property
    def mutations(self) -> list[RecordedRequest]:
        return [request for request in self.requests if request.method != "GET"]

    def runs(self, *, validate_only: bool | None = None) -> list[RecordedRequest]:
        return [request for request in self.mutations if request.url.endswith(":run")
                and (validate_only is None or bool(request.json().get("validateOnly")) is validate_only)]

    def __call__(self, method: str, url: str, *, body: bytes | None, headers: Mapping[str, str]) -> tuple[int, bytes]:
        self.requests.append(RecordedRequest(method, url, dict(headers), body))
        parts = urlsplit(url)
        assert (parts.scheme, parts.netloc) == ("https", "run.googleapis.com") and parts.path.startswith("/v2/")
        path = unquote(parts.path.removeprefix("/v2/"))
        query = {name: values[0] for name, values in parse_qs(parts.query).items()}
        payload = json.loads(body) if body else {}
        try:
            if method == "POST" and path.endswith(":run"):
                result = self.jobs.run_job(path.removesuffix(":run"), etag=payload.get("etag"),
                                           overrides=payload.get("overrides"),
                                           validate_only=bool(payload.get("validateOnly")))
            elif method == "POST" and path.endswith(":cancel"):
                result = self.jobs.cancel_execution(path.removesuffix(":cancel"), etag=payload.get("etag"))
            elif method == "GET" and path.endswith("/executions"):
                result = self.jobs.list_executions(path.removesuffix("/executions"),
                                                   page_size=int(query["pageSize"]) if "pageSize" in query else None,
                                                   page_token=query.get("pageToken"))
            elif method == "GET" and "/executions/" in path:
                result = self.jobs.get_execution(path)
            elif method == "GET":
                result = self.jobs.get_job(path)
            else:
                return 404, b'{"error": {"code": 404, "status": "NOT_FOUND"}}'
        except FakeCloudRunError as exc:
            if exc.status is None:
                raise ConnectionResetError("connection reset after the request was sent") from None
            return exc.status, json.dumps({"error": {"code": exc.status, "status": exc.code}}).encode()
        return 200, json.dumps(result).encode()


def add_bootstrap_job(jobs: FakeCloudRunJobs, *, name: str = JOB, image: str = IMAGE,
                      command: list[str] | None = None, cpu: str = "4", memory: str = "16Gi",
                      **changes: Any) -> dict[str, Any]:
    """The job plan 14 §14 declares: the bootstrap command, no args, one task, no retries, 4 vCPU and 16 GiB."""

    jobs.add_job(name, image=image, command=list(BOOTSTRAP if command is None else command), **changes)
    jobs.jobs[name]["template"]["template"]["containers"][0]["resources"] = {"limits": {"cpu": cpu, "memory": memory}}
    return jobs.get_job(name)
