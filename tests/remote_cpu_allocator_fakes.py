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
            transport_bucket=self.bucket, object_store=(self.store, B2_BUCKET, B2_REGION))
        self.runs = 0

    def sleep(self, seconds: float) -> None:
        self.clock.advance(seconds)

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
                      command: list[str] | None = None, **changes: Any) -> dict[str, Any]:
    """The job plan 14 §14 declares: the bootstrap command, no args, one task, no retries."""

    return jobs.add_job(name, image=image, command=list(BOOTSTRAP if command is None else command), **changes)
