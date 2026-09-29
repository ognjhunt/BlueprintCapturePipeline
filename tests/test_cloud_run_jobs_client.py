# Covers (for impacted-test selection):
#   src/blueprint_pipeline/cloud_run_jobs_client.py
#   src/blueprint_pipeline/paid_resource_admission.py
#   tests/remote_cpu_allocator_fakes.py
"""ADP-009D/day-28, plan 14 PR 2: Cloud Run runs a remote CPU job only under a grant bound to it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from blueprint_pipeline import cloud_run_jobs_client as client_module
from blueprint_pipeline.cloud_run_jobs_client import (
    CLOUD_RUN_CPU_JOB_RESOURCE_CLASS,
    OVERRIDE_VARIABLES,
    CloudRunJobsClient,
    CloudRunJobsError,
    allocation_binding,
    job_definition_blockers,
    load_dispatcher_credentials,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PAID_RESOURCE_CLASSES,
    PaidResourceAdmissionBlocked,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from blueprint_pipeline.safe_outbound_http import SafeOutboundHttpError
from tests.remote_cpu_allocator_fakes import (
    IMAGE,
    JOB,
    TRANSPORT_BUCKET,
    CloudRunRest,
    FakeCredentials,
    add_bootstrap_job,
)
from tests.remote_cpu_fakes import FakeClock, FakeCloudRunJobs, env_value

ATTEMPT = "rcj-ec-" + "a" * 24 + "-a1-" + "b" * 32
DESCRIPTOR_DIGEST = "sha256:" + "d" * 64
TARGET = {"job": JOB, "attempt_id": ATTEMPT, "descriptor_digest": DESCRIPTOR_DIGEST, "timeout_seconds": 1800}
SERVICE_ACCOUNT = {
    "type": "service_account", "project_id": "blueprint-8c1ca",
    "client_email": "remote-cpu-dispatcher@blueprint-8c1ca.iam.gserviceaccount.com",
}


def _overrides(**changes) -> dict:
    overrides = {
        "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID": ATTEMPT,
        "BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256": DESCRIPTOR_DIGEST,
        "BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT":
            f"gs://{TRANSPORT_BUCKET}/transport/{ATTEMPT[:31]}/{ATTEMPT}-{'c' * 32}.json",
        "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION": "1234",
    }
    overrides.update(changes)
    return {name: value for name, value in overrides.items() if value is not None}


def _grant(binding: dict | None, *, resource_class: str = CLOUD_RUN_CPU_JOB_RESOURCE_CLASS):
    admission = build_paid_lane_admission(resource_class=resource_class)
    if binding is not None:
        admission.update({"allocation_binding": binding, "allocation_binding_digest": canonical_digest(binding)})
    return require_paid_resource_admission(
        admission, resource_class=resource_class, expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)


def _client() -> tuple[FakeCloudRunJobs, CloudRunRest, CloudRunJobsClient]:
    jobs = FakeCloudRunJobs(clock=FakeClock())
    add_bootstrap_job(jobs)
    rest = CloudRunRest(jobs)
    return jobs, rest, CloudRunJobsClient(credentials=FakeCredentials(), transport=rest)


def test_cloud_run_cpu_job_is_a_paid_resource_class() -> None:
    assert CLOUD_RUN_CPU_JOB_RESOURCE_CLASS == "cloud_run_cpu_job"
    assert "cloud_run_cpu_job" in PAID_RESOURCE_CLASSES
    admission = build_paid_lane_admission(resource_class="cloud_run_cpu_job")
    assert (admission["status"], admission["blockers"]) == ("admitted", [])
    grant = require_paid_resource_admission(
        admission, resource_class="cloud_run_cpu_job", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)
    assert grant.resource_class == "cloud_run_cpu_job"
    blocked = build_paid_lane_admission(resource_class="cloud_run_cpu_job", blockers=["remote_cpu_x"])
    with pytest.raises(PaidResourceAdmissionBlocked):
        require_paid_resource_admission(
            blocked, resource_class="cloud_run_cpu_job", expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION)


def test_run_job_requires_the_grant_bound_to_descriptor_and_etag() -> None:
    jobs, rest, client = _client()
    etag = client.get_job(JOB)["etag"]

    def bound(**changes) -> dict:
        return allocation_binding(**{"descriptor_digest": DESCRIPTOR_DIGEST, "attempt_id": ATTEMPT, "job": JOB,
                                     "etag": etag, **changes})

    refused = {
        "missing": None,
        "other_class": _grant(bound(), resource_class="cpu_build"),
        "unbound": _grant(None),
        "other_etag": _grant(bound(etag='"job-999"')),
        "other_descriptor": _grant(bound(descriptor_digest="sha256:" + "e" * 64)),
        "other_attempt": _grant(bound(attempt_id="rcj-ec-" + "a" * 24 + "-a2-" + "b" * 32)),
        "other_job": _grant(bound(job=JOB.replace("episode-compilation", "other"))),
    }
    for name, grant in refused.items():
        with pytest.raises(PaidResourceAdmissionBlocked):
            client.run_job(grant=grant, target=TARGET, etag=etag, overrides=_overrides())
        assert rest.mutations == [] and jobs.executions[JOB] == [], name

    operation = client.run_job(grant=_grant(bound()), target=TARGET, etag=etag, overrides=_overrides())
    assert [request.json()["etag"] for request in rest.runs()] == [etag]
    [execution] = jobs.executions[JOB]
    assert operation["metadata"]["name"] == execution["name"]
    assert env_value(jobs.get_execution(execution["name"]), "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID") == ATTEMPT

    # A job changed between jobs.get and :run is refused by its etag and creates nothing.
    jobs.update_job(JOB, timeout_seconds=1800)
    with pytest.raises(CloudRunJobsError) as stale:
        client.run_job(grant=_grant(bound()), target=TARGET, etag=etag, overrides=_overrides())
    assert stale.value.status == 409 and len(jobs.executions[JOB]) == 1


@pytest.mark.parametrize("overrides", [
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_STAGE="episode_compilation"), id="extra_variable"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION=None), id="missing_variable"),
    pytest.param({**_overrides(), "taskCount": "2"}, id="task_count"),
    pytest.param({**_overrides(), "args": "--stage"}, id="args"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_ATTEMPT_ID="rcj-ec-" + "a" * 24 + "-a2-" + "b" * 32), id="attempt"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256="sha256:" + "e" * 64), id="descriptor"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT="s3://bucket/transport/x.json"), id="not_gcs"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT=f"gs://{TRANSPORT_BUCKET}/transport/"
                            f"{ATTEMPT[:31]}/{ATTEMPT}-{'c' * 32}.json?generation=1"), id="query"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT=f"gs://{TRANSPORT_BUCKET}/transport/"
                            f"{ATTEMPT[:31]}/other-{'c' * 32}.json"), id="other_attempt_object"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION="0"), id="generation_zero"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION="12a"), id="generation_text"),
    pytest.param(_overrides(BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION=1234), id="generation_not_string"),
])
def test_client_refuses_overrides_other_than_the_four_identifiers(overrides: dict) -> None:
    jobs, rest, client = _client()
    etag = client.get_job(JOB)["etag"]
    grant = _grant(allocation_binding(descriptor_digest=DESCRIPTOR_DIGEST, attempt_id=ATTEMPT, job=JOB, etag=etag))
    with pytest.raises(CloudRunJobsError) as refused:
        client.run_job(grant=grant, target=TARGET, etag=etag, overrides=overrides)
    assert refused.value.code.startswith("remote_cpu_run_overrides_invalid")
    assert rest.mutations == [] and jobs.executions[JOB] == []

    client.run_job(grant=grant, target=TARGET, etag=etag, overrides=_overrides())
    [run] = rest.runs()
    assert run.json()["overrides"] == {
        "containerOverrides": [{"env": [{"name": name, "value": _overrides()[name]} for name in sorted(OVERRIDE_VARIABLES)]}],
        "taskCount": 1,
        "timeout": "1800s",
    }
    assert sorted(OVERRIDE_VARIABLES) == sorted(_overrides())


def test_job_definition_must_run_the_bootstrap_command_without_args() -> None:
    def blockers(**changes) -> list[str]:
        jobs = FakeCloudRunJobs()
        return job_definition_blockers(add_bootstrap_job(jobs, **changes), image=IMAGE, timeout_seconds=1800,
                                       vcpu=4, memory_bytes=16 * 1024**3)

    assert blockers() == []
    assert client_module.BOOTSTRAP_COMMAND == ("python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap")
    assert "remote_cpu_job_definition_invalid:command" in blockers(
        command=["python", "-m", "blueprint_pipeline.capture_orchestrator"])
    assert "remote_cpu_job_definition_invalid:command" in blockers(command=["python"], args=["-m", "x"])
    assert "remote_cpu_job_definition_invalid:args" in blockers(args=["--stage", "episode_compilation"])
    assert "remote_cpu_job_definition_invalid:max_retries" in blockers(max_retries=3)
    assert "remote_cpu_job_definition_invalid:task_count" in blockers(task_count=2)
    assert "remote_cpu_job_definition_invalid:timeout" in blockers(timeout_seconds=3600)
    assert blockers(image=IMAGE.replace("d" * 64, "e" * 64)) == ["remote_cpu_image_mismatch"]

    job = add_bootstrap_job(FakeCloudRunJobs())
    # proto3 JSON omits an unset oneof: a job without maxRetries retries three times.
    del job["template"]["template"]["maxRetries"]
    job.pop("etag")
    assert job_definition_blockers(job, image=IMAGE, timeout_seconds=1800, vcpu=4, memory_bytes=16 * 1024**3) == [
        "remote_cpu_job_definition_invalid:etag", "remote_cpu_job_definition_invalid:max_retries"]


def test_client_pins_the_run_googleapis_host_and_reads_the_loaded_credential(tmp_path: Path) -> None:
    directory = tmp_path / "credentials"
    directory.mkdir(mode=0o700)
    credential = directory / "remote-cpu-dispatcher"
    credential.write_text(json.dumps(SERVICE_ACCOUNT), encoding="utf-8")
    credential.chmod(0o400)
    decoy = tmp_path / "application-default.json"
    decoy.write_text(json.dumps({**SERVICE_ACCOUNT, "client_email": "decoy@blueprint-8c1ca.iam.gserviceaccount.com"}))
    seen: list[tuple[str, tuple[str, ...]]] = []

    def factory(info: dict, *, scopes: list[str]) -> FakeCredentials:
        seen.append((info["client_email"], tuple(scopes)))
        return FakeCredentials(token="loaded-dispatcher-token")

    environ = {"CREDENTIALS_DIRECTORY": str(directory), "GOOGLE_APPLICATION_CREDENTIALS": str(decoy)}
    credentials = load_dispatcher_credentials(environ=environ, factory=factory)
    assert seen == [(SERVICE_ACCOUNT["client_email"], ("https://www.googleapis.com/auth/cloud-platform",))]

    jobs = FakeCloudRunJobs(clock=FakeClock())
    add_bootstrap_job(jobs)
    rest = CloudRunRest(jobs)
    client = CloudRunJobsClient(credentials=credentials, transport=rest)
    client.get_job(JOB)
    assert client.list_all_executions(JOB) == ([], 1)
    for request in rest.requests:
        parts = urlsplit(request.url)
        assert (parts.scheme, parts.hostname, parts.port) == ("https", "run.googleapis.com", None)
        assert parts.path.startswith("/v2/projects/blueprint-8c1ca/locations/us-central1/jobs/")
        assert request.headers["Authorization"] == "Bearer loaded-dispatcher-token"

    # Only the credential systemd loaded counts; nothing falls back to application-default credentials.
    for broken, code in (({}, "remote_cpu_dispatcher_credential_missing"),
                         ({"CREDENTIALS_DIRECTORY": str(tmp_path / "absent")}, "remote_cpu_dispatcher_credential_missing")):
        with pytest.raises(CloudRunJobsError) as refused:
            load_dispatcher_credentials(environ={**broken, "GOOGLE_APPLICATION_CREDENTIALS": str(decoy)},
                                        factory=factory)
        assert refused.value.code == code
    credential.chmod(0o644)
    with pytest.raises(CloudRunJobsError) as readable:
        load_dispatcher_credentials(environ=environ, factory=factory)
    assert readable.value.code == "remote_cpu_dispatcher_credential_unsafe"
    credential.chmod(0o400)
    linked = tmp_path / "linked"
    linked.mkdir()
    os.symlink(credential, linked / "remote-cpu-dispatcher")
    with pytest.raises(CloudRunJobsError) as symlink:
        load_dispatcher_credentials(environ={"CREDENTIALS_DIRECTORY": str(linked)}, factory=factory)
    assert symlink.value.code == "remote_cpu_dispatcher_credential_unsafe"
    assert len(seen) == 1

    # The production transport refuses any other host before opening a connection.
    with pytest.raises(SafeOutboundHttpError):
        client_module._default_transport("GET", "https://run.googleapis.com.evil.example/v2/x", body=None, headers={})
    requests_before = len(rest.requests)
    for name in ("https://evil.example/v2/" + JOB, JOB + "/../../other", JOB.replace("blueprint-8c1ca", "x@evil"),
                 JOB + "?alt=media", JOB.replace("locations/us-central1", "locations/us-central1/")):
        with pytest.raises(CloudRunJobsError) as bad:
            client.get_job(name)
        assert bad.value.code == "cloud_run_resource_name_invalid"
    assert len(rest.requests) == requests_before


def test_job_definition_must_match_the_descriptors_cpu_and_memory() -> None:
    def blockers(limits: dict | None) -> list[str]:
        job = add_bootstrap_job(FakeCloudRunJobs())
        container = job["template"]["template"]["containers"][0]
        container.pop("resources", None)
        if limits is not None:
            container["resources"] = {"limits": limits}
        return job_definition_blockers(job, image=IMAGE, timeout_seconds=1800, vcpu=4, memory_bytes=16 * 1024**3)

    both = ["remote_cpu_job_definition_invalid:cpu", "remote_cpu_job_definition_invalid:memory"]
    assert blockers({"cpu": "4", "memory": "16Gi"}) == []
    assert blockers({"cpu": "4000m", "memory": str(16 * 1024**3)}) == []
    # A job twice the descriptor's size would be billed at twice the ledger's worst case.
    assert blockers({"cpu": "8", "memory": "32Gi"}) == both
    assert blockers({"cpu": "2", "memory": "16Gi"}) == ["remote_cpu_job_definition_invalid:cpu"]
    assert blockers({"cpu": "4", "memory": "16G"}) == ["remote_cpu_job_definition_invalid:memory"]  # decimal units
    assert blockers({"memory": "16Gi"}) == ["remote_cpu_job_definition_invalid:cpu"]  # unset means the default
    assert blockers(None) == both
    assert blockers({"cpu": "four", "memory": "16 Gi"}) == both
    assert add_bootstrap_job(FakeCloudRunJobs())["template"]["template"]["containers"][0]["resources"] == {
        "limits": {"cpu": "4", "memory": "16Gi"}}
