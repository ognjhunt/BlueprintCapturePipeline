"""Grant-gated Cloud Run Admin API v2 client for remote CPU jobs (plan 14 §2, §4, §8).

``run_job`` is the only call here that creates anything, and it requires the paid-resource grant
whose allocation binding names the attempt's descriptor, the job and the ``jobs.get`` etag it was
admitted against; the etag also rides on the request, so a job changed in between is refused.
Overrides carry only the four attempt identifiers, one task and the descriptor's timeout.
``cancel_execution`` only terminates.  Every request goes to ``run.googleapis.com`` through the
audited outbound boundary, authenticated as the dispatcher credential that systemd loads into the
paid unit (``LoadCredential=``); nothing falls back to application-default credentials.

``GcsTransportBucket`` is the dispatcher's view of the transport bucket: create-if-absent,
generation-pinned reads and deletes, and no listing.  google-auth and google-cloud-storage are
imported only when a credential or bucket is actually used.
"""

from __future__ import annotations

import json
import os
import re
import stat
import urllib.error
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode

from . import safe_outbound_http
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import require_paid_resource_admission_grant

CLOUD_RUN_CPU_JOB_RESOURCE_CLASS = "cloud_run_cpu_job"
CLOUD_RUN_API = "https://run.googleapis.com"
DISPATCHER_CREDENTIAL = "remote-cpu-dispatcher"
SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)
BOOTSTRAP_COMMAND = ("python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap")
ATTEMPT_VARIABLE = "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID"
DESCRIPTOR_VARIABLE = "BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256"
TRANSPORT_OBJECT_VARIABLE = "BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT"
TRANSPORT_GENERATION_VARIABLE = "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION"
OVERRIDE_VARIABLES = (ATTEMPT_VARIABLE, DESCRIPTOR_VARIABLE, TRANSPORT_OBJECT_VARIABLE, TRANSPORT_GENERATION_VARIABLE)
BINDING_SCHEMA_VERSION = "remote_cpu_job_allocation_binding.v1"
MAX_LISTING_PAGES = 1000
_MAX_CREDENTIAL_BYTES = 64 * 1024
_POLICY = safe_outbound_http.pinned_api_policy(CLOUD_RUN_API, max_response_bytes=16 * 1024 * 1024)
_PROJECT = r"(?:[a-z][a-z0-9-]{4,28}[a-z0-9]|[0-9]{1,20})"
_JOB_NAME = re.compile(rf"projects/{_PROJECT}/locations/[a-z]+-[a-z]+[0-9]+/jobs/[a-z](?:[a-z0-9-]{{0,61}}[a-z0-9])?")
_EXECUTION_NAME = re.compile(rf"(?:{_JOB_NAME.pattern})/executions/[a-z](?:[a-z0-9-]{{0,126}}[a-z0-9])?")
_ATTEMPT = re.compile(r"(rcj-[a-z]{2}-[0-9a-f]{24})-a[1-9][0-9]{0,2}-[0-9a-f]{32}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_GENERATION = re.compile(r"[1-9][0-9]{0,19}")
_BUCKET = r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]"

Transport = Callable[..., tuple[int, bytes]]


class CloudRunJobsError(RuntimeError):
    """A definite Cloud Run refusal or failure; ``status`` is the HTTP status when there was one."""

    def __init__(self, status: int | None, code: str) -> None:
        self.status, self.code = status, str(code)
        super().__init__(self.code if status is None else f"{self.code}:http_{status}")


class CloudRunAmbiguousResponse(CloudRunJobsError):
    """A mutation whose outcome is unknown: it may or may not have reached Cloud Run."""


def _named(value: Any, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise CloudRunJobsError(None, "cloud_run_resource_name_invalid")
    return value


def job_resource_name(*, project: str, region: str, job: str) -> str:
    return _named(f"projects/{project}/locations/{region}/jobs/{job}", _JOB_NAME)


def execution_resource_name(job: str, execution: str) -> str:
    return _named(f"{_named(job, _JOB_NAME)}/executions/{execution}", _EXECUTION_NAME)


def run_target(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """What ``run_job`` needs from a validated descriptor: the job, the attempt and its timeout."""

    execution = descriptor["execution"]
    return {"job": job_resource_name(project=execution["project"], region=execution["region"], job=execution["job"]),
            "attempt_id": descriptor["attempt_id"], "descriptor_digest": descriptor["descriptor_digest"],
            "timeout_seconds": descriptor["limits"]["task_timeout_seconds"]}


def allocation_binding(*, descriptor_digest: str, attempt_id: str, job: str, etag: str) -> dict[str, Any]:
    """The admission binding (plan 14 §8): descriptor digest, attempt id, job and ``jobs.get`` etag."""

    return {"schema_version": BINDING_SCHEMA_VERSION, "descriptor_digest": descriptor_digest,
            "attempt_id": attempt_id, "job": job, "job_etag": etag}


def env_value(resource: Mapping[str, Any], name: str) -> str | None:
    """One environment variable of a job's or an execution's container template."""

    template = resource.get("template") or {}
    template = template.get("template", template) if "containers" not in template else template
    for container in template.get("containers") or []:
        for variable in container.get("env") or []:
            if isinstance(variable, Mapping) and variable.get("name") == name:
                return variable.get("value")
    return None


def job_definition_blockers(job: Mapping[str, Any], *, image: str, timeout_seconds: int) -> list[str]:
    """``jobs.get`` must show the bootstrap command without args, one task, no retries and the timeout."""

    template = job.get("template") if isinstance(job.get("template"), Mapping) else {}
    task = template.get("template") if isinstance(template.get("template"), Mapping) else {}
    containers = task.get("containers") if isinstance(task.get("containers"), list) else []
    container = containers[0] if len(containers) == 1 and isinstance(containers[0], Mapping) else {}
    failed = {
        "etag": not isinstance(job.get("etag"), str) or not job["etag"],
        "containers": len(containers) != 1,
        "command": list(container.get("command") or []) != list(BOOTSTRAP_COMMAND),
        "args": bool(container.get("args")),
        # ``maxRetries`` sits in a proto3 oneof: an absent value is the default of three retries.
        "max_retries": task.get("maxRetries") != 0,
        "task_count": template.get("taskCount") != 1,
        "parallelism": template.get("parallelism", 0) not in (0, 1),
        "timeout": task.get("timeout") != f"{timeout_seconds}s",
    }
    blockers = [f"remote_cpu_job_definition_invalid:{name}" for name, failure in failed.items() if failure]
    if container.get("image") != image:
        blockers.append("remote_cpu_image_mismatch")
    return sorted(blockers)


def run_overrides(target: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """The only overrides a run may carry: the four identifiers, one task and the descriptor's timeout."""

    attempt = target.get("attempt_id")
    match = _ATTEMPT.fullmatch(attempt) if isinstance(attempt, str) else None
    if not isinstance(overrides, Mapping) or set(overrides) != set(OVERRIDE_VARIABLES):
        raise CloudRunJobsError(None, "remote_cpu_run_overrides_invalid:variables")
    if not all(isinstance(value, str) for value in overrides.values()) or match is None:
        raise CloudRunJobsError(None, "remote_cpu_run_overrides_invalid:values")
    transport = re.compile(rf"gs://{_BUCKET}/transport/{match.group(1)}/{re.escape(attempt)}-[0-9a-f]{{32}}\.json")
    failed = {
        "attempt": overrides[ATTEMPT_VARIABLE] != attempt,
        "descriptor": overrides[DESCRIPTOR_VARIABLE] != target.get("descriptor_digest")
        or not _DIGEST.fullmatch(overrides[DESCRIPTOR_VARIABLE]),
        "transport_object": transport.fullmatch(overrides[TRANSPORT_OBJECT_VARIABLE]) is None,
        "transport_generation": _GENERATION.fullmatch(overrides[TRANSPORT_GENERATION_VARIABLE]) is None,
    }
    timeout = target.get("timeout_seconds")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 3600:
        failed["timeout"] = True
    reasons = sorted(name for name, failure in failed.items() if failure)
    if reasons:
        raise CloudRunJobsError(None, f"remote_cpu_run_overrides_invalid:{reasons[0]}")
    return {"containerOverrides": [{"env": [{"name": name, "value": overrides[name]} for name in sorted(overrides)]}],
            "taskCount": 1, "timeout": f"{timeout}s"}


def load_dispatcher_credentials(*, environ: Mapping[str, str] | None = None,
                                factory: Callable[..., Any] | None = None) -> Any:
    """The service-account key systemd loaded as ``$CREDENTIALS_DIRECTORY/remote-cpu-dispatcher``.

    The file must be a regular, non-symlinked file with no group-write or other access.  Its bytes
    stay in memory; only ``google.oauth2.service_account`` (imported here, lazily) sees them.
    """

    directory = str((os.environ if environ is None else environ).get("CREDENTIALS_DIRECTORY") or "").strip()
    if not directory:
        raise CloudRunJobsError(None, "remote_cpu_dispatcher_credential_missing")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(os.path.join(directory, DISPATCHER_CREDENTIAL), flags)
    except FileNotFoundError:
        raise CloudRunJobsError(None, "remote_cpu_dispatcher_credential_missing") from None
    except OSError:
        raise CloudRunJobsError(None, "remote_cpu_dispatcher_credential_unsafe") from None
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_mode & 0o027 or status.st_size > _MAX_CREDENTIAL_BYTES:
            raise CloudRunJobsError(None, "remote_cpu_dispatcher_credential_unsafe")
        payload = stream.read(_MAX_CREDENTIAL_BYTES + 1)
    try:
        info = json.loads(payload)
    except ValueError:
        info = None
    if not isinstance(info, dict) or info.get("type") != "service_account":
        raise CloudRunJobsError(None, "remote_cpu_dispatcher_credential_invalid")
    if factory is None:
        from google.oauth2 import service_account

        factory = service_account.Credentials.from_service_account_info
    return factory(info, scopes=list(SCOPES))


def _default_transport(method: str, url: str, *, body: bytes | None, headers: Mapping[str, str]) -> tuple[int, bytes]:
    try:
        response = safe_outbound_http.request(url, method=method, data=body, headers=headers, policy=_POLICY,
                                              timeout_seconds=60)
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read(65536)
        except Exception:  # noqa: BLE001 - the status is what matters
            payload = b""
        return exc.code, payload
    return response.status, response.body


def _error_status(payload: bytes) -> str:
    try:
        error = json.loads(payload).get("error") or {}
        status = str(error.get("status") or "")
    except (ValueError, AttributeError):
        status = ""
    return status if re.fullmatch(r"[A-Z_]{1,64}", status) else "cloud_run_error"


class CloudRunJobsClient:
    """Cloud Run Admin API v2 jobs and executions for one dispatcher identity."""

    def __init__(self, *, credentials: Any = None, transport: Transport | None = None, page_size: int = 100) -> None:
        self._credentials, self._transport, self.page_size = credentials, transport or _default_transport, page_size

    def _token(self) -> str:
        if self._credentials is None:
            self._credentials = load_dispatcher_credentials()
        if not getattr(self._credentials, "valid", False):
            try:
                from google.auth.transport.requests import Request

                self._credentials.refresh(Request())
            except Exception:  # noqa: BLE001 - the refusal is typed; the cause may carry key material
                raise CloudRunJobsError(None, "remote_cpu_dispatcher_token_unavailable") from None
        token = getattr(self._credentials, "token", None)
        if not isinstance(token, str) or not token:
            raise CloudRunJobsError(None, "remote_cpu_dispatcher_token_unavailable")
        return token

    def _call(self, method: str, path: str, body: Mapping[str, Any] | None = None, *, mutation: bool = False,
              query: Mapping[str, Any] | None = None) -> dict[str, Any]:
        url = f"{CLOUD_RUN_API}/v2/{path}" + (f"?{urlencode(query)}" if query else "")
        safe_outbound_http.validate_outbound_url(url, policy=_POLICY)
        headers = {"Authorization": f"Bearer {self._token()}", "Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            status, payload = self._transport(method, url, body=data, headers=headers)
        except Exception as exc:  # noqa: BLE001 - the request may have reached Cloud Run
            refusal = CloudRunAmbiguousResponse if mutation else CloudRunJobsError
            raise refusal(None, f"cloud_run_response_lost:{type(exc).__name__}") from None
        if mutation and status >= 500:
            raise CloudRunAmbiguousResponse(status, _error_status(payload))
        if not 200 <= status < 300:
            raise CloudRunJobsError(status, _error_status(payload))
        try:
            value = json.loads(payload or b"{}")
        except ValueError:
            value = None
        if not isinstance(value, dict):
            raise (CloudRunAmbiguousResponse if mutation else CloudRunJobsError)(status, "cloud_run_response_invalid")
        return value

    def get_job(self, job: str) -> dict[str, Any]:
        return self._call("GET", _named(job, _JOB_NAME))

    def run_job(self, *, grant: Any, target: Mapping[str, Any], etag: str, overrides: Mapping[str, Any],
                validate_only: bool = False) -> dict[str, Any]:
        """Start one execution (or validate the request) under a grant bound to this descriptor and etag."""

        job = _named(target.get("job"), _JOB_NAME)
        if not isinstance(etag, str) or not etag:
            raise CloudRunJobsError(None, "remote_cpu_job_etag_missing")
        binding = allocation_binding(descriptor_digest=target.get("descriptor_digest"),
                                     attempt_id=target.get("attempt_id"), job=job, etag=etag)
        require_paid_resource_admission_grant(
            grant, resource_class=CLOUD_RUN_CPU_JOB_RESOURCE_CLASS,
            allocation_binding_digest=canonical_digest(binding), require_allocation_binding=True)
        body = {"etag": etag, "validateOnly": bool(validate_only), "overrides": run_overrides(target, overrides)}
        return self._call("POST", f"{job}:run", body, mutation=True)

    def get_execution(self, name: str) -> dict[str, Any]:
        return self._call("GET", _named(name, _EXECUTION_NAME))

    def list_all_executions(self, job: str) -> tuple[list[dict[str, Any]], int]:
        """Every execution of the job, following ``nextPageToken`` until it is empty; and the page count."""

        job, rows, token, pages, seen = _named(job, _JOB_NAME), [], None, 0, set()
        while True:
            page = self._call("GET", f"{job}/executions",
                              query={"pageSize": self.page_size, **({"pageToken": token} if token else {})})
            pages += 1
            executions = page.get("executions", [])
            if not isinstance(executions, list) or not all(isinstance(row, dict) for row in executions):
                raise CloudRunJobsError(None, "cloud_run_listing_invalid")
            rows.extend(executions)
            token = page.get("nextPageToken") or None
            if token is None:
                return rows, pages
            if not isinstance(token, str) or token in seen or pages >= MAX_LISTING_PAGES:
                raise CloudRunJobsError(None, "cloud_run_listing_unbounded")
            seen.add(token)

    def cancel_execution(self, name: str, *, etag: str | None = None) -> dict[str, Any]:
        """Terminate one execution; this never starts or changes anything else."""

        return self._call("POST", f"{_named(name, _EXECUTION_NAME)}:cancel", {"etag": etag} if etag else {},
                          mutation=True)


class GcsTransportBucket:
    """The transport bucket as the dispatcher sees it: create-if-absent, generation-pinned get and delete."""

    def __init__(self, name: str, *, credentials: Any, project: str) -> None:
        from google.cloud import storage

        if re.fullmatch(_BUCKET, str(name or "")) is None:
            raise CloudRunJobsError(None, "remote_cpu_transport_bucket_invalid")
        self.name = name
        self._bucket = storage.Client(project=project, credentials=credentials).bucket(name)

    def create(self, object_name: str, data: bytes, *, if_generation_match: int | None) -> int:
        blob = self._bucket.blob(object_name)
        blob.upload_from_string(bytes(data), content_type="application/json", if_generation_match=if_generation_match)
        return int(blob.generation)

    def get(self, object_name: str, *, generation: int) -> bytes:
        return self._bucket.blob(object_name, generation=generation).download_as_bytes()

    def exists(self, object_name: str, *, generation: int) -> bool:
        return bool(self._bucket.blob(object_name, generation=generation).exists())

    def delete(self, object_name: str, *, generation: int | None = None) -> None:
        self._bucket.blob(object_name, generation=generation).delete()
