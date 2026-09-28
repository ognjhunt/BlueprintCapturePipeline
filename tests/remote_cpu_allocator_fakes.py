"""Plan 14 PR 2 test support: drive the real Cloud Run jobs client against ``FakeCloudRunJobs``.

``CloudRunRest`` stands where ``safe_outbound_http`` would: it receives exactly the REST calls the
client makes to ``run.googleapis.com`` and routes them into the PR 1 fake, so the client's URL
building, overrides and error handling are exercised rather than bypassed.  A lost response
becomes a transport error after the fake has already acted, which is what makes it ambiguous.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

from tests.remote_cpu_fakes import FakeCloudRunError, FakeCloudRunJobs

PROJECT = "blueprint-8c1ca"
REGION = "us-central1"
JOB_SHORT = "blueprint-remote-cpu-episode-compilation"
JOB = f"projects/{PROJECT}/locations/{REGION}/jobs/{JOB_SHORT}"
IMAGE = "gcr.io/blueprint-8c1ca/blueprint-pipeline@sha256:" + "d" * 64
BOOTSTRAP = ["python", "-m", "blueprint_pipeline.remote_cpu_worker", "bootstrap"]
TRANSPORT_BUCKET = "blueprint-8c1ca-remote-cpu-transport"


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
