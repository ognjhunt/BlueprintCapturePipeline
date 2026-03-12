"""Shared client for remote Stage 1 world-model backends."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


def _normalize_backend_name(raw: str) -> str:
    value = (raw or "").strip().lower().replace("-", "_")
    if value in {"neoverse"}:
        return "neoverse"
    if value in {"gen3c"}:
        return "gen3c"
    raise ValueError(f"unsupported world-model backend '{raw}'")


def _env(name: str) -> str:
    import os

    return (os.getenv(name) or "").strip()


@dataclass(frozen=True)
class WorldModelServiceConfig:
    backend: str
    service_url: str
    api_key: str
    timeout_seconds: int
    poll_interval_seconds: int

    @classmethod
    def from_env(cls, backend: str) -> "WorldModelServiceConfig":
        normalized_backend = _normalize_backend_name(backend)
        upper = normalized_backend.upper()
        service_url = _env(f"{upper}_SERVICE_URL") or _env("WORLD_MODEL_SERVICE_URL")
        api_key = _env(f"{upper}_SERVICE_API_KEY") or _env("WORLD_MODEL_SERVICE_API_KEY")
        timeout_value = _env("WORLD_MODEL_SERVICE_TIMEOUT_SECONDS") or "14400"
        poll_value = _env("WORLD_MODEL_SERVICE_POLL_SECONDS") or "20"
        return cls(
            backend=normalized_backend,
            service_url=service_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=max(1, int(timeout_value)),
            poll_interval_seconds=max(1, int(poll_value)),
        )


@dataclass(frozen=True)
class WorldModelJobResult:
    job_id: str
    status: str
    payload: Mapping[str, Any]
    result_manifest: Mapping[str, Any]
    latency_seconds: float


class WorldModelServiceClient:
    """HTTP client for async remote world-model execution."""

    def __init__(self, config: WorldModelServiceConfig) -> None:
        self.config = config

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib_request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib_request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail[:400]}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
        if not raw.strip():
            return {}
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"{method} {url} returned non-object JSON")
        return parsed

    def submit_job(
        self,
        *,
        backend: str,
        scene_id: str,
        capture_id: str,
        job_spec: Mapping[str, Any],
    ) -> str:
        if not self.config.service_url:
            raise RuntimeError(f"{self.config.backend} service URL is not configured")
        payload = {
            "backend": _normalize_backend_name(backend),
            "scene_id": scene_id,
            "capture_id": capture_id,
            "job_spec": job_spec,
        }
        response = self._request_json(
            method="POST",
            url=f"{self.config.service_url}/v1/world-model/jobs",
            payload=payload,
        )
        job_id = str(response.get("job_id") or "").strip()
        if not job_id:
            raise RuntimeError("world-model service submit response missing job_id")
        return job_id

    def poll_job(self, job_id: str) -> Mapping[str, Any]:
        return self._request_json(
            method="GET",
            url=f"{self.config.service_url}/v1/world-model/jobs/{urllib_parse.quote(job_id)}",
        )

    def load_result_manifest(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        manifest = payload.get("result_manifest")
        if isinstance(manifest, Mapping):
            return manifest

        location = ""
        for key in ("result_manifest_url", "result_manifest_uri", "result_manifest_path"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                location = value.strip()
                break
        if not location:
            raise RuntimeError("world-model job completed without result manifest")

        parsed = urllib_parse.urlparse(location)
        if parsed.scheme in {"http", "https"}:
            data = self._request_json(method="GET", url=location)
            return data
        if parsed.scheme == "file":
            path = Path(parsed.path)
        else:
            path = Path(location)
        if not path.is_file():
            raise RuntimeError(f"result manifest path not found: {location}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise RuntimeError(f"result manifest at {location} is not a JSON object")
        return loaded

    def wait_for_completion(
        self,
        *,
        backend: str,
        scene_id: str,
        capture_id: str,
        job_spec: Mapping[str, Any],
    ) -> WorldModelJobResult:
        started_at = time.monotonic()
        job_id = self.submit_job(
            backend=backend,
            scene_id=scene_id,
            capture_id=capture_id,
            job_spec=job_spec,
        )
        deadline = started_at + float(self.config.timeout_seconds)

        while True:
            payload = self.poll_job(job_id)
            status = str(payload.get("status") or "").strip().lower()
            if status in {"succeeded", "completed"}:
                result_manifest = self.load_result_manifest(payload)
                return WorldModelJobResult(
                    job_id=job_id,
                    status=status,
                    payload=payload,
                    result_manifest=result_manifest,
                    latency_seconds=max(0.0, time.monotonic() - started_at),
                )
            if status in {"failed", "cancelled", "canceled"}:
                detail = str(payload.get("error") or payload.get("detail") or "world-model job failed").strip()
                raise RuntimeError(f"{self.config.backend} job {job_id} failed: {detail}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{self.config.backend} job {job_id} timed out after {self.config.timeout_seconds}s"
                )
            time.sleep(float(self.config.poll_interval_seconds))
