"""Client for the persistent NeoVerse site-world runtime service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


@dataclass(frozen=True)
class SiteWorldRuntimeServiceConfig:
    service_url: str
    api_key: str
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "SiteWorldRuntimeServiceConfig":
        timeout_raw = _env("NEOVERSE_RUNTIME_SERVICE_TIMEOUT_SECONDS") or "120"
        return cls(
            service_url=_env("NEOVERSE_RUNTIME_SERVICE_URL").rstrip("/"),
            api_key=_env("NEOVERSE_RUNTIME_SERVICE_API_KEY"),
            timeout_seconds=max(1, int(timeout_raw)),
        )


class SiteWorldRuntimeServiceClient:
    def __init__(self, config: SiteWorldRuntimeServiceConfig) -> None:
        self.config = config

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not self.config.service_url:
            raise RuntimeError("NeoVerse runtime service URL is not configured")
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib_request.Request(
            f"{self.config.service_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib_request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail[:400]}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc
        parsed = json.loads(raw) if raw.strip() else {}
        if not isinstance(parsed, Mapping):
            raise RuntimeError(f"{method} {path} returned non-object JSON")
        return parsed

    def build_site_world(self, spec: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._request_json(method="POST", path="/v1/site-worlds", payload=spec)

    def get_site_world(self, site_world_id: str) -> Mapping[str, Any]:
        return self._request_json(
            method="GET",
            path=f"/v1/site-worlds/{urllib_parse.quote(site_world_id)}",
        )

    def get_site_world_health(self, site_world_id: str) -> Mapping[str, Any]:
        return self._request_json(
            method="GET",
            path=f"/v1/site-worlds/{urllib_parse.quote(site_world_id)}/health",
        )

    def create_session(
        self,
        site_world_id: str,
        *,
        robot_profile_id: str,
        task_id: str,
        scenario_id: str,
        start_state_id: str,
        notes: str = "",
    ) -> Mapping[str, Any]:
        return self._request_json(
            method="POST",
            path=f"/v1/site-worlds/{urllib_parse.quote(site_world_id)}/sessions",
            payload={
                "robot_profile_id": robot_profile_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "start_state_id": start_state_id,
                "notes": notes,
            },
        )

    def reset_session(
        self,
        session_id: str,
        *,
        task_id: str | None = None,
        scenario_id: str | None = None,
        start_state_id: str | None = None,
    ) -> Mapping[str, Any]:
        return self._request_json(
            method="POST",
            path=f"/v1/sessions/{urllib_parse.quote(session_id)}/reset",
            payload={
                "task_id": task_id,
                "scenario_id": scenario_id,
                "start_state_id": start_state_id,
            },
        )

