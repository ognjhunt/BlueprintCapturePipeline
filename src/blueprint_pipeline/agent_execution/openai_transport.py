"""Thin HTTPS transport for the documented Agents API beta endpoints.

The pinned OpenAI Python dependency does not yet expose ``beta.agents``. This
adapter uses the published REST interface without changing the shared runtime
installation or adding a second model/tool orchestration implementation.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .contracts import AgentExecutionError, canonical_json


class AgentTransportError(AgentExecutionError):
    def __init__(self, code: str, *, status: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status = status

    @property
    def definitively_rejected(self) -> bool:
        # Request timeout/conflict and server errors do not establish absence.
        return self.status in {400, 401, 403, 404, 405, 413, 415, 422, 429}


class AgentsAPITransport(Protocol):
    project_id: str

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, str | int] | None = None,
    ) -> Mapping[str, Any]: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OpenAIAgentsHTTP:
    """No automatic mutation retries and no authorization-bearing redirects."""

    def __init__(self, *, api_key: str, project_id: str, timeout_seconds: float = 20) -> None:
        if not api_key.strip() or not project_id.strip():
            raise ValueError("agents_api_credentials_missing")
        if not 0 < timeout_seconds <= 60:
            raise ValueError("agents_api_timeout_invalid")
        self._api_key = api_key
        self.project_id = project_id
        self.timeout_seconds = timeout_seconds
        self._opener = build_opener(_NoRedirect())

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, str | int] | None = None,
    ) -> Mapping[str, Any]:
        if method not in {"GET", "POST", "DELETE"} or not path.startswith("/agents/"):
            raise ValueError("agents_api_request_not_allowed")
        if any(value in path for value in ("..", "?", "#", "\\")):
            raise ValueError("agents_api_request_path_invalid")
        url = "https://api.openai.com/v1" + path
        if query:
            url += "?" + urlencode(query)
        request = Request(
            url,
            method=method,
            data=canonical_json(body).encode("utf-8") if body is not None else None,
            headers={
                "Authorization": "Bearer " + self._api_key,
                "OpenAI-Project": self.project_id,
                "OpenAI-Beta": "agents=v1",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(16_000_001)
        except HTTPError as exc:
            # Never propagate a provider response body, request headers or key.
            raise AgentTransportError("agents_api_http_error", status=exc.code) from None
        except (URLError, OSError, TimeoutError):
            raise AgentTransportError("agents_api_connection_uncertain") from None
        if len(raw) > 16_000_000:
            raise AgentTransportError("agents_api_response_too_large")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError
            canonical_json(value)
        except (ValueError, UnicodeError):
            raise AgentTransportError("agents_api_response_invalid") from None
        return value
