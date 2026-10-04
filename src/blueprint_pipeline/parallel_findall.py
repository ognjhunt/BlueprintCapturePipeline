"""Offline FindAll request preparation and explicitly authenticated run reads.

This optional agent utility never creates, ingests, extends, enriches, or cancels
a run. Paid execution needs separate admission; no production lane uses it.
"""

from __future__ import annotations

import argparse
import getpass
import http.client
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import safe_outbound_http

RUNS_URL = "https://api.parallel.ai/v1beta/findall/runs"
AUTH_CHECK_URL = "https://api.parallel.ai/v1/monitors?limit=1"
API_KEY_ENV = "PARALLEL_API_KEY"
GENERATORS = frozenset({"preview", "base", "core", "pro"})
_RUN_ID = re.compile(r"findall_[A-Za-z0-9_-]+")
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class FindAllError(ValueError):
    """Secret-free validation or transport failure."""


def prepare_run(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimal official create body without credentials or I/O.

    Optional enrichment, webhook, memory, and exclusion fields are intentionally
    outside this initial contract. The returned envelope is reviewable input to
    a separately admitted official SDK/API call, never execution authorization.
    """
    fields = {"objective", "entity_type", "match_conditions", "generator", "match_limit"}
    if not isinstance(spec, Mapping) or not fields <= set(spec) or set(spec) - fields - {"metadata"}:
        raise FindAllError("findall_spec_requires_exact_minimal_fields")
    metadata = spec.get("metadata")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str)
            or type(value) not in (str, int, float, bool)
            or (type(value) is float and not math.isfinite(value))
            for key, value in metadata.items()
        )
    ):
        raise FindAllError("findall_spec_metadata_invalid")
    for name in ("objective", "entity_type"):
        if not isinstance(spec[name], str) or not spec[name].strip():
            raise FindAllError(f"findall_spec_invalid_{name}")
    generator = spec["generator"]
    if not isinstance(generator, str) or generator not in GENERATORS:
        raise FindAllError("findall_spec_invalid_generator")
    limit = spec["match_limit"]
    if type(limit) is not int or not 5 <= limit <= 1000:
        raise FindAllError("findall_spec_match_limit_must_be_5_to_1000")
    if generator == "preview" and limit > 10:
        raise FindAllError("findall_preview_limit_must_be_5_to_10")
    conditions = spec["match_conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise FindAllError("findall_spec_requires_match_conditions")
    names: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {"name", "description"}:
            raise FindAllError("findall_spec_invalid_match_condition")
        if not all(isinstance(v, str) and v.strip() for v in condition.values()):
            raise FindAllError("findall_spec_empty_match_condition")
        if condition["name"] in names:
            raise FindAllError("findall_spec_duplicate_condition_name")
        names.add(condition["name"])
    return {
        "method": "POST",
        "url": RUNS_URL,
        "body_json": json.loads(json.dumps(dict(spec))),
        "required_credential_binding": API_KEY_ENV,
        "required_auth_header": "x-api-key",
        "execution_authorized": False,
        "network_called": False,
    }


def _validate_run_id(findall_id: str) -> None:
    if not isinstance(findall_id, str) or not _RUN_ID.fullmatch(findall_id):
        raise FindAllError("findall_id_invalid")


def _validate_run(run: Any, findall_id: str) -> None:
    if not isinstance(run, dict) or run.get("findall_id") != findall_id:
        raise FindAllError("findall_response_run_identity_invalid")
    status = run.get("status")
    if (
        not isinstance(status, dict)
        or not isinstance(status.get("status"), str)
        or not status["status"]
        or type(status.get("is_active")) is not bool
    ):
        raise FindAllError("findall_response_status_invalid")


class FindAllClient:
    """Reads one explicitly selected run, preserving citations and unknown fields.

    Construction never discovers secrets. Supply a key only through a separately
    authorized runtime binding or a user's private, non-echoed terminal entry.
    """

    def __init__(self, api_key: str, *, timeout_seconds: float = 30.0) -> None:
        if not isinstance(api_key, str) or not api_key or any(c.isspace() for c in api_key):
            raise FindAllError("findall_runtime_key_invalid")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def _get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"x-api-key": self._api_key, "Accept": "application/json"},
        )
        try:
            response = safe_outbound_http.open_request(
                request,
                policy=safe_outbound_http.pinned_api_policy(
                    RUNS_URL, max_response_bytes=_MAX_RESPONSE_BYTES
                ),
                timeout_seconds=self._timeout_seconds,
                max_response_bytes=_MAX_RESPONSE_BYTES,
            )
            if not 200 <= response.status < 300:
                raise FindAllError(f"findall_http_error:{response.status}")
            payload = json.loads(response.body.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise FindAllError(f"findall_http_error:{exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            raise FindAllError("findall_transport_failed") from None
        except (UnicodeError, json.JSONDecodeError):
            raise FindAllError("findall_response_invalid_json") from None
        except safe_outbound_http.SafeOutboundHttpError:
            raise FindAllError("findall_outbound_policy_refused") from None
        if not isinstance(payload, dict):
            raise FindAllError("findall_response_must_be_object")
        return payload

    def _read(self, findall_id: str, suffix: str = "") -> dict[str, Any]:
        _validate_run_id(findall_id)
        return self._get_json(f"{RUNS_URL}/{findall_id}{suffix}")

    def auth_check(self) -> dict[str, Any]:
        """Check a standard API key with a documented, non-mutating list GET.

        Needs no run ID and returns no monitor contents or identifiers. Success
        proves access to this endpoint, not FindAll entitlement or a spend grant.
        """
        payload = self._get_json(AUTH_CHECK_URL)
        monitors = payload.get("monitors")
        if not isinstance(monitors, list) or not all(isinstance(m, dict) for m in monitors):
            raise FindAllError("findall_auth_check_response_invalid")
        return {
            "authenticated": True,
            "endpoint": AUTH_CHECK_URL,
            "verified_scope": "monitor_list_read",
            "findall_scope_verified": False,
            "paid_creation_authorized": False,
        }

    def status(self, findall_id: str) -> dict[str, Any]:
        payload = self._read(findall_id)
        _validate_run(payload, findall_id)
        return payload

    def result(self, findall_id: str) -> dict[str, Any]:
        """Return the current snapshot, including pending/non-matching candidates."""
        payload = self._read(findall_id, "/result")
        _validate_run(payload.get("run"), findall_id)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not all(isinstance(c, dict) for c in candidates):
            raise FindAllError("findall_response_candidates_invalid")
        return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Show binding name/presence only; no key read or network")
    prepare = commands.add_parser("prepare", help="Validate a JSON spec entirely offline")
    prepare.add_argument("spec", type=Path)
    for command in ("auth-check", "status", "result"):
        help_text = (
            "Check monitor-list authentication privately; FindAll scope remains unverified"
            if command == "auth-check" else "Read an existing, explicitly selected run"
        )
        read = commands.add_parser(command, help=help_text)
        if command != "auth-check":
            read.add_argument("findall_id")
        auth = read.add_mutually_exclusive_group(required=True)
        auth.add_argument("--prompt-key", action="store_true", help="Private user terminal entry")
        auth.add_argument(
            "--use-runtime-key", action="store_true",
            help="Use PARALLEL_API_KEY only after its runtime use is explicitly authorized",
        )
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            output = {
                "credential_binding": API_KEY_ENV,
                "binding_scope": "current_process_environment",
                "binding_present": API_KEY_ENV in os.environ,
                "credential_validated": False,
                "network_called": False,
                "paid_creation_supported": False,
            }
        elif args.command == "prepare":
            output = prepare_run(json.loads(args.spec.read_text(encoding="utf-8")))
        else:
            if args.command != "auth-check":
                _validate_run_id(args.findall_id)
            if args.prompt_key:
                if not sys.stdin.isatty():
                    raise FindAllError("findall_private_terminal_required")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", getpass.GetPassWarning)
                        key = getpass.getpass("Parallel API key (private; not saved): ")
                except getpass.GetPassWarning:
                    raise FindAllError("findall_private_terminal_echo_control_required") from None
            else:
                key = os.environ.get(API_KEY_ENV, "")
            client = FindAllClient(key)
            output = (
                client.auth_check() if args.command == "auth-check"
                else getattr(client, args.command)(args.findall_id)
            )
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (FindAllError, OSError, json.JSONDecodeError) as exc:
        # Never print a server body, transport exception, or user input on failure.
        message = str(exc) if isinstance(exc, FindAllError) else "findall_input_unreadable"
        print(json.dumps({"error": message}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
