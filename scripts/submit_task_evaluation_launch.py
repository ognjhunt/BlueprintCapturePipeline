#!/usr/bin/env python3
"""Submit one signed Task Evaluation launch request to the production intake.

The website is the normal trigger. This is the same request over the same
canonical HMAC-signed endpoint, so an operator or a scheduled routine can drive
a production run without reconstructing the signing scheme by hand -- which is
otherwise tribal knowledge that has to be rediscovered from the service source
every time.

It performs no provider mutation: the intake queues the request and the
canonical allocator, owned by the dispatcher, remains the only boundary that
can spend. Rights, spend, and execution authority must be supplied by the
caller; this tool never invents them.
"""

from __future__ import annotations

import argparse
import hmac
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.task_evaluation_launch_dispatcher import canonical_digest

DEFAULT_CLIENT_ID = "blueprint-webapp"
SIGNATURE_HEADER = "X-Blueprint-Pipeline-Signature"
TIMESTAMP_HEADER = "X-Blueprint-Pipeline-Timestamp"
NONCE_HEADER = "X-Blueprint-Pipeline-Nonce"
CLIENT_ID_HEADER = "X-Blueprint-Pipeline-Client-Id"


class LaunchSubmissionError(ValueError):
    """Raised when a launch request cannot be built or signed."""


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise LaunchSubmissionError(f"expected a JSON object: {path}")
    return dict(value)


def build_launch_request(
    *,
    profile: Mapping[str, Any],
    launch_id: str,
    run_id: str,
    actor_id: str,
    actor_role: str,
    rights_scope: str,
    rights_uri: str,
    rights_digest: str,
    max_spend_usd: float,
    authority_window_hours: float,
    now: datetime,
) -> dict[str, Any]:
    """Bind a request to one exact published profile.

    The spend ceiling is the caller's authorization, and the dispatcher
    separately refuses any profile whose allocator ceiling exceeds it.
    """

    expires_at = now + timedelta(hours=authority_window_hours)
    request: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": launch_id,
        "run_id": run_id,
        "idempotency_key": launch_id,
        "launch_profile_id": profile["profile_id"],
        "launch_profile_digest": profile["profile_digest"],
        "source_bundle": profile["source_bundle"],
        "evaluation_run_spec": profile["evaluation_run_spec"],
        "required_controls": profile["required_controls"],
        "claim_ceiling": profile["claim_ceiling"],
        "authorization": {
            "actor": {"id": actor_id, "role": actor_role},
            "authorized_at": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "rights": {
                "approved": True,
                "scope": rights_scope,
                "evidence": {"uri": rights_uri, "digest": rights_digest},
            },
            "execution": {"approved": True},
            "spend": {
                "approved": True,
                "currency": "USD",
                "max_spend_usd": max_spend_usd,
                "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            },
        },
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def signed_headers(
    *, secret: str, client_id: str, body: bytes, now: datetime, nonce: str
) -> dict[str, str]:
    """Build the intake's canonical signature headers.

    The timestamp must be ISO-8601; the service parses it with
    ``datetime.fromisoformat`` and rejects epoch seconds outright. The signed
    payload is ``{timestamp}.{client_id}.{nonce}.`` followed by the raw body,
    so the exact bytes posted must be the bytes signed.
    """

    if not secret:
        raise LaunchSubmissionError("intake signing secret is empty")
    timestamp = now.isoformat()
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{client_id}.{nonce}.".encode("utf-8") + body,
        "sha256",
    ).hexdigest()
    return {
        "content-type": "application/json",
        TIMESTAMP_HEADER: timestamp,
        NONCE_HEADER: nonce,
        CLIENT_ID_HEADER: client_id,
        SIGNATURE_HEADER: f"sha256={digest}",
    }


def submit(
    *, endpoint: str, headers: Mapping[str, str], body: bytes, timeout: float = 30.0
) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(endpoint)
    loopback_http = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "::1",
        "localhost",
    }
    if (
        (parsed.scheme != "https" and not loopback_http)
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise LaunchSubmissionError("intake endpoint must be HTTPS or loopback HTTP")
    request = urllib.request.Request(
        endpoint, data=body, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(  # nosec B310 - endpoint validated above
            request, timeout=timeout
        ) as response:
            return {"http_status": response.status, "body": json.loads(response.read())}
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, "body": exc.read().decode("utf-8")[:2000]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="published launch profile JSON")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--secret-file", required=True, help="file holding the intake secret")
    parser.add_argument("--client-id", default=DEFAULT_CLIENT_ID)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--actor-role", default="ops")
    parser.add_argument("--rights-scope", required=True)
    parser.add_argument("--rights-uri", required=True)
    parser.add_argument("--rights-digest", required=True)
    parser.add_argument("--max-spend-usd", type=float, required=True)
    parser.add_argument("--authority-window-hours", type=float, default=3.0)
    parser.add_argument("--request-out")
    args = parser.parse_args(argv)

    try:
        profile = _read_json(args.profile)
        secret = Path(args.secret_file).expanduser().read_text(encoding="utf-8").strip()
        now = datetime.now(timezone.utc)
        request = build_launch_request(
            profile=profile,
            launch_id=args.launch_id,
            run_id=args.run_id,
            actor_id=args.actor_id,
            actor_role=args.actor_role,
            rights_scope=args.rights_scope,
            rights_uri=args.rights_uri,
            rights_digest=args.rights_digest,
            max_spend_usd=args.max_spend_usd,
            authority_window_hours=args.authority_window_hours,
            now=now,
        )
        body = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = signed_headers(
            secret=secret,
            client_id=args.client_id,
            body=body,
            now=now,
            nonce=uuid.uuid4().hex,
        )
        if args.request_out:
            Path(args.request_out).expanduser().write_text(
                json.dumps(request, indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
        result = submit(endpoint=args.endpoint, headers=headers, body=body)
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    # No secret value is ever echoed: only the non-secret client identity.
    print(
        json.dumps(
            {
                "status": "submitted" if result["http_status"] == 202 else "rejected",
                "http_status": result["http_status"],
                "launch_id": args.launch_id,
                "client_id": args.client_id,
                "response": result["body"],
                "provider_mutation_performed_by_this_tool": False,
            },
            sort_keys=True,
        )
    )
    return 0 if result["http_status"] == 202 else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
