"""Client command for an independent calibrated action-consistency scorer."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from blueprint_pipeline import safe_outbound_http


URL_ENV = "BLUEPRINT_WAM_STRICT_SCORER_URL"
TOKEN_FILE_ENV = "BLUEPRINT_WAM_STRICT_SCORER_TOKEN_FILE"
INPUT_ENV = "BLUEPRINT_WAM_CONSISTENCY_INPUT"
OUTPUT_ENV = "BLUEPRINT_WAM_CONSISTENCY_OUTPUT"


def score_via_service(
    request: Mapping[str, object],
    *,
    url: str,
    token_file: str | Path,
    timeout_seconds: float = 600.0,
) -> dict[str, object]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("strict_consistency_scorer_requires_https")
    token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("strict_consistency_scorer_token_file_empty")
    strict = request.get("strict_action_aware_consistency")
    if not isinstance(strict, Mapping) or strict.get("required") is not True:
        raise ValueError("strict_action_aware_consistency_request_required")
    body = json.dumps(dict(request), sort_keys=True, separators=(",", ":")).encode()
    http_request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    response = safe_outbound_http.open_request(
        http_request,
        policy=safe_outbound_http.service_endpoint_policy(url),
        timeout_seconds=max(1.0, float(timeout_seconds)),
    )
    result = json.loads(response.body.decode("utf-8"))
    if not isinstance(result, Mapping):
        raise RuntimeError("strict_consistency_scorer_response_not_object")
    payload = dict(result)
    if payload.get("status") != "completed":
        raise RuntimeError("strict_consistency_scorer_response_not_completed")
    checks = payload.get("rollout_checks")
    if not isinstance(checks, list) or not checks:
        raise RuntimeError("strict_consistency_scorer_rollout_checks_missing")
    payload["client_transport"] = {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "token_source": "file",
        "raw_token_recorded": False,
    }
    return payload


def main() -> int:
    input_path = os.environ.get(INPUT_ENV, "")
    output_path = os.environ.get(OUTPUT_ENV, "")
    url = os.environ.get(URL_ENV, "")
    token_file = os.environ.get(TOKEN_FILE_ENV, "")
    if not all((input_path, output_path, url, token_file)):
        raise SystemExit(
            f"{INPUT_ENV}, {OUTPUT_ENV}, {URL_ENV}, and {TOKEN_FILE_ENV} are required"
        )
    request = json.loads(Path(input_path).read_text(encoding="utf-8"))
    result = score_via_service(request, url=url, token_file=token_file)
    Path(output_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
