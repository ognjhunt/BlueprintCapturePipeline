"""The OpenAI organization Costs API: its pinned endpoint, the Admin key file and one GET.

``task_evaluation_supervisor.openai_cost_authority`` meters one candidate key
through this endpoint, key file and transport, and the capacity tick's spend
alert (``provider_spend_alerts``) reads organization-wide daily totals with
them. They live here, with no package imports, so that the alert does not load
the task-evaluation supervisor package (about 160 modules, among them the Vast
WAM runners). Everything here only reads. Errors carry fixed reason codes, never
the key, the headers or response text.
"""

from __future__ import annotations

import json
import math
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

OPENAI_COSTS_ENDPOINT = "https://api.openai.com/v1/organization/costs"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_PAGES = 200
DAY_SECONDS = 86_400

CostsTransport = Callable[[str, Mapping[str, str], float], Mapping[str, Any]]


class OpenAICostsError(ValueError):
    """A fixed reason code; ``http_status`` is set when OpenAI answered with an error status."""

    def __init__(self, code: str, *, http_status: int | None = None) -> None:
        super().__init__(code)
        self.http_status = http_status


def read_admin_key(path: Path) -> str:
    """The organization Admin key, from a regular file only its owner can read."""

    if path.is_symlink() or not path.is_file():
        raise OpenAICostsError("openai_admin_key_file_missing_or_symlink")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise OpenAICostsError("openai_admin_key_file_permissions_too_open")
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OpenAICostsError("openai_admin_key_file_unreadable") from exc
    if len(value) < 16 or any(character.isspace() for character in value):
        raise OpenAICostsError("openai_admin_key_invalid")
    return value


def get_costs_page(url: str, headers: Mapping[str, str], timeout_seconds: float) -> Mapping[str, Any]:
    """One GET of the pinned Costs endpoint, decoded to a JSON object."""

    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "api.openai.com"
        or parsed.path != "/v1/organization/costs"
    ):
        raise OpenAICostsError("openai_cost_endpoint_invalid")
    request = Request(url, headers=dict(headers), method="GET")
    try:
        # URL origin, scheme, and exact path are pinned immediately above.
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            status_code = int(getattr(response, "status", 200))
    except OSError as exc:
        status = exc.code if isinstance(exc, HTTPError) and isinstance(exc.code, int) else None
        raise OpenAICostsError("openai_cost_request_failed", http_status=status) from exc
    if status_code != 200 or len(payload) > MAX_RESPONSE_BYTES:
        raise OpenAICostsError("openai_cost_response_invalid")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAICostsError("openai_cost_response_invalid") from exc
    if not isinstance(decoded, Mapping):
        raise OpenAICostsError("openai_cost_response_invalid")
    return decoded


def _rows(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _usd(amount: Any) -> float:
    if not isinstance(amount, Mapping) or str(amount.get("currency") or "").lower() != "usd":
        raise OpenAICostsError("openai_cost_currency_invalid")
    value = amount.get("value")
    if value is None or isinstance(value, bool):
        raise OpenAICostsError("openai_cost_amount_invalid")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OpenAICostsError("openai_cost_amount_invalid") from exc
    if not math.isfinite(number) or number < 0:
        raise OpenAICostsError("openai_cost_amount_invalid")
    return number


def organization_daily_costs(
    *,
    admin_api_key_file: Path,
    start_time: int,
    end_time: int,
    transport: CostsTransport | None = None,
    timeout_seconds: float = 30.0,
) -> dict[int, float]:
    """USD per UTC day for the whole organization, keyed by bucket start.

    The endpoint reports whole UTC days only (``bucket_width=1d`` is its one
    width), so both ends must be UTC midnights. There is no project or key
    filter: every project in the organization counts. Any result that is not a
    USD ``organization.costs.result`` raises, so a malformed answer is never
    read as low spend.
    """

    if (
        isinstance(start_time, bool)
        or isinstance(end_time, bool)
        or not isinstance(start_time, int)
        or not isinstance(end_time, int)
        or start_time < 0
        or end_time <= start_time
        or start_time % DAY_SECONDS
        or end_time % DAY_SECONDS
        or end_time - start_time > 180 * DAY_SECONDS
    ):
        raise OpenAICostsError("openai_cost_query_window_invalid")
    headers = {
        "Authorization": f"Bearer {read_admin_key(admin_api_key_file)}",
        "Content-Type": "application/json",
    }
    base_params = [
        ("start_time", str(start_time)),
        ("end_time", str(end_time)),
        ("bucket_width", "1d"),
        ("limit", str((end_time - start_time) // DAY_SECONDS)),
    ]
    send = transport or get_costs_page
    totals: dict[int, float] = {}
    page: str | None = None
    seen: set[str] = set()
    for _ in range(MAX_PAGES):
        params = base_params + ([("page", page)] if page is not None else [])
        response = send(f"{OPENAI_COSTS_ENDPOINT}?{urlencode(params)}", headers, timeout_seconds)
        if not isinstance(response, Mapping) or response.get("object") != "page" or not _rows(response.get("data")):
            raise OpenAICostsError("openai_cost_page_schema_invalid")
        for bucket in response["data"]:
            if not isinstance(bucket, Mapping) or bucket.get("object") != "bucket":
                raise OpenAICostsError("openai_cost_bucket_schema_invalid")
            bucket_start, bucket_end = bucket.get("start_time"), bucket.get("end_time")
            if (
                isinstance(bucket_start, bool)
                or isinstance(bucket_end, bool)
                or not isinstance(bucket_start, int)
                or not isinstance(bucket_end, int)
                or bucket_end <= bucket_start
                or bucket_start < start_time
                or bucket_end > end_time
            ):
                raise OpenAICostsError("openai_cost_bucket_window_invalid")
            results = bucket.get("results")
            if not _rows(results):
                raise OpenAICostsError("openai_cost_result_schema_invalid")
            day = totals.get(bucket_start, 0.0)
            for result in results:
                if not isinstance(result, Mapping) or result.get("object") != "organization.costs.result":
                    raise OpenAICostsError("openai_cost_result_schema_invalid")
                day += _usd(result.get("amount"))
            totals[bucket_start] = day
        has_more = response.get("has_more")
        if not isinstance(has_more, bool):
            raise OpenAICostsError("openai_cost_pagination_invalid")
        if not has_more:
            return totals
        next_page = str(response.get("next_page") or "").strip()
        if not next_page or next_page in seen:
            raise OpenAICostsError("openai_cost_pagination_invalid")
        seen.add(next_page)
        page = next_page
    raise OpenAICostsError("openai_cost_pagination_exhausted")


__all__ = [
    "DAY_SECONDS",
    "OPENAI_COSTS_ENDPOINT",
    "OpenAICostsError",
    "get_costs_page",
    "organization_daily_costs",
    "read_admin_key",
]
