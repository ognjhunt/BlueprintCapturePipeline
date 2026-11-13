"""Small pure helpers shared by Vast offer-selection policy."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def geolocation_selection_kwargs(request: Mapping[str, Any]) -> dict[str, Any]:
    """Keep advisory and allocation geography filters identical."""
    result: dict[str, Any] = {
        "preferred_geolocation_regex": str(request.get("preferred_geolocation_regex") or "").strip(),
    }
    raw = request.get("allowed_geolocation_country_codes")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        countries = [str(item) for item in raw if str(item).strip()]
        if countries:
            result["allowed_geolocation_country_codes"] = countries
    return result


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def keyword_match_rank(value: Any, keywords: Sequence[str]) -> int:
    if not keywords:
        return 0
    haystack = _string(value).lower()
    return 0 if any(_string(keyword).lower() in haystack for keyword in keywords) else 1


def regex_match_rank(value: Any, pattern: str) -> int:
    text, regex = _string(value), _string(pattern)
    if not regex:
        return 0
    try:
        return 0 if re.search(regex, text, flags=re.IGNORECASE) else 1
    except re.error:
        return 1


def machine_id_set(values: Iterable[Any]) -> set[int]:
    result: set[int] = set()
    for value in values:
        number = _number(value)
        if number is not None:
            result.add(int(number))
    return result


def version_at_least(value: Any, minimum: str) -> bool:
    if not minimum:
        return True

    def version_tuple(candidate: Any) -> tuple[int, int, int] | None:
        parts = re.findall(r"\d+", _string(candidate))
        if not parts:
            return None
        numbers = [int(item) for item in parts[:3]]
        numbers.extend([0] * (3 - len(numbers)))
        return numbers[0], numbers[1], numbers[2]

    observed, required = version_tuple(value), version_tuple(minimum)
    if not observed or not required:
        return False
    width = max(len(observed), len(required))
    return observed + (0,) * (width - len(observed)) >= required + (0,) * (width - len(required))


def _approved_vast_offer_matches(approved: Any, observed: Any = None) -> bool:
    """Retain the preflight ask and every mandatory price before a paid create."""
    if not isinstance(approved, Mapping):
        return False
    for key in ("ask_contract_id", "machine_id", "disk_gb"):
        if type(approved.get(key)) is not int or approved[key] <= 0:
            return False
    prices = (
        "compute_hourly_rate_usd", "storage_hourly_rate_usd", "hourly_rate_usd",
        "provider_download_cost_per_gb_usd", "provider_upload_cost_per_gb_usd",
    )
    for key in prices:
        value = approved.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return False
    if approved["compute_hourly_rate_usd"] <= 0 or not math.isclose(
        approved["hourly_rate_usd"],
        approved["compute_hourly_rate_usd"] + approved["storage_hourly_rate_usd"],
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        return False
    if observed is None:
        return True
    if not isinstance(observed, Mapping):
        return False
    for key in ("ask_contract_id", "machine_id", "disk_gb"):
        if type(observed.get(key)) is not int or observed[key] != approved[key]:
            return False
    for key in prices:
        value = observed.get(key)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or value > approved[key]):
            return False
    return True


#: Provider-reported download speed (``inet_down``, megabits per second) at or
#: above which an offer is tried first when the launch admits a cold Isaac
#: image pull, right after the two driver ranks (driver support is correctness,
#: speed a preference). A multi-gigabyte digest-pinned image has to arrive well inside
#: the admitted cold-pull time; on 2026-11-12 two consecutive scene-configuration
#: launches rented the cheapest RTX 4090 offer with no speed check (machines
#: 25706 and 60701), the provider reported "still starting" for the whole
#: window and both ended `no_log_progress_timeout` at 20m51s. This only orders
#: the acceptable offers -- within equal driver ranks, fast first, then the
#: rest in today's order -- and never excludes one, so no launch that finds a
#: machine today finds none.
COLD_PULL_PREFERRED_MIN_DOWNLOAD_MBPS = 500.0


def download_speed_mbps(value: Any) -> float | None:
    """Vast's ``inet_down`` in megabits per second; missing or non-numeric is unknown."""

    speed = _number(value)
    if speed is None or not math.isfinite(speed) or speed < 0:
        return None
    return speed


def meets_download_speed_floor(offer: Mapping[str, Any] | None, floor_mbps: float) -> bool:
    """Known and at or above the floor; an unknown speed counts as below it."""

    speed = download_speed_mbps((offer or {}).get("provider_download_speed_mbps"))
    return speed is not None and speed >= float(floor_mbps)


def download_speed_sort_rank(offer: Mapping[str, Any], floor_mbps: float) -> int:
    """Ranked after the driver ranks: fast first under a floor; no floor ranks all alike.

    By decision it ranks after driver support (correctness) and AHEAD of the GPU
    keyword, location and reliability preferences: the image must arrive. With
    a zero floor every offer gets 0, so the (stable) sort is today's.
    """

    return 0 if not floor_mbps or meets_download_speed_floor(offer, floor_mbps) else 1


def cold_pull_download_speed_floor_mbps(
    image_startup_preflight: Mapping[str, Any], *, opted_in: bool = False
) -> float:
    """The download-speed preference for a launch that asked for it.

    Opt-in: zero (no preference, today's order) unless the launch asked for it
    (``prefer_fast_cold_pull_download``) AND requests the Isaac path, admits a
    cold image pull and has no proven cached template image.
    """

    if (
        opted_in is True
        and image_startup_preflight.get("isaac_path_requested") is True
        and image_startup_preflight.get("allow_cold_isaac_image_pull") is True
        and image_startup_preflight.get("template_image_cache_proven") is not True
    ):
        return COLD_PULL_PREFERRED_MIN_DOWNLOAD_MBPS
    return 0.0


def download_speed_manifest_fields(
    selected_offer: Mapping[str, Any] | None, prefer_min_download_mbps: float
) -> dict[str, Any]:
    """What the chosen machine reported, and the floor in force for this launch.

    The floor (and whether the selected offer met it) is null when no
    preference was in force. Kept in the offer-selection record so a later
    startup failure can be judged from records alone.
    """

    floor = float(prefer_min_download_mbps) if prefer_min_download_mbps else None
    return {
        "download_speed_floor_mbps": floor,
        "download_speed_preference_applied": floor is not None,
        "selected_offer_download_speed_mbps": (
            download_speed_mbps(selected_offer.get("provider_download_speed_mbps"))
            if selected_offer
            else None
        ),
        "selected_offer_meets_download_speed_floor": (
            meets_download_speed_floor(selected_offer, floor)
            if selected_offer and floor is not None
            else None
        ),
    }
