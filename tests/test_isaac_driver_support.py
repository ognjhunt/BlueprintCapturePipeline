"""Pin the attempt-069 driver ceiling: newer is not better for a pinned runtime.

Attempt 069 allocated an L40S on driver 595.71.05 and Isaac never created a GPU
device (``ERROR_INCOMPATIBLE_DRIVER``, ``review_canary_timeout_renderer_never_ready``).
The selector had no upper bound -- it blocklisted only the too-old 570 window and
ranked ``major >= 580`` as the most preferred branch, so 595 sorted first. Because
595 hosts are also cheaper, cost broke the tie toward the broken driver on every
retry: the second allocation drew 595.71.05 again.

Every driver that has actually rendered was on the 580 branch, so these tests use
the real observed versions rather than invented ones.
"""

from __future__ import annotations

import pytest

from blueprint_pipeline.isaac_driver_support import (
    ABOVE_CEILING_STATUS,
    MAX_ISAAC_DRIVER_ENV,
    SUPPORTED_STATUS,
    driver_excluded_count,
    driver_newer_branch_sort_rank,
    driver_version_tuple,
    isaac_driver_support_status,
    meets_max_isaac_driver,
    resolve_max_isaac_driver,
)

# Drivers observed on hosts that rendered successfully (attempts 065-068).
PROVEN_GOOD = ("580.119.02", "580.159.03", "580.159.04")
# The driver observed on attempt 069, which could not initialize the renderer.
BROKEN_TOO_NEW = "595.71.05"


def _offer(driver: str, **extra):
    return {"driver_version": driver, "hourly_rate_usd": 0.471, **extra}


@pytest.mark.parametrize("driver", PROVEN_GOOD)
def test_proven_580_branch_drivers_stay_admitted(driver: str) -> None:
    """The ceiling must not cost us the only hosts known to work."""

    ceiling = resolve_max_isaac_driver()
    assert meets_max_isaac_driver(_offer(driver), ceiling) is True
    assert isaac_driver_support_status(driver) == SUPPORTED_STATUS


def test_attempt_069_driver_is_now_rejected_before_allocation() -> None:
    """The exact host that burned attempt 069 is no longer rentable."""

    ceiling = resolve_max_isaac_driver()
    assert meets_max_isaac_driver(_offer(BROKEN_TOO_NEW), ceiling) is False
    assert isaac_driver_support_status(BROKEN_TOO_NEW) == ABOVE_CEILING_STATUS


def test_cheaper_broken_driver_no_longer_outranks_proven_one() -> None:
    """Cost must not break the tie toward a driver that cannot render.

    Attempt 069 drew 595 twice because both branches ranked 0 and 595 was
    cheaper. The ceiling rank must order the proven branch strictly first.
    """

    good = {"driver_version": "580.159.03"}
    broken = {"driver_version": BROKEN_TOO_NEW}
    assert driver_newer_branch_sort_rank(good) < driver_newer_branch_sort_rank(broken)


def test_ceiling_counts_excluded_offers_for_admission_evidence() -> None:
    ceiling = resolve_max_isaac_driver()
    summaries = [_offer(d) for d in (*PROVEN_GOOD, BROKEN_TOO_NEW, BROKEN_TOO_NEW)]
    assert driver_excluded_count(summaries, ceiling) == 2


def test_unreported_driver_is_admitted_not_guessed() -> None:
    """An absent version is not evidence of an incompatible one.

    The live renderer gate still fails closed on the box itself, so guessing
    here would only shrink the pool without adding safety.
    """

    ceiling = resolve_max_isaac_driver()
    assert meets_max_isaac_driver({"hourly_rate_usd": 0.5}, ceiling) is True
    assert meets_max_isaac_driver({"driver_version": ""}, ceiling) is True


def test_env_override_can_raise_or_disable_the_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly qualified branch must be adoptable without a code change."""

    monkeypatch.setenv(MAX_ISAAC_DRIVER_ENV, "600.0.0")
    assert meets_max_isaac_driver(_offer(BROKEN_TOO_NEW), resolve_max_isaac_driver())

    monkeypatch.setenv(MAX_ISAAC_DRIVER_ENV, "0")
    assert resolve_max_isaac_driver() is None
    assert meets_max_isaac_driver(_offer(BROKEN_TOO_NEW), None) is True


def test_explicit_argument_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_ISAAC_DRIVER_ENV, "600.0.0")
    assert resolve_max_isaac_driver("581.0.0") == (581, 0, 0)


def test_driver_version_parsing_tolerates_vendor_suffixes() -> None:
    assert driver_version_tuple("580.159.03") == (580, 159, 3)
    assert driver_version_tuple("580.159") == (580, 159)
    assert driver_version_tuple("") is None
    assert driver_version_tuple(None) is None
