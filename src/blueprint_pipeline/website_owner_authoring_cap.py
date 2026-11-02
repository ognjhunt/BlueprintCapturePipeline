"""Apply the owner's deploy-installed ARTISAN cap to a website job's stage 3.

A website job freezes its stage-3 configuration when it is first prepared
(``runtime_inputs.object_authoring.configuration``), and every retry reuses
it. capture-447ae4f4, 2026-11-02: the owner raised the per-object ARTISAN cap
from $15 to $25 (installed by the deploy into the release environment file,
#3240). The retry still froze $15, because nothing re-read the cap after the
first preparation.

The control plane now raises, never lowers, a frozen ARTISAN cap to the value
the deploy installed. It reads that value from the release environment file
itself, so a host-local env file cannot shadow it. The validators that
re-derive stage 3 accept exactly this difference: the same configuration with
a higher cap, with the same cumulative scope. Spend still needs the intent's
budget; this module grants nothing.

The cap is sized for the models it was frozen with. Owner direction 2026-11-02
moved ARTISAN from gpt-6-astra to gpt-6.1-sol and its cap from $25 to $12. A
configuration frozen under earlier ``ARTISAN_MODELS`` cannot run (stage 3
requires the release's models), so it is rebound to the current models and
their cap, which may be lower. The validators accept that difference too.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .task_object_authoring_backend import (ARTISAN_AUTHORING_CAP_USD, ARTISAN_BACKEND, ARTISAN_BUDGET_ENV,
                                            ARTISAN_MODELS, frozen_authoring_cap, positive_cost)

#: Installed by every deploy (scripts/deploy_control_plane_commit.py).
RELEASE_ENVIRONMENT_FILE = Path("/etc/blueprint/task-evaluation-scene-configuration-release.env")
_SCOPE = "whole_authoring_stage_cumulative_including_resume"


def owner_artisan_cap(path: Path | None = None) -> float | None:
    """The cap the deploy installed, or ``None`` when absent or unreadable."""

    try:
        text = Path(path or RELEASE_ENVIRONMENT_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == ARTISAN_BUDGET_ENV:
            try:
                return positive_cost(value.strip().strip('"'))
            except ValueError:
                return None
    return None


def _cumulative_artisan(configuration: Mapping[str, Any]) -> bool:
    budget = configuration.get("authoring_budget")
    return (configuration.get("authoring_backend") == ARTISAN_BACKEND and isinstance(budget, Mapping)
            and budget.get("mode") == "finite_cap" and budget.get("scope") == _SCOPE)


def _with_cap(configuration: Mapping[str, Any], cap: float) -> dict[str, Any]:
    return {**configuration, "authoring_budget": {**configuration["authoring_budget"], "maximum_cost_usd": cap}}


def with_owner_artisan_cap(configuration: dict[str, Any], cap: float | None) -> dict[str, Any]:
    """``configuration`` with its ARTISAN cap raised to ``cap``, or rebound to the release's models.

    Under the release's models a cap is only ever raised. A configuration frozen
    under earlier models takes the current models and their cap: ``cap`` when the
    deploy installed one, else ``ARTISAN_AUTHORING_CAP_USD``. Nothing else changes.
    """

    if not _cumulative_artisan(configuration):
        return configuration
    if configuration.get("authoring_models") != ARTISAN_MODELS:
        rebound = _with_cap(configuration, ARTISAN_AUTHORING_CAP_USD if cap is None else cap)
        return {**rebound, "authoring_models": dict(ARTISAN_MODELS)}
    if cap is None or cap <= frozen_authoring_cap(configuration):
        return configuration
    return _with_cap(configuration, cap)


def same_except_owner_cap(stored: Mapping[str, Any], derived: Mapping[str, Any]) -> bool:
    """Whether ``stored`` is ``derived``, or what ``with_owner_artisan_cap`` makes of it.

    That is ``derived`` with only a raised cumulative ARTISAN cap, or, when
    ``derived`` was frozen under earlier models, ``derived`` rebound to the
    release's ``ARTISAN_MODELS`` with any valid cap.
    """

    if stored == derived:
        return True
    if not (_cumulative_artisan(stored) and _cumulative_artisan(derived)):
        return False

    def without_cap(value: Mapping[str, Any]) -> dict[str, Any]:
        return {**value, "authoring_budget": {k: v for k, v in value["authoring_budget"].items()
                                              if k != "maximum_cost_usd"}, "authoring_models": None}
    try:
        if without_cap(stored) != without_cap(derived):
            return False
        if derived.get("authoring_models") != ARTISAN_MODELS:
            return stored.get("authoring_models") == ARTISAN_MODELS and frozen_authoring_cap(stored) > 0
        return (stored.get("authoring_models") == ARTISAN_MODELS
                and frozen_authoring_cap(stored) > frozen_authoring_cap(derived))
    except ValueError:
        return False


__all__ = ["RELEASE_ENVIRONMENT_FILE", "owner_artisan_cap", "same_except_owner_cap", "with_owner_artisan_cap"]
