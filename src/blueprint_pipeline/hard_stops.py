"""The three walls a job may not pass, as types: privacy, rights and money (AGENTS.md, "The Only Hard Stops").

Until 2026-11-17 a controller could tell a wall from any other refusal only by the words of its code
(``job_retry_policy.hard_stop_kind``) and by short lists, and a sample of raise sites found both wrong in
both directions. The step that guards a wall now raises it as a type:

- ``PrivacyRightsStop``: the site's consent and its withdrawal, the owner closing the run, the provider
  terms and the provider retention the site's frames are sent under, what an admission allows to be
  disclosed, and the face blur on frames sent to World Labs.
- ``MoneyStop``: an allowance, a ceiling, an admission that refuses or no longer covers a purchase, and
  the reviewed prices a purchase is bounded by.

Both are ``ValueError`` whose text is the code it always was, so every existing ``except``, every stored
code and every digest is unchanged. Two existing types are walls as they stand: the paid-resource
chokepoint's ``PaidResourceAdmissionBlocked`` (a ``RuntimeError``, whatever its blockers) and the Atlas
worker's withdrawal (``atlas_preparation_job_control._Withdrawn``).

This module only names the walls. What a job does after one is unchanged by it; ``is_wall`` is what a
later change uses to decide. ``hard_stop_kind`` stays the keyword backstop for a code nobody typed.

Standard library only; no I/O.
"""

from __future__ import annotations

from typing import Any

from .job_retry_policy import MONEY, OWNER_CLOSED_CODE, PRIVACY_RIGHTS

__all__ = ["WEBAPP_MONEY", "WEBAPP_PRIVACY_RIGHTS", "HardStop", "MoneyStop", "PrivacyRightsStop", "is_wall",
           "wall_kind", "webapp_refusal"]


class HardStop(ValueError):
    """A refusal that protects privacy, rights or money. ``str(error)`` is its code, exactly as before."""

    kind: str = ""


class PrivacyRightsStop(HardStop):
    kind = PRIVACY_RIGHTS


class MoneyStop(HardStop):
    kind = MONEY


#: The website's typed answers (``website_control_<operation>_http_<status>:<code>``) that are walls: a
#: withdrawal or revocation, the owner closing the run, the capture's consent, provider terms and grant.
WEBAPP_PRIVACY_RIGHTS = frozenset({
    "source_revoked", "website_capture_withdrawn", "consent_revoked", "capture_revoked_or_expired",
    OWNER_CLOSED_CODE, "website_atlas_source_consent_required", "website_atlas_consent_expired",
    "website_atlas_planner_consent_required", "provider_terms_not_configured_or_changed",
    "website_atlas_authorization_invalid", "website_atlas_source_owner_mismatch", "consent_expired"})
#: The job's own allowance, ceiling or safety limit (``websiteSceneSponsorship.ts``).
WEBAPP_MONEY = frozenset({"website_scene_preparation_budget_exhausted"})


def webapp_refusal(message: str, code: Any) -> ValueError:
    """The error for one of the website's typed refusals: a wall's type when ``code`` is a wall, else ValueError.

    ``message`` is what was raised before, unchanged.
    """
    if code in WEBAPP_PRIVACY_RIGHTS:
        return PrivacyRightsStop(message)
    if code in WEBAPP_MONEY:
        return MoneyStop(message)
    return ValueError(message)


def wall_kind(error: BaseException) -> str | None:
    """``PRIVACY_RIGHTS`` or ``MONEY`` when ``error`` is a wall by its type, else None. Never reads a code."""
    if isinstance(error, HardStop):
        return error.kind
    from .atlas_preparation_job_control import _Withdrawn
    from .paid_resource_admission import PaidResourceAdmissionBlocked
    if isinstance(error, _Withdrawn):
        return PRIVACY_RIGHTS
    if isinstance(error, PaidResourceAdmissionBlocked):
        return MONEY
    return None


def is_wall(error: BaseException) -> bool:
    """Whether ``error`` is a privacy, rights or money wall by its type."""
    return wall_kind(error) is not None
