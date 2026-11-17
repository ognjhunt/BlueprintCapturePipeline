"""The three walls a job may not pass, as types: privacy, rights and money (AGENTS.md, "The Only Hard Stops").

Every other failure of a job may be flagged and go on; these may not. Until 2026-11-17 a controller told a
wall from anything else by the words of its code (``job_retry_policy.hard_stop_kind``) and by short lists,
which named some walls wrongly and missed others. Now the step that guards a wall raises it as one of these
types, so a controller can recognise it without reading words:

- ``PrivacyRightsStop``: the site's consent, its withdrawal, the owner closing the run, the provider terms
  and retention the site's frames are sent under, and the face screen before any frame leaves.
  ``final`` is true for a withdrawal and an owner closure: those end the job. Any other privacy or rights
  wall is asked again on a timer until the job's window ends.
- ``MoneyStop``: a ceiling or allowance that refuses new spend. It waits, and its admission is asked again
  each time.

Both are ``ValueError`` with the refusal's own code as their text, so every existing ``except ValueError``,
every stored code and every digest is unchanged. A catch-all that turns a refusal into a held state must
re-raise these (``reraise``), never hold them.

What a wall is stays plain code. No model decision raises, lowers or skips one.

Standard library only; no I/O.
"""

from __future__ import annotations

from typing import Any

from .job_retry_policy import MONEY, OWNER_CLOSED_CODE, PRIVACY_RIGHTS

__all__ = [
    "ACCOUNT_CEILING_BLOCKERS",
    "WEBAPP_FINAL",
    "WEBAPP_MONEY",
    "WEBAPP_PRIVACY_RIGHTS",
    "HardStop",
    "MoneyStop",
    "PrivacyRightsStop",
    "reraise",
    "wall",
    "webapp_refusal",
]


class HardStop(ValueError):
    """A refusal that protects privacy, rights or money. ``str(error)`` is its code, exactly as before."""

    kind: str = ""

    def __init__(self, code: str, *, final: bool = False) -> None:
        super().__init__(code)
        self.final = bool(final)


class PrivacyRightsStop(HardStop):
    kind = PRIVACY_RIGHTS


class MoneyStop(HardStop):
    kind = MONEY


#: The website's typed answers (``website_control_<operation>_http_<status>:<code>``) that are walls.
#: A withdrawal and an owner closure end the job.
WEBAPP_FINAL = frozenset({"source_revoked", "website_capture_withdrawn", "consent_revoked",
                          "capture_revoked_or_expired", OWNER_CLOSED_CODE})
#: Consent for this capture's frames to go to the provider, the provider terms it was given under, the
#: grant itself, and the grant's window: asked again until the job's own window ends.
WEBAPP_PRIVACY_RIGHTS = frozenset({"website_atlas_source_consent_required", "website_atlas_consent_expired",
                                   "website_atlas_planner_consent_required", "provider_terms_not_configured_or_changed",
                                   "website_atlas_authorization_invalid", "website_atlas_source_owner_mismatch",
                                   "consent_expired"})
#: The job's own allowance, ceiling or safety limit (``websiteSceneSponsorship.ts``).
WEBAPP_MONEY = frozenset({"website_scene_preparation_budget_exhausted"})

#: The paid-resource chokepoint's ceilings (``paid_resource_admission``): the account's monthly limit and the
#: Google Cloud monthly budget, including the refusal when either cannot be read (they fail closed).
ACCOUNT_CEILING_BLOCKERS = ("account_monthly_spend_limit_", "gcp_monthly_budget_")


def webapp_refusal(message: str, code: Any) -> ValueError:
    """The error for one of the website's typed refusals: a wall's type when ``code`` is a wall, else ValueError.

    ``message`` is what was raised before, unchanged.
    """
    if code in WEBAPP_FINAL:
        return PrivacyRightsStop(message, final=True)
    if code in WEBAPP_PRIVACY_RIGHTS:
        return PrivacyRightsStop(message)
    if code in WEBAPP_MONEY:
        return MoneyStop(message)
    return ValueError(message)


def wall(error: BaseException) -> tuple[str, bool] | None:
    """``(kind, final)`` when ``error`` is a wall, else None.

    A typed wall, or the paid-resource chokepoint's refusal when every blocker it names is a ceiling
    (``ACCOUNT_CEILING_BLOCKERS``). That refusal is a ``RuntimeError`` many callers already record by name,
    so its type is left as it is and it is recognised here.
    """
    if isinstance(error, HardStop):
        return error.kind, error.final
    from .paid_resource_admission import PaidResourceAdmissionBlocked
    if isinstance(error, PaidResourceAdmissionBlocked):
        blockers = [str(row) for row in error.blockers or []]
        if blockers and all(row.startswith(ACCOUNT_CEILING_BLOCKERS) for row in blockers):
            return MONEY, False
    return None


def reraise(error: BaseException) -> None:
    """Raise ``error`` again when it is a wall: a catch-all that holds a refusal must never hold a wall."""
    if wall(error) is not None:
        raise error
