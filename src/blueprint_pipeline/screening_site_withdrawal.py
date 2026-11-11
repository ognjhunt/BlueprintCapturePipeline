"""Whether a screening site's consent is withdrawn, read where the intake leaves its tombstone.

G52: a planned screening campaign could reserve budget for a site whose consent
was withdrawn; only the launch refused, later, before any spend. The screening
coordinator now asks here before it reserves, and refuses (a privacy stop) when
the site is withdrawn.

A site is withdrawn when ``<root>/scenes/<site>/website_withdrawal/tombstone.json``
or ``<root>/<bucket>/scenes/<site>/website_withdrawal/tombstone.json`` exists
under a configured site root: the file the intake writes on withdrawal
(``website_capture_withdrawal.acknowledge_withdrawal``) and the one automatic
withdrawal deletion looks for (``website_capture_withdrawal_automatic``). Its
presence is enough: a privacy stop never waits for the tombstone to validate.

The roots come only from ``BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS``, which the
screening unit sets, so nothing here depends on what exists on the machine.
Without it the answer is ``not_configured``. A root or directory that cannot
be listed is ``unknown``, never "not withdrawn": the coordinator then reserves
nothing and asks again on its next pass.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SITE_ROOTS_ENV = "BLUEPRINT_WEBSITE_WITHDRAWAL_SITE_ROOTS"
WITHDRAWN_BLOCKER = "beta_screening_site_consent_withdrawn"
_SITE = re.compile(r"site-[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_TOMBSTONE = Path("website_withdrawal") / "tombstone.json"


def configured_site_roots(environ: Mapping[str, str] | None = None) -> list[Path]:
    env = os.environ if environ is None else environ
    return [Path(item) for item in str(env.get(SITE_ROOTS_ENV) or "").split(":") if item.strip()]


def _tombstone(site_dir: Path) -> bool:
    path = site_dir / _TOMBSTONE
    return path.is_file() and not path.is_symlink()


def site_withdrawal_state(site_id: str, *, environ: Mapping[str, str] | None = None) -> str:
    """``withdrawn``, ``not_withdrawn``, ``unknown`` (a root could not be read) or ``not_configured``.

    A site id that is not a website site (``site-<request id>``) has no
    withdrawal to look for and reads ``not_withdrawn``.
    """

    roots = configured_site_roots(environ)
    if not roots:
        return "not_configured"
    if not _SITE.fullmatch(str(site_id)):
        return "not_withdrawn"
    unknown = False
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            unknown = True
            continue
        if _tombstone(root / "scenes" / site_id):
            return "withdrawn"
        try:
            with os.scandir(root) as entries:
                buckets = [Path(entry.path) for entry in entries
                           if entry.name != "scenes" and entry.is_dir(follow_symlinks=False)]
        except OSError:
            unknown = True
            continue
        if any(_tombstone(bucket / "scenes" / site_id) for bucket in buckets):
            return "withdrawn"
    return "unknown" if unknown else "not_withdrawn"


def release_withdrawn_site(ledger: Mapping[str, Any], site_id: str, *, closed_at: str) -> list[dict[str, Any]]:
    """Close every queued row of the site and release every hold not yet past its launch intent.

    Changes the screening ledger in place and returns the campaigns it changed.
    A row ``launching`` or later may already have spent: it keeps its
    reservation and reconciles as before.
    """

    changed = []
    for state in ledger["campaigns"].values():
        if state.get("site_id") != site_id:
            continue
        touched = False
        for row in state["candidates"].values():
            attempt = row.get("attempt")
            if row.get("status") in {"pending", "waitlisted"}:
                row.update(status="blocked", blockers=[WITHDRAWN_BLOCKER])
                touched = True
            elif (row.get("status") == "preparing" and isinstance(attempt, dict)
                  and attempt.get("reservation_status") == "reserved" and not attempt.get("launch_intent_at")):
                attempt.update(reservation_status="released", closed_at=closed_at)
                row.update(status="blocked", blockers=[WITHDRAWN_BLOCKER])
                touched = True
        if touched:
            changed.append(state)
    return changed


__all__ = ["SITE_ROOTS_ENV", "WITHDRAWN_BLOCKER", "configured_site_roots", "release_withdrawn_site",
           "site_withdrawal_state"]
