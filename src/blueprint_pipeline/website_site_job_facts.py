"""The site's quick answers on its job page, kept beside the capture for the job sheet (``job_packet_export``).

A site answers a few quick questions on its job page: how heavy the item is, how big, how many an hour, how many
hours a day the job runs and when it wants a robot. They are the site's own statements, never checked
(``basis: "owner_stated_unverified"``), and deployment companies read them in the job sheet with that label.

They reach the Pipeline only on a task-context read that asks for them (``include_site_job_facts``), never on the
reads that preparation retains, compares or binds: a site taps an answer at any time, and inside a bound context
every tap would read as a changed task and buy preparation again. This module makes that read and writes what it
returns to ``pipeline/website_site_job_facts.json``:

    {"schema_version": "website_site_job_facts_record.v1", "read_at_epoch": <int>,
     "site_job_facts": null | {"basis": "owner_stated_unverified", "updated_at": <str | null>,
                               "answers": {<key>: {"code", "answer", "label", "question"}}}}

with the keys ``item_weight``, ``item_size_longest_side``, ``items_per_hour``, ``hours_per_day`` and ``wanted_by``,
present only for the questions the site answered (``null``: it answered none). It runs after a live rights read
that found the job shared (``company_share_permission.record_live``), so at most once a window per shared capture.
A failed read (the website down, or one that does not know the question yet) keeps the last record and changes
nothing else: the answers are never a reason to stop anything. The file sits in the capture's own folder, so the
site's withdrawal deletes it with the capture.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .website_task_context_contracts import site_job_facts_valid

logger = logging.getLogger(__name__)

RECORD_SCHEMA = "website_site_job_facts_record.v1"
#: Relative to the capture root.
RECORD_PATH = ("pipeline", "website_site_job_facts.json")


def read_live(identity: Mapping[str, str]) -> Mapping[str, Any] | None:
    """The site's quick answers as the website holds them now; None when it answered none."""
    from .website_task_context import load_current_website_task_context
    context = load_current_website_task_context(request_id=identity["request_id"], scene_id=identity["scene_id"],
                                                capture_id=identity["capture_id"], purpose="scene_preparation",
                                                include_site_job_facts=True)
    facts = context.get("site_job_facts")
    return dict(facts) if isinstance(facts, Mapping) else None


def record_site_job_facts(capture_root: Path, identity: Mapping[str, str], now: float,
                          read: Callable[[Mapping[str, str]], Mapping[str, Any] | None] | None = None) -> dict[str, Any] | None:
    """Read the answers once (``read``, else the website's ``read_live``) and keep them; on any failure keep the last
    record and return None. Never raises."""
    try:
        facts = (read or read_live)(identity)
        if facts is not None and not site_job_facts_valid(facts):
            raise ValueError("website_site_job_facts_invalid")
        value = {"schema_version": RECORD_SCHEMA, "read_at_epoch": int(now), "site_job_facts": facts}
        write_json(Path(capture_root).joinpath(*RECORD_PATH), value)
        return value
    except Exception as error:  # noqa: BLE001 - the answers never stop anything; the next window reads again
        logger.warning("website_site_job_facts_unread capture=%s type=%s", identity.get("capture_id"), type(error).__name__)
        return None
