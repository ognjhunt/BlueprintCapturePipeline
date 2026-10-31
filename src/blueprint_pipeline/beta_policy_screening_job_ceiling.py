"""Size each new screening plan to its website job's one dollar ceiling.

Owner decision 2026-10-30 (WebApp #1168): one dollar ceiling per website job,
$40 by default, across Atlas, upstream, native construction and screening. The
WebApp refuses preparation spend and scene intakes past it, but this queue
reserves screening without asking the WebApp, so the plan is where screening
stops: each new plan's budget is the smaller of the $15 site allowance and what
the ceiling leaves, read once over the existing signed transport. Plans already
built are never resized.

If the ceiling cannot be read (a WebApp without the summary, a job without its
retained Atlas configuration, a refusal, the network), planning holds with
``website_job_ceiling_unavailable`` and the source is retried on the next pass.
An earlier read is never reused: the job's other ledgers may have spent since,
so a stale figure could plan more than the job has left.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

UNAVAILABLE = "website_job_ceiling_unavailable"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def read_job_ceiling_summary(*, context: Mapping[str, Any], capture_root: str | Path) -> dict[str, Any]:
    """The WebApp's ``job_ceiling`` from its non-creating Atlas configuration read.

    That read is bound to the job's Atlas source video, whose digest the
    preview retained with its own configuration. Every failure holds alike.
    """
    from .beta_policy_screening_plan import JOB_CEILING_SUMMARY_SCHEMA
    from .website_task_context import load_atlas_configuration

    try:
        retained = json.loads((Path(capture_root) / "pipeline" / "atlas" / "configuration.json").read_text())
        source = retained["source_video_digest"]
        if not isinstance(source, str) or not _DIGEST.fullmatch(source):
            raise ValueError(UNAVAILABLE)
        summary = load_atlas_configuration(context, source, create=False).get("job_ceiling")
    except Exception:  # noqa: BLE001 - a missing file, an old WebApp, a refusal or the network hold planning alike
        raise ValueError(UNAVAILABLE) from None
    if not isinstance(summary, Mapping) or summary.get("schema_version") != JOB_CEILING_SUMMARY_SCHEMA:
        raise ValueError(UNAVAILABLE)
    return dict(summary)


def local_screening_holds(queue_root: str | Path, *, site_id: str, published_usd: Decimal) -> Decimal:
    """Screening this queue holds for the site that the WebApp's figure misses.

    The WebApp counts screening from the last publication it accepted, so it
    cannot see reservations made since, nor any campaign's selected candidates
    not reserved yet (each campaign up to its own plan budget).
    """
    from .beta_policy_screening_coordinator import _amount, _budget, _ledger, _lock, _plan, _root
    from .beta_policy_screening_plan import money

    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        committed = _amount(_budget(ledger, site_id)["committed_upper_bound_usd"])
        holds = max(Decimal(0), committed - published_usd)
        for campaign_id, state in ledger["campaigns"].items():
            pending = {key for key, row in state["candidates"].items() if row["status"] == "pending"}
            if state["site_id"] != site_id or not pending:
                continue
            plan = _plan(root, campaign_id, state["plan_digest"])
            unreserved = sum((money(row["max_cost_usd"]) for row in plan["candidates"]
                              if row["candidate_id"] in pending), Decimal(0))
            reserved = _amount(_budget(ledger, site_id, campaign_id=campaign_id)["committed_upper_bound_usd"])
            holds += min(unreserved, max(Decimal(0), money(plan["budget_usd"]) - reserved))
    return holds


def plan_within_job_ceiling(catalog: Mapping[str, Any], scene: Mapping[str, Any], *, queue_root: str | Path,
                            context: Mapping[str, Any], capture_root: str | Path, **options: Any) -> dict[str, Any]:
    """Read the job's ceiling once, then build the new plan within it."""
    from .beta_policy_screening_plan import BetaPolicyScreeningPlanError, build_job_ceiling_screening_plan
    from .beta_policy_screening_plan import job_ceiling_record

    if scene.get("site_id") != context.get("scene_id"):
        raise ValueError(UNAVAILABLE)
    summary = read_job_ceiling_summary(context=context, capture_root=capture_root)
    read_at = dt.datetime.now(dt.timezone.utc).isoformat()
    # Without the WebApp's screening figure, count every local reservation again: less, never more.
    published = summary.get("screening_cost_usd")
    published = (Decimal(str(published)) if isinstance(published, (int, float)) and not isinstance(published, bool)
                 and math.isfinite(published) and published >= 0 else Decimal(0))
    holds = local_screening_holds(queue_root, site_id=scene["site_id"], published_usd=published)
    try:
        record = job_ceiling_record(summary, local_holds_usd=holds, read_at=read_at)
    except BetaPolicyScreeningPlanError:
        raise ValueError(UNAVAILABLE) from None
    return build_job_ceiling_screening_plan(catalog, scene, job_ceiling=record, **options)


__all__ = ["UNAVAILABLE", "local_screening_holds", "plan_within_job_ceiling", "read_job_ceiling_summary"]
