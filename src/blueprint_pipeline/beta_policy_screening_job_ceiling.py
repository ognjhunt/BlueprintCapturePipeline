"""Size each new screening plan to its website job's one dollar ceiling.

Owner decision 2026-10-30 (WebApp #1168): one dollar ceiling per website job,
across Atlas, upstream, native construction and screening; the WebApp sets its
figure (the job's Atlas grant, $25 on every job granted before 2026-11-11; the
owner moves new jobs to $100). The WebApp refuses preparation spend and scene
intakes past it, but this queue reserves screening without asking the WebApp,
so the plan is where screening stops. Since 2026-11-11 (owner: "yes replace
them - $300 monthly limit, $100 fuse") each new plan's budget is everything the
ceiling leaves, read once over the existing signed transport: no $15 plan size
and no $30 site allowance below it. The ceiling is a fuse against a fault
burning the month in one job, not a ration. Plans already built are never
resized, and every paid launch reads the ceiling again
(``require_current_job_ceiling``).

If the ceiling cannot be read (a WebApp without the summary, a job without its
retained Atlas configuration, a refusal, the network), planning holds with
``website_job_ceiling_unavailable`` and the source is retried on the next pass.
An earlier read is never reused: the job's other ledgers may have spent since,
so a stale figure could plan more than the job has left.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

UNAVAILABLE = "website_job_ceiling_unavailable"
#: The website's picks for the job, read beside its ceiling (``beta_policy_screening_known_good_control``).
PREFERENCE_FIELD = "preferred_robot_model_ids"
LOGGER = logging.getLogger(__name__)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def read_job_ceiling_summary(*, context: Mapping[str, Any], capture_root: str | Path,
                             extras: dict[str, Any] | None = None) -> dict[str, Any]:
    """The WebApp's ``job_ceiling`` from its non-creating Atlas configuration read.

    That read is bound to the job's Atlas source video, whose digest the
    preview retained with its own configuration. Every failure holds alike.
    ``extras``, when given, receives the same read's ``preferred_robot_model_ids``
    (the website's picks for the job, when it sends them). It is kept out of the
    summary, so no ceiling, plan or task digest changes with it.
    """
    from .beta_policy_screening_plan import JOB_CEILING_SUMMARY_SCHEMA
    from .website_task_context import load_atlas_configuration

    try:
        retained = json.loads((Path(capture_root) / "pipeline" / "atlas" / "configuration.json").read_text())
        source = retained["source_video_digest"]
        if not isinstance(source, str) or not _DIGEST.fullmatch(source):
            raise ValueError(UNAVAILABLE)
        configuration = load_atlas_configuration(context, source, create=False)
        summary = configuration.get("job_ceiling")
        if extras is not None and PREFERENCE_FIELD in configuration:
            extras[PREFERENCE_FIELD] = configuration[PREFERENCE_FIELD]
    except Exception:  # noqa: BLE001 - a missing file, an old WebApp, a refusal or the network hold planning alike
        raise ValueError(UNAVAILABLE) from None
    if not isinstance(summary, Mapping) or summary.get("schema_version") != JOB_CEILING_SUMMARY_SCHEMA:
        raise ValueError(UNAVAILABLE)
    return dict(summary)


def local_screening_holds(queue_root: str | Path, *, site_id: str, published_usd: Decimal,
                          superseding: str | None = None) -> Decimal:
    """Screening this queue holds for the site that the WebApp's figure misses.

    The WebApp counts screening from the last publication it accepted, so it
    cannot see reservations made since, nor what any campaign can still reserve:
    its selected candidates and the waitlisted ones released money can fund,
    each campaign up to its own plan budget. A campaign
    being re-planned (``superseding``) is skipped for those unreserved rows:
    the re-plan closes them in the same write that enqueues the new plan, so
    counting them would charge the job twice. What it committed still counts.

    Commitments count as funding does (``_budget``): a settled attempt at its
    settled cost. The WebApp's figure may still carry it at its whole cap (a
    campaign never publishes a lower spent bound); that only makes the
    difference here zero, never negative, so a hold is never credited back.
    """
    from .beta_policy_screening_coordinator import _amount, _budget, _ledger, _lock, _plan, _root
    from .beta_policy_screening_plan import money

    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        committed = _amount(_budget(ledger, site_id)["committed_upper_bound_usd"])
        holds = max(Decimal(0), committed - published_usd)
        for campaign_id, state in ledger["campaigns"].items():
            # Waitlisted rows count too: money a selected row releases before
            # spend can fund them, up to what the plan's budget still allows.
            # An unreserved known-good control is not held here: it runs only on money left over.
            unfunded = {key for key, row in state["candidates"].items()
                        if row["status"] in {"pending", "waitlisted"} and not row.get("known_good_control")}
            if state["site_id"] != site_id or not unfunded or campaign_id == superseding:
                continue
            plan = _plan(root, campaign_id, state["plan_digest"])
            unreserved = sum((money(row["max_cost_usd"]) for row in plan["candidates"]
                              if row["candidate_id"] in unfunded), Decimal(0))
            reserved = _amount(_budget(ledger, site_id, campaign_id=campaign_id)["committed_upper_bound_usd"])
            holds += min(unreserved, max(Decimal(0), money(plan["budget_usd"]) - reserved))
    return holds


def plan_within_job_ceiling(catalog: Mapping[str, Any], scene: Mapping[str, Any], *, queue_root: str | Path,
                            context: Mapping[str, Any], capture_root: str | Path, superseding: str | None = None,
                            **options: Any) -> dict[str, Any]:
    """Read the job's ceiling once, then build the new plan within it.

    ``superseding`` names the campaign a re-plan after a deploy replaces.
    """
    from .beta_policy_screening_plan import BetaPolicyScreeningPlanError, build_job_ceiling_screening_plan
    from .beta_policy_screening_plan import job_ceiling_record

    if scene.get("site_id") != context.get("scene_id"):
        raise ValueError(UNAVAILABLE)
    extras: dict[str, Any] = {}
    summary = read_job_ceiling_summary(context=context, capture_root=capture_root, extras=extras)
    read_at = dt.datetime.now(dt.timezone.utc).isoformat()
    preferred = _preferred_robot_model_ids(extras.get(PREFERENCE_FIELD), options.get("capabilities"))
    if preferred:
        # Orders only the known-good control's body; outside every digest (``robot_fit_rows`` too).
        options["capabilities"] = {**(options.get("capabilities") or {}), PREFERENCE_FIELD: preferred}
    # Without the WebApp's screening figure, count every local reservation again: less, never more.
    published = summary.get("screening_cost_usd")
    published = (Decimal(str(published)) if isinstance(published, (int, float)) and not isinstance(published, bool)
                 and math.isfinite(published) and published >= 0 else Decimal(0))
    holds = local_screening_holds(queue_root, site_id=scene["site_id"], published_usd=published,
                                  superseding=superseding)
    try:
        record = job_ceiling_record(summary, local_holds_usd=holds, read_at=read_at)
    except BetaPolicyScreeningPlanError:
        raise ValueError(UNAVAILABLE) from None
    from .website_job_limits import fuse
    if not fuse():
        # nested (shipped): the site's remaining allowance fits a last quote, as before 2026-11-11.
        options.setdefault("site_remaining_usd", site_screening_remaining(queue_root, site_id=scene["site_id"]))
    # fuse: no site allowance is passed; the ceiling is the only bound (owner decision 2026-11-11).
    return build_job_ceiling_screening_plan(catalog, scene, job_ceiling=record, **options)


def _preferred_robot_model_ids(value: Any, capabilities: Any) -> list[str]:
    """The website's picks when usable; an unusable value is logged by its typed code and ignored.

    Known ids are the job's own robot fit table's models (one row per spec-catalog model), the
    only models the preference can order. Without a table there is nothing to order.
    """
    from .beta_policy_screening_known_good_control import FIT_CAPABILITY, preferred_model_ids

    rows = (capabilities or {}).get(FIT_CAPABILITY) if isinstance(capabilities, Mapping) else None
    if value is None or not isinstance(rows, list):
        return []
    known = [row["model_id"] for row in rows if isinstance(row, Mapping) and isinstance(row.get("model_id"), str)]
    preferred, code = preferred_model_ids(value, known)
    if code:
        LOGGER.warning("%s", code)
    return preferred


EXCEEDED = "beta_screening_job_ceiling_exceeded"


def require_current_job_ceiling(plan: Mapping[str, Any], *, context: Mapping[str, Any],
                                capture_root: str | Path | None = None,
                                queue_root: str | Path | None = None) -> dict[str, Any] | None:
    """Before a paid launch: the job's ceiling in force now still covers what this queue committed.

    A plan is sized once (``plan_within_job_ceiling``), but its money authority can end
    later: an owner row expires, a grant is withdrawn, other ledgers spend. Reading the
    ceiling again here makes that end a stop before spawn, for every candidate of every
    plan, instead of the plan's original figure staying in force (capture-447, 11-06:
    a run launched seven minutes after the owner's $130 row expired). Unreadable is a
    stop too (``website_job_ceiling_unavailable``). A plan made without a website job
    ceiling has none to re-read and returns ``None``.
    """
    from .beta_policy_screening_coordinator import _amount, _budget, _ledger, _lock, _root

    if not plan.get("website_job_ceiling"):
        return None
    if capture_root is None:
        from .team_scene_packet_worker import resolve_capture_root
        capture_root = resolve_capture_root(plan["source"])
    summary = read_job_ceiling_summary(context=context, capture_root=capture_root)
    figures = [summary.get(key) for key in ("screening_cost_usd", "remaining_cost_usd", "maximum_cost_usd",
                                            "reserved_or_settled_cost_usd")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in figures):
        raise ValueError(UNAVAILABLE)
    published, remaining, maximum, accounted = (Decimal(str(value)) for value in figures)
    root = _root(queue_root)
    with _lock(root):
        committed = _amount(_budget(_ledger(root), plan["site_id"])["committed_upper_bound_usd"])
    # What the WebApp's figure cannot see yet: commitments since its last accepted publication.
    unseen = max(Decimal(0), committed - published)
    # The remaining figure floors at zero, so a job already past a lowered ceiling (an owner row
    # expired after this reservation was published) reads as zero left and zero unseen. Compare
    # the whole: everything the WebApp counts plus what it cannot see, against the ceiling now.
    if remaining < unseen or accounted + unseen > maximum:
        raise ValueError(EXCEEDED)
    return summary


def site_screening_remaining(queue_root: str | Path, *, site_id: str) -> str:
    """What the site's screening limit has left after every commitment, as funding counts it.

    The limit is the site's job ceiling once one of its campaigns recorded it
    (``beta_policy_screening_site_limit.site_limit``). A read for people; planning no
    longer passes it to the plan (the ceiling read itself bounds a new plan).
    """
    from .beta_policy_screening_coordinator import SITE_LIMIT, _amount, _budget, _ledger, _lock, _root
    from .beta_policy_screening_site_limit import site_limit

    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
        committed = _amount(_budget(ledger, site_id)["committed_upper_bound_usd"])
        limit = site_limit(ledger, site_id, SITE_LIMIT)
    return f"{max(Decimal(0), limit - committed):.2f}"


__all__ = ["EXCEEDED", "UNAVAILABLE", "local_screening_holds", "plan_within_job_ceiling", "read_job_ceiling_summary",
           "require_current_job_ceiling", "site_screening_remaining"]
