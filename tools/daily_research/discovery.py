"""Adaptive discovery instructions and conservative, explicitly estimated costs.

No provider, credentials, sink writes or scheduling. The daily $1 soft target is
separate from the explicitly authorized one-time test ceiling.
"""
from decimal import Decimal

TARGET_NEW = 10
MAX_CANDIDATES = 100  # Input/resource ceiling, not a research quota.
TEAM_DIRECTORY = "https://app.notion.com/p/3eb80154161d817aa3e6d9b9d7eba938"


def instructions(target_usd=1):
    return (
        "Work breadth-first, then deepen. Before external searches, read the broader Team Directory "
        f"through the existing Notion read connection when available: {TEAM_DIRECTORY}. "
        "Read its index, role groups and relevant full records, not just the four-record supplied capability register. "
        "Record actual directory coverage and original evidence dates; directory entries are research leads, "
        "not qualified partners. Read prior run summaries, tested hypotheses, rejections and actual outcomes "
        "through authorized company history. Browse with an empty query when useful, follow available pages, "
        "then fetch relevant full records. Missing or expired access is a gap, never permission to widen scope. "
        "If Notion reads fail, use available reviewed context and public primary sources to broaden discovery; "
        "do not silently treat the small snapshot as the whole directory or stop all research. "
        "Build a capability/task/industry opportunity map from that evidence. Consider manufacturing machine "
        "tending, kitting/assembly, warehouse picking/unloading/material transport, food handling, commercial "
        "cleaning and inspection, and textiles only where supported; these are starting examples, not fixed filters. "
        "Software, policies, components and world models need a concrete robot/workflow integration hypothesis. "
        "Survey several distinct supported task families before spending most of the run on one. Choose "
        "queries and allocation yourself based on evidence and prior outcomes. An early easy hit, prior laundry "
        "success or the first CRM-ready row is not a reason to end exploration. Avoid recent-announcement-only "
        "searches: established operating sites, operator service/process pages and employer-affiliated job "
        "descriptions can establish current recurring work without new funding or an expansion announcement. "
        "First build a substantial source-backed opportunity_shortlist, then deepen its strongest entries. "
        "Aim for roughly 20-40 distinct site/task leads across supported families when sources and the admitted "
        "envelope allow; this is a planning aim, never a required count, padded quota or stopping rule. "
        "Open operator sources to verify initial leads, dedupe identities early, and record remaining questions "
        "and next checks. Retain promising incomplete, rejected and duplicate leads with their honest status "
        "instead of losing them when they cannot yet enter the CRM. A search passage alone is an unverified lead. "
        "Deepen the strongest entries using specific task evidence, robot limitations, incumbent automation, "
        "geography and credible professional routing. Keep confidence in site/task evidence and robot fit "
        "separate from buying interest. High-confidence public task evidence can coexist with unknown buying "
        "interest; never lower task confidence solely because nobody has replied, or invent demand from a task. "
        "Only candidates satisfying the existing source/evidence contract enter candidate QA and CRM admission. "
        "Before stopping, revisit underexplored supported families and unresolved strong leads; marginal returns "
        "must be assessed across the stated breadth, not just one laundry query sequence. Reserve the existing "
        "QA/publication time and record whether breadth was interrupted rather than claiming completion. "
        "Research a defined, evidence-based scope of NEW distinct commercial site/task opportunities. Never pad the list. "
        "Use exact operator-sourced site/location and evidence of real recurring physical work. "
        "A plausible task/team fit is a labeled hypothesis, not a deployment or pilot qualification. "
        "Unknown buyer interest, owner, budget, provider willingness, support and rights remain explicit "
        "unknowns; they do not alone exclude a sourced discovery opportunity. Check incumbent automation "
        "and a plausible disqualifier before recommending a task. Existing robot deployments are separate "
        "learning contacts in findings, never new-opportunity quota. Existing CRM matches are evidence "
        "refreshes or learning notes, never new opportunities. Blueprint agent QA checks exact and semantic "
        "duplicates before publication. Emit a concise candidate row with one useful non-confidential first "
        "question in proposed_next_action; add deeper notes only for the strongest cases. "
        "Explore relevant tasks, industries and regions before deepening promising cases; count distinct sites/tasks, "
        "never findings as prospects. A site may explore robotics before a proven ROI advantage: do not assume "
        "manual work, costly pain, buying intent or existing robotics interest. Discover the actual decision and "
        "uncertainty; interest in conversation never means agreement to pilot. Search for a publicly evidenced "
        "workflow owner, technology evaluator, decision-maker or credible routing contact and explain relevance. "
        "A named employee or executive is not automatically suitable; search for a relevant person before using "
        "a general inbox, and never guess professional contacts or relationships. Include contact source and "
        "relevance, previous contact/history, interest or unknown, counterevidence, plausible team limitations "
        "and one easy non-confidential question in the sourced findings/brief and candidate next action. "
        "Research plausible robotics teams alongside site opportunities, including early-stage and 2025-2026 "
        "announced, funded or publicly evidenced stealth teams. Task-specific hardware, hands, manipulation "
        "components, robotics software and world models may be relevant only with concrete robotics deployment "
        "or pilot evidence; exclude home-focused offerings, data-only firms and generic 3D reconstruction without "
        "robotics focus or a concrete demo. Keep team discoveries in findings and proposed knowledge changes; "
        "never count them as site prospects. Funding and demos do not prove deployment readiness. Preserve "
        "exact product/version, embodiment, sourced specifications with units, measured evidence and unknowns. "
        "Explicitly read /workspace/capabilities/blueprint/deep-research/SKILL.md and "
        "/workspace/capabilities/blueprint/blueprint-evidence-qualification/SKILL.md and its "
        "references/prospect-contract.md. Use the reviewed versions. Decompose the task, vary searches, "
        "open primary operator and product sources, follow gaps and contradictions, and reuse reviewed "
        "knowledge. There is no two-search/two-open or three-candidate quality cap. Stop when material "
        "questions are answered at a stated evidence scope, next sources add little value, access is "
        "blocked, or the admitted time/spend envelope closes. Retain every defensible prospect within the "
        "resource envelope; explain unresolved coverage and interruptions. There is no minimum or maximum "
        "prospect quota. A source failure is a visible gap, "
        "never a pretend read or evidence of market absence. No dot or parent runtime review step: "
        "Blueprint agents own research QA and publication under the existing approvals. No outreach, "
        "drafts, purchases, credentials, subagents, new models/providers, installs, sandbox networking "
        "or direct external writes. Native web_search only. "
        f"The total model/search/hosted-environment research target is ${target_usd}; model token "
        "estimates are not a hard total-dollar cap. Record actual search query/page counts, branches, "
        "rejections, source failures and stopping reason in coverage. "
    )


def validate_coverage(value, candidate_count):
    fields = {"search_queries", "pages_opened", "branches_checked", "rejection_reasons", "stop_reason", "shortfall_reason"}
    scope_fields = {"defined_run_scope", "unresolved_promising_branches", "completion_state"}
    if not isinstance(value, dict) or set(value) not in (fields, fields | scope_fields):
        raise ValueError("discovery_coverage_invalid")
    for name in ("search_queries", "pages_opened"):
        if type(value[name]) is not int or not 0 <= value[name] <= 10000:
            raise ValueError("discovery_coverage_invalid")
    for name in ("branches_checked", "rejection_reasons"):
        if (not isinstance(value[name], list) or len(value[name]) > 100
                or any(not isinstance(x, str) or not 1 <= len(x.strip()) <= 2000 for x in value[name])):
            raise ValueError("discovery_coverage_invalid")
    if not isinstance(value["stop_reason"], str) or not 1 <= len(value["stop_reason"].strip()) <= 2000:
        raise ValueError("discovery_coverage_invalid")
    reason = value["shortfall_reason"]
    if reason is not None and (not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 2000):
        raise ValueError("discovery_coverage_invalid")
    if scope_fields <= set(value):
        for name in ("defined_run_scope", "unresolved_promising_branches"):
            if (not isinstance(value[name], list) or len(value[name]) > 100
                    or any(not isinstance(x, str) or not 1 <= len(x.strip()) <= 2000 for x in value[name])):
                raise ValueError("discovery_coverage_invalid")
        if not value["defined_run_scope"] or value["completion_state"] not in {
                "coverage_complete", "budget_interrupted", "time_interrupted", "access_blocked"}:
            raise ValueError("discovery_scope_or_completion_invalid")
        if value["completion_state"] == "coverage_complete" and value["unresolved_promising_branches"]:
            raise ValueError("discovery_completion_has_unresolved_branches")
    elif candidate_count < TARGET_NEW and reason is None:
        raise ValueError("discovery_shortfall_reason_required")
    return value


def shortlist_issues(value, run_date):
    """Research backlog only, never candidate admission or contact authority."""
    from datetime import date

    from tools.daily_research.runner import Refusal, public_url

    text_fields = {"organization", "site", "location", "task", "capability_family", "robot_fit", "buying_interest"}
    fields = text_fields | {"site_task_confidence", "robot_fit_confidence", "status", "remaining_questions", "sources"}
    source_fields = {"claim", "url", "publisher", "quote", "source_date", "checked_date"}
    def text(v):
        return isinstance(v, str) and 1 <= len(v.strip()) <= 2000
    if not isinstance(value, list) or len(value) > MAX_CANDIDATES:
        yield "", "discovery_shortlist_invalid"
        return
    for index, item in enumerate(value):
        pointer = f"/{index}"
        if not isinstance(item, dict) or set(item) != fields:
            yield pointer, "discovery_shortlist_invalid"
            continue
        for field in text_fields:
            if not text(item[field]):
                yield pointer + "/" + field, "discovery_shortlist_invalid"
        for field in ("site_task_confidence", "robot_fit_confidence"):
            if item[field] not in ("unknown", "low", "medium", "high"):
                yield pointer + "/" + field, "discovery_shortlist_invalid"
        if item["status"] not in ("needs_research", "candidate", "rejected", "duplicate", "existing_deployment"):
            yield pointer + "/status", "discovery_shortlist_invalid"
        questions = item["remaining_questions"]
        if not isinstance(questions, list) or len(questions) > 20 or any(not text(q) for q in questions):
            yield pointer + "/remaining_questions", "discovery_shortlist_invalid"
        sources = item["sources"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= 12:
            yield pointer + "/sources", "discovery_shortlist_invalid"
            continue
        for si, source in enumerate(sources):
            sp = pointer + f"/sources/{si}"
            if not isinstance(source, dict) or set(source) != source_fields:
                yield sp, "discovery_shortlist_invalid"
                continue
            if any(not text(source[f]) for f in ("claim", "url", "publisher", "quote")):
                yield sp, "discovery_shortlist_invalid"
            try:
                public_url(source["url"])
                published = source["source_date"]
                if published is not None and (not isinstance(published, str) or date.fromisoformat(published).isoformat() != published or published > run_date):
                    raise ValueError
                if source["checked_date"] != run_date:
                    raise ValueError
            except (Refusal, TypeError, ValueError):
                yield sp, "discovery_shortlist_invalid"


ESTIMATOR_VERSION = "blueprint.model-token-estimate.v2"
RATE_REFERENCE = {
    "verified_on": "2026-10-02", "model": "gpt-6.1-sol", "service_tier": "default",
    "sources": ["https://developers.openai.com/api/docs/models/gpt-6.1-sol",
                "https://developers.openai.com/api/docs/guides/prompt-caching"],
    "short_usd_per_million": {"uncached_input": "2", "cached_input": "0.10", "cache_write": "2.50", "output": "10"},
    "long_usd_per_million": {"uncached_input": "4", "cached_input": "0.20", "cache_write": "5", "output": "15"},
    "long_context_threshold_input_tokens_per_request": 272000,
}


def estimated_model_cost(usage, *, model="gpt-6.1-sol", service_tier="default",
                         context_regime="unknown", regional_processing=None):
    """Versioned conditional range for recorded tokens, never an invoice/total cap.

    Read/write/uncached input categories are mutually exclusive. Turn aggregates
    cannot establish the >272K PER REQUEST tier. Reasoning is already in output.
    Unknown cache writes/context are ranges; unknown regional/hosting/tool costs
    stay separate. Defaults describe the pinned researcher, not arbitrary models.
    """
    result = {"estimator_version": ESTIMATOR_VERSION, "rate_reference": RATE_REFERENCE,
              "model": model, "service_tier": service_tier, "known": False,
              "estimate_usd": None, "billed_usd": None, "hard_total_cap": False,
              "estimate_kind": "conditional_upper_for_recorded_model_tokens",
              "excludes": ["unreported_or_lagged_usage", "tool_fees", "hosted_environment"],
              "unknown_components": []}
    if model != RATE_REFERENCE["model"] or service_tier != "default":
        return {**result, "error": "model_or_service_tier_rates_unverified"}
    if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ("input_tokens", "output_tokens"))):
        return {**result, "error": "token_usage_unavailable_or_invalid"}
    if context_regime not in {"short", "long", "unknown"} or (regional_processing is not None and type(regional_processing) is not bool):
        return {**result, "error": "pricing_context_invalid"}
    details = usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        return {**result, "error": "input_token_details_invalid"}
    total, output = usage["input_tokens"], usage["output_tokens"]
    cached, writes = details.get("cached_tokens"), details.get("cache_write_tokens")
    if any(value is not None and (type(value) is not int or not 0 <= value <= total)
           for value in (cached, writes)) or cached is not None and writes is not None and cached + writes > total:
        return {**result, "error": "input_token_categories_invalid"}
    unknown = result["unknown_components"]
    if cached is None:
        unknown.append("cache_read_count")
    if writes is None:
        unknown.append("cache_write_classification")
    if context_regime == "unknown":
        unknown.append("per_request_context_regime")
    if regional_processing is None:
        unknown.append("regional_processing_premium")
        result["excludes"].append("unverified_regional_processing_premium")

    def cost(regime, upper):
        rates = {key: Decimal(value) for key, value in RATE_REFERENCE[regime + "_usd_per_million"].items()}
        write_count = writes if writes is not None else (total - (cached or 0) if upper else 0)
        read_count = cached if cached is not None else (0 if upper else total - write_count)
        uncached_count = total - read_count - write_count
        price = (uncached_count * rates["uncached_input"] + read_count * rates["cached_input"]
                 + write_count * rates["cache_write"] + output * rates["output"]) / 1000000
        return price * (Decimal("1.1") if regional_processing is True else 1)

    minimum = cost("short" if context_regime == "unknown" else context_regime, False)
    maximum = cost("long" if context_regime == "unknown" else context_regime, True)
    return {**result, "known": True, "estimate_usd": str(maximum),
            "recorded_tokens_usd": {"minimum": str(minimum), "maximum": str(maximum)},
            "context_regime": context_regime, "regional_processing": regional_processing,
            "usage": usage}


def model_cost_observation(turns, *, model="gpt-6.1-sol", service_tier="default"):
    """Each turn once; retain its exact token details rather than flattening cache.

    Caller supplies only turns from the exact admitted session. Session aggregate
    usage is deliberately never added. Missing/unsupported turns remain unknown.
    """
    observed, pending, seen, lower, upper, unknown = [], [], set(), Decimal(0), Decimal(0), set()
    for turn in turns:
        turn_id = turn.get("id")
        if not isinstance(turn_id, str) or not turn_id or turn_id in seen:
            return {"estimator_version": ESTIMATOR_VERSION, "known": False, "estimate_usd": None,
                    "reported_estimate_usd": None, "billed_usd": None, "hard_total_cap": False,
                    "usage_state": "invalid_turn_inventory"}
        seen.add(turn_id)
        estimate = estimated_model_cost(turn.get("usage"), model=model, service_tier=service_tier)
        observed.append({"turn_id": turn_id, "status": turn.get("status"), "usage": turn.get("usage")})
        if not estimate["known"]:
            pending.append({"turn_id": turn_id, "status": turn.get("status")})
            continue
        lower += Decimal(estimate["recorded_tokens_usd"]["minimum"])
        upper += Decimal(estimate["recorded_tokens_usd"]["maximum"])
        unknown.update(estimate["unknown_components"])
    reported = len(observed) - len(pending)
    complete = bool(observed) and not pending
    return {"estimator_version": ESTIMATOR_VERSION, "rate_reference": RATE_REFERENCE,
            "model": model, "service_tier": service_tier, "known": complete,
            "estimate_usd": str(upper) if complete else None,
            "reported_estimate_usd": str(upper) if reported else None,
            "recorded_tokens_usd": {"minimum": str(lower), "maximum": str(upper)} if reported else None,
            "usage_state": "reported_best_effort" if complete else "pending",
            "reported_turn_count": reported, "pending_turns": pending, "observed_turns": observed,
            "unknown_components": sorted(unknown), "billed_usd": None, "hard_total_cap": False,
            "excludes": ["unreported_or_lagged_usage", "tool_fees", "hosted_environment", "unverified_regional_processing_premium"]}


def preserve_estimate(row, field, estimate):
    """Never replace original proof with a corrected interpretation without trace."""
    previous = row.get(field)
    if previous and previous.get("estimator_version") != ESTIMATOR_VERSION:
        history = row.setdefault(field + "_history", [])
        if previous not in history:
            history.append(previous)
    row[field] = estimate
