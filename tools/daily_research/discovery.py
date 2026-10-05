"""Adaptive discovery instructions and conservative, explicitly estimated costs.

No provider, credentials, sink writes or scheduling. The daily $1 soft target is
separate from the explicitly authorized one-time test ceiling.
"""
import hashlib
import json
from decimal import Decimal

TARGET_NEW = 10
MAX_CANDIDATES = 100  # Legacy batch size; v3 retention is bounded by bytes.
INVENTORY_PAGE_BYTES = 100_000
INVENTORY_VERSION = "blueprint.discovery-inventory.v1"
# Broader Notion Team Directory (#2561). Its entries are research leads, never qualified partners.
TEAM_DIRECTORY = "https://app.notion.com/p/3eb80154161d817aa3e6d9b9d7eba938"


def inventory_issues(entries):
    """Thin discoveries have their own contract and no promotion authority."""
    fields = {"operator", "site", "location", "task_hypothesis", "source_urls", "evidence_gap", "disposition"}
    if not isinstance(entries, list):
        yield {"pointer": "/discovery_inventory", "code": "discovery_inventory_invalid"}
        return
    from tools.daily_research.runner import Refusal, public_url
    for index, entry in enumerate(entries):
        path = f"/discovery_inventory/{index}"
        if not isinstance(entry, dict) or set(entry) != fields:
            yield {"pointer": path, "code": "discovery_inventory_invalid"}
            continue
        for field in ("operator", "site", "location", "task_hypothesis"):
            value = entry[field]
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 2000):
                yield {"pointer": path + "/" + field, "code": "discovery_inventory_invalid"}
        if not isinstance(entry["disposition"], str) or entry["disposition"] not in {"candidate", "unresolved", "rejected", "learning", "duplicate"} or not isinstance(entry["evidence_gap"], str) or not entry["evidence_gap"].strip() or len(entry["evidence_gap"]) > 2000:
            yield {"pointer": path, "code": "discovery_inventory_invalid"}
        from tools.daily_research.runner import canonical
        if len(canonical(entry).encode()) > INVENTORY_PAGE_BYTES - 1000:
            yield {"pointer": path, "code": "discovery_inventory_record_resource_ceiling_raw_retained"}
        urls = entry["source_urls"]
        if not isinstance(urls, list) or not urls or len(urls) > 12:
            yield {"pointer": path + "/source_urls", "code": "discovery_inventory_invalid"}
        else:
            for number, url in enumerate(urls):
                try:
                    public_url(url)
                except (Refusal, ValueError, TypeError):
                    yield {"pointer": path + f"/source_urls/{number}", "code": "discovery_inventory_invalid"}


def retain_inventory(entries, row, ledger, output_digest):
    """Store all original records in immutable bounded pages; never trim a list.

    The cursor is a page number, so read-only callers can resume without a new
    search or changing eligibility. The full original output remains archived.
    """
    from tools.daily_research.recovery import parse_artifact_json
    from tools.daily_research.runner import Refusal, canonical
    from tools.daily_research.verification import digest as evidence_digest
    records_digest = evidence_digest(entries)
    sources = [(f"{row['date']}-artifact.json", row.get("raw_output_digest"))] + [
        (revision.get("artifact_file"), revision.get("artifact_digest")) for revision in row.get("validation_repairs", [])]
    source_binding = None
    for filename, expected in reversed(sources):
        if not filename or not expected:
            continue
        raw = ledger.read_bytes(filename)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise Refusal("discovery_inventory_source_binding_invalid")
        try:
            original, _ = parse_artifact_json(raw)
            if isinstance(original, dict) and evidence_digest(original.get("discovery_inventory")) == records_digest:
                source_binding = {"source_artifact_file": filename, "source_artifact_sha256": expected}
                break
        except (ValueError, UnicodeError, TypeError):
            continue
    if source_binding is None:
        raise Refusal("discovery_inventory_source_binding_invalid")
    pages, current, start = [], [], 0
    def save(records):
        nonlocal start
        value = {"version": INVENTORY_VERSION, "run_key": row["run_key"], "source_output_digest": output_digest,
                 "start": start, "end": start + len(records), "records": records}
        raw = (canonical(value) + "\n").encode()
        if len(raw) > INVENTORY_PAGE_BYTES:
            raise Refusal("discovery_inventory_record_resource_ceiling_raw_retained")
        filename = f"{row['date']}-inventory-{output_digest}-{len(pages)}.json"
        ledger.write_json(filename, value)
        pages.append({"index": len(pages), "file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "start": start, "end": value["end"]})
        start = value["end"]
    for entry in entries:
        if current and len(canonical(current + [entry]).encode()) > INVENTORY_PAGE_BYTES - 1000:
            save(current)
            current = []
        current.append(entry)
    if current:
        save(current)
    return {"version": INVENTORY_VERSION, "source_output_digest": output_digest,
            "record_count": len(entries), "page_count": len(pages), "pages": pages,
            "records_digest": records_digest, **source_binding,
            "eligibility": "discovery_only_no_promotion", "complete_retention": start == len(entries)}


def read_inventory_page(manifest, ledger, cursor=0):
    """Bounded, digest-verified page read. No model call or business mutation."""
    if (not isinstance(manifest, dict) or manifest.get("version") != INVENTORY_VERSION
            or type(cursor) is not int or cursor < 0 or cursor >= manifest.get("page_count", 0)):
        raise ValueError("discovery_inventory_cursor_invalid")
    page = manifest["pages"][cursor]
    import re
    if (not re.fullmatch(r"[a-f0-9]{64}", str(manifest.get("source_output_digest")))
            or not re.fullmatch(r"\d{4}-\d{2}-\d{2}-inventory-" + manifest["source_output_digest"] + "-" + str(cursor) + r"\.json", str(page.get("file")))):
        raise ValueError("discovery_inventory_page_binding_invalid")
    raw = ledger.read_bytes(page["file"])
    if len(raw) > INVENTORY_PAGE_BYTES or len(raw) != page["bytes"] or hashlib.sha256(raw).hexdigest() != page["sha256"]:
        raise ValueError("discovery_inventory_page_binding_invalid")
    value = json.loads(raw)
    if (value.get("version") != INVENTORY_VERSION or value.get("source_output_digest") != manifest["source_output_digest"]
            or value.get("start") != page["start"] or value.get("end") != page["end"]
            or len(value.get("records", [])) != page["end"] - page["start"]):
        raise ValueError("discovery_inventory_page_binding_invalid")
    return {**value, "record_count": manifest["record_count"],
            "next_cursor": cursor + 1 if cursor + 1 < manifest["page_count"] else None}


def instructions(target_usd=1):
    return (
        "Research a defined, evidence-based scope of NEW distinct commercial site/task opportunities. Never pad the list. "
        "Work in two stages: first discover relevant physical work at named operating sites, then rank and "
        "assess the useful candidates against their evidence and limits. Do not require a verified contact, "
        "buying interest, budget or robot fit before retaining relevant work. Within the requested scope, "
        "consider basic handling families such as loading/unloading, picking/placing, kitting/sorting, "
        "moving bins/totes, line feeding, machine tending, portioning/packing and finishing; these are "
        "search hypotheses, not a requirement to cover every "
        "family or expand beyond the admitted geography, time or spend. "
        "Before external searches, read the broader Team Directory through the existing Notion read "
        f"connection when available: {TEAM_DIRECTORY}. Read its index, role groups and relevant full "
        "records, not just the small capability register in the supplied snapshot. Record actual directory "
        "coverage and original evidence dates; directory entries are research leads, not qualified partners. "
        "Advance useful unqualified backlog records from prior runs when their retained evidence is available "
        "through existing authorized history tools; newness is judged against qualified CRM/history, not just "
        "whether a site appeared in yesterday's raw inventory. Reuse retained lists instead of buying the same "
        "enumeration again. Missing backlog access is an explicit gap, not a claim that it was consumed. "
        "Read prior run summaries, tested hypotheses, rejections and actual outcomes through authorized "
        "company history. Missing or expired access is a gap, never permission to widen scope. If Notion "
        "reads fail, use the reviewed context and public primary sources to broaden discovery; do not treat "
        "the small snapshot as the whole directory or stop all research. Build a capability/task/industry "
        "opportunity map from that evidence and survey several distinct supported task families before "
        "spending most of the run on one. Choose queries and how to divide effort across families yourself, "
        "from evidence and prior outcomes. An early easy hit, a prior success such as laundry or the "
        "first CRM-ready row is not a reason to end exploration. Do not search only for recent "
        "announcements: established operating sites, operator service/process pages and employer-affiliated "
        "job descriptions can show recurring work without new funding or an expansion announcement. Keep "
        "confidence in site/task evidence and robot fit separate from buying interest: strong public task "
        "evidence can coexist with unknown interest; never lower task confidence only because nobody has "
        "replied, and never infer demand from a task. "
        "Make the first stage broad enumeration, not deep qualification of the first few hits. For a "
        "broad United States operating-site scope, seek an inventory on the order of hundreds when available "
        "sources and the admitted envelope support it; this is an exploration objective, never a minimum "
        "quota or permission to invent records. Use operator facility/location directories, sourced site "
        "lists before buying another enumeration; use available FindAll when it adds useful coverage, then "
        "concentrate on individual deep research. Preserve "
        "each identifiable physical site from a list as its own inventory record with its actual source. "
        "A directory entry can establish a possible site while its operator, exact task, current manual "
        "workflow and interest remain unknown. Set unsupported fields to null and state the evidence gap; "
        "do not discard that raw possibility for failing the formal candidate evidence requirements. "
        "After enumeration, prioritize a bounded subset for task/workflow evidence, contradictions, "
        "robot fit and contacts. Keep all other sourced possibilities in the backlog for subsequent runs. "
        "Do not fully investigate every enumerated site in this run or put every raw possibility into "
        "formal candidates. Report raw inventory records, distinct sites after actual deduplication, "
        "formal assessments and qualified/accepted prospects separately, as plain sentences in findings; "
        "report findings are not a site count. Do not add keys to coverage: its fields are fixed by the "
        "contract. State coverage gaps and requested-versus-returned provider counts without implying an "
        "exhaustive market census. Keep the whole final output under 2,000,000 bytes: put each enumerated "
        "site in discovery_inventory with short fields, not in findings, and summarize rather than copy "
        "provider snapshots. "
        "Use exact operator-sourced site/location and evidence of real recurring physical work. "
        "Use employer job postings alongside operator pages, case studies and other discovery sources, "
        "including an employer's own applicant-tracking page. Extract the quoted physical duties linked "
        "to the exact operating site; a role title or whole job is not a robot-capable task. Hiring is "
        "optional: relevant active hiring may prioritize a sourced site/task, but is neither a discovery "
        "requirement nor proof of labor shortage, automation interest, manual work or robot fit. Retain "
        "employer, site, requisition ID when available, original source URL, quoted duties, supported "
        "published/modified dates, observed date and application-status evidence in findings and existing "
        "sourced evidence fields. Unknown dates, status and currentness remain unknown; first-seen dates "
        "and reposts do not prove new vacancies. An open application link alone does not prove active "
        "hiring. Deduplicate posting identities by employer/requisition or canonical posting URL, and "
        "facility/task identities separately, preserving every original source and changed status. "
        "Treat improved conversations or Task Evaluation Runs from hiring-led discovery as a testable "
        "hypothesis: label hiring-supported and other discoveries, propose a relevant first question, "
        "and only record conversations or evaluations when actual retained outcomes support them. "
        "A plausible task/team fit is a labeled hypothesis, not a deployment or pilot qualification. "
        "Unknown buyer interest, owner, budget, provider willingness, support and rights remain explicit "
        "unknowns; they do not alone exclude a sourced discovery opportunity. Check incumbent automation "
        "and a plausible disqualifier before recommending a task. Existing robot deployments are separate "
        "learning contacts in findings, never new-opportunity quota. Existing CRM matches are evidence "
        "refreshes or learning notes, never new opportunities. Blueprint agent QA checks exact and semantic "
        "duplicates before publication. Emit a concise candidate row with one useful non-confidential first "
        "question in proposed_next_action; add deeper notes only for the strongest cases. "
        "Explore relevant tasks, industries and regions before deepening promising cases; count distinct sites/tasks, "
        "never findings as prospects. Retain one operator at one physical site with one bounded task per record; "
        "country lists, company-wide services and multi-site aggregates are not individual sites. Compare exact "
        "facilities and tasks with supplied CRM/history after discovery, preserving aliases and uncertain matches; "
        "the same company or city alone is not a duplicate, and unavailable history cannot establish newness. "
        "Retain every distinct discovered site/task in discovery_inventory, including thin, unresolved, disputed, rejected, "
        "duplicate and learning discoveries. Use separate records for separate sites; unknown operator/site/location/task_hypothesis is null. "
        "Each record has operator, site, location, task_hypothesis, source_urls, evidence_gap and disposition "
        "(candidate, unresolved, rejected, learning or duplicate). Preserve actual sources and the next evidence question. "
        "Never drop discoveries because the brief or one response is full; full inventory is stored in resumable pages. "
        "A raw candidate needs task and geography evidence; robot capability is optional and unmatched potential_robot_match stays unknown. "
        "Missing evidence alone is not rejection or automatic qualification. "
        "Distinguish publication/event dates from current operating state; a fresh page retrieval does not "
        "make an old workflow current. Check automation of the exact proposed physical step: conveyors, "
        "inventory software or downstream pallet handling do not alone prove robotic extraction or eliminate "
        "remaining manual work. A site may explore robotics before a proven ROI advantage: do not assume "
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
        "blocked, or the admitted time/spend envelope closes. Before stopping, revisit underexplored "
        "supported task families and unresolved strong leads; judge marginal returns across the stated "
        "breadth, not one family's query sequence. Retain every defensible prospect within the "
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
