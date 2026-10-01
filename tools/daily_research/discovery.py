"""Adaptive discovery instructions and conservative, explicitly estimated costs.

No provider, credentials, sink writes or scheduling. The daily $1 soft target is
separate from the explicitly authorized one-time test ceiling.
"""
from decimal import Decimal

TARGET_NEW = 10
MAX_CANDIDATES = 100  # Input/resource ceiling, not a research quota.


def instructions(target_usd=1):
    return (
        "Target at least 10 NEW distinct commercial site/task opportunities. Never pad the list. "
        "Use exact operator-sourced site/location and evidence of real recurring physical work. "
        "A plausible task/team fit is a labeled hypothesis, not a deployment or pilot qualification. "
        "Unknown buyer interest, owner, budget, provider willingness, support and rights remain explicit "
        "unknowns; they do not alone exclude a sourced discovery opportunity. Check incumbent automation "
        "and a plausible disqualifier before recommending a task. Existing robot deployments are separate "
        "learning contacts in findings, never new-opportunity quota. Existing CRM matches are evidence "
        "refreshes or learning notes, never new opportunities. Blueprint agent QA checks exact and semantic "
        "duplicates before publication. Emit a concise candidate row with one useful non-confidential first "
        "question in proposed_next_action; add deeper notes only for the strongest cases. "
        "Explicitly read /workspace/capabilities/blueprint/deep-research/SKILL.md and "
        "/workspace/capabilities/blueprint/blueprint-evidence-qualification/SKILL.md and its "
        "references/prospect-contract.md. Use the reviewed versions. Decompose the task, vary searches, "
        "open primary operator and product sources, follow gaps and contradictions, and reuse reviewed "
        "knowledge. There is no two-search/two-open or three-candidate quality cap. Stop when material "
        "questions are answered at a stated evidence scope, next sources add little value, access is "
        "blocked, or the admitted time/spend envelope closes. If fewer than 10 withstand research, keep "
        "the supported subset and give coverage and shortfall reasons. A source failure is a visible gap, "
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
    elif candidate_count < TARGET_NEW and reason is None:
        raise ValueError("discovery_shortfall_reason_required")
    return value


def estimated_model_cost(usage):
    """Upper-rate token estimate, never a bound on unsettled total provider cost.

    Charge all input at combined long-context input + cache-write rates ($9/M), all output
    including reasoning at $15/M, and add the documented 10% regional premium.
    This ignores cache discounts conservatively. Standard service is required
    by test admission; fast mode is excluded. SDK usage can lag or be absent.
    """
    if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ("input_tokens", "output_tokens"))):
        return {"known": False, "estimate_usd": None, "hard_total_cap": False}
    price = (Decimal(usage["input_tokens"]) * 9 + Decimal(usage["output_tokens"]) * 15) * Decimal("1.1") / 1000000
    return {"known": True, "estimate_usd": str(price), "hard_total_cap": False,
            "excludes": ["unreported_or_lagged_usage", "tool_fees", "hosted_environment"]}
