"""Rank screened robot teams for Blueprint's free invited beta (``blueprint.team-rank.v1``). Standard library only.

The reviewed config (``rank.v1.json``, strictly checked) fixes the points and the thresholds. The per-family weights
are an input file the owner keeps private, because they come from the private site screen report: how many
outreach-ready sites each task family has. A site screen ``summary.json`` is accepted too, read as its
outreach-ready count by task focus. Nothing in this repository holds real weights or real teams.

The rule reads verified evidence only (``universe.screen_record``: a proven quote that names its answer). Points,
out of 100 with the reviewed config:

- fit: ``points.fit`` times the best weight, scaled to the largest weight, among the team's proven task families;
- early stage: ``points.early_stage`` when a proven funding quote names a stage in ``early_stages``;
- openness: ``points.openness`` times the proven ``openness_signals`` over ``openness_full_at`` (at most 1);
- policy: ``points.policy`` when the team builds or shares a learned policy (``policy_signals``);
- recent activity: ``points.recent_activity`` when its latest proven dated evidence is at most ``recent_days`` old.

``beta_candidate`` needs a verified identity on the team's own domain, a proven task family with a positive weight,
a published contact route, a policy or a way to evaluate one (``evaluable_signals``) and at least
``thresholds.beta_candidate`` points. ``prospect`` needs a verified identity, a proven task family or robot form and
at least ``thresholds.prospect`` points. Everything else is ``insufficient``: a team not yet screened, an unverified
or contradicted identity, and any defect in the computation. Every tier is a prospect, never a relationship: nothing
here contacts a team, and the contact kept is only a published business route.
"""
import math
from collections import Counter
from pathlib import Path

from tools.daily_research import site_screen as ss
from tools.team_universe import universe as tu

RANK = "blueprint.team-rank.v1"
WEIGHTS = "blueprint.team-family-weights.v1"
RANKED = "blueprint.team-rank.ranked.v1"
TIERS = ("beta_candidate", "prospect", "insufficient")
CONFIG_PATH = Path(__file__).resolve().with_name("rank.v1.json")
POINTS = ("fit", "early_stage", "openness", "policy", "recent_activity")
CONFIG_KEYS = frozenset({"schema_version", "points", "early_stages", "openness_signals", "openness_full_at",
                         "policy_signals", "evaluable_signals", "recent_days", "thresholds"})
MAX_WEIGHT = 1_000_000
CLAIM_BOUNDARY = {"teams_are_prospects": True, "relationship_implied": False,
                  "contact_is_published_business_route_only": True, "nothing_sent": True}


def _ids(value, allowed):
    """A non-empty list of distinct allowed ids."""
    return (isinstance(value, list) and bool(value) and all(isinstance(item, str) and item in allowed for item in value)
            and len(set(value)) == len(value))


def load_config(raw=None):
    """The rank config, strictly checked: exactly its keys, whole points out of 100 in total, known stages and
    signals, an openness count within its signals, a recent window of at most three years, and a beta_candidate
    threshold above the prospect threshold. ``raw`` None reads the reviewed rank.v1.json."""
    if raw is None:
        try:
            raw = CONFIG_PATH.read_bytes()
        except OSError:
            raise tu.TeamError("team_universe_rank_config_unreadable") from None
    config = ss._json(raw)
    try:
        points, thresholds = config["points"], config["thresholds"]
        valid = (set(config) == CONFIG_KEYS and config["schema_version"] == RANK
                 and isinstance(points, dict) and set(points) == set(POINTS)
                 and all(type(points[name]) is int and 0 <= points[name] <= 100 for name in POINTS)
                 and sum(points.values()) == 100
                 and _ids(config["early_stages"], set(tu.STAGE_CHOICES) - {"unknown"})
                 and _ids(config["openness_signals"], tu.SIGNALS) and _ids(config["policy_signals"], tu.SIGNALS)
                 and _ids(config["evaluable_signals"], tu.SIGNALS)
                 and type(config["openness_full_at"]) is int
                 and 1 <= config["openness_full_at"] <= len(config["openness_signals"])
                 and type(config["recent_days"]) is int and 1 <= config["recent_days"] <= 1095
                 and isinstance(thresholds, dict) and set(thresholds) == {"beta_candidate", "prospect"}
                 and all(type(value) is int for value in thresholds.values())
                 and 0 < thresholds["prospect"] < thresholds["beta_candidate"] <= 100)
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise tu.TeamError("team_universe_rank_config_invalid")
    return {**config, "sha256": ss._sha256(bytes(raw))}


def _weight(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= MAX_WEIGHT


def load_weights(raw):
    """Per-family weights, strictly checked: a ``blueprint.team-family-weights.v1`` file
    (``{"schema_version", "reference", "weights": {task family: a number from 0 to 1,000,000}}``), or a site screen
    summary.json read as its outreach-ready count by task focus. A family left out weighs zero, at least one weight
    must be above zero, and each is scaled to the largest."""
    document = ss._json(raw)
    version = document.get("schema_version") if isinstance(document, dict) else None
    if version == WEIGHTS:
        values, reference = document.get("weights"), document.get("reference")
        if (set(document) != {"schema_version", "reference", "weights"} or not isinstance(reference, str)
                or not ss.REFERENCE.fullmatch(reference) or not isinstance(values, dict) or not values
                or not all(family in tu.TASK_FAMILIES and _weight(value) for family, value in values.items())):
            raise tu.TeamError("team_universe_family_weights_invalid")
        source = "weights_file"
    elif version == ss.SUMMARY:
        screen = document.get("screen")
        focus = screen.get("tiers_by_focus") if isinstance(screen, dict) else None
        if not isinstance(focus, dict):
            raise tu.TeamError("team_universe_family_weights_invalid")
        values = {}
        for family, tiers in focus.items():
            if family in tu.TASK_FAMILIES:
                count = tiers.get("outreach_ready") if isinstance(tiers, dict) else None
                if type(count) is not int or not 0 <= count <= MAX_WEIGHT:
                    raise tu.TeamError("team_universe_family_weights_invalid")
                values[family] = count
        reference, source = None, "site_screen_summary"
    else:
        raise tu.TeamError("team_universe_family_weights_invalid")
    top = max(values.values(), default=0)
    if not top > 0:
        raise tu.TeamError("team_universe_family_weights_empty")
    weights = {family: values.get(family, 0) for family in tu.TASK_FAMILIES}
    return {"source": source, "reference": reference, "sha256": ss._sha256(bytes(raw)), "weights": weights,
            "normalized": {family: weights[family] / top for family in tu.TASK_FAMILIES}}


def _recent(record, days):
    latest, checked = tu._day(record.get("latest_activity")), tu._day(record.get("checked_on"))
    return bool(latest and checked and 0 <= (checked - latest).days <= days)


def assess(record, config, weights):
    """One team's score, components, tier and blockers under the config. A team without a screen record is
    insufficient (not_screened), and any defect in the computation is insufficient, never a higher tier."""
    if record is None:
        return {"tier": "insufficient", "score": 0, "components": None, "blockers": ["not_screened"]}
    try:
        points, thresholds, signals = config["points"], config["thresholds"], record["signals"]
        families, identity = record["task"]["proven"], record["identity"]["state"]
        fit = max((weights["normalized"][family] for family in families), default=0)
        opened = [name for name in config["openness_signals"] if signals[name]["proven"]]
        evaluable = any(signals[name]["proven"] for name in config["evaluable_signals"])
        early = record["stage"]["verified"] and record["stage"]["answer"] in config["early_stages"]
        components = {
            "fit": round(points["fit"] * fit, 2),
            "early_stage": points["early_stage"] if early else 0,
            "openness": round(points["openness"] * min(1, len(opened) / config["openness_full_at"]), 2),
            "policy": points["policy"] if any(signals[name]["proven"] for name in config["policy_signals"]) else 0,
            "recent_activity": points["recent_activity"] if _recent(record, config["recent_days"]) else 0}
        score = round(sum(components.values()), 2)
        blockers = []
        if identity != "verified_fact":
            blockers.append("identity_contradicted" if identity == "contradicted" else "identity_not_verified")
        if not families:
            blockers.append("task_family_unproven")
        elif fit <= 0:
            blockers.append("task_family_weight_zero")
        if record["contact"]["route"] not in ("role_inbox", "contact_page"):
            blockers.append("contact_route_missing")
        if not evaluable:
            blockers.append("not_evaluable")
        if score < thresholds["beta_candidate"]:
            blockers.append("score_below_beta_candidate")
        if not blockers:
            tier = "beta_candidate"
        elif (identity == "verified_fact" and (families or record["robot_forms"]["proven"])
              and score >= thresholds["prospect"]):
            tier = "prospect"
        else:
            tier = "insufficient"
        return {"tier": tier, "score": score, "components": components, "blockers": blockers,
                "openness_signals": opened, "evaluable": evaluable}
    except Exception:  # noqa: BLE001 - any defect in the rule yields insufficient, never a higher tier
        return {"tier": "insufficient", "score": 0, "components": None, "blockers": ["rank_computation_unavailable"]}


def ranked_row(team, record, config, weights):
    """One team in the ranked file: its tier and score, the proven evidence behind them, its published contact route
    and its discovery sources. Always a prospect."""
    subject = team or record
    found = (team or {}).get("discovery") or (record or {}).get("discovery") or {}
    contact = (record or {}).get("contact") or {}
    return {"team_key": subject["site_key"], "domain": subject["domain"],
            "company": ((record or {}).get("identity") or {}).get("company")
            or (subject.get("task_input") or subject.get("input") or {}).get("company"),
            "status": "prospect", **assess(record, config, weights),
            "robot_forms": record["robot_forms"]["proven"] if record else [],
            "task_families": record["task"]["proven"] if record else [],
            "stage": {name: record["stage"][name] for name in ("answer", "verified")} if record else None,
            "signals": sorted(name for name, signal in record["signals"].items() if signal["proven"]) if record else [],
            "contact": {name: contact[name] for name in ("route", "address", "url") if name in contact} or None,
            "latest_activity": record.get("latest_activity") if record else None,
            "discovery": {name: found.get(name) for name in ("proven", "mentions", "query_families", "families",
                                                             "robot_forms", "latest", "sources")},
            "screen": {"run_id": record["run_id"], "rule_version": record["rule_version"],
                       "checked_on": record["checked_on"]} if record else None,
            "proving_sources": record["proving_sources"] if record else []}


def rank(workspace, weights_raw, *, config_raw=None):
    """Rank every discovered or screened team under the config and the private weights, and write the ranked file
    (private; names, domains and contact routes included) to the out dir. The result holds counts only. Reads no page
    and calls no provider."""
    config, weights = load_config(config_raw), load_weights(weights_raw)
    with workspace.lock():
        states, _ = workspace.states()
        teams, _ = tu.team_list(tu.stage_records(workspace, states, "discover").values())
        screens = tu.stage_records(workspace, states, "screen")
        listed = {team["site_key"]: team for team in teams}
        rows = [ranked_row(team, screens.get(key), config, weights) for key, team in listed.items()]
        rows += [ranked_row(None, record, config, weights) for key, record in screens.items() if key not in listed]
        rows.sort(key=lambda row: (TIERS.index(row["tier"]), -row["score"], row["domain"]))
        ss._write_derived(workspace.root / tu.RANKED_NAME, {
            "schema_version": RANKED, "rank_config": RANK, "config_sha256": config["sha256"],
            "weights": {"source": weights["source"], "reference": weights["reference"], "sha256": weights["sha256"],
                        "values": weights["weights"]},
            "rules": {"discover": tu.DISCOVERY_RULE, "screen": tu.SCREEN_RULE}, "claim_boundary": CLAIM_BOUNDARY,
            "teams": rows})
    return {"command": "rank", "state": "complete", "rank_config": RANK, "config_sha256": config["sha256"],
            "weights_source": weights["source"], "weights_sha256": weights["sha256"], "teams": len(rows),
            "screened": sum(row["screen"] is not None for row in rows),
            "tiers": {tier: sum(row["tier"] == tier for row in rows) for tier in TIERS},
            "blockers": dict(Counter(code for row in rows for code in row["blockers"])), "written": tu.RANKED_NAME}
