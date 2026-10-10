"""Owner direction for outreach-ready hypotheses in daily QA; shadow mode unless enabled.

The owner command (operators/outreach-ready-direction.py) writes one content-addressed,
create-only direction object and pins it top-level as ``control.outreach_ready =
{enabled, current: {sha256, generation, version, uri, direction}}`` through the fenced
compare-and-swap, like ``paid_expansion``: config keys stay allowlisted and older packages
ignore it. The pin binds the object's generation and SHA-256; control also carries the
direction itself, so the worker never reads object storage.

The runner freezes the pin once per daily row under its lease, before the durable intent
(``freeze``). Absent, disabled or naming only site_screen, the run is in shadow mode: the
row, its create payload, packet, QA input, review, publication payloads and status output
stay byte-identical to a release without this feature. After publication completes, the
consumer records the tier each candidate would get in the separate ledger file
``<date>-outreach-ready-shadow.json`` (``shadow``), within a read and time budget, and
nothing is admitted. An enabled pin that directs daily_qa pins lead-verification result v3
for the row. QA may then list outreach-ready keys; admission also needs the live pin (the
brake applies at once) and publishes them only as rows labelled "Hypothesis, not
verified". The brake does not reach hypotheses that review already bound into the
publication payloads. ``sends_authorized`` is always false, and no CRM write path is added
beyond the existing publication. Standard library only.
"""
import hashlib
import re
from datetime import datetime, timedelta

from tools.daily_research import verification
from tools.daily_research.runner import AGENT, PROJECT, canonical

DIRECTION = "blueprint.outreach-ready-direction.v1"
ADMISSION = "blueprint.outreach-ready-admission.v1"
SHADOW = "blueprint.outreach-ready-shadow.v1"
BUCKET = "blueprint-8c1ca.appspot.com"
OBJECT_PREFIX = "operations/research/outreach-ready/"
PATHS = ("daily_qa", "site_screen")  # Mirrored by the bridge's OR_PATHS.
LABEL = "hypothesis"
MAX_ROWS_PER_BATCH = 50
MAX_OBJECT_BYTES = 16 * 1024  # Mirrored by the bridge; a direction is well under 2 KiB.
MAX_TERM = timedelta(days=366)
BINDING = {"project_id": PROJECT, "agent_id": AGENT, "firestore_root": "blueprintDailyResearch/sites-first",
           "run_key_prefix": "blueprint-researcher:", "timezone": "America/Chicago"}
FIELDS = frozenset({"schema_version", "version", "supersedes", "rule_version", "scope", "binding", "effective_from",
                    "expires_at", "approval_reference", "approved_by", "issued_at", "reason"})
SCOPE_FIELDS = frozenset({"paths", "label", "max_rows_per_batch", "sends_authorized"})
ENTRY_FIELDS = frozenset({"sha256", "generation", "version", "uri", "direction"})
TEXT = re.compile(r"[\x20-\x7e]{1,500}")
STAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00")  # whole UTC seconds, as the bridge
SHA = re.compile(r"[a-f0-9]{64}")
GENERATION = re.compile(r"[1-9][0-9]{0,18}")
CODE = re.compile(r"outreach_ready_[a-z_]{1,80}")
NOT_DIRECTED = "outreach_ready_daily_qa_not_directed"


def digest(direction):
    return hashlib.sha256(canonical(direction).encode()).hexdigest()


def uri(sha256):
    return f"gs://{BUCKET}/{OBJECT_PREFIX}{sha256}/direction.json"


def stamp(value):
    if not isinstance(value, str) or not STAMP.fullmatch(value):
        raise ValueError("outreach_ready_time_invalid")
    return datetime.fromisoformat(value)


def direction_problem(direction):
    """None for a well-formed owner direction, otherwise a named refusal."""
    try:
        if (not isinstance(direction, dict) or set(direction) != FIELDS or direction["schema_version"] != DIRECTION
                or direction["rule_version"] != verification.OUTREACH_RULE_VERSION):
            return "outreach_ready_direction_invalid"
        if direction["binding"] != BINDING:
            return "outreach_ready_binding_mismatch"
        scope, rows = direction["scope"], direction["scope"].get("max_rows_per_batch") if isinstance(direction["scope"], dict) else None
        if (not isinstance(scope, dict) or set(scope) != SCOPE_FIELDS or scope["label"] != LABEL
                or scope["sends_authorized"] is not False or type(rows) is not int or not 1 <= rows <= MAX_ROWS_PER_BATCH
                or not isinstance(scope["paths"], list) or not scope["paths"]
                or any(path not in PATHS for path in scope["paths"]) or scope["paths"] != sorted(set(scope["paths"]))):
            return "outreach_ready_scope_invalid"
        version, supersedes = direction["version"], direction["supersedes"]
        issued, start, end = (stamp(direction[key]) for key in ("issued_at", "effective_from", "expires_at"))
        if (type(version) is not int or not 1 <= version <= 1_000_000 or (supersedes is None) != (version == 1)
                or supersedes is not None and (not isinstance(supersedes, str) or not SHA.fullmatch(supersedes))
                or any(not isinstance(direction[key], str) or not TEXT.fullmatch(direction[key]) or not direction[key].strip()
                       for key in ("approval_reference", "approved_by", "reason"))
                or direction["approval_reference"].strip().upper().startswith("PENDING")
                or not issued <= start < end or end - issued > MAX_TERM):
            return "outreach_ready_direction_invalid"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "outreach_ready_direction_invalid"
    return None


def entry_problem(entry):
    """None when a control entry is exactly a verified, content-addressed, generation-pinned direction."""
    if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
        return "outreach_ready_direction_invalid"
    code = direction_problem(entry["direction"])
    if code:
        return code
    if (not isinstance(entry["sha256"], str) or digest(entry["direction"]) != entry["sha256"]
            or type(entry["version"]) is not int or entry["version"] != entry["direction"]["version"]
            or entry["uri"] != uri(entry["sha256"])
            or not isinstance(entry["generation"], str) or not GENERATION.fullmatch(entry["generation"])):
        return "outreach_ready_direction_digest_mismatch"
    return None


def current(control):
    """(entry, None) for the enabled, verified current direction, else (None, refusal)."""
    pin = control.get("outreach_ready") if isinstance(control, dict) else None
    if not isinstance(pin, dict) or pin.get("enabled") is not True:
        return None, "outreach_ready_disabled"
    if set(pin) != {"enabled", "current"}:
        return None, "outreach_ready_direction_invalid"
    code = entry_problem(pin["current"])
    return (None, code) if code else (pin["current"], None)


def assess(value, row, now):
    """The record a run would freeze from control.outreach_ready, or None when absent or disabled; never raises.

    An unusable, expired, not-yet-effective or screen-only direction gives a refusal record
    with its code. ``freeze`` drops the screen-only refusal: it does not concern daily runs.
    """
    if value is None or isinstance(value, dict) and value.get("enabled") is False:
        return None
    entry, code = current({"outreach_ready": value})
    record = {"schema_version": ADMISSION, "run_key": row.get("run_key"), "frozen_at": now.isoformat(),
              "direction_sha256": entry["sha256"] if entry else None}
    if code is None:
        direction = entry["direction"]
        try:
            if now < stamp(direction["effective_from"]):
                code = "outreach_ready_not_yet_effective"
            elif now >= stamp(direction["expires_at"]):
                code = "outreach_ready_expired"
            elif "daily_qa" not in direction["scope"]["paths"]:
                code = NOT_DIRECTED
            elif not isinstance(record["run_key"], str) or not record["run_key"].startswith(BINDING["run_key_prefix"]):
                code = "outreach_ready_binding_mismatch"
        except (KeyError, TypeError, ValueError):
            code = "outreach_ready_direction_invalid"
    if code:
        return {**record, "state": "refused", "code": code}
    scope = direction["scope"]
    return {**record, "state": "enabled", "generation": entry["generation"], "version": entry["version"],
            "uri": entry["uri"], "rule_version": direction["rule_version"], "paths": list(scope["paths"]),
            "label": scope["label"], "max_rows_per_batch": scope["max_rows_per_batch"], "sends_authorized": False,
            "approval_reference": direction["approval_reference"], "valid_until": direction["expires_at"]}


def freeze(value, row, now):
    """This run's frozen direction record, or None for shadow mode with an unchanged row; never raises.

    ``value`` is control.outreach_ready, read once under the run's lease before the durable
    intent. Absent, disabled or a direction that names only site_screen gives None: the row
    and its bridge manifest stay exactly as without the feature. A daily_qa direction that
    is unusable, expired or not yet effective gives a refusal record: shadow mode with its code.
    """
    record = assess(value, row, now)
    return None if record is not None and record.get("code") == NOT_DIRECTED else record


def frozen_record(reader, row, now, ceiling):
    """(record or None, code or None) for Runner.start_or_resume; never raises.

    A failed control read freezes a refusal with outreach_ready_control_unavailable. A record
    that would take the intent past ``ceiling`` bytes is replaced by a compact refusal with
    outreach_ready_record_too_large; when even that does not fit, nothing is frozen and the
    code is returned for the caller to report.
    """
    try:
        record = freeze(reader(), row, now)
    except Exception:  # noqa: BLE001 - an optional direction never stops the daily run
        record = {"schema_version": ADMISSION, "run_key": row.get("run_key"), "frozen_at": now.isoformat(),
                  "direction_sha256": None, "state": "refused", "code": "outreach_ready_control_unavailable"}
    if record is None:
        return None, None
    for candidate in (record, {key: record[key] for key in ("schema_version", "run_key", "frozen_at", "direction_sha256")}
                      | {"state": "refused", "code": "outreach_ready_record_too_large"}):
        if len(canonical({**row, "outreach_ready": candidate}).encode()) <= ceiling:
            return candidate, candidate.get("code")
    return None, "outreach_ready_record_too_large"


def enabled(row, path="daily_qa"):
    """True when this row froze an enabled direction for ``path``: result v3, QA asks, publication may label."""
    frozen = row.get("outreach_ready") if isinstance(row, dict) else None
    return (isinstance(frozen, dict) and frozen.get("state") == "enabled" and frozen.get("sends_authorized") is False
            and isinstance(frozen.get("paths"), list) and path in frozen["paths"]
            and type(frozen.get("max_rows_per_batch")) is int and 1 <= frozen["max_rows_per_batch"] <= MAX_ROWS_PER_BATCH)


def admission(row, control, now, path="daily_qa"):
    """(row limit, None) while hypotheses may be admitted now, else (0, refusal).

    The frozen record is the run's upper bound. Live control can only tighten it: the brake,
    an expiry, a removed path or a lower row limit apply at once; a new direction never widens
    a run in progress.
    """
    if not enabled(row, path):
        return 0, "outreach_ready_not_enabled_for_run"
    entry, code = current(control)
    if code:
        return 0, code
    direction, frozen = entry["direction"], row["outreach_ready"]
    try:
        if path not in direction["scope"]["paths"]:
            return 0, "outreach_ready_path_not_directed"
        if now < stamp(direction["effective_from"]) or now >= min(stamp(frozen["valid_until"]), stamp(direction["expires_at"])):
            return 0, "outreach_ready_expired"
    except (KeyError, TypeError, ValueError):
        return 0, "outreach_ready_direction_invalid"
    return min(frozen["max_rows_per_batch"], direction["scope"]["max_rows_per_batch"]), None



# Retained evidence reads use the existing elapsed-time guard, without a count quota.
EVIDENCE_MAX_SECONDS = 60
# The shadow runs after publication completed and is skipped with a recorded code when less than
# this remains before the run's absolute QA and publication deadline.
SHADOW_DEADLINE_MARGIN = timedelta(seconds=120)
SHADOW_FILE_SUFFIX = "-outreach-ready-shadow.json"
SHADOW_MAX_BYTES = 64 * 1024
# A published valid_until must be null or this exact form (Blueprint-WebApp #855 ISO_TIMESTAMP);
# the assessment itself also accepts a space separator, no seconds or an offset without a colon.
VALID_UNTIL = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})")


def admissible_until(result, deadline):
    """True when the assessment stays valid through the run: valid_until is null, or a timestamp in
    the form the WebApp records (VALID_UNTIL) later than ``deadline`` (consumer.qa_deadline, which
    also bounds publication). A hypothesis admitted under this rule never expires while its run's
    QA, review and publication can still act."""
    if not isinstance(deadline, datetime):
        return False
    try:
        valid_until = result["assessment"]["valid_until"]
        return valid_until is None or (isinstance(valid_until, str) and bool(VALID_UNTIL.fullmatch(valid_until))
                                       and verification.moment(valid_until) > deadline)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def shadow_file(row):
    return row["date"] + SHADOW_FILE_SUFFIX


def read_budget():
    return verification.ReadBudget(None, EVIDENCE_MAX_SECONDS)


def shadow_skipped(row, code, now):
    return {"schema_version": SHADOW, "state": "skipped", "code": code, "run_key": row.get("run_key"),
            "recorded_at": now.isoformat(), "admitted": []}


def shadow(row, qa_result, evidence, now, deadline):
    """Shadow-mode record of the tier each candidate would get; written outside the row. Never raises.

    Runs once publication completed. Recomputes result v3 over the QA checks with the retained
    evidence at the QA decision's own evaluation time. QA was not asked to list keys, so
    nothing is admitted; would_admit counts promotable, unaccepted candidates the rule rates
    outreach_ready whose QA check is source-verified and not a duplicate and whose assessment
    stays valid past ``deadline``, before the CRM recheck an enabled run would add.
    """
    try:
        packet, checks, review = row["packet"], qa_result["checks"], row["review"]
        results = (review.get("lead_verification") or {}).get("results") or []
        evaluated_at = verification.moment(results[0]["evaluated_at"]) if results else now
        value = verification.cohort(
            verification.packet_candidates(packet), {c["candidate_key"]: c.get("lead_verification") for c in checks},
            evaluated_at, duplicate_checks={c["candidate_key"]: {"duplicate": c["duplicate"], "duplicate_of": c.get("duplicate_of"),
                                                                  "reason": c["reason"]} for c in checks},
            result_version=verification.OUTREACH_RESULT_VERSION, evidence=evidence)
        promotable = {c["candidate_key"] for c in packet["candidates"]}
        accepted = {key for key in review["accepted_keys"] if isinstance(key, str)}
        supported = {c["candidate_key"] for c in checks if c.get("source_support_verified") is True and c.get("duplicate") is False}
        would = [r["candidate_key"] for r in value["results"] if r["eligible_for_outreach_ready"] is True
                 and r["candidate_key"] in promotable and r["candidate_key"] not in accepted
                 and r["candidate_key"] in supported and admissible_until(r, deadline)]
        frozen = row.get("outreach_ready") or {}
        record = {"schema_version": SHADOW, "state": "recorded", "rule_version": verification.OUTREACH_RULE_VERSION,
                  "run_key": row["run_key"], "evaluated_at": evaluated_at.isoformat(), "recorded_at": now.isoformat(),
                  "direction_state": frozen.get("state", "absent"), "direction_code": frozen.get("code"),
                  "tier_evidence": value["tier_evidence"],
                  "tiers": {tier: sum(r["tier"] == tier for r in value["results"]) for tier in ("verified", "outreach_ready", "none")},
                  "would_admit_count": len(would), "would_admit": would, "admitted": [],
                  "results": [{"candidate_key": r["candidate_key"], "tier": r["tier"], "blockers": r["outreach_ready"]["blockers"],
                               "open_checks": r["outreach_ready"]["open_checks"],
                               "open_questions": r["outreach_ready"]["open_questions"],
                               "proof_levels": {p["claim"]: p["level"] for p in r["outreach_ready"]["proving_sources"]}}
                              for r in value["results"]]}
        if len(canonical(record).encode()) > SHADOW_MAX_BYTES:
            record = {key: item for key, item in record.items() if key not in {"results", "would_admit"}}
        return record
    except Exception:  # noqa: BLE001 - shadow measurement never blocks QA or publication
        return shadow_skipped(row, "outreach_ready_shadow_unavailable", now)
