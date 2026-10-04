"""Owner-directed paid expansion allowance: the single producer and arbiter.

The owner sets one combined per-run USD limit for paid expansion sources (Exa
now; FindAll later) as a content-addressed direction record. Company control
carries it top-level as ``control.paid_expansion = {enabled, current: {sha256,
version, uri, direction}}``: config keys stay allowlisted and older packages
ignore it. The runner freezes one grant per daily row under its lease before the
durable intent (``grant``); every paid start is admitted only by ``problem``.
A frozen grant is the run's upper bound: live control can only tighten it (the
brake, a changed release, a lower current amount or a removed source apply at
once), while a higher amount waits for the next run's grant.
Durable claims are the debits and an unknown cost holds its whole reservation.
This allowance is host-reserved and separate from the research soft target.
Standard library only; the worker never reads object storage.

FindAll interface (PR 2, TODO): add its name to SOURCES and the bridge's
PAID_SOURCES, return its durable claims from claims() with reserved_micros, and
admit each start with problem(..., source="findall"). The limit stays combined.
"""
import hashlib
import re
from datetime import datetime, timedelta, timezone

from tools.daily_research.runner import AGENT, PROJECT, canonical

DIRECTION = "blueprint.research-paid-expansion-direction.v1"
GRANT = "blueprint.research-paid-expansion-grant.v1"
STATUS = "blueprint.research-paid-expansion-status.v1"
HARD_CEILING_MICROS = 100_000_000  # $100.00 per run; a typo above it refuses instead of spending.
FLOOR_MICROS = 1_000_000  # $1.00
PER_START_CEILING_MICROS = 50_000_000
AMOUNT = re.compile(r"[1-9][0-9]{0,2}(\.[0-9]{2})?")  # ASCII digits only, like the bridge
# TODO(FindAll): add "findall" here, in the bridge's PAID_SOURCES and as a claims() reader.
SOURCES = ("exa",)
BUCKET = "blueprint-8c1ca.appspot.com"
OBJECT_PREFIX = "operations/research/paid-expansion/"
SCOPE = {"project_id": PROJECT, "agent_id": AGENT, "firestore_root": "blueprintDailyResearch/sites-first",
         "run_key_prefix": "blueprint-researcher:", "timezone": "America/Chicago"}
FIELDS = frozenset({"schema_version", "version", "supersedes", "per_run_limit_usd", "sources", "scope",
                    "effective_from", "expires_at", "approval_reference", "approved_by", "issued_at", "reason"})
GRANT_FIELDS = frozenset({"schema_version", "state", "run_key", "frozen_at", "direction_sha256", "grant_id",
                          "direction_uri", "version", "sources", "limit_micros", "per_start_max_micros",
                          "source_commit", "approval_reference", "valid_until"})
MAX_TERM = timedelta(days=366)
TEXT = re.compile(r"[\x20-\x7e]{1,500}")
STAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+00:00")  # whole UTC seconds: exact in Python and Node
SHA = re.compile(r"[a-f0-9]{64}")
COMMIT = re.compile(r"[a-f0-9]{40}")
CODE = re.compile(r"paid_expansion_[a-z_]{1,80}")


def micros(amount):
    """Exact integer microdollars for an owner amount such as "10.00" within $1–$100, else None."""
    if not isinstance(amount, str) or not AMOUNT.fullmatch(amount):
        return None
    whole, _, cents = amount.partition(".")
    value = int(whole) * 1_000_000 + int(cents or "0") * 10_000
    return value if FLOOR_MICROS <= value <= HARD_CEILING_MICROS else None


def usd(value):
    """Two-decimal display of an owner amount in microdollars."""
    return f"{value // 1_000_000}.{value % 1_000_000 // 10_000:02d}"


def per_start_max(limit_micros):
    """Half the run limit for one start, clamped to $1–$50 ($10→$5, $20→$10, $30→$15)."""
    return min(max(limit_micros // 2, FLOOR_MICROS), PER_START_CEILING_MICROS)


def digest(direction):
    return hashlib.sha256(canonical(direction).encode()).hexdigest()


def uri(sha256):
    return f"gs://{BUCKET}/{OBJECT_PREFIX}{sha256}/direction.json"


def grant_id(direction_sha256, run_key):
    return hashlib.sha256(canonical([direction_sha256, run_key]).encode()).hexdigest()


def stamp(value):
    if not isinstance(value, str) or not STAMP.fullmatch(value):
        raise ValueError("paid_expansion_time_invalid")
    return datetime.fromisoformat(value)


def direction_problem(direction):
    """None for a well-formed owner direction, otherwise a named refusal."""
    try:
        if not isinstance(direction, dict) or set(direction) != FIELDS or direction["schema_version"] != DIRECTION:
            return "paid_expansion_direction_invalid"
        if micros(direction["per_run_limit_usd"]) is None:
            return "paid_expansion_limit_invalid"
        if direction["scope"] != SCOPE:
            return "paid_expansion_scope_mismatch"
        version, supersedes, sources = direction["version"], direction["supersedes"], direction["sources"]
        issued, start, end = (stamp(direction[key]) for key in ("issued_at", "effective_from", "expires_at"))
        if (type(version) is not int or not 1 <= version <= 1_000_000 or (supersedes is None) != (version == 1)
                or supersedes is not None and (not isinstance(supersedes, str) or not SHA.fullmatch(supersedes))
                or not isinstance(sources, list) or not sources or any(source not in SOURCES for source in sources)
                or sources != sorted(set(sources))
                or any(not isinstance(direction[key], str) or not TEXT.fullmatch(direction[key]) or not direction[key].strip()
                       for key in ("approval_reference", "approved_by", "reason"))
                or direction["approval_reference"].strip().upper().startswith("PENDING")
                or not issued <= start < end or end - issued > MAX_TERM):
            return "paid_expansion_direction_invalid"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "paid_expansion_direction_invalid"
    return None


def entry_problem(entry):
    """None when a control entry is exactly a verified, content-addressed direction."""
    if not isinstance(entry, dict) or set(entry) != {"sha256", "version", "uri", "direction"}:
        return "paid_expansion_direction_invalid"
    code = direction_problem(entry["direction"])
    if code:
        return code
    if (not isinstance(entry["sha256"], str) or digest(entry["direction"]) != entry["sha256"]
            or type(entry["version"]) is not int or entry["version"] != entry["direction"]["version"]
            or entry["uri"] != uri(entry["sha256"])):
        return "paid_expansion_direction_digest_mismatch"
    return None


def current(control):
    """(entry, None) for the enabled, verified current direction, else (None, refusal)."""
    paid = control.get("paid_expansion") if isinstance(control, dict) else None
    if not isinstance(paid, dict) or paid.get("enabled") is not True:
        return None, "paid_expansion_disabled"
    if set(paid) != {"enabled", "current"}:
        return None, "paid_expansion_direction_invalid"
    code = entry_problem(paid["current"])
    return (None, code) if code else (paid["current"], None)


def grant(control, row, now):
    """Freeze this run's grant from control, or a named refusal record; never raises.

    The runner calls this once per daily row under its lease, before the durable
    intent. A crash or retry reuses the stored row, so an owner amount change
    applies from the next run; the brake and release fences apply at every start.
    """
    entry, code = current(control)
    record = {"schema_version": GRANT, "run_key": row.get("run_key"), "frozen_at": now.isoformat(),
              "direction_sha256": entry["sha256"] if entry else None}
    if code is None:
        direction, commit, seconds = entry["direction"], control.get("source_commit"), row.get("research_runtime_seconds")
        try:
            started = datetime.fromisoformat(row["started_at"])
            if started.tzinfo is None or type(seconds) is not int or not 0 < seconds <= 1800:
                raise ValueError("paid_expansion_run_context_invalid")
            valid_until = min(stamp(direction["expires_at"]), started + timedelta(seconds=seconds))
        except (KeyError, TypeError, ValueError):
            code = "paid_expansion_run_context_invalid"
        else:
            if not isinstance(record["run_key"], str) or not record["run_key"].startswith(SCOPE["run_key_prefix"]):
                code = "paid_expansion_scope_mismatch"
            elif not isinstance(commit, str) or not COMMIT.fullmatch(commit):
                code = "paid_expansion_source_commit_unverified"
            elif now < stamp(direction["effective_from"]):
                code = "paid_expansion_not_yet_effective"
            elif now >= valid_until:
                code = "paid_expansion_expired"
    if code:
        return {**record, "state": "refused", "code": code}
    limit = micros(direction["per_run_limit_usd"])
    return {**record, "state": "granted", "grant_id": grant_id(entry["sha256"], record["run_key"]),
            "direction_uri": entry["uri"], "version": entry["version"], "sources": list(direction["sources"]),
            "limit_micros": limit, "per_start_max_micros": per_start_max(limit), "source_commit": commit,
            "approval_reference": direction["approval_reference"], "valid_until": valid_until.isoformat()}


def refused(value, code):
    """The refusal record for a grant the company store would not admit at the intent."""
    return {"schema_version": GRANT, "state": "refused", "code": code,
            **{key: value.get(key) for key in ("run_key", "frozen_at", "direction_sha256")}}


def granted(value):
    """None for a structurally valid granted record, else its named refusal."""
    if not isinstance(value, dict) or value.get("schema_version") != GRANT:
        return "paid_expansion_grant_missing"
    if value.get("state") == "refused":
        code = value.get("code")
        return code if isinstance(code, str) and CODE.fullmatch(code) else "paid_expansion_grant_invalid"
    try:
        limit, start = value["limit_micros"], value["per_start_max_micros"]
        if (value["state"] != "granted" or set(value) != GRANT_FIELDS
                or type(limit) is not int or not FLOOR_MICROS <= limit <= HARD_CEILING_MICROS
                or type(start) is not int or start != per_start_max(limit)
                or not isinstance(value["sources"], list) or not set(value["sources"]) <= set(SOURCES)
                or value["grant_id"] != grant_id(value["direction_sha256"], value["run_key"])
                or datetime.fromisoformat(value["valid_until"]).tzinfo is None):
            return "paid_expansion_grant_invalid"
    except (KeyError, TypeError, ValueError):
        return "paid_expansion_grant_invalid"
    return None


def standing(value, control, now, *, source="exa"):
    """None while a frozen grant may admit a new paid start of ``source``.

    Live fences checked at every start: the owner's brake (``enabled``), a
    verifiable current direction that still names the source, and the reviewed
    release (``source_commit``).
    """
    code = granted(value)
    if code:
        return code
    if source not in value["sources"]:
        return "paid_expansion_source_not_directed"
    entry, code = current(control)
    if code:
        return code
    if source not in entry["direction"]["sources"]:
        return "paid_expansion_source_not_directed"
    if control.get("source_commit") != value["source_commit"]:
        return "paid_expansion_source_commit_changed"
    if now < stamp(entry["direction"]["effective_from"]):
        return "paid_expansion_not_yet_effective"
    if now >= stamp(entry["direction"]["expires_at"]):
        return "paid_expansion_expired"
    if now >= datetime.fromisoformat(value["valid_until"]):
        return "paid_expansion_expired"
    return None


def effective_limit(value, control):
    """The frozen limit, lowered by a smaller verified current direction; never raised."""
    entry, _ = current(control)
    live = micros(entry["direction"]["per_run_limit_usd"]) if entry else None
    return value["limit_micros"] if live is None else min(value["limit_micros"], live)


def claims(row):
    """This run's durable paid claims. Each debits its whole reserved cap: nothing is
    released without a provider-reported terminal cost, and none is pinned yet."""
    found = []
    exa = row.get("exa_expansion") if isinstance(row, dict) else None
    if exa:
        found.append({"source": "exa", "intent_sha256": exa.get("intent_sha256"), "reserved_micros": exa.get("cap_micros")})
    # TODO(FindAll): append this row's FindAll claims under the same reserved_micros contract.
    return found


def headroom(value, found, limit_micros=None):
    """Remaining allowance and largest admissible start under ``limit_micros`` (default the
    frozen limit); None amounts when a debit is unknowable."""
    limit = value["limit_micros"] if limit_micros is None else min(limit_micros, value["limit_micros"])
    reserved = 0
    for claim in found:
        amount = claim.get("reserved_micros") if isinstance(claim, dict) else None
        if type(amount) is not int or amount <= 0:
            return {"reserved_micros": None, "remaining_micros": None, "max_start_micros": None}
        reserved += amount
    remaining = max(0, limit - reserved)
    return {"reserved_micros": reserved, "remaining_micros": remaining,
            "max_start_micros": min(remaining, per_start_max(limit))}


def problem(value, found, cap_micros, now, *, control, source="exa"):
    """None when a paid start of ``cap_micros`` is admitted; otherwise a named refusal.

    Remaining is the effective limit (the frozen grant, lowered by a smaller
    current direction) minus every existing durable claim's reserved cap for this
    run. This replaces the v1 all-in snapshot predicate.
    """
    code = standing(value, control, now, source=source)
    if code:
        return code
    limit = effective_limit(value, control)
    room = headroom(value, found, limit)
    if room["remaining_micros"] is None:
        return "paid_expansion_claims_unverified"
    if type(cap_micros) is not int or cap_micros <= 0:
        return "paid_expansion_cap_invalid"
    if cap_micros > room["remaining_micros"]:
        return "paid_expansion_cap_exceeds_remaining"
    if cap_micros > per_start_max(limit):
        return "paid_expansion_cap_exceeds_per_start_maximum"
    return None


def diagnostic(row, control=None, *, now=None):
    """Read-only, allowlisted explanation of the frozen grant; never manufactures cost."""
    value = row.get("paid_expansion_grant") if isinstance(row, dict) else None
    result = {"schema_version": STATUS, "grant_state": value.get("state") if isinstance(value, dict) else "missing",
              "reasons": [], "remaining_micros": None, "max_start_micros": None,
              "claim_created": False, "provider_started": False}
    code = granted(value)
    if code:
        result["reasons"].append(code)
        return result
    result.update({key: value[key] for key in ("version", "direction_sha256", "limit_micros", "per_start_max_micros", "valid_until")})
    limit = value["limit_micros"] if control is None else effective_limit(value, control)
    result.update(headroom(value, claims(row), limit), effective_limit_micros=limit)
    if control is not None:
        code = standing(value, control, now or datetime.now(timezone.utc))
        if code:
            result["reasons"].append(code)
            result["max_start_micros"] = 0  # No start is admissible while a live fence refuses.
    return result
