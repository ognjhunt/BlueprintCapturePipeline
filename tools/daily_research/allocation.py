"""Owner-directed paid research sources with immutable authority and usage records.

Direction and grant dollar fields remain readable for historical signed records.
They do not limit admission. Live source, expiry, brake and release fences remain;
durable claims retain provider costs, estimates and unknown exposure without replay.
"""
import hashlib
import re
from datetime import datetime, timedelta, timezone

from tools.daily_research.runner import AGENT, MAX_ADAPTIVE_RUNTIME_SECONDS, PROJECT, canonical

DIRECTION = "blueprint.research-paid-expansion-direction.v1"
GRANT = "blueprint.research-paid-expansion-grant.v1"
STATUS = "blueprint.research-paid-expansion-status.v1"

FLOOR_MICROS = 1_000_000  # $1.00

AMOUNT = re.compile(r"[1-9][0-9]*(\.[0-9]{2})?")  # ASCII digits only, like the bridge
SOURCES = ("exa", "findall")  # Mirrored by the bridge's PAID_SOURCES.
# FindAll claims live in the owner journal field parallel_findall_owner.SUBMISSIONS_FIELD.
FINDALL_FIELD = "parallel_findall_submissions"
FINDALL_AMOUNT = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]{1,2})?")  # USD, at most cents, like the bridge
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
    """Exact integer microdollars for an owner amount such as "10.00" as historical reporting data, else None."""
    if not isinstance(amount, str) or not AMOUNT.fullmatch(amount):
        return None
    whole, _, cents = amount.partition(".")
    value = int(whole) * 1_000_000 + int(cents or "0") * 10_000
    return value if 0 < value <= 2**53 - 1 else None


def findall_micros(amount):
    """Exact integer microdollars of a FindAll ``maximum_cost_usd`` such as "2.5" (above $0), else None."""
    if not isinstance(amount, str) or not FINDALL_AMOUNT.fullmatch(amount):
        return None
    whole, _, cents = amount.partition(".")
    value = int(whole) * 1_000_000 + int(cents.ljust(2, "0")) * 10_000
    return value if 0 < value <= 2**53 - 1 else None


def usd(value):
    """Two-decimal display of an owner amount in microdollars."""
    return f"{value // 1_000_000}.{value % 1_000_000 // 10_000:02d}"


def per_start_max(limit_micros):
    """Reproduce legacy signed metadata only; this value never gates new work."""
    return None if limit_micros is None else min(max(limit_micros // 2, FLOOR_MICROS), 50_000_000)


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
        if direction["per_run_limit_usd"] is not None and micros(direction["per_run_limit_usd"]) is None:
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
            if started.tzinfo is None or type(seconds) is not int or not 0 < seconds <= MAX_ADAPTIVE_RUNTIME_SECONDS:
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
                or limit is not None and (type(limit) is not int or limit <= 0)
                or start != per_start_max(limit)
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
    """No Blueprint dollar limit; retained legacy amounts are reporting data."""
    return


def claims(row):
    """Retain this run's durable claims as accounting data without admission ceilings.

    Historical Exa caps and FindAll exact-request estimates remain recorded in
    every state. New uncapped Exa starts and malformed claims stay unknown.
    Integer amounts must remain exactly representable in the existing JS bridge.
    """
    found = []
    exa = row.get("exa_expansion") if isinstance(row, dict) else None
    if exa:
        found.append({"source": "exa", "intent_sha256": exa.get("intent_sha256"), "reserved_micros": exa.get("cap_micros")})
    findall = row.get(FINDALL_FIELD) if isinstance(row, dict) else None
    if findall:
        entries = sorted(findall.items()) if isinstance(findall, dict) else [(None, findall)]
        for key, entry in entries:
            prepared = entry.get("prepared") if isinstance(entry, dict) else None
            amount = prepared.get("maximum_cost_usd") if isinstance(prepared, dict) else None
            found.append({"source": "findall", "operation_sha256": key, "reserved_micros": findall_micros(amount)})
    return found


def headroom(value, found, limit_micros=None):
    """Report retained reservations, including unknowns, without a dollar ceiling."""
    reserved = 0
    for claim in found:
        amount = claim.get("reserved_micros") if isinstance(claim, dict) else None
        if type(amount) is not int or amount <= 0:
            reserved = None
            break
        reserved += amount
    return {"reserved_micros": reserved, "remaining_micros": None, "max_start_micros": None}


def problem(value, found, cap_micros, now, *, control, source="exa"):
    """None when a paid start of ``cap_micros`` is admitted; otherwise a named refusal.

    Source authority, brake, expiry and release identity are still required.
    Historical dollar metadata and retained exposure never become a new ceiling.
    """
    code = standing(value, control, now, source=source)
    if code:
        return code
    if cap_micros is not None and (type(cap_micros) is not int or cap_micros <= 0):
        return "paid_expansion_cap_invalid"

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
    limit = effective_limit(value, control)
    result.update(headroom(value, claims(row), limit), effective_limit_micros=limit)
    if control is not None:
        code = standing(value, control, now or datetime.now(timezone.utc))
        if code:
            result["reasons"].append(code)
            result["max_start_micros"] = 0  # No start is admissible while a live fence refuses.
    return result
