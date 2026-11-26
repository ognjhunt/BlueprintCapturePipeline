"""The build plan's dollar budget: each extra paid object reserves its own share of the job's $40.

Owner decision 2026-11-23: the job's one cap stays $40 (``website_job_spend_cap``, mirrored byte for byte in the
WebApp), and the simulation may spend whatever the scan leaves under it. The scan (Atlas, SAM, Gemini, the planner)
spends first. The first object the build plan builds, its top-ranked one, is built under the scene intake's allowance
exactly as today: the WebApp holds that allowance in full on the job's ledger, and the Pipeline holds each scene
attempt to it (``task_evaluation_scene_intake.scene_spend_budget``). Then, one object at a time in the plan's order:

- an object compile builds with ARTISAN (at its frozen per-object cap: $12 today,
  ``task_object_authoring_backend.new_authoring_budget``) reserves that cap as its own row on the job's WebApp ledger,
  through the existing preparation reservation (``website_task_context.reserve_website_preparation_spend``: the
  account-limit precheck, then WebApp ``reserveWebsitePreparationSpend``, whose ``checkWebsiteJobCeiling`` counts
  everything the job holds or has settled on every ledger, then the paid chokepoint). The row is an ordinary OpenAI
  preparation row, ARTISAN's one OpenAI gate whichever provider writes its text: no new money channel;
- an object compile builds from a library model (Google Scanned Objects, SimReady) or a parametric shape (a box,
  tote, bin, tray or can) costs nothing and reserves nothing.

Which it is, is read from compile's own result for the object (``configurations``: its stage-3 configuration,
``website_supplied_mesh.supplied_mesh_configuration``), never from the plan's ``sourcing`` (stress round 2,
2026-11-24): the library comes first whatever the plan said, so an object the plan calls ``generate`` that the
library builds for free reserves nothing, and a "free" parametric shape that does not fit the object's bounds, which
compile hands to ARTISAN, is reserved like any paid object. An extra object is authored with ARTISAN only under its
own admitted reservation (``authoring_allocation`` refuses any other entry): one compile has not decided cannot be
budgeted at all.

Raising the scene allowance up front instead was tried and reverted: the scan can spend past its nominal figure (a
$20.50 Atlas hold) and the WebApp holds the intake's whole allowance up front, so a bigger allowance was refused at the
cap and built nothing.

When the WebApp refuses (``website_scene_preparation_budget_exhausted``), that object and every paid object ranked
below it are kept as filmed, each by name, under the machine reason ``spending_cap_reached``. The WebApp sends that one
code for both of the job's limits on paid work, the $40 dollar cap and the native ledger's paid-request count
(``preparationRequestLimit``), and does not say which refused. So the records keep the code exactly as received with
the two limits it may mean (``REFUSAL_CODE_MEANS_ONE_OF``), never which one, and the company README's plain notes say
what is true of either ("Kept as filmed: <name>, the job reached its limit on paid work.", ``website_build_plan_notes``);
the handoff's build plan disclosure carries the same (``kept_disclosure``). The job goes on without them; a refusal
never stops it, and nothing spends past the cap. A refusal is final: money a later settlement frees never brings a
refused object back. Any other refusal (the account's monthly limit, privacy, rights, an outage) is raised exactly as
the reservation raises it, so the job's own retry asks again.

Each object's reservation is written once (``reservations/<binding>.json``), keyed by the job and the object (its
target and name), never by the plan: a retry, a resume or a new plan (a new Atlas result, a correction) reuses it and
never reserves twice, so a superseded plan's $12 is never held beside the new plan's for the same object; the WebApp
keys its row by the same binding. It is written as soon as the WebApp admits it, before the local paid recheck, so a
run that stops after the row was held still finds it. An admitted reservation stays admitted; before it counts again,
the capture's current source is read (a site that withdrew is never authored for under an earlier hold) and the
local limits are asked again (``_recheck_retained``). A resumed authoring run spends only what its reservation's
frozen cap leaves (``authoring_allocation``: the rule the first object's successor uses). When the object's authoring
has ended, its row settles at what it cost (``settle_object``: the existing OpenAI ``preparation-settlement``), never
above the reservation; until then it holds its whole quote.

Several objects per job, stage 6 (``reserve_at_removal``): a paid further object must be taken out of the scan before
the room is reconstructed, which is before the scene intake's allowance is held, and only under its own admitted
reservation, or else a refusal could not leave it in the scan. So the Atlas job reserves each one as it seals its
job-object decision (``website_atlas_pipeline._build_plan``), in rank order, each reservation naming what the scored
object still needs from the job's cap (``held_ahead_usd``: its scene intake's allowance, from the grant the capture runs
under, and the scan's share left, from the Atlas configuration), which the WebApp counts beside the $12 and never holds.
That is the guarantee the design's "once the scene intake is held" gives, given before the intake: a further object
never takes the scored object's money. The figure is frozen once per job (``HOLD_AHEAD_RECORD``), so a run asking again
asks with the same command. A WebApp that cannot count it refuses the field, and then nothing is reserved: that object
and each paid one ranked below it stay in the scan as filmed, under the reason that claims no limit refused them. An
admitted one is taken out of the scan with the job's object and authored with ARTISAN at compile, one at a time in rank
order, under its own row (``website_further_artisan``); a refused one, and each paid one ranked below it, stays in the
scan as filmed under ``spending_cap_reached``. None is reserved unless this host can author one
(``website_further_artisan.installed``). Ranks are the intake request's (``website_build_plan.built_rank``): the scored
object is 1, each further object 2 to 8. An extra object's ARTISAN usage settles through ``settle_object`` only; a scene
attempt's meter never records it.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_build_plan_notes import (BUDGET_RECORD, BUDGET_SCHEMA, KEPT_AT_CAP as KEPT, REFUSAL_CODE_MEANS_ONE_OF,
                                       budget_root)
from .website_job_spend_cap import SPENDING_CAP_REACHED

RESERVATION_SCHEMA = "website_build_object_reservation.v1"
#: r2 (stress round 2): a reservation is keyed by the job and the object, no longer by the sealed plan.
REVISION = "website_build_budget.r2"
#: The plan's sourcing for an object that costs nothing to build: a licensed library asset or a parametric shape.
FREE_SOURCING = frozenset({"library", "parametric"})
#: The plan's sourcing for an object that needs its own generated model (ARTISAN): the paid one.
PAID_SOURCING = "generate"
#: Compile's builders that generate a model and are paid (``website_half_preparation.GENERATING_BACKENDS``).
GENERATING_BACKENDS = frozenset({"artisan_v1", "astra_cad_blender_v1"})
#: An object's status: the first (the scene intake's allowance, as today), free, its own reservation admitted, or kept
#: as filmed at a limit on the job's paid work (``KEPT``). A reservation record is admitted or ``REFUSED`` there.
FIRST, FREE, ADMITTED, REFUSED = "first_object", "free", "admitted", "refused_at_cap"
#: The WebApp row: ARTISAN's one OpenAI gate (``website_native_submission``: one reservation and one cap).
RESOURCE_CLASS, PROVIDER = "openai_api_candidate", "openai"
#: An object not reserved, kept in the scan as filmed under a reason that claims no limit refused it: the WebApp could
#: not count the hold ahead (``HOLD_UNCONFIRMED``), or already holds the object's row with no record of it here.
UNRESERVED = "not_reserved"
HOLD_UNCONFIRMED = "website_build_budget_hold_ahead_unconfirmed"
HELD_WITHOUT_RECORD = "website_build_budget_reservation_held_without_record"
#: What the scored object still needs from the job's cap, frozen once per job beside the reservations.
HOLD_AHEAD_SCHEMA, HOLD_AHEAD_RECORD = "website_build_budget_hold_ahead.v1", "hold_ahead.json"
OBJECTS_INVALID = "website_build_budget_objects_invalid"
RECORD_INVALID = "website_build_budget_record_invalid"
SETTLEMENT_INVALID = "website_build_budget_settlement_invalid"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def reservation_binding(*, task_context: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """The one reservation an object may hold: its job and the object (target and name). A retry, a resume, a later
    task context and a new plan name the same one."""
    return {"kind": "website_build_extra_object", "revision": REVISION, "request_id": str(task_context["request_id"]),
            "target_id": str(row["target_id"]), "name": str(row["name"])}


def route_of(configuration: Any) -> str:
    """What compile builds an object with, by its stage-3 configuration (``website_supplied_mesh``): ``library`` or
    ``parametric`` (supplied geometry, free) or ``generate`` (a paid generating builder). Raises ``OBJECTS_INVALID``
    for anything else: an object compile has not decided is never budgeted, so never authored as free."""
    if isinstance(configuration, Mapping):
        source = configuration.get("authoring_source")
        if source in FREE_SOURCING and "authoring_backend" not in configuration:
            return str(source)
        if configuration.get("authoring_backend") in GENERATING_BACKENDS:
            return PAID_SOURCING
    raise ValueError(OBJECTS_INVALID)


def _path(root: Path, digest: str, suffix: str = ".json") -> Path:
    return Path(root) / "reservations" / (digest.removeprefix("sha256:") + suffix)


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    """One run of a job's budget at a time: a concurrent run waits, then reuses what the first wrote."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _read_reservation(root: Path, digest: str) -> dict[str, Any] | None:
    """The object's written reservation, or None when none was written. One that does not seal itself raises: money
    fails closed, and asking again could not tell whether the WebApp already holds its row."""
    path = _path(root, digest)
    if not os.path.lexists(path):
        return None
    try:
        if path.is_symlink():
            raise ValueError(RECORD_INVALID)
        record = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(RECORD_INVALID) from exc
    if (not isinstance(record, dict) or record.get("schema_version") != RESERVATION_SCHEMA
            or record.get("binding_digest") != digest or canonical_digest(record.get("binding") or {}) != digest
            or record.get("status") not in (ADMITTED, REFUSED)
            or (record["status"] == ADMITTED) != isinstance(record.get("admission"), Mapping)
            or (record["status"] == REFUSED) != isinstance(record.get("refusal_code"), str)
            or record.get("digest") != canonical_digest(record, digest_field="digest")):
        raise ValueError(RECORD_INVALID)
    return record


def _write_once(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Seal ``record`` and create ``path`` with it, whole or not at all. A written reservation is never replaced."""
    sealed = {**record, "digest": canonical_digest(record, digest_field="digest")}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(sealed, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return sealed


def _at_cap(error: BaseException) -> bool:
    """Whether ``error`` is the website's refusal at a limit on the job's paid work, as every preparation lane reads it
    (``spending_cap_code``): its one code covers the dollar cap and the paid-request count alike."""
    from .website_scene_handoff import spending_cap_code
    return isinstance(error, ValueError) and spending_cap_code(str(error))


def _refusal(code: str) -> dict[str, Any]:
    """The refusal as a record keeps it: the website's code exactly as received and the limits it may mean."""
    return {"refusal_code": code, "refusal_code_means_one_of": list(REFUSAL_CODE_MEANS_ONE_OF)}


def _reserve(*, root: Path, digest: str, value: Mapping[str, Any], task_context: Mapping[str, Any],
             reserve: Callable[..., Any] | None, now: Callable[[], float],
             held_ahead_usd: float | None = None) -> dict[str, Any]:
    """Ask the job's ledger for the object's frozen ARTISAN cap: the admitted reservation, or the refusal at a limit on
    the job's paid work, kept with the website's code as received (the code after the transport's
    ``website_control_preparation-spend_http_409:``). Anything else is raised and writes nothing, unless the WebApp
    already admitted the row: that stays written. With ``held_ahead_usd`` the request names what the scored object
    still needs; a WebApp that does not parse it (``_http_400``: it wrote nothing) raises ``HOLD_UNCONFIRMED``."""
    from .task_object_authoring_backend import frozen_authoring_cap, new_authoring_budget
    if reserve is None:
        from .website_task_context import reserve_website_preparation_spend as reserve
    budget = new_authoring_budget()
    cap = frozen_authoring_cap({"authoring_budget": budget})
    path = _path(root, digest)
    base = {"schema_version": RESERVATION_SCHEMA, "binding_digest": digest, "binding": dict(value),
            "authoring_budget": budget, "maximum_cost_usd": cap}

    def retain(admission: Mapping[str, Any]) -> None:
        _write_once(path, {**base, "status": ADMITTED, "admission": dict(admission), "recorded_at_epoch": now()})

    try:
        reserve(task_context=task_context, binding_digest=digest, maximum_cost_usd=cap, request_count=1,
                resource_class=RESOURCE_CLASS, provider=PROVIDER, retain_admission=retain,
                **({"held_ahead_usd": held_ahead_usd} if held_ahead_usd is not None else {}))
    except Exception as exc:
        if (held_ahead_usd is not None and not os.path.lexists(path) and isinstance(exc, ValueError)
                and "preparation-spend_http_400:" in str(exc)):
            raise ValueError(HOLD_UNCONFIRMED) from exc
        if os.path.lexists(path) or not _at_cap(exc):
            raise
        return _write_once(path, {**base, "status": REFUSED, **_refusal(str(exc).rsplit(":", 1)[-1]),
                                  "recorded_at_epoch": now()})
    record = _read_reservation(root, digest)
    if record is None or record["status"] != ADMITTED:
        raise ValueError(RECORD_INVALID)
    return record


def _recheck_retained(record: Mapping[str, Any], digest: str, task_context: Mapping[str, Any]) -> None:
    """A reservation written by an earlier run is a hold, not a grant: before it counts as admitted again, the
    capture's current source is read (``require_current_website_source``, as ``retained_gemini_call`` does before it
    sends anything under an earlier hold: a site that withdrew raises its privacy stop, stress round 2), then the same
    local checks a new reservation passes run again (the account's month, the resource's monthly budget and its class,
    as ``require_paid_resource_budget`` reads them), and its admission must still be this binding's, admitted, with no
    blockers. A refusal is raised as the first run's was; the hold stays written for the next run, so the website is
    never asked twice (Codex review of #3713, 2026-11-24: a hold kept when the local recheck refused must not turn
    that refusal into an admission on the retry). No grant is issued and nothing is metered here: dispatch still
    takes its own grant."""
    from .paid_resource_admission import require_paid_resource_budget
    from .website_task_context import require_current_website_source
    require_current_website_source(task_context)
    held = record.get("admission")
    if (not isinstance(held, Mapping) or held.get("allocation_binding_digest") != digest
            or held.get("status") != "admitted" or held.get("schema_version") != "paid_lane_admission.v1"
            or held.get("blockers") not in ([], None) or held.get("resource_class") != RESOURCE_CLASS
            or held.get("provider") != PROVIDER):
        raise ValueError(RECORD_INVALID)
    # The account's month must still have room for the object's frozen cap, as a new reservation's precheck asks.
    require_paid_resource_budget(resource_class=RESOURCE_CLASS, maximum_cost_usd=record.get("maximum_cost_usd"),
                                 allocation_binding_digest=digest)


def _ranks(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    """Each object's rank as the intake request ranks it (``website_build_plan.built_rank``): its own ``rank`` where
    the caller names one, else its position plus one. The first object is 1, the scored one; each later one is above
    the one before it and at most 8. Anything else raises ``OBJECTS_INVALID``."""
    from .task_evaluation_scene_intent_contracts import ADDITIONAL_OBJECT_RANKS
    from .website_build_plan import built_rank
    ranks = [row.get("rank", built_rank(position)) for position, row in enumerate(rows)]
    if (any(type(rank) is not int for rank in ranks) or ranks[0] != 1
            or any(later <= earlier for earlier, later in zip(ranks, ranks[1:]))
            or ranks[-1] > ADDITIONAL_OBJECT_RANKS[1]):
        raise ValueError(OBJECTS_INVALID)
    return ranks


def admit_extra_objects(objects: Sequence[Mapping[str, Any]], *, task_context: Mapping[str, Any],
                        build_plan_digest: str, root: Path, configurations: Mapping[str, Any] | None = None,
                        routes: Mapping[str, str] | None = None, held_ahead_usd: float | None = None,
                        reserve: Callable[..., Any] | None = None,
                        now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Admit or refuse each extra paid object the build will build; returns the job's budget record, also written to
    ``root`` (``budget_root(capture_root)``, where the readers look).

    ``objects`` are the build plan's objects the build will build, in the plan's order, each as ``validate_plan`` kept
    it (``target_id``, ``name``, ``sourcing``, and its ``rank`` where it is not its position plus one: ``_ranks``);
    ``build_plan_digest`` is the sealed plan's record digest, kept on the record. What each extra object is built with
    decides what it costs, never the plan's ``sourcing``: ``configurations`` is compile's stage-3 configuration of each,
    by its ``target_id`` (``route_of``), and ``routes`` (in its place) the route decided before the room is
    reconstructed (``reserve_at_removal``): ``library``, ``parametric`` or ``generate``. The first is the first object,
    under the scene intake's allowance as today: nothing is reserved for it here. Each later one is free (compile
    builds it from a library model or a parametric shape) or, built with ARTISAN, reserved on the job's ledger
    (``reserve``, by default ``reserve_website_preparation_spend``), unless a paid object ranked above it was refused
    at a limit on the job's paid work: then it is kept as filmed without asking, under that refusal's code. A refusal
    there keeps that object as filmed too, and is final. Any other failure is raised as the reservation raised it,
    after the objects before it were recorded in their reservations. A rerun, and a new plan, reuse every written
    reservation and ask only for paid objects never asked. Call it only once the scene intake's allowance is held, or
    with ``held_ahead_usd``: what the scored object still needs, which the WebApp counts beside each quote and never
    holds. Then a WebApp that cannot count it (``HOLD_UNCONFIRMED``) leaves that object and each paid one below it
    unreserved (``UNRESERVED``), and one whose row the WebApp holds with no record here (``HELD_WITHOUT_RECORD``) is
    left unreserved alone: each stays in the scan as filmed, under a reason that claims no limit refused it.
    """
    rows = [dict(row) for row in objects if isinstance(row, Mapping)]
    try:
        identities = [(str(row["target_id"]), str(row["name"])) for row in rows]
    except KeyError:
        raise ValueError(OBJECTS_INVALID) from None
    if (not rows or len(rows) != len(objects) or len(set(identities)) != len(identities)
            or any(row.get("sourcing") not in FREE_SOURCING | {PAID_SOURCING} for row in rows)
            or not isinstance(build_plan_digest, str) or not _DIGEST.fullmatch(build_plan_digest)
            or (configurations is None) == (routes is None)
            or not isinstance(configurations if routes is None else routes, Mapping)
            or held_ahead_usd is not None and (isinstance(held_ahead_usd, bool)
                                               or not isinstance(held_ahead_usd, (int, float))
                                               or not math.isfinite(held_ahead_usd) or held_ahead_usd <= 0)):
        raise ValueError(OBJECTS_INVALID)
    ranks = _ranks(rows)
    # What compile builds each extra object with, decided before anything is asked: library, parametric, or ARTISAN.
    if routes is not None:
        found = {str(row["target_id"]): routes.get(str(row["target_id"])) for row in rows[1:]}
        if any(route not in FREE_SOURCING | {PAID_SOURCING} for route in found.values()):
            raise ValueError(OBJECTS_INVALID)
    else:
        found = {str(row["target_id"]): route_of(configurations.get(str(row["target_id"]))) for row in rows[1:]}
    routes = found
    root = Path(root)

    def named(rank: int, row: Mapping[str, Any]) -> dict[str, Any]:
        return {"rank": rank, "target_id": str(row["target_id"]), "name": str(row["name"]), "sourcing": row["sourcing"],
                **({"route": routes[str(row["target_id"])]} if rank > 1 else {})}

    entries: list[dict[str, Any]] = [{**named(ranks[0], rows[0]), "status": FIRST}]
    with _locked(root):
        capped = None  # the refusal that keeps every later paid object as filmed, once there is one
        unconfirmed = False  # the WebApp could not count the hold ahead: no later paid object is asked
        for rank, row in zip(ranks[1:], rows[1:]):
            if routes[str(row["target_id"])] in FREE_SOURCING:
                entries.append({**named(rank, row), "status": FREE})
                continue
            value = reservation_binding(task_context=task_context, row=row)
            digest = canonical_digest(value)
            record = _read_reservation(root, digest)
            if record is not None and record["status"] == ADMITTED:
                _recheck_retained(record, digest, task_context)
            reason = HOLD_UNCONFIRMED if unconfirmed and record is None and capped is None else None
            if record is None and capped is None and not unconfirmed:
                try:
                    record = _reserve(root=root, digest=digest, value=value, task_context=task_context,
                                      reserve=reserve, now=now, held_ahead_usd=held_ahead_usd)
                except Exception as exc:
                    from .paid_resource_admission import PaidResourceAlreadyReserved
                    if held_ahead_usd is None or not (str(exc) == HOLD_UNCONFIRMED
                                                      or isinstance(exc, PaidResourceAlreadyReserved)):
                        raise
                    unconfirmed = str(exc) == HOLD_UNCONFIRMED
                    reason = HOLD_UNCONFIRMED if unconfirmed else HELD_WITHOUT_RECORD
            if reason is not None:
                entries.append({**named(rank, row), "status": UNRESERVED, "reason": reason})
                continue
            if record is not None and record["status"] == ADMITTED:
                entries.append({**named(rank, row), "status": ADMITTED, "binding_digest": digest,
                                "maximum_cost_usd": record["maximum_cost_usd"],
                                "authoring_budget": record["authoring_budget"]})
                continue
            if record is not None:
                capped = record["refusal_code"]
            entries.append({**named(rank, row), "status": KEPT, "reason": SPENDING_CAP_REACHED,
                            "refused": record is not None, **({"binding_digest": digest} if record else {}),
                            **_refusal(record["refusal_code"] if record else capped)})
        kept = [row["name"] for row in entries if row["status"] == KEPT]
        budget = {"schema_version": BUDGET_SCHEMA, "revision": REVISION, "request_id": str(task_context["request_id"]),
                  "build_plan_digest": build_plan_digest, "objects": entries, "kept_at_cap": kept,
                  **({"flags": [SPENDING_CAP_REACHED]} if kept else {}),
                  **({"held_ahead_usd": held_ahead_usd} if held_ahead_usd is not None else {}),
                  "claim_boundary": "Reservations inside the job's limits on paid work, in the plan's order; a refusal "
                                    "names the website's code, which covers the dollar cap and the paid-request count "
                                    "alike, never which one. No object is built by this record."}
        budget["digest"] = canonical_digest(budget, digest_field="digest")
        write_json(root / BUDGET_RECORD, budget)
    return budget


#: Why a further object needs ARTISAN before the room is reconstructed (``website_extra_objects``) that one rigid
#: generated model answers: no free shape for it, or a can that may miss its size. One that opens never is one: a further
#: object is composed as one rigid body, so it stays in the scan as filmed, as before.
PAID_AT_REMOVAL = ("website_extra_object_no_free_source", "website_extra_object_cylinder_can_miss_size_tolerance")


def _positive(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def scored_object_needs(*, capture_root: Path, task_context: Mapping[str, Any],
                        source_digest: str | None) -> dict[str, Any] | None:
    """What the scored object still needs from the job's cap before the room is reconstructed: its scene intake's
    allowance, from the grant the capture runs under (``website_task_context.RETAINED_GRANT_FILE``: the figure its intake
    request names and the WebApp holds in full), and the scan's share left, from the Atlas configuration (what the
    Pipeline still plans Atlas purchases within, never more than the job has left), or None when either cannot be read:
    then nothing is reserved. Reads only; creates no grant."""
    from .website_task_context import RETAINED_GRANT_FILE, load_atlas_configuration
    if not isinstance(source_digest, str) or not _DIGEST.fullmatch(source_digest):
        return None
    try:
        grant = json.loads((Path(capture_root) / RETAINED_GRANT_FILE).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(grant, Mapping) or grant.get("schema_version") != "website_scene_sponsorship.v1"
            or grant.get("authority_digest") != canonical_digest(grant, digest_field="authority_digest")
            or any(grant.get(key) != task_context.get(key) for key in ("request_id", "scene_id", "capture_id"))
            or not _positive(grant.get("max_total_spend_usd"))):
        return None
    left = load_atlas_configuration(task_context, source_digest, create=False).get("remaining_cost_usd")
    if isinstance(left, bool) or not isinstance(left, (int, float)) or not math.isfinite(left) or left < 0:
        return None
    allowance = float(grant["max_total_spend_usd"])
    return {"scene_allowance_usd": allowance, "scan_share_left_usd": float(left),
            "held_ahead_usd": round(allowance + float(left), 6)}


def _hold_ahead(root: Path, *, capture_root: Path, task_context: Mapping[str, Any], source_digest: str | None,
                now: Callable[[], float]) -> float | None:
    """The job's hold ahead, read once (``scored_object_needs``) and frozen beside its reservations, so each run asks
    with the same command: the WebApp then answers a row it holds with that row, never a conflict. None when it cannot
    be read; one written that does not seal itself raises (money fails closed)."""
    path = Path(root) / HOLD_AHEAD_RECORD
    if os.path.lexists(path):
        try:
            record = None if path.is_symlink() else json.loads(path.read_text())
        except (OSError, ValueError):
            record = None
        if (not isinstance(record, Mapping) or record.get("schema_version") != HOLD_AHEAD_SCHEMA
                or record.get("request_id") != str(task_context["request_id"])
                or not _positive(record.get("held_ahead_usd"))
                or record.get("digest") != canonical_digest(record, digest_field="digest")):
            raise ValueError(RECORD_INVALID)
        return float(record["held_ahead_usd"])
    needs = scored_object_needs(capture_root=capture_root, task_context=task_context, source_digest=source_digest)
    if needs is None:
        return None
    _write_once(path, {"schema_version": HOLD_AHEAD_SCHEMA, "request_id": str(task_context["request_id"]), **needs,
                       "recorded_at_epoch": now(),
                       "claim_boundary": "What the job's scored object may still need from its cap, counted beside "
                                         "each further object's reservation and never held; not a cost."})
    return needs["held_ahead_usd"]


def reserve_at_removal(further: Sequence[Mapping[str, Any]], *, built: Sequence[Mapping[str, Any]],
                       plan: Mapping[str, Any], task_context: Mapping[str, Any], atlas_root: Path,
                       source_digest: str | None, now: Callable[[], float] = time.time) -> list[dict[str, Any]]:
    """The further objects of a decision being sealed (``website_build_plan.further_removal``), with each paid one
    (``PAID_AT_REMOVAL``) reserved before the room is reconstructed, in rank order (``admit_extra_objects``, with the
    hold ahead): admitted, it is taken out of the scan with the job's object (``removed``, its ``paid`` reservation
    named) to be authored with ARTISAN at compile; refused at a limit on the job's paid work, it and each paid one ranked
    below it stay in the scan as filmed under ``spending_cap_reached``; not reserved (``UNRESERVED``), it stays as it
    was. ``built`` is ``built_objects``' list, the scored object first; ``plan`` the plan record; ``atlas_root`` the
    Atlas job's ``pipeline/atlas``. The rows come back unchanged, and nothing is asked or written, without a paid one,
    when this host cannot author one (``website_further_artisan.installed``) or when the hold ahead cannot be read; a
    privacy or rights stop reading it is raised. Any other refusal is raised as the reservation raised it (a wait)."""
    from .hard_stops import wall_kind
    from .job_retry_policy import PRIVACY_RIGHTS
    from .website_further_artisan import installed
    from .website_further_objects import NEEDS_ARTISAN

    def paid(row: Mapping[str, Any]) -> bool:
        head, _, detail = str(row.get("kept_in_scan_reason") or "").partition(":")
        return row.get("removed") is False and head == NEEDS_ARTISAN and detail in PAID_AT_REMOVAL
    rows = [dict(row) for row in further]
    if not any(paid(row) for row in rows) or not built or not installed():
        return rows
    capture_root = Path(atlas_root).parent.parent
    root = budget_root(capture_root)
    try:
        with _locked(root):
            held = _hold_ahead(root, capture_root=capture_root, task_context=task_context,
                               source_digest=source_digest, now=now)
    except Exception as exc:  # noqa: BLE001 - a hold ahead that cannot be read reserves nothing, never a stop
        if wall_kind(exc) == PRIVACY_RIGHTS or str(exc) == RECORD_INVALID:
            raise
        held = None
    if held is None:
        return rows
    by_target = {str(row["target_id"]): row for row in built}
    taken = [row for row in rows if row.get("removed") is True or paid(row)]
    objects = [built[0], *({**by_target[row["target_id"]], "rank": row["rank"]} for row in taken)]
    routes = {row["target_id"]: PAID_SOURCING if paid(row) else "parametric" for row in taken}
    record = admit_extra_objects(objects, task_context=task_context, build_plan_digest=plan["digest"], root=root,
                                 routes=routes, held_ahead_usd=held, now=now)
    outcome = {row["target_id"]: row for row in record["objects"][1:]}
    found = []
    for row in rows:
        entry = outcome.get(row["target_id"]) if paid(row) else None
        if entry is None or entry["status"] == UNRESERVED:
            found.append(row)
            continue
        base = {key: value for key, value in row.items() if key != "kept_in_scan_reason"}
        found.append({**base, "removed": True, "paid": {"binding_digest": entry["binding_digest"],
                                                       "maximum_cost_usd": entry["maximum_cost_usd"]}}
                     if entry["status"] == ADMITTED else
                     {**base, "removed": False, "kept_in_scan_reason": SPENDING_CAP_REACHED})
    return found


def admitted_reservation(root: Path, binding_digest: str) -> dict[str, Any] | None:
    """The object's written reservation when it was admitted, else None; one that does not seal itself raises."""
    record = _read_reservation(Path(root), binding_digest)
    return record if record is not None and record["status"] == ADMITTED else None


def settlement(root: Path, binding_digest: str) -> dict[str, Any] | None:
    """The object's kept settlement receipt (``settle_object``), or None while its row holds its whole quote."""
    kept = _path(Path(root), binding_digest, ".settlement.json")
    if not kept.is_file():
        return None
    receipt = json.loads(kept.read_text())
    if receipt.get("allocation_binding_digest") != binding_digest or receipt.get("status") != "settled":
        raise ValueError(SETTLEMENT_INVALID)
    return receipt


def authoring_allocation(entry: Mapping[str, Any], carried: Mapping[str, Any] | None) -> float:
    """What one authoring run of an admitted object may spend: its reservation's frozen cap less what earlier runs
    under it spent (``carried``, as ``factory_carried_spend`` reports it), by the rule the first object's successor
    uses (``resumed_authoring_allocation``). A resume never reserves again; it spends what its reservation left. Any
    entry but an admitted one (a free object, one kept as filmed) raises: an extra object is never authored with
    ARTISAN without its own reservation."""
    from .task_object_authoring_backend import frozen_authoring_cap, resumed_authoring_allocation
    if entry.get("status") != ADMITTED:
        raise ValueError(OBJECTS_INVALID)
    return resumed_authoring_allocation(frozen_authoring_cap({"authoring_budget": entry["authoring_budget"]}), carried)


def settle_object(*, binding_digest: str, task_context: Mapping[str, Any], root: Path, cost_usd: float, usage: Any,
                  request: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Settle an admitted object's row at what its authoring cost, once its authoring has ended.

    ``cost_usd`` is the object's whole authoring cost across every run under its reservation (ARTISAN's meter) and
    ``usage`` the metered usage it rests on. The row then counts that cost instead of its quote, never more than the
    reservation (the WebApp refuses more), through the existing OpenAI ``preparation-settlement`` (``request``, by
    default ``preparation_webapp_request``). Call it only once every call of the object's authoring has its usage:
    until then the row keeps its whole quote, which over-counts and never under-counts. Settled once: a rerun returns
    the kept receipt and asks nothing.
    """
    root = Path(root)
    with _locked(root):
        record = _read_reservation(root, binding_digest)
        if record is None or record["status"] != ADMITTED:
            raise ValueError(SETTLEMENT_INVALID)
        kept = _path(root, binding_digest, ".settlement.json")
        if kept.is_file():
            receipt = json.loads(kept.read_text())
            if receipt.get("allocation_binding_digest") != binding_digest or receipt.get("status") != "settled":
                raise ValueError(SETTLEMENT_INVALID)
            return receipt
        if isinstance(cost_usd, bool) or not isinstance(cost_usd, (int, float)) or not math.isfinite(cost_usd) \
                or cost_usd < 0:
            raise ValueError(SETTLEMENT_INVALID)
        admission = record["admission"]
        amount = round(min(float(cost_usd), float(record["maximum_cost_usd"])), 6)
        command = {"task_context_digest": admission["task_context_digest"], "allocation_binding_digest": binding_digest,
                   "provider": PROVIDER, "completed_request_count": admission["request_count"],
                   "provider_charge_amount_usd": amount, "usage_receipt_digest": canonical_digest({"usage": usage})}
        if request is None:
            from .website_task_context import preparation_webapp_request as request
        receipt = dict(request(task_context=task_context, operation="preparation-settlement",
                               payload={"request_id": task_context["request_id"], "scene_id": task_context["scene_id"],
                                        "settlement": command}))
        if (receipt.get("status") != "settled" or receipt.get("allocation_binding_digest") != binding_digest
                or receipt.get("actual_cost_usd") != amount):
            raise ValueError(SETTLEMENT_INVALID)
        write_json(kept, receipt)
        return receipt


__all__ = ["ADMITTED", "FIRST", "FREE", "FREE_SOURCING", "GENERATING_BACKENDS", "HELD_WITHOUT_RECORD",
           "HOLD_AHEAD_RECORD", "HOLD_UNCONFIRMED", "KEPT", "PAID_AT_REMOVAL", "PAID_SOURCING", "REFUSED", "UNRESERVED",
           "admit_extra_objects", "admitted_reservation", "authoring_allocation", "budget_root", "reservation_binding",
           "reserve_at_removal", "route_of", "scored_object_needs", "settle_object", "settlement"]
