"""The build plan's dollar budget: each extra paid object reserves its own share of the job's $40.

Owner decision 2026-11-23: the job's one cap stays $40 (``website_job_spend_cap``, mirrored byte for byte in the
WebApp), and the simulation may spend whatever the scan leaves under it. The scan (Atlas, SAM, Gemini, the planner)
spends first. The first object the build plan builds, its top-ranked one, is built under the scene intake's allowance
exactly as today: the WebApp holds that allowance in full on the job's ledger, and the Pipeline holds each scene
attempt to it (``task_evaluation_scene_intake.scene_spend_budget``). Then, one object at a time in the plan's order:

- an object the plan sources as ``generate`` (ARTISAN, at its frozen per-object cap: $12 today,
  ``task_object_authoring_backend.new_authoring_budget``) reserves that cap as its own row on the job's WebApp ledger,
  through the existing preparation reservation (``website_task_context.reserve_website_preparation_spend``: the
  account-limit precheck, then WebApp ``reserveWebsitePreparationSpend``, whose ``checkWebsiteJobCeiling`` counts
  everything the job holds or has settled on every ledger, then the paid chokepoint). The row is an ordinary OpenAI
  preparation row, ARTISAN's one OpenAI gate whichever provider writes its text: no new money channel;
- a library object (Google Scanned Objects, SimReady) or a parametric shape (a box, tote, bin, tray or can) costs
  nothing and reserves nothing.

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

Each object's reservation is written once (``reservations/<binding>.json``), keyed by the job, the sealed plan and the
object, so a retry or a resume reuses it and never reserves twice; the WebApp keys its row by the same binding. It is
written as soon as the WebApp admits it, before the local paid recheck, so a run that stops after the row was held
still finds it. An admitted reservation stays admitted. A resumed authoring run spends only what its reservation's
frozen cap leaves (``authoring_allocation``: the rule the first object's successor uses). When the object's authoring
has ended, its row settles at what it cost (``settle_object``: the existing OpenAI ``preparation-settlement``), never
above the reservation; until then it holds its whole quote.

The several-objects build calls ``admit_extra_objects`` with the objects it will build, only once the scene intake's
allowance is held, so an extra object never competes with the first. It settles an extra object's ARTISAN usage through
``settle_object`` only; a scene attempt's meter must never also record it. Today's route builds only the first object
(``website_build_plan.built_object``), so it reserves nothing; the readers (``website_build_plan_notes``, the handoff)
already name an object a limit kept out.
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
REVISION = "website_build_budget.r1"
#: The plan's sourcing for an object that costs nothing to build: a licensed library asset or a parametric shape.
FREE_SOURCING = frozenset({"library", "parametric"})
#: The plan's sourcing for an object that needs its own generated model (ARTISAN): the paid one.
PAID_SOURCING = "generate"
#: An object's status: the first (the scene intake's allowance, as today), free, its own reservation admitted, or kept
#: as filmed at a limit on the job's paid work (``KEPT``). A reservation record is admitted or ``REFUSED`` there.
FIRST, FREE, ADMITTED, REFUSED = "first_object", "free", "admitted", "refused_at_cap"
#: The WebApp row: ARTISAN's one OpenAI gate (``website_native_submission``: one reservation and one cap).
RESOURCE_CLASS, PROVIDER = "openai_api_candidate", "openai"
OBJECTS_INVALID = "website_build_budget_objects_invalid"
RECORD_INVALID = "website_build_budget_record_invalid"
SETTLEMENT_INVALID = "website_build_budget_settlement_invalid"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def reservation_binding(*, task_context: Mapping[str, Any], build_plan_digest: str,
                        row: Mapping[str, Any]) -> dict[str, Any]:
    """The one reservation an object may hold: its job, the sealed plan and the object. A retry, a resume and a later
    task context name the same one."""
    return {"kind": "website_build_extra_object", "revision": REVISION, "request_id": str(task_context["request_id"]),
            "build_plan_digest": build_plan_digest, "target_id": str(row["target_id"]), "name": str(row["name"])}


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
             reserve: Callable[..., Any] | None, now: Callable[[], float]) -> dict[str, Any]:
    """Ask the job's ledger for the object's frozen ARTISAN cap: the admitted reservation, or the refusal at a limit on
    the job's paid work, kept with the website's code as received (the code after the transport's
    ``website_control_preparation-spend_http_409:``). Anything else is raised and writes nothing, unless the WebApp
    already admitted the row: that stays written."""
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
                resource_class=RESOURCE_CLASS, provider=PROVIDER, retain_admission=retain)
    except Exception as exc:
        if os.path.lexists(path) or not _at_cap(exc):
            raise
        return _write_once(path, {**base, "status": REFUSED, **_refusal(str(exc).rsplit(":", 1)[-1]),
                                  "recorded_at_epoch": now()})
    record = _read_reservation(root, digest)
    if record is None or record["status"] != ADMITTED:
        raise ValueError(RECORD_INVALID)
    return record


def admit_extra_objects(objects: Sequence[Mapping[str, Any]], *, task_context: Mapping[str, Any],
                        build_plan_digest: str, root: Path, reserve: Callable[..., Any] | None = None,
                        now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Admit or refuse each extra paid object the build will build; returns the job's budget record, also written to
    ``root`` (``budget_root(capture_root)``, where the readers look).

    ``objects`` are the build plan's objects the build will build, in the plan's order, each as ``validate_plan`` kept
    it (``target_id``, ``name``, ``sourcing``); ``build_plan_digest`` is the sealed plan's record digest. The first is
    the first object, under the scene intake's allowance as today: nothing is reserved for it here. Each later one is
    free (``FREE_SOURCING``) or, sourced ``generate``, reserved on the job's ledger (``reserve``, by default
    ``reserve_website_preparation_spend``), unless a paid object ranked above it was refused at a limit on the job's
    paid work: then it is kept as filmed without asking, under that refusal's code. A refusal there keeps that object as
    filmed too, and is final. Any other failure is raised as the reservation raised it, after the objects before it
    were recorded in their reservations. A rerun reuses every written reservation and asks only for paid objects never
    asked. Call it only once the scene intake's allowance is held.
    """
    rows = [dict(row) for row in objects if isinstance(row, Mapping)]
    try:
        identities = [(str(row["target_id"]), str(row["name"])) for row in rows]
    except KeyError:
        raise ValueError(OBJECTS_INVALID) from None
    if (not rows or len(rows) != len(objects) or len(set(identities)) != len(identities)
            or any(row.get("sourcing") not in FREE_SOURCING | {PAID_SOURCING} for row in rows)
            or not isinstance(build_plan_digest, str) or not _DIGEST.fullmatch(build_plan_digest)):
        raise ValueError(OBJECTS_INVALID)
    root = Path(root)

    def named(rank: int, row: Mapping[str, Any]) -> dict[str, Any]:
        return {"rank": rank, "target_id": str(row["target_id"]), "name": str(row["name"]), "sourcing": row["sourcing"]}

    entries: list[dict[str, Any]] = [{**named(0, rows[0]), "status": FIRST}]
    with _locked(root):
        capped = None  # the refusal that keeps every later paid object as filmed, once there is one
        for rank, row in enumerate(rows[1:], start=1):
            if row["sourcing"] in FREE_SOURCING:
                entries.append({**named(rank, row), "status": FREE})
                continue
            value = reservation_binding(task_context=task_context, build_plan_digest=build_plan_digest, row=row)
            digest = canonical_digest(value)
            record = _read_reservation(root, digest)
            if record is None and capped is None:
                record = _reserve(root=root, digest=digest, value=value, task_context=task_context, reserve=reserve,
                                  now=now)
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
                  "claim_boundary": "Reservations inside the job's limits on paid work, in the plan's order; a refusal "
                                    "names the website's code, which covers the dollar cap and the paid-request count "
                                    "alike, never which one. No object is built by this record."}
        budget["digest"] = canonical_digest(budget, digest_field="digest")
        write_json(root / BUDGET_RECORD, budget)
    return budget


def authoring_allocation(entry: Mapping[str, Any], carried: Mapping[str, Any] | None) -> float:
    """What one authoring run of an admitted object may spend: its reservation's frozen cap less what earlier runs
    under it spent (``carried``, as ``factory_carried_spend`` reports it), by the rule the first object's successor
    uses (``resumed_authoring_allocation``). A resume never reserves again; it spends what its reservation left."""
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


__all__ = ["ADMITTED", "FIRST", "FREE", "FREE_SOURCING", "KEPT", "PAID_SOURCING", "REFUSED", "admit_extra_objects",
           "authoring_allocation", "budget_root", "reservation_binding", "settle_object"]
