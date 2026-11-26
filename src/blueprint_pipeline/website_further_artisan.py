"""Several objects per job, stage 6: each paid further object authored with ARTISAN under its own reservation.

A further object that needs a generated model (the build plan's object after the scored one, with no free shape that
fits it) is reserved before the room is reconstructed, one object at a time in rank order, each on its own row of the
job's WebApp ledger at ARTISAN's frozen per-object cap ($12), with what the scored object still needs counted beside it
(``website_build_budget.reserve_at_removal``). Admitted, it is taken out of the scan with the job's object, and compile
authors it here (``author``, called by ``website_further_objects.place`` in rank order), as it authors a free one:

- under its admitted reservation only: what one run may spend is what that reservation's frozen cap leaves
  (``website_build_budget.authoring_allocation``, the rule the first object's successor resumes under), after the
  capture's current source and the account's month are read again (``_recheck_retained``): after a withdrawal nothing is
  sent;
- by the host's ARTISAN run for one object (``installed_runner``), which keeps what it spent as it goes (``METER``);
- settled once its authoring has ended, at what it cost (``website_build_budget.settle_object``, the existing OpenAI
  ``preparation-settlement``), never above the reservation. A run that fails is settled at what it spent, so the rest of
  the hold is released, and the object is named not built (``GENERATION_FAILED``); a run whose spend cannot be read
  keeps its whole hold (over, never under). An object whose reservation was settled is never authored again: money a
  settlement released is never spent under it.

Its usage is on its own row only: no scene attempt's meter records it. ``close`` settles a reservation whose object was
not authored (left in the scan, not placeable, the room shipped as a room package): at what any earlier run spent, $0
when none started. The scored object's job never waits on any of it.

No host runs ARTISAN outside a scene attempt today: the stage runs inside one (``task_evaluation_scene_configuration_
artisan_driver``), under that attempt's OpenAI gate and meter, and a further object's usage must never be on that meter.
So ``installed_runner`` is None, nothing is reserved (``reserve_at_removal`` asks ``installed`` first), and a paid
further object stays in the scan as filmed, as before stage 6.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "website_further_object_artisan_asset.v1"
REQUEST_SCHEMA = "website_further_object_artisan_request.v1"
RESULT_SCHEMA = "website_further_object_artisan_result.v1"
METER_SCHEMA = "website_further_object_artisan_meter.v1"
#: How a further object authored here was built, on its preparation's record (``website_further_objects.compose``).
SOURCE = "artisan"
#: Each paid object's authoring, under the job's build budget, keyed by its reservation.
AUTHORING = "authoring"
#: The runner's record of what it spent (``cost_usd``, ``usage``, ``complete``), kept as it goes.
METER = "meter.json"
STARTED, RESULT, FAILED = "started.json", "result.json", "failed.json"
ASSET_FILE = "generated_object.usda"
#: Why a paid further object taken out of the scan was not built; ``:`` adds the detail. Never a stop.
GENERATION_FAILED = "further_object_generation_failed"
RUNNER_MISSING = "artisan_runner_not_installed"
NOT_ADMITTED = "reservation_not_admitted"
SETTLED = "reservation_settled"
SPEND_UNKNOWN = "spend_unreadable"
CLAIM = ("A further object of the job, a 3D model Blueprint generated from frames of the video and sized from its "
         "estimated bounds; its shape and look are the generator's estimate, never a measurement. The success rule "
         "does not score it.")


def installed_runner() -> Callable[..., Any] | None:
    """The host's ARTISAN run for one further object, or None: none is installed today (the module's docstring).

    A runner takes ``request`` (``REQUEST_SCHEMA``: the object, its estimated size, its physics bounds, its task masks'
    target and the frames they index, the job's task context, its reservation's frozen budget), ``output_root`` and
    ``maximum_cost_usd`` (what its reservation leaves this run); keeps what it spent in ``output_root / METER`` as it
    goes, across every run under the same ``output_root`` (``METER_SCHEMA``: ``cost_usd``, ``usage``, and ``complete``
    once every call it made has its usage; a stopped run is resumed there with what the reservation left); and
    returns the path, inside ``output_root``, of one self-contained USD rigid body that passes
    ``website_further_objects.static_check``."""
    return None


def installed() -> bool:
    """Whether this host can author a paid further object: only then is one reserved and taken out of the scan."""
    return installed_runner() is not None


def authoring_root(budget_root: Path, binding_digest: str) -> Path:
    return Path(budget_root) / AUTHORING / str(binding_digest).removeprefix("sha256:")


def _sealed(path: Path, schema: str) -> dict[str, Any] | None:
    try:
        value = None if path.is_symlink() else json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(value, Mapping) or value.get("schema_version") != schema
            or value.get("digest") != canonical_digest(value, digest_field="digest")):
        return None
    return dict(value)


def _seal(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    record = {**value, "digest": canonical_digest(value, digest_field="digest")}
    write_json(path, record)
    return record


def _spent(work: Path) -> tuple[float, Any] | None:
    """What this object's authoring spent so far: $0 when no run started; a run's own meter when every call it made has
    its usage; None when it cannot be read (then the row keeps its whole hold)."""
    if not (work / STARTED).exists():
        return 0.0, {"runs": 0}
    try:
        meter = json.loads((work / "run" / METER).read_text())
    except (OSError, ValueError):
        return None
    cost = meter.get("cost_usd") if isinstance(meter, Mapping) else None
    if (not isinstance(meter, Mapping) or meter.get("schema_version") != METER_SCHEMA or meter.get("complete") is not True
            or isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0):
        return None
    return float(cost), meter.get("usage")


def _paid(row: Mapping[str, Any]) -> tuple[Path, str]:
    paid = row.get("paid") if isinstance(row.get("paid"), Mapping) else row
    return Path(str(paid["budget_root"])), str(paid["binding_digest"])


def close(paid: Mapping[str, Any], *, task_context: Mapping[str, Any]) -> dict[str, Any] | None:
    """Settle an admitted reservation at what its object's authoring cost (``settle_object``): its kept receipt when it
    was settled; $0 when no run started (the hold released whole); a run's own meter otherwise. None, the row keeping its
    whole quote, when the reservation was never admitted here or its spend cannot be read."""
    from .website_build_budget import admitted_reservation, settle_object, settlement
    root, digest = _paid(paid)
    if admitted_reservation(root, digest) is None:
        return None
    kept = settlement(root, digest)
    if kept is not None:
        return kept
    spent = _spent(authoring_root(root, digest))
    if spent is None:
        return None
    return settle_object(binding_digest=digest, task_context=task_context, root=root, cost_usd=spent[0],
                         usage=spent[1])


def close_all(capture_root: Path, *, task_context: Mapping[str, Any]) -> None:
    """Settle every admitted further object's reservation of the capture's build budget (``close``): called when the job
    ships a room package, in which no further object is built. Nothing to do, and nothing written, for a job that holds
    none. One that cannot be settled now keeps its whole hold (over, never under); never a stop."""
    from .website_build_plan_notes import budget_root
    root = budget_root(capture_root)
    reservations = root / "reservations"
    if not reservations.is_dir():
        return
    for path in sorted(reservations.glob("*.json")):
        if path.name.endswith(".settlement.json") or path.name.startswith("."):
            continue
        try:
            close({"budget_root": str(root), "binding_digest": "sha256:" + path.stem}, task_context=task_context)
        except Exception:  # noqa: BLE001 - an unsettled row keeps its whole quote
            continue


def _settled(row: Mapping[str, Any], task_context: Mapping[str, Any]) -> None:
    """``close``, never a stop: a row the WebApp cannot settle now keeps its whole quote (over, never under), and a later
    pass, or the room package, settles it."""
    try:
        close(row, task_context=task_context)
    except Exception:  # noqa: BLE001 - an unsettled row keeps its whole quote
        return


def _usd_bounds(path: Path) -> dict[str, list[float]]:
    """The file's own box about its origin, as ``seat`` reads an authored asset's (its receipt's ``bounds_m``)."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    box = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    return {"minimum": [float(box.GetMin()[i]) for i in range(3)], "maximum": [float(box.GetMax()[i]) for i in range(3)]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _authored(result: Mapping[str, Any], work: Path, output_root: Path) -> dict[str, Any] | None:
    """The authored object for composition from its sealed result: its file copied into compile's folder for it, with
    its receipt; None when the generated file is no longer the one the result names."""
    source = work / "run" / str(result["asset"]["path"])
    if (source.is_symlink() or not source.is_file() or not source.resolve().is_relative_to(work.resolve())
            or _sha256(source) != result["asset"]["sha256"]):
        return None
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / ASSET_FILE
    shutil.copyfile(source, path)
    receipt = {"schema_version": SCHEMA_VERSION, "source": SOURCE,
               "reservation_binding_digest": result["reservation_binding_digest"],
               "maximum_cost_usd": result["maximum_cost_usd"], "authoring_cost_usd": result["cost_usd"],
               "physics_bounds": result["physics_bounds"], "bounds_m": result["bounds_m"],
               "asset": {"path": path.name, "sha256": result["asset"]["sha256"],
                         "size_bytes": result["asset"]["size_bytes"]},
               "model_called": True, "physical_measurement": False, "claim": CLAIM}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    return {"source": SOURCE, "receipt": receipt, "path": path,
            "artisan_authoring": {"reservation_binding_digest": result["reservation_binding_digest"],
                                  "maximum_cost_usd": result["maximum_cost_usd"],
                                  "authoring_cost_usd": result["cost_usd"], "result_digest": result["digest"]}}


def author(row: Mapping[str, Any], *, dimensions: Sequence[float], tolerance: float,
           physics_bounds: Mapping[str, Any], task_context: Mapping[str, Any] | None,
           target: Mapping[str, Any] | None, frames: Sequence[Mapping[str, Any]],
           output_root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """One paid further object authored with ARTISAN under its own admitted reservation (``row["paid"]``), or None and
    why (``GENERATION_FAILED``). A finished authoring is reused, never run again; a run that stopped part way resumes
    with what its reservation's cap leaves; a run that failed is final. Settles the row once the authoring has ended
    (``close``). A privacy or rights stop reading the capture's source is raised: nothing is sent after a withdrawal."""
    from .website_build_budget import (ADMITTED, _recheck_retained, admitted_reservation, authoring_allocation,
                                       settlement)
    root, digest = _paid(row)
    record = admitted_reservation(root, digest)
    if record is None or task_context is None:
        return None, f"{GENERATION_FAILED}:{NOT_ADMITTED}"
    work = authoring_root(root, digest)
    result = _sealed(work / RESULT, RESULT_SCHEMA)
    if result is not None and result.get("reservation_binding_digest") == digest:
        _settled(row, task_context)
        authored = _authored(result, work, output_root)
        return (authored, None) if authored is not None else (None, f"{GENERATION_FAILED}:asset_changed")
    if settlement(root, digest) is not None:
        return None, f"{GENERATION_FAILED}:{SETTLED}"
    failed = _sealed(work / FAILED, "website_further_object_artisan_failure.v1")
    if failed is not None:
        _settled(row, task_context)
        return None, f"{GENERATION_FAILED}:{failed['reason']}"
    runner = installed_runner()
    if runner is None:
        _settled(row, task_context)
        return None, f"{GENERATION_FAILED}:{RUNNER_MISSING}"

    def fail(reason: str) -> tuple[None, str]:
        _seal(work / FAILED, {"schema_version": "website_further_object_artisan_failure.v1",
                              "reservation_binding_digest": digest, "reason": reason[:200]})
        _settled(row, task_context)
        return None, f"{GENERATION_FAILED}:{reason[:200]}"
    # The capture's current source and the account's month, read again before anything is sent under the hold.
    _recheck_retained(record, digest, task_context)
    carried = None
    if (work / STARTED).exists():
        spent = _spent(work)
        if spent is None:
            return fail(SPEND_UNKNOWN)  # what the stopped run spent cannot be read: never spend more under it
        carried = {"cost_usd": spent[0], "completed": False}
    allocation = authoring_allocation({"status": ADMITTED, "authoring_budget": record["authoring_budget"]}, carried)
    (work / "run").mkdir(parents=True, exist_ok=True)
    if not (work / STARTED).exists():
        _seal(work / STARTED, {"schema_version": "website_further_object_artisan_started.v1",
                               "reservation_binding_digest": digest, "maximum_cost_usd": record["maximum_cost_usd"]})
    request = {"schema_version": REQUEST_SCHEMA, "rank": row.get("rank"), "target_id": str(row.get("target_id")),
               "name": row.get("name"), "dimensions_m": [float(value) for value in dimensions],
               "tolerance": float(tolerance), "physics_bounds": dict(physics_bounds), "target": dict(target or {}),
               "frames": list(frames), "task_context": dict(task_context), "reservation_binding_digest": digest,
               "authoring_budget": dict(record["authoring_budget"])}
    try:
        asset = Path(runner(request=request, output_root=work / "run", maximum_cost_usd=allocation))
    except Exception as exc:  # noqa: BLE001 - a further object never stops the scored object's job
        return fail("runner_failed:" + type(exc).__name__)
    if (asset.is_symlink() or not asset.is_file() or not asset.resolve().is_relative_to((work / "run").resolve())):
        return fail("asset_missing")
    spent = _spent(work)
    if spent is None:
        return fail(SPEND_UNKNOWN)
    try:
        bounds = _usd_bounds(asset)
    except Exception as exc:  # noqa: BLE001 - a file that cannot be read is not built, by its error's type
        return fail("asset_unreadable:" + type(exc).__name__)
    result = _seal(work / RESULT, {
        "schema_version": RESULT_SCHEMA, "reservation_binding_digest": digest,
        "maximum_cost_usd": record["maximum_cost_usd"], "cost_usd": round(min(spent[0], record["maximum_cost_usd"]), 6),
        "usage_digest": canonical_digest({"usage": spent[1]}), "physics_bounds": dict(physics_bounds),
        "bounds_m": bounds, "asset": {"path": asset.resolve().relative_to((work / "run").resolve()).as_posix(),
                                      "sha256": _sha256(asset), "size_bytes": asset.stat().st_size}})
    _settled(row, task_context)
    authored = _authored(result, work, output_root)
    return (authored, None) if authored is not None else (None, f"{GENERATION_FAILED}:asset_changed")


__all__ = ["ASSET_FILE", "GENERATION_FAILED", "METER", "METER_SCHEMA", "REQUEST_SCHEMA", "RUNNER_MISSING", "SOURCE",
           "author", "authoring_root", "close", "close_all", "installed", "installed_runner"]
