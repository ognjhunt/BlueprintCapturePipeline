"""Settle each rigid robot-free probe attempt's website-grant hold to what it ran, once nothing can spend.

Beta journey step: L5 team policy evaluation. Each funded attempt reserves its
$1.00 cap from the capture's website preparation grant (``gpu_render``/``vast``
on the upstream ledger). Nothing settled those holds, so capture-447ae4f4's four
attempts kept $4.00 of the $5 upstream cap while their ledgers estimated $0.62,
and the next $1.00 hold was refused (``rigid_robot_free_probe_website_grant_exhausted``,
2026-11-04 01:07Z).

``settle_attempt_spend`` settles one attempt through the WebApp's signed
``preparation-settlement`` transport, the route every preparation lane settles
on, once the attempt's own records prove the run is over:

- the allocator result says nothing continues to spend and nothing is left
  staged, its provider closeout confirms provider zero, and its independent
  watchdog confirms the instance is gone;
- the teardown manifest it names (checked by digest, inside the attempt's own
  directory) completed every destroy for exactly one instance;
- the adapter's budget ledger beside it is complete for that instance.

Vast's instance API reports no bill for these runs (the ledger's
``actual_cost_usd`` is null, ``actual_cost_source`` is
``not_available_from_instance_probe_api``), so the hold settles at the ledger's
estimate (the observed all-in hourly rate over the observed live time, plus the
priced transfer), sent and recorded as ``adapter_budget_ledger_estimate`` with
the ledger's digest, never as a provider charge. It is the same figure the
remote CPU lane settles at once its teardown is sealed
(``remote_cpu_job_allocator.settle_remote_cpu_attempt``). Beta screening has
no settlement to reuse: its site ledger keeps every allocated cap in full.

The settlement record is written once beside the attempt
(``rigid_robot_free_probe_spend_settlement.json``). The WebApp settles each
reservation key once, never above its quote, and returns the same receipt to
the same command, so a lost answer is resent on the next pass. An attempt
whose teardown is unproven, or whose records do not bind one funded run, is
never settled and keeps its full hold. Settlement only records accounting: it
authorizes nothing and never raises into the runner.

``provider_no_start`` reads the same proven records for a run whose machine
never started our container: the adapter classified it
``pre_execution_provider_null`` and avoided the machine for
``vast_startup_control_plane_did_not_reach_onstart_heartbeat``. The runner
counts such a run apart from its funded attempts (bounded on its own), and
``refresh_lane_avoidlist`` keeps every machine an attempt's adapter avoided in
the lane's shared avoidlist, so the next attempt never rents it again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest

SETTLEMENT_SCHEMA = "blueprint.rigid_robot_free_probe_spend_settlement.v1"
SETTLEMENT_FILENAME = "rigid_robot_free_probe_spend_settlement.json"
CHARGE_BASIS = "adapter_budget_ledger_estimate"
LEDGER_COST_SOURCE = "not_available_from_instance_probe_api"
FUNDING_FILENAME = "website_preparation_spend_admission.json"
RESULT_FILENAME = "allocator_result.json"
ADMISSION_FILENAME = "paid_admission.json"
LEDGER_FILENAME = "vast_budget_ledger.json"
JOB_AVOIDLIST_FILENAME = "adp_arena_vast_machine_avoidlist.json"
NO_START_REASON = "vast_startup_control_plane_did_not_reach_onstart_heartbeat"
AVOIDLIST_SCHEMA = "vast_machine_avoidlist.v1"
LANE_AVOIDLIST_DIRNAME = "provider-machine-avoidlists"
LANE_AVOIDLIST_FILENAME = "rigid-robot-free-probe-vast-machine-avoidlist.json"
ATTEMPT_AVOIDLIST_FILENAME = "machine_avoidlist.json"
#: The lane's cap; a funding admission above it is not this lane's.
MAXIMUM_RESERVATION_USD = 1.00
_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_REQUEST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")

_LOG = logging.getLogger(__name__)


class ProbeSettlementEvidenceError(ValueError):
    """The attempt's records do not prove one funded, torn-down run; nothing is settled."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ProbeSettlementEvidenceError(reason)


def _safe(path: Path) -> bool:
    return path.is_absolute() and not any(part.is_symlink() for part in (path, *path.parents))


def _bytes(path: Path) -> bytes:
    _require(_safe(path) and path.is_file(), f"rigid_robot_free_probe_settlement_record_missing:{path.name}")
    return path.read_bytes()


def _json(raw: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except ValueError:
        value = None
    _require(isinstance(value, dict), f"rigid_robot_free_probe_settlement_record_invalid:{name}")
    return value


def _read(path: Path) -> dict[str, Any]:
    return _json(_bytes(path), path.name)


def _read_optional(path: Path) -> dict[str, Any] | None:
    try:
        return _read(path) if path.exists() else None
    except ProbeSettlementEvidenceError:
        return None


def _sha256(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _referenced(ref: Any, *, inside: Path, name: str) -> tuple[dict[str, Any], str]:
    """A file a result names by path and digest, re-read and re-hashed, inside the attempt's own directory."""

    _require(isinstance(ref, Mapping) and isinstance(ref.get("path"), str)
             and _SHA.fullmatch(str(ref.get("sha256") or "")) is not None,
             f"rigid_robot_free_probe_settlement_reference_invalid:{name}")
    path = Path(ref["path"])
    _require(path.is_absolute() and ".." not in path.parts and inside in path.parents,
             f"rigid_robot_free_probe_settlement_reference_outside_attempt:{name}")
    raw = _bytes(path)
    _require(_sha256(raw) == ref["sha256"], f"rigid_robot_free_probe_settlement_reference_changed:{name}")
    return _json(raw, name), ref["sha256"]


def _instance_ids(value: Any) -> list[int]:
    if not isinstance(value, list) or not all(type(item) is int and item > 0 for item in value):
        return []
    return list(value)


def _money(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def funding_admission(attempt: Path, work: Path) -> dict[str, Any] | None:
    """The website reservation that funded this attempt, bound to its own allocation, or None when unfunded.

    Raises when a reservation is there but does not bind this attempt's admission (never settled then).
    """

    path = Path(work) / "allocator" / FUNDING_FILENAME
    if not path.exists():
        return None
    funding = _read(path)
    admission = _read(Path(attempt) / ADMISSION_FILENAME)
    maximum = _money(funding.get("maximum_cost_usd"))
    _require(funding.get("schema_version") == "paid_lane_admission.v1" and funding.get("status") == "admitted"
             and funding.get("provider") == "vast" and funding.get("resource_class") == "gpu_render"
             and maximum is not None and 0 < maximum <= MAXIMUM_RESERVATION_USD
             and _SHA.fullmatch(str(funding.get("task_context_digest") or "")) is not None
             and _SHA.fullmatch(str(funding.get("allocation_binding_digest") or "")) is not None,
             "rigid_robot_free_probe_settlement_funding_invalid")
    _require(admission.get("allocation_binding_digest") == funding["allocation_binding_digest"]
             and admission.get("website_preparation_admission_digest") == canonical_digest(funding),
             "rigid_robot_free_probe_settlement_funding_not_this_attempts")
    return {"funding": funding, "admission": admission}


def terminal_closeout(attempt: Path, work: Path) -> dict[str, Any]:
    """The attempt's proven end: one instance destroyed, provider zero, its ledger's estimate; raises otherwise."""

    allocator = Path(work) / "allocator"
    raw = _bytes(Path(attempt) / RESULT_FILENAME)
    result = _json(raw, RESULT_FILENAME)
    closeout = result.get("provider_closeout")
    watchdog = result.get("independent_watchdog_close")
    _require(result.get("continuing_spend_from_this_run") is False and result.get("all_staged_objects_absent") is True
             and isinstance(closeout, Mapping) and closeout.get("provider_zero_confirmed") is True
             and closeout.get("all_staged_objects_absent") is True and closeout.get("warm_session_retained") is False
             and isinstance(watchdog, Mapping) and watchdog.get("provider_absence_confirmed") is True,
             "rigid_robot_free_probe_settlement_teardown_unproven")
    teardown, teardown_digest = _referenced(closeout.get("teardown_manifest"), inside=allocator, name="teardown")
    ids = _instance_ids(teardown.get("vast_instance_ids"))
    actions = teardown.get("teardown_actions_performed")
    _require(teardown.get("schema_version") == "vast_teardown_manifest.v1" and teardown.get("status") == "completed"
             and teardown.get("runner_gpu_teardown_completed") is True
             and teardown.get("continuing_spend_from_this_run") is False and teardown.get("retention_authorized") is False
             and len(ids) == 1 and isinstance(actions, list) and actions
             and all(isinstance(row, Mapping) and row.get("status") == "completed" and row.get("instance_id") in ids
                     for row in actions)
             and {row.get("instance_id") for row in actions} == set(ids)
             and _instance_ids(watchdog.get("instance_ids")) == ids,
             "rigid_robot_free_probe_settlement_teardown_unproven")
    adapter, _adapter_digest = _referenced(closeout.get("adapter_result"), inside=allocator, name="adapter_result")
    _require(_instance_ids(adapter.get("vast_instance_ids")) == ids
             and adapter.get("continuing_spend_from_this_run") is False,
             "rigid_robot_free_probe_settlement_adapter_mismatch")
    ledger_path = Path(str(closeout["teardown_manifest"]["path"])).parent / LEDGER_FILENAME
    ledger_raw = _bytes(ledger_path)
    ledger = _json(ledger_raw, LEDGER_FILENAME)
    estimate = _money(ledger.get("estimated_cost_usd"))
    _require(ledger.get("schema_version") == "vast_budget_ledger.v1" and ledger.get("status") == "completed"
             and ledger.get("continuing_spend_from_this_run") is False
             and _instance_ids(ledger.get("vast_instance_ids")) == ids
             and ledger.get("provider_transfer_cost_priced") is True
             and ledger.get("actual_cost_usd") is None and ledger.get("actual_cost_source") == LEDGER_COST_SOURCE
             and estimate is not None,
             "rigid_robot_free_probe_settlement_ledger_invalid")
    # One allocation, so the session's figures are this ledger's: anything else is not one run's estimate.
    for figure in (result.get("estimated_cost_usd"), closeout.get("estimated_cost_usd")):
        _require(_money(figure) is not None and abs(float(figure) - estimate) <= 1e-9,
                 "rigid_robot_free_probe_settlement_estimate_inconsistent")
    return {"instance_id": ids[0], "estimated_cost_usd": round(estimate, 6), "ledger_path": str(ledger_path),
            "budget_ledger_digest": _sha256(ledger_raw), "execution_result_digest": _sha256(raw),
            "teardown_receipt_digest": teardown_digest, "provider_zero_digest": canonical_digest(dict(closeout)),
            "ledger": ledger, "teardown": teardown, "adapter": adapter}


def provider_no_start(attempt: Path, work: Path) -> dict[str, Any] | None:
    """The machine that never started our container on this proven, torn-down run, or None.

    It needs all of: the adapter's own ``pre_execution_provider_null`` classification (our bundle and
    entrypoint never started and nothing came back), no container health proven at teardown, and the
    adapter's avoidlist entry for exactly this instance naming the on-start heartbeat. Unreadable or
    partial evidence is None, so the run counts as an ordinary funded attempt.
    """

    from .task_evaluation_scene_recovery import provider_null_evidence

    try:
        closeout = terminal_closeout(attempt, work)
        adapter, teardown = closeout["adapter"], closeout["teardown"]
        if not provider_null_evidence(adapter):
            return None
        decision = teardown.get("retention_decision")
        if not isinstance(decision, Mapping) or decision.get("container_health_proven") is not False:
            return None
        avoided = _read(Path(work) / "allocator" / JOB_AVOIDLIST_FILENAME)
        rows = [row for row in avoided.get("entries") or [] if isinstance(row, Mapping)
                and row.get("instance_id") == closeout["instance_id"] and row.get("reason") == NO_START_REASON
                and type(row.get("machine_id")) is int and row["machine_id"] > 0]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if avoided.get("schema_version") != AVOIDLIST_SCHEMA or len(rows) != 1:
        return None
    classification = adapter["provider_attempt_classification"]
    return {"instance_id": closeout["instance_id"], "machine_id": rows[0]["machine_id"],
            "offer_id": rows[0].get("offer_id"), "reason": NO_START_REASON,
            "classification": classification.get("classification"),
            "blockers": [str(item) for item in classification.get("blockers") or []][:10],
            "estimated_cost_usd": closeout["estimated_cost_usd"]}


def settlement_command(attempt: Path, work: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """The settlement this attempt's records prove, its funding and closeout; None when it was never funded."""

    funded = funding_admission(attempt, work)
    if funded is None:
        return None
    funding = funded["funding"]
    closeout = terminal_closeout(attempt, work)
    _require(closeout["estimated_cost_usd"] <= float(funding["maximum_cost_usd"]),
             "rigid_robot_free_probe_settlement_estimate_above_reservation")
    command = {"task_context_digest": funding["task_context_digest"],
               "allocation_binding_digest": funding["allocation_binding_digest"], "provider": "vast",
               "instance_id": str(closeout["instance_id"]), "provider_charge_basis": CHARGE_BASIS,
               "provider_charge_amount_usd": closeout["estimated_cost_usd"],
               "budget_ledger_digest": closeout["budget_ledger_digest"],
               "execution_result_digest": closeout["execution_result_digest"],
               "teardown_receipt_digest": closeout["teardown_receipt_digest"],
               "provider_zero_digest": closeout["provider_zero_digest"],
               "continuing_spend_from_this_run": False, "all_staged_objects_absent": True}
    return command, funded, closeout


def _website_identity(admission: Mapping[str, Any]) -> dict[str, str]:
    """The request the reservation was made under, from the capture's retained website task context."""

    capture_root = (admission.get("allocation_binding") or {}).get("capture_root")
    _require(isinstance(capture_root, str) and Path(capture_root).is_absolute(),
             "rigid_robot_free_probe_settlement_capture_unbound")
    context = _read(Path(capture_root) / "pipeline" / "website_task_context.json")
    request_id, scene_id, capture_id = (context.get(key) for key in ("request_id", "scene_id", "capture_id"))
    _require(isinstance(request_id, str) and _REQUEST.fullmatch(request_id) is not None
             and scene_id == f"site-{request_id}" and capture_id == Path(capture_root).name,
             "rigid_robot_free_probe_settlement_capture_context_invalid")
    return {"request_id": request_id, "scene_id": scene_id, "capture_id": capture_id}


def _write_once(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    text = json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False,
                                     encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(stream.name, 0o640)
        os.link(stream.name, path)
    except FileExistsError:
        return _read(path)
    finally:
        os.unlink(stream.name)
    return dict(value)


def settle_attempt_spend(attempt: Path, work: Path) -> dict[str, Any]:
    """Settle one attempt's hold once its teardown is proven; never raises, never spends.

    ``status`` is ``settled`` (now or before), ``not_funded``, ``pending`` (no result yet: still
    running or unknown), ``held`` (the records do not prove one torn-down run; the full hold stays)
    or ``deferred`` (the WebApp did not settle it; the next pass resends the same command).
    """

    from . import website_task_context

    attempt, work = Path(attempt), Path(work)
    path = attempt / SETTLEMENT_FILENAME
    retained = _read_optional(path)
    if retained is not None:
        return {"status": "settled", "attempt": str(attempt), "settled_cost_usd": retained.get("settled_cost_usd"),
                "record": str(path)}
    try:
        if not (work / "allocator" / FUNDING_FILENAME).exists():
            return {"status": "not_funded", "attempt": str(attempt)}
        if not (attempt / RESULT_FILENAME).is_file():
            # The allocator is still running, or its fate is unknown: nothing is settled before it ends.
            return {"status": "pending", "attempt": str(attempt)}
        built = settlement_command(attempt, work)
        if built is None:
            return {"status": "not_funded", "attempt": str(attempt)}
        command, funded, closeout = built
        identity = _website_identity(funded["admission"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        # Not one proven, funded, torn-down run: the full hold stays.
        return {"status": "held", "attempt": str(attempt), "reason": f"{type(exc).__name__}:{str(exc)[:200]}"}
    try:
        receipt = website_task_context.website_webapp_request(
            capture_id=identity["capture_id"], operation="preparation-settlement",
            payload={"request_id": identity["request_id"], "scene_id": identity["scene_id"], "settlement": command})
        if (receipt.get("status") != "settled" or any(receipt.get(key) != value for key, value in command.items())
                or receipt.get("actual_cost_usd") != command["provider_charge_amount_usd"]):
            raise ValueError("rigid_robot_free_probe_settlement_receipt_invalid")
    except Exception as exc:  # noqa: BLE001 - accounting only: the hold stays and the next pass resends
        _LOG.warning("rigid_robot_free_probe_settlement_deferred attempt=%s binding=%s reason=%s", attempt,
                     command["allocation_binding_digest"], str(exc)[:200])
        return {"status": "deferred", "attempt": str(attempt), "reason": str(exc)[:200]}
    no_start = provider_no_start(attempt, work)
    record = {"schema_version": SETTLEMENT_SCHEMA, "status": "settled", "attempt": attempt.name,
              "capture_id": identity["capture_id"], "request_id": identity["request_id"],
              "reserved_maximum_cost_usd": float(funded["funding"]["maximum_cost_usd"]),
              "settled_cost_usd": receipt["actual_cost_usd"], "charge_basis": CHARGE_BASIS,
              "charge_source": {"budget_ledger_path": closeout["ledger_path"],
                                "actual_cost_usd": closeout["ledger"].get("actual_cost_usd"),
                                "actual_cost_source": closeout["ledger"].get("actual_cost_source")},
              "provider_no_start": no_start is not None, "command": command, "receipt": dict(receipt),
              "settled_at_iso": utc_now_iso(), "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    try:
        written = _write_once(path, record)
    except (OSError, ProbeSettlementEvidenceError) as exc:
        # The WebApp kept the settlement; the next pass resends the same command and records its receipt.
        return {"status": "deferred", "attempt": str(attempt), "reason": f"record_not_written:{type(exc).__name__}"}
    return {"status": "settled", "attempt": str(attempt), "settled_cost_usd": written.get("settled_cost_usd"),
            "record": str(path), "settled_now": True}


def lane_avoidlist_path(queue_root: Path) -> Path:
    return Path(queue_root) / LANE_AVOIDLIST_DIRNAME / LANE_AVOIDLIST_FILENAME


def _avoidlist_rows(path: Path) -> list[dict[str, Any]]:
    """The admitted entries of one avoidlist; an unreadable one contributes nothing (it only narrows offers)."""

    from .provider_machine_avoidlist import load_machine_avoidlist, machine_avoidlist_ids

    if not _safe(path) or not path.is_file():
        return []
    value = load_machine_avoidlist(path)
    try:
        machine_avoidlist_ids(value)
    except ValueError:
        _LOG.warning("rigid_robot_free_probe_machine_avoidlist_unreadable path=%s", path)
        return []
    rows = [dict(row) for row in value.get("entries") or [] if isinstance(row, Mapping)]
    rows += [{"machine_id": item} for item in value.get("machine_ids") or []
             if type(item) is int and item > 0 and item not in {row.get("machine_id") for row in rows}]
    return rows


def _merged(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entries, seen = [], set()
    for row in rows:
        identity = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
        if type(row.get("machine_id")) is int and row["machine_id"] > 0 and identity not in seen:
            seen.add(identity)
            entries.append(row)
    return {"schema_version": AVOIDLIST_SCHEMA, "status": "completed",
            "machine_ids": sorted({row["machine_id"] for row in entries}), "entries": entries,
            "raw_secret_values_recorded": False}


def refresh_lane_avoidlist(queue_root: Path, work_dirs: list[Path]) -> dict[str, Any]:
    """Merge every machine an attempt's adapter avoided into the lane's shared avoidlist (never drops one)."""

    shared = lane_avoidlist_path(queue_root)
    rows = _avoidlist_rows(shared)
    for work in work_dirs:
        rows += _avoidlist_rows(Path(work) / "allocator" / JOB_AVOIDLIST_FILENAME)
    value = _merged(rows)
    if value["machine_ids"] and _merged(_avoidlist_rows(shared)) != value:
        shared.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        with tempfile.NamedTemporaryFile("w", dir=shared.parent, prefix=f".{shared.name}.", delete=False,
                                         encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(stream.name, 0o640)
        os.replace(stream.name, shared)
    return value


def attempt_machine_avoidlist(queue_root: Path, attempt: Path, profile_path: str | None) -> str | None:
    """The avoidlist a new attempt's allocator reads: the profile's, plus every machine the lane avoids.

    With nothing in the lane's list this is the profile's path, unchanged. Otherwise the attempt gets
    its own snapshot of both (the profile's file is only read), written once beside its records.
    """

    lane = _avoidlist_rows(lane_avoidlist_path(queue_root))
    if not lane:
        return profile_path
    rows = (_avoidlist_rows(Path(profile_path)) if profile_path else []) + lane
    path = Path(attempt) / ATTEMPT_AVOIDLIST_FILENAME
    try:
        _write_once(path, _merged(rows))
    except (OSError, ProbeSettlementEvidenceError) as exc:
        # It only narrows offer selection: without it the attempt reads the profile's list, as before.
        _LOG.warning("rigid_robot_free_probe_machine_avoidlist_not_staged attempt=%s reason=%s", attempt,
                     type(exc).__name__)
        return profile_path
    return str(path)


__all__ = [
    "CHARGE_BASIS",
    "LANE_AVOIDLIST_FILENAME",
    "NO_START_REASON",
    "SETTLEMENT_FILENAME",
    "ProbeSettlementEvidenceError",
    "attempt_machine_avoidlist",
    "funding_admission",
    "lane_avoidlist_path",
    "provider_no_start",
    "refresh_lane_avoidlist",
    "settle_attempt_spend",
    "settlement_command",
    "terminal_closeout",
]
