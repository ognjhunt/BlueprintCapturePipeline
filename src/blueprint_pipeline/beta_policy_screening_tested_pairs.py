"""Which (policy, scene, harness, cells) pairs a job already has a finished scored result for.

Plan 1.2 (``docs/launch-readiness/2026-11-11-evaluation-service-plan.md``): never test the same pair
twice. A pair is one policy as released (its ``policy_identity``), on one scene revision, under one
screening harness revision, over one set of screening cells (the digest of the row's frozen
``scenario_cells``). Two results of the same pair are the same measurement, so the second is never
paid for.

The policy identity holds only the catalog fields that can change a result: the checkpoint (kind,
source, revision, digest), the interface (robot, cameras, action settings and horizons) and the runtime
(backend and runtime id). Descriptive fields (display name, deployment notes and links, licence notes,
eligibility) are left out, so editing them never makes a new pair, and the same checkpoint entered under
a new candidate id is the same policy. A row with no stable identity (no checkpoint revision or digest,
no interface, or a scripted controller) also carries its candidate id, and its record says so
(``identity_basis``). A pair recorded before the identity existed keeps its old key (candidate id and
whole-row ``configuration_digest``, ``legacy_pair_key``): a row that still has that key is still tested,
so nothing finished is run or paid for again.

What counts as finished is what the coordinator already treats as a final scored result
(``beta_policy_screening_coordinator._result_not_counted``): a completed or closed row that scored at
least one episode. A run our harness failed (``_harness_failed``) or the rented machine failed
(``_machine_faulted``) is not finished: the coordinator's own bounded retries still apply to it. A result
made under an older harness revision has another key, so it never blocks the candidate's run under the
current one.

The ledger and the retained plans are the record; ``record_tested_pairs`` also writes the job's derived
list under ``<queue>/tested_pairs/<site_id>.json`` for operators. Nothing here moves money.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

RECORD_SCHEMA = "blueprint.beta_screening_tested_pairs.v1"
RECORD_DIRNAME = "tested_pairs"
#: Raised by the planner when every requested row is a pair the job already has a result for.
EVERY_PAIR_TESTED = "beta_screening_every_pair_already_tested"


def cells_digest(cells: Sequence[Mapping[str, Any]]) -> str:
    return canonical_digest({"cells": list(cells)})


#: The basis of a policy identity: the checkpoint itself, or the candidate id where the row has nothing stable.
CHECKPOINT_BASIS = "checkpoint"
CANDIDATE_ID_BASIS = "candidate_id_fallback"
#: A pair read from a campaign whose catalog was not kept: only its old key is known.
LEGACY_BASIS = "legacy_configuration_digest"
EVERY_PAIR_RECORD_SCHEMA = "blueprint.beta_screening_every_pair_tested.v1"
#: One planning lock per job (``site_planning_lock``), taken by both planners.
PLANNING_LOCK_PREFIX = ".pair-planning-"
#: The order every screening lock is taken in, outermost first. A planner holding one never waits on one
#: earlier in this list, so the two planners cannot deadlock: the worker holds ``.source.lock`` for its pass,
#: then a job's planning lock; ``on_job`` takes the planning lock, then ``.standing.lock``; either takes the
#: ledger lock last and briefly.
LOCK_ORDER = (".source.lock", PLANNING_LOCK_PREFIX, ".standing.lock", ".ledger.lock")


def planning_lock_name(site_id: str) -> str:
    return PLANNING_LOCK_PREFIX + canonical_digest({"site_id": site_id})[7:39] + ".lock"


@contextmanager
def site_planning_lock(queue_root: str | Path | None, site_id: str) -> Iterator[None]:
    """Held from reading a job's tested and in-flight pairs until its new campaign is enqueued.

    Both planners (the worker's re-plan and ``beta_policy_screening_on_job``) take it, so two requests at
    the same moment cannot both read the same exclusions and both plan, and pay for, the same pair."""
    from .beta_policy_screening_coordinator import _lock, _root

    with _lock(_root(queue_root), planning_lock_name(site_id)):
        yield


def policy_identity(row: Mapping[str, Any]) -> dict[str, str]:
    """``{"policy_identity", "identity_basis"}`` of one catalog row, from the fields that can change a result."""
    policy = row.get("policy") or {}
    runtime = row.get("runtime")
    fields: dict[str, Any] = {
        "policy": {key: policy.get(key) for key in ("kind", "source", "revision", "digest")},
        "interface": row.get("interface"),
        "runtime": None if runtime is None else {key: runtime.get(key) for key in ("backend", "runtime_id")}}
    stable = bool(policy.get("digest") or policy.get("revision")) and row.get("interface") is not None
    if not stable or policy.get("kind") == "scripted":
        fields["candidate_id"] = row["candidate_id"]
    return {"policy_identity": canonical_digest(fields),
            "identity_basis": CHECKPOINT_BASIS if "candidate_id" not in fields else CANDIDATE_ID_BASIS}


def pair_key(*, candidate_id: str, configuration_digest: str, scene_revision_digest: str, harness_revision: int,
             cells: Sequence[Mapping[str, Any]]) -> str:
    """The old key of one measurement (candidate id and whole catalog row): kept so earlier results still count."""
    return canonical_digest({"candidate_id": candidate_id, "configuration_digest": configuration_digest,
                             "scene_revision_digest": scene_revision_digest,
                             "screening_harness_revision": int(harness_revision),
                             "cells_digest": cells_digest(cells)})


def identity_pair_key(*, policy_identity: str, scene_revision_digest: str, harness_revision: int,
                      cells: Sequence[Mapping[str, Any]]) -> str:
    """The identity of one measurement: the same key is the same test."""
    return canonical_digest({"policy_identity": policy_identity, "scene_revision_digest": scene_revision_digest,
                             "screening_harness_revision": int(harness_revision), "cells_digest": cells_digest(cells)})


def candidate_keys(candidate: Mapping[str, Any], *, scene_revision_digest: str, harness_revision: int,
                   cells: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    """Both keys of a catalog row about to be planned: tested when either is already recorded."""
    identity = policy_identity(candidate)["policy_identity"]
    return frozenset({
        identity_pair_key(policy_identity=identity, scene_revision_digest=scene_revision_digest,
                          harness_revision=harness_revision, cells=cells),
        pair_key(candidate_id=candidate["candidate_id"], configuration_digest=candidate["configuration_digest"],
                 scene_revision_digest=scene_revision_digest, harness_revision=harness_revision, cells=cells)})


def campaign_catalog_rows(root: Path, plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The catalog rows a plan was frozen from (by candidate id), or ``{}`` when that catalog was not kept."""
    from .beta_policy_candidate_catalog import validate_catalog
    from .website_candidate_selection import CATALOG_FILENAME

    try:
        catalog = validate_catalog(json.loads((root / "inputs" / plan["campaign_id"] / CATALOG_FILENAME)
                                              .read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    if catalog["catalog_digest"] != plan.get("catalog_digest"):
        return {}
    return {row["candidate_id"]: row for row in catalog["candidates"]}


def plan_row_key(plan: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    """A planned row's key; ``None`` for a row that froze no cells (it never ran)."""
    from .beta_policy_screening_plan import plan_harness_revision

    cells = row.get("scenario_cells")
    if not isinstance(cells, list) or not cells:
        return None
    return pair_key(candidate_id=row["candidate_id"], configuration_digest=row["configuration_digest"],
                    scene_revision_digest=plan["scene_revision_digest"],
                    harness_revision=plan_harness_revision(plan), cells=cells)


def finished(row: Mapping[str, Any]) -> bool:
    """A ledger row holding a final scored result (never an infrastructure failure)."""
    from .beta_policy_screening_coordinator import _harness_failed, _machine_faulted
    from .beta_policy_screening_known_good_control import is_control

    summary = row.get("summary") or {}
    return (not is_control(row) and row.get("status") in {"completed", "blocked"}
            and bool(summary.get("episodes_scored")) and not _harness_failed(row) and not _machine_faulted(row))


def in_flight(row: Mapping[str, Any]) -> bool:
    """A ledger row that may still run (queued or active), in a campaign that was not superseded."""
    from .beta_policy_screening_known_good_control import is_control
    from .beta_policy_screening_result import ACTIVE_STATUSES, QUEUED_STATUSES

    return not is_control(row) and row.get("status") in QUEUED_STATUSES | ACTIVE_STATUSES


def site_pairs(queue_root: str | Path | None, site_id: str) -> dict[str, dict[str, dict[str, Any]]]:
    """``{"finished": {key: entry}, "in_flight": {key: entry}}`` for one job, from the ledger and its plans.

    Each entry is keyed by its policy-identity key when its campaign's catalog was kept, else by its old key;
    it carries both (``legacy_pair_key``), and ``planning_exclusions`` gives every key a planner must skip."""
    from .beta_policy_screening_coordinator import _ledger, _lock, _plan, _root
    from .beta_policy_screening_plan import plan_harness_revision

    root = _root(queue_root)
    with _lock(root):
        ledger = _ledger(root)
    found: dict[str, dict[str, dict[str, Any]]] = {"finished": {}, "in_flight": {}}
    for campaign_id, state in sorted(ledger["campaigns"].items(), key=lambda item: (item[1]["created_at"], item[0])):
        if state["site_id"] != site_id:
            continue
        rows = state["candidates"]
        wanted = {key for key, row in rows.items()
                  if finished(row) or in_flight(row) and not state.get("superseded_by")}
        if not wanted:
            continue
        plan = _plan(root, campaign_id, state["plan_digest"])
        catalog_rows = campaign_catalog_rows(root, plan)
        for configured in plan["candidates"]:
            row = rows.get(configured["candidate_id"]) or {}
            if configured["candidate_id"] not in wanted:
                continue
            legacy = plan_row_key(plan, configured)
            if legacy is None:
                continue
            source = catalog_rows.get(configured["candidate_id"])
            if source is not None and source.get("configuration_digest") == configured["configuration_digest"]:
                identity = policy_identity(source)
                key = identity_pair_key(policy_identity=identity["policy_identity"],
                                        scene_revision_digest=plan["scene_revision_digest"],
                                        harness_revision=plan_harness_revision(plan),
                                        cells=configured["scenario_cells"])
            else:
                identity, key = {"policy_identity": None, "identity_basis": LEGACY_BASIS}, legacy
            kind = "finished" if finished(row) else "in_flight"
            found[kind].setdefault(key, {
                "pair_key": key, "legacy_pair_key": legacy, **identity, "candidate_id": configured["candidate_id"],
                "configuration_digest": configured["configuration_digest"],
                "scene_revision_digest": plan["scene_revision_digest"],
                "screening_harness_revision": plan.get("screening_harness_revision", 0),
                "cells_digest": cells_digest(configured["scenario_cells"]), "campaign_id": campaign_id,
                "status": row.get("status"),
                **({"session_result_digest": (row.get("summary") or {}).get("session_result_digest")}
                   if kind == "finished" else {})})
    # A finished pair is finished even while another campaign still lists it (by either of its keys).
    done = planning_exclusions({"finished": found["finished"], "in_flight": {}})
    for key, entry in list(found["in_flight"].items()):
        if {entry["pair_key"], entry["legacy_pair_key"]} & done:
            found["in_flight"].pop(key)
    return found


def planning_exclusions(pairs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> frozenset[str]:
    """Every key a new plan must leave out: each finished pair and each pair still queued or running.

    The one rule both planners use (the worker's re-plan and ``beta_policy_screening_on_job``), so a pair
    in flight is never paid for a second time by either."""
    return frozenset(key for kind in ("finished", "in_flight") for entry in pairs[kind].values()
                     for key in (entry["pair_key"], entry["legacy_pair_key"]))


def excluded_pair_keys(queue_root: str | Path | None, site_id: str) -> frozenset[str]:
    return planning_exclusions(site_pairs(queue_root, site_id))


def finished_pair_keys(queue_root: str | Path | None, site_id: str) -> frozenset[str]:
    return planning_exclusions({"finished": site_pairs(queue_root, site_id)["finished"], "in_flight": {}})


def every_pair_inputs_digest(*, campaign_id: str, profile: Mapping[str, Any], implementation_commit: str | None,
                             excluded: frozenset[str]) -> str:
    """What could change an "every pair already tested" outcome for one job.

    ``campaign_id`` already binds the scene request (its scene revision and task) and the effective catalog
    (registry entries included); the profile, the release, the harness revision and the keys left out are
    added, so a change to any of them plans again."""
    from .beta_policy_screening_plan import SCREENING_HARNESS_REVISION

    return canonical_digest({"campaign_id": campaign_id, "profile": dict(profile),
                             "implementation_commit": implementation_commit,
                             "screening_harness_revision": SCREENING_HARNESS_REVISION,
                             "excluded_pair_keys": sorted(excluded)})


def _every_pair_path(root: Path, campaign_id: str) -> Path:
    return root / RECORD_DIRNAME / f"{campaign_id}.every-pair-tested.json"


def every_pair_tested_recorded(queue_root: str | Path | None, campaign_id: str, inputs_digest: str) -> bool:
    """Whether this campaign was already found to have nothing new, for exactly these inputs."""
    from .beta_policy_screening_coordinator import _root

    try:
        value = json.loads(_every_pair_path(_root(queue_root), campaign_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(value, dict) and value.get("schema_version") == EVERY_PAIR_RECORD_SCHEMA
            and value.get("inputs_digest") == inputs_digest
            and value.get("record_digest") == canonical_digest(value, digest_field="record_digest"))


def record_every_pair_tested(queue_root: str | Path | None, campaign_id: str, inputs_digest: str) -> None:
    """Remember that planning this campaign found every pair tested; moves no money."""
    from .beta_policy_screening_coordinator import _root, _write

    root = _root(queue_root)
    value = {"schema_version": EVERY_PAIR_RECORD_SCHEMA, "campaign_id": campaign_id, "inputs_digest": inputs_digest,
             "outcome": EVERY_PAIR_TESTED, "record_digest": ""}
    value["record_digest"] = canonical_digest(value, digest_field="record_digest")
    (root / RECORD_DIRNAME).mkdir(exist_ok=True, mode=0o700)
    _write(_every_pair_path(root, campaign_id), value)


def record_tested_pairs(queue_root: str | Path | None, site_id: str) -> dict[str, Any]:
    """Write (and return) the job's derived record of finished pairs. Idempotent; moves no money."""
    from .beta_policy_screening_coordinator import _root, _write

    root = _root(queue_root)
    pairs = site_pairs(root, site_id)
    record = {"schema_version": RECORD_SCHEMA, "site_id": site_id,
              "finished": [pairs["finished"][key] for key in sorted(pairs["finished"])],
              "in_flight": [pairs["in_flight"][key] for key in sorted(pairs["in_flight"])], "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    directory = root / RECORD_DIRNAME
    directory.mkdir(exist_ok=True, mode=0o700)
    _write(directory / f"{site_id}.json", record)
    return record


__all__ = ["CANDIDATE_ID_BASIS", "CHECKPOINT_BASIS", "EVERY_PAIR_TESTED", "LEGACY_BASIS", "LOCK_ORDER", "PLANNING_LOCK_PREFIX", "RECORD_SCHEMA",
           "campaign_catalog_rows", "candidate_keys", "cells_digest", "every_pair_inputs_digest",
           "every_pair_tested_recorded", "excluded_pair_keys", "finished", "finished_pair_keys", "identity_pair_key",
           "in_flight", "pair_key", "plan_row_key", "planning_exclusions", "planning_lock_name", "policy_identity",
           "record_every_pair_tested", "record_tested_pairs", "site_pairs", "site_planning_lock"]
