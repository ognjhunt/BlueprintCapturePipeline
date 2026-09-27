"""Release storage pins once evidence proves they protect nothing.

The collector previously retained already-archived runs for the pin's full
30-day TTL. This reconciliation changes only the cache ledger; the normal
collector separately rechecks references and removes reproducible directories.

Two proofs always apply to an activation pin: every run it owns that exists is
archived behind a verified pointer (``archived_run``), or sealed cold without a
result registry (``sealed_cold_run``). The extended proofs are evaluated on every tick but
release a pin only with the owner's opt-in,
``BLUEPRINT_CONTROL_PLANE_GC_EXTENDED_PIN_PROOFS=1``; until then their
candidates are listed with ``"enabled": false``:

* ``sealed_registry_run``: an activation pin whose every run carries its
  terminal receipt and a result registry the artifact store accepts as sealed
  (delivered completed_unqualified, blocked or cancelled, its closeout receipts
  intact), idle past the hot window, with no whole-run pointer. Whole-run
  offload never archives a registry run, so neither original proof could
  release one.
* ``activation_expired_unlaunched``: a profile-authority activation pin with no
  run directory and no pointer under any of its evidence names in any evidence
  root, whose one sealed result in the activation queue says it was prepared
  more than a week and a day ago, and whose standing authorization expired more
  than a day ago. A policy-campaign activation is out of scope
  (``policy_campaign_activation_out_of_scope``): it publishes no standing
  authorization and dispatches through the policy canary queue on the scene
  execution window, and the canary dispatcher releases its pin on completion. The mutation window (a week at most) has lapsed, and launch
  admission checks that authorization, which the activation request dates
  with no maximum. A launch id the WebApp or an operator chooses names no
  directory a search could guess, so the proof also needs positive evidence:
  no record under ``<standing authorization dir>/consumed/<profile id>/`` and
  no launch queue row, in any state, naming the activation or its profile.
  Without the activation queue, the launch queue or the standing authorization
  directory this proof is off. For a profile without the one-use standing
  authorization requirement, an operator's per-launch handshake can still admit
  a launch after the authorization lapsed; if that happens after the pin was
  released, the launch fails its input verification rather than using missing
  inputs, which re-preparing recovers. The proof accepts that risk only after
  both the window and the authorization lapsed and neither record exists.
* ``unconsumed_stale_pin``: a preparation or compilation pin that no live pin
  depends on, created more than a week and a day ago, whose paths are all
  ``cache``, and whose preparation no activation can take any more: its sealed
  envelope sits in ``materialized/`` bound to a release other than the running
  one, or in ``blocked/``. The activation worker verifies materialized inputs
  and never re-fetches them, so age alone proves nothing: a materialized
  preparation waits for its activation intent with no age limit.

The extended proofs live in ``control_plane_pin_proofs``, which only reads;
this module decides and mutates the ledger. Both activation proofs look for a
run under every name an activation can launch as (``_launch_evidence_names``):
its id, ``<id>-launch``, and the bounded launch id the launch paths derive for a
long id, with their own functions.
Every proof keeps the six-hour minimum pin age, the dependency closure (a pin
is released only when no queue row or process references any pin in it), and
a re-derivation at the mutation edge. A dependency a release takes with it is
covered by its parent's closure checks and by the ledger's ``_still_needed``
(no other live pin depends on it), not by a proof of its own: the original
proofs have always released an activation's preparation and compilation that
way, and a preparation's lifecycle ends with the activation that consumed it. The extended proofs read queues
strictly: they also count a row parked in a queue state that will still run,
such as a preparation awaiting its source preparation, and a row they cannot
read (linked, oversized, not UTF-8 or unreadable) keeps their candidates as
``queue_unreadable``, where the original proofs skip it as they always did. The report names every live pin: as a
candidate with its ``proof``, or in ``kept`` with a typed reason. A candidate
whose references change at the mutation edge is kept too, as
``reference_changed``.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from .control_plane_pin_proofs import (
    MINIMUM_PIN_AGE_SECONDS, _evidence_names, _pin_path_allowed, _present, _read, extended_proof,
    launch_queue_snapshot,
)
from .control_plane_storage_pins import depends_on, load_storage_pins, release_storage_pin, storage_pin_guard
from .control_plane_storage_references import QueueReferenceUnreadable, queue_reference_text
from .decision_evidence_contracts import canonical_digest
from . import completed_replay_cache_retention as retention
from .control_plane_evidence_offload import (
    DEFAULT_HOT_WINDOW_SECONDS, POINTER_SUFFIX, _has_result_registry, _terminal_receipt, _tree_snapshot,
)
from .control_plane_replay_cache_gc import _truthy_setting
from .control_plane_storage_roots import require_storage_class

EXTENDED_PIN_PROOFS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_EXTENDED_PIN_PROOFS"
EXTENDED_PIN_PROOFS_INVALID = "extended_pin_proofs_setting_invalid"
#: Report rows per list; production holds about 142 live pins. Counts always cover every pin.
_MAX_ROWS = 200


def extended_pin_proofs_setting(environ=os.environ):
    """Whether the extended proofs may release pins, and an alert when the setting is invalid.

    Its own opt-in, parsed exactly like the storage GC's others: an invalid value
    only lists candidates and alerts, and it never follows another opt-in.
    """

    return _truthy_setting(environ, EXTENDED_PIN_PROOFS_ENV, EXTENDED_PIN_PROOFS_INVALID)


def _closed_proof(pin, evidence_roots, *, hot_window_seconds=DEFAULT_HOT_WINDOW_SECONDS, now=None):
    """Proof that the runs this activation pin protects no longer need the pin.

    Every name the activation owns that exists, in every evidence root, must be
    closed: archived behind a verified pointer, or its run directory sealed by a
    terminal receipt, idle past the hot window, and carrying no result registry.
    The proof is the first closed name's. The second case exists because the
    collector will not offload a pinned run and used to release the pin only
    after offload: a launch that ended blocked or cancelled without releasing
    its own pin kept its evidence on disk indefinitely. Until 10c this stopped at
    the first closed name, so a website activation's sealed own directory could
    release the pin while its ``<id>-launch`` run was still going.
    """

    owner, kind = pin["owner_id"], pin["kind"]
    if kind != "activation":
        return None
    proof = None
    for root in evidence_roots:
        root = Path(root)
        for evidence_name in _evidence_names(owner):
            directory = root / evidence_name
            pointer = root / (evidence_name + POINTER_SUFFIX)
            present = (_present(directory), _present(pointer))
            if present == (False, False):
                continue
            found = None if None in present else (
                _sealed_cold_run(directory, pointer, hot_window_seconds=hot_window_seconds, now=now)
                or _archived_run(evidence_name, directory, pointer))
            if found is None:
                return None
            proof = proof or found
    return proof


def _sealed_cold_run(directory, pointer, *, hot_window_seconds, now):
    if now is None or not directory.is_dir() or directory.is_symlink() or pointer.exists():
        return None
    receipt = _terminal_receipt(directory)
    if receipt is None or _has_result_registry(directory):
        return None
    latest, size, count = _tree_snapshot(directory)
    if now - latest < hot_window_seconds:
        return None
    return {"kind": "sealed_cold_run", "path": str(directory), "terminal_receipt": receipt,
            "latest_mtime_epoch": latest, "size_bytes": size, "file_count": count}


def _archived_run(evidence_name, directory, pointer):
    value = _read(pointer)
    if (value is None or directory.exists()
            or value.get("schema_version") != "control_plane_evidence_offload_pointer.v1"
            or value.get("pointer_digest") != canonical_digest(value, digest_field="pointer_digest")
            or value.get("status") != "offloaded" or value.get("directory") != evidence_name
            or value.get("evidence_deleted") is not False
            or not str(value.get("uri", "")).startswith("s3://blueprint-task-evaluation-artifacts-prod/")
            or value.get("terminal_receipt") not in {"dispatch_receipt.json", "launch_receipt.json", "abandoned_idle"}
            or type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0):
        return None
    return {"kind": "archived_run", "path": str(pointer), "pointer_digest": value["pointer_digest"],
            "terminal_receipt": value["terminal_receipt"], "archive_digest": value["digest"]}


def _derive(pin, live_pins, context):
    """``_extended_proof`` and the type of any error it raised: one unreadable pin never costs the tick."""

    try:
        return (*extended_proof(pin, live_pins, **context), None)
    except Exception as exc:  # noqa: BLE001 - the report keeps the type, never a message with a path
        return None, "proof_error", type(exc).__name__


def _live_queue_text(queue_roots):
    """Every queue row that will still run, read strictly: ``(text, None)``, or ``(None, "queue_unreadable")``.

    Pending and processing rows of every root, and the rows a queue parks in a
    state that will still run (``LIVE_QUEUE_STATES``: a preparation awaiting its
    source or capacity), each read twice so a row moving between states mid-read
    is seen. Only the extended proofs read this way; a row that cannot be read
    keeps their candidates and nothing else.
    """

    from .control_plane_release_leases import LIVE_QUEUE_STATES

    try:
        return "\n".join(queue_reference_text(queue_roots, states=LIVE_QUEUE_STATES, strict=True)
                         for _read_pass in range(2)), None
    except QueueReferenceUnreadable:
        return None, "queue_unreadable"


def _live_pins(pins_root, now):
    return {(p["kind"], p["owner_id"]): p for p in load_storage_pins(pins_root, now=lambda: now) if p["status"] == "live"}


def _closure(identity, pins):
    """The pin and every live pin it depends on, transitively: what releasing it can release."""

    closure, pending = {}, [identity]
    while pending:
        key = pending.pop()
        if key in closure or key not in pins:
            continue
        closure[key] = pins[key]
        pending.extend((d["kind"], d["owner_id"]) for d in pins[key].get("depends_on", []))
    return closure


def _partial_release(pins_root, now, identity, closure, error_type):
    """What the ledger holds after a release raised: a partial release receipt when it recorded the pin, else {}.

    ``release_storage_pin`` writes the pin's release and then walks its
    dependencies, so it can fail having recorded the pin. The receipt lists
    every pin of the closure the ledger now shows released.
    """

    try:
        status = {(p["kind"], p["owner_id"]): p["status"] for p in load_storage_pins(pins_root, now=lambda: now)}
    except Exception:  # noqa: BLE001 - a ledger that cannot be read is reported as the failure it is
        return {}
    if status.get(identity) != "released":
        return {}
    return {"schema_version": "control_plane_storage_pin_release.v1", "kind": identity[0], "owner_id": identity[1],
            "status": "release_partial", "error_type": error_type,
            "released": [{"kind": key[0], "owner_id": key[1]} for key in closure if status.get(key) == "released"]}


def _process_checker(reference_checker):
    """``reference_checker`` as given, or one sweep of the process table, taken when first asked and reused.

    A pass checks every path of every candidate's closure; one sweep answers for
    all of them as a sweep per path would (``process_reference_index``).
    """

    if reference_checker is not None:
        return reference_checker
    swept = []

    def checker(path):
        if not swept:
            swept.append(retention.process_reference_index())
        return swept[0](path)

    return checker


def _referenced(closure, queue_text, reference_checker):
    return any(p["owner_id"] in queue_text or any(reference_checker(Path(path)) for path in p["paths"])
               for p in closure.values())


def _closure_reason(identity, closure, pins, queue_text, reference_checker):
    """Why the closure keeps the pin: a queue row or process references it, or another pin depends on it."""

    if _referenced(closure, queue_text, reference_checker):
        return "active_reference"
    if any(depends_on(other, *identity) for key, other in pins.items() if key not in closure):
        return "depended_on"
    return None


def reconcile_terminal_cache_pins(*, pins_root, queue_roots, evidence_roots, now, apply=False,
                                  reference_checker=None, classifier=require_storage_class,
                                  hot_window_seconds=DEFAULT_HOT_WINDOW_SECONDS, extended_proofs_enabled=False,
                                  activation_queue_root=None, preparation_queue_root=None, running_commit="",
                                  launch_queue_root=None, standing_authorization_dir=None):
    """Plan, and with ``apply`` release, every live pin a proof closes.

    ``enabled`` is the extended proofs' opt-in; the original proofs always apply.
    ``activation_queue_root`` is where the activation worker seals its results
    (``activation_queue_root_of(queue_roots)``); without it an unlaunched activation is kept.
    ``preparation_queue_root`` (``preparation_queue_root_of(queue_roots)``) and the
    ``running_commit`` tell whether an activation can still take a preparation;
    without them a stale preparation or compilation is kept. ``launch_queue_root``
    (``launch_queue_root_of(queue_roots)``) and ``standing_authorization_dir``, where
    launch admission records consumed authorizations, hold the evidence of a launch
    under any id; without them an unlaunched activation is kept.
    ``reference_checker`` answers whether a process still reads a path; by
    default planning sweeps the process table once, and each release sweeps it
    again. The launch queue is read once for planning and once for the releases.
    Each candidate carries its ``proof`` and whether it is ``enabled``, each kept
    pin a typed ``reason`` (and ``error_type`` for ``proof_error``), and
    ``released_count_by_kind`` counts every pin a release receipt lists, its
    dependencies included. Nothing here removes a byte.
    """

    pins_root = Path(pins_root)
    for root in evidence_roots:
        classifier(str(root), expected="evidence_cold", code="terminal_cache_pin_evidence_root_invalid")
    pins = _live_pins(pins_root, now)
    queue_text = queue_reference_text(queue_roots)
    live_queue_text, queue_unreadable = _live_queue_text(queue_roots)
    planning_checker = _process_checker(reference_checker)
    context = {"classifier": classifier, "now": now, "evidence_roots": evidence_roots,
               "hot_window_seconds": hot_window_seconds, "activation_queue_root": activation_queue_root,
               "preparation_queue_root": preparation_queue_root, "running_commit": running_commit,
               "launch_queue_root": launch_queue_root, "standing_authorization_dir": standing_authorization_dir,
               "launch_queue": launch_queue_snapshot(launch_queue_root)}
    # The releases re-derive their proofs against a launch queue read again, once, after planning.
    edge_context = {**context, "launch_queue": launch_queue_snapshot(launch_queue_root)}
    candidates, kept, released = [], [], []
    for identity, pin in pins.items():
        row = {"kind": pin["kind"], "owner_id": pin["owner_id"]}
        proof = _closed_proof(pin, evidence_roots, hot_window_seconds=hot_window_seconds, now=now)
        original = proof is not None
        if original:
            if now - pin["created_at_epoch"] < MINIMUM_PIN_AGE_SECONDS:
                kept.append({**row, "reason": "pin_young"})
                continue
            for path in pin["paths"]:
                _pin_path_allowed(classifier, path)
            # Check the entire dependency closure before releasing a parent pin.
            closure = _closure(identity, pins)
            reason = _closure_reason(identity, closure, pins, queue_text, planning_checker)
        else:
            proof, reason, error = _derive(pin, pins, context)
            if proof is not None and queue_unreadable is not None:
                reason = queue_unreadable
            elif proof is not None:
                try:
                    closure = _closure(identity, pins)
                    reason = _closure_reason(identity, closure, pins, live_queue_text, planning_checker)
                except Exception as exc:  # noqa: BLE001 - an extended candidate never costs the original proofs
                    reason, error = "proof_error", type(exc).__name__
            if error is not None:
                kept.append({**row, "reason": reason, "error_type": error})
                continue
        if reason is not None:
            kept.append({**row, "reason": reason})
            continue
        candidate = {**row, "proof": proof, "enabled": original or bool(extended_proofs_enabled)}
        candidates.append(candidate)
        if not (apply and candidate["enabled"]):
            continue
        if original:
            # Re-read live queue references and the proof at the mutation edge.
            fresh = queue_reference_text(queue_roots)
            if (proof != _closed_proof(pin, evidence_roots, hot_window_seconds=hot_window_seconds, now=now)
                    or _referenced(closure, fresh, _process_checker(reference_checker))):
                kept.append({**candidate, "reason": "reference_changed"})
                continue
            released.append(release_storage_pin(pins_root=pins_root, kind=pin["kind"],
                                                 owner_id=pin["owner_id"], now=lambda: now))
            continue
        # An extended proof is re-derived before the release: queue rows first, then the
        # ledger, the proof and a fresh sweep of the processes, all outside the pin lock, which
        # blocks every producer. Under the lock only the ledger is re-read, so a pin a
        # consumer published meanwhile shows up there if its queue row no longer did. A
        # failure keeps this pin, with its error type, and costs no other.
        stage = "proof_error"
        try:
            fresh, fresh_unreadable = _live_queue_text(queue_roots)
            if fresh_unreadable is not None:
                kept.append({**candidate, "reason": fresh_unreadable})
                continue
            fresh_proof, fresh_reason, fresh_error = _derive(pin, _live_pins(pins_root, now), edge_context)
            if fresh_proof is None:
                # The fresh derivation says why the proof no longer holds.
                kept.append({**candidate, "reason": fresh_reason,
                             **({"error_type": fresh_error} if fresh_error else {})})
                continue
            if fresh_proof != proof or _referenced(closure, fresh, _process_checker(reference_checker)):
                kept.append({**candidate, "reason": "reference_changed"})
                continue
            stage = "release_failed"
            with storage_pin_guard(pins_root, exclusive=True):
                if any(depends_on(other, *identity) for other in _live_pins(pins_root, now).values()):
                    kept.append({**candidate, "reason": "depended_on"})
                    continue
                released.append(release_storage_pin(pins_root=pins_root, kind=pin["kind"],
                                                     owner_id=pin["owner_id"], now=lambda: now))
        except Exception as exc:  # noqa: BLE001 - one failed check or release never costs the tick or its report
            partial = _partial_release(pins_root, now, identity, closure, type(exc).__name__)
            if partial:
                released.append(partial)
            else:
                kept.append({**candidate, "reason": stage, "error_type": type(exc).__name__})
    # A pin a later release took with it (a dependency no live pin needed any more) was released, not kept.
    released_pins = {(row["kind"], row["owner_id"]) for receipt in released for row in receipt["released"]}
    kept = [row for row in kept if (row["kind"], row["owner_id"]) not in released_pins]
    by_kind = Counter(row["kind"] for receipt in released for row in receipt["released"])
    return {"schema_version": "control_plane_terminal_cache_pin_reconciliation.v1",
        "status": "applied" if apply else "dry_run", "enabled": bool(extended_proofs_enabled),
        "candidates": candidates[:_MAX_ROWS], "omitted_candidates_count": max(0, len(candidates) - _MAX_ROWS),
        "candidate_count": len(candidates),
        "candidate_count_by_proof": dict(sorted(Counter(row["proof"]["kind"] for row in candidates).items())),
        "released": released, "released_count": sum(by_kind.values()),
        "released_count_by_kind": dict(sorted(by_kind.items())),
        "kept": kept[:_MAX_ROWS], "omitted_kept_count": max(0, len(kept) - _MAX_ROWS),
        "retained_counts": dict(sorted(Counter(row["reason"] for row in kept).items())),
        "cache_or_evidence_bytes_removed": False}
