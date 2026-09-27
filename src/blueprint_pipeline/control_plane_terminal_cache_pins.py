"""Release storage pins once evidence proves they protect nothing.

The collector previously retained already-archived runs for the pin's full
30-day TTL. This reconciliation changes only the cache ledger; the normal
collector separately rechecks references and removes reproducible directories.

Two proofs always apply to an activation pin: its run is archived behind a
verified pointer (``archived_run``), or sealed cold without a result registry
(``sealed_cold_run``). The extended proofs are evaluated on every tick but
release a pin only with the owner's opt-in,
``BLUEPRINT_CONTROL_PLANE_GC_EXTENDED_PIN_PROOFS=1``; until then their
candidates are listed with ``"enabled": false``:

* ``sealed_registry_run``: an activation pin whose run carries a result
  registry the artifact store accepts as sealed (delivered completed_unqualified,
  blocked or cancelled, its closeout receipts intact), idle past the hot
  window, with no whole-run pointer. Whole-run offload never archives a
  registry run, so neither original proof could release one.
* ``activation_expired_unlaunched``: an activation pin with no run directory
  and no pointer under any of its evidence names in any evidence root, whose
  one sealed result in the activation queue says it was prepared more than a
  week and a day ago. Its mutation window has lapsed and launch re-validates
  the window, so it can never start. Without an activation queue root this
  proof is off.
* ``unconsumed_stale_pin``: a preparation or compilation pin that no live pin
  depends on, created more than a week and a day ago, whose paths are all
  ``cache``. Its content is reproducible and re-fetched by digest.

Every proof keeps the six-hour minimum pin age, the dependency closure (a pin
is released only when no queue row or process references any pin in it), and
a re-derivation at the mutation edge. The report names every live pin: as a
candidate with its ``proof``, or in ``kept`` with a typed reason. A candidate
whose references change at the mutation edge is kept too, as
``reference_changed``.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

from .control_plane_storage_pins import load_storage_pins, release_storage_pin, storage_pin_guard
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .completed_replay_cache_retention import active_reference
from .control_plane_evidence_offload import (
    DEFAULT_HOT_WINDOW_SECONDS, POINTER_SUFFIX, _has_result_registry, _terminal_receipt, _tree_snapshot,
)
from .control_plane_storage_roots import require_storage_class

# A pin may name the reproducible activation inputs (cache or work class) or the
# retained run directory itself (evidence_cold). Releasing a pin removes no
# bytes, so any of these is acceptable; hot evidence and state never are.
_PIN_PATH_CLASSES = ("cache", "work", "evidence_cold")
MINIMUM_PIN_AGE_SECONDS = 6 * 3600
#: A shared mutation window is valid for at most a week and launch re-validates
#: it, so a day past that nothing it released can still be consumed.
MAXIMUM_MUTATION_WINDOW_SECONDS = 604_800
LAPSE_SECONDS = MAXIMUM_MUTATION_WINDOW_SECONDS + 86_400
#: The queue root whose ``results`` the activation worker seals; it is also a queue root.
ACTIVATION_QUEUE_NAME = "task-evaluation-launch-activations"
#: ``task_evaluation_launch_activation_queue.RESULT_SCHEMA_VERSION``, without importing its contracts.
ACTIVATION_RESULT_SCHEMA_VERSION = "task_evaluation_launch_activation_result.v1"
#: The statuses the activation worker writes, only ever for an activation it prepared,
#: the one terminal state in which it pins the activation.
PREPARED_ACTIVATION_STATUSES = frozenset({
    "policy_campaign_queue_materialized_no_execution",
    "profile_authority_materialized_no_execution",
})


def _read(path):
    if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
            or path.stat().st_size > 16 * 1024**2):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _pin_path_allowed(classifier, path, classes=_PIN_PATH_CLASSES):
    last = None
    for expected in classes:
        try:
            classifier(str(path), expected=expected, code="terminal_cache_pin_path_class_invalid")
            return
        except ValueError as exc:
            last = exc
    raise last


def _evidence_names(owner):
    """The run directories an activation owns: its id, and ``<id>-launch`` for a website auto activation."""

    return (owner, owner + "-launch") if owner.endswith("-activation-auto") else (owner,)


def _closed_proof(pin, evidence_roots, *, hot_window_seconds=DEFAULT_HOT_WINDOW_SECONDS, now=None):
    """Proof that the run this activation pin protects no longer needs the pin.

    Either the run has been archived behind a verified pointer, or the run
    directory itself is sealed by a terminal receipt, idle past the hot window,
    and carries no result registry. The second case exists because the
    collector will not offload a pinned run and used to release the pin only
    after offload: a launch that ended blocked or cancelled without releasing
    its own pin kept its evidence on disk indefinitely.
    """

    owner, kind = pin["owner_id"], pin["kind"]
    if kind != "activation":
        return None
    for root in evidence_roots:
        root = Path(root)
        for evidence_name in _evidence_names(owner):
            directory = root / evidence_name
            if (now is not None and directory.is_dir() and not directory.is_symlink()
                    and not (root / (evidence_name + POINTER_SUFFIX)).exists()):
                receipt = _terminal_receipt(directory)
                if receipt is None or _has_result_registry(directory):
                    continue
                latest, size, count = _tree_snapshot(directory)
                if now - latest < hot_window_seconds:
                    continue
                return {"kind": "sealed_cold_run", "path": str(directory), "terminal_receipt": receipt,
                        "latest_mtime_epoch": latest, "size_bytes": size, "file_count": count}
            path = root / (evidence_name + ".offloaded.v1.json")
            value = _read(path)
            if (value is None or directory.exists()
                    or value.get("schema_version") != "control_plane_evidence_offload_pointer.v1"
                    or value.get("pointer_digest") != canonical_digest(value, digest_field="pointer_digest")
                    or value.get("status") != "offloaded" or value.get("directory") != evidence_name
                    or value.get("evidence_deleted") is not False
                    or not str(value.get("uri", "")).startswith("s3://blueprint-task-evaluation-artifacts-prod/")
                    or value.get("terminal_receipt") not in {"dispatch_receipt.json", "launch_receipt.json", "abandoned_idle"}
                    or type(value.get("size_bytes")) is not int or value["size_bytes"] <= 0):
                continue
            return {"kind": "archived_run", "path": str(path), "pointer_digest": value["pointer_digest"],
                    "terminal_receipt": value["terminal_receipt"], "archive_digest": value["digest"]}
    return None


def _paths_classify(classifier, paths, classes):
    try:
        for path in paths:
            _pin_path_allowed(classifier, path, classes)
    except ValueError:
        return False
    return True


def _unconsumed_stale_pin(pin, live_pins, *, classifier, now, **_context):
    """A preparation or compilation that nothing consumes and that has outlived every mutation window."""

    identity = (pin["kind"], pin["owner_id"])
    if any((row.get("kind"), row.get("owner_id")) == identity
           for other in live_pins.values() for row in other.get("depends_on") or []):
        return None, "depended_on"
    if now - pin["created_at_epoch"] < LAPSE_SECONDS:
        return None, "pin_not_stale"
    if not _paths_classify(classifier, pin["paths"], ("cache",)):
        return None, "path_class_invalid"
    return {"kind": "unconsumed_stale_pin", "created_at_epoch": pin["created_at_epoch"]}, None


def _present(path):
    """Whether anything is at ``path``, or None when that cannot be told: an unreadable root proves nothing."""

    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return True


def _sealed_registry_run(directory, *, hot_window_seconds, now):
    """A run whose result registry the artifact store accepts as sealed, the registry idle past the hot window."""

    if directory.is_symlink() or not directory.is_dir():
        return None, "run_path_unsafe"
    if not _has_result_registry(directory):
        # The sealed-cold-run proof already declined it: unsealed, or sealed and still hot.
        return None, "run_not_sealed" if _terminal_receipt(directory) is None else "run_hot"
    from .task_evaluation_result_artifact_store import _sealed_registry
    try:
        registry, registry_path, _raw = _sealed_registry(directory.resolve())
        idle_since = registry_path.stat().st_mtime
        # The delivery the registry seals, re-read only to name its status.
        delivery = _read(registry_path.parent / "delivery.json")
        sealed = registry["delivery_digest"] == cross_runtime_canonical_digest(delivery, digest_field="delivery_digest")
    except Exception:  # noqa: BLE001 - whatever the store will not accept as sealed keeps the pin
        return None, "registry_unsealed"
    if not sealed:
        return None, "registry_unsealed"
    if now - idle_since < hot_window_seconds:
        return None, "registry_hot"
    return {"kind": "sealed_registry_run", "run_directory": directory.name, "registry_digest": registry["registry_digest"],
            "delivery_status": delivery["result_status"], "registry_mtime_epoch": idle_since}, None


def activation_queue_root_of(queue_roots):
    """The activation queue: the one configured queue root named ``task-evaluation-launch-activations``, else None."""

    roots = {Path(root).expanduser() for root in queue_roots}
    matches = [root for root in roots if root.name == ACTIVATION_QUEUE_NAME]
    return matches[0] if len(matches) == 1 else None


def _expired_unlaunched(owner, activation_queue_root, *, now):
    """A prepared activation whose every mutation window has lapsed; the caller found no run of it.

    The worker names a result for the activation's queue envelope,
    ``<activation id>-<request digest>.json``, and an activation id has one request.
    """

    if activation_queue_root is None:
        return None, "activation_queue_unconfigured"
    results = Path(activation_queue_root) / "results"
    pattern = re.compile(re.escape(owner) + r"-[0-9a-f]{64}\.json")
    try:
        if results.is_symlink() or not results.is_dir():
            return None, "activation_queue_unavailable"
        names = sorted(entry.name for entry in os.scandir(results) if pattern.fullmatch(entry.name))
    except OSError:
        return None, "activation_queue_unavailable"
    if len(names) != 1:
        return None, "activation_result_ambiguous" if names else "activation_result_missing"
    path = results / names[0]
    value = _read(path)
    if (value is None or value.get("schema_version") != ACTIVATION_RESULT_SCHEMA_VERSION
            or value.get("activation_id") != owner
            or value.get("result_digest") != canonical_digest(value, digest_field="result_digest")):
        return None, "activation_result_invalid"
    status = value.get("status")
    if not isinstance(status, str) or status not in PREPARED_ACTIVATION_STATUSES:
        return None, "activation_result_not_prepared"
    written = path.lstat().st_mtime
    if now - written < LAPSE_SECONDS:
        return None, "activation_result_not_stale"
    return {"kind": "activation_expired_unlaunched", "result_name": path.name, "result_digest": value["result_digest"],
            "result_status": status, "result_mtime_epoch": written}, None


def _activation_proof(pin, live_pins, *, evidence_roots, activation_queue_root, hot_window_seconds, classifier, now,
                      **_context):
    """Every run under the activation's evidence names is a sealed registry run, or it never launched.

    Any whole-run pointer keeps the pin: the archived-run proof already declined
    it. A root that is linked, or where a name cannot be looked up, proves
    nothing; a configured root that does not exist holds no run.
    """

    roots = [Path(root) for root in evidence_roots]
    if not roots or any(root.is_symlink() for root in roots):
        return None, "evidence_root_unavailable"
    runs = []
    for root in roots:
        for name in _evidence_names(pin["owner_id"]):
            directory, pointer = _present(root / name), _present(root / (name + POINTER_SUFFIX))
            if directory is None or pointer is None:
                return None, "evidence_root_unavailable"
            if pointer:
                return None, "run_pointer_present"
            if directory:
                runs.append(root / name)
    proof = None
    if not runs:
        proof, reason = _expired_unlaunched(pin["owner_id"], activation_queue_root, now=now)
        if proof is None:
            return None, reason
    for run in runs:
        found, reason = _sealed_registry_run(run, hot_window_seconds=hot_window_seconds, now=now)
        if found is None:
            return None, reason
        proof = proof or found
    if not _paths_classify(classifier, pin["paths"], _PIN_PATH_CLASSES):
        return None, "path_class_invalid"
    return proof, None


_EXTENDED_PROOFS = {"preparation": _unconsumed_stale_pin, "compilation": _unconsumed_stale_pin,
                    "activation": _activation_proof}


def _extended_proof(pin, live_pins, **context):
    """``(proof, None)`` when an extended proof holds for ``pin``, else ``(None, reason)``. It only reads.

    The six-hour minimum pin age is checked first, as the original proofs check it.
    """

    created = pin.get("created_at_epoch")
    if type(created) not in (int, float):
        return None, "pin_invalid"
    if context["now"] - created < MINIMUM_PIN_AGE_SECONDS:
        return None, "pin_young"
    proof = _EXTENDED_PROOFS.get(pin["kind"])
    return (None, "no_proof") if proof is None else proof(pin, live_pins, **context)


def _derive(pin, live_pins, context):
    """``_extended_proof`` and the type of any error it raised: one unreadable pin never costs the tick."""

    try:
        return (*_extended_proof(pin, live_pins, **context), None)
    except Exception as exc:  # noqa: BLE001 - the report keeps the type, never a message with a path
        return None, "proof_error", type(exc).__name__


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


def _referenced(closure, queue_text, reference_checker):
    return any(p["owner_id"] in queue_text or any(reference_checker(Path(path)) for path in p["paths"])
               for p in closure.values())


def _closure_reason(identity, closure, pins, queue_text, reference_checker):
    """Why the closure keeps the pin: a queue row or process references it, or another pin depends on it."""

    if _referenced(closure, queue_text, reference_checker):
        return "active_reference"
    if any(any((d["kind"], d["owner_id"]) == identity for d in other.get("depends_on", []))
           for key, other in pins.items() if key not in closure):
        return "depended_on"
    return None


def reconcile_terminal_cache_pins(*, pins_root, queue_roots, evidence_roots, now, apply=False,
                                  reference_checker=active_reference, classifier=require_storage_class,
                                  hot_window_seconds=DEFAULT_HOT_WINDOW_SECONDS, extended_proofs_enabled=False,
                                  activation_queue_root=None):
    """Plan, and with ``apply`` release, every live pin a proof closes.

    ``enabled`` is the extended proofs' opt-in; the original proofs always apply.
    ``activation_queue_root`` is where the activation worker seals its results
    (``activation_queue_root_of(queue_roots)``); without it an unlaunched activation is kept.
    Each candidate carries its ``proof`` and whether it is ``enabled``, each kept
    pin a typed ``reason`` (and ``error_type`` for ``proof_error``), and
    ``released_count_by_kind`` counts every pin a release receipt lists, its
    dependencies included. Nothing here removes a byte.
    """

    from .control_plane_storage_gc import _queue_reference_text
    pins_root = Path(pins_root)
    for root in evidence_roots:
        classifier(str(root), expected="evidence_cold", code="terminal_cache_pin_evidence_root_invalid")
    pins = _live_pins(pins_root, now)
    queue_text = _queue_reference_text(queue_roots)
    context = {"classifier": classifier, "now": now, "evidence_roots": evidence_roots,
               "hot_window_seconds": hot_window_seconds, "activation_queue_root": activation_queue_root}
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
            reason = _closure_reason(identity, closure, pins, queue_text, reference_checker)
        else:
            proof, reason, error = _derive(pin, pins, context)
            if proof is not None:
                try:
                    closure = _closure(identity, pins)
                    reason = _closure_reason(identity, closure, pins, queue_text, reference_checker)
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
            fresh = _queue_reference_text(queue_roots)
            if (proof != _closed_proof(pin, evidence_roots, hot_window_seconds=hot_window_seconds, now=now)
                    or _referenced(closure, fresh, reference_checker)):
                kept.append({**candidate, "reason": "reference_changed"})
                continue
            released.append(release_storage_pin(pins_root=pins_root, kind=pin["kind"],
                                                 owner_id=pin["owner_id"], now=lambda: now))
            continue
        # An extended proof is re-derived under the lock producers hold to publish a pin,
        # reading queue rows before the ledger: the stale-pin proof rests on the ledger,
        # and a consumer arriving meanwhile shows up in one or the other.
        with storage_pin_guard(pins_root, exclusive=True):
            fresh = _queue_reference_text(queue_roots)
            if (proof != _derive(pin, _live_pins(pins_root, now), context)[0]
                    or _referenced(closure, fresh, reference_checker)):
                kept.append({**candidate, "reason": "reference_changed"})
                continue
            released.append(release_storage_pin(pins_root=pins_root, kind=pin["kind"],
                                                 owner_id=pin["owner_id"], now=lambda: now))
    by_kind = Counter(row["kind"] for receipt in released for row in receipt["released"])
    return {"schema_version": "control_plane_terminal_cache_pin_reconciliation.v1",
        "status": "applied" if apply else "dry_run", "enabled": bool(extended_proofs_enabled),
        "candidates": candidates, "candidate_count": len(candidates),
        "candidate_count_by_proof": dict(sorted(Counter(row["proof"]["kind"] for row in candidates).items())),
        "released": released, "released_count": sum(by_kind.values()),
        "released_count_by_kind": dict(sorted(by_kind.items())),
        "kept": kept, "retained_counts": dict(sorted(Counter(row["reason"] for row in kept).items())),
        "cache_or_evidence_bytes_removed": False}
