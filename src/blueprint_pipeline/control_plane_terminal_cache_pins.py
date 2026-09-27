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

* ``sealed_registry_run``: an activation pin whose every run carries its
  terminal receipt and a result registry the artifact store accepts as sealed
  (delivered completed_unqualified, blocked or cancelled, its closeout receipts
  intact), idle past the hot window, with no whole-run pointer. Whole-run
  offload never archives a registry run, so neither original proof could
  release one.
* ``activation_expired_unlaunched``: an activation pin with no run directory
  and no pointer under any of its evidence names in any evidence root, whose
  one sealed result in the activation queue says it was prepared more than a
  week and a day ago, and whose standing authorization expired more than a
  day ago. The mutation window (a week at most) has lapsed, and launch
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

Both activation proofs look for a run under every name an activation can
launch as (``_launch_evidence_names``): its id, ``<id>-launch``, and the bounded
launch id the launch paths derive for a long id, with their own functions.
Every proof keeps the six-hour minimum pin age, the dependency closure (a pin
is released only when no queue row or process references any pin in it), and
a re-derivation at the mutation edge. The extended proofs also count a row
parked in a queue state that will still run, such as a preparation awaiting
its source preparation. The report names every live pin: as a
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
from .control_plane_replay_cache_gc import _truthy_setting
from .control_plane_storage_roots import require_storage_class

EXTENDED_PIN_PROOFS_ENV = "BLUEPRINT_CONTROL_PLANE_GC_EXTENDED_PIN_PROOFS"
EXTENDED_PIN_PROOFS_INVALID = "extended_pin_proofs_setting_invalid"
# A pin may name the reproducible activation inputs (cache or work class) or the
# retained run directory itself (evidence_cold). Releasing a pin removes no
# bytes, so any of these is acceptable; hot evidence and state never are.
_PIN_PATH_CLASSES = ("cache", "work", "evidence_cold")
MINIMUM_PIN_AGE_SECONDS = 6 * 3600
#: Report rows per list; production holds about 142 live pins. Counts always cover every pin.
_MAX_ROWS = 200
#: A shared mutation window is valid for at most a week; the proofs wait a day
#: past it, and past the activation's standing authorization, which launch
#: admission checks.
MAXIMUM_MUTATION_WINDOW_SECONDS = 604_800
LAPSE_GRACE_SECONDS = 86_400
LAPSE_SECONDS = MAXIMUM_MUTATION_WINDOW_SECONDS + LAPSE_GRACE_SECONDS
#: The queue root whose ``results`` the activation worker seals; it is also a queue root.
ACTIVATION_QUEUE_NAME = "task-evaluation-launch-activations"
#: The queue root launch requests are staged in, whoever chose their launch ids.
LAUNCH_QUEUE_NAME = "task-evaluation-launches"
#: An identifier that is safe as one path component.
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}")
#: The queue root whose ``materialized/`` envelopes the activation worker takes preparations from.
PREPARATION_QUEUE_NAME = "task-evaluation-launch-preparations"
#: ``task_evaluation_launch_preparation_queue.ENVELOPE_SCHEMA_VERSION``.
PREPARATION_ENVELOPE_SCHEMA_VERSION = "task_evaluation_launch_preparation_envelope.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
#: ``task_evaluation_launch_activation_queue.RESULT_SCHEMA_VERSION`` and ``ENVELOPE_SCHEMA_VERSION``.
ACTIVATION_RESULT_SCHEMA_VERSION = "task_evaluation_launch_activation_result.v1"
ACTIVATION_ENVELOPE_SCHEMA_VERSION = "task_evaluation_launch_activation_envelope.v1"
#: The statuses the activation worker writes, only ever for an activation it prepared,
#: the one terminal state in which it pins the activation.
PREPARED_ACTIVATION_STATUSES = frozenset({
    "policy_campaign_queue_materialized_no_execution",
    "profile_authority_materialized_no_execution",
})


def extended_pin_proofs_setting(environ=os.environ):
    """Whether the extended proofs may release pins, and an alert when the setting is invalid.

    Its own opt-in, parsed exactly like the storage GC's others: an invalid value
    only lists candidates and alerts, and it never follows another opt-in.
    """

    return _truthy_setting(environ, EXTENDED_PIN_PROOFS_ENV, EXTENDED_PIN_PROOFS_INVALID)


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


def _launch_evidence_names(owner):
    """Every run directory an activation may own, as the extended proofs look for it.

    The original proofs' names, ``<id>-launch`` for every activation (configured
    controls activations such as ``<run>-controls`` launch that way too), and the
    bounded launch id each launch path derives for an id too long for that: from
    the paths' own functions, so the two can never drift. A name this cannot
    derive fails the proof rather than being skipped.
    """

    from .task_evaluation_configured_controls_progression import _bounded_launch_id as controls_launch_id
    from .task_evaluation_scene_configuration_activation_automation import _bounded_launch_id as scene_launch_id

    names = (*_evidence_names(owner), owner + "-launch", controls_launch_id(owner), scene_launch_id(owner))
    return tuple(dict.fromkeys(names))


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


def _unconsumed_stale_pin(pin, live_pins, *, classifier, now, preparation_queue_root, running_commit, **_context):
    """A preparation or compilation that nothing consumes, that has outlived every mutation window, and that
    no activation can take any more.

    A compilation is named for its preparation, so both read the preparation's envelope.
    """

    identity = (pin["kind"], pin["owner_id"])
    if any((row.get("kind"), row.get("owner_id")) == identity
           for other in live_pins.values() for row in other.get("depends_on") or []):
        return None, "depended_on"
    if now - pin["created_at_epoch"] < LAPSE_SECONDS:
        return None, "pin_not_stale"
    if not _paths_classify(classifier, pin["paths"], ("cache",)):
        return None, "path_class_invalid"
    ended, reason = _unactivatable_preparation(pin["owner_id"], preparation_queue_root, running_commit)
    if ended is None:
        return None, reason
    state, commit = ended
    return {"kind": "unconsumed_stale_pin", "created_at_epoch": pin["created_at_epoch"],
            "preparation_state": state, "preparation_commit": commit}, None


def _unactivatable_preparation(preparation_id, preparation_queue_root, running_commit):
    """``(state, commit)`` when no activation can take the preparation any more, else ``(None, reason)``.

    The activation worker takes a preparation only from ``materialized/``, and only
    when its sealed envelope names the release the worker runs; it verifies the
    materialized inputs and never re-fetches them. So the preparation's one sealed
    envelope must sit in ``materialized/`` bound to another release, or in
    ``blocked/``. Anything else, including an envelope this queue does not hold, keeps
    the pin. A rollback to that release would make it activatable again.
    """

    if preparation_queue_root is None:
        return None, "preparation_queue_unconfigured"
    if not isinstance(running_commit, str) or _COMMIT.fullmatch(running_commit) is None:
        return None, "running_commit_unknown"
    pattern = re.compile(re.escape(preparation_id) + r"-[0-9a-f]{64}\.json")
    found = []
    for state in ("materialized", "blocked"):
        directory = Path(preparation_queue_root) / state
        try:
            if directory.is_symlink():
                return None, "preparation_queue_unavailable"
            if not directory.is_dir():
                continue
            found.extend((state, directory / entry.name) for entry in os.scandir(directory)
                         if pattern.fullmatch(entry.name))
        except OSError:
            return None, "preparation_queue_unavailable"
    if len(found) != 1:
        return None, "preparation_envelope_ambiguous" if found else "preparation_envelope_missing"
    state, path = found[0]
    envelope = _read(path)
    request = envelope.get("request") if envelope is not None else None
    commit = request.get("expected_production_commit") if isinstance(request, dict) else None
    if (not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None
            or envelope.get("schema_version") != PREPARATION_ENVELOPE_SCHEMA_VERSION
            or envelope.get("envelope_digest") != canonical_digest(envelope, digest_field="envelope_digest")
            or request.get("preparation_id") != preparation_id):
        return None, "preparation_envelope_invalid"
    if state == "materialized" and commit == running_commit:
        return None, "preparation_release_current"
    return (state, commit), None


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
    """A run with its terminal receipt whose result registry the store accepts as sealed, idle past the hot window.

    Without the receipt the canary dispatcher can still recover a stranded
    delivery after a deploy, re-reading the activation's launch set.
    """

    if directory.is_symlink() or not directory.is_dir():
        return None, "run_path_unsafe"
    receipt = _terminal_receipt(directory)
    if receipt is None:
        return None, "run_not_sealed"
    if not _has_result_registry(directory):
        # Not this proof's run; the original sealed-cold-run proof reads only the original names.
        return None, "run_hot" if now - _tree_snapshot(directory)[0] < hot_window_seconds else "run_without_registry"
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
    return {"kind": "sealed_registry_run", "run_directory": directory.name, "terminal_receipt": receipt,
            "registry_digest": registry["registry_digest"], "delivery_status": delivery["result_status"],
            "registry_mtime_epoch": idle_since}, None


def _queue_root_named(queue_roots, name):
    roots = {Path(root).expanduser() for root in queue_roots}
    matches = [root for root in roots if root.name == name]
    return matches[0] if len(matches) == 1 else None


def activation_queue_root_of(queue_roots):
    """The activation queue: the one configured queue root named ``task-evaluation-launch-activations``, else None."""

    return _queue_root_named(queue_roots, ACTIVATION_QUEUE_NAME)


def launch_queue_root_of(queue_roots):
    """The launch queue: the one configured queue root named ``task-evaluation-launches``, else None."""

    return _queue_root_named(queue_roots, LAUNCH_QUEUE_NAME)


def preparation_queue_root_of(queue_roots):
    """The preparation queue: the one configured queue root named ``task-evaluation-launch-preparations``, else None."""

    return _queue_root_named(queue_roots, PREPARATION_QUEUE_NAME)


def _expired_unlaunched(owner, activation_queue_root, *, now, standing_authorization_dir=None, launch_queue_root=None):
    """A prepared activation whose windows have lapsed with positive evidence it was never launched.

    The caller found no run of it, but a launch id the WebApp or an operator chose
    names no directory a search could guess. So the proof also needs the
    evidence launch leaves: no record that the activation's standing
    authorization was consumed, and no launch queue row, in any state, naming
    the activation or its profile.

    The worker names a result for the activation's queue envelope, as the queue's
    own ``_queue_filename`` names it: ``<activation id>-<request digest>.json``, or
    a hashed name for an id too long for that. An activation id has one request.
    """

    from .task_evaluation_launch_activation_queue import _queue_filename

    if activation_queue_root is None:
        return None, "activation_queue_unconfigured"
    results = Path(activation_queue_root) / "results"
    digest_suffix = "-" + "0" * 64 + ".json"
    stem = _queue_filename(activation_id=owner, request_digest="sha256:" + "0" * 64)
    if not stem.endswith(digest_suffix):
        raise ValueError("terminal_cache_pin_activation_result_name_unknown")
    pattern = re.compile(re.escape(stem[: -len(digest_suffix)]) + r"-[0-9a-f]{64}\.json")
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
    # A profile-authority activation publishes a profile, the one the WebApp launches.
    profile_id = value.get("profile_id")
    if status == "profile_authority_materialized_no_execution" and (
            not isinstance(profile_id, str) or _IDENTIFIER.fullmatch(profile_id) is None):
        return None, "activation_result_invalid"
    written = path.lstat().st_mtime
    if now - written < LAPSE_SECONDS:
        return None, "activation_result_not_stale"
    expires, reason = _authorization_expiry(owner, Path(activation_queue_root) / "prepared" / path.name)
    if expires is None:
        return None, reason
    if now - expires < LAPSE_GRACE_SECONDS:
        return None, "activation_authorization_not_lapsed"
    if status == "profile_authority_materialized_no_execution":
        reason = _authorization_consumption(profile_id, standing_authorization_dir)
        if reason is not None:
            return None, reason
    reason = _launch_requested(launch_queue_root, (owner, profile_id) if profile_id else (owner,))
    if reason is not None:
        return None, reason
    return {"kind": "activation_expired_unlaunched", "result_name": path.name, "result_digest": value["result_digest"],
            "result_status": status, "result_mtime_epoch": written, "authorization_expires_epoch": expires,
            **({"profile_id": profile_id} if profile_id else {})}, None


def _authorization_consumption(profile_id, standing_authorization_dir):
    """Why a consumed-authorization record for the profile proves a launch, or proves nothing; None when none exists.

    Launch admission records each launch it admits under a standing authorization
    at ``<dir>/consumed/<profile id>/<launch id>.json`` (the dispatcher resolves
    ``<dir>`` from its environment or beside its launch state root). Any entry
    there is a launch. A directory that is unconfigured, missing, linked or
    unreadable proves nothing.
    """

    if not standing_authorization_dir:
        return "standing_authorization_unavailable"
    root = Path(standing_authorization_dir)
    consumed = root / "consumed"
    try:
        if root.is_symlink() or not root.is_dir() or consumed.is_symlink():
            return "standing_authorization_unavailable"
        records = consumed / profile_id
        present = _present(records)
        if present is not True:
            return None if present is False else "standing_authorization_unavailable"
        if records.is_symlink() or not records.is_dir():
            return "standing_authorization_unavailable"
        if any(True for _entry in os.scandir(records)):
            return "activation_authorization_consumed"
    except OSError:
        return "standing_authorization_unavailable"
    return None


def _launch_requested(launch_queue_root, names):
    """Whether a launch queue row, in any state, names the activation or its profile; None when none does.

    Every file under the launch queue is read, whatever state directory holds it,
    bounded as ``queue_reference_text`` bounds a row. A linked file or directory,
    a row over the bound, or one that cannot be read proves nothing.
    """

    from .control_plane_storage_references import MAX_QUEUE_MESSAGE_BYTES

    if launch_queue_root is None:
        return "launch_queue_unconfigured"
    root = Path(launch_queue_root)
    needles = [name.encode("utf-8") for name in names]
    try:
        if root.is_symlink() or not root.is_dir():
            return "launch_queue_unavailable"
        for directory, subdirectories, files in os.walk(root, onerror=_raise):
            for name in (*subdirectories, *files):
                if (Path(directory) / name).is_symlink():
                    return "launch_queue_unavailable"
            for name in files:
                path = Path(directory) / name
                if path.lstat().st_size > MAX_QUEUE_MESSAGE_BYTES:
                    return "launch_queue_unavailable"
                data = path.read_bytes()
                if any(needle in data for needle in needles):
                    return "activation_launch_requested"
    except OSError:
        return "launch_queue_unavailable"
    return None


def _raise(error):
    raise error


def _authorization_expiry(owner, envelope_path):
    """When the standing authorization the activation published expires, from its prepared envelope.

    Launch admission checks that authorization, not the mutation window, and the
    request sets its expiry with no maximum. It is parsed with the admission's
    own parser; an envelope that is missing, unsealed, someone else's or without
    an expiry proves nothing.
    """

    from .task_evaluation_standing_launch_authorization import _parse_timestamp

    if not _present(envelope_path):
        return None, "activation_envelope_missing"
    envelope = _read(envelope_path)
    request = envelope.get("request") if envelope is not None else None
    authorization = request.get("authorization") if isinstance(request, dict) else None
    expires = (_parse_timestamp(authorization.get("standing_authorization_expires_at"))
               if isinstance(authorization, dict) else None)
    if (expires is None or envelope.get("schema_version") != ACTIVATION_ENVELOPE_SCHEMA_VERSION
            or envelope.get("envelope_digest") != canonical_digest(envelope, digest_field="envelope_digest")
            or request.get("activation_id") != owner):
        return None, "activation_envelope_invalid"
    return expires.timestamp(), None


def _activation_proof(pin, live_pins, *, evidence_roots, activation_queue_root, hot_window_seconds, classifier, now,
                      standing_authorization_dir=None, launch_queue_root=None, **_context):
    """Every run under the activation's evidence names is a sealed registry run, or it never launched.

    Any whole-run pointer keeps the pin: the archived-run proof already declined
    it. A root that is missing (unmounted or renamed, say), linked, or where a
    name cannot be looked up proves nothing.
    """

    roots = [Path(root) for root in evidence_roots]
    if not roots or any(root.is_symlink() or not root.is_dir() for root in roots):
        return None, "evidence_root_unavailable"
    runs = []
    for root in roots:
        for name in _launch_evidence_names(pin["owner_id"]):
            directory, pointer = _present(root / name), _present(root / (name + POINTER_SUFFIX))
            if directory is None or pointer is None:
                return None, "evidence_root_unavailable"
            if pointer:
                return None, "run_pointer_present"
            if directory:
                runs.append(root / name)
    proof = None
    if not runs:
        proof, reason = _expired_unlaunched(pin["owner_id"], activation_queue_root, now=now,
                                            standing_authorization_dir=standing_authorization_dir,
                                            launch_queue_root=launch_queue_root)
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
    return _EXTENDED_PROOFS[pin["kind"]](pin, live_pins, **context)


def _derive(pin, live_pins, context):
    """``_extended_proof`` and the type of any error it raised: one unreadable pin never costs the tick."""

    try:
        return (*_extended_proof(pin, live_pins, **context), None)
    except Exception as exc:  # noqa: BLE001 - the report keeps the type, never a message with a path
        return None, "proof_error", type(exc).__name__


def _parked_queue_text(queue_roots):
    """Rows parked in a queue state that will still run, beyond pending and processing.

    ``LIVE_QUEUE_STATES`` names those states: a preparation that paused on its
    source preparation or on capacity is pinned and still in flight, yet its row
    sits in ``awaiting_source_preparation`` or ``awaiting_capacity``. The extended
    proofs read these rows as references too; rows are read as
    ``queue_reference_text`` reads them.
    """

    from .control_plane_release_leases import LIVE_QUEUE_STATES
    from .control_plane_storage_references import MAX_QUEUE_MESSAGE_BYTES, QUEUE_STATES
    chunks = []
    for raw_root in queue_roots:
        root = Path(raw_root).expanduser()
        for state in LIVE_QUEUE_STATES.get(root.name, ()):
            directory = root / state
            if state in QUEUE_STATES or not directory.is_dir() or directory.is_symlink():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    if not path.is_symlink() and path.stat().st_size <= MAX_QUEUE_MESSAGE_BYTES:
                        chunks.append(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError):
                    continue
    return "\n".join(chunks)


def _parked_or_error(queue_roots):
    """``(text, None)``, or ``(None, error type)`` when the parked rows cannot be read."""

    try:
        return _parked_queue_text(queue_roots), None
    except Exception as exc:  # noqa: BLE001 - an unreadable parked row keeps the extended candidates, never the tick
        return None, type(exc).__name__


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
    # Parked rows before and after the pending and processing rows, so a row moving
    # between them either way is seen; only the extended proofs read them.
    parked_before = _parked_or_error(queue_roots)
    queue_text = _queue_reference_text(queue_roots)
    parked_after = _parked_or_error(queue_roots)
    parked_error = parked_before[1] or parked_after[1]
    live_queue_text = None if parked_error else "\n".join((parked_before[0], queue_text, parked_after[0]))
    context = {"classifier": classifier, "now": now, "evidence_roots": evidence_roots,
               "hot_window_seconds": hot_window_seconds, "activation_queue_root": activation_queue_root,
               "preparation_queue_root": preparation_queue_root, "running_commit": running_commit,
               "launch_queue_root": launch_queue_root, "standing_authorization_dir": standing_authorization_dir}
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
            if proof is not None and parked_error is not None:
                reason, error = "proof_error", parked_error
            elif proof is not None:
                try:
                    closure = _closure(identity, pins)
                    reason = _closure_reason(identity, closure, pins, live_queue_text, reference_checker)
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
        # and a consumer arriving meanwhile shows up in one or the other. A failure here
        # keeps this pin, with its error type, and costs no other.
        try:
            with storage_pin_guard(pins_root, exclusive=True):
                fresh = "\n".join((_parked_queue_text(queue_roots), _queue_reference_text(queue_roots),
                                   _parked_queue_text(queue_roots)))
                if (proof != _derive(pin, _live_pins(pins_root, now), context)[0]
                        or _referenced(closure, fresh, reference_checker)):
                    kept.append({**candidate, "reason": "reference_changed"})
                    continue
                released.append(release_storage_pin(pins_root=pins_root, kind=pin["kind"],
                                                     owner_id=pin["owner_id"], now=lambda: now))
        except Exception as exc:  # noqa: BLE001 - one failed release never costs the tick or its report
            kept.append({**candidate, "reason": "release_failed", "error_type": type(exc).__name__})
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
