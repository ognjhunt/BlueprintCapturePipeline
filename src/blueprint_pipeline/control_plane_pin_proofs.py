"""The extended proofs that a storage pin protects nothing: read-only evidence, never a mutation.

``control_plane_terminal_cache_pins`` evaluates them on every tick and releases
a pin by one only with the owner's opt-in. Each returns ``(proof, None)`` when
it holds, or ``(None, reason)`` with a typed reason. Nothing here writes, locks
or removes anything.

* ``sealed_registry_run``: an activation pin whose every run carries its
  terminal receipt and a result registry the artifact store accepts as sealed
  (delivered completed_unqualified, blocked or cancelled, its closeout receipts
  intact), idle past the hot window, with no whole-run pointer. Whole-run
  offload never archives a registry run, so neither original proof could
  release one. Without the receipt the canary dispatcher can still recover a
  stranded delivery from the launch set.
* ``activation_expired_unlaunched``: a profile-authority activation pin with no
  run directory and no pointer under any of its evidence names in any evidence
  root, whose one sealed result in the activation queue says it was prepared
  more than a week and a day ago (a shared mutation window lives at most a
  week), and whose standing authorization expired more than a day ago (launch
  admission checks that authorization, and the request dates it with no
  maximum). A launch id the WebApp or an operator chooses names no directory a
  search could guess, so the proof also needs positive evidence: no record
  under ``<standing authorization dir>/consumed/<profile id>/`` and no launch
  queue row, in any state, naming the activation or its profile. Without the
  activation queue, the launch queue or the standing authorization directory
  this proof is off. For a profile without the one-use standing authorization
  requirement, an operator's per-launch handshake can still admit a launch
  after the authorization lapsed; if that happens after the pin was released,
  the launch fails its input verification rather than using missing inputs,
  which re-preparing recovers. The proof accepts that risk only after both the
  window and the authorization lapsed and neither record exists. A
  policy-campaign activation is out of scope
  (``policy_campaign_activation_out_of_scope``): it publishes no standing
  authorization and dispatches through the policy canary queue on the scene
  execution window, and the canary dispatcher releases its pin on completion.
* ``unconsumed_stale_pin``: a preparation or compilation pin that no live pin
  depends on, created more than a week and a day ago, whose paths are all
  ``cache``, and whose preparation no activation can take any more: its sealed
  envelope sits in ``materialized/`` bound to a release other than the running
  one, or in ``blocked/``. The activation worker verifies materialized inputs
  and never re-fetches them, so age alone proves nothing: a materialized
  preparation waits for its activation intent with no age limit.

Both activation proofs look for a run under every name an activation can launch
as (``_launch_evidence_names``): its id, ``<id>-launch``, and the bounded launch
id the launch paths derive for a long id, with their own functions.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .control_plane_evidence_offload import POINTER_SUFFIX, _has_result_registry, _terminal_receipt
from .control_plane_storage_pins import depends_on
from .control_plane_storage_references import QueueReferenceUnreadable, queue_reference_text
from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

# A pin may name the reproducible activation inputs (cache or work class) or the
# retained run directory itself (evidence_cold). Releasing a pin removes no
# bytes, so any of these is acceptable; hot evidence and state never are.
_PIN_PATH_CLASSES = ("cache", "work", "evidence_cold")
MINIMUM_PIN_AGE_SECONDS = 6 * 3600
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


def _read(path):
    if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
            or path.stat().st_size > 16 * 1024**2):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _record(path):
    """``(record, None)``, or ``(None, "unreadable")`` for a file that cannot be read as it stands (linked, not a
    regular file, oversized or refused), or ``(None, "invalid")`` for one that is not a JSON object."""

    try:
        if (any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file()
                or path.stat().st_size > 16 * 1024**2):
            return None, "unreadable"
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, "unreadable"
    try:
        value = json.loads(text)
    except ValueError:
        return None, "invalid"
    return (value, None) if isinstance(value, dict) else (None, "invalid")


def _pin_path_allowed(classifier, path, classes=_PIN_PATH_CLASSES):
    last = None
    for expected in classes:
        try:
            classifier(str(path), expected=expected, code="terminal_cache_pin_path_class_invalid")
            return
        except ValueError as exc:
            last = exc
    raise last


def _paths_classify(classifier, paths, classes):
    try:
        for path in paths:
            _pin_path_allowed(classifier, path, classes)
    except ValueError:
        return False
    return True


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


def _unconsumed_stale_pin(pin, live_pins, *, classifier, now, preparation_queue_root, running_commit,
                          preparation_envelopes=None, **_context):
    """A preparation or compilation that nothing consumes, that has outlived every mutation window, and that
    no activation can take any more.

    A compilation is named for its preparation, so both read the preparation's envelope.
    """

    if any(depends_on(other, pin["kind"], pin["owner_id"]) for other in live_pins.values()):
        return None, "depended_on"
    if now - pin["created_at_epoch"] < LAPSE_SECONDS:
        return None, "pin_not_stale"
    if not _paths_classify(classifier, pin["paths"], ("cache",)):
        return None, "path_class_invalid"
    envelopes = preparation_envelopes or preparation_envelope_snapshot(preparation_queue_root)
    ended, reason = _unactivatable_preparation(pin["owner_id"], envelopes, running_commit)
    if ended is None:
        return None, reason
    state, commit = ended
    return {"kind": "unconsumed_stale_pin", "created_at_epoch": pin["created_at_epoch"],
            "preparation_state": state, "preparation_commit": commit}, None


_ENVELOPE_NAME = re.compile(r"(.+)-[0-9a-f]{64}\.json")


def preparation_envelope_snapshot(preparation_queue_root):
    """The preparation queue's ended envelopes, indexed once however many pins ask.

    Call it for ``(index, None)`` or ``(None, reason)``: the index maps a
    preparation id to the ``(state, path)`` of every envelope named for it in
    ``materialized/`` and ``blocked/``. Both directories are listed twice and the
    listings unioned, so an envelope moving between them is seen (twice, and so
    ambiguous: the pin is kept). A linked or unlistable state proves nothing.
    """

    taken = []

    def snapshot():
        if not taken:
            taken.append(_index_envelopes(preparation_queue_root))
        return taken[0]

    return snapshot


def _index_envelopes(preparation_queue_root):
    if preparation_queue_root is None:
        return None, "preparation_queue_unconfigured"
    found = set()
    for _read_pass in range(2):
        for state in ("materialized", "blocked"):
            directory = Path(preparation_queue_root) / state
            try:
                if directory.is_symlink():
                    return None, "preparation_queue_unavailable"
                if not directory.is_dir():
                    continue
                with os.scandir(directory) as entries:
                    found.update((state, entry.name) for entry in entries if _ENVELOPE_NAME.fullmatch(entry.name))
            except OSError:
                return None, "preparation_queue_unavailable"
    index = {}
    for state, name in sorted(found):
        index.setdefault(_ENVELOPE_NAME.fullmatch(name).group(1), []).append(
            (state, Path(preparation_queue_root) / state / name))
    return index, None


def _unactivatable_preparation(preparation_id, envelopes, running_commit):
    """``(state, commit)`` when no activation can take the preparation any more, else ``(None, reason)``.

    The activation worker takes a preparation only from ``materialized/``, and only
    when its sealed envelope names the release the worker runs; it verifies the
    materialized inputs and never re-fetches them. So the preparation's one sealed
    envelope must sit in ``materialized/`` bound to another release, or in
    ``blocked/``. Anything else, including an envelope this queue does not hold, keeps
    the pin. A rollback to that release would make it activatable again.
    ``envelopes`` is a ``preparation_envelope_snapshot``.
    """

    index, reason = envelopes()
    if index is None:
        return None, reason
    if not isinstance(running_commit, str) or _COMMIT.fullmatch(running_commit) is None:
        return None, "running_commit_unknown"
    found = index.get(preparation_id, [])
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


def _evidence_root_state(root):
    """True for a real directory, False for a missing root its readable parent shows absent, else None."""

    if root.is_symlink():
        return None
    if root.is_dir():
        return True
    if _present(root) is not False or any(path.is_symlink() for path in root.parents):
        return None
    try:
        if root.parent.is_symlink() or not root.parent.is_dir():
            return None
        with os.scandir(root.parent):
            pass
    except OSError:
        return None
    return False


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
        # Not this proof's run. Which keep reason fits is judged by the receipt's age, not a walk of the tree.
        sealed_at = (directory / receipt).lstat().st_mtime
        return None, "run_hot" if now - sealed_at < hot_window_seconds else "run_without_registry"
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


def _expired_unlaunched(owner, activation_queue_root, *, now, standing_authorization_dir=None, launch_queue=None):
    """A prepared profile-authority activation whose windows have lapsed, with positive evidence it never launched.

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
    if status != "profile_authority_materialized_no_execution":
        # A policy campaign publishes no standing authorization and dispatches through the
        # policy canary queue on the scene execution window: this proof's premise does not
        # hold, and the canary dispatcher releases its pin when the campaign completes.
        return None, "policy_campaign_activation_out_of_scope"
    # A profile-authority activation publishes a profile, the one the WebApp launches.
    profile_id = value.get("profile_id")
    if not isinstance(profile_id, str) or _IDENTIFIER.fullmatch(profile_id) is None:
        return None, "activation_result_invalid"
    written = path.lstat().st_mtime
    if now - written < LAPSE_SECONDS:
        return None, "activation_result_not_stale"
    expires, reason = _authorization_expiry(owner, Path(activation_queue_root) / "prepared" / path.name)
    if expires is None:
        return None, reason
    if now - expires < LAPSE_GRACE_SECONDS:
        return None, "activation_authorization_not_lapsed"
    reason = _authorization_consumption(profile_id, standing_authorization_dir) or _launch_requested(
        launch_queue, (owner, profile_id))
    if reason is not None:
        return None, reason
    return {"kind": "activation_expired_unlaunched", "result_name": path.name, "result_digest": value["result_digest"],
            "result_status": status, "result_mtime_epoch": written, "authorization_expires_epoch": expires,
            "profile_id": profile_id}, None


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


def launch_queue_snapshot(launch_queue_root):
    """The launch queue, read once however many activations ask: call it for ``(text, None)`` or ``(None, reason)``.

    Every row in every state directory is read with the strict queue reader,
    twice, so a row renamed between states mid-read is seen. A linked,
    oversized or unreadable row or state proves nothing (``queue_unreadable``).
    """

    taken = []

    def snapshot():
        if not taken:
            taken.append(_read_launch_queue(launch_queue_root))
        return taken[0]

    return snapshot


def _read_launch_queue(launch_queue_root):
    if launch_queue_root is None:
        return None, "launch_queue_unconfigured"
    root = Path(launch_queue_root)
    if root.is_symlink() or not root.is_dir():
        return None, "launch_queue_unavailable"
    try:
        return "\n".join(queue_reference_text([root], states=None, strict=True) for _read_pass in range(2)), None
    except QueueReferenceUnreadable:
        return None, "queue_unreadable"


def _launch_requested(launch_queue, names):
    """Whether a launch queue row, in any state, names the activation or its profile; None when none does."""

    text, reason = launch_queue()
    if reason is not None:
        return reason
    return "activation_launch_requested" if any(name in text for name in names) else None


def _authorization_expiry(owner, envelope_path):
    """When the standing authorization the activation published expires, from its prepared envelope.

    Launch admission checks that authorization, not the mutation window, and the
    request sets its expiry with no maximum. It is parsed with the admission's
    own parser; an envelope that is missing, unsealed, someone else's or without
    an expiry proves nothing.
    """

    from .task_evaluation_standing_launch_authorization import _parse_timestamp

    present = _present(envelope_path)
    if present is not True:
        return None, "activation_envelope_missing" if present is False else "activation_envelope_unreadable"
    envelope, reason = _record(envelope_path)
    if reason is not None:
        return None, f"activation_envelope_{reason}"
    request = envelope.get("request")
    authorization = request.get("authorization") if isinstance(request, dict) else None
    expires = (_parse_timestamp(authorization.get("standing_authorization_expires_at"))
               if isinstance(authorization, dict) else None)
    if (expires is None or not isinstance(request, dict)
            or envelope.get("schema_version") != ACTIVATION_ENVELOPE_SCHEMA_VERSION
            or envelope.get("envelope_digest") != canonical_digest(envelope, digest_field="envelope_digest")
            or request.get("activation_id") != owner):
        return None, "activation_envelope_invalid"
    return expires.timestamp(), None


def _activation_proof(pin, live_pins, *, evidence_roots, activation_queue_root, hot_window_seconds, classifier, now,
                      standing_authorization_dir=None, launch_queue_root=None, launch_queue=None, **_context):
    """Every run under the activation's evidence names is a sealed registry run, or it never launched.

    Any whole-run pointer keeps the pin: the archived-run proof already declined
    it. A root that is linked, unreadable or not a directory, or where a name
    cannot be looked up, proves nothing; so does a missing root whose parent is
    missing too (unmounted, say). A missing root whose parent is present and
    readable, with no linked ancestor, holds no runs: the unit marks two roots
    optional, and a host without them would otherwise never release.
    """

    roots = []
    for root in (Path(root) for root in evidence_roots):
        state = _evidence_root_state(root)
        if state is None:
            return None, "evidence_root_unavailable"
        if state:
            roots.append(root)
    if not evidence_roots:
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
                                            launch_queue=launch_queue or launch_queue_snapshot(launch_queue_root))
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


def extended_proof(pin, live_pins, **context):
    """``(proof, None)`` when an extended proof holds for ``pin``, else ``(None, reason)``. It only reads.

    The six-hour minimum pin age is checked first, as the original proofs check it.
    """

    created = pin.get("created_at_epoch")
    if type(created) not in (int, float):
        return None, "pin_invalid"
    if context["now"] - created < MINIMUM_PIN_AGE_SECONDS:
        return None, "pin_young"
    return _EXTENDED_PROOFS[pin["kind"]](pin, live_pins, **context)


__all__ = [
    "ACTIVATION_ENVELOPE_SCHEMA_VERSION",
    "ACTIVATION_QUEUE_NAME",
    "ACTIVATION_RESULT_SCHEMA_VERSION",
    "LAPSE_SECONDS",
    "LAUNCH_QUEUE_NAME",
    "MINIMUM_PIN_AGE_SECONDS",
    "PREPARATION_ENVELOPE_SCHEMA_VERSION",
    "PREPARATION_QUEUE_NAME",
    "PREPARED_ACTIVATION_STATUSES",
    "activation_queue_root_of",
    "extended_proof",
    "launch_queue_root_of",
    "launch_queue_snapshot",
    "preparation_envelope_snapshot",
    "preparation_queue_root_of",
]
