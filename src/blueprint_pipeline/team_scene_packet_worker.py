"""Compile a prepared website scene's team half without a person, and republish its job's offer.

Beta journey step: L5 team policy evaluation. A robot team can run on a website
scene only once the scene's own half of the evaluation is compiled
(``team_scene_packet``), recorded where the team registry reads it, and the
job's offer is republished with it. That took an operator running
``python -m blueprint_pipeline.team_scene_packet`` with eight arguments for
every scene, so no website scene reached a team without a person.

Scene preparation now enqueues one small sealed request here once it has
retained the scene's robot-free readiness receipt, unless that receipt is a
decided failure no team can be admitted past
(``task_evaluation_scene_preparation_completion``; that join never compiles),
and this no-spend worker, woken by its path unit, does the rest on the CPU:

- the capture is resolved under the agent-run partition, the root the executor
  admits, and must be the website capture whose retained task context the
  scene's task was derived from; a withdrawn capture is refused;
- an operator row for the capture always wins, as does a derived row holding
  the operator's frozen internal pair: nothing is compiled or written;
- a derived row that already holds this capture's scene half at this revision
  is reused and only the offer is republished;
- otherwise the launch's published revision is compiled with the configured
  scene bundle the scene configuration retained beside that revision, and with
  what only the operator fixes, in the registry's ``derived_profiles.packet_inputs``
  block: the DROID profile (path and digest), the runtime-source receipt, and
  the root compiled halves are written under. A registry without that block
  keeps this off, by name; an invalid block refuses. Every team body the
  derived defaults register for screening is compiled with the scene too, so
  each gets its own station (``registered_embodiments``).

The scenario is ``nominal``: the website journey never names one (the WebApp
copies the offer's scenario into the run request it prepares), and a team
evaluation runs the scene's nominal cell and its hidden variations
(``team_candidate_quick10_registry.NOMINAL_CELL``), not a capture's episode specs.

Every attempt compiles into a new directory of its own. The derived row,
written by rename, is the only pointer to it, so a reader never sees a partial
half, and a failed or interrupted attempt's directory is removed unless a row
names it. Every request ends in a status file: ``completed``, ``skipped`` with
a reason, or ``blocked`` with typed blockers; ``--requeue-unfinished`` retries
the rest once their cause is fixed. Nothing here allocates a provider, starts a
run or spends: the offer publishes facts, never authority, and every rights,
consent and spend gate after it is unchanged.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

REQUEST_SCHEMA = "blueprint.team_scene_packet_request.v1"
STATUS_SCHEMA = "blueprint.team_scene_packet_status.v1"
RUN_SCHEMA = "blueprint.team_scene_packet_queue_run.v1"
QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
# The root the agent-run executor admits captures under (agent_run_executor._row_capture_root).
CAPTURE_PARTITION_ENV = "BLUEPRINT_AGENT_RUN_CAPTURE_PARTITION_ROOT"
SCENARIO_ID = "nominal"
# A website scene's task is "website-" plus its task context digest (website_task_preparation).
WEBSITE_TASK_PREFIX = "website-"
PACKET_INPUT_FIELDS = ("droid_profile_path", "droid_profile_digest", "runtime_source_receipt_path", "output_root")
# Published beside the revision (task_evaluation_scene_configuration_publication), where
# configured-controls autostart also reads it; materialize_team_scene re-checks its digest.
CONFIGURED_BUNDLE_FILENAME = "configured_scene_bundle.v1.zip"
REQUEST_FIELDS = frozenset({"schema_version", "intent_id", "intent_digest", "capture_id", "task_id",
                            "source_launch_root", "launch_receipt_digest", "scene_revision_digest",
                            "robot_free_readiness_status", "request_digest"})
QUEUE_STATES = ("pending", "processing", "processed", "status")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ATTEMPT = re.compile(r"[0-9a-f]{16}-[0-9a-f]{8}")
_CAPTURE_KEY = re.compile(r"[0-9a-f]{32}")
Fetcher = Callable[[Mapping[str, Any]], bytes]


class TeamScenePacketRequestRefused(ValueError):
    """This request cannot become a team scene half; the message names why."""


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = {**value, field: ""}
    sealed[field] = canonical_digest(sealed, digest_field=field)
    return sealed


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _temporary(directory: Path, value: Mapping[str, Any]) -> Path:
    # Never named *.json, so no queue glob can see a half-written file.
    with tempfile.NamedTemporaryFile("w", dir=directory, prefix=".team-scene-", suffix=".tmp",
                                     delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, indent=1, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return temporary


def _write_replace(path: Path, value: Mapping[str, Any]) -> None:
    os.replace(_temporary(path.parent, value), path)


def _write_new(path: Path, value: Mapping[str, Any], *, staging: Path) -> bool:
    """Write ``path`` only if absent: staged outside its directory, then linked in whole."""

    temporary = _temporary(staging, value)
    try:
        os.link(temporary, path)
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _queue(queue_root: str | Path) -> Path:
    root = Path(queue_root)
    if not root.is_absolute() or any(part.is_symlink() for part in (root, *root.parents)):
        raise TeamScenePacketRequestRefused("team_scene_packet_queue_root_unsafe")
    for state in QUEUE_STATES:
        (root / state).mkdir(parents=True, exist_ok=True, mode=0o750)
    return root


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    descriptor = os.open(root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def request_key(request: Mapping[str, Any]) -> str:
    """One request per capture and published revision."""

    return canonical_digest({"capture_id": request["capture_id"],
                             "scene_revision_digest": request["scene_revision_digest"]})[7:39]


def capture_key(capture_root: Path) -> str:
    """The name the team registry files this capture's derived row under."""

    return hashlib.sha256(str(capture_root).encode()).hexdigest()[:32]


def validated_request(value: Any) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != REQUEST_FIELDS
            or value.get("schema_version") != REQUEST_SCHEMA
            or value.get("request_digest") != canonical_digest(dict(value), digest_field="request_digest")
            or not isinstance(value.get("capture_id"), str) or not _ID.fullmatch(value["capture_id"])
            or not str(value.get("task_id") or "").startswith(WEBSITE_TASK_PREFIX)
            or not isinstance(value.get("scene_revision_digest"), str)
            or not _DIGEST.fullmatch(value["scene_revision_digest"])
            or not Path(str(value.get("source_launch_root") or "")).is_absolute()):
        raise TeamScenePacketRequestRefused("team_scene_request_invalid")
    return dict(value)


def enqueue_team_scene_request(*, intent: Mapping[str, Any], run_root: str | Path,
                               receipt: Mapping[str, Any], result: Mapping[str, Any],
                               readiness: Mapping[str, Any], queue_root: str | Path | None = None
                               ) -> dict[str, Any]:
    """Queue the one request that compiles a completed website scene's team half.

    Called by scene preparation completion after the robot-free readiness
    receipt is retained. It writes one small sealed file and nothing else: no
    compile, no process, no provider. Only an I/O failure raises (the caller
    retries); everything else is a typed status the completion records.
    """

    raw = str(queue_root or os.environ.get(QUEUE_ROOT_ENV, "")).strip()
    if not raw:
        return {"status": "not_configured"}
    task_id = str(intent["request"]["task"]["task_id"])
    if not task_id.startswith(WEBSITE_TASK_PREFIX):
        return {"status": "not_a_website_scene"}
    if readiness.get("retained") is not True:
        return {"status": "not_enqueued", "reason": "robot_free_readiness_not_retained"}
    if readiness.get("status") == "failed":
        # A decided failure is sealed for this revision: no team can ever be admitted to it.
        return {"status": "not_enqueued", "reason": "robot_free_probe_failed"}
    try:
        from .website_preparation_revision_identity import capture_id
        request = validated_request(_seal({
            "schema_version": REQUEST_SCHEMA, "intent_id": str(intent["intent_id"]),
            "intent_digest": str(intent["intent_digest"]), "capture_id": str(capture_id(intent["request"])),
            "task_id": task_id, "source_launch_root": str(Path(run_root).resolve()),
            "launch_receipt_digest": str(receipt.get("receipt_digest") or ""),
            "scene_revision_digest": str(result.get("configured_scene_revision_digest") or ""),
            "robot_free_readiness_status": str(readiness.get("status") or "")}, "request_digest"))
        root = _queue(raw)
    except TeamScenePacketRequestRefused as exc:
        return {"status": "not_enqueued", "reason": str(exc)}
    key = request_key(request)
    for state in ("pending", "processing", "processed"):
        if (root / state / f"{key}.json").exists():
            return {"status": "already_enqueued", "request_key": key, "state": state}
    # Staged in the queue root, so the watched pending/ only ever sees the whole request appear.
    written = _write_new(root / "pending" / f"{key}.json", request, staging=root)
    return {"status": "enqueued" if written else "already_enqueued", "request_key": key,
            "request_digest": request["request_digest"]}


def packet_inputs(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    """The operator's fixed compile inputs under ``derived_profiles.packet_inputs``; None when absent.

    Exactly the four fields, fail-closed like the derived ``defaults``: the
    DROID profile and runtime-source receipt are operator-owned regular files,
    the profile digest is a sha256 (``materialize_team_scene`` re-checks it
    against the bytes), and the output root is absolute.
    """

    from .team_candidate_quick10_registry import TeamCandidateRegistryError, _operator_file

    settings = registry.get("derived_profiles")
    block = settings.get("packet_inputs") if isinstance(settings, Mapping) else None
    if block is None:
        return None
    if not isinstance(block, Mapping) or set(block) != set(PACKET_INPUT_FIELDS):
        raise TeamCandidateRegistryError("team_scene_packet_inputs_invalid")
    output = Path(str(block.get("output_root") or ""))
    if (not _DIGEST.fullmatch(str(block.get("droid_profile_digest") or "")) or not output.is_absolute()
            or output.is_symlink()):
        raise TeamCandidateRegistryError("team_scene_packet_inputs_invalid")
    files = {field: _operator_file(Path(str(block[field])), code="team_scene_packet_inputs_not_operator_owned")
             for field in ("droid_profile_path", "runtime_source_receipt_path")}
    return {**files, "droid_profile_digest": str(block["droid_profile_digest"]), "output_root": output}


def registered_embodiments(registry: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The team bodies registered for screening, once each, so the scene also compiles their own stations.

    Each is a ``screening_candidates.<id>.robot_policy.manifest`` under the derived defaults, the rows every
    derived scene's screening reads. One whose bytes changed or that does not validate is named and skipped.
    """

    from .beta_policy_screening_launch import _bound_path, _read
    from .team_candidate_quick10_registry import _derived_settings
    from .team_embodiment import validate_embodiment_manifest

    configured = ((_derived_settings(registry) or {}).get("defaults") or {}).get("screening_candidates") or {}
    manifests: dict[str, dict[str, Any]] = {}
    unreadable: list[str] = []
    for key, settings in sorted(configured.items()):
        registration = settings.get("robot_policy") if isinstance(settings, Mapping) else None
        if not isinstance(registration, Mapping) or not isinstance(registration.get("manifest"), Mapping):
            continue
        try:
            manifest = validate_embodiment_manifest(_read(_bound_path(registration["manifest"])))
        except (OSError, ValueError, KeyError, TypeError):
            unreadable.append(str(key))
            continue
        manifests[manifest["embodiment_digest"]] = manifest
    return [manifests[digest] for digest in sorted(manifests)], unreadable


def resolve_capture_root(request: Mapping[str, Any], *, partition: str | Path | None = None) -> Path:
    """The request's website capture inside the executor's partition, bound to the scene's task."""

    from .website_atlas_preview_store import capture_withdrawn

    raw = str(partition or os.environ.get(CAPTURE_PARTITION_ENV, "")).strip()
    if not raw:
        raise TeamScenePacketRequestRefused("team_scene_capture_partition_not_configured")
    root = Path(raw)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise TeamScenePacketRequestRefused("team_scene_capture_partition_invalid")
    capture_id = str(request["capture_id"])
    matches = sorted(path for path in (root / "scenes").glob(f"*/captures/{capture_id}")
                     if path.is_dir() and not path.is_symlink())
    if len(matches) != 1:
        raise TeamScenePacketRequestRefused(
            "team_scene_capture_root_" + ("missing" if not matches else "ambiguous"))
    capture = matches[0].resolve()
    scene_id = capture.parent.parent.name
    descriptor = _read_json(capture / "capture_descriptor.json") or {}
    if (not capture.is_relative_to(root.resolve()) or descriptor.get("capture_id") != capture_id
            or descriptor.get("scene_id") != scene_id):
        raise TeamScenePacketRequestRefused("team_scene_capture_root_identity_mismatch")
    if capture_withdrawn(capture):
        raise TeamScenePacketRequestRefused("team_scene_capture_withdrawn")
    context = _read_json(capture / "pipeline" / "website_task_context.json") or {}
    digest = str(context.get("context_digest") or "")
    if (context.get("capture_id") != capture_id or context.get("scene_id") != scene_id
            or digest != canonical_digest(context, digest_field="context_digest")
            or request["task_id"] != WEBSITE_TASK_PREFIX + digest[7:27]):
        raise TeamScenePacketRequestRefused("team_scene_website_task_context_mismatch")
    return capture


def configured_scene_bundle(result: Mapping[str, Any]) -> Path:
    """The configured bundle retained beside the launch's published revision."""

    revision_path = Path(str(result.get("configured_scene_revision_path") or ""))
    bundle = revision_path.parent / CONFIGURED_BUNDLE_FILENAME
    if not revision_path.is_absolute() or bundle.is_symlink() or not bundle.is_file():
        raise TeamScenePacketRequestRefused("team_scene_configured_bundle_unavailable")
    return bundle


def _current_half(profile: Mapping[str, Any] | None, revision: str) -> str | None:
    """This capture's compiled scene half at ``revision``, when its row names one that still reads."""

    from .team_scene_packet import read_team_scene_inputs

    path = str((profile or {}).get("team_scene_inputs_path") or "")
    if not path or (profile or {}).get("scene_revision_digest") != revision:
        return None
    try:
        read_team_scene_inputs(path)
    except (OSError, ValueError, KeyError):
        return None
    return path


def _discard_unreferenced(attempt_root: Any, capture_root: Any) -> None:
    """Remove a failed attempt's own directory unless a derived row names it (unknown keeps it)."""

    from .team_candidate_quick10_registry import profile_for_capture

    attempt = Path(str(attempt_root or ""))
    if (not attempt.is_absolute() or not _ATTEMPT.fullmatch(attempt.name)
            or not _CAPTURE_KEY.fullmatch(attempt.parent.name)
            or any(part.is_symlink() for part in (attempt, *attempt.parents)) or not attempt.is_dir()):
        return
    try:
        profile = profile_for_capture(str(capture_root)) if capture_root else None
    except (OSError, ValueError):
        return
    named = str((profile or {}).get("team_scene_inputs_path") or "")
    if not (named and Path(named).is_relative_to(attempt)):
        shutil.rmtree(attempt, ignore_errors=True)


def _compile(root: Path, request: Mapping[str, Any], *, status: Mapping[str, Any], capture: Path,
             inputs: Mapping[str, Any], fetcher: Fetcher | None,
             embodiments: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    from .task_evaluation_configured_controls_progression_worker import _validate_source
    from .team_scene_packet import prepare_team_scene

    run = Path(request["source_launch_root"])
    if run.is_symlink() or not run.is_dir():
        raise TeamScenePacketRequestRefused("team_scene_source_launch_unavailable")
    result, _receipt, _zero = _validate_source(run)
    if result.get("configured_scene_revision_digest") != request["scene_revision_digest"]:
        raise TeamScenePacketRequestRefused("team_scene_source_launch_revision_changed")
    bundle = configured_scene_bundle(result)
    parent = Path(inputs["output_root"]) / capture_key(capture)
    attempt = parent / f"{request['scene_revision_digest'][7:23]}-{secrets.token_hex(4)}"
    # Named before it exists, so an interrupted attempt is found and removed on the next run.
    _write_replace(root / "status" / f"{status['request_key']}.json",
                   {**status, "status": "processing", "attempt_root": str(attempt)})
    parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    attempt.mkdir(mode=0o750)
    try:
        prepared = prepare_team_scene(
            run_root=run, capture_root=capture, scenario_id=SCENARIO_ID, output_root=attempt, fetcher=fetcher,
            configured_scene_bundle_path=bundle, droid_profile_path=inputs["droid_profile_path"],
            droid_profile_digest=inputs["droid_profile_digest"],
            runtime_source_receipt_path=inputs["runtime_source_receipt_path"],
            **({"embodiment_manifests": list(embodiments)} if embodiments else {}))
        written = dict(prepared.get("registry") or {})
        if written.get("status") != "written":
            raise TeamScenePacketRequestRefused(
                "team_scene_derived_profile_" + str(written.get("status") or "not_written"))
    except Exception:
        # Nothing names this attempt yet (the derived row is the last write), so it goes.
        shutil.rmtree(attempt, ignore_errors=True)
        raise
    return {"outcome": "compiled", "attempt_root": str(attempt), "task_id": prepared["task_id"],
            "team_scene_inputs_path": prepared["path"], "team_scene_inputs_digest": prepared["inputs_digest"],
            "derived_profile": {key: written[key] for key in ("path", "profile_digest")}}


def _blocker(exc: BaseException) -> str:
    text = str(exc).strip()
    if isinstance(exc, (ValueError, RuntimeError)) and text:
        return text[:200]
    return f"team_scene_packet_failed:{type(exc).__name__}"


def _process(root: Path, claimed: Path, *, fetcher: Fetcher | None,
             offer_transport: Callable[..., Any] | None) -> dict[str, Any]:
    from . import team_candidate_quick10_registry as registry_module
    from .agent_execution_offer import publish_agent_execution_offer

    status: dict[str, Any] = {"schema_version": STATUS_SCHEMA, "request_key": claimed.stem,
                              "scenario_id": SCENARIO_ID, "provider_mutation_performed": False,
                              "paid_execution_requested": False}

    def skipped(reason: str) -> dict[str, Any]:
        return {**status, "status": "skipped", "reason": reason, "blockers": []}

    try:
        request = validated_request(_read_json(claimed))
        if request_key(request) != claimed.stem:
            raise TeamScenePacketRequestRefused("team_scene_request_invalid")
        status.update({key: request[key] for key in ("request_digest", "capture_id", "scene_revision_digest")})
        registry = registry_module.load_registry()
        if registry is None:
            return skipped("team_candidate_registry_not_configured")
        if registry_module._derived_settings(registry) is None:
            return skipped("derived_profiles_not_configured")
        inputs = packet_inputs(registry)
        if inputs is None:
            return skipped("team_scene_packet_inputs_not_configured")
        capture = resolve_capture_root(request)
        status["capture_root"] = str(capture)
        if any(isinstance(row, Mapping) and row.get("capture_root") == str(capture) for row in registry["profiles"]):
            # The operator's row is what every reader uses; nothing is compiled beside it.
            return skipped("team_candidate_operator_row_present")
        current = registry_module.profile_for_capture(capture)
        if (current or {}).get("pair_runtime_inputs_path"):
            # A frozen internal pair is the operator's choice of scene half, never replaced here.
            return skipped("team_candidate_frozen_pair_present")
        existing = _current_half(current, request["scene_revision_digest"])
        if existing is not None:
            status.update(outcome="already_compiled", team_scene_inputs_path=existing)
        else:
            embodiments, unreadable = registered_embodiments(registry)
            if unreadable:
                status["embodiment_manifests_unreadable"] = unreadable
            status.update(_compile(root, request, status=status, capture=capture, inputs=inputs, fetcher=fetcher,
                                   embodiments=embodiments))
    except Exception as exc:  # noqa: BLE001 - every request ends in a typed status; the queue keeps moving
        blocker = _blocker(exc)
        if not blocker.startswith("team_scene_packet_failed:"):
            return {**status, "status": "blocked", "blockers": [blocker]}
        from .bounded_error import error_description

        # The type alone hides a full disk or a missing input; keep a redacted, trimmed message.
        return {**status, "status": "blocked", "blockers": [blocker], "error": error_description(exc)}
    published = publish_agent_execution_offer(capture, transport=offer_transport)
    status["offer"] = {"status": published.get("status"), "reason": published.get("reason"),
                       "team_evaluation_listed": bool((published.get("offer") or {}).get("team_evaluation")),
                       "team_candidate_offer_refused": published.get("team_candidate_offer_refused")}
    if published.get("status") == "failed":
        # The half is recorded; a requeue reuses it and only publishes again.
        return {**status, "status": "blocked", "blockers": ["team_scene_offer_publication_failed"]}
    return {**status, "status": "completed", "blockers": []}


def _finish(root: Path, claimed: Path, status: Mapping[str, Any]) -> dict[str, Any]:
    _write_replace(root / "status" / claimed.name, status)
    os.replace(claimed, root / "processed" / claimed.name)
    return dict(status)


def _recover_interrupted(root: Path, claimed: Path) -> dict[str, Any]:
    prior = _read_json(root / "status" / claimed.name) or {}
    _discard_unreferenced(prior.get("attempt_root"), prior.get("capture_root"))
    kept = {key: prior[key] for key in ("request_digest", "capture_id", "scene_revision_digest", "capture_root")
            if key in prior}
    return _finish(root, claimed, {"schema_version": STATUS_SCHEMA, "request_key": claimed.stem, **kept,
                                   "scenario_id": SCENARIO_ID, "status": "blocked",
                                   "blockers": ["team_scene_packet_interrupted"],
                                   "provider_mutation_performed": False, "paid_execution_requested": False})


def process_team_scene_queue(*, queue_root: str | Path, max_messages: int = 2, fetcher: Fetcher | None = None,
                             offer_transport: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Advance up to ``max_messages`` pending requests to a terminal status, on the CPU."""

    if not isinstance(max_messages, int) or isinstance(max_messages, bool) or not 1 <= max_messages <= 16:
        raise TeamScenePacketRequestRefused("team_scene_packet_max_messages_invalid")
    root = _queue(queue_root)
    results: list[dict[str, Any]] = []
    with _locked(root):
        # The lock is held for the whole run, so anything still processing was interrupted.
        recovered = [_recover_interrupted(root, path) for path in sorted((root / "processing").glob("*.json"))]
        for pending in sorted((root / "pending").glob("*.json"))[:max_messages]:
            claimed = root / "processing" / pending.name
            try:
                os.replace(pending, claimed)
            except FileNotFoundError:
                continue
            results.append(_finish(root, claimed, _process(root, claimed, fetcher=fetcher,
                                                           offer_transport=offer_transport)))
    return {"schema_version": RUN_SCHEMA, "status": "processed" if results or recovered else "idle",
            "processed_count": len(results), "results": results, "recovered": recovered,
            "provider_mutation_performed": False, "paid_execution_requested": False}


def requeue_unfinished(*, queue_root: str | Path) -> dict[str, Any]:
    """Return every processed request whose status is not completed to pending, for one more run."""

    root = _queue(queue_root)
    requeued: list[str] = []
    with _locked(root):
        for path in sorted((root / "processed").glob("*.json")):
            target = root / "pending" / path.name
            if (_read_json(root / "status" / path.name) or {}).get("status") == "completed" or target.exists():
                continue
            os.replace(path, target)
            requeued.append(path.stem)
    return {"schema_version": RUN_SCHEMA, "status": "requeued", "requeued": requeued,
            "provider_mutation_performed": False, "paid_execution_requested": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue-root", default=os.environ.get(QUEUE_ROOT_ENV, ""))
    parser.add_argument("--max-messages", type=int, default=2)
    parser.add_argument("--requeue-unfinished", action="store_true",
                        help="return skipped and blocked requests to pending once their cause is fixed")
    args = parser.parse_args(argv)
    if not args.queue_root:
        print(json.dumps({"schema_version": RUN_SCHEMA, "status": "blocked",
                          "blockers": ["team_scene_packet_queue_root_not_configured"]}, sort_keys=True))
        return 2
    result = (requeue_unfinished(queue_root=args.queue_root) if args.requeue_unfinished
              else process_team_scene_queue(queue_root=args.queue_root, max_messages=args.max_messages))
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "CAPTURE_PARTITION_ENV",
    "PACKET_INPUT_FIELDS",
    "QUEUE_ROOT_ENV",
    "REQUEST_SCHEMA",
    "SCENARIO_ID",
    "STATUS_SCHEMA",
    "TeamScenePacketRequestRefused",
    "capture_key",
    "configured_scene_bundle",
    "enqueue_team_scene_request",
    "packet_inputs",
    "process_team_scene_queue",
    "registered_embodiments",
    "request_key",
    "requeue_unfinished",
    "resolve_capture_root",
    "validated_request",
]


if __name__ == "__main__":
    raise SystemExit(main())
