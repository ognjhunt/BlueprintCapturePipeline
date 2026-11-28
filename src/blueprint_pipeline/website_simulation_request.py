"""A website job's simulation is built when an approved deployment company asks for it on that job.

Owner decision, 2026-11-28: the simulation of the site is built only on request. Until a request for the job
exists, its preparation stage keeps everything that is free and needed to share the job: the hand-off, consent,
rights and withdrawal as they are, the scene grant, the normalized video (the company clip is made from it, so
it is not rebuilt later), the task context and its evidence. It then ends as a completed stage that waits
(``WAITING_STATUS``): no clean plate, grounding or SAM step, no Atlas job, review or planner, no World Labs call,
no object authoring, no scene intent, compile or package, and nothing is reserved, admitted or spent for them.
The company clip is a separate pass (``website_company_clip``) and publishes as promptly as before.

**Which jobs wait.** Only a job whose task context the website sealed with ``simulation_on_request: true``
(``CONTEXT_FIELD``), read from the context the hand-off kept, and only when its job ledger keeps the delivery
key the replay below needs. A website that predates this never sends the field, so its jobs build at once, as
before: the change is safe whichever repository deploys first. A job that waits reports ``awaiting_inputs``;
the reason ``awaiting_simulation_request`` goes only to a website that lists the new reasons
(``website_preparation_outcome.website_lists_new_reasons``).

**The request.** The website's signed push (``POST /api/live-pipeline/website-simulation-requests/{capture_id}``,
``website_atlas_preview_http``) writes one small record, ``pipeline/simulation_request/request.json``
(``record_request``): written atomically without following links, only for a capture the hand-off kept and
that is not withdrawn, once. The first request is the job's: a later one, from the same row or another company,
is ``already_recorded`` and changes nothing, since one simulation serves every company. No company name, no
person, no free text.

**The start.** The held-handoff replay (``website_held_handoff_replay``) reads the waiting handoff as held on
kind ``simulation_request`` and, once the record exists, its fact ``simulation_requested`` keyed by the
request's digest: one replay per digest, never through the rules reopen. The replay re-checks consent and the
claim as every replay does, and the stage then runs today's path unchanged (``site_package_orchestrator``),
through every paid admission and the job's cap. A failure after that retries as any preparation does
(``job_retry_policy``). The wait itself is a completed stage, so nothing retries, pages or ends it.

A leaf apart from the sealing helper and, where named, the stores it reads.
"""
from __future__ import annotations

import json
import os
import re
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_simulation_request.v1"
DIRECTORY = ("pipeline", "simulation_request")
MARK_FILE = "request.json"
#: The field of the website's task context that makes a job wait for a request (absent: it builds at once).
CONTEXT_FIELD = "simulation_on_request"
#: The waiting handoff's status, and the reason the status reports to a website that lists it.
WAITING_STATUS = "awaiting_simulation_request"
#: The held-handoff replay's hold kind and the fact that starts it.
HOLD_KIND = "simulation_request"
FACT_KIND = "simulation_requested"
SOURCES = frozenset({"company", "internal_test"})
BODY_FIELDS = frozenset({"request_id", "scene_id", "request_digest", "source", "requested_at_epoch"})
_RECORD_FIELDS = BODY_FIELDS | {"schema_version", "capture_id", "received_at_epoch", "digest"}
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_MAX_BYTES = 4096
HANDOFF = ("pipeline", "website_scene_preparation", "handoff.json")
#: What ``record_request`` answers: the route turns each into its response.
RECORDED, ALREADY_RECORDED = "recorded", "already_recorded"
CAPTURE_UNKNOWN, CAPTURE_WITHDRAWN, INVALID = "capture_unknown", "capture_withdrawn", "invalid"


def request_path(capture_root: Path) -> Path:
    return Path(capture_root).joinpath(*DIRECTORY) / MARK_FILE


def _plain_json(path: Path, limit: int) -> Any:
    """A small file read without following a link at it or at its folder; None when absent or unreadable."""
    try:
        if path.parent.is_symlink() or path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def valid_body(body: Any, *, capture_id: str) -> bool:
    """The route's body, exactly: the job, its scene, the website's digest of its request row, who asked, when."""
    from .website_atlas_preview_store import capture_name_matches_request

    return (isinstance(body, Mapping) and set(body) == BODY_FIELDS
            and isinstance(body["request_id"], str) and _REQUEST_ID.fullmatch(body["request_id"]) is not None
            and body["scene_id"] == f"site-{body['request_id']}"
            and capture_name_matches_request(body["request_id"], capture_id)
            and isinstance(body["request_digest"], str) and _DIGEST.fullmatch(body["request_digest"]) is not None
            and body["source"] in SOURCES
            and type(body["requested_at_epoch"]) is int and body["requested_at_epoch"] >= 0)


def read_request(capture_root: Path) -> dict[str, Any] | None:
    """The capture's recorded request, sealed and its own; None while there is none (or none that reads)."""
    root = Path(capture_root)
    value = _plain_json(request_path(root), _MAX_BYTES)
    if (not isinstance(value, dict) or set(value) != _RECORD_FIELDS or value.get("schema_version") != SCHEMA
            or value.get("digest") != canonical_digest(value, digest_field="digest")
            or value.get("capture_id") != root.name or value.get("scene_id") != root.parent.parent.name
            or not valid_body({key: value[key] for key in BODY_FIELDS}, capture_id=root.name)):
        return None
    return value


def _write_once(folder: Path, value: Mapping[str, Any]) -> bool:
    """Write the record unless one is there: a partial file, then a hard link that fails if the name exists.
    A record that is there but does not read is replaced. True when this call wrote it."""
    partial = folder / f".{MARK_FILE}.{os.getpid()}.{time.monotonic_ns()}.partial"
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        target = folder / MARK_FILE
        try:
            os.link(partial, target, follow_symlinks=False)
            return True
        except FileExistsError:
            if target.is_symlink():
                raise OSError("website_simulation_request_record_is_a_link") from None
            if read_request(folder.parent.parent) is not None:
                return False
            os.replace(partial, target)  # an unreadable record names no request: this one replaces it
            return True
    finally:
        partial.unlink(missing_ok=True)


def record_request(*, capture_id: str, body: Any, storage_root: Path | None, now: float | None = None) -> str:
    """The website asks for this job's simulation: keep the request once. Never raises.

    ``"recorded"`` on the first write; ``"already_recorded"`` when the capture already has a request (the
    same digest, or another one: the first stands); ``"capture_unknown"`` when no capture the hand-off kept
    has this name for this job, or nothing can be checked; ``"capture_withdrawn"`` when the site withdrew;
    ``"invalid"`` for a body that is not the contract's. Nothing else happens here: the held-handoff replay
    starts the build on its next pass.
    """
    from .website_atlas_preview_store import capture_withdrawn

    try:
        if not isinstance(capture_id, str) or not valid_body(body, capture_id=capture_id):
            return INVALID
        if storage_root is None:
            return CAPTURE_UNKNOWN
        request_id = body["request_id"]
        roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
                 if root.is_dir() and not root.is_symlink()]
        if len(roots) != 1:
            return CAPTURE_UNKNOWN
        root = roots[0]
        pipeline = root / DIRECTORY[0]
        if pipeline.is_symlink() or not pipeline.is_dir():
            return CAPTURE_UNKNOWN  # not handed off yet: the website asks again
        if capture_withdrawn(root):
            return CAPTURE_WITHDRAWN
        if read_request(root) is not None:
            return ALREADY_RECORDED
        folder = root.joinpath(*DIRECTORY)
        if folder.is_symlink():
            return CAPTURE_UNKNOWN
        folder.mkdir(mode=0o750, exist_ok=True)
        if not stat.S_ISDIR(os.lstat(folder).st_mode):
            return CAPTURE_UNKNOWN
        value = {"schema_version": SCHEMA, **{key: body[key] for key in sorted(BODY_FIELDS)},
                 "capture_id": capture_id, "received_at_epoch": int(time.time() if now is None else now)}
        value["digest"] = canonical_digest(value, digest_field="digest")
        return RECORDED if _write_once(folder, value) else ALREADY_RECORDED
    except Exception:  # noqa: BLE001 - the website asks again until it hears 202
        return CAPTURE_UNKNOWN


def _retained_context(capture_root: Path) -> dict[str, Any] | None:
    value = _plain_json(Path(capture_root) / "pipeline" / "website_task_context.json", 2_000_000)
    return value if isinstance(value, dict) else None


def waits_for_request(capture_root: Path) -> bool:
    """Whether this run's preparation waits for a request instead of building. Never raises: on doubt it builds.

    The kept task context asks for it (``CONTEXT_FIELD`` true), the ledger keeps the delivery key the replay
    runs on, and no request is recorded yet.
    """
    try:
        root = Path(capture_root)
        context = _retained_context(root)
        if not isinstance(context, dict) or context.get(CONTEXT_FIELD) is not True:
            return False
        ledger = _plain_json(root / "pipeline_job_ledger.json", 8_000_000)
        key = ledger.get("producer_delivery_key") if isinstance(ledger, dict) else None
        return isinstance(key, str) and bool(key) and read_request(root) is None
    except Exception:  # noqa: BLE001 - a job that cannot tell builds, as before
        return False


def retain_awaiting_simulation_request(*, capture_root: Path, pipeline_dir: Path,
                                       task_context: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the waiting handoff and return the completed stage's row. Writes nothing else."""
    from .core.handoff_lease import guarded_handoff_write_json

    handoff = {"schema_version": "website_scene_handoff.v1", "scene_id": task_context["scene_id"],
               "capture_id": task_context["capture_id"], "status": WAITING_STATUS, "blockers": [],
               "task_context_digest": task_context["context_digest"], "simulator_ready": False,
               "visual_reconstruction_ready": False, "provider_mutation_performed": False,
               "claim_ceiling": "development_only"}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    guarded_handoff_write_json(Path(capture_root).joinpath(*HANDOFF), handoff)
    guarded_handoff_write_json(Path(pipeline_dir) / "provider_run_manifest.json", {
        "schema_version": "v1", "provider_name": None, "status": "not_requested",
        "website_scene_preparation": handoff})
    return {"status": "completed", "lane": "qualification", "website_scene_preparation": handoff,
            "native_execution_complete": False}


def _waiting_shape(handoff: Any) -> bool:
    return (isinstance(handoff, Mapping) and handoff.get("schema_version") == "website_scene_handoff.v1"
            and handoff.get("status") == WAITING_STATUS and handoff.get("blockers") == []
            and handoff.get("simulator_ready") is False and handoff.get("provider_mutation_performed") is False
            and isinstance(handoff.get("digest"), str)
            and handoff["digest"] == canonical_digest(handoff, digest_field="digest")
            and not any(key in handoff for key in ("website_intake_outbox", "navigation_intake", "runtime_inputs",
                                                   "source_registration", "base_scene_path")))


def retains_awaiting_simulation_request(handoff: Mapping[str, Any], task_context: Mapping[str, Any]) -> bool:
    """A completed preparation stage that waits for a request for the job's simulation, under this context."""
    return (_waiting_shape(handoff) and isinstance(task_context, Mapping)
            and all(handoff.get(key) == task_context.get(key) for key in ("scene_id", "capture_id"))
            and handoff.get("task_context_digest") == task_context.get("context_digest"))


def awaiting(capture_root: Path) -> bool:
    """The capture's retained handoff is the waiting one (request recorded or not). Never raises."""
    root = Path(capture_root)
    handoff = _plain_json(root.joinpath(*HANDOFF), 1_000_000)
    return (_waiting_shape(handoff) and handoff.get("capture_id") == root.name
            and handoff.get("scene_id") == root.parent.parent.name)


def arrived_fact(capture_root: Path) -> dict[str, Any] | None:
    """The replay's fact once a request is recorded: its kind and the request's digest (no digest of its own yet)."""
    request = read_request(capture_root)
    if request is None:
        return None
    return {"kind": FACT_KIND, "request_digest": request["request_digest"]}


def waiting_outcome(capture_root: Path, *, ledger: Mapping[str, Any]) -> dict[str, Any] | None:
    """Status v7's outcome for a completed job that waits for a request, while the website lists new reasons."""
    from .website_preparation_outcome import website_lists_new_reasons

    root = Path(capture_root)
    if not website_lists_new_reasons() or ledger.get("status") != "completed" or not awaiting(root):
        return None
    handoff = _plain_json(root.joinpath(*HANDOFF), 1_000_000)
    return {"category": "held", "reason": WAITING_STATUS, "basis": "simulation_on_request", "robot_id": None,
            "evidence_digest": handoff["digest"]}


__all__ = ["ALREADY_RECORDED", "CAPTURE_UNKNOWN", "CAPTURE_WITHDRAWN", "CONTEXT_FIELD", "FACT_KIND", "HOLD_KIND",
           "INVALID", "MARK_FILE", "RECORDED", "SCHEMA", "SOURCES", "WAITING_STATUS", "arrived_fact", "awaiting",
           "read_request", "record_request", "request_path", "retain_awaiting_simulation_request",
           "retains_awaiting_simulation_request", "valid_body", "waiting_outcome", "waits_for_request"]
