"""The paid unit of remote team-scene compilation: dispatch through the allocator, then collect (plan 14's lane).

``blueprint-team-scene-packet-remote.service`` runs this from a 60 s timer and a ``PathChanged=`` watch on the
team stage's hand-off and shadow directories; it never compiles. For each hand-off it stages the plan's
inputs and the release source by digest, seals the attempt's descriptor and runs
``python -m blueprint_pipeline.paid_resource_allocator remote-cpu-job --action dispatch``, the only creator
of executions, under the owner's standing authority, the worst-case ledger and the live-execution cap. It
renews the lease on each advancing heartbeat, and once the execution is terminal it commits the attempt,
every step idempotent so a resume starts at the first one missing: compute-zero, promotion and one readback,
validation of the worker's result and team scene inputs record, landing of the whole ``<attempt>/scene`` tree
at the host's own path (assembled aside and renamed into place, every blob re-hashed), the record re-read on
the host, and the ``collected`` marker; then staging deletion and provider-zero, the sealed teardown, the
settlement and the lease's terminal state, which frees the capacity slot. The no-spend worker then records the
derived row and republishes the offer exactly as after a host compile.

An infrastructure failure (an OOM-killed or timed-out stage child, a lost receipt, a ``MemoryError`` or
``OSError`` in the compile) gets one fresh attempt once the failed one is compute-zero; then, or after a
refused dispatch, the request goes back to the no-spend unit as a fallback marker and compiles on the host.
Before a first dispatch and before a retry the capture's consent is read again: a withdrawn capture is never
dispatched (privacy is a hard stop). A shadow attempt compares the worker's output with the host's tree,
records parity and lands nothing. In ``host`` mode (an unset flag without this stage's config) the unit only
drains: it collects what already runs and returns undispatched hand-offs.
``summary.team_scene_packet.json`` says which mode a run was in and why.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import remote_cpu_job_allocator as allocator
from . import remote_cpu_job_lease as leases
from . import task_evaluation_episode_compilation_collector as shared
from . import team_scene_packet_remote as remote
from .cloud_run_jobs_client import job_resource_name
from .decision_evidence_contracts import canonical_digest
from .remote_cpu_job_contract import (
    build_descriptor,
    execution_name_of,
    job_id_for,
    stage_limits,
    validate_descriptor,
    validate_receipt,
)
from .remote_cpu_job_records import compute_zero_proven, replace_remote_cpu_record, validate_teardown
from .remote_cpu_output_archive import RemoteCpuArchiveError, land_subset
from .remote_cpu_transport import publish_release_source, stage_inputs
from .task_evaluation_configured_scene_object_store import (
    DEFAULT_KEY_PREFIX,
    TaskEvaluationConfiguredSceneObjectStoreError,
    read_remote_cpu_staging_object,
)

STAGE = remote.STAGE
ROW_SCHEMA_VERSION = "team_scene_packet_collection.v1"
SUMMARY_SCHEMA_VERSION = "team_scene_packet_remote_summary.v1"
SUMMARY_NAME = f"summary.{STAGE}.json"
QUEUE_ROOT_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT"
DEFAULT_QUEUE_ROOT = "/var/lib/blueprint/pipeline-control-plane/team-scene-packets"
CollectorError, _AttemptFailed = shared.CollectorError, shared._AttemptFailed
Allocate = Callable[[list[str]], Mapping[str, Any]]


def _after_step(step: str, attempt_id: str) -> None:
    """Called after each commit step's durable effect (a seam: the resume test crashes here)."""


@dataclass
class Collector:
    """What one paid-unit run touches: the allocator's provider runtime and this host's roots.

    It offers the attributes the episode collector's stage-agnostic helpers read (``runtime``, ``config``,
    ``jobs_root``, ``now``, ``action()``), which this module reuses for monitoring, promotion and teardown."""

    runtime: allocator.RemoteCpuRuntime
    config: dict[str, Any]
    jobs_root: Path
    queue_root: Path
    source_commit: str
    mode: str
    allocate: Allocate
    stage_release: Callable[[tuple[Any, str, str], str], Mapping[str, Any]]
    filesystem_root: Path = Path("/")
    summary: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)

    @property
    def now(self) -> float:
        return float(self.runtime.clock())

    def action(self) -> Any:
        return allocator._Action(runtime=self.runtime, config=self.config, root=self.jobs_root, execute=True,
                                 now=self.now)

    def host(self, worker_path: str) -> Path:
        return Path(self.filesystem_root) / str(worker_path).lstrip("/")


def plan_descriptor(plan: remote.RemotePlan, *, config: Mapping[str, Any], mode: str, attempt: int,
                    staged: list[Mapping[str, Any]], source: Mapping[str, Any], object_prefix: str,
                    nonce: str | None = None) -> dict[str, Any]:
    """Seal a plan into ``remote_cpu_job_descriptor.v1`` once its inputs and release source are staged."""

    limits = stage_limits(config, STAGE)
    return build_descriptor(
        config=config, stage=STAGE, mode=mode, attempt=attempt, queue_row=plan.queue_row,
        code={"source_commit": plan.source_commit, "image": plan.image, "environment_digest": plan.environment_digest,
              "source_archive": {name: source[name] for name in ("digest", "size_bytes", "uri")}},
        environment=plan.environment,
        inputs=[{**{name: row[name] for name in ("role", "contract_path", "digest", "size_bytes", "mode",
                                                "materialize_at")}, "uri": staged_row["uri"]}
                for row, staged_row in zip(plan.inputs, staged)],
        outputs={"output_root": plan.output_root, "declared_scratch": list(plan.declared_scratch),
                 "object_prefix": object_prefix},
        limits=limits, closure=plan.closure,
        spend={"worst_case_usd": allocator.worst_case_usd(limits=limits, rate_table=config["rate_table"]),
               "rate_table_digest": canonical_digest(config["rate_table"])}, nonce=nonce)


# ---------------------------------------------------------------------------------------------- records


def _row_path(c: Collector, name: str) -> Path:
    return c.jobs_root / "rows" / STAGE / name


def _row(c: Collector, name: str, queue_row: Mapping[str, Any]) -> dict[str, Any]:
    record = shared._read_json(_row_path(c, name))
    if record is not None and record.get("state_digest") == canonical_digest(record, digest_field="state_digest"):
        return record
    return {"schema_version": ROW_SCHEMA_VERSION, "queue_row": dict(queue_row), "attempt_id": None, "failures": 0,
            "first_failure_at_epoch": None, "last_failure": None, "promoted": None, "gave_up": None,
            "collected": None, "state_digest": ""}


def _save_row(c: Collector, row: dict[str, Any], **changes: Any) -> dict[str, Any]:
    previous = row["state_digest"] or None
    updated = {**row, **changes, "state_digest": ""}
    updated["state_digest"] = canonical_digest(updated, digest_field="state_digest")
    replace_remote_cpu_record(_row_path(c, row["queue_row"]["name"]), updated, previous_digest=previous,
                              digest_field="state_digest", mode=0o640)
    return updated


def _for_attempt(c: Collector, row: dict[str, Any], attempt_id: str) -> dict[str, Any]:
    if row["attempt_id"] == attempt_id:
        return row
    return _save_row(c, row, attempt_id=attempt_id, failures=0, first_failure_at_epoch=None, last_failure=None,
                     promoted=None)


def _record_failure(c: Collector, row: dict[str, Any], code: str) -> dict[str, Any]:
    """Count one promotion, landing or commit failure; past the budget the attempt itself fails (plan 14 §9)."""

    first = row["first_failure_at_epoch"] if row["first_failure_at_epoch"] is not None else c.now
    row = _save_row(c, row, failures=row["failures"] + 1, first_failure_at_epoch=first,
                    last_failure=shared._outcome(code))
    if row["failures"] >= shared.COLLECTION_RETRIES or c.now - first >= shared.COLLECTION_WINDOW_SECONDS:
        raise _AttemptFailed(shared._outcome(code))
    return row


def _lease(c: Collector, job_id: str) -> dict[str, Any] | None:
    return allocator._lease(c.jobs_root, job_id)


def _descriptor(c: Collector, attempt_id: str) -> tuple[dict[str, Any], Path]:
    path = c.jobs_root / "descriptors" / f"{attempt_id}.json"
    raw = shared._read_json(path)
    if raw is None:
        raise CollectorError("remote_team_scene_packet_descriptor_missing")
    return validate_descriptor(raw, config=c.config), path


def _allocate(c: Collector, action: str, descriptor_path: Path | None, label: str) -> Mapping[str, Any]:
    import secrets

    # A fresh result path per call: a result an earlier call left can never be read as this one's.
    out = c.jobs_root / "allocator" / f"{label}.{action}.{secrets.token_hex(8)}.json"
    argv = ["remote-cpu-job", "--action", action, "--stage", STAGE, "--lease", str(c.jobs_root), "--out", str(out),
            "--execute", *(["--descriptor", str(descriptor_path)] if descriptor_path is not None else [])]
    return c.allocate(argv)


def _job_id(marker: Mapping[str, Any]) -> str:
    return job_id_for(STAGE, marker["queue_row"]["name"])


def _claimed(c: Collector, marker: Mapping[str, Any]) -> Path:
    return c.queue_root / "processing" / f"{marker['request_id']}.json"


# ---------------------------------------------------------------------------------------------- dispatch


def _sealed_descriptors(c: Collector, plan: remote.RemotePlan, attempt: int) -> list[Path]:
    return sorted((c.jobs_root / "descriptors").glob(f"{job_id_for(STAGE, plan.queue_row['name'])}-a{attempt}-*.json"))


def _admission_refusal(c: Collector) -> str | None:
    """What the allocator's admission would refuse, read-only and before anything is staged: the standing
    authority, then the ledger's caps at this attempt's worst case. The allocator checks again."""

    try:
        authority, blockers = allocator.load_standing_authority(stage=STAGE, now=c.now)
        if authority is not None and not blockers:
            limits = stage_limits(c.config, STAGE)
            blockers = allocator.spend_ledger_blockers(authority=authority, now=c.now, worst_case_usd=(
                allocator.worst_case_usd(limits=limits, rate_table=c.config["rate_table"])))
    except Exception as exc:  # noqa: BLE001 - an authority no one can read refuses, typed, before any upload
        blockers = [f"remote_cpu_standing_authority_invalid:{type(exc).__name__}"]
    return blockers[0] if blockers else None


def _withdrawn(plan: remote.RemotePlan) -> bool:
    """Privacy is a hard stop: a capture whose consent was withdrawn is never shipped. Unreadable is withdrawn."""

    from .website_atlas_preview_store import capture_withdrawn

    try:
        return bool(capture_withdrawn(Path(plan.capture_root)))
    except Exception:  # noqa: BLE001 - consent that cannot be read is never assumed given
        return True


def _attempt_descriptor(c: Collector, plan: remote.RemotePlan, mode: str, attempt: int) -> tuple[dict[str, Any], Path]:
    """This attempt's sealed descriptor: the one already written, or staged and sealed now (never two)."""

    from .remote_cpu_job_records import write_remote_cpu_record

    existing = _sealed_descriptors(c, plan, attempt)
    if len(existing) > 1:
        raise CollectorError("remote_team_scene_packet_descriptor_ambiguous")
    if existing:
        return _descriptor(c, existing[0].stem)
    client, bucket, _ = c.runtime.object_store
    staged = stage_inputs(remote.input_sources(plan), client=client, bucket=bucket)
    source = c.stage_release(c.runtime.object_store, plan.source_commit)
    descriptor = plan_descriptor(plan, config=c.config, mode=mode, attempt=attempt, staged=staged, source=source,
                                 object_prefix=f"s3://{bucket}/{DEFAULT_KEY_PREFIX}")
    path = c.jobs_root / "descriptors" / f"{descriptor['attempt_id']}.json"
    write_remote_cpu_record(path, descriptor)
    return descriptor, path


def _dispatch(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan,
              attempt: int) -> dict[str, Any]:
    # An attempt already staged and sealed goes straight to the allocator; a new one is staged only if admitted.
    reason = None if _sealed_descriptors(c, plan, attempt) else _admission_refusal(c)
    if reason is None:
        descriptor, descriptor_path = _attempt_descriptor(c, plan, marker["mode"], attempt)
        result = _allocate(c, "dispatch", descriptor_path, descriptor["attempt_id"])
        status = str(result.get("status") or "blocked")
        if status in {"dispatched", "already_dispatched", "awaiting_capacity", "ambiguous_dispatch_unresolved",
                      "teardown_pending"}:
            return {"status": status, "attempt_id": descriptor["attempt_id"]}
        reason = next(iter(result.get("blockers") or []), "remote_cpu_dispatch_refused")
    # Refused before anything ran (or, for a failed mint, torn down by the allocator): the host compiles it.
    returned = _hand_back(c, path, marker, _lease(c, _job_id(marker)), reason, attempts=attempt - 1)
    return {**returned, "status": "dispatch_refused", "blocker": reason}


def _give_up(c: Collector, marker: Mapping[str, Any], *, reason: str, attempts: int) -> None:
    """No further attempt: an authoritative request goes back to the no-spend unit (plan 14 §10)."""

    name = marker["queue_row"]["name"]
    row = _row(c, name, marker["queue_row"])
    if row["gave_up"] is None:
        row = _save_row(c, row, gave_up={"reason": shared._outcome(reason), "at_epoch": c.now})
    # Only a request still claimed goes back: one the no-spend unit already finished needs nothing more.
    if (marker["mode"] == "authoritative" and _claimed(c, marker).is_file()
            and not remote.marker_path(c.jobs_root, "fallback", name).exists()
            and not remote.marker_path(c.jobs_root, "collected", name).exists()):
        attempt = Path(marker["plan"]["attempt_root"])
        for partial in attempt.glob(".scene.landing-*"):
            shutil.rmtree(partial, ignore_errors=True)
        remote.write_fallback(c.jobs_root, marker, reason=row["gave_up"]["reason"], attempts=attempts,
                              now=row["gave_up"]["at_epoch"])


def _hand_back(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any] | None, reason: str, *,
               attempts: int) -> dict[str, Any]:
    """No (further) dispatch for this hand-off: the request goes back and the hand-off goes.

    The fallback is written durably first, then the hand-off is removed. A hand-off stays only while an attempt
    that may have started still owes its provider-zero teardown; the lease then closes as ``fallback_host``."""

    if lease is not None and lease["state"] in {"claimed", "awaiting_capacity"}:
        descriptor_path = c.jobs_root / "descriptors" / f"{lease['attempt_id']}.json"
        if allocator._consumption_path(lease["attempt_id"]).exists() and descriptor_path.exists():
            _allocate(c, "reconcile", descriptor_path, lease["attempt_id"])  # a crashed dispatch: settle it at zero
        lease = _lease(c, lease["job_id"])
    if lease is not None and lease["state"] not in {"claimed", "awaiting_capacity", "expired", *leases.TERMINAL_STATES}:
        return {"status": lease["state"], "blocker": "remote_cpu_attempt_in_flight"}  # followed, never handed back
    _give_up(c, marker, reason=reason, attempts=attempts)
    if lease is None or lease["state"] in leases.TERMINAL_STATES:
        _retire(c, path, marker)
        return {"status": "returned_to_host", "reason": shared._outcome(reason)}
    return _close(c, path, marker, lease, terminal="fallback_host", outcome=reason)


def _retire(c: Collector, path: Path, marker: Mapping[str, Any]) -> None:
    """Remove a marker; a shadow's envelope goes with it (an authoritative one's goes when its request settles)."""

    if marker["mode"] == "shadow":
        remote.discard_envelope(c.jobs_root, marker["queue_row"]["name"])
    path.unlink(missing_ok=True)


def _handed_back(c: Collector, marker: Mapping[str, Any]) -> str | None:
    """Why this hand-off must never dispatch again: given up, handed back, or its request no longer claimed."""

    name = marker["queue_row"]["name"]
    row = _row(c, name, marker["queue_row"])
    if row["gave_up"] is not None:
        return row["gave_up"]["reason"]
    if remote.marker_path(c.jobs_root, "fallback", name).exists():
        return "remote_cpu_row_handed_back"
    if marker["mode"] == "authoritative" and not _claimed(c, marker).is_file():
        return "remote_cpu_row_not_claimed"
    return None


def _dispatchable(c: Collector, marker: Mapping[str, Any]) -> bool:
    return c.mode == "cloud_run" if marker["mode"] == "authoritative" else c.mode in remote.REMOTE_MODES


def _refusal(c: Collector, marker: Mapping[str, Any], plan: remote.RemotePlan) -> str | None:
    """Why a waiting marker must not be dispatched now; asked before a first dispatch and before a retry."""

    if _withdrawn(plan):
        return "team_scene_capture_withdrawn"
    if not _dispatchable(c, marker):
        return "remote_cpu_mode_rolled_back"
    if plan.source_commit != c.source_commit:
        return "remote_cpu_release_changed"
    return _handed_back(c, marker)


# ---------------------------------------------------------------------------------------------- commit


def _compute_zero(c: Collector, lease: dict[str, Any], descriptor: dict[str, Any]) -> bool:
    """Plan 14 §11 compute-zero for the current attempt, recorded on the lease once proven."""

    if lease["compute_zero_proven"] or not lease["dispatch_started"]:
        return True
    compute = allocator.prove_compute_zero(c.action(), lease, descriptor)
    if not compute_zero_proven(compute, worker_identity=lease["worker_identity"]):
        if compute["unfinished_executions_for_attempt"]:
            _allocate(c, "cancel", c.jobs_root / "descriptors" / f"{lease['attempt_id']}.json", lease["attempt_id"])
        return False
    leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state=None, now=c.now,
                      updates={"compute_zero": compute})
    return True


def _promotion(c: Collector, row: dict[str, Any], descriptor: dict[str, Any],
               output: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Promote once per attempt; a failure retries from staging, never by rerunning the execution."""

    if row["promoted"] is not None:
        return row["promoted"], row
    try:
        promoted = shared._promote(c, descriptor, output)
    except Exception as exc:  # noqa: BLE001 - a failed copy or readback retries from staging, within the budget
        return None, _record_failure(c, row, shared._failure(exc, "remote_cpu_promotion_failed"))
    return promoted, _save_row(c, row, promoted=promoted)


def _read_cas(c: Collector, reference: Mapping[str, Any], maximum: int) -> bytes:
    import hashlib

    client, bucket, _ = c.runtime.object_store
    body = client.get_object(Bucket=bucket, Key=str(reference["uri"]).removeprefix(f"s3://{bucket}/"))["Body"]
    data = body.read(maximum + 1)
    if len(data) != reference["size_bytes"] or "sha256:" + hashlib.sha256(data).hexdigest() != reference["digest"]:
        raise CollectorError("remote_team_scene_packet_readback_mismatch")
    return data


def _validate(c: Collector, plan: remote.RemotePlan, descriptor: dict[str, Any], result: Mapping[str, Any],
              index: Mapping[str, Any], archive: Mapping[str, Any]) -> list[str]:
    """The worker's result and its sealed team scene inputs record, read from the index and the archive by range,
    against what the host compile would have recorded: this request, this release, this revision, its packet in
    the tree, and the host's own runtime-source receipt."""

    from .team_scene_packet import CLAIM_CEILING, RECORD_FILENAME, SCHEMA

    reasons = remote.validate_result(result, request_id=plan.request_id,
                                     envelope_digest=plan.queue_row["envelope_digest"],
                                     source_commit=c.source_commit, output_root=plan.output_root)
    if reasons or result.get("status") != remote.COMPLETED:
        return reasons or ["result_status"]
    root = os.path.realpath(c.host(plan.output_root))
    entries = {row["path"]: row for row in index["entries"]}
    offsets = {row["blob"]: row["offset"] for row in index["blobs"]}

    def relative(value: Any) -> str | None:
        text = os.path.realpath(str(value or ""))
        return text[len(root) + 1:] if text.startswith(root + os.sep) else None

    entry = entries.get(RECORD_FILENAME)
    if entry is None or entry["origin"] != "archive" or entry["size_bytes"] > shared.MAX_JSON_BLOB_BYTES:
        return ["record"]
    try:
        with shared._range(c, archive["uri"], offsets[entry["blob"]], max(entry["size_bytes"], 1)) as stream:
            data = stream.read(entry["size_bytes"])
    except Exception as exc:  # noqa: BLE001 - a read that failed says nothing about the output
        raise shared._ReadFailed(f"remote_cpu_output_read_failed:{type(exc).__name__}") from None
    try:
        record = json.loads(data)
    except ValueError:
        return ["record"]
    compiled = result["compiled"]
    if (not isinstance(record, dict) or record.get("schema_version") != SCHEMA
            or record.get("claim_ceiling") != CLAIM_CEILING
            or record.get("inputs_digest") != canonical_digest(record, digest_field="inputs_digest")
            or record.get("inputs_digest") != compiled.get("inputs_digest")):
        return ["record"]
    if record.get("scene_revision_digest") != plan.scene_revision_digest:
        reasons.append("record_revision")
    packet = record.get("base_native_packet") or {}
    found = entries.get(relative(packet.get("path")) or "")
    if (record.get("construction_result") != packet or found is None
            or (found["blob"], found["size_bytes"]) != (packet.get("sha256"), packet.get("size_bytes"))):
        reasons.append("packet")
    runtime = record.get("runtime_source") or {}
    [receipt] = [row for row in plan.inputs if row["contract_path"] == "runtime_source_receipt_path"] or [{}]
    if (os.path.realpath(str(runtime.get("path") or "")) != os.path.realpath(str(receipt.get("host_path") or "-"))
            or (runtime.get("sha256"), runtime.get("size_bytes")) != (receipt.get("digest"), receipt.get("size_bytes"))):
        reasons.append("runtime_source")
    return reasons


def _land(c: Collector, plan: remote.RemotePlan, descriptor: dict[str, Any], index: Mapping[str, Any],
          archive: Mapping[str, Any]) -> dict[str, Any]:
    """Land the whole tree at the host's own ``<attempt>/scene``, assembled aside and renamed into place, then
    re-read its record on the host as the team registry will (``read_team_scene_inputs``)."""

    from .team_scene_packet import RECORD_FILENAME, read_team_scene_inputs

    destination = c.host(descriptor["outputs"]["output_root"])
    sources = {row["digest"]: row["host_path"] for row in plan.inputs}
    landed = land_subset(index=index, reader=lambda offset, length: shared._range(c, archive["uri"], offset, length),
                         host_sources=sources, destination_root=destination, selectors=["**"], member_store=None)
    try:
        read_team_scene_inputs(destination / RECORD_FILENAME)
    except (OSError, ValueError, KeyError):
        shutil.rmtree(destination, ignore_errors=True)
        raise _AttemptFailed("remote_cpu_output_invalid:team_scene_inputs_unreadable") from None
    return {name: landed[name] for name in ("state", "paths", "bytes", "linked_from_host")}


def _commit(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan, lease: dict[str, Any],
            descriptor: dict[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The commit order; each step is idempotent, so a resume starts at the first one missing."""

    name, attempt_id, result = plan.queue_row["name"], lease["attempt_id"], receipt["result"]
    abandoned = _row(c, name, plan.queue_row)["gave_up"]
    if abandoned is not None:  # a commit that could not finish: only its teardown is left
        return _close(c, path, marker, lease, terminal="blocked", outcome=abandoned["reason"])
    blocked = receipt["status"] == "blocked"
    terminal = "blocked" if blocked else "completed"
    if _decided(c, plan, lease, descriptor):
        return _close(c, path, marker, lease, terminal=terminal, outcome=result["status"])
    if not _compute_zero(c, lease, descriptor):
        return {"status": "collecting", "blocker": "remote_cpu_compute_zero_unproven"}
    _after_step("compute_zero", attempt_id)
    landed = None
    if not blocked:
        row = _for_attempt(c, _row(c, name, plan.queue_row), attempt_id)
        promoted, row = _promotion(c, row, descriptor, receipt["output"])
        if promoted is None:
            return {"status": "collecting", "blocker": row["last_failure"]}
        _after_step("promotion", attempt_id)
        try:
            index = json.loads(_read_cas(c, promoted["index"], shared.MAX_INDEX_BYTES))
            reasons = _validate(c, plan, descriptor, result, index, promoted["archive"])
        except Exception as exc:  # noqa: BLE001 - a read that failed says nothing about the output: retry it
            row = _record_failure(c, row, shared._failure(exc, "remote_cpu_output_read_failed"))
            return {"status": "collecting", "blocker": row["last_failure"]}
        if reasons:
            raise _AttemptFailed(f"remote_cpu_output_invalid:{reasons[0]}")
        _after_step("validation", attempt_id)
        try:
            landed = _land(c, plan, descriptor, index, promoted["archive"])
        except _AttemptFailed:
            raise
        except Exception as exc:  # noqa: BLE001 - a landing that failed is assembled aside and retried
            row = _record_failure(c, row, shared._failure(exc, "remote_cpu_landing_failed"))
            return {"status": "collecting", "blocker": row["last_failure"]}
        _after_step("landing", attempt_id)
    # Recorded on the row first, so the attempt stays decided once the no-spend unit consumes the marker.
    _save_row(c, _for_attempt(c, _row(c, name, plan.queue_row), attempt_id), collected=attempt_id)
    remote.write_collected(c.jobs_root, marker, attempt_id=attempt_id, result=result, landed=landed, now=c.now)
    _after_step("collected", attempt_id)
    return _close(c, path, marker, lease, terminal=terminal, outcome=result["status"])


def _compare_shadow(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan,
                    lease: dict[str, Any], descriptor: dict[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The host stayed authoritative: compare the worker's tree with the host's, record parity, land nothing."""

    import hashlib

    if not _compute_zero(c, lease, descriptor):
        return {"status": "collecting", "blocker": "remote_cpu_compute_zero_unproven"}
    abandoned = _row(c, plan.queue_row["name"], plan.queue_row)["gave_up"]
    if abandoned is not None:
        return _close(c, path, marker, lease, terminal="blocked", outcome=abandoned["reason"])
    recorded = shared._read_json(c.jobs_root / "parity" / STAGE / f"{lease['attempt_id']}.json")
    if recorded is not None:
        return _close(c, path, marker, lease, terminal="shadow_compared", outcome=f"shadow_parity_{recorded['parity']}")
    worker, host = receipt["result"], dict(marker.get("host") or {})
    mismatches: list[str] = []
    if receipt["status"] == "succeeded" and host.get("inputs_digest"):
        if (worker.get("compiled") or {}).get("inputs_digest") != host["inputs_digest"]:
            mismatches.append("result")
        client, bucket, _ = c.runtime.object_store
        raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "index.json",
                                             maximum_size_bytes=shared.MAX_INDEX_BYTES, client=client, bucket=bucket)
        output = receipt["output"]["index"]
        if raw is None or ("sha256:" + hashlib.sha256(raw).hexdigest(), len(raw)) != (output["digest"],
                                                                                    output["size_bytes"]):
            mismatches.append("index")
        else:
            mismatches += shared._tree_mismatches(c.host(plan.output_root), json.loads(raw))
        parity = "passed" if not mismatches else "failed"
    elif host.get("inputs_digest"):
        parity, mismatches = "failed", ["status"]  # the host compiled and the worker did not
    else:
        parity = "inconclusive"
    remote.record_shadow_parity(c.jobs_root, {
        "closure_class": plan.closure["class"], "attempt_id": lease["attempt_id"], "queue_row": dict(plan.queue_row),
        "request_id": plan.request_id, "image": plan.image, "host_environment_digest": plan.host_environment_digest,
        "worker_environment_digest": receipt["environment"]["environment_digest"],
        "cpu_class": receipt["environment"]["cpu_class"], "parity": parity, "mismatches": mismatches[:16],
        "source_commit": plan.source_commit, "compared_at_epoch": c.now})
    return _close(c, path, marker, lease, terminal="shadow_compared", outcome=f"shadow_parity_{parity}")


def _collect(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan, lease: dict[str, Any],
             descriptor: dict[str, Any]) -> dict[str, Any]:
    # The receipt is kept on the host once fenced: provider-zero deletes staging before the write URLs expire,
    # so a resumed commit must never depend on staging.
    from .remote_cpu_job_records import write_remote_cpu_record

    recorded = c.jobs_root / "receipts" / f"{lease['attempt_id']}.json"
    raw: Any = shared._read_json(recorded)
    if raw is None:
        client, bucket, _ = c.runtime.object_store
        try:
            raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "receipt.json",
                                                 maximum_size_bytes=shared.MAX_RECEIPT_BYTES, client=client, bucket=bucket)
        except TaskEvaluationConfiguredSceneObjectStoreError:
            if c.now < lease["deadlines"]["hard_deadline_epoch"] + shared.RECEIPT_UNKNOWN_GRACE_SECONDS:
                return {"status": "collecting", "blocker": "remote_cpu_receipt_state_unknown"}
            raw = None
        if raw is None:
            return shared._fail(c, lease, "remote_cpu_receipt_missing")
        try:
            raw = json.loads(raw)
        except ValueError:
            return shared._fail(c, lease, "remote_cpu_receipt_invalid")
    try:
        verdict = validate_receipt(raw, descriptor=descriptor, execution_name=execution_name_of(lease["worker_identity"]))
    except (ValueError, TypeError):  # RemoteCpuContractError is a ValueError
        return shared._fail(c, lease, "remote_cpu_receipt_invalid")
    write_remote_cpu_record(recorded, verdict["receipt"])
    if verdict["outcome"] == "infrastructure_failed":
        return shared._fail(c, lease, verdict["infrastructure_failures"][0])
    try:
        if descriptor["mode"] == "shadow":
            return _compare_shadow(c, path, marker, plan, lease, descriptor, verdict["receipt"])
        return _commit(c, path, marker, plan, lease, descriptor, verdict["receipt"])
    except _AttemptFailed as exc:
        return shared._fail(c, _lease(c, lease["job_id"]), str(exc))
    except Exception as exc:  # noqa: BLE001 - a named conflict is abandoned; anything else is bounded, then abandoned
        current, code = _lease(c, lease["job_id"]), _typed(exc)
        if current["state"] != "collecting" or (current["dispatch_started"] and not current["compute_zero_proven"]):
            raise  # nothing is committed before compute-zero: the next run tries again
        if _decided(c, plan, current, descriptor):
            raise  # this attempt's outcome is written: the next run resumes _close, the slot held by design
        if code.startswith("remote_cpu_landing_conflict:") or code.startswith("remote_cpu_record_conflict:"):
            return _abandon_commit(c, path, marker, current, f"remote_cpu_commit_failed:{code}")
        row = _for_attempt(c, _row(c, plan.queue_row["name"], plan.queue_row), current["attempt_id"])
        try:
            row = _record_failure(c, row, shared._failure(exc, "remote_cpu_commit_failed"))
        except _AttemptFailed as spent:
            return _abandon_commit(c, path, marker, current, str(spent))
        return {"status": "collecting", "blocker": row["last_failure"]}


def _decided(c: Collector, plan: remote.RemotePlan, lease: Mapping[str, Any], descriptor: Mapping[str, Any]) -> bool:
    """Whether this attempt's outcome is already written: its collected marker, or, in shadow, its parity."""

    if descriptor["mode"] == "shadow":
        return (c.jobs_root / "parity" / STAGE / f"{lease['attempt_id']}.json").is_file()
    # The row records the attempt before its collected marker, which the no-spend unit consumes once it settles.
    return _row(c, plan.queue_row["name"], plan.queue_row).get("collected") == lease["attempt_id"]


def _abandon_commit(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any],
                    reason: str) -> dict[str, Any]:
    """A commit that cannot finish must not hold the attempt's slot forever: the request, if still claimed, goes
    back to the host; the attempt is torn down to provider-zero and settled; the lease ends ``blocked``."""

    _give_up(c, marker, reason=reason, attempts=lease["attempt"])
    return _close(c, path, marker, lease, terminal="blocked", outcome=reason)


# ---------------------------------------------------------------------------------------------- teardown


def _close(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any], *, terminal: str,
           outcome: str | None = None) -> dict[str, Any]:
    """Tear every started attempt down to provider-zero and settle it, then turn the lease terminal."""

    if outcome is not None and lease["outcome"] != shared._outcome(outcome):
        lease = leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state=None,
                                  now=c.now, updates={"outcome": shared._outcome(outcome)})
    pending, failures = 0, []
    for attempt in [*lease["prior_attempts"], lease]:
        if not attempt["dispatch_started"]:
            continue
        if attempt["provider_zero_proven"] and shared._settled(attempt["attempt_id"]):
            continue  # torn down and settled: nothing of it is left to read or do
        try:
            descriptor, _ = _descriptor(c, attempt["attempt_id"])
            if attempt["provider_zero_proven"]:
                sealed = shared._read_json_present(c.jobs_root / "teardowns" / f"{attempt['attempt_id']}.json")
                if sealed is None:
                    raise CollectorError("remote_cpu_teardown_record_missing")
                record = validate_teardown(sealed)
            else:
                record = shared._seal_teardown(c, attempt, descriptor)
                if record is None:
                    pending += 1
                    continue
                lease = leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"],
                                          to_state=None, now=c.now, updates={"teardown": record})
            if not shared._settled(attempt["attempt_id"]):
                shared._settle(c, attempt, descriptor, record)
        except Exception as exc:  # noqa: BLE001 - one attempt's failure never stops another's teardown
            failures.append(exc)
    if failures:
        raise failures[0]
    if pending:
        c.summary["teardown_unproven"] = c.summary.get("teardown_unproven", 0) + pending
        return {"status": lease["state"], "blocker": "remote_cpu_provider_zero_unproven"}
    _after_step("provider_zero", lease["attempt_id"])
    if lease["state"] not in leases.TERMINAL_STATES:
        lease = leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state=terminal,
                                  now=c.now)
    return _finish(c, path, marker, lease)


def _finish(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any]) -> dict[str, Any]:
    """A terminal lease: a fallback or abandoned authoritative request goes back to the host; the marker goes."""

    if lease["state"] in {"fallback_host", "abandoned_dispatch"}:
        _give_up(c, marker, reason=lease["outcome"] or lease["state"], attempts=lease["attempt"])
    if marker["mode"] == "shadow" and lease["state"] != "shadow_compared":
        _record_abandoned(c, marker, lease)
    _retire(c, path, marker)
    return {"status": lease["state"], "attempt_id": lease["attempt_id"]}


def _record_abandoned(c: Collector, marker: Mapping[str, Any], lease: Mapping[str, Any]) -> None:
    """A shadow that ran but was never compared is an outcome; one that never ran spent nothing and is none."""

    record = c.jobs_root / "parity" / STAGE / f"{lease['attempt_id']}.json"
    if record.exists() or not any(attempt["dispatch_started"] for attempt in [*lease["prior_attempts"], lease]):
        return
    plan = remote.RemotePlan.from_record(marker["plan"])
    remote.record_shadow_parity(c.jobs_root, {
        "closure_class": plan.closure["class"], "attempt_id": lease["attempt_id"], "queue_row": dict(plan.queue_row),
        "request_id": plan.request_id, "image": plan.image, "host_environment_digest": plan.host_environment_digest,
        "worker_environment_digest": None, "cpu_class": None, "parity": "abandoned", "mismatches": [],
        "outcome": shared._outcome(lease["outcome"] or lease["state"]), "source_commit": plan.source_commit,
        "compared_at_epoch": c.now})


def _after_expiry(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan,
                  lease: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
    """Compute-zero first; then one fresh attempt, or the host (plan 14 §10). A retry is refused as a first
    dispatch is (a withdrawn capture, a rolled-back mode, a new release, a request handed back)."""

    if not _compute_zero(c, lease, descriptor):
        return {"status": "expired", "blocker": "remote_cpu_compute_zero_unproven"}
    lease = _lease(c, lease["job_id"])
    refused = _refusal(c, marker, plan)
    if refused is None and lease["attempt"] < c.config["max_attempts"]:
        return _dispatch(c, path, marker, plan, lease["attempt"] + 1)
    _give_up(c, marker, reason=refused or lease["outcome"] or "remote_cpu_attempts_exhausted",
             attempts=lease["attempt"])
    return _close(c, path, marker, lease, terminal="fallback_host")


def _advance(c: Collector, path: Path, marker: Mapping[str, Any]) -> dict[str, Any]:
    plan = remote.RemotePlan.from_record(marker["plan"])
    lease = _lease(c, _job_id(marker))
    state = None if lease is None else lease["state"]
    if state in {None, "claimed", "awaiting_capacity"}:
        retired = _refusal(c, marker, plan)
        if retired is not None:
            prior = 0 if lease is None else sum(attempt["dispatch_started"] for attempt in lease["prior_attempts"])
            return _hand_back(c, path, marker, lease, retired, attempts=prior)
        return _dispatch(c, path, marker, plan, 1 if lease is None else lease["attempt"])
    descriptor, descriptor_path = _descriptor(c, lease["attempt_id"])
    if state in {"dispatching", "expired"} and lease["worker_identity"] is None and lease["dispatch_started"]:
        # A dispatch whose response was lost, or that never started: the allocator lists every execution of the
        # attempt, binds a late one, deletes the transport and tears it down (abandoned_dispatch, plan 14 §10).
        started = (lease["deadlines"] or {}).get("dispatch_started_at_epoch", 0.0)
        if state == "dispatching" and c.now < started + allocator.RECONCILE_AFTER_SECONDS:
            return {"status": state}
        return {"status": str(_allocate(c, "reconcile", descriptor_path, lease["attempt_id"]).get("status"))}
    if state == "dispatching":
        return {"status": state}
    if state in {"dispatched", "running"}:
        return shared._monitor(c, lease, descriptor)
    if state == "collecting":
        return _collect(c, path, marker, plan, lease, descriptor)
    if state == "expired":
        return _after_expiry(c, path, marker, plan, lease, descriptor)
    return _finish(c, path, marker, lease)


# ---------------------------------------------------------------------------------------------- the run


def _observe_job_image(c: Collector) -> None:
    """Record the job template's image against the config's, for drift (plan 14 §15)."""

    entry = c.config["stages"][STAGE]
    try:
        job = c.runtime.cloud_run.get_job(job_resource_name(project=c.config["project"], region=c.config["region"],
                                                            job=entry["job"]))
        containers = ((job.get("template") or {}).get("template") or {}).get("containers") or [{}]
        image = str(containers[0].get("image") or "")
    except Exception:  # noqa: BLE001 - an unobserved job is not drift; dispatch checks the job itself
        return
    record = remote.record_job_image(c.jobs_root, job_image=image, config_image=entry["image"], now=c.now)
    c.summary["drift"] = int(record["drift"] or remote.image_drift(c.jobs_root, config_image=entry["image"]))


def _typed(exc: BaseException) -> str:
    code = getattr(exc, "code", None) or (getattr(exc, "reasons", None) or [None])[0]
    if isinstance(exc, (CollectorError, TaskEvaluationConfiguredSceneObjectStoreError, RemoteCpuArchiveError)):
        code = str(exc).split(" ")[0]
    return shared._outcome(str(code)) if code else f"remote_team_scene_packet_failed:{type(exc).__name__}"


def _parity_counts(c: Collector) -> dict[str, int]:
    outcomes = ("passed", "failed", "inconclusive", "abandoned")
    counts = dict.fromkeys(outcomes, 0)
    for record_path in sorted((c.jobs_root / "parity" / STAGE).glob("*.json")):
        record = shared._read_json(record_path) or {}
        counts[record["parity"] if record.get("parity") in outcomes else "failed"] += 1
    return counts


def _each_marker(c: Collector, step: Callable[[Path, Mapping[str, Any]], dict[str, Any]]) -> None:
    for kind in ("authoritative", "shadow"):
        for path, marker in remote.markers(c.jobs_root, kind):
            if marker is None or marker.get("schema_version") != remote.HANDOFF_SCHEMA_VERSION:
                c.summary["blockers"].append(f"remote_team_scene_packet_marker_unreadable:{path.name}")
                continue
            try:
                c.summary["rows"][path.name] = step(path, marker)
            except Exception as exc:  # noqa: BLE001 - one row's failure never stops the others; its cause stays typed
                c.summary["rows"][path.name] = {"status": "blocked", "blocker": _typed(exc)}


def _start_summary(c: Collector, blockers: list[str]) -> None:
    execution = {"requested": c.execution.get("requested"), "effective": c.mode, "reason": c.execution.get("reason"),
                 "findings": list(c.execution.get("findings") or [])}
    c.summary = {"schema_version": SUMMARY_SCHEMA_VERSION, "stage": STAGE, "mode": c.mode, "observed_at_epoch": c.now,
                 "execution_mode": execution, "rows": {}, "drift": 0, "teardown_unproven": 0, "orphans_cancelled": 0,
                 "blockers": list(blockers)}


def _write_summary(c: Collector) -> dict[str, Any]:
    census = leases.slot_census(c.jobs_root)
    c.summary.update(slots_in_use=census["slots_in_use"], unreadable_leases=census["unreadable"],
                     parity=_parity_counts(c))
    summary_path = c.jobs_root / SUMMARY_NAME
    summary_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(c.summary, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o644)
    os.replace(temporary, summary_path)
    return c.summary


def run_collector(c: Collector) -> dict[str, Any]:
    """One paid-unit run: every hand-off and shadow marker advanced one step, then a sweep and the summary."""

    _start_summary(c, [])
    _observe_job_image(c)
    _each_marker(c, lambda path, marker: _advance(c, path, marker))
    # After every row was followed, so a finished execution is collected rather than expired.
    c.summary["expired"] = leases.expire_stale(c.jobs_root, now=c.now)
    swept = _allocate(c, "sweep", None, "stage")
    c.summary["orphans_cancelled"] = len(swept.get("cancelled") or [])
    return _write_summary(c)


def drain_without_provider(c: Collector, blockers: list[str]) -> dict[str, Any]:
    """With no provider connection (the config or the dispatcher credential is unusable), every hand-off that never
    dispatched, consumed nothing and owes no teardown still goes back, and the summary is still written. Anything
    that may have started keeps its hand-off for a connected run's teardown."""

    _start_summary(c, blockers)
    reason = shared._outcome(f"remote_cpu_provider_unavailable:{blockers[0]}")

    def untouched(path: Path, marker: Mapping[str, Any]) -> dict[str, Any]:
        lease = _lease(c, _job_id(marker))
        if lease is not None and (lease["state"] not in {"claimed", "awaiting_capacity"} or lease["prior_attempts"]
                                  or allocator._consumption_path(lease["attempt_id"]).exists()):
            return {"status": "held_for_provider", "lease_state": lease["state"]}
        if lease is not None:
            leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state="fallback_host",
                              now=c.now, updates={"outcome": reason})
        _give_up(c, marker, reason=reason, attempts=0)
        _retire(c, path, marker)
        return {"status": "returned_to_host", "reason": reason}

    _each_marker(c, untouched)
    c.summary["expired"] = leases.expire_stale(c.jobs_root, now=c.now)
    return _write_summary(c)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch and collect remote team-scene compilations")
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--source-commit")
    parser.add_argument("--jobs-root", default=os.getenv(remote.JOBS_ROOT_ENV, remote.DEFAULT_JOBS_ROOT))
    args = parser.parse_args(argv)
    execution = remote.resolve_execution_mode()
    mode, findings = execution["effective"], execution["findings"]
    try:
        config, blockers = allocator.load_remote_cpu_config()
    except Exception as exc:  # noqa: BLE001 - a config no loader can validate is no provider: drain
        config, blockers = {}, [f"remote_cpu_config_invalid:{type(exc).__name__}"]
    if not blockers and STAGE not in config.get("stages", {}):
        blockers = [f"remote_cpu_config_invalid:stage_missing:{STAGE}"]
    runtime = allocator.RemoteCpuRuntime()
    blockers = blockers or allocator._connect(runtime, config)
    if not args.source_commit:
        blockers = [*blockers, "remote_team_scene_packet_commit_missing"]
    repository = Path(__file__).resolve().parents[2]
    collector = Collector(
        runtime=runtime, config=config, jobs_root=Path(args.jobs_root),
        queue_root=Path(os.environ.get(QUEUE_ROOT_ENV) or DEFAULT_QUEUE_ROOT), source_commit=args.source_commit or "",
        mode=mode, execution=execution, allocate=shared.subprocess_allocate,
        stage_release=lambda store, commit: publish_release_source(repository=repository, source_commit=commit,
                                                                   client=store[0], bucket=store[1]))
    if blockers:
        summary = drain_without_provider(collector, blockers)
        print(json.dumps({"status": "blocked", "blockers": blockers, "rows": len(summary["rows"]),
                          "findings": findings}, sort_keys=True))
        return 0
    summary = run_collector(collector)
    print(json.dumps({"status": "collected", "rows": len(summary["rows"]), "findings": findings}, sort_keys=True))
    return 0


__all__ = ["Collector", "CollectorError", "drain_without_provider", "main", "plan_descriptor", "run_collector"]


if __name__ == "__main__":
    raise SystemExit(main())
