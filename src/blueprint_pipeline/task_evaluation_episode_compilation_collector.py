"""The paid unit of remote episode compilation: dispatch through the allocator, then collect (plan 14 §1, §9-§12).

``blueprint-task-evaluation-episode-compilation-remote.service`` runs this from a 60 s timer and a
``PathChanged=`` watch on the hand-off and shadow directories; it never compiles.  For each hand-off it
stages the plan's inputs and the release source by digest, seals the attempt's descriptor and runs
``python -m blueprint_pipeline.paid_resource_allocator remote-cpu-job --action dispatch``, the only creator
of executions.  It renews the lease on each advancing heartbeat, and once the execution is terminal it
commits the attempt in plan 14 §9's order, every step idempotent so a resume starts at the first one
missing: compute-zero, promotion and one readback, validation, landing of only the consumer subset, the
output pointer, the pin, the result, the row move; then staging deletion and provider-zero, the sealed
teardown and the lease's terminal state, which frees the capacity slot.

An infrastructure failure gets one fresh attempt once the failed one is compute-zero; then, or after a
refused dispatch, the row goes back to the no-spend unit as a fallback marker.  A shadow attempt compares
the worker's result and output with the host's, records parity for its closure class and lands nothing.
In ``host`` mode the unit only drains: it collects what already runs and returns undispatched hand-offs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import remote_cpu_job_allocator as allocator
from . import remote_cpu_job_lease as leases
from . import task_evaluation_episode_compilation_remote as remote
from .cloud_run_jobs_client import execution_resource_name, execution_seconds, job_resource_name, run_target
from .control_plane_disk_budget import reserve_control_plane_disk
from .control_plane_storage_pins import ControlPlaneStoragePinError, write_storage_pin
from .decision_evidence_contracts import canonical_digest
from .remote_cpu_job_contract import (
    EXECUTION_PROVIDER,
    MAX_RECEIPT_BYTES,
    RemoteCpuContractError,
    execution_name_of,
    job_id_for,
    validate_descriptor,
    validate_receipt,
)
from .remote_cpu_job_records import (
    compute_zero_proven,
    pointer_record,
    replace_remote_cpu_record,
    teardown_record,
    validate_teardown,
    write_remote_cpu_record,
)
from .remote_cpu_output_archive import RemoteCpuArchiveError, land_subset, verify_blobs_stream
from .remote_cpu_transport import publish_release_source, stage_inputs
from .task_evaluation_configured_scene_object_store import (
    DEFAULT_KEY_PREFIX,
    TaskEvaluationConfiguredSceneObjectStoreError,
    copy_remote_cpu_staging_to_cas,
    discard_remote_cpu_output_object,
    read_remote_cpu_staging_object,
)
from .task_evaluation_episode_compilation_worker import INPUT_ROOT_ENV, OUTPUT_ROOT_ENV, QUEUE_ROOT_ENV
from .task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive
from .task_evaluation_scene_construction_queue import _canonical_bytes

STAGE = remote.STAGE
ROW_SCHEMA_VERSION = "task_evaluation_episode_compilation_collection.v1"
SUMMARY_SCHEMA_VERSION = "task_evaluation_episode_compilation_remote_summary.v1"
RESULT_SCHEMA_VERSION = "task_evaluation_episode_compilation_result.v1"
POINTER_SUFFIX = ".remote-output.v1.json"
CONSUMER_SUBSET = "episode_compilation_consumer.v1"
ADAPTER_RESULT = "native-arena-adapter/task_evaluation_native_arena_adapter_result.v1.json"
# Promotion and landing retry from staging and CAS this often, over this long, before the attempt fails.
COLLECTION_RETRIES = 6
COLLECTION_WINDOW_SECONDS = 3600
# A receipt whose state is unknown is read again, never taken as success, until this long after the deadline.
RECEIPT_UNKNOWN_GRACE_SECONDS = 3600
LANDING_MARGIN_BYTES = 16 * 1024 * 1024
MAX_INDEX_BYTES = 64 * 1024 * 1024
MAX_JSON_BLOB_BYTES = 4 * 1024 * 1024
Allocate = Callable[[list[str]], Mapping[str, Any]]


class CollectorError(RuntimeError):
    """A typed refusal of the collector; the message is the blocker and never carries a URL."""


class _AttemptFailed(CollectorError):
    """This attempt is an infrastructure failure: it expires, then retries or falls back."""


class _ReadFailed(CollectorError):
    """A promoted object could not be read now: the step retries, within the collection budget."""


def _after_step(step: str, attempt_id: str) -> None:
    """Called after each commit step's durable effect (a seam: the resume test crashes here)."""


@dataclass
class Collector:
    """What one paid-unit run touches: the allocator's provider runtime and this host's roots."""

    runtime: allocator.RemoteCpuRuntime
    config: dict[str, Any]
    jobs_root: Path
    queue_root: Path
    outputs_root: Path
    source_commit: str
    mode: str
    allocate: Allocate
    stage_release: Callable[[tuple[Any, str, str], str], Mapping[str, Any]]
    filesystem_root: Path = Path("/")
    disk_reservation_root: Path | None = None
    storage_pins_root: Path | None = None
    summary: dict[str, Any] = field(default_factory=dict)

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

    from .remote_cpu_job_contract import build_descriptor, stage_limits

    limits = stage_limits(config, STAGE, allowed_cpu_classes=plan.allowed_cpu_classes)
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


def _read_json(path: Path) -> dict[str, Any] | None:
    return remote._read_record(path)


def _row_path(c: Collector, name: str) -> Path:
    return c.jobs_root / "rows" / STAGE / name


def _row(c: Collector, name: str, queue_row: Mapping[str, Any]) -> dict[str, Any]:
    record = _read_json(_row_path(c, name))
    if record is not None and record.get("state_digest") == canonical_digest(record, digest_field="state_digest"):
        return record
    return {"schema_version": ROW_SCHEMA_VERSION, "queue_row": dict(queue_row), "attempt_id": None, "failures": 0,
            "first_failure_at_epoch": None, "last_failure": None, "promoted": None, "gave_up": None,
            "state_digest": ""}


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


def _lease(c: Collector, job_id: str) -> dict[str, Any] | None:
    return allocator._lease(c.jobs_root, job_id)


def _descriptor(c: Collector, attempt_id: str) -> tuple[dict[str, Any], Path]:
    path = c.jobs_root / "descriptors" / f"{attempt_id}.json"
    raw = _read_json(path)
    if raw is None:
        raise CollectorError("remote_episode_compilation_descriptor_missing")
    return validate_descriptor(raw, config=c.config), path


def _allocate(c: Collector, action: str, descriptor_path: Path | None, label: str) -> Mapping[str, Any]:
    out = c.jobs_root / "allocator" / f"{label}.{action}.json"
    argv = ["remote-cpu-job", "--action", action, "--stage", STAGE, "--lease", str(c.jobs_root), "--out", str(out),
            "--execute", *(["--descriptor", str(descriptor_path)] if descriptor_path is not None else [])]
    return c.allocate(argv)


def subprocess_allocate(argv: list[str]) -> Mapping[str, Any]:
    """Run the canonical allocator subcommand (AGENTS.md) and read its door-safe ``--out`` result."""

    out = Path(argv[argv.index("--out") + 1])
    completed = subprocess.run([sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", *argv],
                               stdin=subprocess.DEVNULL, capture_output=True, check=False, timeout=900)
    result = _read_json(out)
    if result is None:
        return {"status": "blocked", "blockers": [f"remote_cpu_allocator_result_unreadable:exit_{completed.returncode}"]}
    return result


# ---------------------------------------------------------------------------------------------- dispatch


def _attempt_descriptor(c: Collector, plan: remote.RemotePlan, mode: str, attempt: int) -> tuple[dict[str, Any], Path]:
    """This attempt's sealed descriptor: the one already written, or staged and sealed now (never two)."""

    job_id = job_id_for(STAGE, plan.queue_row["name"])
    existing = sorted((c.jobs_root / "descriptors").glob(f"{job_id}-a{attempt}-*.json"))
    if len(existing) > 1:
        raise CollectorError("remote_episode_compilation_descriptor_ambiguous")
    if existing:
        return _descriptor(c, existing[0].stem)
    client, bucket, _ = c.runtime.object_store
    staged = stage_inputs(remote.input_sources(plan, c.queue_root), client=client, bucket=bucket)
    source = c.stage_release(c.runtime.object_store, plan.source_commit)
    descriptor = plan_descriptor(plan, config=c.config, mode=mode, attempt=attempt, staged=staged, source=source,
                                 object_prefix=f"s3://{bucket}/{DEFAULT_KEY_PREFIX}")
    path = c.jobs_root / "descriptors" / f"{descriptor['attempt_id']}.json"
    write_remote_cpu_record(path, descriptor)
    return descriptor, path


def _dispatch(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan,
              attempt: int) -> dict[str, Any]:
    descriptor, descriptor_path = _attempt_descriptor(c, plan, marker["mode"], attempt)
    result = _allocate(c, "dispatch", descriptor_path, descriptor["attempt_id"])
    status = str(result.get("status") or "blocked")
    if status in {"dispatched", "already_dispatched", "awaiting_capacity", "ambiguous_dispatch_unresolved",
                  "teardown_pending"}:
        return {"status": status, "attempt_id": descriptor["attempt_id"]}
    # Refused before anything ran (or, for a failed mint, torn down by the allocator): the host compiles it.
    reason = next(iter(result.get("blockers") or []), "remote_cpu_dispatch_refused")
    returned = _hand_back(c, path, marker, _lease(c, descriptor["job_id"]), reason, attempts=attempt - 1)
    return {**returned, "status": "dispatch_refused", "blocker": reason}


def _outcome(text: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "_.:/-" else "_" for character in text)
    return cleaned[:256] or "remote_cpu_unknown"


def _give_up(c: Collector, marker: Mapping[str, Any], *, reason: str, attempts: int) -> None:
    """No further attempt: an authoritative row goes back to the no-spend unit (plan 14 §10)."""

    name = marker["queue_row"]["name"]
    row = _row(c, name, marker["queue_row"])
    if row["gave_up"] is None:
        row = _save_row(c, row, gave_up={"reason": _outcome(reason), "at_epoch": c.now})
    # Only a row still claimed goes back: one the no-spend unit already compiled needs nothing more.
    if (marker["mode"] == "authoritative" and (c.queue_root / "processing" / name).is_file()
            and not remote.marker_path(c.jobs_root, "fallback", name).exists()):
        remote.write_fallback(c.jobs_root, marker["queue_row"], reason=row["gave_up"]["reason"], attempts=attempts,
                              now=row["gave_up"]["at_epoch"])
        for partial in c.outputs_root.glob(f".{marker['plan']['compilation_id']}.landing-*"):
            shutil.rmtree(partial, ignore_errors=True)


def _hand_back(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any] | None, reason: str, *,
               attempts: int) -> dict[str, Any]:
    """No (further) dispatch for this hand-off: the row goes back and the hand-off goes (review C1).

    The fallback is written durably first, then the hand-off is removed, so a crash between them leaves a
    hand-off that ``_advance`` retires rather than dispatches.  A hand-off stays only while an attempt that
    may have started still owes its provider-zero teardown; the lease then closes as ``fallback_host``.
    """

    if lease is not None and lease["state"] in {"claimed", "awaiting_capacity"}:
        descriptor_path = c.jobs_root / "descriptors" / f"{lease['attempt_id']}.json"
        if allocator._consumption_path(lease["attempt_id"]).exists() and descriptor_path.exists():
            _allocate(c, "reconcile", descriptor_path, lease["attempt_id"])  # a crashed dispatch: settle it at zero
        lease = _lease(c, lease["job_id"])
    if lease is not None and lease["state"] not in {"claimed", "awaiting_capacity", "expired", *leases.TERMINAL_STATES}:
        return {"status": lease["state"], "blocker": "remote_cpu_attempt_in_flight"}  # followed, never handed back
    _give_up(c, marker, reason=reason, attempts=attempts)
    if lease is None or lease["state"] in leases.TERMINAL_STATES:
        path.unlink(missing_ok=True)
        return {"status": "returned_to_host", "reason": _outcome(reason)}
    return _close(c, path, marker, lease, terminal="fallback_host", outcome=reason)


def _handed_back(c: Collector, marker: Mapping[str, Any]) -> str | None:
    """Why this hand-off must never dispatch again: its row was given up, handed back, or is no longer claimed."""

    name = marker["queue_row"]["name"]
    row = _row(c, name, marker["queue_row"])
    if row["gave_up"] is not None:
        return row["gave_up"]["reason"]
    if remote.marker_path(c.jobs_root, "fallback", name).exists():
        return "remote_cpu_row_handed_back"
    if marker["mode"] == "authoritative" and not (c.queue_root / "processing" / name).is_file():
        return "remote_cpu_row_not_claimed"
    return None


def _dispatchable(c: Collector, marker: Mapping[str, Any]) -> bool:
    return c.mode == "cloud_run" if marker["mode"] == "authoritative" else c.mode in {"cloud_run", "cloud_run_shadow"}


# ---------------------------------------------------------------------------------------------- follow


def _monitor(c: Collector, lease: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
    """Renew the lease on each advancing heartbeat; a terminal execution moves it to collecting."""

    job, identity = run_target(descriptor)["job"], lease["worker_identity"]
    execution = c.runtime.cloud_run.get_execution(execution_resource_name(job, execution_name_of(identity)))
    client, bucket, _ = c.runtime.object_store
    try:
        raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "heartbeat.json",
                                             maximum_size_bytes=16 * 1024, client=client, bucket=bucket)
        if raw is not None:
            leases.observe_heartbeat(c.jobs_root, lease["job_id"], json.loads(raw),
                                     execution_running=bool(execution.get("runningCount")), now=c.now)
    except (ValueError, TaskEvaluationConfiguredSceneObjectStoreError):
        pass  # no heartbeat, an unreadable or a fenced one: none renews the lease
    if execution.get("completionTime"):
        leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state="collecting",
                          now=c.now)
        return {"status": "collecting"}
    return {"status": lease["state"]}


def _fail(c: Collector, lease: dict[str, Any], code: str) -> dict[str, Any]:
    leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state="expired", now=c.now,
                      updates={"outcome": _outcome(code)})
    return {"status": "expired", "outcome": _outcome(code)}


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


# ---------------------------------------------------------------------------------------------- commit


def _record_failure(c: Collector, row: dict[str, Any], code: str) -> dict[str, Any]:
    """Count one promotion or landing failure; past the budget the attempt itself fails (plan 14 §9)."""

    first = row["first_failure_at_epoch"] if row["first_failure_at_epoch"] is not None else c.now
    row = _save_row(c, row, failures=row["failures"] + 1, first_failure_at_epoch=first, last_failure=_outcome(code))
    if row["failures"] >= COLLECTION_RETRIES or c.now - first >= COLLECTION_WINDOW_SECONDS:
        raise _AttemptFailed(_outcome(code))
    return row


def _range(c: Collector, uri: str, offset: int, length: int) -> Any:
    client, bucket, _ = c.runtime.object_store
    key = uri.removeprefix(f"s3://{bucket}/")
    return client.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset + length - 1}")["Body"]


def _read_cas(c: Collector, reference: Mapping[str, Any], maximum: int) -> bytes:
    import hashlib

    client, bucket, _ = c.runtime.object_store
    body = client.get_object(Bucket=bucket, Key=str(reference["uri"]).removeprefix(f"s3://{bucket}/"))["Body"]
    data = body.read(maximum + 1)
    if len(data) != reference["size_bytes"] or "sha256:" + hashlib.sha256(data).hexdigest() != reference["digest"]:
        raise CollectorError("remote_episode_compilation_readback_mismatch")
    return data


def _promote(c: Collector, descriptor: dict[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    """Server-side copy of ``blobs.tar`` and ``index.json`` into CAS, then one streaming readback (plan 14 §9).

    A promoted object whose bytes do not match its key is discarded, every version, so a retry from staging
    copies again rather than finding it."""

    client, bucket, _ = c.runtime.object_store
    staging = descriptor["outputs"]["staging_prefix"]
    promoted: dict[str, Any] = {}
    for name, label in (("index.json", "index"), ("blobs.tar", "archive")):
        key = (staging + name).removeprefix(f"s3://{bucket}/")
        etag = str(client.head_object(Bucket=bucket, Key=key).get("ETag") or "")
        copied = copy_remote_cpu_staging_to_cas(
            staging_uri=staging + name, digest=output[label]["digest"], size_bytes=output[label]["size_bytes"],
            etag=etag, artifact_kind="remote-cpu-output", filename=name, client=client, bucket=bucket)
        promoted[label] = {"uri": copied["uri"], "digest": copied["digest"], "size_bytes": copied["size_bytes"]}
    try:
        index = json.loads(_read_cas(c, promoted["index"], MAX_INDEX_BYTES))
        body = client.get_object(Bucket=bucket, Key=promoted["archive"]["uri"].removeprefix(f"s3://{bucket}/"))["Body"]
        verify_blobs_stream(body, index, expected_digest=promoted["archive"]["digest"])
    except (CollectorError, RemoteCpuArchiveError, ValueError) as exc:
        for label in ("archive", "index"):
            discard_remote_cpu_output_object(uri=promoted[label]["uri"], digest=promoted[label]["digest"],
                                             client=client, bucket=bucket)
        raise CollectorError(f"remote_cpu_promotion_readback_failed:{type(exc).__name__}") from None
    return promoted


def _promotion(c: Collector, row: dict[str, Any], descriptor: dict[str, Any],
               output: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Promote once per attempt; a failure retries from staging, never by rerunning the execution."""

    if row["promoted"] is not None:
        return row["promoted"], row
    try:
        promoted = _promote(c, descriptor, output)
    except (CollectorError, TaskEvaluationConfiguredSceneObjectStoreError, RemoteCpuArchiveError) as exc:
        return None, _record_failure(c, row, str(exc).split(" ")[0] or "remote_cpu_promotion_failed")
    return promoted, _save_row(c, row, promoted=promoted)


def _envelope(c: Collector, name: str) -> dict[str, Any]:
    for state in ("processing", "completed", "blocked"):
        path = c.queue_root / state / name
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("envelope_digest") == canonical_digest(value, digest_field="envelope_digest"):
                return value
    raise CollectorError("remote_episode_compilation_row_unlocated")


def _validate(c: Collector, envelope: Mapping[str, Any], descriptor: dict[str, Any], result: Mapping[str, Any],
              index: Mapping[str, Any], archive: Mapping[str, Any]) -> list[str]:
    """Plan 14 §9, mirroring ``_validated_compiler_output`` from the index and the JSON blobs by range."""

    root = os.path.realpath(c.host(descriptor["outputs"]["output_root"]))
    entries = {row["path"]: row for row in index["entries"]}
    offsets = {row["blob"]: row["offset"] for row in index["blobs"]}
    reasons: list[str] = []

    def relative(value: Any) -> str | None:
        text = os.path.realpath(str(value or ""))
        return text[len(root) + 1:] if text.startswith(root + os.sep) else None

    def document(path: str | None) -> dict[str, Any]:
        entry = entries.get(path or "")
        if entry is None or entry["origin"] != "archive" or entry["size_bytes"] > MAX_JSON_BLOB_BYTES:
            return {}
        try:
            with _range(c, archive["uri"], offsets[entry["blob"]], max(entry["size_bytes"], 1)) as stream:
                data = stream.read(entry["size_bytes"])
        except Exception as exc:  # noqa: BLE001 - a read that failed says nothing about the output
            raise _ReadFailed(f"remote_cpu_output_read_failed:{type(exc).__name__}") from None
        try:
            value = json.loads(data)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    expected = {"schema_version": RESULT_SCHEMA_VERSION, "status": "compiled_for_production_launch",
                "compilation_id": envelope["compilation_id"], "run_id": envelope["run_id"],
                "team_namespace": envelope["team_namespace"], "source_commit": c.source_commit,
                "configured_scene_revision_digest": envelope["configured_scene_revision_digest"],
                "customer_supplied_prebuilt_episode_packet": False, "compiled_by_production": True,
                "provider_mutation_performed": False, "paid_execution_requested": False,
                "automatic_progression_required": True, "blockers": []}
    reasons += [f"result_{name}" for name, value in expected.items() if result.get(name) != value]
    if result.get("result_digest") != canonical_digest(result, digest_field="result_digest"):
        reasons.append("result_seal")
    packet = entries.get(relative(result.get("compiled_episode_packet_path")) or "")
    if packet is None or (packet["blob"], packet["size_bytes"]) != (result.get("compiled_episode_packet_digest"),
                                                                   result.get("compiled_episode_packet_size_bytes")):
        reasons.append("packet")
    adapter_path = relative(result.get("adapter_result_path"))
    adapter = document(adapter_path)
    runtime = relative(adapter.get("runtime_source_receipt"))
    packet_root = relative(adapter.get("packet_root"))
    if (adapter_path != ADAPTER_RESULT or adapter.get("status") != "native_arena_adapter_materialized"
            or adapter.get("result_digest") != result.get("adapter_result_digest")
            or adapter.get("result_digest") != canonical_digest(adapter, digest_field="result_digest")
            or adapter.get("preparation_id") != envelope["preparation_id"]
            or adapter.get("source_commit") != c.source_commit or runtime not in entries
            or not (runtime or "").startswith("native-arena-adapter/")
            or not (packet_root or "").startswith("native-arena-adapter/")
            or not any(path.startswith(packet_root + "/") for path in entries)):
        reasons.append("adapter_result")
    probe_fields = ("destination_native_probe_request_path", "destination_native_probe_request_digest",
                    "destination_native_probe_request_document_digest")
    if envelope["request"].get("run_mode") == "destination_qualification":
        probe_path = relative(result.get(probe_fields[0]))
        probe = entries.get(probe_path or "")
        if (probe is None or probe["blob"] != result.get(probe_fields[1]) or probe_path not in {
                "rigid_destination_native_probe_request.v1.json"}
                or document(probe_path).get("request_digest") != result.get(probe_fields[2])):
            reasons.append("probe")
    elif any(name in result for name in probe_fields):
        reasons.append("probe")
    if not str(result.get("compiler_output_digest") or "").startswith("sha256:"):
        reasons.append("compiler_output_digest")
    return reasons


def _land(c: Collector, plan: remote.RemotePlan, descriptor: dict[str, Any], index: Mapping[str, Any],
          archive: Mapping[str, Any]) -> dict[str, Any]:
    """Land only the consumer subset, assembled aside and renamed into place (plan 14 §9)."""

    destination = c.host(descriptor["outputs"]["output_root"])
    sources = {row["digest"]: row["path"] for row in remote.input_sources(plan, c.queue_root)}
    members = c.outputs_root / "content-addressed" / "adapter-members" / "sha256"
    selected = [row for row in index["entries"] if any(
        row["path"] == pattern or (pattern.endswith("/**") and row["path"].startswith(pattern[:-2]))
        for pattern in remote.EPISODE_COMPILATION_CONSUMER_SUBSET)]
    missing = sum({(row["blob"], row["mode"]): row["size_bytes"] for row in selected
                   if row["origin"] == "archive" or "input_member" in row["origin"]}.values())
    reservation = None
    if c.disk_reservation_root is not None:
        reservation = reserve_control_plane_disk(
            "episode_compilation", target_root=c.outputs_root, expected_bytes=missing + LANDING_MARGIN_BYTES,
            reservation_root=c.disk_reservation_root, workspace=destination, workload="compiled_episode_landing")
    landed = None
    try:
        landed = land_subset(index=index, reader=lambda offset, length: _range(c, archive["uri"], offset, length),
                             host_sources=sources, destination_root=destination,
                             selectors=list(remote.EPISODE_COMPILATION_CONSUMER_SUBSET), member_store=members)
    finally:
        if reservation is not None:
            reservation.release(outcome="completed" if landed is not None else "failed")
    return landed


def _consumption(attempt_id: str) -> dict[str, Any]:
    return allocator._sealed(allocator._consumption_path(attempt_id), allocator.CONSUMPTION_SCHEMA_VERSION,
                             "consumption_digest")


def _pointer_path(c: Collector, compilation_id: str) -> Path:
    return c.outputs_root / f"{compilation_id}{POINTER_SUFFIX}"


def _write_pointer(c: Collector, plan: remote.RemotePlan, lease: Mapping[str, Any], descriptor: dict[str, Any],
                   receipt: Mapping[str, Any], promoted: Mapping[str, Any], index: Mapping[str, Any],
                   landed: Mapping[str, Any]) -> dict[str, Any]:
    consumption = _consumption(lease["attempt_id"])
    # The packet the result names, which did not land, as a host path under the output root: validation has
    # placed it there, and the worker compiled at the host's own paths, so this is the result's path.
    output_root = descriptor["outputs"]["output_root"]
    packet = os.path.realpath(str(receipt["result"]["compiled_episode_packet_path"]))
    packet = output_root + packet[len(os.path.realpath(c.host(output_root))):]
    fields = {
        "stage": STAGE, "compilation_id": plan.compilation_id, "queue_row": dict(plan.queue_row),
        "attempt_id": lease["attempt_id"], "descriptor_digest": descriptor["descriptor_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "execution": {"provider": EXECUTION_PROVIDER, "job": descriptor["execution"]["job"],
                      "worker_identity": lease["worker_identity"],
                      "allocation_binding_digest": consumption["allocation_binding_digest"],
                      "spend_consumption": consumption["consumption_digest"]},
        "code": {"source_commit": descriptor["code"]["source_commit"],
                 "source_archive_digest": descriptor["code"]["source_archive"]["digest"],
                 "image": descriptor["code"]["image"], "environment_digest": descriptor["code"]["environment_digest"]},
        "archive": dict(promoted["archive"]), "index": dict(promoted["index"]),
        "output_root": output_root, "paths_total": index["paths_total"],
        "bytes_total": index["bytes_total"], "host_known": dict(index["host_known"]),
        "landed": {"subset": CONSUMER_SUBSET, "paths": landed["paths"], "bytes": landed["bytes"]},
        # What the result names that did not land: the packet, validated against the index above (§16).
        "raw_references": [{"path": packet, "digest": receipt["result"]["compiled_episode_packet_digest"],
                            "size_bytes": receipt["result"]["compiled_episode_packet_size_bytes"]}],
        "state": "landed",
    }
    pointer = pointer_record(fields)
    path = _pointer_path(c, plan.compilation_id)
    existing = _read_json(path)
    if existing is not None and existing.get("provider_zero_proven"):
        return existing  # already resealed with its teardown: this step is behind us
    replace_remote_cpu_record(path, pointer, previous_digest=None, digest_field="pointer_digest")
    return pointer


def _pin(c: Collector, plan: remote.RemotePlan, envelope: Mapping[str, Any]) -> None:
    if c.storage_pins_root is None:
        return
    try:
        write_storage_pin(pins_root=c.storage_pins_root, kind="compilation", owner_id=plan.compilation_id,
                          paths=[c.outputs_root / plan.compilation_id, _pointer_path(c, plan.compilation_id)],
                          depends_on=[{"kind": "preparation", "owner_id": str(envelope["preparation_id"])}])
    except (ControlPlaneStoragePinError, OSError, KeyError):
        pass  # as the host compile: a missing pin never blocks the result


def _write_result(c: Collector, name: str, result: Mapping[str, Any]) -> None:
    path = c.queue_root / "results" / name
    try:
        write_launch_preparation_record_exclusive(path, result)
    except FileExistsError:
        if path.read_bytes() != _canonical_bytes(result):
            raise CollectorError("remote_cpu_result_conflict") from None


def _move_row(c: Collector, name: str, state: str) -> None:
    source, target = c.queue_root / "processing" / name, c.queue_root / state / name
    if source.exists():
        os.replace(source, target)
    elif not target.is_file():
        raise CollectorError("remote_episode_compilation_row_unlocated")


def _commit(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan, lease: dict[str, Any],
            descriptor: dict[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Plan 14 §9's commit order; each step is idempotent, so a resume starts at the first one missing."""

    name, attempt_id, result = plan.queue_row["name"], lease["attempt_id"], receipt["result"]
    blocked = receipt["status"] == "blocked"
    terminal = "blocked" if blocked else "completed"
    if (c.queue_root / terminal / name).is_file() and not (c.queue_root / "processing" / name).exists():
        return _close(c, path, marker, lease, terminal=terminal, outcome=result["status"])
    if not _compute_zero(c, lease, descriptor):
        return {"status": "collecting", "blocker": "remote_cpu_compute_zero_unproven"}
    _after_step("compute_zero", attempt_id)
    envelope = _envelope(c, name)
    if not blocked:
        row = _for_attempt(c, _row(c, name, plan.queue_row), attempt_id)
        promoted, row = _promotion(c, row, descriptor, receipt["output"])
        if promoted is None:
            return {"status": "collecting", "blocker": row["last_failure"]}
        _after_step("promotion", attempt_id)
        try:
            index = json.loads(_read_cas(c, promoted["index"], MAX_INDEX_BYTES))
            reasons = _validate(c, envelope, descriptor, result, index, promoted["archive"])
        except (_ReadFailed, CollectorError, TaskEvaluationConfiguredSceneObjectStoreError, OSError) as exc:
            row = _record_failure(c, row, str(exc).split(" ")[0] or "remote_cpu_output_read_failed")
            return {"status": "collecting", "blocker": row["last_failure"]}
        if reasons:
            raise _AttemptFailed(f"remote_cpu_output_invalid:{reasons[0]}")
        _after_step("validation", attempt_id)
        try:
            landed = _land(c, plan, descriptor, index, promoted["archive"])
        except (RemoteCpuArchiveError, TaskEvaluationConfiguredSceneObjectStoreError, OSError) as exc:
            row = _record_failure(c, row, str(exc).split(" ")[0] or "remote_cpu_landing_failed")
            return {"status": "collecting", "blocker": row["last_failure"]}
        _after_step("landing", attempt_id)
        _write_pointer(c, plan, lease, descriptor, receipt, promoted, index, landed)
        _after_step("pointer", attempt_id)
        _pin(c, plan, envelope)
        _after_step("pin", attempt_id)
    _write_result(c, name, result)
    _after_step("result", attempt_id)
    _move_row(c, name, terminal)
    _after_step("move", attempt_id)
    return _close(c, path, marker, lease, terminal=terminal, outcome=result["status"])


def _tree_mismatches(root: Path, index: Mapping[str, Any]) -> list[str]:
    """Where the host's own output tree differs from the worker's index: paths, bytes, modes."""

    import hashlib

    if not root.is_dir():
        return ["$root"]
    found: dict[str, Any] = {}
    for current in sorted(root.rglob("*")):
        info, relative = current.lstat(), current.relative_to(root).as_posix()
        mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if stat.S_ISDIR(info.st_mode):
            found[relative] = ("dir", mode)
        else:
            found[relative] = ("sha256:" + hashlib.sha256(current.read_bytes()).hexdigest(), info.st_size, mode)
    expected = {row["path"]: ("dir", row["mode"]) for row in index["directories"]}
    expected.update({row["path"]: (row["blob"], row["size_bytes"], row["mode"]) for row in index["entries"]})
    mismatches = sorted(path for path in set(found) | set(expected) if found.get(path) != expected.get(path))
    if f"{stat.S_IMODE(root.lstat().st_mode):04o}" != index["root_mode"]:
        mismatches.insert(0, "$root_mode")
    return mismatches


def _compare_shadow(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan,
                    lease: dict[str, Any], descriptor: dict[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """The host stayed authoritative: compare, record parity for the closure class, land nothing (§13)."""

    if not _compute_zero(c, lease, descriptor):
        return {"status": "collecting", "blocker": "remote_cpu_compute_zero_unproven"}
    recorded = _read_json(c.jobs_root / "parity" / STAGE / f"{lease['attempt_id']}.json")
    if recorded is not None:  # compared already: only the teardown is left
        return _close(c, path, marker, lease, terminal="shadow_compared", outcome=f"shadow_parity_{recorded['parity']}")
    name = plan.queue_row["name"]
    host, worker = _read_json(c.queue_root / "results" / name), receipt["result"]
    compiled = "compiled_for_production_launch"
    mismatches: list[str] = []
    if host is not None and receipt["status"] == "succeeded" and host.get("status") == compiled == worker.get("status"):
        # Both sides compiled: results and output trees are compared byte for byte, the only way to a pass.
        mismatches = [] if host == worker else ["result"]
        client, bucket, _ = c.runtime.object_store
        raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "index.json",
                                             maximum_size_bytes=MAX_INDEX_BYTES, client=client, bucket=bucket)
        output = receipt["output"]["index"]
        import hashlib

        if raw is None or ("sha256:" + hashlib.sha256(raw).hexdigest(), len(raw)) != (output["digest"],
                                                                                    output["size_bytes"]):
            mismatches.append("index")
        else:
            mismatches += _tree_mismatches(c.host(descriptor["outputs"]["output_root"]), json.loads(raw))
        parity = "passed" if not mismatches else "failed"
    elif host is not None and host.get("status") != worker.get("status"):
        parity, mismatches = "failed", ["status"]  # one side compiled and the other did not
    else:
        parity = "inconclusive"  # nothing compiled on both sides to compare (review I2): no pass, no reset
    remote.record_shadow_parity(c.jobs_root, {
        "closure_class": plan.closure["class"], "attempt_id": lease["attempt_id"], "queue_row": dict(plan.queue_row),
        "image": plan.image, "host_environment_digest": plan.host_environment_digest,
        "worker_environment_digest": receipt["environment"]["environment_digest"],
        "cpu_class": receipt["environment"]["cpu_class"], "parity": parity, "mismatches": mismatches[:16],
        "compared_at_epoch": c.now})
    return _close(c, path, marker, lease, terminal="shadow_compared", outcome=f"shadow_parity_{parity}")


def _collect(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan, lease: dict[str, Any],
             descriptor: dict[str, Any]) -> dict[str, Any]:
    # The receipt is kept on the host once fenced: provider-zero deletes staging before the write URLs expire,
    # so a resumed commit must never depend on staging.
    recorded = c.jobs_root / "receipts" / f"{lease['attempt_id']}.json"
    raw: Any = _read_json(recorded)
    if raw is None:
        client, bucket, _ = c.runtime.object_store
        try:
            raw = read_remote_cpu_staging_object(staging_uri=descriptor["outputs"]["staging_prefix"] + "receipt.json",
                                                 maximum_size_bytes=MAX_RECEIPT_BYTES, client=client, bucket=bucket)
        except TaskEvaluationConfiguredSceneObjectStoreError:
            # Its state is unknown, which is never success: read again later, then count it as missing.
            if c.now < lease["deadlines"]["hard_deadline_epoch"] + RECEIPT_UNKNOWN_GRACE_SECONDS:
                return {"status": "collecting", "blocker": "remote_cpu_receipt_state_unknown"}
            raw = None
        if raw is None:
            return _fail(c, lease, "remote_cpu_receipt_missing")
        try:
            raw = json.loads(raw)
        except ValueError:
            return _fail(c, lease, "remote_cpu_receipt_invalid")
    try:
        verdict = validate_receipt(raw, descriptor=descriptor, execution_name=execution_name_of(lease["worker_identity"]))
    except (ValueError, TypeError):  # RemoteCpuContractError is a ValueError
        return _fail(c, lease, "remote_cpu_receipt_invalid")
    write_remote_cpu_record(recorded, verdict["receipt"])
    if verdict["outcome"] == "infrastructure_failed":
        return _fail(c, lease, verdict["infrastructure_failures"][0])
    try:
        if descriptor["mode"] == "shadow":
            return _compare_shadow(c, path, marker, plan, lease, descriptor, verdict["receipt"])
        return _commit(c, path, marker, plan, lease, descriptor, verdict["receipt"])
    except _AttemptFailed as exc:
        return _fail(c, _lease(c, lease["job_id"]), str(exc))


# ---------------------------------------------------------------------------------------------- teardown


def _seal_teardown(c: Collector, attempt: Mapping[str, Any], descriptor: dict[str, Any]) -> dict[str, Any] | None:
    """One started attempt's provider-zero teardown, sealed once (plan 14 §11); ``None`` while unprovable."""

    path = c.jobs_root / "teardowns" / f"{attempt['attempt_id']}.json"
    sealed = _read_json(path)
    if sealed is not None:
        return validate_teardown(sealed)
    compute = allocator.prove_compute_zero(c.action(), attempt, descriptor)
    if not compute_zero_proven(compute, worker_identity=attempt["worker_identity"]):
        return None
    record = teardown_record(descriptor=descriptor, worker_identity=attempt["worker_identity"],
                             outcome=attempt["outcome"] or "remote_cpu_collected", compute=compute,
                             provider=allocator.prove_provider_zero(c.action(), attempt, descriptor),
                             observed_at_epoch=c.now)
    if not record["provider_zero_proven"]:
        return None
    write_remote_cpu_record(path, record)
    return record


def _settle(c: Collector, attempt: Mapping[str, Any], descriptor: dict[str, Any], record: Mapping[str, Any]) -> None:
    limits, seconds = descriptor["limits"], None
    if attempt["worker_identity"]:
        seconds = execution_seconds(c.runtime.cloud_run.get_execution(execution_resource_name(
            run_target(descriptor)["job"], execution_name_of(attempt["worker_identity"]))))
    if attempt["worker_identity"] is None and record["compute_zero"]["executions_for_attempt"] == 0:
        settled, basis = 0.0, "no_execution"
    else:
        usage = {**limits, "task_timeout_seconds": min(limits["task_timeout_seconds"], seconds or math.inf)}
        settled = allocator.worst_case_usd(limits=usage, rate_table=c.config["rate_table"])
        basis = "worst_case" if seconds is None else "execution_runtime"
    allocator.settle_remote_cpu_attempt(descriptor=descriptor, teardown_digest=record["teardown_digest"],
                                        settled_usd=settled, basis=basis, now=c.now)


def _close(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any], *, terminal: str,
           outcome: str | None = None) -> dict[str, Any]:
    """Tear every started attempt down to provider-zero, reseal the pointer, then turn the lease terminal."""

    if outcome is not None and lease["outcome"] != _outcome(outcome):
        lease = leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state=None,
                                  now=c.now, updates={"outcome": _outcome(outcome)})
    pending = 0
    for attempt in [*lease["prior_attempts"], lease]:
        if not attempt["dispatch_started"] or attempt["provider_zero_proven"]:
            continue
        descriptor, _ = _descriptor(c, attempt["attempt_id"])
        record = _seal_teardown(c, attempt, descriptor)
        if record is None:
            pending += 1
            continue
        lease = leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state=None,
                                  now=c.now, updates={"teardown": record})
        _settle(c, attempt, descriptor, record)
        if terminal == "completed" and attempt["attempt_id"] == lease["attempt_id"]:
            _reseal_pointer(c, marker["plan"]["compilation_id"], record)
    if pending:
        c.summary["teardown_unproven"] = c.summary.get("teardown_unproven", 0) + pending
        return {"status": lease["state"], "blocker": "remote_cpu_provider_zero_unproven"}
    _after_step("provider_zero", lease["attempt_id"])
    if lease["state"] not in leases.TERMINAL_STATES:
        lease = leases.transition(c.jobs_root, lease["job_id"], attempt_id=lease["attempt_id"], to_state=terminal,
                                  now=c.now)
    return _finish(c, path, marker, lease)


def _reseal_pointer(c: Collector, compilation_id: str, record: Mapping[str, Any]) -> None:
    path = _pointer_path(c, compilation_id)
    pointer = _read_json(path)
    if pointer is None or pointer.get("teardown_receipt_digest") == record["teardown_digest"]:
        return
    resealed = pointer_record({}, previous=pointer, teardown=record)
    replace_remote_cpu_record(path, resealed, previous_digest=pointer["pointer_digest"], digest_field="pointer_digest")


def _finish(c: Collector, path: Path, marker: Mapping[str, Any], lease: dict[str, Any]) -> dict[str, Any]:
    """A terminal lease: a fallback or abandoned authoritative row goes back to the host; the marker goes."""

    if lease["state"] in {"fallback_host", "abandoned_dispatch"}:
        _give_up(c, marker, reason=lease["outcome"] or lease["state"], attempts=lease["attempt"])
    path.unlink(missing_ok=True)
    return {"status": lease["state"], "attempt_id": lease["attempt_id"]}


def _after_expiry(c: Collector, path: Path, marker: Mapping[str, Any], plan: remote.RemotePlan,
                  lease: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any]:
    """Compute-zero first; then one fresh attempt, or the host (plan 14 §10)."""

    if not _compute_zero(c, lease, descriptor):
        return {"status": "expired", "blocker": "remote_cpu_compute_zero_unproven"}
    lease = _lease(c, lease["job_id"])
    if (_handed_back(c, marker) is None and _dispatchable(c, marker) and lease["attempt"] < c.config["max_attempts"]
            and plan.source_commit == c.source_commit):
        return _dispatch(c, path, marker, plan, lease["attempt"] + 1)
    _give_up(c, marker, reason=lease["outcome"] or "remote_cpu_attempts_exhausted", attempts=lease["attempt"])
    return _close(c, path, marker, lease, terminal="fallback_host")


def _advance(c: Collector, path: Path, marker: Mapping[str, Any]) -> dict[str, Any]:
    plan = remote.RemotePlan.from_record(marker["plan"])
    lease = _lease(c, job_id_for(STAGE, marker["queue_row"]["name"]))
    state = None if lease is None else lease["state"]
    if state in {None, "claimed", "awaiting_capacity"}:
        retired = _handed_back(c, marker)
        if retired is None and not _dispatchable(c, marker):
            retired = "remote_cpu_mode_rolled_back"
        if retired is None and plan.source_commit != c.source_commit:
            retired = "remote_cpu_release_changed"
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
        return _monitor(c, lease, descriptor)
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
    if isinstance(exc, (CollectorError, TaskEvaluationConfiguredSceneObjectStoreError)):
        code = str(exc).split(" ")[0]
    return _outcome(str(code)) if code else f"remote_episode_compilation_failed:{type(exc).__name__}"


def _parity_counts(c: Collector) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for record_path in sorted((c.jobs_root / "parity" / STAGE).glob("*.json")):
        record = _read_json(record_path) or {}
        klass = counts.setdefault(str(record.get("closure_class")), {"passed": 0, "failed": 0, "inconclusive": 0})
        klass[record["parity"] if record.get("parity") in {"passed", "inconclusive"} else "failed"] += 1
    return counts


def run_collector(c: Collector) -> dict[str, Any]:
    """One paid-unit run: every hand-off and shadow marker advanced one step, then a sweep and the summary."""

    c.summary = {"schema_version": SUMMARY_SCHEMA_VERSION, "stage": STAGE, "mode": c.mode, "observed_at_epoch": c.now,
                 "rows": {}, "drift": 0, "teardown_unproven": 0, "orphans_cancelled": 0, "blockers": []}
    _observe_job_image(c)
    for kind in ("authoritative", "shadow"):
        for path, marker in remote.markers(c.jobs_root, kind):
            if marker is None or marker.get("schema_version") != remote.HANDOFF_SCHEMA_VERSION:
                c.summary["blockers"].append(f"remote_episode_compilation_marker_unreadable:{path.name}")
                continue
            try:
                c.summary["rows"][path.name] = _advance(c, path, marker)
            except Exception as exc:  # noqa: BLE001 - one row's failure never stops the others; its cause stays typed
                c.summary["rows"][path.name] = {"status": "blocked", "blocker": _typed(exc)}
    # After every row was followed, so a finished execution is collected rather than expired.
    c.summary["expired"] = leases.expire_stale(c.jobs_root, now=c.now)
    swept = _allocate(c, "sweep", None, "stage")
    c.summary["orphans_cancelled"] = len(swept.get("cancelled") or [])
    census = leases.slot_census(c.jobs_root)
    c.summary.update(slots_in_use=census["slots_in_use"], unreadable_leases=census["unreadable"],
                     parity=_parity_counts(c))
    summary_path = c.jobs_root / "summary.json"
    temporary = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(c.summary, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o644)
    os.replace(temporary, summary_path)
    return c.summary


def should_run(jobs_root: str | Path, environ: Mapping[str, str] | None = None) -> bool:
    """The paid unit's ExecCondition: a remote mode, a live lease, or a marker still to finish (plan 14 §1)."""

    mode, _ = remote.execution_mode(environ)
    root = Path(jobs_root)
    return (mode != "host" or any((root / "live").glob("*"))
            or any(remote.markers(root, kind) for kind in ("authoritative", "shadow")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch and collect remote episode compilations")
    parser.add_argument("command", choices=("run", "should-run"))
    parser.add_argument("--source-commit")
    parser.add_argument("--jobs-root", default=os.getenv(remote.JOBS_ROOT_ENV, remote.DEFAULT_JOBS_ROOT))
    args = parser.parse_args(argv)
    if args.command == "should-run":
        return 0 if should_run(args.jobs_root) else 1
    mode, findings = remote.execution_mode()
    config, blockers = allocator.load_remote_cpu_config()
    runtime = allocator.RemoteCpuRuntime()
    blockers = blockers or allocator._connect(runtime, config)
    if blockers or not args.source_commit:
        print(json.dumps({"status": "blocked", "blockers": blockers or ["remote_episode_compilation_commit_missing"],
                          "findings": findings}, sort_keys=True))
        return 0
    repository = Path(__file__).resolve().parents[2]
    collector = Collector(
        runtime=runtime, config=config, jobs_root=Path(args.jobs_root),
        queue_root=Path(os.environ[QUEUE_ROOT_ENV]), outputs_root=Path(os.environ[OUTPUT_ROOT_ENV]),
        source_commit=args.source_commit, mode=mode, allocate=subprocess_allocate,
        stage_release=lambda store, commit: publish_release_source(repository=repository, source_commit=commit,
                                                                   client=store[0], bucket=store[1]),
        disk_reservation_root=Path(os.environ["BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"])
        if os.getenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT") else None,
        storage_pins_root=Path(os.environ["BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT"])
        if os.getenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT") else None)
    summary = run_collector(collector)
    print(json.dumps({"status": "collected", "rows": len(summary["rows"]), "findings": findings}, sort_keys=True))
    return 0


__all__ = [
    "Collector",
    "CollectorError",
    "INPUT_ROOT_ENV",
    "RemoteCpuContractError",
    "main",
    "plan_descriptor",
    "run_collector",
    "should_run",
    "subprocess_allocate",
]


if __name__ == "__main__":
    raise SystemExit(main())
