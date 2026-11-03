"""A website scene's robot-team half compiled on a remote CPU worker: plan 14's lane, second stage.

Why: the control-plane host has 8.3 GB of RAM and no swap. capture-447ae4f4's compile reached 6.3-6.7 GB
of anonymous memory on a 292 MB USDA room collider and a 2 M-splat, 130 MB appearance and was killed by
the global OOM killer five times (2026-11-03). The owner kept the input fidelity and chose machines rented
per job (2026-11-03), so ``team_scene_packet.materialize_team_scene`` runs here on a Cloud Run job instead.

``BLUEPRINT_TEAM_SCENE_PACKET_EXECUTION`` is ``host`` (today's path), ``cloud_run_shadow`` or ``cloud_run``.
Unset or empty is auto: ``cloud_run`` once the remote-CPU config is there and usable for this stage, exactly
as the paid unit loads it, and ``host`` otherwise, which is today's path byte for byte. An invalid value runs
as ``host``.

The no-spend team worker (``team_scene_packet_worker``) owns the request queue in every mode and holds no
credential. In a remote mode it fetches the revision and its documents into the attempt directory as the
host compile would, then asks ``plan_remote_team_scene``, with no credential, whether the compile can run
remotely: every file the compile reads ships content-addressed at its host path (the configured bundle,
the documents and tray assets, the DROID profile, the runtime-source receipt, each registered body's model,
the owner's sealed success confirmations and, for a cached particlefield, the cache entry), the appearance
closure is one the worker reproduces, the worker's in-memory disk fits, and the probe-recorded worker
environment equals the host's. An ineligible request compiles on the host, as before.

In ``cloud_run`` an eligible request gets a hand-off and stays in ``processing/``; the paid unit
(``team_scene_packet_collector``) dispatches it through the paid-resource allocator, lands the whole
``<attempt>/scene`` tree where the host compile would have written it, and leaves a ``collected`` marker.
The no-spend worker then records the derived row and republishes the offer exactly as after a host compile.
A request the paid unit gives up on comes back as a ``fallback`` marker and compiles on the host, from a
fresh attempt directory; the host never compiles a request while a remote attempt of it may still land.
``cloud_run`` is authoritative without a shadow-pass gate, unlike episode compilation: the scenes this lane
exists for cannot compile on the host at all, so shadows could never earn passes for them. In
``cloud_run_shadow`` the host compiles every request and a completed, eligible one also gets a shadow
marker: the paid unit compiles it remotely at the same paths, records byte parity and lands nothing.

Inside the worker, ``run_team_scene_packet_in_worker`` calls the host's own ``materialize_team_scene`` with
the same arguments (the request's ``owner_success_confirmation_scope`` included) at the host's paths. This
module imports no allocation or staging authority: the worker's stage child imports it from the release.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_environment import DIGESTED_FIELDS, environment_digest, environment_record
from .remote_cpu_job_contract import (
    CACHE_ROOT_VARIABLE,
    PERMITTED_PATH_ROOTS,
    STAGE_LIMITS,
    STAGES,
    config_blockers,
)
from .remote_cpu_job_records import write_remote_cpu_record

EXECUTION_ENV = "BLUEPRINT_TEAM_SCENE_PACKET_EXECUTION"
MODES = ("host", "cloud_run_shadow", "cloud_run")
REMOTE_MODES = ("cloud_run_shadow", "cloud_run")
STAGE = "team_scene_packet"
QUEUE = STAGES[STAGE]["queue"]
COMPLETED = STAGES[STAGE]["success_status"]
JOBS_ROOT_ENV = "BLUEPRINT_REMOTE_CPU_JOBS_ROOT"
DEFAULT_JOBS_ROOT = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
# The allocator reads the same file; this no-spend copy of its loader may not import the allocator.
CONFIG_ENV = "BLUEPRINT_REMOTE_CPU_WORKERS_CONFIG"
DEFAULT_CONFIG_PATH = "/etc/blueprint/remote-cpu-workers.json"
CONFIG_MAX_BYTES = 256 * 1024
DEFAULT_CACHE_ROOT = "/var/lib/blueprint/task-evaluation-inputs/particlefield-runtime-assets"
ENVELOPE_SCHEMA_VERSION = "team_scene_packet_remote_envelope.v1"
RESULT_SCHEMA_VERSION = "team_scene_packet_remote_result.v1"
HANDOFF_SCHEMA_VERSION = "team_scene_packet_handoff.v1"
FALLBACK_SCHEMA_VERSION = "team_scene_packet_fallback.v1"
COLLECTED_SCHEMA_VERSION = "team_scene_packet_collected.v1"
PARITY_SCHEMA_VERSION = "team_scene_packet_shadow_parity.v1"
WORKER_ENVIRONMENT_SCHEMA_VERSION = "remote_cpu_worker_environment.v1"
JOB_IMAGE_SCHEMA_VERSION = "remote_cpu_job_image_observation.v1"
# Under the jobs root: a hand-off (``cloud_run``) or shadow marker names a plan for the paid unit; a fallback
# returns a request to the no-spend unit; a collected marker says its tree landed. Envelopes are the inputs.
MARKERS = {"authoritative": "handoffs", "shadow": "shadow", "fallback": "fallback", "collected": "collected"}
ENVELOPES = "envelopes"
# Memory-backed /var/lib/blueprint holds the inputs, the extracted assets and the compile's copies of them.
ASSET_COPIES = 3
_MAX_RECORD_BYTES = 4 * 1024 * 1024
_ATTEMPT = re.compile(r"[0-9a-f]{16}-[0-9a-f]{8}")
_CHUNK = 1024 * 1024
PATH_ARGUMENTS = ("configured_scene_bundle_path", "droid_profile_path", "runtime_source_receipt_path", "output_root")


class TeamScenePacketRemoteError(RuntimeError):
    """A typed refusal on the remote team-scene path; the message is the blocker."""


# ---------------------------------------------------------------------------------------------- mode


def _read_record(path: Path, *, forbidden_mode: int = 0o022, maximum: int = _MAX_RECORD_BYTES) -> dict[str, Any] | None:
    """A regular, non-symlinked JSON object no other account may write; ``None`` when absent or unusable."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode) or status.st_mode & forbidden_mode:
        os.close(descriptor)
        return None
    with os.fdopen(descriptor, "rb") as stream:
        payload = stream.read(maximum + 1)
    try:
        value = json.loads(payload) if len(payload) <= maximum else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def load_config(environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The host's ``remote_cpu_workers_config.v1`` (0640 at most), or ``None`` when absent or unusable: what the
    allocator's ``load_remote_cpu_config`` loads without blockers."""

    values = os.environ if environ is None else environ
    try:
        config = _read_record(Path(values.get(CONFIG_ENV) or DEFAULT_CONFIG_PATH), forbidden_mode=0o027,
                              maximum=CONFIG_MAX_BYTES)
        return config if config is not None and not config_blockers(config) else None
    except Exception:  # noqa: BLE001 - deep nesting, a lone surrogate, an I/O error: no config, host
        return None


def remote_configured(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the remote-CPU config is there and usable, exactly as the paid unit loads it, and names this stage."""

    config = load_config(environ)
    return config is not None and STAGE in config["stages"]


def resolve_execution_mode(environ: Mapping[str, str] | None = None, *,
                           configured: bool | None = None) -> dict[str, Any]:
    """The mode a run asks for, the mode it runs in, and why; ``requested`` is door-safe (a mode, ``invalid``
    or ``None`` when unset), never the raw value."""

    values = os.environ if environ is None else environ
    raw = str(values.get(EXECUTION_ENV) or "").strip()
    if raw:
        selected, reason = raw, "explicit"
    elif remote_configured(values) if configured is None else configured:
        selected, reason = "cloud_run", "auto_with_config"
    else:
        selected, reason = "host", "auto_without_config"
    findings = [] if selected in MODES else ["team_scene_packet_execution_mode_invalid"]
    return {"requested": (raw if raw in MODES else "invalid") if raw else None,
            "effective": "host" if findings else selected, "reason": reason, "findings": findings}


@dataclass
class Dispatch:
    """What the no-spend worker needs to route a request: the mode and, for a remote one, the plan's inputs."""

    execution: dict[str, Any]
    jobs_root: Path
    config: dict[str, Any] | None = None
    source_commit: str | None = None
    filesystem_root: Path = Path("/")
    cache_root: Path | None = None
    host_environment: Mapping[str, Any] | None = None
    clock: Callable[[], float] = time.time

    @property
    def mode(self) -> str:
        return str(self.execution["effective"])


def dispatch_context(environ: Mapping[str, str] | None = None, *, source_commit: str | None = None) -> Dispatch:
    """The worker's routing from its environment; in ``host`` mode nothing beyond the config's path is read."""

    values = os.environ if environ is None else environ
    execution = resolve_execution_mode(values)
    config = load_config(values) if execution["effective"] != "host" else None
    return Dispatch(execution=execution, jobs_root=Path(values.get(JOBS_ROOT_ENV) or DEFAULT_JOBS_ROOT),
                    config=config, source_commit=source_commit,
                    cache_root=Path(values.get(CACHE_ROOT_VARIABLE) or DEFAULT_CACHE_ROOT))


# ---------------------------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class HostDecision:
    """The request compiles on the host, for this reason."""

    reason: str


@dataclass(frozen=True)
class RemotePlan:
    """Everything a descriptor needs but staged URIs and the attempt. ``output_root`` and each input's
    ``materialize_at`` are the worker's (``/``-relative) paths; each input's ``host_path`` is the file its bytes
    are staged from. ``attempt_root`` and ``capture_root`` are host paths."""

    queue_row: dict[str, Any]
    request_id: str
    scene_revision_digest: str
    capture_root: str
    attempt_root: str
    output_root: str
    source_commit: str
    image: str
    environment_digest: str
    host_environment_digest: str
    closure: dict[str, Any]
    environment: dict[str, str]
    inputs: tuple[dict[str, Any], ...]
    declared_scratch: tuple[str, ...] = field(default=())
    ephemeral_bytes_required: int = 0

    def record(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> RemotePlan:
        fields = {name: value[name] for name in cls.__dataclass_fields__}
        for name in ("inputs", "declared_scratch"):
            fields[name] = tuple(fields[name])
        return cls(**fields)


def worker_path(path: str | Path, filesystem_root: str | Path) -> str | None:
    """A canonical host path as the worker names it: ``/``-relative and under the permitted roots.

    A path with a symbolic link anywhere on it is not canonical: the worker materializes inputs at their
    declared path, so the host's resolved path is the only one both sides read alike."""

    candidate, root = Path(path), Path(filesystem_root)
    try:
        if not candidate.is_absolute() or candidate.resolve() != candidate:
            return None
        relative = candidate.relative_to(root.resolve())
    except (OSError, ValueError, RuntimeError):
        return None
    text = "/" + relative.as_posix()
    if any(part in {"", ".", ".."} for part in text.split("/")[1:]):
        return None
    return text if any(text.startswith(prefix) for prefix in PERMITTED_PATH_ROOTS) else None


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _input(role: str, contract_path: str, path: Path, filesystem_root: Path) -> dict[str, Any] | None:
    """One input row: a canonical, regular, non-empty host file under the permitted roots (staging refuses an
    empty object), with its digest, size and mode."""

    target = worker_path(path, filesystem_root)
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if target is None or not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        return None
    digest, size = _sha256_and_size(path)
    return {"role": role, "contract_path": contract_path, "digest": digest, "size_bytes": size,
            "mode": f"{stat.S_IMODE(info.st_mode):04o}", "materialize_at": target, "host_path": str(path)}


def _closure(bundle: Path, cache_root: Path) -> tuple[dict[str, Any], list[Path], int] | HostDecision:
    """The appearance closure class, the cache files it ships, and the configured-scene asset bytes, classified
    exactly as remote episode compilation classifies them (the compile converts the same appearance)."""

    from . import task_evaluation_episode_compilation_remote as episode

    parsed = episode._bundle_appearance(bundle)
    if parsed is None:
        return HostDecision("remote_ineligible:configured_scene_bundle_unreadable")
    appearance, rows = parsed
    assets = sum(int(row["size_bytes"]) for row in rows)
    digest = str(appearance["digest"])
    entry = episode._cache_entry(cache_root, digest)
    if entry == "invalid":
        return HostDecision("remote_ineligible:particlefield_cache_invalid")
    if entry:
        return {"class": "shipped", "source_appearance_digest": digest}, list(entry), assets
    nurec = episode._appearance_is_nurec(bundle, str(appearance.get("relative_path") or ""))
    if nurec is None:
        return HostDecision("remote_ineligible:configured_appearance_unreadable")
    if not nurec:
        return {"class": "not_applicable", "source_appearance_digest": None}, [], assets
    if int(appearance["size_bytes"]) <= episode.MAX_INLINE_NUREC_BYTES:
        # Its inline conversion writes a random temporary path into the converted layer (review I4).
        return HostDecision("remote_ineligible:inline_nurec_conversion_nondeterministic")
    return HostDecision("remote_ineligible:particlefield_transcode_required")


def _confirmation_files(scope: Mapping[str, Any] | None) -> list[tuple[str, Path]]:
    """Every file ``confirmed_owner_authority`` may read under the request's intent directory."""

    if scope is None:
        return []
    from .task_evaluation_owner_success_confirmation import DIRECTORY

    directory = Path(scope["directory"])
    files = [("owner_success_confirmation.revoked", directory / "revoked.json")] if (
        os.path.lexists(directory / "revoked.json")) else []
    store = directory / DIRECTORY
    if store.is_dir() and not store.is_symlink():
        files += [(f"owner_success_confirmation.{path.name}", path) for path in sorted(store.iterdir())
                  if path.suffix == ".json"]
    return files


def _plan_files(arguments: Mapping[str, Any], attempt: Path) -> list[tuple[str, str, Path]] | HostDecision:
    """``(role, contract_path, host path)`` of every file ``materialize_team_scene`` reads with ``arguments``."""

    files = [("materialized_reference", name, Path(arguments[name]))
             for name in ("configured_scene_bundle_path", "droid_profile_path", "runtime_source_receipt_path")]
    documents = attempt / "documents"
    # Every document and tray asset the host fetched for the revision, including those the mapping names.
    fetched = sorted(path for path in documents.rglob("*") if not path.is_dir()) if documents.is_dir() else []
    named = {Path(path) for path in dict(arguments["documents"]).values()}
    if not named <= set(fetched):
        return HostDecision("remote_ineligible:documents_outside_attempt")
    files += [("materialized_reference", f"documents.{path.relative_to(documents).as_posix()}", path)
              for path in fetched]
    files += [("materialized_reference", f"embodiment_model.{digest}", Path(path))
              for digest, path in sorted(dict(arguments.get("embodiment_models") or {}).items())]
    files += [("materialized_reference", label, path)
              for label, path in _confirmation_files(arguments.get("owner_success_confirmation_scope"))]
    return files


def worker_environment(jobs_root: str | Path) -> dict[str, Any] | None:
    """The worker environment the allocator's preflight probe recorded for this stage (plan 14 §5, §8)."""

    record = _read_record(Path(jobs_root) / "environment" / f"{STAGE}.json")
    worker = (record or {}).get("worker_environment")
    if (record is None or record.get("schema_version") != WORKER_ENVIRONMENT_SCHEMA_VERSION
            or not isinstance(worker, Mapping) or not set(DIGESTED_FIELDS) <= set(worker)
            or not record.get("environment_digest") == worker.get("environment_digest") == environment_digest(worker)):
        return None  # absent, unsealed, or sealed under a digest this host no longer computes: probe again
    return record


def record_job_image(jobs_root: str | Path, *, job_image: str, config_image: str, now: float) -> dict[str, Any]:
    """The paid unit's observation of this stage's job template image against the config's (plan 14 §15)."""

    record = {"schema_version": JOB_IMAGE_SCHEMA_VERSION, "stage": STAGE, "job_image": job_image,
              "config_image": config_image, "drift": job_image != config_image, "observed_at_epoch": float(now),
              "observation_digest": ""}
    record["observation_digest"] = canonical_digest(record, digest_field="observation_digest")
    path = Path(jobs_root) / "drift" / f"{STAGE}.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    temporary.chmod(0o640)
    os.replace(temporary, path)
    return record


def image_drift(jobs_root: str | Path, *, config_image: str) -> bool:
    """Drift: the probe's image or the last observed job template image is not the config's image."""

    recorded = worker_environment(jobs_root)
    if recorded is None or recorded.get("image") != config_image:
        return True
    observed = _read_record(Path(jobs_root) / "drift" / f"{STAGE}.json")
    return observed is not None and observed.get("job_image") != config_image


def _environment_refusal(jobs_root: Path, image: str, host: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    recorded = worker_environment(jobs_root)
    if recorded is None:
        return None, "remote_ineligible:environment_unrecorded"
    if image_drift(jobs_root, config_image=image):
        return None, "remote_ineligible:image_drift"
    worker = recorded["worker_environment"]
    for name in DIGESTED_FIELDS:  # everything but the CPU class, which the digest leaves out (plan 14 §5)
        if worker.get(name) != host.get(name):
            return None, f"remote_ineligible:environment_mismatch:{name}"
    if (worker.get("stage_distributions") or {}).get(STAGE) != (host.get("stage_distributions") or {}).get(STAGE):
        return None, "remote_ineligible:environment_mismatch:stage_distributions"
    return recorded, ""


def _worker_arguments(arguments: Mapping[str, Any], filesystem_root: Path) -> dict[str, Any] | None:
    """``materialize_team_scene``'s arguments with every path as the worker names it; ``None`` when one is not."""

    converted: dict[str, Any] = {"revision": arguments["revision"],
                                 "droid_profile_digest": arguments["droid_profile_digest"]}
    for name in PATH_ARGUMENTS:
        converted[name] = worker_path(arguments[name], filesystem_root)
    converted["documents"] = {name: worker_path(path, filesystem_root)
                              for name, path in sorted(dict(arguments["documents"]).items())}
    if "embodiment_manifests" in arguments:
        converted["embodiment_manifests"] = list(arguments["embodiment_manifests"])
    if "embodiment_models" in arguments:
        converted["embodiment_models"] = {digest: worker_path(path, filesystem_root)
                                          for digest, path in sorted(dict(arguments["embodiment_models"]).items())}
    scope = arguments.get("owner_success_confirmation_scope")
    if scope is not None:
        converted["owner_success_confirmation_scope"] = {"intent": scope["intent"],
                                                          "directory": worker_path(scope["directory"], filesystem_root)}
    paths = [converted[name] for name in PATH_ARGUMENTS] + list(converted["documents"].values()) + list(
        (converted.get("embodiment_models") or {}).values()) + (
        [converted["owner_success_confirmation_scope"]["directory"]] if scope is not None else [])
    return None if any(path is None for path in paths) else converted


def envelope_path(jobs_root: str | Path, name: str) -> Path:
    return Path(jobs_root) / ENVELOPES / STAGE / name


def _write_envelope(jobs_root: Path, attempt: Path, envelope: dict[str, Any]) -> tuple[Path, str]:
    """The sealed envelope, the queue row the worker materializes: ``<attempt name>-<envelope hex>.json``."""

    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    payload = (json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
               + "\n").encode("utf-8")
    name = f"{attempt.name}-{envelope['envelope_digest'].removeprefix('sha256:')}.json"
    path = envelope_path(jobs_root, name)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = path.with_name(f".{name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "xb") as stream:
            os.fchmod(stream.fileno(), 0o640)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise TeamScenePacketRemoteError("team_scene_packet_remote_envelope_conflict") from None
    finally:
        temporary.unlink(missing_ok=True)
    return path, name


def plan_remote_team_scene(
    *, request: Mapping[str, Any], capture_root: str | Path, attempt_root: str | Path, arguments: Mapping[str, Any],
    source_commit: str | None, config: Mapping[str, Any] | None, jobs_root: str | Path,
    filesystem_root: str | Path = "/", cache_root: str | Path | None = None,
    host_environment: Mapping[str, Any] | None = None,
) -> RemotePlan | HostDecision:
    """Decide, with no credential, whether one request's compile runs remotely, and write its envelope if so.

    ``arguments`` are exactly what the host compile would pass ``materialize_team_scene``; the revision's
    documents are already fetched under ``attempt_root/documents``."""

    root, attempt, jobs = Path(filesystem_root), Path(attempt_root), Path(jobs_root)
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        return HostDecision("remote_ineligible:release_unknown")
    if not isinstance(config, Mapping) or config_blockers(config) or STAGE not in config.get("stages", {}):
        return HostDecision("remote_ineligible:config_invalid")
    entry = config["stages"][STAGE]
    if _ATTEMPT.fullmatch(attempt.name) is None or Path(arguments["output_root"]) != attempt / "scene":
        return HostDecision("remote_ineligible:attempt_unrecognized")
    closure = _closure(Path(arguments["configured_scene_bundle_path"]), Path(cache_root or DEFAULT_CACHE_ROOT))
    if isinstance(closure, HostDecision):
        return closure
    closure_record, cache_files, asset_bytes = closure
    files = _plan_files(arguments, attempt)
    if isinstance(files, HostDecision):
        return files
    files += [("particlefield_cache_member", f"particlefield_cache.{path.name}", path) for path in cache_files]
    rows, seen = [], set()
    for role, contract_path, path in files:
        if str(path) in seen:
            continue
        seen.add(str(path))
        rows.append(_input(role, contract_path, path, root))
    converted = _worker_arguments(arguments, root)
    output_root, cache_variable = worker_path(attempt, root), worker_path(Path(cache_root or DEFAULT_CACHE_ROOT), root)
    if any(row is None for row in rows) or converted is None or output_root is None or (
            closure_record["class"] == "shipped" and cache_variable is None):
        return HostDecision("remote_ineligible:path_outside_permitted_roots")
    input_bytes = sum(row["size_bytes"] for row in rows)
    required = input_bytes + ASSET_COPIES * asset_bytes
    if input_bytes > STAGE_LIMITS["max_input_bytes"]:
        return HostDecision("remote_ineligible:input_bytes_exceed_limit")
    if required > entry["ephemeral_bytes"]:
        return HostDecision("remote_ineligible:ephemeral_budget_exceeded")
    host = dict(host_environment) if host_environment is not None else environment_record()
    recorded, refusal = _environment_refusal(jobs, entry["image"], host)
    if recorded is None:
        return HostDecision(refusal)
    envelope = {"schema_version": ENVELOPE_SCHEMA_VERSION, "request_id": str(request["request_id"]),
                "request_digest": str(request["request_digest"]),
                "scene_revision_digest": str(request["scene_revision_digest"]),
                "source_commit": source_commit, "arguments": converted, "envelope_digest": ""}
    envelope_file, name = _write_envelope(jobs, attempt, envelope)
    envelope_row = _input("queue_envelope", "queue_envelope", envelope_file, root)
    if envelope_row is None:
        envelope_file.unlink(missing_ok=True)
        return HostDecision("remote_ineligible:path_outside_permitted_roots")
    return RemotePlan(
        queue_row={"queue": QUEUE, "name": name, "envelope_digest": envelope["envelope_digest"]},
        request_id=str(request["request_id"]), scene_revision_digest=str(request["scene_revision_digest"]),
        capture_root=str(capture_root), attempt_root=str(attempt), output_root=f"{output_root}/scene",
        source_commit=source_commit, image=entry["image"], environment_digest=str(recorded["environment_digest"]),
        host_environment_digest=str(host["environment_digest"]), closure=closure_record,
        environment={CACHE_ROOT_VARIABLE: str(cache_variable)} if closure_record["class"] == "shipped" else {},
        inputs=(envelope_row, *rows), ephemeral_bytes_required=required + envelope_row["size_bytes"])


def input_sources(plan: RemotePlan) -> list[dict[str, Any]]:
    """Each plan input's host file, as the paid unit's staging takes it (path, digest and size)."""

    return [{"path": row["host_path"], "digest": row["digest"], "size_bytes": row["size_bytes"]} for row in plan.inputs]


def discard_envelope(jobs_root: str | Path, name: str) -> None:
    envelope_path(jobs_root, name).unlink(missing_ok=True)


def discard_attempt_envelopes(jobs_root: str | Path, attempt_root: str | Path) -> None:
    """The envelopes an interrupted hand-off of this attempt left (its row names begin with the attempt's name)."""

    directory = Path(jobs_root) / ENVELOPES / STAGE
    if directory.is_dir():
        for path in directory.glob(f"{Path(attempt_root).name}-*.json"):
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------- markers


def marker_path(jobs_root: str | Path, kind: str, name: str) -> Path:
    return Path(jobs_root) / MARKERS[kind] / STAGE / name


def _sealed(record: dict[str, Any], field_name: str) -> dict[str, Any]:
    record[field_name] = canonical_digest(record, digest_field=field_name)
    return record


def write_handoff(jobs_root: str | Path, plan: RemotePlan, *, mode: str, now: float,
                  host: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Hand a request's plan to the paid unit: one immutable marker, created exclusively (plan 14 §1).
    A shadow marker also names what the host compile wrote (``host``), for the comparison."""

    if mode not in {"authoritative", "shadow"}:
        raise TeamScenePacketRemoteError("team_scene_packet_handoff_mode_invalid")
    record = _sealed({"schema_version": HANDOFF_SCHEMA_VERSION, "mode": mode, "queue_row": dict(plan.queue_row),
                      "request_id": plan.request_id, "plan": plan.record(), "host": dict(host or {}),
                      "created_at_epoch": float(now), "handoff_digest": ""}, "handoff_digest")
    write_remote_cpu_record(marker_path(jobs_root, mode, plan.queue_row["name"]), record)
    return record


def write_fallback(jobs_root: str | Path, marker: Mapping[str, Any], *, reason: str, attempts: int,
                   now: float) -> dict[str, Any]:
    """Return a request to the no-spend unit, which compiles it on the host (plan 14 §10)."""

    record = _sealed({"schema_version": FALLBACK_SCHEMA_VERSION, "queue_row": dict(marker["queue_row"]),
                      "request_id": marker["request_id"], "attempt_root": marker["plan"]["attempt_root"],
                      "reason": reason, "attempts": int(attempts), "created_at_epoch": float(now),
                      "fallback_digest": ""}, "fallback_digest")
    write_remote_cpu_record(marker_path(jobs_root, "fallback", marker["queue_row"]["name"]), record)
    return record


def write_collected(jobs_root: str | Path, marker: Mapping[str, Any], *, attempt_id: str, result: Mapping[str, Any],
                    landed: Mapping[str, Any] | None, now: float) -> dict[str, Any]:
    """The paid unit's word that this request's remote compile is decided: its sealed result, and for a completed
    one, the tree it landed at the host's own output root."""

    record = _sealed({"schema_version": COLLECTED_SCHEMA_VERSION, "queue_row": dict(marker["queue_row"]),
                      "request_id": marker["request_id"], "attempt_root": marker["plan"]["attempt_root"],
                      "output_root": marker["plan"]["output_root"], "attempt_id": attempt_id, "result": dict(result),
                      "landed": dict(landed) if landed is not None else None, "collected_at_epoch": float(now),
                      "collected_digest": ""}, "collected_digest")
    write_remote_cpu_record(marker_path(jobs_root, "collected", marker["queue_row"]["name"]), record)
    return record


_MARKER_SEALS = {HANDOFF_SCHEMA_VERSION: "handoff_digest", FALLBACK_SCHEMA_VERSION: "fallback_digest",
                 COLLECTED_SCHEMA_VERSION: "collected_digest"}


def read_marker(path: str | Path) -> dict[str, Any] | None:
    """A sealed hand-off, shadow, fallback or collected marker; ``None`` when absent, foreign or unsealed."""

    record = _read_record(Path(path))
    seal = _MARKER_SEALS.get(str((record or {}).get("schema_version")))
    if (record is None or seal is None or record.get(seal) != canonical_digest(record, digest_field=seal)
            or not isinstance(record.get("queue_row"), Mapping) or record["queue_row"].get("name") != Path(path).name):
        return None
    return record


def markers(jobs_root: str | Path, kind: str) -> list[tuple[Path, dict[str, Any] | None]]:
    """Every marker of one kind, oldest name first; an unreadable one is listed as ``None``."""

    directory = Path(jobs_root) / MARKERS[kind] / STAGE
    return [(path, read_marker(path)) for path in sorted(directory.glob("*.json"))] if directory.is_dir() else []


def claims(jobs_root: str | Path) -> dict[str, dict[str, tuple[Path, dict[str, Any]]]]:
    """The requests the remote lane holds, by request id: each one's authoritative hand-off, collected and
    fallback markers. An unreadable marker names no request."""

    held: dict[str, dict[str, tuple[Path, dict[str, Any]]]] = {}
    for kind in ("authoritative", "collected", "fallback"):
        for path, marker in markers(jobs_root, kind):
            if marker is not None and isinstance(marker.get("request_id"), str):
                held.setdefault(marker["request_id"], {})[kind] = (path, marker)
    return held


def record_shadow_parity(jobs_root: str | Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one shadow comparison: ``parity/team_scene_packet/<attempt_id>.json``, once."""

    record = _sealed({"schema_version": PARITY_SCHEMA_VERSION, **dict(fields), "record_digest": ""}, "record_digest")
    write_remote_cpu_record(Path(jobs_root) / "parity" / STAGE / f"{record['attempt_id']}.json", record)
    return record


# ---------------------------------------------------------------------------------------------- worker


def _remote_blocker(exc: BaseException) -> str:
    """The host worker's blocker for this failure; resource exhaustion is typed as infrastructure, which the
    contract's retryable prefixes send to a fresh attempt and then to the host."""

    if isinstance(exc, MemoryError):
        return "team_scene_packet_failed:MemoryError"
    if isinstance(exc, OSError):
        return f"team_scene_packet_failed:OSError:{type(exc).__name__}"
    from .team_scene_packet_worker import _blocker

    return _blocker(exc)


def _local_arguments(arguments: Mapping[str, Any], roots: Any) -> dict[str, Any]:
    local: dict[str, Any] = {"revision": arguments["revision"], "droid_profile_digest": arguments["droid_profile_digest"]}
    for name in PATH_ARGUMENTS:
        local[name] = roots.local(arguments[name])
    local["documents"] = {name: roots.local(path) for name, path in arguments["documents"].items()}
    if "embodiment_manifests" in arguments:
        local["embodiment_manifests"] = list(arguments["embodiment_manifests"])
    if "embodiment_models" in arguments:
        local["embodiment_models"] = {digest: roots.local(path) for digest, path in arguments["embodiment_models"].items()}
    scope = arguments.get("owner_success_confirmation_scope")
    if scope is not None:
        local["owner_success_confirmation_scope"] = {"intent": scope["intent"],
                                                      "directory": roots.local(scope["directory"])}
    return local


def _worker_view(path: str, roots: Any) -> str:
    return "/" + Path(path).relative_to(Path(roots.filesystem_root)).as_posix()


def run_team_scene_packet_in_worker(descriptor: Mapping[str, Any], roots: Any) -> dict[str, Any]:
    """The worker's ``team_scene_packet`` stage: the host's ``materialize_team_scene`` with the envelope's
    arguments at the host's paths. Returns the sealed result; a compile refusal is a ``blocked`` result whose
    blocker is the host worker's own."""

    [row] = [item for item in descriptor["inputs"] if item["role"] == "queue_envelope"]
    envelope = json.loads(roots.local(row["materialize_at"]).read_bytes())
    if (envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION
            or envelope.get("envelope_digest") != canonical_digest(envelope, digest_field="envelope_digest")
            or envelope.get("source_commit") != descriptor["code"]["source_commit"]):
        raise TeamScenePacketRemoteError("team_scene_packet_remote_envelope_invalid")
    from .team_scene_packet import materialize_team_scene

    compiled, blockers, error_type = None, [], None
    try:
        made = materialize_team_scene(**_local_arguments(envelope["arguments"], roots))
        compiled = {"path": _worker_view(made["path"], roots), "inputs_digest": made["inputs_digest"],
                    "scene_id": made["scene_id"], "task_id": made["task_id"],
                    **({"owner_success_confirmation": made["owner_success_confirmation"]}
                       if made.get("owner_success_confirmation") else {})}
    except Exception as exc:  # noqa: BLE001 - every refusal is a typed result, as the host worker types it
        blockers, error_type = [_remote_blocker(exc)], type(exc).__name__
    result = {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked" if blockers else COMPLETED,
              "blockers": blockers, "source_commit": descriptor["code"]["source_commit"],
              "request_id": envelope["request_id"], "envelope_digest": envelope["envelope_digest"],
              "compiled": compiled, "error_type": error_type, "provider_mutation_performed": False,
              "paid_execution_requested": False, "result_digest": ""}
    return _sealed(result, "result_digest")


def validate_result(result: Any, *, request_id: str, envelope_digest: str, source_commit: str,
                    output_root: str) -> list[str]:
    """Why a worker's sealed result does not stand for this request's envelope, release and output root."""

    from .team_scene_packet import RECORD_FILENAME

    if not isinstance(result, Mapping):
        return ["result"]
    expected = {"schema_version": RESULT_SCHEMA_VERSION, "source_commit": source_commit, "request_id": request_id,
                "envelope_digest": envelope_digest, "provider_mutation_performed": False,
                "paid_execution_requested": False}
    reasons = [f"result_{name}" for name, value in expected.items() if result.get(name) != value]
    if result.get("result_digest") != canonical_digest(result, digest_field="result_digest"):
        reasons.append("result_seal")
    blockers, compiled = result.get("blockers"), result.get("compiled")
    if result.get("status") == COMPLETED:
        if blockers != [] or not isinstance(compiled, Mapping) or compiled.get("path") != f"{output_root}/{RECORD_FILENAME}":
            reasons.append("result_compiled")
    elif result.get("status") != "blocked" or not isinstance(blockers, list) or len(blockers) != 1 or compiled is not None:
        reasons.append("result_status")
    return reasons


__all__ = [
    "COMPLETED",
    "Dispatch",
    "EXECUTION_ENV",
    "HostDecision",
    "MODES",
    "RemotePlan",
    "STAGE",
    "TeamScenePacketRemoteError",
    "claims",
    "dispatch_context",
    "image_drift",
    "input_sources",
    "load_config",
    "marker_path",
    "markers",
    "plan_remote_team_scene",
    "read_marker",
    "record_job_image",
    "record_shadow_parity",
    "remote_configured",
    "resolve_execution_mode",
    "run_team_scene_packet_in_worker",
    "validate_result",
    "worker_environment",
    "worker_path",
    "write_collected",
    "write_fallback",
    "write_handoff",
]
