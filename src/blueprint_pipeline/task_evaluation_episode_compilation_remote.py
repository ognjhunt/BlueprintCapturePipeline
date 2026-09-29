"""Episode compilation on a remote CPU worker, behind a default-off flag (plan 14 §1, §5, §13).

``BLUEPRINT_EPISODE_COMPILATION_EXECUTION`` is ``host`` (the default: today's path),
``cloud_run_shadow`` or ``cloud_run``.  The no-spend unit owns ``pending/`` in every mode and never
holds a credential: it decides here, from the claimed envelope and the host's own files, whether a
row can compile remotely (``plan_remote_compilation``), and it leaves the paid unit a hand-off (or,
in shadow mode, a shadow marker) naming the plan.  An ineligible row compiles on the host.

Eligibility, in order: the envelope verifies exactly as the host compile would verify it; the
appearance closure class is ``not_applicable``, ``shipped`` (a valid host cache entry, whose files
ship as inputs with the cache-root environment) or ``absent_inline_only`` (a NuRec source small
enough to convert inline); the worker's ephemeral disk fits the compile; the probe-recorded worker
environment equals the host's on everything but the CPU class, for the configured image; and, for
``cloud_run`` only, the class has three consecutive shadow parity passes on the current image,
host environment and CPU class.

Inside the worker, ``run_episode_compilation_in_worker`` runs the host's own
``compile_claimed_envelope`` at the host's paths.  Nothing here imports allocation authority: the
worker's stage child imports this module from the release.
"""

from __future__ import annotations

import json
import os
import re
import stat
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .remote_cpu_environment import DIGESTED_FIELDS, environment_record
from .remote_cpu_job_contract import (
    CACHE_ROOT_VARIABLE,
    PERMITTED_PATH_ROOTS,
    STAGE_LIMITS,
    STAGES,
    RemoteCpuContractError,
    config_blockers,
)
from .remote_cpu_job_records import write_remote_cpu_record
from .task_evaluation_episode_compilation_worker import (
    TaskEvaluationEpisodeCompilationWorkerError,
    _load_envelope,
    _verified_references,
    compile_claimed_envelope,
)
from .task_evaluation_native_arena_episode_compiler import (
    MAXIMUM_INLINE_NUREC_CONVERSION_BYTES,
    compile_native_arena_episode,
)
from .task_evaluation_native_arena_preparation_adapter import MANIFEST_NAME

EXECUTION_ENV = "BLUEPRINT_EPISODE_COMPILATION_EXECUTION"
MODES = ("host", "cloud_run_shadow", "cloud_run")
JOBS_ROOT_ENV = "BLUEPRINT_REMOTE_CPU_JOBS_ROOT"
DEFAULT_JOBS_ROOT = "/var/lib/blueprint/pipeline-control-plane/remote-cpu-jobs"
STAGE = "episode_compilation"
QUEUE = STAGES[STAGE]["queue"]
# The unit's input root as the worker sees it: the host verifies every reference under it, and so does the worker.
PREPARED_REFERENCES = "/var/lib/blueprint/task-evaluation-inputs/prepared-references"
DEFAULT_CACHE_ROOT = "/var/lib/blueprint/task-evaluation-inputs/particlefield-runtime-assets"
MAX_INLINE_NUREC_BYTES = MAXIMUM_INLINE_NUREC_CONVERSION_BYTES
SHADOW_PASSES_REQUIRED = 3
# Plan 14 §9: the files later host stages read by path; nothing else lands on the host.
EPISODE_COMPILATION_CONSUMER_SUBSET = ("native-arena-adapter/**", "rigid_destination_native_probe_request.v1.json")
WORKER_ENVIRONMENT_SCHEMA_VERSION = "remote_cpu_worker_environment.v1"
JOB_IMAGE_SCHEMA_VERSION = "remote_cpu_job_image_observation.v1"
PARITY_SCHEMA_VERSION = "task_evaluation_episode_compilation_shadow_parity.v1"
BUNDLE_MANIFEST = "configured_scene_bundle_candidate.v1.json"
CACHE_MANIFEST = "particlefield_runtime_asset_cache.v1.json"
CACHE_SCHEMA_VERSION = "particlefield_runtime_asset_cache.v1"
_MAX_RECORD_BYTES = 4 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}")


class TaskEvaluationEpisodeCompilationRemoteError(RuntimeError):
    """A typed refusal on the remote episode-compilation path; the message is the blocker."""


@dataclass(frozen=True)
class HostDecision:
    """The row compiles on the host, for this reason."""

    reason: str


@dataclass(frozen=True)
class RemotePlan:
    """Everything a descriptor needs except staged URIs and the attempt: paths are the worker's (``/``-relative),
    and each input also names ``host_path``, the host file its bytes are staged and landed from."""

    queue_row: dict[str, Any]
    compilation_id: str
    source_commit: str
    image: str
    environment_digest: str
    closure: dict[str, Any]
    environment: dict[str, str]
    inputs: tuple[dict[str, Any], ...]
    output_root: str
    declared_scratch: tuple[str, ...]
    allowed_cpu_classes: tuple[str, ...]
    ephemeral_bytes_required: int

    def record(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> RemotePlan:
        fields = {name: value[name] for name in cls.__dataclass_fields__}
        for name in ("inputs", "declared_scratch", "allowed_cpu_classes"):
            fields[name] = tuple(fields[name])
        return cls(**fields)


def _read_record(path: Path, *, forbidden_mode: int = 0o022) -> dict[str, Any] | None:
    """A regular, non-symlinked JSON object no other account may write; ``None`` when absent or unusable."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    with os.fdopen(descriptor, "rb") as stream:
        status = os.fstat(stream.fileno())
        if not stat.S_ISREG(status.st_mode) or status.st_mode & forbidden_mode:
            return None
        payload = stream.read(_MAX_RECORD_BYTES + 1)
    try:
        value = json.loads(payload) if len(payload) <= _MAX_RECORD_BYTES else None
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _worker_path(path: str | Path, filesystem_root: Path) -> str | None:
    """A host path as the worker names it: relative to the host's ``/`` and under the permitted roots."""

    candidate = Path(path)
    for base in (Path(filesystem_root), Path(filesystem_root).resolve()):
        try:
            relative = candidate.relative_to(base)
            break
        except ValueError:
            continue
    else:
        try:
            relative = candidate.resolve().relative_to(Path(filesystem_root).resolve())
        except (OSError, ValueError):
            return None
    text = "/" + relative.as_posix()
    if any(part in {"", ".", ".."} for part in text.split("/")[1:]):
        return None
    return text if any(text.startswith(root) for root in PERMITTED_PATH_ROOTS) else None


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(os.stat(path).st_mode):04o}"


def _file_identity(path: Path) -> tuple[str, int]:
    from .task_evaluation_episode_compilation_worker import _sha256_and_size

    return _sha256_and_size(path)


def _bundle_appearance(bundle: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """The configured-scene bundle's appearance row and all asset rows, from its one manifest member."""

    try:
        with zipfile.ZipFile(bundle) as archive:
            manifest = json.loads(archive.read(BUNDLE_MANIFEST).decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None
    rows = manifest.get("assets") if isinstance(manifest, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return None
    appearance = [row for row in rows if row.get("role") == "appearance"]
    sizes = [row.get("size_bytes") for row in rows]
    if (len(appearance) != 1 or not all(isinstance(size, int) and not isinstance(size, bool) for size in sizes)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(appearance[0].get("digest")))):
        return None
    return dict(appearance[0]), [dict(row) for row in rows]


def _appearance_is_nurec(bundle: Path, member: str) -> bool | None:
    """A NuRec appearance is a USDZ holding a ``.nurec`` payload; ``None`` when that cannot be read."""

    try:
        with zipfile.ZipFile(bundle) as archive, archive.open(member) as stream:
            if stream.read(4) != b"PK\x03\x04":
                return False
            stream.seek(0)
            with zipfile.ZipFile(stream) as usdz:
                return any(name.endswith(".nurec") for name in usdz.namelist())
    except (OSError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile):
        return None


def _cache_entry(cache_root: Path, source_digest: str) -> list[Path] | str | None:
    """The files of a valid host cache entry for one appearance digest (plan 14 §13, credential-free: the
    worker re-validates the particlefield itself), ``"invalid"`` when an entry exists but does not verify."""

    root = Path(cache_root) / source_digest.removeprefix("sha256:")
    manifest_path = root / CACHE_MANIFEST
    if not os.path.lexists(manifest_path):
        return None
    manifest = _read_record(manifest_path)
    if (manifest is None or manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("source_configured_appearance_digest") != source_digest
            or manifest.get("immutable") is not True
            or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")):
        return "invalid"
    files = [manifest_path]
    for name in ("particlefield", "authoring_receipt"):
        row = manifest.get(name)
        relative = str((row or {}).get("relative_path") or "") if isinstance(row, Mapping) else ""
        path = root / relative
        if (not relative or "/" in relative or relative in {".", ".."} or path.is_symlink() or not path.is_file()
                or _file_identity(path) != (row.get("digest"), row.get("size_bytes"))):
            return "invalid"
        files.append(path)
    return sorted(files)


def _runtime_bundle_bytes(path: Path, size: int) -> int:
    """Bytes the runtime-source members extract to (its manifest), or the bundle's own size when unreadable."""

    try:
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        entries = manifest.get("entries")
        return sum(int(row["size_bytes"]) for row in entries if isinstance(row, Mapping))
    except (OSError, KeyError, ValueError, TypeError, AttributeError, zipfile.BadZipFile):
        return size


def worker_environment(jobs_root: str | Path) -> dict[str, Any] | None:
    """The worker environment the allocator's preflight probe recorded for this stage (plan 14 §5, §8)."""

    record = _read_record(Path(jobs_root) / "environment" / f"{STAGE}.json")
    worker = (record or {}).get("worker_environment")
    if (record is None or record.get("schema_version") != WORKER_ENVIRONMENT_SCHEMA_VERSION
            or not isinstance(worker, Mapping) or not set(DIGESTED_FIELDS) <= set(worker)
            or record.get("environment_digest") != worker.get("environment_digest")):
        return None
    return record


def record_job_image(jobs_root: str | Path, *, job_image: str, config_image: str, now: float) -> dict[str, Any]:
    """The paid unit's observation of the job template's image against the config's (plan 14 §15 drift)."""

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


def record_shadow_parity(jobs_root: str | Path, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one shadow comparison (plan 14 §13): ``parity/episode_compilation/<attempt_id>.json``, once."""

    record = {"schema_version": PARITY_SCHEMA_VERSION, **dict(fields), "record_digest": ""}
    record["record_digest"] = canonical_digest(record, digest_field="record_digest")
    write_remote_cpu_record(Path(jobs_root) / "parity" / STAGE / f"{record['attempt_id']}.json", record)
    return record


def shadow_passes(jobs_root: str | Path, *, closure_class: str, image: str, host_environment_digest: str,
                  cpu_class: str | None) -> int:
    """Consecutive trailing parity passes for one class on this image, host environment and CPU class."""

    rows = []
    for path in sorted((Path(jobs_root) / "parity" / STAGE).glob("*.json")):
        record = _read_record(path)
        if (record is not None and record.get("schema_version") == PARITY_SCHEMA_VERSION
                and record.get("record_digest") == canonical_digest(record, digest_field="record_digest")
                and (record.get("closure_class"), record.get("image"), record.get("host_environment_digest"),
                     record.get("cpu_class")) == (closure_class, image, host_environment_digest, cpu_class)):
            rows.append((float(record.get("compared_at_epoch") or 0.0), record.get("parity") == "passed"))
    count = 0
    for _, passed in sorted(rows, reverse=True):
        if not passed:
            break
        count += 1
    return count


def _closure(envelope: Mapping[str, Any], references: Mapping[str, Mapping[str, Any]],
             cache_root: Path) -> tuple[dict[str, Any], list[Path], int] | HostDecision:
    """The appearance closure class, the cache files it ships, and the configured-scene asset bytes."""

    bundle = Path(str(references["scene.configured_revision.configured_scene_bundle"]["materialized_path"]))
    parsed = _bundle_appearance(bundle)
    if parsed is None:
        return HostDecision("remote_ineligible:configured_scene_bundle_unreadable")
    appearance, rows = parsed
    assets = sum(int(row["size_bytes"]) for row in rows)
    setup = (envelope["request"].get("execution_adapter") or {}).get("policy_observation_setup")
    if setup is not None:
        return {"class": "not_applicable", "source_appearance_digest": None}, [], assets
    digest = str(appearance["digest"])
    entry = _cache_entry(cache_root, digest)
    if entry == "invalid":
        return HostDecision("remote_ineligible:particlefield_cache_invalid")
    if entry:
        return {"class": "shipped", "source_appearance_digest": digest}, list(entry), assets
    nurec = _appearance_is_nurec(bundle, str(appearance.get("relative_path") or ""))
    if nurec is None:
        return HostDecision("remote_ineligible:configured_appearance_unreadable")
    if not nurec:
        return {"class": "not_applicable", "source_appearance_digest": None}, [], assets
    if int(appearance["size_bytes"]) <= MAX_INLINE_NUREC_BYTES:
        return {"class": "absent_inline_only", "source_appearance_digest": digest}, [], assets
    return HostDecision("remote_ineligible:particlefield_transcode_required")


def _input(role: str, contract_path: str, path: Path, filesystem_root: Path, *,
           identity: tuple[str, int] | None = None) -> dict[str, Any] | None:
    target = _worker_path(path, filesystem_root)
    if target is None:
        return None
    digest, size = identity or _file_identity(path)
    return {"role": role, "contract_path": contract_path, "digest": digest, "size_bytes": size, "mode": _mode(path),
            "materialize_at": target, "host_path": str(path)}


def _inputs(claimed: Path, references: Mapping[str, Mapping[str, Any]], cache_files: Sequence[Path],
            filesystem_root: Path) -> list[dict[str, Any]] | None:
    rows = [_input("queue_envelope", "queue_envelope", claimed, filesystem_root)]
    seen = set()
    for contract_path, row in sorted(references.items()):
        path = Path(str(row["materialized_path"]))
        if str(path) not in seen:
            seen.add(str(path))
            rows.append(_input("materialized_reference", contract_path, path, filesystem_root,
                               identity=(row["digest"], row["size_bytes"])))
    rows.extend(_input("particlefield_cache_member", f"particlefield_cache.{path.name}", path, filesystem_root)
                for path in cache_files)
    return None if any(row is None for row in rows) else rows


def plan_remote_compilation(
    claimed: str | Path, *, inputs: str | Path, outputs: str | Path, source_commit: str, config: Mapping[str, Any],
    jobs_root: str | Path, filesystem_root: str | Path = "/", cache_root: str | Path | None = None,
    host_environment: Mapping[str, Any] | None = None, require_shadow_gate: bool = True,
) -> RemotePlan | HostDecision:
    """Decide, with no credential, whether one claimed row compiles remotely (plan 14 §13)."""

    claimed, root = Path(claimed), Path(filesystem_root)
    try:
        envelope = _load_envelope(claimed)
        if envelope["expected_production_commit"] != source_commit:
            raise TaskEvaluationEpisodeCompilationWorkerError("episode_compilation_source_commit_mismatch")
        references = _verified_references(envelope, input_root=Path(inputs).resolve(strict=True))
    except (TaskEvaluationEpisodeCompilationWorkerError, OSError):
        return HostDecision("remote_ineligible:envelope_invalid")
    if not isinstance(config, Mapping) or config_blockers(config) or STAGE not in config.get("stages", {}):
        return HostDecision("remote_ineligible:config_invalid")
    entry = config["stages"][STAGE]
    if _worker_path(Path(inputs), root) != PREPARED_REFERENCES:
        return HostDecision("remote_ineligible:input_root_unsupported")
    closure = _closure(envelope, references, Path(cache_root or DEFAULT_CACHE_ROOT))
    if isinstance(closure, HostDecision):
        return closure
    closure_record, cache_files, asset_bytes = closure
    rows = _inputs(claimed, references, cache_files, root)
    output_parent = _worker_path(Path(outputs), root)
    if rows is None or output_parent is None:
        return HostDecision("remote_ineligible:path_outside_permitted_roots")
    input_bytes = sum(row["size_bytes"] for row in rows)
    runtime = references.get("execution_adapter.runtime_source_bundle")
    runtime_bytes = 0 if runtime is None else _runtime_bundle_bytes(
        Path(str(runtime["materialized_path"])), int(runtime["size_bytes"]))
    required = input_bytes + 3 * asset_bytes + runtime_bytes
    if input_bytes > STAGE_LIMITS["max_input_bytes"]:
        return HostDecision("remote_ineligible:input_bytes_exceed_limit")
    if required > entry["ephemeral_bytes"]:
        return HostDecision("remote_ineligible:ephemeral_budget_exceeded")
    recorded = worker_environment(jobs_root)
    if recorded is None:
        return HostDecision("remote_ineligible:environment_unrecorded")
    if image_drift(jobs_root, config_image=entry["image"]):
        return HostDecision("remote_ineligible:image_drift")
    host = dict(host_environment) if host_environment is not None else environment_record()
    worker = recorded["worker_environment"]
    for field in DIGESTED_FIELDS:
        if field != "cpu_class" and worker.get(field) != host.get(field):
            return HostDecision(f"remote_ineligible:environment_mismatch:{field}")
    cpu_class = recorded.get("cpu_class")
    inline = closure_record["class"] == "absent_inline_only"
    if inline and not cpu_class:
        return HostDecision("remote_ineligible:cpu_class_unmeasured")
    if require_shadow_gate and shadow_passes(
            jobs_root, closure_class=closure_record["class"], image=entry["image"],
            host_environment_digest=host["environment_digest"], cpu_class=cpu_class) < SHADOW_PASSES_REQUIRED:
        return HostDecision(f"remote_ineligible:shadow_parity_unproven:{closure_record['class']}")
    cache_variable = _worker_path(Path(cache_root or DEFAULT_CACHE_ROOT), root)
    if closure_record["class"] == "shipped" and cache_variable is None:
        return HostDecision("remote_ineligible:path_outside_permitted_roots")
    return RemotePlan(
        queue_row={"queue": QUEUE, "name": claimed.name, "envelope_digest": envelope["envelope_digest"]},
        compilation_id=str(envelope["compilation_id"]), source_commit=source_commit, image=entry["image"],
        environment_digest=str(recorded["environment_digest"]), closure=closure_record,
        environment={CACHE_ROOT_VARIABLE: cache_variable} if closure_record["class"] == "shipped" else {},
        inputs=tuple(rows), output_root=f"{output_parent}/{envelope['compilation_id']}",
        declared_scratch=(f"{output_parent}/content-addressed/",),
        allowed_cpu_classes=(str(cpu_class),) if inline else (), ephemeral_bytes_required=required)


def input_sources(plan: RemotePlan, queue_root: str | Path) -> list[dict[str, Any]]:
    """Each plan input's current host file: the row's envelope moves between queue states after the plan
    (a shadow row is compiled and moved before its attempt stages), every other input stays where it was."""

    rows = []
    for row in plan.inputs:
        path = Path(row["host_path"])
        if row["role"] == "queue_envelope":
            found = [Path(queue_root) / state / path.name for state in ("processing", "completed", "blocked", "pending")
                     if (Path(queue_root) / state / path.name).is_file()]
            if len(found) != 1:
                raise TaskEvaluationEpisodeCompilationRemoteError("remote_episode_compilation_row_unlocated")
            path = found[0]
        rows.append({"path": str(path), "digest": row["digest"], "size_bytes": row["size_bytes"]})
    return rows


def run_episode_compilation_in_worker(descriptor: Mapping[str, Any], roots: Any, *,
                                      episode_compiler: Any = None) -> dict[str, Any]:
    """The worker's ``episode_compilation`` stage: the host's ``compile_claimed_envelope`` at the host's paths,
    with no disk reservation and no pin (plan 14 §13).  Returns the sealed result the host would write."""

    [envelope] = [row for row in descriptor["inputs"] if row["role"] == "queue_envelope"]
    claimed = roots.local(envelope["materialize_at"])
    outputs = roots.local(str(PurePosixPath(descriptor["outputs"]["output_root"]).parent))
    outputs.mkdir(parents=True, exist_ok=True, mode=0o750)
    _, result = compile_claimed_envelope(
        claimed, source_name=claimed.name, inputs=roots.local(PREPARED_REFERENCES).resolve(strict=True),
        outputs=outputs.resolve(strict=True), source_commit=descriptor["code"]["source_commit"],
        episode_compiler=episode_compiler or compile_native_arena_episode, disk_reservation_root=None,
        storage_pins_root=None)
    return result


__all__ = [
    "EPISODE_COMPILATION_CONSUMER_SUBSET",
    "EXECUTION_ENV",
    "HostDecision",
    "MODES",
    "RemotePlan",
    "RemoteCpuContractError",
    "TaskEvaluationEpisodeCompilationRemoteError",
    "image_drift",
    "input_sources",
    "plan_remote_compilation",
    "record_job_image",
    "record_shadow_parity",
    "run_episode_compilation_in_worker",
    "shadow_passes",
    "worker_environment",
]
