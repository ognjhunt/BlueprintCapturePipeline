"""Remote CPU job records (plan 14): descriptors, receipts, heartbeats, teardowns, output pointers.

The host seals a digest-bound descriptor, the worker echoes it in heartbeats and a receipt, the
host derives compute- and provider-zero from evidence, and a landed output gets a resealable
pointer.  No record holds a URL or a credential; the ``remote_cpu_job_transport.v1`` object that
carries presigned authority is refused by every host writer.  Nothing here contacts a provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

DESCRIPTOR_SCHEMA_VERSION = "remote_cpu_job_descriptor.v1"
TRANSPORT_SCHEMA_VERSION = "remote_cpu_job_transport.v1"
RECEIPT_SCHEMA_VERSION = "remote_cpu_job_receipt.v1"
HEARTBEAT_SCHEMA_VERSION = "remote_cpu_job_heartbeat.v1"
TEARDOWN_SCHEMA_VERSION = "remote_cpu_job_teardown.v1"
POINTER_SCHEMA_VERSION = "remote_cpu_output_pointer.v1"
CONFIG_SCHEMA_VERSION = "remote_cpu_workers_config.v1"
OUTPUT_FORMAT = "remote_cpu_output.v1"
SOURCE_ARCHIVE_RECIPE = "git_archive_tar.v2"
SOURCE_ARCHIVE_PATHS = ("src", "docs/schemas", "pyproject.toml")
EXECUTION_PROVIDER = "gcp_cloud_run_job"
PERMITTED_PATH_ROOTS = ("/var/lib/blueprint/",)
CACHE_ROOT_VARIABLE = "BLUEPRINT_PARTICLEFIELD_RUNTIME_ASSET_CACHE_ROOT"
ENVIRONMENT_VARIABLES = frozenset({CACHE_ROOT_VARIABLE})
INPUT_ROLES = ("queue_envelope", "materialized_reference", "particlefield_cache_member")
CLOSURE_CLASSES = ("not_applicable", "shipped", "absent_inline_only")
PHASES = ("bootstrap", "fetch", "stage", "seal_upload")
POINTER_STATES = ("landed", "restored_full")
MAX_ATTEMPTS_CAP = 2
PHASE_MARGIN_SECONDS = 180
INFRASTRUCTURE_FAILED = "infrastructure_failed:"
RELEASE_PATH_MISSING = INFRASTRUCTURE_FAILED + "release_path_missing:"

# Stages the host may describe; an unregistered stage is refused everywhere.
STAGES: Mapping[str, Mapping[str, Any]] = {
    "episode_compilation": {
        "abbreviation": "ec", "stage_contract": "episode_compilation_remote.v1",
        "queue": "task-evaluation-episode-compilations", "success_status": "compiled_for_production_launch",
        # Resource exhaustion inside the worker is infrastructure, not the row.
        "retryable_blocker_prefixes": ("episode_compilation_failed:OSError:", "episode_compilation_failed:MemoryError"),
    },
}

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_NAME = re.compile(r"[a-z][a-z0-9-]{0,126}[a-z0-9]")
_QUEUE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\.json")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9.-]*(?::[0-9]+)?/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]")
_KEY_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")
_IDENTITY = re.compile(r"gcp-cloud-run:([a-z][a-z0-9-]{4,28}[a-z0-9])/(us-[a-z0-9-]+)/([a-z][a-z0-9-]{0,62})"
                       r"/executions/([a-z][a-z0-9-]{0,126}[a-z0-9])")
_MODE = re.compile(r"0[0-7]{3}")
_OUTCOME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,255}")
_FORBIDDEN_KEY_FRAGMENTS = ("token", "secret", "password", "key", "credential", "authorization", "signature")
_URI_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://")
_CREDENTIAL_TEXT = re.compile(
    r"x-amz-|x-goog-|awsaccesskeyid=|googleaccessid=|signature=|bearer\s|-----begin|ya29\.", re.IGNORECASE)
_ACCESS_KEY_ID = re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")


class RemoteCpuContractError(ValueError):
    """A remote CPU record is malformed, unbound, or carries forbidden content."""

    def __init__(self, reasons: str | Sequence[str]):
        items = [reasons] if isinstance(reasons, str) else list(reasons)
        self.reasons = tuple(sorted({str(item) for item in items if str(item)}))
        super().__init__("; ".join(self.reasons))


def _raise_if(reasons: Sequence[str]) -> None:
    if reasons:
        raise RemoteCpuContractError(reasons)


def _url_or_credential_text(text: str) -> bool:
    if _CREDENTIAL_TEXT.search(text) or _ACCESS_KEY_ID.search(text):
        return True
    schemes = [match.group(1).lower() for match in _URI_SCHEME.finditer(text)]
    if any(scheme not in {"s3", "gs"} for scheme in schemes):
        return True
    # Object names never carry a query, a fragment or user information.
    return bool(schemes) and any(character in text for character in "?#@")


def forbidden_record_content(value: Any, path: str = "") -> list[str]:
    """Name every credential-shaped key, URL-shaped value and embedded transport, never the value."""
    reasons: list[str] = []
    if isinstance(value, Mapping):
        if value.get("schema_version") == TRANSPORT_SCHEMA_VERSION:
            reasons.append(f"remote_cpu_transport_never_persisted:{path or '$'}")
        for key, item in value.items():
            where = f"{path}.{key}" if path else str(key)
            if any(fragment in str(key).lower() for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                reasons.append(f"remote_cpu_record_credential_shaped_key:{where}")
            elif _url_or_credential_text(str(key)):
                reasons.append(f"remote_cpu_record_url_or_credential_value:{where}")
            reasons.extend(forbidden_record_content(item, where))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reasons.extend(forbidden_record_content(item, f"{path}[{index}]"))
    elif isinstance(value, str) and _url_or_credential_text(value):
        reasons.append(f"remote_cpu_record_url_or_credential_value:{path or '$'}")
    return reasons


def record_bytes(value: Mapping[str, Any]) -> bytes:
    """Canonical bytes of a host record, after the transport and credential guards."""
    if not isinstance(value, Mapping):
        raise RemoteCpuContractError("remote_cpu_record_not_mapping")
    reasons = forbidden_record_content(value)
    _raise_if(reasons)
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RemoteCpuContractError("remote_cpu_record_not_json") from exc
    return (text + "\n").encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_remote_cpu_record(path: str | Path, value: Mapping[str, Any]) -> bool:
    """Create one immutable record (exclusive, fsynced, 0640); ``False`` if it already exists identically."""
    payload = record_bytes(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(8)}.tmp")
    try:
        with open(temporary, "xb") as stream:
            os.fchmod(stream.fileno(), 0o640)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        created = True
    except FileExistsError:
        created = False
    finally:
        temporary.unlink(missing_ok=True)
    if created:
        fsync_directory(destination.parent)
        return True
    with suppress(OSError):
        if not destination.is_symlink() and destination.read_bytes() == payload:
            return False
    raise RemoteCpuContractError(f"remote_cpu_record_conflict:{destination.name}")


def job_id_for(stage: str, queue_name: str) -> str:
    """``rcj-<stage>-<24 hex of sha256(queue name)>``: one job per queue row."""
    if stage not in STAGES:
        raise RemoteCpuContractError("remote_cpu_stage_unknown")
    return f"rcj-{STAGES[stage]['abbreviation']}-{hashlib.sha256(queue_name.encode('utf-8')).hexdigest()[:24]}"


def worker_identity_for(execution: Mapping[str, Any], execution_name: str) -> str:
    identity = (f"gcp-cloud-run:{execution.get('project')}/{execution.get('region')}/"
                f"{execution.get('job')}/executions/{execution_name}")
    if not _IDENTITY.fullmatch(identity):
        raise RemoteCpuContractError("remote_cpu_worker_identity_invalid")
    return identity


def execution_name_of(worker_identity: str) -> str:
    match = _IDENTITY.fullmatch(str(worker_identity or ""))
    if match is None:
        raise RemoteCpuContractError("remote_cpu_worker_identity_invalid")
    return match.group(4)


def _is_count(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_amount(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _matches(pattern: re.Pattern[str]) -> Callable[[Any], bool]:
    return lambda value: isinstance(value, str) and pattern.fullmatch(value) is not None


def _one_of(*options: Any) -> Callable[[Any], bool]:
    return lambda value: any(value == option and type(value) is type(option) for option in options)


def _nullable(predicate: Callable[[Any], bool]) -> Callable[[Any], bool]:
    return lambda value: value is None or predicate(value)


_is_digest = _matches(_DIGEST)
_text = _matches(re.compile(r"(?s).{1,1024}"))
_positive = partial(_is_count, minimum=1)
_flag = _one_of(True, False)
_optional_digest = _nullable(_is_digest)


def _check(value: Any, spec: Any, where: str, reasons: list[str]) -> None:
    """Exact-key structure: a dict spec, a one-item list spec, or a predicate."""
    if isinstance(spec, (dict, list)) and not isinstance(value, Mapping if isinstance(spec, dict) else list):
        reasons.append(f"remote_cpu_field_invalid:{where or '$'}")
    elif isinstance(spec, dict):
        prefix = f"{where}." if where else ""
        reasons.extend(f"remote_cpu_field_unexpected:{prefix}{key}" for key in sorted(set(value) - set(spec)))
        reasons.extend(f"remote_cpu_field_missing:{prefix}{key}" for key in spec if key not in value)
        for key in set(spec) & set(value):
            _check(value[key], spec[key], prefix + key, reasons)
    elif isinstance(spec, list):
        for index, item in enumerate(value):
            _check(item, spec[0], f"{where}[{index}]", reasons)
    elif not spec(value):
        reasons.append(f"remote_cpu_field_invalid:{where}")


def _clone(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RemoteCpuContractError(f"remote_cpu_{label}_not_mapping")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RemoteCpuContractError(f"remote_cpu_{label}_not_json") from exc


def _path_reasons(value: Any, where: str, roots: Sequence[str], *, directory: bool = False) -> list[str]:
    if not isinstance(value, str) or not value.startswith("/"):
        return [f"remote_cpu_path_not_absolute:{where}"]
    text = value[:-1] if directory and value.endswith("/") else value
    if "\x00" in text or any(part in {"", ".", ".."} for part in text.split("/")[1:]):
        return [f"remote_cpu_path_not_normalized:{where}"]
    if not any(text.startswith(root) and len(text) > len(root) for root in roots):
        return [f"remote_cpu_path_outside_allowed_roots:{where}"]
    return []


def _within(path: str, root: str) -> bool:
    return path.rstrip("/") == root.rstrip("/") or path.startswith(root.rstrip("/") + "/")


def _cas_uri_reasons(uri: Any, digest: Any, where: str, *, kind: str | None = None,
                     filename: str | None = None) -> list[str]:
    """``s3://<bucket>/…/<kind>/sha256/<hex>/<file>`` whose hex is the digest."""
    if not isinstance(uri, str) or not uri.startswith("s3://") or _url_or_credential_text(uri):
        return [f"remote_cpu_uri_invalid:{where}"]
    bucket, _, key = uri[5:].partition("/")
    parts = key.split("/")
    if not _BUCKET.fullmatch(bucket) or not all(map(_KEY_PART.fullmatch, parts)):
        return [f"remote_cpu_uri_invalid:{where}"]
    if (not _is_digest(digest) or len(parts) < 4 or parts[-3] != "sha256" or parts[-2] != digest[7:]
            or kind not in {None, parts[-4]} or filename not in {None, parts[-1]}):
        return [f"remote_cpu_uri_not_content_addressed:{where}"]
    return []


def _row_id(row: Mapping[str, Any]) -> str | None:
    """The queue row's id when its name binds the envelope digest (``<id>-<hex>.json``)."""
    suffix = f"-{row['envelope_digest'].removeprefix('sha256:')}.json"
    return row["name"].removesuffix(suffix) if row["name"].endswith(suffix) else None


_JOB_LIMITS = ("vcpu", "memory_bytes", "ephemeral_bytes", "task_timeout_seconds")
_ROW_SPEC = {"queue": _text, "name": _matches(_QUEUE_NAME), "envelope_digest": _is_digest}
_DESCRIPTOR_SPEC = {
    "schema_version": _one_of(DESCRIPTOR_SCHEMA_VERSION), "job_id": _text, "attempt": _positive,
    "attempt_id": _matches(_NAME), "mode": _one_of("authoritative", "shadow"), "stage": _one_of(*STAGES),
    "stage_contract": _text, "queue_row": _ROW_SPEC, "environment": lambda v: isinstance(v, Mapping),
    "code": {
        "source_commit": _matches(_COMMIT), "image": _matches(_IMAGE), "environment_digest": _is_digest,
        "source_archive": {"recipe": _one_of(SOURCE_ARCHIVE_RECIPE), "paths": _one_of(list(SOURCE_ARCHIVE_PATHS)),
                           "digest": _is_digest, "size_bytes": _positive, "uri": _text},
    },
    "inputs": [{"role": _one_of(*INPUT_ROLES), "contract_path": _text, "digest": _is_digest,
                "size_bytes": _is_count, "mode": _matches(_MODE), "materialize_at": _text, "uri": _text}],
    "outputs": {"output_root": _text, "declared_scratch": lambda v: isinstance(v, list),
                "format": _one_of(OUTPUT_FORMAT), "staging_prefix": _text},
    "limits": {
        **{name: _positive for name in (*_JOB_LIMITS, "start_allowance_seconds", "heartbeat_interval_seconds",
                                         "heartbeat_stale_seconds", "max_input_bytes", "max_output_bytes",
                                         "max_output_paths", "max_attempts")},
        "phase_seconds": {"fetch": _positive, "stage": _positive, "seal_upload": _positive},
        "allowed_path_roots": lambda v: isinstance(v, list) and bool(v) and set(v) <= set(PERMITTED_PATH_ROOTS),
        "allowed_cpu_classes": lambda v: isinstance(v, list) and all(map(_is_digest, v)) and len(set(v)) == len(v),
    },
    "execution": {"provider": _one_of(EXECUTION_PROVIDER), "project": _text, "region": _text, "job": _text},
    "closure": {"class": _one_of(*CLOSURE_CLASSES), "source_appearance_digest": _optional_digest},
    "spend": {"worst_case_usd": _is_amount, "rate_table_digest": _is_digest},
    "private_url_recorded": _one_of(False), "descriptor_digest": _is_digest,
}
_OUTPUT_SPEC = {
    "format": _one_of(OUTPUT_FORMAT), "paths_total": _is_count, "bytes_total": _is_count,
    "index": {"digest": _is_digest, "size_bytes": _positive}, "archive": {"digest": _is_digest, "size_bytes": _positive},
    "host_known": {"count": _is_count, "bytes": _is_count},
}
_RECEIPT_SPEC = {
    "schema_version": _one_of(RECEIPT_SCHEMA_VERSION), "job_id": _text, "attempt": _positive, "attempt_id": _text,
    "stage": _text, "descriptor_digest": _is_digest, "execution_name": _matches(_NAME),
    "status": _one_of("succeeded", "blocked", "infrastructure_failed"),
    "result": _nullable(lambda v: isinstance(v, Mapping)), "output": _nullable(lambda v: isinstance(v, Mapping)),
    "infrastructure_failures": [lambda v: _text(v) and v.startswith(INFRASTRUCTURE_FAILED)],
    "release_path_misses": [lambda v: _text(v) and not v.startswith("/")
                            and all(part not in {"", ".", ".."} for part in v.split("/"))],
    "environment": {"environment_digest": _is_digest, "cpu_class": _optional_digest},
    "phases": {phase: _nullable(_is_amount) for phase in PHASES},
    "bytes_fetched": _is_count, "bytes_uploaded": _is_count,
    "private_url_recorded": _one_of(False), "receipt_digest": _is_digest,
}
_HEARTBEAT_SPEC = {
    "schema_version": _one_of(HEARTBEAT_SCHEMA_VERSION), "attempt_id": _text, "execution_name": _text,
    "sequence": _positive, "phase": _one_of(*PHASES), "elapsed_seconds": _is_amount,
    "bytes_fetched": _is_count, "bytes_uploaded": _is_count,
}
_COMPUTE_SPEC = {
    "execution_completed": _flag, "running_count": _is_count, "listing_complete": _flag, "listing_pages": _is_count,
    "executions_for_attempt": _is_count, "unfinished_executions_for_attempt": _is_count,
    "transport_deleted": _flag, "transport_absent_at_generation": _flag,
}
_PROVIDER_SPEC = {
    "staging_versions_deleted": _is_count, "staging_versions_remaining": _is_count,
    "staging_listing_complete": _flag, "urls_expire_at_epoch": _is_amount, "named_objects_absent": _flag,
}
_TEARDOWN_SPEC = {
    "schema_version": _one_of(TEARDOWN_SCHEMA_VERSION), "job_id": _text, "attempt": _positive,
    "attempt_id": _matches(_NAME), "stage": _one_of(*STAGES), "descriptor_digest": _is_digest,
    "worker_identity": _nullable(_matches(_IDENTITY)), "outcome": _matches(_OUTCOME),
    "compute_zero": _COMPUTE_SPEC, "compute_zero_proven": _flag, "provider_zero": _PROVIDER_SPEC,
    "provider_zero_proven": _flag, "observed_at_epoch": _is_amount, "private_url_recorded": _one_of(False),
    "teardown_digest": _is_digest,
}
_POINTER_SPEC = {
    "schema_version": _one_of(POINTER_SCHEMA_VERSION), "stage": _one_of(*STAGES), "compilation_id": _text,
    "queue_row": _ROW_SPEC, "attempt_id": _matches(_NAME), "descriptor_digest": _is_digest,
    "receipt_digest": _is_digest,
    "execution": {"provider": _one_of(EXECUTION_PROVIDER), "job": _text, "worker_identity": _matches(_IDENTITY),
                  "allocation_binding_digest": _is_digest, "spend_consumption": _is_digest},
    "code": {"source_commit": _matches(_COMMIT), "source_archive_digest": _is_digest, "image": _matches(_IMAGE),
             "environment_digest": _is_digest},
    "archive": {"uri": _text, "digest": _is_digest, "size_bytes": _positive},
    "index": {"uri": _text, "digest": _is_digest, "size_bytes": _positive},
    "output_root": _text, "paths_total": _is_count, "bytes_total": _is_count,
    "host_known": {"count": _is_count, "bytes": _is_count},
    "landed": {"subset": _matches(_OUTCOME), "paths": _is_count, "bytes": _is_count},
    "state": _one_of(*POINTER_STATES), "teardown_receipt_digest": _optional_digest,
    "provider_zero_proven": _flag, "pointer_digest": _is_digest,
}


def _config_stage(config: Mapping[str, Any], stage: str, reasons: list[str]) -> Mapping[str, Any]:
    entry = config.get("stages", {}).get(stage) if isinstance(config.get("stages"), Mapping) else None
    reasons.extend(reason for reason, failed in {
        "remote_cpu_config_invalid:schema_version": config.get("schema_version") != CONFIG_SCHEMA_VERSION,
        "remote_cpu_config_invalid:config_digest": config.get("config_digest") != canonical_digest(
            config, digest_field="config_digest"),
        "remote_cpu_config_region_not_us": not str(config.get("region") or "").startswith("us-"),
        f"remote_cpu_config_invalid:stage_missing:{stage}": not isinstance(entry, Mapping),
    }.items() if failed)
    return entry if isinstance(entry, Mapping) else {}


def _descriptor_path_reasons(descriptor: Mapping[str, Any], reasons: list[str]) -> list[str]:
    """Absolute, normalized, under the allowed roots, unique and bound; returns cache members."""
    roots, outputs, row = descriptor["limits"]["allowed_path_roots"], descriptor["outputs"], descriptor["queue_row"]
    output_root = outputs["output_root"]
    output_problems = _path_reasons(output_root, "outputs.output_root", roots)
    reasons.extend(output_problems)
    if not output_problems and output_root.rsplit("/", 1)[-1] != _row_id(row):
        reasons.append("remote_cpu_descriptor_output_root_unbound")
    for index, path in enumerate(outputs["declared_scratch"]):
        problems = _path_reasons(path, f"outputs.declared_scratch[{index}]", roots, directory=True)
        reasons.extend(problems)
        if not problems and not output_problems and (_within(path, output_root) or _within(output_root, path)):
            reasons.append(f"remote_cpu_descriptor_scratch_overlaps_output_root:{index}")
    for variable, path in sorted(descriptor["environment"].items()):
        if variable not in ENVIRONMENT_VARIABLES:
            reasons.append(f"remote_cpu_descriptor_environment_variable_not_allowed:{variable}")
        reasons.extend(_path_reasons(path, f"environment.{variable}", roots, directory=True))
    seen, cache_members = set(), []
    for index, item in enumerate(descriptor["inputs"]):
        where, target = f"inputs[{index}]", item["materialize_at"]
        reasons.extend(_cas_uri_reasons(item["uri"], item["digest"], f"{where}.uri"))
        problems = _path_reasons(target, f"{where}.materialize_at", roots)
        reasons.extend(problems)
        if problems:
            continue
        if target in seen:
            reasons.append(f"remote_cpu_descriptor_materialize_at_duplicate:{where}")
        seen.add(target)
        if not output_problems and _within(target, output_root):
            reasons.append(f"remote_cpu_descriptor_input_inside_output_root:{where}")
        if item["role"] == "queue_envelope" and target.rsplit("/", 1)[-1] != row["name"]:
            reasons.append(f"remote_cpu_descriptor_queue_envelope_unbound:{where}")
        if item["role"] == "particlefield_cache_member":
            cache_members.append(target)
    return cache_members


def validate_descriptor(value: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one sealed descriptor against the host's remote-worker config."""
    descriptor = _clone(value, "descriptor")
    reasons = forbidden_record_content(descriptor)
    _check(descriptor, _DESCRIPTOR_SPEC, "", reasons)
    if descriptor.get("descriptor_digest") != canonical_digest(descriptor, digest_field="descriptor_digest"):
        reasons.append("remote_cpu_descriptor_digest_mismatch")
    if not isinstance(config, Mapping):
        reasons.append("remote_cpu_config_invalid:not_mapping")
    _raise_if(reasons)
    stage, entry = STAGES[descriptor["stage"]], _config_stage(config, descriptor["stage"], reasons)
    row, code, limits, closure = descriptor["queue_row"], descriptor["code"], descriptor["limits"], descriptor["closure"]
    job_id, attempt_id, execution = descriptor["job_id"], descriptor["attempt_id"], descriptor["execution"]
    reasons.extend(_cas_uri_reasons(code["source_archive"]["uri"], code["source_archive"]["digest"],
                                    "code.source_archive.uri", kind="remote-cpu-source", filename="source.tar"))
    if code["image"] != entry.get("image"):
        reasons.append("remote_cpu_image_mismatch")
    reasons.extend(f"remote_cpu_descriptor_limits_do_not_match_job:{name}"
                   for name in _JOB_LIMITS if limits[name] != entry.get(name))
    cache_members = _descriptor_path_reasons(descriptor, reasons)
    cache_root = descriptor["environment"].get(CACHE_ROOT_VARIABLE)
    staging = descriptor["outputs"]["staging_prefix"]
    bucket, _, prefix = staging.removeprefix("s3://").partition("/")
    rate_table = config.get("rate_table")
    failed = {
        "stage_contract_invalid": descriptor["stage_contract"] != stage["stage_contract"] or row["queue"] != stage["queue"],
        "queue_row_unbound": _row_id(row) is None,
        "attempt_unbound": job_id != job_id_for(descriptor["stage"], row["name"])
        or not re.fullmatch(rf"{re.escape(job_id)}-a{descriptor['attempt']}-[0-9a-f]{{32}}", attempt_id),
        "attempt_exceeds_max_attempts": descriptor["attempt"] > limits["max_attempts"],
        "phase_budget_exceeds_task_timeout":
            sum(limits["phase_seconds"].values()) + PHASE_MARGIN_SECONDS > limits["task_timeout_seconds"],
        "limits_out_of_bounds": limits["task_timeout_seconds"] > 3600 or limits["memory_bytes"] > 32 * 1024**3
        or limits["heartbeat_interval_seconds"] >= limits["heartbeat_stale_seconds"],
        "max_attempts_invalid": not limits["max_attempts"] <= MAX_ATTEMPTS_CAP
        or limits["max_attempts"] != config.get("max_attempts"),
        "queue_envelope_input_count": [item["role"] for item in descriptor["inputs"]].count("queue_envelope") != 1,
        "input_bytes_exceed_limit": sum(item["size_bytes"] for item in descriptor["inputs"]) > limits["max_input_bytes"],
        "staging_prefix_unbound": not staging.startswith("s3://") or not _BUCKET.fullmatch(bucket)
        or not staging.endswith(f"/remote-cpu/staging/{job_id}/{attempt_id}/")
        or not all(map(_KEY_PART.fullmatch, prefix.rstrip("/").split("/"))),
        "region_not_us": not execution["region"].startswith("us-"),
        "execution_does_not_match_config": not execution["job"].startswith("blueprint-remote-cpu-") or execution != {
            "provider": EXECUTION_PROVIDER, "project": config.get("project"), "region": config.get("region"),
            "job": entry.get("job")},
        "closure_invalid": (closure["source_appearance_digest"] is None) != (closure["class"] == "not_applicable"),
        "closure_requires_qualified_cpu_classes":
            closure["class"] == "absent_inline_only" and not limits["allowed_cpu_classes"],
        "shipped_closure_unbound": closure["class"] == "shipped" and (
            not cache_members or not isinstance(cache_root, str) or not all(_within(p, cache_root) for p in cache_members)),
        "cache_members_without_shipped_closure": closure["class"] != "shipped" and bool(cache_members),
        "rate_table_unbound": not isinstance(rate_table, Mapping)
        or descriptor["spend"]["rate_table_digest"] != canonical_digest(rate_table),
    }
    reasons.extend(f"remote_cpu_descriptor_{name}" for name, failure in failed.items() if failure)
    _raise_if(reasons)
    return descriptor


def build_descriptor(
    *, config: Mapping[str, Any], stage: str, mode: str, attempt: int, queue_row: Mapping[str, Any],
    code: Mapping[str, Any], environment: Mapping[str, str], inputs: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Any], limits: Mapping[str, Any], closure: Mapping[str, Any], spend: Mapping[str, Any],
    nonce: str | None = None,
) -> dict[str, Any]:
    """Seal and validate a descriptor; the attempt id takes a random nonce unless one is given."""
    if stage not in STAGES or not all(isinstance(item, Mapping) for item in (config, queue_row, code, outputs)):
        raise RemoteCpuContractError("remote_cpu_descriptor_arguments_invalid")
    job_id = job_id_for(stage, str(queue_row.get("name") or ""))
    attempt_id = f"{job_id}-a{attempt}-{secrets.token_hex(16) if nonce is None else nonce}"
    archive = {"recipe": SOURCE_ARCHIVE_RECIPE, "paths": list(SOURCE_ARCHIVE_PATHS), **(code.get("source_archive") or {})}
    descriptor: dict[str, Any] = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION, "job_id": job_id, "attempt": attempt, "attempt_id": attempt_id,
        "mode": mode, "stage": stage, "stage_contract": STAGES[stage]["stage_contract"], "queue_row": dict(queue_row),
        "code": {**code, "source_archive": archive}, "environment": dict(environment),
        "inputs": [dict(row) for row in inputs],
        "outputs": {"output_root": outputs.get("output_root"), "declared_scratch": list(outputs.get("declared_scratch")
                    or []), "format": OUTPUT_FORMAT, "staging_prefix":
                    f"{str(outputs.get('object_prefix')).rstrip('/')}/remote-cpu/staging/{job_id}/{attempt_id}/"},
        "limits": _clone(limits, "limits"),
        "execution": {"provider": EXECUTION_PROVIDER, "project": config.get("project"), "region": config.get("region"),
                      "job": _config_stage(config, stage, []).get("job")},
        "closure": dict(closure), "spend": dict(spend), "private_url_recorded": False, "descriptor_digest": "",
    }
    descriptor["descriptor_digest"] = canonical_digest(descriptor, digest_field="descriptor_digest")
    return validate_descriptor(descriptor, config=config)


def validate_receipt(value: Mapping[str, Any], *, descriptor: Mapping[str, Any], execution_name: str) -> dict[str, Any]:
    """Fence a receipt to its attempt and execution, then classify it; a release-path miss, environment
    mismatch, unqualified CPU class or retryable blocker makes it an infrastructure failure."""
    receipt = _clone(value, "receipt")
    if not isinstance(descriptor, Mapping):
        raise RemoteCpuContractError("remote_cpu_descriptor_not_mapping")
    reasons = forbidden_record_content(receipt)
    _check(receipt, _RECEIPT_SPEC, "", reasons)
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        reasons.append("remote_cpu_receipt_digest_mismatch")
    expected = {field: descriptor.get(field) for field in ("job_id", "attempt", "attempt_id", "stage", "descriptor_digest")}
    expected["execution_name"] = execution_name if _matches(_NAME)(execution_name) else None
    reasons.extend(f"remote_cpu_receipt_fenced:{field}" for field, value in expected.items()
                   if value is None or receipt.get(field) != value)
    stage = STAGES.get(str(descriptor.get("stage")))
    if reasons or stage is None:
        raise RemoteCpuContractError(reasons or ["remote_cpu_stage_unknown"])
    status, output, limits = receipt["status"], receipt["output"], descriptor.get("limits") or {}
    result = receipt["result"] if isinstance(receipt["result"], Mapping) else {}
    blockers = result.get("blockers")
    if status == "succeeded":
        _check(output, _OUTPUT_SPEC, "output", reasons)
    reasons.extend(f"remote_cpu_receipt_{name}" for name, failed in ({} if reasons else {
        "result_invalid": status != "infrastructure_failed" and not result,
        "result_digest_mismatch": bool(result) and result.get("result_digest") != canonical_digest(
            result, digest_field="result_digest"),
        "result_source_commit_mismatch": bool(result) and result.get("source_commit") != (
            descriptor.get("code") or {}).get("source_commit"),
        "result_status_mismatch": status != "infrastructure_failed" and (
            result.get("status") != (stage["success_status"] if status == "succeeded" else "blocked")
            or not isinstance(blockers, list) or not all(map(_text, blockers)) or bool(blockers) != (status == "blocked")),
        "output_without_success": status != "succeeded" and output is not None,
        "output_host_known_exceeds_totals": status == "succeeded" and (
            output["host_known"]["count"] > output["paths_total"] or output["host_known"]["bytes"] > output["bytes_total"]),
        "output_exceeds_limits": status == "succeeded" and (output["paths_total"] > limits.get("max_output_paths", -1)
                                                            or output["bytes_total"] > limits.get("max_output_bytes", -1)),
        "infrastructure_failure_unnamed": status == "infrastructure_failed"
        and not receipt["infrastructure_failures"] + receipt["release_path_misses"],
    }).items() if failed)
    _raise_if(reasons)
    derived = [*receipt["infrastructure_failures"], *(RELEASE_PATH_MISSING + p for p in receipt["release_path_misses"])]
    if receipt["environment"]["environment_digest"] != (descriptor.get("code") or {}).get("environment_digest"):
        derived.append(INFRASTRUCTURE_FAILED + "environment_mismatch")
    if (descriptor.get("closure") or {}).get("class") == "absent_inline_only" and (
            receipt["environment"]["cpu_class"] not in limits.get("allowed_cpu_classes", [])):
        derived.append(INFRASTRUCTURE_FAILED + "cpu_class_unqualified")
    if status == "blocked":
        derived.extend(f"{INFRASTRUCTURE_FAILED}retryable_stage_blocker:{blocker}"
                       for blocker in receipt["result"]["blockers"] if blocker.startswith(stage["retryable_blocker_prefixes"]))
    derived = sorted(set(derived))
    return {"receipt": receipt, "outcome": "infrastructure_failed" if derived else status, "terminal": not derived,
            "infrastructure_failures": derived, "receipt_digest": receipt["receipt_digest"]}


def validate_heartbeat(value: Mapping[str, Any], *, attempt_id: str, execution_name: str) -> dict[str, Any]:
    """A heartbeat counts only under the current attempt id and its execution."""
    heartbeat = _clone(value, "heartbeat")
    reasons = forbidden_record_content(heartbeat)
    _check(heartbeat, _HEARTBEAT_SPEC, "", reasons)
    reasons.extend(f"remote_cpu_heartbeat_fenced:{name}" for name, expected in (
        ("attempt_id", attempt_id), ("execution_name", execution_name)) if not expected or heartbeat.get(name) != expected)
    _raise_if(reasons)
    return heartbeat


def compute_zero_proven(compute: Mapping[str, Any], *, worker_identity: str | None) -> bool:
    """Terminal execution, a complete listing with nothing unfinished, transport gone at its generation."""
    reasons: list[str] = []
    _check(compute, _COMPUTE_SPEC, "compute", reasons)
    _raise_if(reasons)
    # Without a known execution (a lost dispatch response) no execution may carry the attempt.
    ran = (compute["execution_completed"] and compute["running_count"] == 0) if worker_identity else (
        compute["executions_for_attempt"] == 0)
    return bool(ran and compute["listing_complete"] and compute["listing_pages"] >= 1
                and compute["unfinished_executions_for_attempt"] == 0
                and compute["transport_deleted"] and compute["transport_absent_at_generation"])


def _provider_zero(compute_zero: bool, provider: Mapping[str, Any], observed_at_epoch: float) -> bool:
    # B2 only hides a deleted object, so the proof is an empty version listing; a URL is
    # spent once it has expired or the object it names is gone.
    return bool(compute_zero and provider["staging_listing_complete"] and provider["staging_versions_remaining"] == 0
                and (observed_at_epoch >= provider["urls_expire_at_epoch"] or provider["named_objects_absent"]))


def teardown_record(
    *, descriptor: Mapping[str, Any], worker_identity: str | None, outcome: str, compute: Mapping[str, Any],
    provider: Mapping[str, Any], observed_at_epoch: float,
) -> dict[str, Any]:
    """Seal ``remote_cpu_job_teardown.v1``; both zero proofs are derived from evidence, never asserted."""
    reasons = forbidden_record_content({"outcome": outcome, "compute": compute, "provider": provider})
    _check(compute, _COMPUTE_SPEC, "compute", reasons)
    _check(provider, _PROVIDER_SPEC, "provider", reasons)
    if not isinstance(descriptor, Mapping) or descriptor.get("descriptor_digest") != canonical_digest(
            descriptor, digest_field="descriptor_digest"):
        reasons.append("remote_cpu_teardown_descriptor_unsealed")
    if reasons or not _is_amount(observed_at_epoch):
        raise RemoteCpuContractError(reasons or ["remote_cpu_field_invalid:observed_at_epoch"])
    if worker_identity is not None and worker_identity != worker_identity_for(
            descriptor.get("execution") or {}, execution_name_of(worker_identity)):
        raise RemoteCpuContractError("remote_cpu_teardown_worker_identity_unbound")
    compute_zero = compute_zero_proven(compute, worker_identity=worker_identity)
    record: dict[str, Any] = {
        "schema_version": TEARDOWN_SCHEMA_VERSION,
        **{name: descriptor.get(name) for name in ("job_id", "attempt", "attempt_id", "stage", "descriptor_digest")},
        "worker_identity": worker_identity, "outcome": outcome,
        "compute_zero": dict(compute), "compute_zero_proven": compute_zero, "provider_zero": dict(provider),
        "provider_zero_proven": _provider_zero(compute_zero, provider, float(observed_at_epoch)),
        "observed_at_epoch": float(observed_at_epoch), "private_url_recorded": False, "teardown_digest": "",
    }
    record["teardown_digest"] = canonical_digest(record, digest_field="teardown_digest")
    return validate_teardown(record)


def validate_teardown(value: Mapping[str, Any]) -> dict[str, Any]:
    """Check a sealed teardown and re-derive both proofs from its own evidence."""
    record = _clone(value, "teardown")
    reasons = forbidden_record_content(record)
    _check(record, _TEARDOWN_SPEC, "", reasons)
    if record.get("teardown_digest") != canonical_digest(record, digest_field="teardown_digest"):
        reasons.append("remote_cpu_teardown_digest_mismatch")
    _raise_if(reasons)
    compute_zero = compute_zero_proven(record["compute_zero"], worker_identity=record["worker_identity"])
    if record["compute_zero_proven"] is not compute_zero or record["provider_zero_proven"] is not _provider_zero(
            compute_zero, record["provider_zero"], record["observed_at_epoch"]):
        raise RemoteCpuContractError("remote_cpu_teardown_proof_not_derived")
    return record


def _pointer_reasons(pointer: Mapping[str, Any]) -> list[str]:
    reasons = forbidden_record_content(pointer)
    _check(pointer, _POINTER_SPEC, "", reasons)
    if reasons:
        return reasons
    row, stage, execution = pointer["queue_row"], pointer["stage"], pointer["execution"]
    paths, total, known, landed = pointer["paths_total"], pointer["bytes_total"], pointer["host_known"], pointer["landed"]
    reasons.extend(_path_reasons(pointer["output_root"], "output_root", PERMITTED_PATH_ROOTS))
    for name in ("archive", "index"):
        reasons.extend(_cas_uri_reasons(pointer[name]["uri"], pointer[name]["digest"], f"{name}.uri"))
    reasons.extend(f"remote_cpu_pointer_{name}" for name, failed in {
        "queue_row_unbound": row["queue"] != STAGES[stage]["queue"] or pointer["compilation_id"] != _row_id(row),
        "attempt_unbound": not re.fullmatch(rf"{re.escape(job_id_for(stage, row['name']))}-a[1-9][0-9]*-[0-9a-f]{{32}}",
                                            pointer["attempt_id"]),
        "output_root_unbound": pointer["output_root"].rsplit("/", 1)[-1] != pointer["compilation_id"],
        "worker_identity_unbound": _IDENTITY.fullmatch(execution["worker_identity"]).group(3) != execution["job"],
        "totals_invalid": known["count"] > paths or known["bytes"] > total or landed["paths"] > paths
        or landed["bytes"] > total,
        "provider_zero_unbound": pointer["provider_zero_proven"] is not (pointer["teardown_receipt_digest"] is not None),
    }.items() if failed)
    return reasons


def pointer_record(fields: Mapping[str, Any], *, previous: Mapping[str, Any] | None = None,
                   teardown: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Seal ``remote_cpu_output_pointer.v1``, or reseal ``previous``: only ``state`` advances
    (``landed`` then ``restored_full``) and a provider-zero teardown attaches; nothing is dropped."""
    fields = _clone(fields, "pointer_fields")
    reasons = forbidden_record_content(fields)
    _raise_if(reasons)
    if previous is None:
        reasons.extend(f"remote_cpu_pointer_field_derived:{name}" for name in (
            "schema_version", "teardown_receipt_digest", "provider_zero_proven", "pointer_digest") if name in fields)
        if fields.get("state") != "landed":
            reasons.append("remote_cpu_pointer_initial_state_invalid")
        pointer = {**fields, "schema_version": POINTER_SCHEMA_VERSION, "teardown_receipt_digest": None,
                   "provider_zero_proven": False, "pointer_digest": ""}
    else:
        pointer = _clone(previous, "pointer")
        prior = _pointer_reasons(pointer)
        if pointer.get("pointer_digest") != canonical_digest(pointer, digest_field="pointer_digest"):
            prior.append("remote_cpu_pointer_digest_mismatch")
        _raise_if(prior)
        for name, item in fields.items():
            if name != "state":
                if pointer.get(name, object()) != item:
                    reasons.append(f"remote_cpu_pointer_field_immutable:{name}")
            elif item not in POINTER_STATES:
                reasons.append("remote_cpu_pointer_state_invalid")
            elif POINTER_STATES.index(item) < POINTER_STATES.index(pointer["state"]):
                reasons.append("remote_cpu_pointer_state_regression")
            else:
                pointer["state"] = item
    if teardown is not None:
        record = validate_teardown(teardown)
        reasons.extend(name for name, failed in {
            "remote_cpu_pointer_teardown_attempt_mismatch": (record["attempt_id"], record["descriptor_digest"])
            != (pointer.get("attempt_id"), pointer.get("descriptor_digest")),
            "remote_cpu_pointer_teardown_not_provider_zero": not record["provider_zero_proven"],
            "remote_cpu_pointer_field_immutable:teardown_receipt_digest":
                pointer.get("teardown_receipt_digest") not in {None, record["teardown_digest"]},
        }.items() if failed)
        pointer["teardown_receipt_digest"], pointer["provider_zero_proven"] = record["teardown_digest"], True
    _raise_if(reasons)
    pointer["pointer_digest"] = canonical_digest(pointer, digest_field="pointer_digest")
    reasons = _pointer_reasons(pointer)
    _raise_if(reasons)
    return pointer
