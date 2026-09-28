"""Remote CPU job contract (plan 14): descriptors, receipts, heartbeats and the shared record guard.

The host seals a digest-bound descriptor, and the worker echoes it in heartbeats and a receipt;
``remote_cpu_job_records`` holds the teardowns, output pointers and record writers built on top.
No record holds a URL or a credential: the ``remote_cpu_job_transport.v1`` object that carries
presigned authority is refused by every host writer, and every record is scanned for
credential-shaped keys and URL-shaped values.  Nothing here contacts a provider.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

from .decision_evidence_contracts import canonical_digest

DESCRIPTOR_SCHEMA_VERSION = "remote_cpu_job_descriptor.v1"
TRANSPORT_SCHEMA_VERSION = "remote_cpu_job_transport.v1"
RECEIPT_SCHEMA_VERSION = "remote_cpu_job_receipt.v1"
HEARTBEAT_SCHEMA_VERSION = "remote_cpu_job_heartbeat.v1"
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
MAX_ATTEMPTS_CAP = 2
PHASE_MARGIN_SECONDS = 180
MAX_TASK_TIMEOUT_SECONDS = 3600
MAX_MEMORY_BYTES = 32 * 1024**3
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


def _credential_shaped_key(text: str) -> bool:
    return any(fragment in text.lower() for fragment in _FORBIDDEN_KEY_FRAGMENTS)


def safe_label(value: Any) -> str:
    """A key fit for a refusal message: a credential- or URL-shaped key becomes ``<key#sha256[:12]>``."""
    text = str(value)
    if _credential_shaped_key(text) or _url_or_credential_text(text):
        return f"<key#{hashlib.sha256(text.encode('utf-8', 'surrogatepass')).hexdigest()[:12]}>"
    return text


def forbidden_record_content(value: Any, path: str = "") -> list[str]:
    """Name every credential-shaped key, URL-shaped value and embedded transport, never the value."""
    reasons: list[str] = []
    if isinstance(value, Mapping):
        if value.get("schema_version") == TRANSPORT_SCHEMA_VERSION:
            reasons.append(f"remote_cpu_transport_never_persisted:{path or '$'}")
        for key, item in value.items():
            where = f"{path}.{safe_label(key)}" if path else safe_label(key)
            if _credential_shaped_key(str(key)):
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
        reasons.extend(f"remote_cpu_field_unexpected:{prefix}{safe_label(key)}"
                       for key in sorted(set(value) - set(spec), key=str))
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


def _config_stage(config: Mapping[str, Any], stage: str, reasons: list[str]) -> Mapping[str, Any]:
    """The config's entry for ``stage``; the config itself must be sealed and in a US region."""
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        reasons.append("remote_cpu_config_invalid:schema_version")
    if config.get("config_digest") != canonical_digest(config, digest_field="config_digest"):
        reasons.append("remote_cpu_config_invalid:config_digest")
    if not str(config.get("region") or "").startswith("us-"):
        reasons.append("remote_cpu_config_region_not_us")
    stages = config.get("stages")
    entry = stages.get(stage) if isinstance(stages, Mapping) else None
    if not isinstance(entry, Mapping):
        reasons.append(f"remote_cpu_config_invalid:stage_missing:{stage}")
        return {}
    return entry


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
            reasons.append(f"remote_cpu_descriptor_environment_variable_not_allowed:{safe_label(variable)}")
        reasons.extend(_path_reasons(path, f"environment.{safe_label(variable)}", roots, directory=True))
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


def _identity_reasons(descriptor: Mapping[str, Any]) -> list[str]:
    """The stage, job id, attempt id, envelope input and staging prefix all derive from the queue row."""
    stage, row = STAGES[descriptor["stage"]], descriptor["queue_row"]
    job_id, attempt_id = descriptor["job_id"], descriptor["attempt_id"]
    reasons = []
    if descriptor["stage_contract"] != stage["stage_contract"] or row["queue"] != stage["queue"]:
        reasons.append("remote_cpu_descriptor_stage_contract_invalid")
    if _row_id(row) is None:
        reasons.append("remote_cpu_descriptor_queue_row_unbound")
    attempt_pattern = rf"{re.escape(job_id)}-a{descriptor['attempt']}-[0-9a-f]{{32}}"
    if job_id != job_id_for(descriptor["stage"], row["name"]) or not re.fullmatch(attempt_pattern, attempt_id):
        reasons.append("remote_cpu_descriptor_attempt_unbound")
    if [item["role"] for item in descriptor["inputs"]].count("queue_envelope") != 1:
        reasons.append("remote_cpu_descriptor_queue_envelope_input_count")
    staging = descriptor["outputs"]["staging_prefix"]
    bucket, _, prefix = staging.removeprefix("s3://").partition("/")
    if (not staging.startswith("s3://") or not _BUCKET.fullmatch(bucket)
            or not staging.endswith(f"/remote-cpu/staging/{job_id}/{attempt_id}/")
            or not all(map(_KEY_PART.fullmatch, prefix.rstrip("/").split("/")))):
        reasons.append("remote_cpu_descriptor_staging_prefix_unbound")
    return reasons


def _limit_reasons(descriptor: Mapping[str, Any], entry: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """Limits match the config's job, fit the phase budget and stay inside hard bounds."""
    limits = descriptor["limits"]
    reasons = [f"remote_cpu_descriptor_limits_do_not_match_job:{name}"
               for name in _JOB_LIMITS if limits[name] != entry.get(name)]
    if descriptor["attempt"] > limits["max_attempts"]:
        reasons.append("remote_cpu_descriptor_attempt_exceeds_max_attempts")
    if limits["max_attempts"] > MAX_ATTEMPTS_CAP or limits["max_attempts"] != config.get("max_attempts"):
        reasons.append("remote_cpu_descriptor_max_attempts_invalid")
    if sum(limits["phase_seconds"].values()) + PHASE_MARGIN_SECONDS > limits["task_timeout_seconds"]:
        reasons.append("remote_cpu_descriptor_phase_budget_exceeds_task_timeout")
    if (limits["task_timeout_seconds"] > MAX_TASK_TIMEOUT_SECONDS or limits["memory_bytes"] > MAX_MEMORY_BYTES
            or limits["heartbeat_interval_seconds"] >= limits["heartbeat_stale_seconds"]):
        reasons.append("remote_cpu_descriptor_limits_out_of_bounds")
    if sum(item["size_bytes"] for item in descriptor["inputs"]) > limits["max_input_bytes"]:
        reasons.append("remote_cpu_descriptor_input_bytes_exceed_limit")
    return reasons


def _execution_reasons(descriptor: Mapping[str, Any], entry: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    """The image and execution are the config's job in a US region, and spend uses its rate table."""
    execution = descriptor["execution"]
    reasons = []
    if descriptor["code"]["image"] != entry.get("image"):
        reasons.append("remote_cpu_image_mismatch")
    if not execution["region"].startswith("us-"):
        reasons.append("remote_cpu_descriptor_region_not_us")
    expected = {"provider": EXECUTION_PROVIDER, "project": config.get("project"), "region": config.get("region"),
                "job": entry.get("job")}
    if execution != expected or not execution["job"].startswith("blueprint-remote-cpu-"):
        reasons.append("remote_cpu_descriptor_execution_does_not_match_config")
    rate_table = config.get("rate_table")
    if not isinstance(rate_table, Mapping) or descriptor["spend"]["rate_table_digest"] != canonical_digest(rate_table):
        reasons.append("remote_cpu_descriptor_rate_table_unbound")
    return reasons


def _closure_reasons(descriptor: Mapping[str, Any], cache_members: Sequence[str]) -> list[str]:
    """Shipped NuRec cache members sit under the cache root; inline NuRec math needs qualified CPUs."""
    closure = descriptor["closure"]
    kind, cache_root = closure["class"], descriptor["environment"].get(CACHE_ROOT_VARIABLE)
    reasons = []
    if (closure["source_appearance_digest"] is None) != (kind == "not_applicable"):
        reasons.append("remote_cpu_descriptor_closure_invalid")
    if kind == "absent_inline_only" and not descriptor["limits"]["allowed_cpu_classes"]:
        reasons.append("remote_cpu_descriptor_closure_requires_qualified_cpu_classes")
    if kind == "shipped" and (not cache_members or not isinstance(cache_root, str)
                              or not all(_within(path, cache_root) for path in cache_members)):
        reasons.append("remote_cpu_descriptor_shipped_closure_unbound")
    if kind != "shipped" and cache_members:
        reasons.append("remote_cpu_descriptor_cache_members_without_shipped_closure")
    return reasons


def validate_descriptor(value: Mapping[str, Any], *, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one sealed descriptor against the host's remote-worker config."""
    descriptor, config = _clone(value, "descriptor"), _clone(config, "config")
    reasons = forbidden_record_content(descriptor)
    _check(descriptor, _DESCRIPTOR_SPEC, "", reasons)
    if descriptor.get("descriptor_digest") != canonical_digest(descriptor, digest_field="descriptor_digest"):
        reasons.append("remote_cpu_descriptor_digest_mismatch")
    _raise_if(reasons)
    entry = _config_stage(config, descriptor["stage"], reasons)
    archive = descriptor["code"]["source_archive"]
    reasons.extend(_cas_uri_reasons(archive["uri"], archive["digest"], "code.source_archive.uri",
                                    kind="remote-cpu-source", filename="source.tar"))
    reasons.extend(_identity_reasons(descriptor))
    reasons.extend(_limit_reasons(descriptor, entry, config))
    reasons.extend(_execution_reasons(descriptor, entry, config))
    reasons.extend(_closure_reasons(descriptor, _descriptor_path_reasons(descriptor, reasons)))
    _raise_if(reasons)
    return descriptor


def build_descriptor(
    *, config: Mapping[str, Any], stage: str, mode: str, attempt: int, queue_row: Mapping[str, Any],
    code: Mapping[str, Any], environment: Mapping[str, str], inputs: Sequence[Mapping[str, Any]],
    outputs: Mapping[str, Any], limits: Mapping[str, Any], closure: Mapping[str, Any], spend: Mapping[str, Any],
    nonce: str | None = None,
) -> dict[str, Any]:
    """Seal and validate a descriptor; the attempt id takes a random nonce unless one is given.

    ``code`` holds ``source_commit``, ``source_archive`` (digest, size_bytes, uri), ``image`` and
    ``environment_digest``; ``outputs`` holds ``output_root``, ``declared_scratch`` and the B2
    ``object_prefix`` under which the attempt's staging prefix is derived.
    """
    if stage not in STAGES or not all(isinstance(item, Mapping) for item in (config, queue_row, code, outputs)):
        raise RemoteCpuContractError("remote_cpu_descriptor_arguments_invalid")
    job_id = job_id_for(stage, str(queue_row.get("name") or ""))
    attempt_id = f"{job_id}-a{attempt}-{secrets.token_hex(16) if nonce is None else nonce}"
    staging_prefix = f"{str(outputs.get('object_prefix')).rstrip('/')}/remote-cpu/staging/{job_id}/{attempt_id}/"
    archive = {"recipe": SOURCE_ARCHIVE_RECIPE, "paths": list(SOURCE_ARCHIVE_PATHS), **(code.get("source_archive") or {})}
    descriptor: dict[str, Any] = {
        "schema_version": DESCRIPTOR_SCHEMA_VERSION, "job_id": job_id, "attempt": attempt, "attempt_id": attempt_id,
        "mode": mode, "stage": stage, "stage_contract": STAGES[stage]["stage_contract"], "queue_row": dict(queue_row),
        "code": {**code, "source_archive": archive}, "environment": dict(environment),
        "inputs": [dict(row) for row in inputs],
        "outputs": {"output_root": outputs.get("output_root"),
                    "declared_scratch": list(outputs.get("declared_scratch") or []),
                    "format": OUTPUT_FORMAT, "staging_prefix": staging_prefix},
        "limits": _clone(limits, "limits"),
        "execution": {"provider": EXECUTION_PROVIDER, "project": config.get("project"), "region": config.get("region"),
                      "job": _config_stage(config, stage, []).get("job")},
        "closure": dict(closure), "spend": dict(spend), "private_url_recorded": False, "descriptor_digest": "",
    }
    descriptor["descriptor_digest"] = canonical_digest(descriptor, digest_field="descriptor_digest")
    return validate_descriptor(descriptor, config=config)


def _fence_reasons(receipt: Mapping[str, Any], descriptor: Mapping[str, Any], execution_name: str) -> list[str]:
    """A receipt counts only when it echoes this descriptor's attempt and the dispatched execution."""
    expected = {field: descriptor.get(field) for field in ("job_id", "attempt", "attempt_id", "stage", "descriptor_digest")}
    expected["execution_name"] = execution_name if _matches(_NAME)(execution_name) else None
    return [f"remote_cpu_receipt_fenced:{field}" for field, value in expected.items()
            if value is None or receipt.get(field) != value]


def _result_reasons(receipt: Mapping[str, Any], descriptor: Mapping[str, Any], stage: Mapping[str, Any]) -> list[str]:
    """A succeeded or blocked receipt carries the stage's sealed result, built from this release."""
    status, result = receipt["status"], receipt["result"]
    if status == "infrastructure_failed" and not result:
        return []
    if not isinstance(result, Mapping) or not result:
        return ["remote_cpu_receipt_result_invalid"]
    reasons = []
    if result.get("result_digest") != canonical_digest(result, digest_field="result_digest"):
        reasons.append("remote_cpu_receipt_result_digest_mismatch")
    if result.get("source_commit") != (descriptor.get("code") or {}).get("source_commit"):
        reasons.append("remote_cpu_receipt_result_source_commit_mismatch")
    if status != "infrastructure_failed":
        blockers = result.get("blockers")
        expected_status = stage["success_status"] if status == "succeeded" else "blocked"
        if (result.get("status") != expected_status or not isinstance(blockers, list)
                or not all(map(_text, blockers)) or bool(blockers) != (status == "blocked")):
            reasons.append("remote_cpu_receipt_result_status_mismatch")
    return reasons


def _output_reasons(receipt: Mapping[str, Any], limits: Mapping[str, Any]) -> list[str]:
    """Only a succeeded receipt carries an output, and that output fits the descriptor's limits."""
    status, output = receipt["status"], receipt["output"]
    if status != "succeeded":
        return [] if output is None else ["remote_cpu_receipt_output_without_success"]
    reasons: list[str] = []
    _check(output, _OUTPUT_SPEC, "output", reasons)
    if reasons:
        return reasons
    known = output["host_known"]
    if known["count"] > output["paths_total"] or known["bytes"] > output["bytes_total"]:
        reasons.append("remote_cpu_receipt_output_host_known_exceeds_totals")
    if (output["paths_total"] > limits.get("max_output_paths", -1)
            or output["bytes_total"] > limits.get("max_output_bytes", -1)):
        reasons.append("remote_cpu_receipt_output_exceeds_limits")
    return reasons


def _infrastructure_failures(receipt: Mapping[str, Any], descriptor: Mapping[str, Any],
                             stage: Mapping[str, Any]) -> list[str]:
    """The worker's named failures plus those the host derives; any one makes the attempt retryable."""
    code, closure = descriptor.get("code") or {}, descriptor.get("closure") or {}
    allowed = (descriptor.get("limits") or {}).get("allowed_cpu_classes", [])
    failures = [*receipt["infrastructure_failures"],
                *(RELEASE_PATH_MISSING + path for path in receipt["release_path_misses"])]
    if receipt["environment"]["environment_digest"] != code.get("environment_digest"):
        failures.append(INFRASTRUCTURE_FAILED + "environment_mismatch")
    if closure.get("class") == "absent_inline_only" and receipt["environment"]["cpu_class"] not in allowed:
        failures.append(INFRASTRUCTURE_FAILED + "cpu_class_unqualified")
    if receipt["status"] == "blocked":
        failures.extend(f"{INFRASTRUCTURE_FAILED}retryable_stage_blocker:{blocker}"
                        for blocker in receipt["result"]["blockers"]
                        if blocker.startswith(stage["retryable_blocker_prefixes"]))
    return sorted(set(failures))


def validate_receipt(value: Mapping[str, Any], *, descriptor: Mapping[str, Any], execution_name: str) -> dict[str, Any]:
    """Fence a receipt to its attempt and execution, then classify it.

    A release-path miss, an environment mismatch, an unqualified CPU class or a retryable stage
    blocker overrides the worker's own status: that attempt is an infrastructure failure.
    """
    receipt = _clone(value, "receipt")
    if not isinstance(descriptor, Mapping):
        raise RemoteCpuContractError("remote_cpu_descriptor_not_mapping")
    reasons = forbidden_record_content(receipt)
    _check(receipt, _RECEIPT_SPEC, "", reasons)
    if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
        reasons.append("remote_cpu_receipt_digest_mismatch")
    reasons.extend(_fence_reasons(receipt, descriptor, execution_name))
    stage = STAGES.get(str(descriptor.get("stage")))
    if reasons or stage is None:
        raise RemoteCpuContractError(reasons or ["remote_cpu_stage_unknown"])
    reasons.extend(_result_reasons(receipt, descriptor, stage))
    reasons.extend(_output_reasons(receipt, descriptor.get("limits") or {}))
    if receipt["status"] == "infrastructure_failed" and not (
            receipt["infrastructure_failures"] or receipt["release_path_misses"]):
        reasons.append("remote_cpu_receipt_infrastructure_failure_unnamed")
    _raise_if(reasons)
    failures = _infrastructure_failures(receipt, descriptor, stage)
    return {"receipt": receipt, "outcome": "infrastructure_failed" if failures else receipt["status"],
            "terminal": not failures, "infrastructure_failures": failures, "receipt_digest": receipt["receipt_digest"]}


def validate_heartbeat(value: Mapping[str, Any], *, attempt_id: str, execution_name: str) -> dict[str, Any]:
    """A heartbeat counts only under the current attempt id and its execution."""
    heartbeat = _clone(value, "heartbeat")
    reasons = forbidden_record_content(heartbeat)
    _check(heartbeat, _HEARTBEAT_SPEC, "", reasons)
    reasons.extend(f"remote_cpu_heartbeat_fenced:{name}" for name, expected in (
        ("attempt_id", attempt_id), ("execution_name", execution_name)) if not expected or heartbeat.get(name) != expected)
    _raise_if(reasons)
    return heartbeat
