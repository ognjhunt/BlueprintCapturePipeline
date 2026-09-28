"""Pure retained preparation/activation reference evidence for ADP-009D.

Canonical seals establish supplied document integrity only. This unused leaf
neither observes queues/payloads nor grants admission, reference clearance or GC.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .control_plane_queue_observation import _Blocked, _finite, _Scan

MAX_RECORDS = 10_000
MAX_RECORD_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 20 * 1024 * 1024
MAX_FACTS = 20_000
MAX_ROOTS = 16
MAX_PATH_BYTES, MAX_PATH_COMPONENTS = 4096, 64
MAX_CONTRACT_PATH_BYTES = 1024
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,191}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PREPARATION_STATES = {"pending", "processing", "awaiting_source_preparation", "awaiting_capacity",
                       "materialized", "completed", "blocked"}
_ACTIVATION_STATES = {"pending", "processing", "prepared", "blocked"}
_ROLES = {"envelope", "identity", "result", "result_conflict"}
_UNFINISHED = ("construction_compilation_launch_profile", "policy_canary", "auxiliary_consumer_joins",
               "runtime_wrapper_revision_recipe", "registry_process_enabled_sources",
               "fencing_grants_offload_restore")


class PreparationActivationReferenceError(ValueError):
    """Invalid API inputs; fixed text only."""


class _Invalid(Exception):
    def __init__(self, code: str = "supplied_record_invalid"):
        self.code = code


@dataclass(frozen=True)
class ReferenceFamilyContract:
    family: Literal["preparation", "activation"]
    queue_root: str


@dataclass(frozen=True)
class RetainedReferenceRecord:
    family: Literal["preparation", "activation"]
    queue_root: str
    role: Literal["envelope", "identity", "result", "result_conflict"]
    row_path: str
    raw_bytes: bytes
    observed_identity: tuple[int, int, int, int, int] | None = None


@dataclass(frozen=True)
class RawReferenceProvenance:
    family: str
    queue_root: str
    role: str
    row_path: str
    raw_sha256: str
    raw_size_bytes: int
    observed_identity: tuple[int, ...] | None


@dataclass(frozen=True)
class ReferenceRecordDisposition:
    source: RawReferenceProvenance
    disposition: str
    reason: str
    canonical_digest: str | None


@dataclass(frozen=True)
class ReferenceFact:
    source: RawReferenceProvenance
    contract_path: str
    binding_status: str
    reason: str
    digest_meaning: str
    digest: str | None = None
    path: str | None = None
    uri: str | None = None
    size_bytes: int | None = None
    related_sources: tuple[RawReferenceProvenance, ...] = ()


@dataclass(frozen=True)
class PreparationActivationReferenceInterpretation:
    complete_supplied_supported_projection: bool
    records: tuple[ReferenceRecordDisposition, ...]
    local_path_protections: tuple[ReferenceFact, ...]
    remote_raw_references: tuple[ReferenceFact, ...]
    raw_digest_selector_obligations: tuple[ReferenceFact, ...]
    canonical_document_selector_obligations: tuple[ReferenceFact, ...]
    missing_edge_obligations: tuple[ReferenceFact, ...]
    blockers: tuple[str, ...]
    scope: str = "preparation_activation_supplied_reference_contracts_only"
    unfinished_scopes: tuple[str, ...] = _UNFINISHED
    full_request_schema_verified: bool = False
    request_admission_verified: bool = False
    producer_authorization_verified: bool = False
    payload_bytes_verified: bool = False
    general_reference_inventory_complete: bool = False
    consumer_fence_checked: bool = False
    references_clear: bool = False
    execution_authorized: bool = False
    mutations: int = 0


def _check(condition: bool, code: str = "supplied_record_invalid") -> None:
    if not condition:
        raise _Invalid(code)


def _path(value: Any) -> str:
    if not isinstance(value, str) or len(value) > MAX_PATH_BYTES:
        raise PreparationActivationReferenceError("reference_parameters_invalid")
    try:
        valid = (len(value.encode("utf-8")) <= MAX_PATH_BYTES and value.startswith("/")
                 and not value.startswith("//") and not any(ord(c) < 32 or ord(c) == 127
                                                           or c in "\\<>*?[]" for c in value))
    except UnicodeError:
        valid = False
    if not valid:
        raise PreparationActivationReferenceError("reference_parameters_invalid")
    parts = value[1:].split("/") if value != "/" else []
    if len(parts) > MAX_PATH_COMPONENTS or any(p in {"", ".", ".."} or len(p.encode()) > 255 for p in parts):
        raise PreparationActivationReferenceError("reference_parameters_invalid")
    return value


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _filename(family: str, identifier: str, digest: str) -> str:
    name = identifier + "-" + digest[7:] + ".json"
    if family == "activation" and len(name.encode()) > 255:
        name = "activation-" + hashlib.sha256(identifier.encode()).hexdigest() + "-" + digest[7:] + ".json"
    _check(len(name.encode()) <= 255)
    return name


def _source_key(source: RawReferenceProvenance) -> tuple[str, ...]:
    return source.queue_root, source.family, source.role, source.row_path, source.raw_sha256


@dataclass
class _Document:
    source: RawReferenceProvenance
    value: dict[str, Any]
    identifier: str
    request_digest: str
    canonical_digest: str
    state: str
    contested: bool = False


class _Interpretation:
    def __init__(self, clock: Callable[[], float], budget: float):
        # Only unchanged pure methods are used; no descriptor/queue methods.
        self.scan = _Scan((), 0.0, clock, budget)
        self.records: list[ReferenceRecordDisposition] = []
        self.documents: list[_Document] = []
        self.facts: dict[str, list[ReferenceFact]] = {k: [] for k in (
            "local_path_protections", "remote_raw_references", "raw_digest_selector_obligations",
            "canonical_document_selector_obligations", "missing_edge_obligations")}
        self.fact_count = 0

    def block(self, reason: str) -> None:
        self.scan.block(reason)

    def fact(self, kind: str, source: RawReferenceProvenance, contract_path: str, **kwargs: Any) -> None:
        self.scan.tick()
        if self.fact_count >= MAX_FACTS:
            raise _Blocked("reference_facts_limit")
        self.fact_count += 1
        self.facts[kind].append(ReferenceFact(source, contract_path, **kwargs))

    def canonical(self, value: dict[str, Any], field: str | None = None) -> str:
        self.scan.tick()
        document = {k: v for k, v in value.items() if k != field}
        # Default-spaced size is an upper bound for the compact canonical form.
        self.scan.output_size(document)
        self.scan.tick()
        output = hashlib.sha256()
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False).iterencode(document):
            self.scan.tick()
            for offset in range(0, len(chunk), 1024):
                self.scan.tick()
                output.update(chunk[offset:offset + 1024].encode("utf-8"))
        self.scan.tick()
        return "sha256:" + output.hexdigest()

    def decode(self, row: RetainedReferenceRecord) -> None:
        self.scan.tick()
        digest = hashlib.sha256()
        for offset in range(0, len(row.raw_bytes), 4096):
            self.scan.tick()
            digest.update(row.raw_bytes[offset:offset + 4096])
        source = RawReferenceProvenance(row.family, row.queue_root, row.role, row.row_path,
                                        "sha256:" + digest.hexdigest(), len(row.raw_bytes), row.observed_identity)
        canonical = None
        try:
            text = self.scan.parse(row.raw_bytes)
            self.scan.tick()
            value = json.loads(text)
            self.scan.tick()
            field = {"envelope": "envelope_digest", "identity": "identity_digest"}.get(row.role, "result_digest")
            schema_role = "result" if row.role == "result_conflict" else row.role
            _check(value.get("schema_version") == "task_evaluation_launch_" + row.family + "_" + schema_role + ".v1")
            _check(_digest(value.get(field)))
            canonical = self.canonical(value, field)
            _check(value[field] == canonical)
            request = value.get("request") if row.role == "envelope" else value
            _check(isinstance(request, dict))
            identifier = request.get(row.family + "_id")
            _check(_identifier(identifier))
            relative = row.row_path[len(row.queue_root.rstrip("/")) + 1:].split("/")
            state = relative[0]
            if row.role == "identity":
                _check(relative == ["identities", identifier + ".json"])
                request_digest = value.get("request_digest")
            elif row.role == "envelope":
                _check(state in (_PREPARATION_STATES if row.family == "preparation" else _ACTIVATION_STATES))
                _check(request.get("schema_version") == "task_evaluation_launch_" + row.family + "_request.v1")
                _check(_identifier(request.get("team_namespace")) and isinstance(request.get("expected_production_commit"), str)
                       and _COMMIT.fullmatch(request["expected_production_commit"]) is not None)
                request_digest = self.canonical(request)
                _check(value.get("request_digest") == request_digest)
                flags = ["provider_mutation_performed_inside_intake", "catalog_mutation_performed_inside_intake"]
                if row.family == "activation":
                    flags += ["standing_authorization_published_inside_intake", "paid_execution_requested"]
                _check(all(value.get(flag) is False for flag in flags))
                _check(all(isinstance(value.get(key), str) and 0 < len(value[key]) <= 4096
                           for key in ("submitted_by", "submitted_at_iso")))
                _check(relative == [state, _filename(row.family, identifier, request_digest)])
            else:
                _check(relative[0] == "results" and isinstance(value.get("status"), str))
                filename = relative[-1]
                suffix = "-" + canonical[7:] + ".json" if row.role == "result_conflict" else ".json"
                _check(filename.endswith(suffix))
                prefix = filename[:-len(suffix)]
                _check(len(prefix) >= 65 and prefix[-65] == "-")
                request_digest = "sha256:" + prefix[-64:]
                expected = _filename(row.family, identifier, request_digest)
                if row.role == "result_conflict":
                    expected = expected[:-5] + suffix
                _check(relative == (["results", "conflicts", expected] if row.role == "result_conflict" else ["results", expected]))
            _check(_digest(request_digest))
            self.documents.append(_Document(source, value, identifier, request_digest, canonical, state))
            self.records.append(ReferenceRecordDisposition(source, "supported", "supplied_integrity_only", canonical))
        except _Invalid as error:
            self.block(error.code)
            self.records.append(ReferenceRecordDisposition(source, "invalid", error.code, canonical))
        except _Blocked as error:
            if error.code.endswith("limit") or error.code in {"queue_deadline_exceeded", "queue_clock_invalid", "queue_result_invalid"}:
                raise
            self.block("supplied_json_invalid")
            self.records.append(ReferenceRecordDisposition(source, "invalid", "supplied_json_invalid", canonical))

    def conflicts(self) -> None:
        paths: dict[tuple[str, str], list[_Document]] = {}
        placements: dict[tuple[str, str, str, str], list[_Document]] = {}
        for doc in self.documents:
            self.scan.tick()
            paths.setdefault((doc.source.queue_root, doc.source.row_path), []).append(doc)
            if doc.source.role in {"envelope", "identity"}:
                placements.setdefault((doc.source.family, doc.source.queue_root, doc.source.role, doc.identifier), []).append(doc)
        for group in list(paths.values()) + list(placements.values()):
            self.scan.tick()
            if len(group) <= 1 or (group[0].source.family == "preparation" and group[0].source.role == "result"):
                continue
            self.block("immutable_record_conflict")
            for doc in group:
                self.scan.tick()
                doc.contested = True
        contested = {_source_key(doc.source) for doc in self.documents if doc.contested}
        for index, row in enumerate(self.records):
            self.scan.tick()
            if _source_key(row.source) in contested:
                self.records[index] = ReferenceRecordDisposition(row.source, "conflict", "immutable_record_conflict", row.canonical_digest)

    def result(self) -> PreparationActivationReferenceInterpretation:
        self.scan.tick()
        records = tuple(sorted(self.records, key=lambda row: _source_key(row.source)))
        facts = {}
        for kind, rows in self.facts.items():
            self.scan.tick()
            facts[kind] = tuple(sorted(rows, key=lambda row: (_source_key(row.source), row.contract_path,
                                                             row.path or "", row.uri or "", row.digest or "", row.reason)))
        self.scan.tick()
        result = PreparationActivationReferenceInterpretation(not self.scan.blockers, records, **facts,
                                                              blockers=tuple(sorted(self.scan.blockers)))
        self.scan.tick()
        self.scan.output_size(asdict(result))
        self.scan.tick()
        return result


def interpret_preparation_activation_references(
    contracts: Sequence[ReferenceFamilyContract], records: Sequence[RetainedReferenceRecord], *,
    monotonic: Callable[[], float] = time.monotonic, time_budget_seconds: float = 5.0,
) -> PreparationActivationReferenceInterpretation:
    """Interpret finite supplied bytes; never load payloads or confer action authority."""
    if (not isinstance(contracts, (tuple, list)) or not 1 <= len(contracts) <= MAX_ROOTS
            or not isinstance(records, (tuple, list)) or not callable(monotonic)
            or not _finite(time_budget_seconds) or not 0 < time_budget_seconds <= 5):
        raise PreparationActivationReferenceError("reference_parameters_invalid")
    roots: set[tuple[str, str]] = set()
    for contract in contracts:
        if not isinstance(contract, ReferenceFamilyContract) or contract.family not in {"preparation", "activation"}:
            raise PreparationActivationReferenceError("reference_parameters_invalid")
        root = _path(contract.queue_root)
        if any(root == prior or root.startswith(prior.rstrip("/") + "/") or prior.startswith(root.rstrip("/") + "/")
               for _, prior in roots):
            raise PreparationActivationReferenceError("reference_parameters_invalid")
        roots.add((contract.family, root))
    engine = _Interpretation(monotonic, float(time_budget_seconds))
    try:
        engine.scan.tick()
        if len(records) > MAX_RECORDS:
            raise _Blocked("reference_records_limit")
        total = 0
        for row in records:
            engine.scan.tick()
            if (not isinstance(row, RetainedReferenceRecord) or (row.family, row.queue_root) not in roots
                    or row.role not in _ROLES or type(row.raw_bytes) is not bytes or not row.raw_bytes):
                raise PreparationActivationReferenceError("reference_parameters_invalid")
            path = _path(row.row_path)
            if not path.startswith(row.queue_root.rstrip("/") + "/"):
                raise PreparationActivationReferenceError("reference_parameters_invalid")
            identity = row.observed_identity
            if identity is not None and (not isinstance(identity, tuple) or len(identity) != 5
                    or any(type(n) is not int for n in identity) or any(n < 0 for n in identity[:3])
                    or identity[2] != len(row.raw_bytes)):
                raise PreparationActivationReferenceError("reference_parameters_invalid")
            total += len(row.raw_bytes)
            if len(row.raw_bytes) > MAX_RECORD_BYTES or total > MAX_TOTAL_BYTES:
                raise _Blocked("reference_bytes_limit")
        seen = set()
        for row in records:
            engine.decode(row)
            source = engine.records[-1].source
            key = _source_key(source) + (source.raw_size_bytes,)
            if key in seen:
                raise PreparationActivationReferenceError("reference_duplicate_input")
            seen.add(key)
        engine.conflicts()
        return engine.result()
    except _Blocked as error:
        engine.block(error.code)
        return PreparationActivationReferenceInterpretation(False, (), (), (), (), (), (), tuple(sorted(engine.scan.blockers)))
