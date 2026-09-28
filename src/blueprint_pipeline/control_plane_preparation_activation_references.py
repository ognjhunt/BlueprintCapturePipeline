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
_URI = re.compile(r"(gs|s3|https)://[^\s]+\Z")
_PREPARATION_SUCCESS = {"inputs_materialized_awaiting_construction_adapter",
                        "native_arena_inputs_verified_awaiting_profile_authority",
                        "queued_for_production_episode_compilation", "queued_for_production_scene_configuration"}
_ACTIVATION_PREPARATION_SUCCESS = _PREPARATION_SUCCESS - {"inputs_materialized_awaiting_construction_adapter"}
_LANES = {"task_evaluation_scene_configuration", "native_task_arena_construction",
          "native_task_arena_construction_after_destination", "native_task_arena_destination_qualification",
          "native_task_arena_controls", "native_task_arena_zero_action", "native_task_arena_scripted_positive",
          "native_task_arena_policy_evaluation"}
_PREPARATION_STATES = {"pending", "processing", "awaiting_source_preparation", "awaiting_capacity",
                       "materialized", "completed", "blocked"}
_ACTIVATION_STATES = {"pending", "processing", "prepared", "blocked"}
_ROLES = {"envelope", "identity", "result", "result_conflict"}
_UNFINISHED = ("construction_compilation_launch_profile", "policy_canary", "auxiliary_consumer_joins",
               "runtime_wrapper_revision_recipe", "registry_process_enabled_sources",
               "fencing_grants_offload_restore")
# Finite known structured fields; scalar metadata cannot conceal new containers.
# These classify shapes only, never recursively discover references.
_STRUCTURED = {
    "envelope": "request", "identity": "",
    "request": "scene robot construction controller task sensors runtime execution_adapter publication spend policy_run_setup policy_run_selection policy_run_configuration policy_canary_activation preparation release_window lineage authorization requested_mutations episode_interpretation_authority episode_interpretation_source_rights_admission",
    "scene": "identity source_manifest appearance geometry registration rights website_native_inputs configured_revision",
    "scene.appearance": "representation renderer_qualification", "scene.geometry": "collision validation source_derivation",
    "scene.registration": "metric_registration support_plane robot_mount_interface workspace_clearance camera_calibration",
    "scene.rights": "admission evidence public_display_authorization",
    "scene.rights.evidence.i": "artifact",
    "scene.rights.public_display_authorization": "scene_identity task_identity subject_identity allowed_fields",
    "scene.website_native_inputs": "runtime_inputs appearance observations candidate frames",
    "construction": "recipe output_identity", "robot": "identity configuration kinematics joint_bounds base_registration controller_configuration",
    "controller": "identity configuration model_or_asset_rights",
    "task": "identity destination subject definition success_criteria execution surface_target",
    "task.subject": "identity asset physics_validation rights_admission source_object",
    "task.destination": "identity asset rights_admission static_qualification native_import_qualification geometry placement_qualification native_probe pose_world",
    "task.surface_target": "surface_position_world_m", "task.destination.pose_world": "position_world_m orientation_xyzw",
    "task.destination.native_probe": "placement_support_scene_prim_paths qualification_limits",
    "task.destination.native_probe.qualification_limits": "minimum_camera_pixels",
    "runtime": "identity entrypoint health_protocol requirements network secret_refs mounts",
    "runtime.network": "allowlist", "runtime.requirements": "", "runtime.mounts.i": "source",
    "sensors": "configuration", "execution_adapter": "runtime_source_bundle policy_observation_setup",
    "execution_adapter.policy_observation_setup": "appearance_asset appearance_authoring_receipt wrist_camera_mount_registry policy_master_resolution_wh overview_review_resolution_wh",
    "publication": "", "spend": "provider_allowlist external_service_caps",
    "spend.external_service_caps": "openai anthropic", "spend.external_service_caps.openai": "stage_max_cost_usd",
    "preparation": "", "lineage": "project_spend_reconciliation initial_provider_zero construction_result prior_authority prior_result prior_launch_receipt prior_webapp_sync prior_provider_zero prior_spend_reconciliation destination_qualification_result zero_action_result controls_qualification_manifest",
    "authorization": "scene_owner_attempt", "requested_mutations": "",
    "references": "host_source_readback",
    "result": "references blockers policy_run_plan task_success_contract episode_interpretation_authority episode_interpretation_source_rights_admission construction_output_identity",
}


class PreparationActivationReferenceError(ValueError):
    """Invalid API inputs; fixed text only."""


class _Invalid(Exception):
    def __init__(self, code: str = "supplied_record_invalid"):
        self.code = code


def _resource(code: str) -> bool:
    return code.endswith("limit") or code in {"queue_deadline_exceeded", "queue_clock_invalid", "queue_result_invalid"}


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


def _enum(value: Any, choices: set[str]) -> bool:
    return isinstance(value, str) and value in choices


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
    supported_semantics: bool = True


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
        self.output_bytes = 0
        self.request_references: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
        self.envelopes: dict[tuple[str, str, str, str], list[_Document]] = {}
        self.identities: dict[tuple[str, str, str, str], list[_Document]] = {}
        self.results: dict[tuple[str, str, str, str], list[_Document]] = {}

    def wire_size(self, value: Any) -> int:
        """Count the fixed output's JSON representation without encoding it."""
        self.scan.tick()
        if isinstance(value, str):
            size = 2
            for offset, char in enumerate(value):
                if offset % 1024 == 0:
                    self.scan.tick()
                n = ord(char)
                if char in '\\"\b\f\n\r\t':
                    size += 2
                elif n < 32:
                    size += 6
                elif n < 128:
                    size += 1
                elif n < 2048:
                    size += 2
                elif 0xD800 <= n <= 0xDFFF:
                    raise _Blocked("reference_output_limit")
                elif n < 65536:
                    size += 3
                else:
                    size += 4
                if size > MAX_OUTPUT_BYTES:
                    raise _Blocked("reference_output_limit")
            return size
        if isinstance(value, dict):
            size = 2
            for index, (key, child) in enumerate(value.items()):
                self.scan.tick()
                size += (2 if index else 0) + self.wire_size(key) + 2 + self.wire_size(child)
                if size > MAX_OUTPUT_BYTES:
                    raise _Blocked("reference_output_limit")
            return size
        if isinstance(value, (tuple, list)):
            size = 2
            for index, child in enumerate(value):
                self.scan.tick()
                size += (2 if index else 0) + self.wire_size(child)
                if size > MAX_OUTPUT_BYTES:
                    raise _Blocked("reference_output_limit")
            return size
        if value is None:
            return 4
        if isinstance(value, bool):
            return 4 if value else 5
        if type(value) is int:
            try:
                return len(str(value))
            except ValueError:
                raise _Blocked("reference_output_limit") from None
        raise _Blocked("reference_output_limit")

    def charge(self, value: Any) -> None:
        size = self.wire_size(value) + 2
        if size > MAX_OUTPUT_BYTES - self.output_bytes:
            raise _Blocked("reference_output_limit")
        self.output_bytes += size

    def provenance_document(self, source: RawReferenceProvenance) -> dict[str, Any]:
        return {"family": source.family, "queue_root": source.queue_root, "role": source.role,
                "row_path": source.row_path, "raw_sha256": source.raw_sha256,
                "raw_size_bytes": source.raw_size_bytes, "observed_identity": source.observed_identity}

    def block(self, reason: str) -> None:
        self.scan.block(reason)

    def fact(self, kind: str, source: RawReferenceProvenance, contract_path: str, **kwargs: Any) -> None:
        self.scan.tick()
        related = kwargs.get("related_sources", ())
        count = 1 + len(related)
        if count > MAX_FACTS - self.fact_count:
            raise _Blocked("reference_facts_limit")
        self.fact_count += count
        wire = {"source": self.provenance_document(source), "contract_path": contract_path, "digest": None,
                "path": None, "uri": None, "size_bytes": None, "related_sources": (), **kwargs}
        wire["related_sources"] = tuple(self.provenance_document(item) for item in related)
        self.charge(wire)
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

    def decode(self, row: RetainedReferenceRecord, value: dict[str, Any] | None, parse_error: str | None) -> None:
        self.scan.tick()
        # Reserve a conservative fixed disposition envelope before allocating
        # any provenance/record object; facts are charged separately per copy.
        self.charge({"source": {"family": row.family, "queue_root": row.queue_root, "role": row.role,
                     "row_path": row.row_path, "raw_sha256": "sha256:" + "0" * 64,
                     "raw_size_bytes": len(row.raw_bytes), "observed_identity": row.observed_identity},
                     "disposition": "unsupported", "reason": "x" * 64, "canonical_digest": "sha256:" + "0" * 64})
        digest = hashlib.sha256()
        for offset in range(0, len(row.raw_bytes), 4096):
            self.scan.tick()
            digest.update(row.raw_bytes[offset:offset + 4096])
        source = RawReferenceProvenance(row.family, row.queue_root, row.role, row.row_path,
                                        "sha256:" + digest.hexdigest(), len(row.raw_bytes), row.observed_identity)
        canonical = None
        try:
            _check(parse_error is None and value is not None, "supplied_json_invalid")
            field = {"envelope": "envelope_digest", "identity": "identity_digest"}.get(row.role, "result_digest")
            schema_role = "result" if row.role == "result_conflict" else row.role
            schema = value.get("schema_version")
            if isinstance(schema, str) and schema != "task_evaluation_launch_" + row.family + "_" + schema_role + ".v1":
                self.block("unsupported_schema")
                self.records.append(ReferenceRecordDisposition(source, "unsupported", "unsupported_schema", None))
                return
            _check(schema == "task_evaluation_launch_" + row.family + "_" + schema_role + ".v1")
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
            if _resource(error.code):
                raise
            self.block("supplied_json_invalid")
            self.records.append(ReferenceRecordDisposition(source, "invalid", "supplied_json_invalid", canonical))

    def conflicts(self) -> None:
        paths: dict[tuple[str, str], list[_Document]] = {}
        raw_paths: dict[tuple[str, str], list[RawReferenceProvenance]] = {}
        for row in self.records:
            self.scan.tick()
            raw_paths.setdefault((row.source.queue_root, row.source.row_path), []).append(row.source)
        placements: dict[tuple[str, str, str, str], list[_Document]] = {}
        for doc in self.documents:
            self.scan.tick()
            if (len(raw_paths[(doc.source.queue_root, doc.source.row_path)]) > 1
                    and not (doc.source.family == "preparation" and doc.source.role == "result")):
                doc.contested = True
                self.block("immutable_record_conflict")
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

    def defer(self, doc: _Document, path: str, reason: str) -> None:
        if reason in {"supported_reference_invalid", "supported_request_invalid", "supported_selector_invalid",
                      "unsupported_metadata_shape", "materialized_reference_count_invalid",
                      "materialized_reference_path_ambiguous", "materialized_reference_binding_mismatch",
                      "request_materialized_reference_missing", "preparation_result_binding_invalid",
                      "activation_execution_declaration_invalid", "activation_result_binding_invalid",
                      "canary_companions_invalid"}:
            doc.supported_semantics = False
        self.block(reason)
        self.fact("missing_edge_obligations", doc.source, path, binding_status="unresolved",
                  reason=reason, digest_meaning="no_inferred_raw_identity")

    def shape(self, doc: _Document, value: Any, path: str, fields: str) -> dict[str, Any]:
        self.scan.tick()
        _check(isinstance(value, dict), "supported_reference_invalid")
        allowed = set(fields.split())
        location = re.sub(r"(?<=\.)[0-9]+(?=\.|$)", "i", path)
        structured = set(_STRUCTURED.get(location, "").split())
        for key in value:
            self.scan.tick()
            if key not in allowed:
                self.defer(doc, path, "unsupported_metadata_shape")
                break
            if isinstance(value[key], (dict, list)) and key not in structured:
                self.defer(doc, path + "." + key, "unsupported_metadata_shape")
            if key in {"identity", "output_identity", "scene_identity", "task_identity", "subject_identity"} and key in structured:
                identity = self.shape(doc, value[key], path + "." + key, "id version")
                _check(_identifier(identity.get("id")) and _identifier(identity.get("version")), "supported_request_invalid")
            if key in {"entrypoint", "secret_refs", "allowlist", "allowed_fields", "provider_allowlist",
                       "surface_position_world_m", "position_world_m", "orientation_xyzw",
                       "policy_master_resolution_wh", "overview_review_resolution_wh", "placement_support_scene_prim_paths", "blockers"} and key in structured:
                _check(isinstance(value[key], list), "supported_request_invalid")
                for item in value[key]:
                    self.scan.tick()
                    if isinstance(item, (dict, list)):
                        self.defer(doc, path + "." + key, "unsupported_metadata_shape")
        return value

    def remote(self, doc: _Document, path: str, value: Any) -> dict[str, Any]:
        self.scan.tick()
        _check(isinstance(value, dict) and set(value) == {"uri", "digest", "size_bytes"}, "supported_reference_invalid")
        uri = value["uri"]
        _check(isinstance(uri, str) and len(uri) <= 4096 and len(uri.encode()) <= 4096
               and _URI.fullmatch(uri) is not None and _digest(value["digest"])
               and type(value["size_bytes"]) is int and value["size_bytes"] >= 1, "supported_reference_invalid")
        self.fact("remote_raw_references", doc.source, path, binding_status="declared_remote_raw",
                  reason="supplied_reference_only", digest_meaning="remote_raw_bytes", uri=uri,
                  digest=value["digest"], size_bytes=value["size_bytes"])
        return value

    def leaves(self, doc: _Document, value: dict[str, Any], path: str, required: str,
               optional: str = "") -> dict[str, dict[str, Any]]:
        output = {}
        for key in required.split() + [k for k in optional.split() if k in value]:
            self.scan.tick()
            name = path + "." + key if path else key
            try:
                output[name] = self.remote(doc, name, value.get(key))
            except _Invalid as error:
                self.defer(doc, name, error.code)
        return output

    def selector(self, doc: _Document, path: str, value: Any, *, raw: bool = False,
                 reason: str = "declared_document_selector") -> None:
        _check(_digest(value), "supported_selector_invalid")
        self.fact("raw_digest_selector_obligations" if raw else "canonical_document_selector_obligations",
                  doc.source, path, binding_status="selector_only", reason=reason,
                  digest_meaning="raw_digest_only" if raw else "canonical_document_seal", digest=value)

    def preparation_request(self, doc: _Document) -> dict[str, dict[str, Any]]:
        request = self.shape(doc, doc.value["request"], "request",
            "schema_version scene_intent_digest run_mode expected_production_commit preparation_id team_namespace run_id "
            "scene robot construction controller task sensors runtime execution_adapter publication spend "
            "policy_run_setup policy_run_selection policy_run_configuration policy_canary_activation "
            "replacement_authoring_backend replacement_authoring_model_provider replacement_authoring_agent_runtime replacement_authoring_model")
        _check(_identifier(request.get("run_id")), "supported_request_invalid")
        _check(_enum(request.get("run_mode"), {"scene_configuration", "destination_qualification", "episode_evaluation"}), "supported_request_invalid")
        refs = {}
        for name in ("scene", "task", "robot", "controller", "runtime"):
            container = request.get(name)
            if isinstance(container, dict) and "identity" in container:
                identity = self.shape(doc, container["identity"], name + ".identity", "id version")
                _check(_identifier(identity.get("id")) and _identifier(identity.get("version")), "supported_request_invalid")
        if "publication" in request:
            self.shape(doc, request["publication"], "publication", "input_namespace service_account_readback_required")
        if "spend" in request:
            spend = self.shape(doc, request["spend"], "spend", "maximum_hourly_rate_usd hard_cap_usd hard_ttl_seconds provider_compute_spend_cap_usd external_service_caps retry_cap selected_provider provider_allowlist")
            if "external_service_caps" in spend:
                caps = self.shape(doc, spend["external_service_caps"], "spend.external_service_caps", "openai anthropic")
                for provider in ("openai", "anthropic"):
                    if provider in caps:
                        path = "spend.external_service_caps." + provider
                        cap = self.shape(doc, caps[provider], path, "maximum_cost_usd maximum_requests stage_max_cost_usd" if provider == "openai" else "maximum_cost_usd maximum_requests")
                        if "stage_max_cost_usd" in cap:
                            self.shape(doc, cap["stage_max_cost_usd"], path + ".stage_max_cost_usd", "artifixer_semantic_teacher artifixer_visual_review content_agents")
        for key in ("policy_run_setup", "policy_run_selection", "policy_run_configuration", "policy_canary_activation"):
            if key in request:
                _check(isinstance(request[key], dict), "supported_request_invalid")
                self.defer(doc, key, "deferred_semantic_object")
        if "scene_intent_digest" in request:
            self.selector(doc, "scene_intent_digest", request["scene_intent_digest"])
        scene = self.shape(doc, request.get("scene"), "scene", "mode identity source_manifest appearance geometry registration rights website_native_inputs configured_revision")
        mode = scene.get("mode")
        if mode == "reuse_configured_revision":
            self.shape(doc, scene, "scene", "mode identity configured_revision")
            refs.update(self.leaves(doc, scene, "scene", "configured_revision"))
        elif mode == "configure_source_scene":
            self.shape(doc, scene, "scene", "mode identity source_manifest appearance geometry registration rights website_native_inputs")
            refs.update(self.leaves(doc, scene, "scene", "source_manifest"))
            for key, required, optional, metadata in (
                ("appearance", "representation renderer_qualification", "", "kind"),
                ("geometry", "collision validation", "source_derivation", "kind"),
                ("registration", "metric_registration support_plane robot_mount_interface workspace_clearance camera_calibration", "", ""),
                ("rights", "admission", "", "evidence source_bytes_redistributable provider_disclosure_scope public_display_authorization")):
                container = self.shape(doc, scene.get(key), "scene." + key, required + " " + optional + " " + metadata)
                refs.update(self.leaves(doc, container, "scene." + key, required, optional))
            evidence = scene["rights"].get("evidence")
            _check(isinstance(evidence, list) and 2 <= len(evidence) <= 16, "supported_reference_invalid")
            for index, item in enumerate(evidence):
                path = "scene.rights.evidence." + str(index)
                item = self.shape(doc, item, path, "role artifact")
                _check(_enum(item.get("role"), {"publisher_terms", "publisher_readme", "upstream_license", "human_authority_record"}), "supported_reference_invalid")
                refs.update(self.leaves(doc, item, path, "artifact"))
            if "public_display_authorization" in scene["rights"]:
                authorization = self.shape(doc, scene["rights"]["public_display_authorization"], "scene.rights.public_display_authorization",
                    "schema_version status scope scene_identity task_identity subject_identity rights_admission_digest human_authority_record_digest "
                    "public_slug title summary category allowed_fields thumbnail_publication_authorized derived_metadata_publication_authorized "
                    "private_artifact_uri_publication_authorized raw_media_publication_authorized authority_reference authorized_by authorization_digest")
                for key in ("rights_admission_digest", "human_authority_record_digest", "authorization_digest"):
                    self.selector(doc, "scene.rights.public_display_authorization." + key, authorization.get(key))
            if "website_native_inputs" in scene:
                native = self.shape(doc, scene["website_native_inputs"], "scene.website_native_inputs", "runtime_inputs appearance observations candidate frames")
                refs.update(self.leaves(doc, native, "scene.website_native_inputs", "runtime_inputs appearance observations candidate"))
                frames = native.get("frames")
                _check(isinstance(frames, list) and 1 <= len(frames) <= 256, "supported_reference_invalid")
                for index, item in enumerate(frames):
                    path = "scene.website_native_inputs.frames." + str(index)
                    refs[path] = self.remote(doc, path, item)
        else:
            self.defer(doc, "scene.mode", "unsupported_metadata_shape")
        construction = self.shape(doc, request.get("construction"), "construction", "mode recipe output_identity")
        if construction.get("mode") == "production_recipe":
            refs.update(self.leaves(doc, construction, "construction", "recipe"))
        elif construction.get("mode") == "reuse_configured_scene":
            self.shape(doc, construction, "construction", "mode")
        else:
            self.defer(doc, "construction.mode", "unsupported_metadata_shape")
        for key, required, optional, metadata in (
            ("robot", "configuration kinematics joint_bounds base_registration controller_configuration", "", "identity"),
            ("controller", "configuration", "model_or_asset_rights", "identity kind")):
            if key not in request and request["run_mode"] != "scene_configuration":
                self.defer(doc, key, "supported_reference_invalid")
            if key in request:
                container = self.shape(doc, request[key], key, required + " " + optional + " " + metadata)
                if key == "controller":
                    _check(_enum(container.get("kind"), {"zero_action", "deterministic_scripted", "policy_container"}), "supported_request_invalid")
                    if container["kind"] == "policy_container":
                        required += " model_or_asset_rights"
                        optional = ""
                refs.update(self.leaves(doc, container, key, required, optional))
        task = self.shape(doc, request.get("task"), "task", "identity binding_mode kind strategy configured_scene_revision_digest destination subject definition success_criteria execution surface_target")
        if task.get("binding_mode") == "define_configuration_template":
            refs.update(self.leaves(doc, task, "task", "definition success_criteria execution"))
        elif task.get("binding_mode") == "reuse_configured_template":
            _check(not any(key in task for key in ("definition", "success_criteria", "execution")), "supported_request_invalid")
        else:
            self.defer(doc, "task.binding_mode", "unsupported_metadata_shape")
        subject = self.shape(doc, task.get("subject"), "task.subject", "mode identity representation_kind asset physics_validation rights_admission provider_disclosure_allowed source_object physics_authority")
        subject_mode = subject.get("mode")
        if subject_mode == "supplied_qualified_asset":
            self.shape(doc, subject, "task.subject", "mode identity representation_kind asset physics_validation rights_admission provider_disclosure_allowed")
            refs.update(self.leaves(doc, subject, "task.subject", "asset physics_validation rights_admission"))
        elif subject_mode == "construct_from_scene_object":
            self.shape(doc, subject, "task.subject", "mode identity representation_kind source_object rights_admission provider_disclosure_allowed")
            refs.update(self.leaves(doc, subject, "task.subject", "source_object rights_admission"))
        elif subject_mode == "configured_scene_object":
            self.shape(doc, subject, "task.subject", "mode identity physics_authority")
        else:
            self.defer(doc, "task.subject.mode", "unsupported_metadata_shape")
        if "configured_scene_revision_digest" in task:
            self.selector(doc, "task.configured_scene_revision_digest", task["configured_scene_revision_digest"])
        if "surface_target" in task:
            target = self.shape(doc, task["surface_target"], "task.surface_target", "schema_version shape non_colliding visible_label radius_m surface_position_world_m support_prim_path support_source_instance_id maximum_tilt_rad stable_seconds maximum_linear_speed_m_s maximum_angular_speed_rad_s target_digest")
            self.selector(doc, "task.surface_target.target_digest", target.get("target_digest"))
        if "destination" in task:
            destination = self.shape(doc, task["destination"], "task.destination", "schema_version identity relation visible_label asset rights_admission static_qualification native_import_qualification geometry placement_qualification native_probe pose_world provider_disclosure_allowed")
            needed = "asset rights_admission static_qualification"
            if request["run_mode"] != "scene_configuration":
                needed += " native_import_qualification geometry"
            else:
                _check("native_import_qualification" not in destination and "geometry" not in destination, "supported_request_invalid")
            if request["run_mode"] == "episode_evaluation":
                needed += " placement_qualification"
            else:
                _check("placement_qualification" not in destination, "supported_request_invalid")
                _check(isinstance(destination.get("native_probe"), dict), "supported_request_invalid")
            refs.update(self.leaves(doc, destination, "task.destination", needed))
            if "pose_world" in destination:
                self.shape(doc, destination["pose_world"], "task.destination.pose_world", "position_world_m orientation_xyzw")
            if "native_probe" in destination:
                probe = self.shape(doc, destination["native_probe"], "task.destination.native_probe", "schema_version placement_support_scene_prim_paths qualification_limits settle_sample_count settle_steps_per_sample")
                if "qualification_limits" in probe:
                    limits = self.shape(doc, probe["qualification_limits"], "task.destination.native_probe.qualification_limits", "maximum_penetration_m minimum_support_contact_force_n maximum_forbidden_contact_force_n settle_translation_tolerance_m settle_rotation_tolerance_rad reset_translation_tolerance_m reset_rotation_tolerance_rad minimum_camera_pixels")
                    if "minimum_camera_pixels" in limits:
                        self.shape(doc, limits["minimum_camera_pixels"], "task.destination.native_probe.qualification_limits.minimum_camera_pixels", "external wrist overview")
        sensors = self.shape(doc, request.get("sensors"), "sensors", "configuration")
        refs.update(self.leaves(doc, sensors, "sensors", "configuration"))
        runtime = self.shape(doc, request.get("runtime"), "runtime", "identity oci_image entrypoint health_protocol requirements network secret_refs mounts output_limit_bytes")
        if "requirements" in runtime:
            self.shape(doc, runtime["requirements"], "runtime.requirements", "cpu_cores memory_gib gpu_count disk_gib")
        if "network" in runtime:
            self.shape(doc, runtime["network"], "runtime.network", "default allowlist")
        refs.update(self.leaves(doc, runtime, "runtime", "health_protocol"))
        mounts = runtime.get("mounts")
        _check(isinstance(mounts, list) and len(mounts) <= 128, "supported_reference_invalid")
        for index, item in enumerate(mounts):
            path = "runtime.mounts." + str(index)
            item = self.shape(doc, item, path, "source container_path mode")
            if item.get("mode") == "read_only":
                refs.update(self.leaves(doc, item, path, "source"))
            elif item.get("mode") == "output":
                _check("source" not in item, "supported_request_invalid")
            else:
                self.defer(doc, path + ".mode", "unsupported_metadata_shape")
        adapter = self.shape(doc, request.get("execution_adapter"), "execution_adapter", "kind version runtime_source_bundle runtime_source_implementation_commit policy_observation_setup")
        refs.update(self.leaves(doc, adapter, "execution_adapter", "runtime_source_bundle"))
        if "policy_observation_setup" in adapter:
            setup = self.shape(doc, adapter["policy_observation_setup"], "execution_adapter.policy_observation_setup", "schema_version appearance_asset appearance_authoring_receipt wrist_camera_mount_registry fresh_native_mount_sweep_required policy_master_resolution_wh overview_review_resolution_wh")
            refs.update(self.leaves(doc, setup, "execution_adapter.policy_observation_setup", "appearance_asset appearance_authoring_receipt wrist_camera_mount_registry"))
        return refs

    def activation_request(self, doc: _Document) -> dict[str, dict[str, Any]]:
        request = self.shape(doc, doc.value["request"], "request", "schema_version expected_production_commit activation_id team_namespace run_kind capture_session_id intake_id episode_interpretation_authority episode_interpretation_source_rights_admission lane preparation release_window lineage authorization requested_mutations")
        lane, kind = request.get("lane"), request.get("run_kind", "qualified_evaluation")
        _check(_enum(lane, _LANES) and _enum(kind, {"qualified_evaluation", "internal_policy_canary"}), "supported_request_invalid")
        if kind == "internal_policy_canary":
            _check(lane == "native_task_arena_policy_evaluation" and _identifier(request.get("capture_session_id")) and _identifier(request.get("intake_id")), "supported_request_invalid")
        binding = self.shape(doc, request.get("preparation"), "preparation", "preparation_id request_digest result_digest")
        _check(_identifier(binding.get("preparation_id")) and _digest(binding.get("request_digest")) and _digest(binding.get("result_digest")), "supported_request_invalid")
        self.selector(doc, "preparation.request_digest", binding["request_digest"])
        self.selector(doc, "preparation.result_digest", binding["result_digest"])
        refs = self.leaves(doc, request, "", "release_window")
        lineage = self.shape(doc, request.get("lineage"), "lineage", "kind project_spend_reconciliation initial_provider_zero construction_result prior_authority prior_result prior_launch_receipt prior_webapp_sync prior_provider_zero prior_spend_reconciliation destination_qualification_result zero_action_result controls_qualification_manifest")
        initial = lane in {"task_evaluation_scene_configuration", "native_task_arena_destination_qualification", "native_task_arena_construction"} or kind == "internal_policy_canary"
        _check(lineage.get("kind") == ("initial_project" if initial else "predecessor"), "supported_request_invalid")
        if initial:
            self.shape(doc, lineage, "lineage", "kind project_spend_reconciliation initial_provider_zero construction_result")
            required, optional = "project_spend_reconciliation initial_provider_zero", "construction_result"
            if lane == "native_task_arena_policy_evaluation":
                required += " construction_result"
                optional = ""
        else:
            self.shape(doc, lineage, "lineage", "kind prior_authority prior_result prior_launch_receipt prior_webapp_sync prior_provider_zero prior_spend_reconciliation construction_result destination_qualification_result zero_action_result controls_qualification_manifest")
            required = "prior_authority prior_result prior_launch_receipt prior_webapp_sync prior_provider_zero prior_spend_reconciliation"
            optional = "construction_result destination_qualification_result zero_action_result controls_qualification_manifest"
            _check("construction_result" in lineage or "destination_qualification_result" in lineage, "supported_request_invalid")
            if lane == "native_task_arena_scripted_positive":
                required += " zero_action_result"
            if lane == "native_task_arena_policy_evaluation":
                required += " controls_qualification_manifest"
            optional = " ".join(key for key in optional.split() if key not in required.split())
        refs.update(self.leaves(doc, lineage, "lineage", required, optional))
        authorization = self.shape(doc, request.get("authorization"), "authorization", "reference authorized_by authorized_on standing_authorization_expires_at profile_revision scene_owner_attempt")
        _check(isinstance(authorization.get("reference"), str) and 0 < len(authorization["reference"]) <= 1000, "supported_request_invalid")
        if "scene_owner_attempt" in authorization:
            _check(isinstance(authorization["scene_owner_attempt"], dict), "supported_request_invalid")
            self.defer(doc, "authorization.scene_owner_attempt", "deferred_semantic_object")
        for key in ("episode_interpretation_authority", "episode_interpretation_source_rights_admission"):
            if key in request:
                _check(isinstance(request[key], dict), "supported_request_invalid")
                self.defer(doc, key, "deferred_semantic_object")
        mutations = self.shape(doc, request.get("requested_mutations"), "requested_mutations", "profile_publication catalog_synchronization standing_authorization policy_campaign_queue")
        _check(all(type(value) is bool for value in mutations.values()), "supported_request_invalid")
        return refs

    def index(self) -> None:
        for doc in self.documents:
            self.scan.tick()
            key = (doc.source.family, doc.source.queue_root, doc.identifier, doc.request_digest)
            collection = {"envelope": self.envelopes, "identity": self.identities, "result": self.results}.get(doc.source.role)
            if collection is not None and not doc.contested:
                collection.setdefault(key, []).append(doc)
            if doc.source.role == "envelope":
                fields = "schema_version request_digest request submitted_by submitted_at_iso provider_mutation_performed_inside_intake catalog_mutation_performed_inside_intake envelope_digest"
                if doc.source.family == "activation":
                    fields += " standing_authorization_published_inside_intake paid_execution_requested"
                self.shape(doc, doc.value, "envelope", fields)
                try:
                    self.request_references[_source_key(doc.source)] = (self.preparation_request(doc) if doc.source.family == "preparation" else self.activation_request(doc))
                except _Invalid as error:
                    self.defer(doc, "request", error.code)
                    self.request_references[_source_key(doc.source)] = {}
            elif doc.source.role == "identity":
                self.shape(doc, doc.value, "identity", "schema_version " + doc.source.family + "_id request_digest identity_digest")

    def parents(self, doc: _Document) -> list[_Document]:
        rows = self.envelopes.get((doc.source.family, doc.source.queue_root, doc.identifier, doc.request_digest), [])
        return [row for row in rows if row.supported_semantics]

    def materialized(self, doc: _Document) -> None:
        value = doc.value
        for flag in ("provider_mutation_performed", "paid_execution_requested"):
            if flag in value and value[flag] is not False:
                self.defer(doc, flag, "supported_request_invalid")
        success = value["status"] in _PREPARATION_SUCCESS
        if not success:
            self.defer(doc, "status", "preparation_status_incomplete")
        refs = value.get("references")
        if refs is None:
            self.defer(doc, "references", "preparation_references_missing")
            return
        _check(isinstance(refs, list) and len(refs) <= MAX_FACTS, "supported_reference_invalid")
        if type(value.get("reference_count")) is not int or value["reference_count"] != len(refs):
            self.defer(doc, "reference_count", "materialized_reference_count_invalid")
        parents = self.parents(doc)
        if len(parents) != 1:
            self.defer(doc, "envelope", "preparation_envelope_unresolved")
        matched = set()
        declared: dict[str, set[tuple[str, str, int]]] = {}
        for item in refs:
            self.scan.tick()
            if (isinstance(item, dict) and isinstance(item.get("contract_path"), str)
                    and isinstance(item.get("uri"), str) and isinstance(item.get("digest"), str)
                    and type(item.get("size_bytes")) is int):
                declared.setdefault(item["contract_path"], set()).add((item["uri"], item["digest"], item["size_bytes"]))
        for item in refs:
            self.scan.tick()
            try:
                item = self.shape(doc, item, "references", "contract_path uri digest size_bytes materialized_path content_addressed_reuse full_byte_service_account_readback_passed host_source_readback")
                path = item.get("contract_path")
                _check(isinstance(path, str) and 0 < len(path) <= MAX_CONTRACT_PATH_BYTES and len(path.encode()) <= MAX_CONTRACT_PATH_BYTES
                       and len(path.split(".")) <= 64 and all(re.fullmatch(r"[A-Za-z0-9_]+", part) for part in path.split(".")), "supported_reference_invalid")
                remote = {key: item.get(key) for key in ("uri", "digest", "size_bytes")}
                self.remote(doc, path, remote)
                try:
                    local = _path(item.get("materialized_path"))
                except PreparationActivationReferenceError:
                    raise _Invalid("supported_reference_invalid") from None
                _check(type(item.get("content_addressed_reuse")) is bool and item.get("full_byte_service_account_readback_passed") is True, "supported_reference_invalid")
                if "host_source_readback" in item:
                    host = self.shape(doc, item["host_source_readback"], path + ".host_source_readback", "installation_receipt_digest publisher_intake_sha256 publisher_uri network_fetch_performed raw_source_uploaded")
                    _check(_digest(host.get("installation_receipt_digest")) and _digest(host.get("publisher_intake_sha256"))
                           and isinstance(host.get("publisher_uri"), str) and len(host["publisher_uri"]) <= 4096
                           and _URI.fullmatch(host["publisher_uri"]) is not None
                           and host.get("network_fetch_performed") is False and host.get("raw_source_uploaded") is False, "supported_reference_invalid")
                    self.selector(doc, path + ".host_source_readback.installation_receipt_digest", host["installation_receipt_digest"])
                    self.selector(doc, path + ".host_source_readback.publisher_intake_sha256", host["publisher_intake_sha256"], raw=True)
                binding, reason, related = "receipt_only", "deferred_parent_reference_proof", ()
                ambiguous = len(declared.get(path, ())) > 1
                if ambiguous:
                    self.defer(doc, path, "materialized_reference_path_ambiguous")
                if len(parents) == 1 and not doc.contested and not ambiguous:
                    parent = parents[0]
                    expected = self.request_references.get(_source_key(parent.source), {}).get(path)
                    if expected is not None:
                        if expected == remote:
                            binding, reason, related = "request_bound", "exact_supplied_request_binding", (parent.source,)
                            matched.add(path)
                        else:
                            self.defer(doc, path, "materialized_reference_binding_mismatch")
                    else:
                        self.defer(doc, path, "deferred_parent_reference_proof")
                else:
                    self.defer(doc, path, "deferred_parent_reference_proof")
                self.fact("local_path_protections", doc.source, path, binding_status=binding, reason=reason,
                          digest_meaning="declared_materialized_raw_bytes", path=local, digest=remote["digest"],
                          size_bytes=remote["size_bytes"], related_sources=related)
            except _Invalid as error:
                self.defer(doc, "references", error.code)
        if success and len(parents) == 1:
            request = parents[0].value["request"]
            for key, request_key in (("run_id", "run_id"), ("team_namespace", "team_namespace"), ("source_commit", "expected_production_commit")):
                _check(value.get(key) == request[request_key], "preparation_result_binding_invalid")
            _check(value.get("full_byte_service_account_readback_passed") is True, "supported_reference_invalid")
            for path in self.request_references.get(_source_key(parents[0].source), {}):
                if path not in matched:
                    self.defer(doc, path, "request_materialized_reference_missing")

    def activation_result(self, doc: _Document) -> None:
        value = doc.value
        for flag in ("provider_mutation_performed", "paid_execution_requested"):
            _check(value.get(flag) is False, "activation_execution_declaration_invalid")
        status = value["status"]
        for key in ("preparation_result_digest", "release_window_digest"):
            if key in value:
                self.selector(doc, key, value[key])
        if status == "profile_authority_materialized_no_execution":
            _check(_identifier(value.get("profile_id")), "supported_selector_invalid")
            _check("preparation_result_digest" in value and "release_window_digest" in value, "supported_selector_invalid")
            self.selector(doc, "profile_digest", value.get("profile_digest"))
            for key in ("profile_publication_receipt_digest", "standing_authorization_digest"):
                self.selector(doc, key, value.get(key), raw=True)
        elif status == "policy_campaign_queue_materialized_no_execution":
            self.selector(doc, "policy_campaign_activation_digest", value.get("policy_campaign_activation_digest"))
            self.selector(doc, "policy_campaign_activation_sha256", value.get("policy_campaign_activation_sha256"), raw=True)
            self.defer(doc, "policy_campaign", "deferred_campaign_graph")
            companions = ("policy_canary_runtime_inputs_path", "policy_canary_runtime_inputs_sha256", "policy_canary_runtime_inputs_digest")
            if any(key in value for key in companions):
                _check(all(key in value for key in companions), "canary_companions_invalid")
                try:
                    path = _path(value[companions[0]])
                except PreparationActivationReferenceError:
                    raise _Invalid("canary_companions_invalid") from None
                _check(_digest(value[companions[1]]) and _digest(value[companions[2]]), "canary_companions_invalid")
                self.fact("local_path_protections", doc.source, companions[0], binding_status="receipt_only",
                          reason="canary_raw_size_missing", digest_meaning="raw_digest_only", path=path, digest=value[companions[1]])
                self.selector(doc, companions[1], value[companions[1]], raw=True, reason="canary_raw_size_missing")
                self.selector(doc, companions[2], value[companions[2]])
                self.defer(doc, companions[0], "canary_raw_size_missing")
        else:
            self.defer(doc, "status", "activation_status_incomplete")
        parents = self.parents(doc)
        if len(parents) != 1:
            self.defer(doc, "envelope", "activation_envelope_unresolved")
        elif status in {"profile_authority_materialized_no_execution", "policy_campaign_queue_materialized_no_execution"}:
            request = parents[0].value["request"]
            _check(all(value.get(key) == request[request_key] for key, request_key in (
                ("team_namespace", "team_namespace"), ("lane", "lane"), ("source_commit", "expected_production_commit"))), "activation_result_binding_invalid")
            _check(value.get("preparation_id") == request["preparation"]["preparation_id"]
                   and value.get("preparation_result_digest") == request["preparation"]["result_digest"], "activation_result_binding_invalid")
            _check(value.get("full_byte_activation_reference_readback_passed") is True, "supported_reference_invalid")

    def edges(self) -> None:
        prep_targets: dict[tuple[str, str, str], list[_Document]] = {}
        for doc in self.documents:
            self.scan.tick()
            if (doc.source.family == "preparation" and doc.source.role == "result"
                    and not doc.contested and doc.supported_semantics):
                prep_targets.setdefault((doc.identifier, doc.request_digest, doc.canonical_digest), []).append(doc)
            if doc.source.role in {"envelope", "identity"}:
                key = (doc.source.family, doc.source.queue_root, doc.identifier, doc.request_digest)
                opposite = self.identities if doc.source.role == "envelope" else self.envelopes
                if len(opposite.get(key, [])) != 1:
                    self.defer(doc, "identity" if doc.source.role == "envelope" else "envelope", "identity_envelope_unresolved")
            if doc.source.role == "envelope":
                versions = self.results.get((doc.source.family, doc.source.queue_root, doc.identifier, doc.request_digest), [])
                if not versions:
                    self.defer(doc, "result", "result_selector_missing")
                elif len({version.canonical_digest for version in versions}) > 1:
                    self.defer(doc, "result", "unqualified_result_selector_ambiguous")
        for doc in self.documents:
            self.scan.tick()
            if doc.source.family != "activation" or doc.source.role != "envelope":
                continue
            if not doc.supported_semantics:
                self.defer(doc, "preparation", "activation_preparation_unresolved")
                continue
            binding = doc.value["request"].get("preparation")
            if not isinstance(binding, dict):
                continue
            targets = prep_targets.get((binding.get("preparation_id"), binding.get("request_digest"), binding.get("result_digest")), [])
            roots = {row.source.queue_root for row in targets}
            resolved = bool(targets) and len(roots) == 1
            for target in targets:
                self.scan.tick()
                parents = self.parents(target)
                request = doc.value["request"]
                resolved = resolved and len(parents) == 1 and parents[0].state == "materialized" and target.value.get("status") in _ACTIVATION_PREPARATION_SUCCESS and target.value.get("full_byte_service_account_readback_passed") is True
                if len(parents) == 1:
                    other = parents[0].value["request"]
                    resolved = resolved and all(other.get(key) == request.get(key) for key in ("team_namespace", "expected_production_commit"))
            if not resolved or doc.contested:
                self.defer(doc, "preparation", "activation_preparation_unresolved")

    def interpret(self) -> None:
        self.index()
        for doc in self.documents:
            self.scan.tick()
            if doc.source.role not in {"result", "result_conflict"}:
                continue
            try:
                fields = ("schema_version status preparation_id activation_id source_commit run_id team_namespace result_digest "
                    "references reference_count unique_object_count content_addressed_reuse_count service_account service_account_uid "
                    "full_byte_service_account_readback_passed provider_mutation_performed "
                    "catalog_mutation_performed paid_execution_requested blockers observed_at_iso run_mode "
                    "existing_result_digest candidate_result_digest adapter_result_digest construction_recipe_digest "
                    "configured_scene_revision_digest configured_scene_bundle_digest episode_compilation_queue_envelope_digest "
                    "construction_queue_envelope_digest construction_queue_receipt_digest episode_compilation_queue_receipt_digest "
                    "configuration_render_inputs_result_digest scene_intent_digest website_request_digest task_success_contract_digest "
                    "policy_run_plan task_success_contract episode_interpretation_authority episode_interpretation_source_rights_admission "
                    "construction_output_identity construction_stage_configuration_count construction_stage_configurations_readback_passed "
                    "construction_packet_materialized configuration_render_input_count raw_interiorgs_bytes_in_provider_packet "
                    "construction_orchestration_id automatic_progression_required runtime_source_bundle_readback_passed "
                    "episode_compilation_id customer_supplied_prebuilt_episode_packet")
                if doc.source.family == "activation":
                    fields += (" lane preparation_result_digest release_window_digest profile_id profile_digest "
                        "profile_publication_receipt_digest standing_authorization_digest full_byte_activation_reference_readback_passed "
                        "profile_publication_performed standing_authorization_published policy_campaign_activation_digest "
                        "policy_campaign_activation_sha256 campaign_unit_count run_kind claim_ceiling policy_canary_runtime_inputs_path "
                        "policy_canary_runtime_inputs_sha256 policy_canary_runtime_inputs_digest capture_session_id intake_id request_digest")
                self.shape(doc, doc.value, "result", fields)
                for key in ("unique_object_count", "content_addressed_reuse_count", "service_account_uid"):
                    if key in doc.value:
                        _check(type(doc.value[key]) is int and doc.value[key] >= 0, "supported_request_invalid")
                if "service_account" in doc.value:
                    _check(isinstance(doc.value["service_account"], str) and 0 < len(doc.value["service_account"]) <= 4096, "supported_request_invalid")
                if doc.source.family == "preparation":
                    self.materialized(doc)
                else:
                    self.activation_result(doc)
                for key in ("existing_result_digest", "candidate_result_digest", "adapter_result_digest", "construction_recipe_digest",
                            "configured_scene_revision_digest", "configured_scene_bundle_digest", "episode_compilation_queue_envelope_digest",
                            "construction_queue_envelope_digest", "construction_queue_receipt_digest", "episode_compilation_queue_receipt_digest",
                            "configuration_render_inputs_result_digest", "scene_intent_digest", "website_request_digest", "task_success_contract_digest"):
                    if key in doc.value:
                        self.selector(doc, key, doc.value[key])
                        self.defer(doc, key, "deferred_downstream_document")
                for key in ("policy_run_plan", "task_success_contract", "episode_interpretation_authority", "episode_interpretation_source_rights_admission"):
                    if key in doc.value:
                        self.defer(doc, key, "deferred_semantic_object")
            except _Invalid as error:
                self.defer(doc, "result", error.code)
        self.edges()
        for kind, key_name in (("local_path_protections", "path"), ("remote_raw_references", "uri")):
            identities: dict[str, set[tuple[str | None, int | None]]] = {}
            for fact in self.facts[kind]:
                self.scan.tick()
                identities.setdefault(getattr(fact, key_name), set()).add((fact.digest, fact.size_bytes))
            for name, values in identities.items():
                self.scan.tick()
                if len(values) > 1:
                    self.block("local_path_identity_ambiguous" if kind == "local_path_protections" else "remote_uri_identity_ambiguous")

    def result(self) -> PreparationActivationReferenceInterpretation:
        self.scan.tick()
        def record_key(row: ReferenceRecordDisposition) -> tuple[str, ...]:
            self.scan.tick()
            return _source_key(row.source)
        def fact_key(row: ReferenceFact) -> tuple[Any, ...]:
            self.scan.tick()
            return _source_key(row.source), row.contract_path, row.path or "", row.uri or "", row.digest or "", row.reason
        records = tuple(sorted(self.records, key=record_key))
        facts = {}
        for kind, rows in self.facts.items():
            self.scan.tick()
            facts[kind] = tuple(sorted(rows, key=fact_key))
        self.scan.tick()
        result = PreparationActivationReferenceInterpretation(not self.scan.blockers, records, **facts,
                                                              blockers=tuple(sorted(self.scan.blockers)))
        self.scan.tick()
        document = asdict(result)
        self.scan.tick()
        _check(self.wire_size(document) <= MAX_OUTPUT_BYTES, "reference_output_limit")
        self.scan.output_size(document)
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
        if (not isinstance(contract, ReferenceFamilyContract) or not isinstance(contract.family, str)
                or contract.family not in {"preparation", "activation"}):
            raise PreparationActivationReferenceError("reference_parameters_invalid")
        root = _path(contract.queue_root)
        if any(root == prior or root.startswith(prior.rstrip("/") + "/") or prior.startswith(root.rstrip("/") + "/")
               for _, prior in roots):
            raise PreparationActivationReferenceError("reference_parameters_invalid")
        roots.add((contract.family, root))
    engine = _Interpretation(monotonic, float(time_budget_seconds))
    try:
        engine.scan.tick()
        # Fixed result overhead is charged once, before retained collections.
        engine.charge(PreparationActivationReferenceInterpretation(False, (), (), (), (), (), (), tuple("x" * 64 for _ in range(33))).__dict__)
        if len(records) > MAX_RECORDS:
            raise _Blocked("reference_records_limit")
        total = 0
        for row in records:
            engine.scan.tick()
            if (not isinstance(row, RetainedReferenceRecord)
                    or any(not isinstance(value, str) for value in (row.family, row.queue_root, row.role, row.row_path))
                    or (row.family, row.queue_root) not in roots
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
        # Prove lexical aggregate limits for every row before any parser/hash.
        prepared = []
        for row in records:
            engine.scan.tick()
            try:
                text = row.raw_bytes.decode("utf-8")
                engine.scan.preflight(text)
                prepared.append((row, text, None))
            except UnicodeError:
                prepared.append((row, None, "supplied_json_invalid"))
            except _Blocked as error:
                if _resource(error.code):
                    raise
                prepared.append((row, None, "supplied_json_invalid"))
        # Reuse the strict parser's decoded scalar proof globally as a second
        # pass. No canonical or raw identity hash runs until all rows finish it.
        engine.scan.values = 0
        decoded = []
        for row, text, error in prepared:
            engine.scan.tick()
            if error is not None:
                decoded.append((row, None, error))
                continue
            try:
                checked = engine.scan.parse(row.raw_bytes)
                engine.scan.tick()
                value = json.loads(checked)
                engine.scan.tick()
                decoded.append((row, value, None))
            except _Blocked as blocked:
                if _resource(blocked.code):
                    raise
                decoded.append((row, None, "supplied_json_invalid"))
        seen = set()
        for row, value, error in decoded:
            engine.decode(row, value, error)
            source = engine.records[-1].source
            key = _source_key(source) + (source.raw_size_bytes,)
            if key in seen:
                raise PreparationActivationReferenceError("reference_duplicate_input")
            seen.add(key)
        engine.conflicts()
        engine.interpret()
        return engine.result()
    except _Blocked as error:
        engine.block(error.code)
        return PreparationActivationReferenceInterpretation(False, (), (), (), (), (), (), tuple(sorted(engine.scan.blockers)))
