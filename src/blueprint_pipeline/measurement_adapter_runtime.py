"""Execution-boundary contracts for measurement-method candidates.

Research dossiers say what should be investigated.  They are not executable
method profiles and must never be converted directly into router booleans.  This
module supplies the missing boundary:

* versioned adapter descriptors for the report's priority engines and tools;
* side-effect-free local availability/version probes (no package imports);
* tri-state capability drafts where unknown is distinct from unsupported; and
* an admission packet that binds a future adapter implementation to the exact
  R0 dossier, probe, capability observations, benchmark protocols, and source
  digest without granting route or execution authority.

The output is deliberately not ``method_capability_profile.v1``.  Promotion to
that production contract remains an R6/R7 human decision backed by independent
held-out benchmark evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
from dataclasses import dataclass
from typing import Any, Mapping

from .measurement_method_research_catalog import (
    research_intake_catalog,
    validate_research_method_candidate,
)
from .task_site_measurement_routing import ALL_CAPABILITY_FIELDS


ADAPTER_DESCRIPTOR_SCHEMA_VERSION = "measurement_adapter_descriptor.v1"
ADAPTER_PROBE_SCHEMA_VERSION = "measurement_adapter_probe.v1"
CAPABILITY_DRAFT_SCHEMA_VERSION = "measurement_capability_draft.v1"
ADMISSION_PACKET_SCHEMA_VERSION = "measurement_adapter_admission_packet.v1"

CAPABILITY_STATES = frozenset({"supported", "unsupported", "unknown"})
PROBE_STATUSES = frozenset(
    {"available", "partial", "unavailable", "access_required", "manual_review"}
)


class MeasurementAdapterError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


@dataclass(frozen=True)
class AdapterRecipe:
    python_distributions: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()
    benchmark_ids: tuple[str, ...] = ()
    execution_mode: str = "local_library"
    access_required: bool = False
    target_version: str | None = None


# Probes are intentionally conservative.  A package or command name appearing
# on a host is availability evidence only; it says nothing about solver,
# backend, scene, robot, sensor, material, or benchmark qualification.
ADAPTER_RECIPES: dict[str, AdapterRecipe] = {
    "exact-geometry-stack": AdapterRecipe(
        python_distributions=("pin", "coal"),
        executables=("move_group",),
        benchmark_ids=("capture-to-geometry-and-contact",),
        target_version="pin-4.1.0+coal-3.0.3",
    ),
    "mujoco-3": AdapterRecipe(
        python_distributions=("mujoco",),
        benchmark_ids=("capture-to-geometry-and-contact", "capture-to-deformation"),
    ),
    "isaac-sim-6-physx": AdapterRecipe(
        python_distributions=("isaacsim",),
        executables=("isaac-sim.sh", "isaac-sim"),
        benchmark_ids=("capture-to-geometry-and-contact", "capture-to-deformation"),
        target_version="6.0.1",
    ),
    "isaac-rtx-openusd-sensor-path": AdapterRecipe(
        python_distributions=("isaacsim",),
        executables=("isaac-sim.sh", "isaac-sim"),
        benchmark_ids=("capture-to-observation",),
    ),
    "newton-1-4": AdapterRecipe(
        python_distributions=("newton", "warp-lang"),
        benchmark_ids=("capture-to-geometry-and-contact", "capture-to-deformation"),
    ),
    "drake-1-55": AdapterRecipe(
        python_distributions=("drake",),
        # Drake removed drake-visualizer in v1.27. Requiring that retired
        # executable made every current pip installation probe as partial.
        benchmark_ids=("capture-to-geometry-and-contact",),
        target_version="1.55.0",
    ),
    "sapien-maniskill-3": AdapterRecipe(
        # The first executable port is deliberately SAPIEN-physics-only.
        # ManiSkill task/runtime integration remains a separate unproven layer.
        python_distributions=("sapien",),
        benchmark_ids=("capture-to-geometry-and-contact",),
    ),
    "project-chrono-10": AdapterRecipe(
        # Official PyChrono binaries are distributed through the
        # projectchrono conda channel or a source build, not the unrelated
        # PyPI project that can share the `pychrono` distribution name.
        benchmark_ids=("capture-to-deformation",),
        execution_mode="isolated_external_conda",
        target_version="10.0.0",
    ),
    "flash": AdapterRecipe(
        benchmark_ids=("capture-to-deformation",),
        execution_mode="unreleased_research_code",
        access_required=True,
    ),
    "garmentdynamics-rgbench": AdapterRecipe(
        benchmark_ids=("capture-to-deformation",),
        execution_mode="dataset_benchmark",
        access_required=True,
    ),
    "simweaver-sim1": AdapterRecipe(
        benchmark_ids=("capture-to-deformation",),
        execution_mode="unreleased_research_code",
        access_required=True,
    ),
    "dlo-lab": AdapterRecipe(
        benchmark_ids=("capture-to-deformation",),
        execution_mode="isolated_source_checkout",
        target_version="c5026a9416b03c6bc5186eba13cd4ffd4c0e7796",
    ),
    "pyelastica": AdapterRecipe(
        python_distributions=("pyelastica",),
        benchmark_ids=("capture-to-deformation",),
    ),
    "sofa-26-06": AdapterRecipe(
        python_distributions=("SofaPython3",),
        executables=("runSofa",),
        benchmark_ids=("capture-to-deformation",),
    ),
    "tacsl": AdapterRecipe(
        python_distributions=("tacsl",),
        benchmark_ids=("capture-to-observation", "capture-to-deformation"),
    ),
    "difftactile": AdapterRecipe(
        python_distributions=("difftactile",),
        benchmark_ids=("capture-to-observation", "capture-to-deformation"),
    ),
    "direct-captured-observations": AdapterRecipe(
        python_distributions=("opencv-python-headless",),
        benchmark_ids=("capture-to-observation", "capture-to-deformation"),
        execution_mode="pipeline_native_read_only",
    ),
    "gigaworld-wmbench": AdapterRecipe(
        benchmark_ids=("world-model-action-fidelity",),
        execution_mode="dataset_benchmark",
    ),
    "world-labs-marble": AdapterRecipe(
        benchmark_ids=("capture-to-observation",),
        execution_mode="provider_api",
        access_required=True,
    ),
    "lightwheel-simready": AdapterRecipe(
        benchmark_ids=("capture-to-geometry-and-contact", "capture-to-observation"),
        execution_mode="provider_api",
        access_required=True,
    ),
}


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise MeasurementAdapterError("measurement_adapter_artifact_not_json") from exc
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _catalog_by_id() -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in research_intake_catalog()}


def build_measurement_adapter_descriptor(candidate_id: str) -> dict[str, Any]:
    """Build a stable, non-authorizing adapter descriptor for one candidate."""

    candidate = _catalog_by_id().get(_string(candidate_id))
    if candidate is None:
        raise MeasurementAdapterError(f"measurement_adapter_candidate_unknown:{candidate_id}")
    candidate = validate_research_method_candidate(candidate)
    recipe = ADAPTER_RECIPES.get(candidate["candidate_id"])
    if recipe is None:
        recipe = AdapterRecipe(
            benchmark_ids=(),
            execution_mode="research_dossier_only",
            access_required=True,
        )
    value = {
        "schema_version": ADAPTER_DESCRIPTOR_SCHEMA_VERSION,
        "adapter_id": f"measurement-adapter-{candidate['candidate_id']}",
        "adapter_reference": f"measurement://{candidate['candidate_id']}/v1",
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["research_candidate_digest"],
        "method_id": candidate["method_id"],
        "method_family": candidate["method_family"],
        "target_version": recipe.target_version or candidate["version_observed"],
        "required_qualification_protocols": candidate["required_qualification_protocols"],
        "benchmark_ids": list(recipe.benchmark_ids),
        "probe_contract": {
            "python_distributions": list(recipe.python_distributions),
            "executables": list(recipe.executables),
            "imports_packages": False,
            "launches_processes": False,
        },
        "execution_mode": recipe.execution_mode,
        "execution_contract": {
            "request_schema_version": "measurement_adapter_execution_request.v1",
            "worker_result_schema_version": "measurement_adapter_worker_result.v1",
            "receipt_schema_version": "measurement_adapter_execution_receipt.v1",
            "worker_arguments": ["--request", "<path>", "--output", "<path>"],
            "shell_allowed": False,
            "explicit_execution_gate_required": True,
            "local_runner_scope": "development_only",
        },
        "access_required": recipe.access_required,
        "capability_semantics": "tri_state_until_independently_observed",
        "production_execution_authorized": False,
        "production_route_eligible": False,
        "physical_robot_execution_authorized": False,
        "agent_may_authorize": False,
    }
    value["adapter_descriptor_digest"] = _digest(value, "adapter_descriptor_digest")
    return validate_measurement_adapter_descriptor(value)


def validate_measurement_adapter_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = _clone(value)
    errors: list[str] = []
    if descriptor.get("schema_version") != ADAPTER_DESCRIPTOR_SCHEMA_VERSION:
        errors.append("measurement_adapter_descriptor_schema_invalid")
    for key in (
        "adapter_id",
        "adapter_reference",
        "candidate_id",
        "candidate_digest",
        "method_id",
        "method_family",
        "execution_mode",
    ):
        if not _string(descriptor.get(key)):
            errors.append(f"measurement_adapter_descriptor_{key}_missing")
    if not _string(descriptor.get("candidate_digest")).startswith("sha256:"):
        errors.append("measurement_adapter_descriptor_candidate_digest_invalid")
    for key in ("required_qualification_protocols", "benchmark_ids"):
        if not isinstance(descriptor.get(key), list):
            errors.append(f"measurement_adapter_descriptor_{key}_invalid")
    probe = descriptor.get("probe_contract")
    if not isinstance(probe, Mapping):
        errors.append("measurement_adapter_descriptor_probe_contract_invalid")
    execution = descriptor.get("execution_contract")
    if not isinstance(execution, Mapping):
        errors.append("measurement_adapter_descriptor_execution_contract_invalid")
    elif (
        execution.get("shell_allowed") is not False
        or execution.get("explicit_execution_gate_required") is not True
        or execution.get("local_runner_scope") != "development_only"
    ):
        errors.append("measurement_adapter_descriptor_execution_boundary_invalid")
    for key in (
        "production_execution_authorized",
        "production_route_eligible",
        "physical_robot_execution_authorized",
        "agent_may_authorize",
    ):
        if descriptor.get(key) is not False:
            errors.append(f"measurement_adapter_descriptor_{key}_must_be_false")
    expected = _digest(descriptor, "adapter_descriptor_digest")
    supplied = descriptor.get("adapter_descriptor_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_adapter_descriptor_digest_mismatch")
    if errors:
        raise MeasurementAdapterError(*errors)
    descriptor["adapter_descriptor_digest"] = expected
    return descriptor


def _distribution_probe(name: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "probe_type": "python_distribution",
        "name": name,
        "available": version is not None,
        "observed_version": version,
        "package_imported": False,
    }


def _executable_probe(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {
        "probe_type": "executable",
        "name": name,
        "available": path is not None,
        "resolved_path": path,
        "process_launched": False,
    }


def probe_measurement_adapter(value: Mapping[str, Any]) -> dict[str, Any]:
    """Probe installed metadata and PATH without importing or executing code."""

    descriptor = validate_measurement_adapter_descriptor(value)
    contract = dict(descriptor["probe_contract"])
    probes = [
        *(_distribution_probe(name) for name in contract.get("python_distributions") or []),
        *(_executable_probe(name) for name in contract.get("executables") or []),
    ]
    available = [row for row in probes if row["available"] is True]
    if descriptor["access_required"] is True and not probes:
        status = "access_required"
    elif not probes:
        status = "manual_review"
    elif len(available) == len(probes):
        status = "available"
    elif available:
        status = "partial"
    else:
        status = "unavailable"
    observed_versions = sorted(
        {
            _string(row.get("observed_version"))
            for row in probes
            if _string(row.get("observed_version"))
        }
    )
    target = _string(descriptor.get("target_version"))
    result = {
        "schema_version": ADAPTER_PROBE_SCHEMA_VERSION,
        "adapter_id": descriptor["adapter_id"],
        "adapter_descriptor_digest": descriptor["adapter_descriptor_digest"],
        "candidate_id": descriptor["candidate_id"],
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "status": status,
        "probes": probes,
        "observed_versions": observed_versions,
        "target_version": target,
        "target_version_observed": bool(target and target in observed_versions),
        "package_imported": False,
        "process_launched": False,
        "credentials_inspected": False,
        "capabilities_established": False,
        "qualification_established": False,
        "production_route_eligible": False,
        "execution_authorized": False,
    }
    result["adapter_probe_digest"] = _digest(result, "adapter_probe_digest")
    return validate_measurement_adapter_probe(result)


def validate_measurement_adapter_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    probe = _clone(value)
    errors: list[str] = []
    if probe.get("schema_version") != ADAPTER_PROBE_SCHEMA_VERSION:
        errors.append("measurement_adapter_probe_schema_invalid")
    if probe.get("status") not in PROBE_STATUSES:
        errors.append("measurement_adapter_probe_status_invalid")
    if not isinstance(probe.get("probes"), list):
        errors.append("measurement_adapter_probe_rows_invalid")
    for key in (
        "package_imported",
        "process_launched",
        "credentials_inspected",
        "capabilities_established",
        "qualification_established",
        "production_route_eligible",
        "execution_authorized",
    ):
        if probe.get(key) is not False:
            errors.append(f"measurement_adapter_probe_{key}_must_be_false")
    expected = _digest(probe, "adapter_probe_digest")
    supplied = probe.get("adapter_probe_digest")
    if supplied is not None and supplied != expected:
        errors.append("measurement_adapter_probe_digest_mismatch")
    if errors:
        raise MeasurementAdapterError(*errors)
    probe["adapter_probe_digest"] = expected
    return probe


def build_capability_draft(
    descriptor_value: Mapping[str, Any],
    probe_value: Mapping[str, Any],
    observations: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a tri-state draft from independently supplied observations.

    Each non-unknown value needs an evidence reference.  The dossier itself is
    not capability evidence and a successful install probe establishes no
    solver capability.
    """

    descriptor = validate_measurement_adapter_descriptor(descriptor_value)
    probe = validate_measurement_adapter_probe(probe_value)
    if probe["adapter_descriptor_digest"] != descriptor["adapter_descriptor_digest"]:
        raise MeasurementAdapterError("measurement_capability_draft_probe_mismatch")
    supplied = dict(observations or {})
    unknown = sorted(set(supplied) - set(ALL_CAPABILITY_FIELDS))
    if unknown:
        raise MeasurementAdapterError(
            "measurement_capability_draft_field_unknown:" + ",".join(unknown)
        )
    capabilities: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for field in sorted(ALL_CAPABILITY_FIELDS):
        observation = supplied.get(field)
        observation = dict(observation) if isinstance(observation, Mapping) else {}
        state = _string(observation.get("state")) or "unknown"
        evidence_refs = observation.get("evidence_refs") or []
        if state not in CAPABILITY_STATES:
            errors.append(f"measurement_capability_draft_state_invalid:{field}")
        if not isinstance(evidence_refs, list) or (state != "unknown" and not evidence_refs):
            errors.append(f"measurement_capability_draft_evidence_missing:{field}")
        capabilities[field] = {
            "state": state,
            "evidence_refs": sorted({_string(item) for item in evidence_refs if _string(item)}),
            "independently_verified": observation.get("independently_verified") is True,
        }
    if errors:
        raise MeasurementAdapterError(*errors)
    draft = {
        "schema_version": CAPABILITY_DRAFT_SCHEMA_VERSION,
        "candidate_id": descriptor["candidate_id"],
        "method_id": descriptor["method_id"],
        "adapter_descriptor_digest": descriptor["adapter_descriptor_digest"],
        "adapter_probe_digest": probe["adapter_probe_digest"],
        "capabilities": capabilities,
        "unknown_is_wildcard": False,
        "research_dossier_is_capability_evidence": False,
        "install_probe_is_qualification": False,
        "production_route_eligible": False,
        "agent_may_promote": False,
    }
    draft["capability_draft_digest"] = _digest(draft, "capability_draft_digest")
    return validate_capability_draft(draft)


def validate_capability_draft(value: Mapping[str, Any]) -> dict[str, Any]:
    draft = _clone(value)
    errors: list[str] = []
    if draft.get("schema_version") != CAPABILITY_DRAFT_SCHEMA_VERSION:
        errors.append("measurement_capability_draft_schema_invalid")
    capabilities = draft.get("capabilities")
    if not isinstance(capabilities, Mapping) or set(capabilities) != set(ALL_CAPABILITY_FIELDS):
        errors.append("measurement_capability_draft_fields_incomplete")
    else:
        for field, row in capabilities.items():
            if not isinstance(row, Mapping) or row.get("state") not in CAPABILITY_STATES:
                errors.append(f"measurement_capability_draft_row_invalid:{field}")
            elif row.get("state") != "unknown" and not row.get("evidence_refs"):
                errors.append(f"measurement_capability_draft_evidence_missing:{field}")
    for key, expected in (
        ("unknown_is_wildcard", False),
        ("research_dossier_is_capability_evidence", False),
        ("install_probe_is_qualification", False),
        ("production_route_eligible", False),
        ("agent_may_promote", False),
    ):
        if draft.get(key) is not expected:
            errors.append(f"measurement_capability_draft_{key}_invalid")
    expected_digest = _digest(draft, "capability_draft_digest")
    supplied_digest = draft.get("capability_draft_digest")
    if supplied_digest is not None and supplied_digest != expected_digest:
        errors.append("measurement_capability_draft_digest_mismatch")
    if errors:
        raise MeasurementAdapterError(*errors)
    draft["capability_draft_digest"] = expected_digest
    return draft


def build_adapter_admission_packet(
    descriptor_value: Mapping[str, Any],
    probe_value: Mapping[str, Any],
    draft_value: Mapping[str, Any],
    *,
    source_snapshot_digest: str,
) -> dict[str, Any]:
    """Bind R1/R3 evidence inputs without advancing or approving admission."""

    descriptor = validate_measurement_adapter_descriptor(descriptor_value)
    probe = validate_measurement_adapter_probe(probe_value)
    draft = validate_capability_draft(draft_value)
    if not _string(source_snapshot_digest).startswith("sha256:"):
        raise MeasurementAdapterError("adapter_admission_source_snapshot_digest_invalid")
    if any(
        (
            probe["adapter_descriptor_digest"] != descriptor["adapter_descriptor_digest"],
            draft["adapter_descriptor_digest"] != descriptor["adapter_descriptor_digest"],
            draft["adapter_probe_digest"] != probe["adapter_probe_digest"],
        )
    ):
        raise MeasurementAdapterError("adapter_admission_packet_binding_mismatch")
    packet = {
        "schema_version": ADMISSION_PACKET_SCHEMA_VERSION,
        "candidate_id": descriptor["candidate_id"],
        "method_id": descriptor["method_id"],
        "adapter_descriptor_digest": descriptor["adapter_descriptor_digest"],
        "adapter_probe_digest": probe["adapter_probe_digest"],
        "capability_draft_digest": draft["capability_draft_digest"],
        "source_snapshot_digest": source_snapshot_digest,
        "benchmark_ids": descriptor["benchmark_ids"],
        "required_qualification_protocols": descriptor["required_qualification_protocols"],
        "r1_source_verification_complete": False,
        "r2_rights_review_complete": False,
        "r3_adapter_feasibility_complete": False,
        "r4_benchmark_preregistered": False,
        "r5_independent_heldout_complete": False,
        "r6_human_decision_complete": False,
        "r7_catalog_admitted": False,
        "production_route_eligible": False,
        "execution_authorized": False,
    }
    packet["adapter_admission_packet_digest"] = _digest(packet, "adapter_admission_packet_digest")
    return packet


def priority_adapter_descriptors() -> tuple[dict[str, Any], ...]:
    """Descriptors for every explicitly implemented candidate recipe."""

    return tuple(
        build_measurement_adapter_descriptor(candidate_id)
        for candidate_id in sorted(ADAPTER_RECIPES)
    )


__all__ = [
    "ADAPTER_DESCRIPTOR_SCHEMA_VERSION",
    "ADAPTER_PROBE_SCHEMA_VERSION",
    "ADAPTER_RECIPES",
    "ADMISSION_PACKET_SCHEMA_VERSION",
    "CAPABILITY_DRAFT_SCHEMA_VERSION",
    "CAPABILITY_STATES",
    "MeasurementAdapterError",
    "build_adapter_admission_packet",
    "build_capability_draft",
    "build_measurement_adapter_descriptor",
    "priority_adapter_descriptors",
    "probe_measurement_adapter",
    "validate_capability_draft",
    "validate_measurement_adapter_descriptor",
    "validate_measurement_adapter_probe",
]
