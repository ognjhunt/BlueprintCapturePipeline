"""Provider-neutral Evaluation Run contract and compiler.

The public interface is deliberately small: callers submit one versioned run
specification containing exactly six replaceable parts and receive one
canonical, proof-bounded execution plan.  Scene-, robot-, task-, policy-, and
provider-specific implementations stay behind those bindings instead of
leaking into the orchestration contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_contracts import ArtifactContractError, validate_sellable_artifact
from .common import read_json, utc_now_iso, write_json


EVALUATION_RUN_SCHEMA_VERSION = "evaluation_run.v1"
EVALUATION_RUN_PLAN_SCHEMA_VERSION = "evaluation_run_plan.v1"
EVALUATION_RUN_COMPONENTS = (
    "scene_bundle",
    "robot_adapter",
    "task_scenario_pack",
    "policy_adapter",
    "runtime_provider_profile",
    "proof_contract",
)
EVALUATION_RUN_MODES = {"evaluate", "startup_canary", "serve"}
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}


@dataclass(frozen=True)
class EvaluationRunSpec:
    """The stable six-part interface consumed by the Evaluation Run compiler."""

    run_id: str
    mode: str
    scene_bundle: Mapping[str, Any]
    robot_adapter: Mapping[str, Any]
    task_scenario_pack: Mapping[str, Any]
    policy_adapter: Mapping[str, Any]
    runtime_provider_profile: Mapping[str, Any]
    proof_contract: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationRunSpec":
        return cls(
            run_id=_string(value.get("run_id")),
            mode=_string(value.get("mode")) or "evaluate",
            scene_bundle=_mapping(value.get("scene_bundle")),
            robot_adapter=_mapping(value.get("robot_adapter")),
            task_scenario_pack=_mapping(value.get("task_scenario_pack")),
            policy_adapter=_mapping(value.get("policy_adapter")),
            runtime_provider_profile=_mapping(value.get("runtime_provider_profile")),
            proof_contract=_mapping(value.get("proof_contract")),
            metadata=_mapping(value.get("metadata")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": self.mode,
            "scene_bundle": dict(self.scene_bundle),
            "robot_adapter": dict(self.robot_adapter),
            "task_scenario_pack": dict(self.task_scenario_pack),
            "policy_adapter": dict(self.policy_adapter),
            "runtime_provider_profile": dict(self.runtime_provider_profile),
            "proof_contract": dict(self.proof_contract),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EvaluationRunAdapterDescriptor:
    """One implementation available at a six-part Evaluation Run seam."""

    component: str
    adapter_id: str
    adapter_version: str
    capabilities: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "capabilities": list(self.capabilities),
        }


class EvaluationRunAdapterRegistry:
    """Resolve concrete adapters without coupling the compiler to their implementation."""

    def __init__(
        self, descriptors: Sequence[EvaluationRunAdapterDescriptor] = ()
    ) -> None:
        self._descriptors: dict[
            tuple[str, str, str], EvaluationRunAdapterDescriptor
        ] = {}
        for descriptor in descriptors:
            self.register(descriptor)

    def register(self, descriptor: EvaluationRunAdapterDescriptor) -> None:
        if descriptor.component not in EVALUATION_RUN_COMPONENTS:
            raise ValueError(f"unsupported_evaluation_run_component:{descriptor.component}")
        key = (
            descriptor.component,
            descriptor.adapter_id,
            descriptor.adapter_version,
        )
        if key in self._descriptors:
            raise ValueError(
                "duplicate_evaluation_run_adapter:"
                f"{descriptor.component}:{descriptor.adapter_id}@{descriptor.adapter_version}"
            )
        self._descriptors[key] = descriptor

    def resolve(
        self, *, component: str, adapter_id: str, adapter_version: str
    ) -> EvaluationRunAdapterDescriptor | None:
        return self._descriptors.get((component, adapter_id, adapter_version))

    def manifest(self) -> list[dict[str, Any]]:
        return [
            descriptor.to_mapping()
            for _, descriptor in sorted(self._descriptors.items())
        ]


DEFAULT_EVALUATION_RUN_ADAPTERS = (
    EvaluationRunAdapterDescriptor(
        "scene_bundle", "openusd_scene_bundle", "1", ("openusd", "content_inventory")
    ),
    EvaluationRunAdapterDescriptor(
        "scene_bundle", "capture_site_scene_bundle", "1", ("capture", "site_package")
    ),
    EvaluationRunAdapterDescriptor(
        "robot_adapter", "isaac_unitree_g1", "1", ("unitree_g1", "isaac_sim")
    ),
    EvaluationRunAdapterDescriptor(
        "robot_adapter", "robot_profile_adapter", "1", ("profile", "multi_robot")
    ),
    EvaluationRunAdapterDescriptor(
        "robot_adapter", "isaac_robot_asset", "1", ("openusd", "isaac_sim")
    ),
    EvaluationRunAdapterDescriptor(
        "task_scenario_pack", "manifest_task_scenario_pack", "1", ("manifest",)
    ),
    EvaluationRunAdapterDescriptor(
        "task_scenario_pack",
        "robot_eval_matrix_task_scenario_pack",
        "1",
        ("matrix", "variations"),
    ),
    EvaluationRunAdapterDescriptor(
        "task_scenario_pack",
        "benchmark_task_scenario_pack",
        "1",
        ("frozen_splits", "hidden_test", "seen_unseen", "fixed_rollouts"),
    ),
    EvaluationRunAdapterDescriptor(
        "policy_adapter", "isaac_g1_deterministic_controller", "1", ("in_process",)
    ),
    EvaluationRunAdapterDescriptor(
        "policy_adapter",
        "unitree_groot_n17_sonic",
        "1",
        ("persistent_worker", "command"),
    ),
    EvaluationRunAdapterDescriptor(
        "policy_adapter", "robot_eval_policy_package", "1", ("multi_modality",)
    ),
    EvaluationRunAdapterDescriptor(
        "policy_adapter",
        "blueprint_agentic_candidate_policy",
        "1",
        ("frozen_manifest", "composite_agent", "no_evaluator_authority"),
    ),
    EvaluationRunAdapterDescriptor(
        "policy_adapter", "http_policy_worker", "1", ("http", "persistent_worker")
    ),
    EvaluationRunAdapterDescriptor(
        "runtime_provider_profile",
        "isaac_provider_runtime",
        "1",
        ("isaac_sim", "paid_provider"),
    ),
    EvaluationRunAdapterDescriptor(
        "runtime_provider_profile",
        "robot_eval_runtime_provider",
        "1",
        ("multi_simulator", "multi_provider"),
    ),
    EvaluationRunAdapterDescriptor(
        "proof_contract",
        "declared_evidence_proof_contract",
        "1",
        ("claim_ceiling", "required_evidence"),
    ),
    EvaluationRunAdapterDescriptor(
        "proof_contract",
        "robot_eval_proof_contract",
        "1",
        ("claim_ceiling", "runtime_closure"),
    ),
)


def default_evaluation_run_adapter_registry() -> EvaluationRunAdapterRegistry:
    return EvaluationRunAdapterRegistry(DEFAULT_EVALUATION_RUN_ADAPTERS)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_string(item) for item in value if _string(item)]
    return []


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _redact(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_KEYS):
        return "REDACTED_SECRET_FIELD" if value not in (None, "") else value
    if isinstance(value, Mapping):
        return {str(k): _redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact(item) for item in value]
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _valid_id(value: str) -> bool:
    return bool(_IDENTIFIER.fullmatch(value))


def _adapter_binding_errors(name: str, value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    adapter_id = _string(value.get("adapter_id"))
    adapter_version = _string(value.get("adapter_version"))
    if not adapter_id:
        errors.append(f"{name}.adapter_id:missing")
    elif not _valid_id(adapter_id):
        errors.append(f"{name}.adapter_id:invalid")
    if not adapter_version:
        errors.append(f"{name}.adapter_version:missing")
    elif not _valid_id(adapter_version):
        errors.append(f"{name}.adapter_version:invalid")
    return errors


def _scene_bundle_errors(spec: EvaluationRunSpec) -> tuple[list[str], list[str]]:
    value = spec.scene_bundle
    errors = _adapter_binding_errors("scene_bundle", value)
    warnings: list[str] = []
    bundle_id = _string(value.get("bundle_id"))
    uri = _string(value.get("uri"))
    entrypoint = _string(value.get("entrypoint"))
    digest = _string(value.get("content_digest"))
    identity_status = _string(value.get("identity_status"))
    if not bundle_id or not _valid_id(bundle_id):
        errors.append("scene_bundle.bundle_id:missing_or_invalid")
    if not uri and spec.mode != "startup_canary":
        errors.append("scene_bundle.uri:missing")
    if not entrypoint and spec.mode != "startup_canary":
        errors.append("scene_bundle.entrypoint:missing")
    if digest and not _SHA256.fullmatch(digest):
        errors.append("scene_bundle.content_digest:invalid_sha256")
    if not digest:
        if identity_status in {"legacy_unverified", "pending_materialization"}:
            warnings.append(f"scene_bundle_identity_{identity_status}")
        elif spec.mode != "startup_canary":
            errors.append("scene_bundle.content_identity:missing")
    return errors, warnings


def _robot_adapter_errors(spec: EvaluationRunSpec) -> list[str]:
    value = spec.robot_adapter
    errors = _adapter_binding_errors("robot_adapter", value)
    if not _string(value.get("robot_profile_id")):
        errors.append("robot_adapter.robot_profile_id:missing")
    if not _string(value.get("asset_ref")) and spec.mode != "startup_canary":
        errors.append("robot_adapter.asset_ref:missing")
    return errors


def _task_pack_errors(spec: EvaluationRunSpec) -> list[str]:
    value = spec.task_scenario_pack
    errors = _adapter_binding_errors("task_scenario_pack", value)
    pack_id = _string(value.get("pack_id"))
    if not pack_id or not _valid_id(pack_id):
        errors.append("task_scenario_pack.pack_id:missing_or_invalid")
    tasks = _rows(value.get("tasks"))
    scenarios = _rows(value.get("scenarios"))
    if spec.mode == "evaluate" and not tasks:
        errors.append("task_scenario_pack.tasks:missing")
    if spec.mode == "evaluate" and not scenarios:
        errors.append("task_scenario_pack.scenarios:missing")
    task_ids = [_string(item.get("task_id") or item.get("id")) for item in tasks]
    scenario_ids = [_string(item.get("scenario_id") or item.get("id")) for item in scenarios]
    if any(not value for value in task_ids):
        errors.append("task_scenario_pack.tasks:missing_task_id")
    if any(not value for value in scenario_ids):
        errors.append("task_scenario_pack.scenarios:missing_scenario_id")
    if len(set(task_ids)) != len(task_ids):
        errors.append("task_scenario_pack.tasks:duplicate_task_id")
    if len(set(scenario_ids)) != len(scenario_ids):
        errors.append("task_scenario_pack.scenarios:duplicate_scenario_id")
    known_tasks = set(task_ids)
    for scenario in scenarios:
        task_id = _string(scenario.get("task_id"))
        if task_id and known_tasks and task_id not in known_tasks:
            errors.append("task_scenario_pack.scenarios:unknown_task_id")
            break
    return errors


def _policy_adapter_errors(spec: EvaluationRunSpec) -> list[str]:
    value = spec.policy_adapter
    errors = _adapter_binding_errors("policy_adapter", value)
    if not _string(value.get("policy_id")) and spec.mode != "startup_canary":
        errors.append("policy_adapter.policy_id:missing")
    observation_schema = _string(value.get("observation_schema_ref"))
    action_schema = _string(value.get("action_schema_ref"))
    if spec.mode == "evaluate" and not observation_schema:
        errors.append("policy_adapter.observation_schema_ref:missing")
    if spec.mode == "evaluate" and not action_schema:
        errors.append("policy_adapter.action_schema_ref:missing")
    return errors


def _runtime_profile_errors(spec: EvaluationRunSpec) -> list[str]:
    value = spec.runtime_provider_profile
    errors = _adapter_binding_errors("runtime_provider_profile", value)
    if not _string(value.get("profile_id")):
        errors.append("runtime_provider_profile.profile_id:missing")
    providers = _string_list(value.get("providers") or value.get("provider"))
    if not providers:
        errors.append("runtime_provider_profile.providers:missing")
    if not _string(value.get("simulator")):
        errors.append("runtime_provider_profile.simulator:missing")
    max_spend = value.get("max_spend_usd")
    if max_spend is not None:
        try:
            # Zero is an explicit no-spend ceiling for local/fixture runs.
            if float(max_spend) < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("runtime_provider_profile.max_spend_usd:invalid")
    return errors


def _proof_contract_errors(spec: EvaluationRunSpec) -> list[str]:
    value = spec.proof_contract
    errors = _adapter_binding_errors("proof_contract", value)
    if not _string(value.get("contract_id")):
        errors.append("proof_contract.contract_id:missing")
    if not _string_list(value.get("required_evidence")):
        errors.append("proof_contract.required_evidence:missing")
    claim_ceiling = _mapping(value.get("claim_ceiling"))
    if not claim_ceiling:
        errors.append("proof_contract.claim_ceiling:missing")
    prohibited_claims = _string_list(value.get("prohibited_claims"))
    if not prohibited_claims:
        errors.append("proof_contract.prohibited_claims:missing")
    return errors


def validate_evaluation_run_spec(
    value: Mapping[str, Any],
    *,
    adapter_registry: EvaluationRunAdapterRegistry | None = None,
) -> dict[str, Any]:
    """Validate the complete six-part interface without performing I/O."""

    raw_schema = _string(value.get("schema_version"))
    spec = EvaluationRunSpec.from_mapping(value)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        validate_sellable_artifact("evaluation_run", value)
    except ArtifactContractError as exc:
        errors.append(str(exc))
    if raw_schema != EVALUATION_RUN_SCHEMA_VERSION:
        errors.append(f"schema_version:must_be:{EVALUATION_RUN_SCHEMA_VERSION}")
    if not spec.run_id or not _valid_id(spec.run_id):
        errors.append("run_id:missing_or_invalid")
    if spec.mode not in EVALUATION_RUN_MODES:
        errors.append("mode:unsupported")
    for component in EVALUATION_RUN_COMPONENTS:
        if not isinstance(value.get(component), Mapping):
            errors.append(f"{component}:missing")
    scene_errors, scene_warnings = _scene_bundle_errors(spec)
    errors.extend(scene_errors)
    warnings.extend(scene_warnings)
    errors.extend(_robot_adapter_errors(spec))
    errors.extend(_task_pack_errors(spec))
    errors.extend(_policy_adapter_errors(spec))
    errors.extend(_runtime_profile_errors(spec))
    errors.extend(_proof_contract_errors(spec))
    registry = adapter_registry or default_evaluation_run_adapter_registry()
    adapter_resolution: dict[str, dict[str, Any]] = {}
    for component in EVALUATION_RUN_COMPONENTS:
        binding = _mapping(value.get(component))
        adapter_id = _string(binding.get("adapter_id"))
        adapter_version = _string(binding.get("adapter_version"))
        descriptor = registry.resolve(
            component=component,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
        )
        if descriptor is None and adapter_id and adapter_version:
            errors.append(
                f"{component}.adapter:unsupported:{adapter_id}@{adapter_version}"
            )
        adapter_resolution[component] = {
            "status": "resolved" if descriptor is not None else "unresolved",
            "adapter_id": adapter_id or None,
            "adapter_version": adapter_version or None,
            "capabilities": list(descriptor.capabilities) if descriptor else [],
        }
    normalized = _redact(spec.to_mapping())
    return {
        "schema_version": "evaluation_run_validation.v1",
        "status": "passed" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "run_id": spec.run_id or None,
        "mode": spec.mode,
        "spec_digest": _canonical_digest(normalized),
        "component_count": len(EVALUATION_RUN_COMPONENTS),
        "required_components": list(EVALUATION_RUN_COMPONENTS),
        "adapter_resolution": adapter_resolution,
        "raw_secret_values_recorded": False,
    }


def compile_evaluation_run(
    value: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    generated_at: str | None = None,
    adapter_registry: EvaluationRunAdapterRegistry | None = None,
) -> dict[str, Any]:
    """Compile one run specification into a canonical execution plan.

    Compilation is provider-neutral and side-effect free except for optional
    artifact writes.  It never stages assets, launches a provider, invokes a
    policy, or upgrades a proof claim.
    """

    sanitized_input = _redact(dict(value))
    spec = EvaluationRunSpec.from_mapping(sanitized_input)
    normalized = _redact(spec.to_mapping())
    validation = validate_evaluation_run_spec(
        sanitized_input,
        adapter_registry=adapter_registry,
    )
    component_bindings = {
        name: {
            "adapter_id": _string(_mapping(normalized.get(name)).get("adapter_id")) or None,
            "adapter_version": _string(
                _mapping(normalized.get(name)).get("adapter_version")
            )
            or None,
        }
        for name in EVALUATION_RUN_COMPONENTS
    }
    plan = {
        "schema_version": EVALUATION_RUN_PLAN_SCHEMA_VERSION,
        "generated_at": generated_at or utc_now_iso(),
        "status": "prepared" if validation["status"] == "passed" else "blocked",
        "run_id": spec.run_id or None,
        "mode": spec.mode,
        "spec_digest": validation["spec_digest"],
        "validation": validation,
        "component_bindings": component_bindings,
        "adapter_resolution": validation["adapter_resolution"],
        "execution_order": [
            "resolve_scene_bundle",
            "bind_robot_adapter",
            "materialize_task_scenario_pack",
            "bind_policy_adapter",
            "apply_runtime_provider_profile",
            "enforce_proof_contract",
            "execute",
            "collect_and_close",
        ],
        "execution_handoff": {
            "adapter_id": _string(spec.runtime_provider_profile.get("execution_adapter_id"))
            or None,
            "provider_mutation_allowed": False,
            "requires_explicit_runtime_gate": True,
        },
        "claim_boundary": {
            "plan_is_not_execution_proof": True,
            "plan_is_not_provider_startup_proof": True,
            "plan_is_not_policy_execution_proof": True,
            "plan_is_not_task_success_proof": True,
            "raw_capture_truth_is_not_rewritten": True,
            "raw_secret_values_recorded": False,
        },
    }
    if output_dir is not None:
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        spec_path = root / "evaluation_run_spec.json"
        plan_path = root / "evaluation_run_plan.json"
        write_json(spec_path, normalized)
        write_json(plan_path, plan)
        plan["artifacts"] = {
            "spec": str(spec_path),
            "plan": str(plan_path),
        }
        write_json(plan_path, plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = compile_evaluation_run(read_json(args.spec), output_dir=args.output_dir)
    print(json.dumps(plan, sort_keys=True))
    return 0 if plan["status"] == "prepared" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
