"""Backend-neutral policy -> one WAM -> same-policy closed-loop harness.

The harness owns orchestration and evidence, not model-specific conversion.  A
registered transition adapter converts the policy action into exactly one WAM
arm's conditioning contract and reconstructs the next policy observation from
the WAM prediction plus explicitly declared state propagation.  This keeps an
OSCAR skeleton arm, native Cosmos raw-action arm, and future WAMs attributable
without allowing one WAM to feed another.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .common import ensure_dir, write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "policy_wam_closed_loop.v1"
TRACE_SCHEMA_VERSION = "policy_wam_closed_loop_trace.v1"
ALLOWED_STATE_SOURCES = frozenset(
    {
        "commanded_prefix_kinematics",
        "wam_estimated_state",
        "registered_state_estimator",
    }
)
FORBIDDEN_WAM_IDENTITY_KEYS = frozenset(
    {
        "policy",
        "policy_id",
        "policy_name",
        "candidate_policy",
        "candidate_policy_id",
        "physical_outcome",
        "success_rate",
        "score",
    }
)


class PolicyClient(Protocol):
    """The frozen candidate policy queried throughout one episode."""

    policy_id: str

    def infer(self, observation: Mapping[str, Any]) -> Any:
        raise NotImplementedError


class WamArm(Protocol):
    """One attributable WAM arm.  No collection of WAMs is accepted here."""

    arm_id: str

    def predict(
        self,
        request: Mapping[str, Any],
        *,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class TransitionAdapter(Protocol):
    """Registered policy-action/WAM-conditioning/observation conversion."""

    adapter_id: str

    def prepare_transition(
        self,
        *,
        observation: Mapping[str, Any],
        policy_action: Any,
        task_prompt: str,
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Return a mapping containing the provider-safe ``wam_request``."""
        raise NotImplementedError

    def advance_policy_observation(
        self,
        *,
        previous_observation: Mapping[str, Any],
        prepared_transition: Mapping[str, Any],
        wam_prediction: Mapping[str, Any],
        executed_prefix_steps: int,
        query_index: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """Return ``observation`` and explicit visual/state provenance."""
        raise NotImplementedError


class ReliabilityGate(Protocol):
    """Frozen per-arm collapse, uncertainty, and causal-reliability gate."""

    gate_id: str

    def assess(
        self,
        *,
        previous_observation: Mapping[str, Any],
        prepared_transition: Mapping[str, Any],
        wam_prediction: Mapping[str, Any],
        query_index: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


class TerminalCriterion(Protocol):
    """Task-specific terminal predicate that is not a candidate-policy judge."""

    criterion_id: str

    def assess(
        self,
        *,
        observation: Mapping[str, Any],
        query_index: int,
    ) -> Mapping[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ClosedLoopConfig:
    task_prompt: str
    executed_prefix_steps: int
    max_policy_queries: int
    control_hz: float = 15.0
    execution_mode: str = "scientific"

    def validate(self) -> None:
        if not self.task_prompt.strip():
            raise ValueError("task_prompt_missing")
        if isinstance(self.executed_prefix_steps, bool) or not isinstance(
            self.executed_prefix_steps, int
        ):
            raise ValueError("executed_prefix_steps_must_be_integer")
        if self.executed_prefix_steps <= 0:
            raise ValueError("executed_prefix_steps_must_be_positive")
        if isinstance(self.max_policy_queries, bool) or not isinstance(
            self.max_policy_queries, int
        ):
            raise ValueError("max_policy_queries_must_be_integer")
        if self.max_policy_queries <= 0:
            raise ValueError("max_policy_queries_must_be_positive")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz_must_be_positive_finite")
        if self.execution_mode not in {"engineering_smoke", "scientific"}:
            raise ValueError("closed_loop_execution_mode_invalid")

    @property
    def executed_prefix_seconds_derived(self) -> float:
        return self.executed_prefix_steps / self.control_hz


def _jsonable(value: Any) -> Any:
    """Return deterministic evidence material without embedding raw image arrays."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a project dependency
        np = None
    if np is not None and isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "ndarray",
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        resolved = value.expanduser().resolve()
        return {
            "kind": "file",
            "path": str(resolved),
            "sha256": file_sha256(resolved) if resolved.is_file() else None,
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(child) for child in value]
    if isinstance(value, bytes):
        return {"kind": "bytes", "size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    return value


def _extract_policy_action(response: Any) -> Any:
    if isinstance(response, Mapping):
        for key in ("actions", "action", "action_chunk"):
            if key in response:
                return response[key]
        raise ValueError("policy_response_missing_action_chunk")
    return response


def _forbidden_wam_request_keys(value: Any, *, prefix: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            path = f"{prefix}.{normalized}" if prefix else normalized
            if normalized in FORBIDDEN_WAM_IDENTITY_KEYS:
                violations.append(path)
            violations.extend(_forbidden_wam_request_keys(child, prefix=path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            violations.extend(_forbidden_wam_request_keys(child, prefix=f"{prefix}[{index}]"))
    return violations


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def run_policy_wam_closed_loop(
    *,
    initial_observation: Mapping[str, Any],
    policy_client: PolicyClient,
    wam_arm: WamArm,
    transition_adapter: TransitionAdapter,
    reliability_gate: ReliabilityGate,
    terminal_criterion: TerminalCriterion,
    config: ClosedLoopConfig,
    output_dir: str | Path,
    conditioning_fidelity_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same candidate policy against exactly one WAM until terminal.

    The evaluator is intentionally absent from this loop.  A WAM or candidate
    policy can neither grade itself nor choose another WAM's output.
    """

    config.validate()
    validated_conditioning_certificate: dict[str, Any] | None = None
    if config.execution_mode == "scientific":
        from .wam_conditioning_fidelity import validate_conditioning_fidelity_certificate

        validated_conditioning_certificate = validate_conditioning_fidelity_certificate(
            conditioning_fidelity_certificate,
            backend_id=str(wam_arm.arm_id),
        )
    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    trace_path = output / "policy_wam_closed_loop_trace.jsonl"
    current_observation = dict(initial_observation)
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    terminal_reason = "maximum_horizon_reached"
    status = "max_horizon"
    started_episode = time.monotonic()

    for query_index in range(config.max_policy_queries):
        step_dir = output / f"transition_{query_index:04d}"
        ensure_dir(step_dir)
        row: dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "query_index": query_index,
            "policy_observation_sha256": canonical_sha256(_jsonable(current_observation)),
            "policy_id_internal_only": str(policy_client.policy_id),
            "wam_arm_id": str(wam_arm.arm_id),
            "transition_adapter_id": str(transition_adapter.adapter_id),
            "reliability_gate_id": str(reliability_gate.gate_id),
        }
        try:
            policy_started = time.monotonic()
            response = policy_client.infer(current_observation)
            policy_latency = time.monotonic() - policy_started
            action = _extract_policy_action(response)
            row["policy_latency_seconds"] = policy_latency
            row["policy_action_sha256"] = canonical_sha256(_jsonable(action))

            prepared = transition_adapter.prepare_transition(
                observation=current_observation,
                policy_action=action,
                task_prompt=config.task_prompt,
                executed_prefix_steps=config.executed_prefix_steps,
                query_index=query_index,
                output_dir=step_dir,
            )
            wam_request = prepared.get("wam_request")
            if not isinstance(wam_request, Mapping):
                raise ValueError("transition_adapter_missing_wam_request")
            leaked = _forbidden_wam_request_keys(wam_request)
            if leaked:
                raise ValueError(f"wam_request_policy_or_outcome_leakage:{leaked[0]}")
            row["prepared_transition_sha256"] = canonical_sha256(_jsonable(prepared))
            row["wam_request_sha256"] = canonical_sha256(_jsonable(wam_request))

            wam_started = time.monotonic()
            prediction = wam_arm.predict(wam_request, output_dir=step_dir)
            wam_latency = time.monotonic() - wam_started
            if not isinstance(prediction, Mapping):
                raise ValueError("wam_prediction_not_mapping")
            row["wam_latency_seconds"] = wam_latency
            row["wam_prediction_sha256"] = canonical_sha256(_jsonable(prediction))

            gate_result = reliability_gate.assess(
                previous_observation=current_observation,
                prepared_transition=prepared,
                wam_prediction=prediction,
                query_index=query_index,
                output_dir=step_dir,
            )
            if not isinstance(gate_result, Mapping):
                raise ValueError("reliability_gate_result_not_mapping")
            row["reliability"] = _jsonable(gate_result)
            if gate_result.get("abstain") is True:
                row["status"] = "abstained"
                rows.append(row)
                status = "abstained"
                terminal_reason = "wam_reliability_gate_abstention"
                break

            advanced = transition_adapter.advance_policy_observation(
                previous_observation=current_observation,
                prepared_transition=prepared,
                wam_prediction=prediction,
                executed_prefix_steps=config.executed_prefix_steps,
                query_index=query_index,
                output_dir=step_dir,
            )
            next_observation = advanced.get("observation")
            provenance = advanced.get("provenance")
            if not isinstance(next_observation, Mapping) or not isinstance(provenance, Mapping):
                raise ValueError("advanced_policy_observation_contract_invalid")
            if provenance.get("visual_source") != "wam_prediction":
                raise ValueError("next_policy_visual_not_attributable_to_wam_prediction")
            if provenance.get("state_source") not in ALLOWED_STATE_SOURCES:
                raise ValueError("next_policy_state_source_unregistered")
            if provenance.get("physical_future_observation_used") is not False:
                raise ValueError("physical_future_observation_use_not_explicitly_false")
            row["next_observation_sha256"] = canonical_sha256(_jsonable(next_observation))
            row["next_observation_provenance"] = _jsonable(provenance)

            terminal = terminal_criterion.assess(
                observation=next_observation,
                query_index=query_index,
            )
            if not isinstance(terminal, Mapping):
                raise ValueError("terminal_criterion_result_not_mapping")
            row["terminal"] = _jsonable(terminal)
            row["status"] = "completed"
            rows.append(row)
            current_observation = dict(next_observation)
            if terminal.get("terminal") is True:
                status = "completed"
                terminal_reason = str(terminal.get("reason") or "task_terminal")
                break
        except Exception as exc:  # noqa: BLE001 - failure is preserved as evidence
            blocker = f"transition_{query_index}:{type(exc).__name__}:{exc}"
            blockers.append(blocker)
            row["status"] = "blocked"
            row["blocker"] = blocker
            rows.append(row)
            status = "blocked"
            terminal_reason = "fail_closed_transition_error"
            break

    _write_jsonl(trace_path, rows)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "terminal_reason": terminal_reason,
        "policy_id_internal_only": str(policy_client.policy_id),
        "wam_arm_id": str(wam_arm.arm_id),
        "transition_adapter_id": str(transition_adapter.adapter_id),
        "reliability_gate_id": str(reliability_gate.gate_id),
        "terminal_criterion_id": str(terminal_criterion.criterion_id),
        "task_prompt_sha256": hashlib.sha256(config.task_prompt.encode("utf-8")).hexdigest(),
        "task_prompt_provider_visible": True,
        "executed_prefix_steps": config.executed_prefix_steps,
        "executed_prefix_seconds_derived": config.executed_prefix_seconds_derived,
        "control_hz": config.control_hz,
        "execution_mode": config.execution_mode,
        "conditioning_fidelity_certificate_sha256": (
            validated_conditioning_certificate["manifest_sha256"]
            if validated_conditioning_certificate is not None
            else None
        ),
        "conditioning_fidelity_certificate_passed": bool(
            validated_conditioning_certificate is not None
        ),
        "maximum_policy_queries": config.max_policy_queries,
        "completed_policy_queries": sum(row.get("status") == "completed" for row in rows),
        "policy_call_count": len(rows),
        "wam_call_count": sum("wam_prediction_sha256" in row for row in rows),
        "elapsed_seconds": time.monotonic() - started_episode,
        "blockers": blockers,
        "trace_path": str(trace_path),
        "trace_sha256": file_sha256(trace_path),
        "architecture": {
            "same_policy_requeried": True,
            "exactly_one_wam_arm_per_transition": True,
            "wam_to_wam_chaining": False,
            "evaluator_in_control_loop": False,
            "physical_future_observation_allowed": False,
        },
        "claim_boundary": {
            "engineering_smoke_only": config.execution_mode == "engineering_smoke",
            "scientific_execution_admitted": bool(
                config.execution_mode == "scientific"
                and validated_conditioning_certificate is not None
            ),
            "domain_wam_qualification_proven_by_loop": False,
            "policy_rank_fidelity_proven": False,
            "physical_success_proven": False,
            "explanation": (
                "Technical closed-loop execution only; a conditioning certificate admits "
                "the run but does not make generated observations physical evidence."
            ),
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output / "policy_wam_closed_loop_manifest.json", manifest)
    return manifest


__all__ = [
    "ALLOWED_STATE_SOURCES",
    "ClosedLoopConfig",
    "PolicyClient",
    "ReliabilityGate",
    "SCHEMA_VERSION",
    "TerminalCriterion",
    "TransitionAdapter",
    "WamArm",
    "run_policy_wam_closed_loop",
]
