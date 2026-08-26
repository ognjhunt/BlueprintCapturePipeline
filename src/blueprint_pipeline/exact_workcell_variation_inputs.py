"""Typed scene/task/embodiment input assembly for exact-workcell matrices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .decision_evidence_contracts import canonical_digest, canonical_json
from .exact_workcell_variation_matrix import (
    DEFAULT_CELL_COUNT,
    EMBODIMENT_INPUT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    REQUIRED_CONTROLS,
    SCENE_INPUT_SCHEMA_VERSION,
    TASK_INPUT_SCHEMA_VERSION,
    ExactWorkcellVariationError,
    VariationProposalAgent,
    _is_digest,
    _json_clone,
    _rows,
    _string,
    build_agent_proposal_brief,
    seal_agent_proposal,
    validate_variation_request,
)
from .task_evaluation_supervisor.agents_sdk import (
    DEFAULT_SUPERVISOR_AGENT_MODEL,
    AgentsSDKAgentSpec,
    AgentsSDKInvoker,
)


class VariationDimensionPriorityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_id: str = Field(min_length=1, max_length=192)
    weight: float = Field(gt=0, le=100)


class VariationTargetedInteractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_ids: list[str] = Field(min_length=2, max_length=4)
    rationale: str = Field(min_length=1, max_length=1_000)


class ExactWorkcellVariationAgentOutput(BaseModel):
    """Strict proposal-only output; it contains no authority or proof fields."""

    model_config = ConfigDict(extra="forbid")

    dimension_priorities: list[VariationDimensionPriorityOutput] = Field(
        default_factory=list, max_length=100
    )
    targeted_interactions: list[VariationTargetedInteractionOutput] = Field(
        default_factory=list, max_length=100
    )
    object_cousins: list[str] = Field(default_factory=list, max_length=0)


@dataclass
class AgentsSDKVariationProposalAgent:
    """Canonical OpenAI Agents SDK adapter for bounded variation proposals."""

    invoker: AgentsSDKInvoker
    run_id: str
    model: str = DEFAULT_SUPERVISOR_AGENT_MODEL
    max_turns: int = 1
    max_output_tokens: int = 4_000
    _observed_model_identity: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ExactWorkcellVariationError(["variation_agent_run_id_missing"])
        if not self.model.strip():
            raise ExactWorkcellVariationError(["variation_agent_model_missing"])
        if self.max_turns != 1:
            raise ExactWorkcellVariationError(["variation_agent_single_turn_required"])
        if not 256 <= self.max_output_tokens <= 16_000:
            raise ExactWorkcellVariationError(
                ["variation_agent_output_token_ceiling_invalid"]
            )

    @property
    def model_identity(self) -> str:
        return self._observed_model_identity or f"openai-agents-sdk:{self.model}"

    def propose(self, *, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        spec = AgentsSDKAgentSpec(
            run_id=self.run_id,
            capability="exact_workcell_variation_proposal",
            name="Blueprint Exact-Workcell Variation Designer",
            instructions=(
                "Return only a bounded proposal over the supplied admitted dimensions. "
                "Prioritize useful dimensions and propose scientifically useful two-to-four "
                "dimension interactions. Never widen ranges, invent measurements, include "
                "object cousins, change scene/task/embodiment identity, or claim authority."
            ),
            model=self.model,
            max_turns=1,
            max_output_tokens=self.max_output_tokens,
            output_type=ExactWorkcellVariationAgentOutput,
        )
        invocation = self.invoker.invoke(spec, canonical_json({"brief": brief}))
        if invocation.model != self.model or not invocation.provider.strip():
            raise ExactWorkcellVariationError(
                ["variation_agent_runtime_identity_mismatch"]
            )
        output = ExactWorkcellVariationAgentOutput.model_validate(invocation.output)
        self._observed_model_identity = (
            f"{invocation.provider}-agents-sdk:{invocation.model}@{invocation.sdk_version}"
        )
        return output.model_dump(mode="json")


def build_variation_request_from_admitted_contracts(
    *,
    matrix_id: str,
    implementation_commit: str,
    seed_root: int,
    scene_contract: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    embodiment_contract: Mapping[str, Any],
    cell_count: int = DEFAULT_CELL_COUNT,
    agent: VariationProposalAgent | None = None,
) -> dict[str, Any]:
    """Merge typed envelopes and seal optional agent-proposed emphasis."""

    scene = _json_clone(dict(scene_contract))
    task = _json_clone(dict(task_contract))
    embodiment = _json_clone(dict(embodiment_contract))
    blockers: list[str] = []
    for label, contract, schema in (
        ("scene_contract", scene, SCENE_INPUT_SCHEMA_VERSION),
        ("task_contract", task, TASK_INPUT_SCHEMA_VERSION),
        ("embodiment_contract", embodiment, EMBODIMENT_INPUT_SCHEMA_VERSION),
    ):
        if contract.get("schema_version") != schema:
            blockers.append(f"{label}_schema_invalid")
        if not _is_digest(contract.get("contract_digest")) or contract.get(
            "contract_digest"
        ) != canonical_digest(contract, digest_field="contract_digest"):
            blockers.append(f"{label}_digest_invalid")
        if contract.get("outcome_data_accessed_for_variation_design") is not False:
            blockers.append(f"{label}_outcome_access_invalid")
        if not _is_digest(contract.get("measurement_authority_digest")):
            blockers.append(f"{label}_measurement_authority_digest_invalid")
    dimensions: list[dict[str, Any]] = []
    for source, contract in (
        ("scene", scene),
        ("task", task),
        ("embodiment", embodiment),
    ):
        raw_dimensions = contract.get("variation_dimensions")
        rows = _rows(raw_dimensions)
        if not isinstance(raw_dimensions, list) or len(rows) != len(raw_dimensions):
            blockers.append(f"{source}_variation_dimensions_invalid")
        for row in rows:
            normalized = dict(row)
            if normalized.get("authority_digest") != contract.get(
                "measurement_authority_digest"
            ):
                blockers.append(
                    "dimension_authority_not_source_contract:"
                    f"{_string(row.get('dimension_id')) or source}"
                )
            normalized["source_contract"] = source
            dimensions.append(normalized)
    if blockers:
        raise ExactWorkcellVariationError(blockers)
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "matrix_id": matrix_id,
        "matrix_kind": "exact_workcell_primary",
        "implementation_commit": implementation_commit,
        "cell_count": cell_count,
        "seed_root": seed_root,
        "scene_binding": {
            key: scene.get(key)
            for key in (
                "scene_id",
                "scene_digest",
                "coordinate_frame_digest",
                "canonical_object_asset_id",
                "canonical_object_asset_digest",
            )
        },
        "task_binding": {
            key: task.get(key)
            for key in (
                "task_id",
                "task_digest",
                "reset_contract_digest",
                "success_contract_digest",
            )
        },
        "embodiment_binding": {
            key: embodiment.get(key)
            for key in (
                "embodiment_id",
                "embodiment_digest",
                "joint_limits_digest",
                "camera_calibration_digest",
            )
        },
        "controls": {
            "control_ids": list(REQUIRED_CONTROLS),
            "run_on_every_cell": True,
            "same_resolved_cell_required": True,
        },
        "variation_dimensions": dimensions,
        "object_cousins": [],
    }
    if agent is not None:
        brief = build_agent_proposal_brief(request)
        raw = agent.propose(brief=brief)
        if not isinstance(raw, Mapping):
            raise ExactWorkcellVariationError(["agent_proposal_not_mapping"])
        request["agent_proposal"] = seal_agent_proposal(
            raw, brief=brief, model_identity=agent.model_identity
        )
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    return validate_variation_request(request)


__all__ = ["build_variation_request_from_admitted_contracts"]
