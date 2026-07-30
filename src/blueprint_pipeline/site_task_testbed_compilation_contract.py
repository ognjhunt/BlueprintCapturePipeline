"""Closed WebApp-to-Pipeline contract for authoritative testbed compilation."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError, model_validator


IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$"
DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
IDEMPOTENCY_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,191}$"

Identifier = Annotated[str, Field(pattern=IDENTIFIER_PATTERN)]
Digest = Annotated[str, Field(pattern=DIGEST_PATTERN)]
BoundedText = Annotated[str, Field(min_length=1, max_length=512)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveDistance = Annotated[float, Field(gt=0.0, le=20.0)]


class _ClosedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseFootprint(_ClosedModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {"shape": {"const": "circle"}},
                    "required": ["shape", "radius_m"],
                    "not": {"anyOf": [{"required": ["length_m"]}, {"required": ["width_m"]}]},
                },
                {
                    "properties": {"shape": {"const": "rectangle"}},
                    "required": ["shape", "length_m", "width_m"],
                    "not": {"required": ["radius_m"]},
                },
            ]
        },
    )

    shape: Literal["circle", "rectangle"]
    radius_m: Annotated[float, Field(gt=0.0, le=10.0)] | None = None
    length_m: PositiveDistance | None = None
    width_m: PositiveDistance | None = None

    @model_validator(mode="after")
    def validate_shape_dimensions(self) -> "BaseFootprint":
        if self.shape == "circle":
            if self.radius_m is None or self.length_m is not None or self.width_m is not None:
                raise ValueError("circle requires radius_m and forbids length_m/width_m")
        elif self.radius_m is not None or self.length_m is None or self.width_m is None:
            raise ValueError("rectangle requires length_m/width_m and forbids radius_m")
        return self


class ReachEnvelope(_ClosedModel):
    minimum_m: Annotated[float, Field(ge=0.0, le=20.0)]
    maximum_m: PositiveDistance

    @model_validator(mode="after")
    def validate_range(self) -> "ReachEnvelope":
        if self.maximum_m <= self.minimum_m:
            raise ValueError("maximum_m must exceed minimum_m")
        return self


class RobotBinding(_ClosedModel):
    robot_id: Annotated[str, Field(min_length=1, max_length=128)]
    embodiment_version: Annotated[str, Field(min_length=1, max_length=128)]
    base_footprint: BaseFootprint
    sensors: Annotated[dict[str, BoundedText], Field(min_length=1, max_length=32)]
    controller_id: Annotated[str, Field(min_length=1, max_length=128)]
    end_effector_id: Annotated[str, Field(min_length=1, max_length=128)]
    reach_envelope: ReachEnvelope


class DecisionCandidate(_ClosedModel):
    robot_id: Annotated[str, Field(min_length=1, max_length=128)]
    embodiment_version: Annotated[str, Field(min_length=1, max_length=128)]
    robot_binding: RobotBinding


class MeasurableThreshold(_ClosedModel):
    operator: Annotated[str, Field(min_length=1, max_length=32)]
    value: Any
    units: Annotated[str, Field(min_length=1, max_length=64)]
    metric: Annotated[str, Field(min_length=1, max_length=128)]


class DesiredConfidenceOrCoverage(_ClosedModel):
    minimum_coverage: Annotated[float, Field(gt=0.0, le=1.0)]
    minimum_independent_methods: Annotated[int, Field(ge=1, le=8)]


class PermittedAbstentionBehavior(_ClosedModel):
    allowed: Literal[True]


class SiteDomainConditions(_ClosedModel):
    scope: Literal["accepted_capture_observation"]


class ClaimEmbodiment(_ClosedModel):
    robot_id: Annotated[str, Field(min_length=1, max_length=128)]
    version: Annotated[str, Field(min_length=1, max_length=128)]
    base_footprint: BaseFootprint
    reach_envelope: ReachEnvelope
    end_effector_id: Annotated[str, Field(min_length=1, max_length=128)]


class ControllerActionRepresentation(_ClosedModel):
    controller_id: Annotated[str, Field(min_length=1, max_length=128)]


ClaimType = Literal[
    "perception_visibility",
    "task_discovery",
    "appearance_review",
    "reachability",
    "robot_placement",
    "navigation_clearance",
    "collision_contact",
    "grasp_contact",
    "articulation",
    "containment",
    "mass_inertia",
    "friction_compliance",
    "object_state_transition",
]


class DecisionClaim(_ClosedModel):
    claim_id: Identifier
    claim_type: ClaimType
    subject: BoundedText
    measurable_threshold: MeasurableThreshold
    false_safe_consequence: Literal["low", "moderate", "high", "critical"]
    acceptable_false_safe_risk: UnitInterval
    desired_confidence_or_coverage: DesiredConfidenceOrCoverage
    permitted_abstention_behavior: PermittedAbstentionBehavior
    task_family: Annotated[str, Field(min_length=1, max_length=128)]
    site_domain_conditions: SiteDomainConditions
    embodiment: ClaimEmbodiment
    sensors: Annotated[dict[str, BoundedText], Field(min_length=1, max_length=32)]
    controller_action_representation: ControllerActionRepresentation


class DecisionBudget(_ClosedModel):
    max_cost_usd: Annotated[float, Field(ge=0.0, le=100_000.0)]
    max_latency_seconds: Annotated[float, Field(gt=0.0, le=2_592_000.0)]


class DecisionRestrictions(BaseModel):
    model_config = ConfigDict(extra="allow")

    webapp_provider_selection_allowed: Literal[False]
    live_robot_execution_allowed: Literal[False]
    paid_compute_authorized: Literal[False]

    @model_validator(mode="after")
    def reject_method_selection(self) -> "DecisionRestrictions":
        forbidden = {
            "selected_method",
            "selected_provider",
            "selected_simulator",
            "runtime_provider_profile",
        }.intersection(self.model_extra or {})
        if forbidden:
            raise ValueError(
                "request method selection forbidden: " + ",".join(sorted(forbidden))
            )
        return self


EvidenceMethod = Literal[
    "analytic_geometry_kinematics",
    "captured_real_observation",
    "traditional_simulation",
    "learned_world_model",
    "external_provider_tool",
    "physical_evidence",
    "owner_attested_operational_input",
]


class DecisionRequestConstraints(_ClosedModel):
    request_id: Identifier
    decision_id: Identifier
    candidates: Annotated[list[DecisionCandidate], Field(min_length=1, max_length=16)]
    claims: Annotated[list[DecisionClaim], Field(min_length=1, max_length=16)]
    budget: DecisionBudget
    deadline: AwareDatetime
    permitted_evidence_methods: Annotated[list[EvidenceMethod], Field(min_length=1, max_length=7)]
    restrictions: DecisionRestrictions
    requested_result_audience: Annotated[str, Field(min_length=1, max_length=128)]
    idempotency_key: Annotated[str, Field(pattern=IDEMPOTENCY_PATTERN)]

    @model_validator(mode="after")
    def validate_unique_claims_and_methods(self) -> "DecisionRequestConstraints":
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claim_id values must be unique")
        if len(set(self.permitted_evidence_methods)) != len(self.permitted_evidence_methods):
            raise ValueError("permitted_evidence_methods must be unique")
        return self


class SiteTaskTestbedCompilationSubmissionV2(_ClosedModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "$id": "https://tryblueprint.io/contracts/site-task-testbed-compilation-submission.v2.schema.json"
        },
    )

    schema_version: Literal["site_task_testbed_compilation_submission.v2"]
    capture_session_id: Identifier
    intake_id: Identifier
    testbed_id: Identifier
    version: Identifier
    approved_task_digest: Digest
    reconstruction_plan_id: Identifier
    reconstruction_execution_result_digest: Digest
    robot_binding: RobotBinding
    decision_request_constraints: DecisionRequestConstraints

    @model_validator(mode="after")
    def validate_robot_binding_consistency(self) -> "SiteTaskTestbedCompilationSubmissionV2":
        binding = self.robot_binding
        for candidate in self.decision_request_constraints.candidates:
            if (
                candidate.robot_id != binding.robot_id
                or candidate.embodiment_version != binding.embodiment_version
                or candidate.robot_binding != binding
            ):
                raise ValueError("decision candidate does not match robot_binding")
        for claim in self.decision_request_constraints.claims:
            if (
                claim.embodiment.robot_id != binding.robot_id
                or claim.embodiment.version != binding.embodiment_version
                or claim.embodiment.base_footprint != binding.base_footprint
                or claim.embodiment.reach_envelope != binding.reach_envelope
                or claim.embodiment.end_effector_id != binding.end_effector_id
                or claim.sensors != binding.sensors
                or claim.controller_action_representation.controller_id
                != binding.controller_id
            ):
                raise ValueError("decision claim does not match robot_binding")
        return self


def validate_testbed_compilation_submission(value: Any) -> dict[str, Any]:
    """Validate and normalize one closed v2 compilation submission."""

    try:
        parsed = SiteTaskTestbedCompilationSubmissionV2.model_validate(value)
    except ValidationError as exc:
        errors = []
        for row in exc.errors(include_url=False, include_context=False, include_input=False):
            location = ".".join(str(item) for item in row["loc"])
            prefix = f"submission.{location}" if location else "submission"
            detail = str(row["type"])
            if detail == "value_error":
                detail = str(row.get("msg") or detail).removeprefix("Value error, ")
            errors.append(f"{prefix}:{detail}")
        raise ValueError("; ".join(sorted(set(errors)))) from exc
    normalized = parsed.model_dump(mode="json", exclude_none=True)
    assert isinstance(normalized, dict)
    return normalized


def testbed_compilation_submission_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema mirrored by customer-facing clients."""

    schema = SiteTaskTestbedCompilationSubmissionV2.model_json_schema(
        mode="validation",
        union_format="any_of",
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema
