"""Runtime-neutral contracts for the ADP-009D autonomous workflow."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Literal, Mapping

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


RUNTIME_SDK = "openai_agents_sdk"
RUNTIME_API = "openai_agents_api"
TASK_SCHEMA = "blueprint_agent_task.v1"
RESULT_SCHEMA = "blueprint_agent_task_result.v1"
IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$"
DIGEST = r"^sha256:[0-9a-f]{64}$"
Effect = Literal["read_only", "idempotent_write", "external_side_effect"]
ToolOutput = Mapping[str, Any] | list[dict[str, Any]]


def canonical_json(value: Any) -> str:
    """Reject non-JSON and non-finite values before persistence or disclosure."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class AgentExecutionError(RuntimeError):
    """A stable refusal code, never an upstream exception body or secret."""


class AgentAdmission(BaseModel):
    """Trusted server-issued authority, kept outside the model's input.

    The owning service validates this against its authority store. A model may
    never create or amend this object. Accepted managed-budget uncertainty does
    not authorize external tools; their existing authority checks still apply.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    authority_digest: str = Field(pattern=DIGEST)
    authority_reference: str = Field(min_length=1, max_length=512)
    project_id: str = Field(pattern=IDENTIFIER)
    runtime: Literal["openai_agents_sdk", "openai_agents_api"]
    disclosure_scope: str = Field(min_length=1, max_length=192)
    allowed_input_digests: tuple[str, ...]
    allowed_tool_ids: tuple[str, ...] = ()
    session_retention: Literal["not_admitted", "until_deleted"] = "not_admitted"
    trace_retention: Literal["not_admitted", "provider_default"] = "not_admitted"
    region: Literal["us", "default"] = "default"
    budget_policy: Literal["strict_per_call", "project_guard_accepted_uncertainty"]
    inference_budget_usd: float = Field(gt=0, allow_inf_nan=False)
    project_guard_receipt_digest: str | None = Field(default=None, pattern=DIGEST)
    expires_at: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _managed_compatibility(self) -> AgentAdmission:
        if len(set(self.allowed_tool_ids)) != len(self.allowed_tool_ids):
            raise ValueError("agent_admission_duplicate_tools")
        if len(set(self.allowed_input_digests)) != len(self.allowed_input_digests):
            raise ValueError("agent_admission_duplicate_input_digests")
        import re

        if any(re.fullmatch(DIGEST, value) is None for value in self.allowed_input_digests):
            raise ValueError("agent_admission_invalid_input_digest")
        if self.runtime == RUNTIME_API and (
            self.session_retention != "until_deleted"
            or self.trace_retention != "provider_default"
            or self.region != "us"
            or self.budget_policy != "project_guard_accepted_uncertainty"
            or self.project_guard_receipt_digest is None
        ):
            raise ValueError("agents_api_retention_or_budget_policy_not_admitted")
        return self


class AgentTask(BaseModel):
    """One immutable task revision with a stable product operation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    _sealed_task_digest: str = PrivateAttr(default="")

    schema_version: Literal["blueprint_agent_task.v1"] = TASK_SCHEMA
    task_id: str = Field(pattern=IDENTIFIER)
    parent_task_id: str | None = Field(default=None, pattern=IDENTIFIER)
    run_id: str = Field(pattern=IDENTIFIER)
    capability: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    context_revision: str = Field(pattern=DIGEST)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    instructions: str = Field(min_length=1, max_length=100_000)
    model: str = Field(min_length=1, max_length=192)
    reasoning_effort: str = Field(default="medium", min_length=1, max_length=32)
    input: list[dict[str, Any]] = Field(min_length=1, max_length=256)
    input_digests: tuple[str, ...]
    output_schema: dict[str, Any]
    tool_ids: tuple[str, ...] = ()
    tool_digests: dict[str, str] = Field(default_factory=dict)
    admission: AgentAdmission
    max_tool_calls: int = Field(default=24, ge=0, le=256)
    max_model_turns: int = Field(default=4, ge=1, le=12)
    max_input_tokens: int = Field(default=120_000, ge=1, le=1_000_000)
    max_output_tokens: int = Field(default=4_000, ge=256, le=32_000)
    max_tool_output_bytes: int = Field(default=2_000_000, ge=1, le=128_000_000)
    max_output_bytes: int = Field(default=200_000, ge=1, le=2_000_000)
    deadline: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _scope(self) -> AgentTask:
        if self.parent_task_id == self.task_id:
            raise ValueError("agent_task_cannot_continue_itself")
        canonical_json(self.input)
        Draft202012Validator.check_schema(self.output_schema)
        if self.output_schema.get("type") != "object":
            raise ValueError("agent_output_schema_must_be_object")
        if not set(self.input_digests) <= set(self.admission.allowed_input_digests):
            raise ValueError("agent_input_disclosure_not_admitted")
        # Bind the actual complete payload as well as the source-artifact ids.
        if digest(self.input) not in self.admission.allowed_input_digests:
            raise ValueError("agent_input_payload_not_admitted")
        if len(set(self.tool_ids)) != len(self.tool_ids):
            raise ValueError("agent_task_duplicate_tools")
        if not set(self.tool_ids) <= set(self.admission.allowed_tool_ids):
            raise ValueError("agent_tools_not_admitted")
        if set(self.tool_digests) != set(self.tool_ids):
            raise ValueError("agent_tool_identity_inventory_incomplete")
        import re

        if any(re.fullmatch(DIGEST, value) is None for value in self.tool_digests.values()):
            raise ValueError("agent_tool_identity_digest_invalid")
        if self.deadline > self.admission.expires_at:
            raise ValueError("agent_deadline_exceeds_authority")
        self._sealed_task_digest = digest(self.model_dump(mode="json"))
        return self

    @property
    def task_digest(self) -> str:
        return digest(self.model_dump(mode="json"))

    def snapshot(self) -> AgentTask:
        """Revalidate a detached copy; Pydantic's frozen setting is shallow."""

        document = canonical_json(self.model_dump(mode="json"))
        if digest(json.loads(document)) != self._sealed_task_digest:
            raise AgentExecutionError("agent_task_mutated_after_validation")
        return AgentTask.model_validate_json(document)

    def validate_output(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise AgentExecutionError("agent_output_not_object")
        if len(canonical_json(value).encode("utf-8")) > self.max_output_bytes:
            raise AgentExecutionError("agent_output_too_large")
        if not Draft202012Validator(self.output_schema).is_valid(value):
            raise AgentExecutionError("agent_output_schema_invalid")
        return value


@dataclass(frozen=True)
class ToolContext:
    run_id: str
    task_id: str
    context_revision: str
    operation_id: str
    authority_digest: str
    deadline: float
    allowed_input_digests: tuple[str, ...] = ()
    reconciliation_only: bool = False


@dataclass(frozen=True)
class ToolReconciliation:
    """Only an authoritative positive ``not_started`` permits re-execution."""

    status: Literal["completed", "not_started", "pending"]
    output: ToolOutput | None = None


@dataclass(frozen=True)
class AgentTool:
    tool_id: str
    version: str
    description: str
    input_schema: Mapping[str, Any]
    effect: Effect
    invoke: Callable[[Mapping[str, Any], ToolContext], ToolOutput]
    reconcile: Callable[[Mapping[str, Any], ToolContext], ToolReconciliation] | None = None

    def __post_init__(self) -> None:
        import re

        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.tool_id) is None:
            raise ValueError("agent_tool_id_invalid")
        if not self.version or not self.description:
            raise ValueError("agent_tool_metadata_missing")
        if self.effect not in {"read_only", "idempotent_write", "external_side_effect"}:
            raise ValueError("agent_tool_effect_invalid")
        Draft202012Validator.check_schema(self.input_schema)
        if self.input_schema.get("type") != "object":
            raise ValueError("agent_tool_schema_must_be_object")

    @property
    def tool_digest(self) -> str:
        return digest({
            "tool_id": self.tool_id,
            "version": self.version,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "effect": self.effect,
        })

    def validate_arguments(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict) or not Draft202012Validator(
            self.input_schema
        ).is_valid(arguments):
            raise AgentExecutionError("agent_tool_arguments_invalid")
        canonical_json(arguments)
        return arguments
