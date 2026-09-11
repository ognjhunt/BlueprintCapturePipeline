"""Adapt the existing capability registry without replacing its controllers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

from ..task_evaluation_supervisor.capabilities import SupervisorContext
from ..task_evaluation_supervisor.contracts import AuthorityEnvelope
from ..task_evaluation_supervisor.tools import (
    ToolRegistry, non_spend_tool_bindings, validate_tool_observation_binding,
)
from .contracts import AgentExecutionError, AgentTool, ToolContext, ToolReconciliation, canonical_json, digest
from .journal import AgentJournal


@dataclass(frozen=True)
class BoundSupervisorService:
    """Server-bound implementation and configuration of an injected executor.

    The admission service supplies the immutable release/profile identities;
    these values never come from model arguments. An anonymous callable cannot
    be substituted for a previously bound executor.
    """
    source_commit: str
    implementation_digest: str
    configuration_digest: str
    service: Any = field(repr=False, compare=False)

    def __post_init__(self):
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in (self.implementation_digest, self.configuration_digest)
        ):
            raise ValueError("agent_supervisor_service_identity_invalid")

    @property
    def identity_digest(self) -> str:
        return digest({"source_commit": self.source_commit, "implementation_digest": self.implementation_digest,
                       "configuration_digest": self.configuration_digest})

    def __call__(self, *args, **kwargs):
        return self.service(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.service, name)


def context_revision(context: SupervisorContext) -> str:
    """Hash serializable context truth; injected service handles are not inputs."""
    payload = {}
    for context_field in fields(context):
        value = getattr(context, context_field.name)
        if context_field.name == "supervisor_output_dir":
            payload[context_field.name] = digest(str(Path(value).expanduser().resolve())) if value else None
            continue
        if isinstance(value, BoundSupervisorService):
            payload[context_field.name] = {"executor_identity_digest": value.identity_digest}
            continue
        if callable(value) or (context_field.name == "recovery_controller" and value is not None):
            raise AgentExecutionError("agent_supervisor_executor_identity_missing")
        payload[context_field.name] = value
    return digest(payload)


class SupervisorCapabilityBridge:
    def __init__(
        self, *, capability: str, registry: ToolRegistry, journal: AgentJournal,
        load_context: Callable[[str], SupervisorContext],
        reconcilers: Mapping[str, Callable[[Mapping, ToolContext], ToolReconciliation]] | None = None,
    ) -> None:
        self.capability, self.registry, self.journal = capability, registry, journal
        self.load_context = load_context
        self.reconcilers = dict(reconcilers or {})
        self._registry_digest = registry.digest

    def _scope(self, context: ToolContext, arguments: Mapping):
        current = self.load_context(context.run_id)
        if not isinstance(current, SupervisorContext) or current.run_id != context.run_id:
            raise AgentExecutionError("agent_supervisor_context_run_mismatch")
        if (self.registry.digest != self._registry_digest
                or context_revision(current) != context.context_revision
                or arguments["context_revision"] != context.context_revision):
            raise AgentExecutionError("agent_supervisor_context_revision_stale")
        authority = AuthorityEnvelope.from_mapping(current.authority_envelope).to_mapping()
        if authority["authority_digest"] != context.authority_digest:
            raise AgentExecutionError("agent_supervisor_authority_changed")
        if not set(authority["immutable_input_digests"]) <= set(context.allowed_input_digests):
            raise AgentExecutionError("agent_supervisor_disclosure_scope_mismatch")
        return current, authority

    def tools(self) -> tuple[AgentTool, ...]:
        """Expose capability-scoped descriptors; execution availability stays live."""
        result = []
        for tool_id in self.registry.allowed_tool_ids_for_capability(self.capability):
            descriptor = self.registry.resolve(tool_id).to_mapping()
            effect = {"read_only": "read_only", "reversible_mutation": "idempotent_write",
                      "external_side_effect": "external_side_effect"}[descriptor["mutability"]]
            if effect == "external_side_effect" and tool_id not in self.reconcilers:
                continue  # Authoritative recovery inspection is required first.

            def invoke(arguments, context, selected=tool_id):
                current, authority = self._scope(context, arguments)
                bindings = {binding.tool_id: binding for binding in non_spend_tool_bindings(
                    capability=self.capability, context=current, registry=self.registry, authority=authority,
                )}
                binding = bindings.get(selected)
                if binding is None:
                    return {"status": "unavailable", "tool_id": selected,
                            "reason": "required_bound_input_or_runtime_missing", "proof_effect": "none"}
                observation = validate_tool_observation_binding(
                    binding.invoke(arguments["arguments"]), run_id=context.run_id,
                    capability=self.capability, registry=self.registry, authority=authority,
                )
                if (context_revision(current) != context.context_revision
                        or context_revision(self.load_context(context.run_id)) != context.context_revision):
                    raise AgentExecutionError("agent_supervisor_context_mutated_during_execution")
                saved = {
                    "context_revision": context.context_revision, "operation_id": context.operation_id,
                    "arguments_digest": digest(arguments), "observation": observation,
                }
                self.journal.record_event("registry_observation_" + context.operation_id[7:], saved)
                return observation

            def reconcile(arguments, context, selected=tool_id, selected_effect=effect):
                saved = self.journal.event("registry_observation_" + context.operation_id[7:])
                if saved is not None:
                    if (saved["operation_id"] != context.operation_id
                            or saved["context_revision"] != context.context_revision
                            or saved["arguments_digest"] != digest(arguments)):
                        raise AgentExecutionError("agent_supervisor_observation_binding_invalid")
                    return ToolReconciliation("completed", saved["observation"])
                external = self.reconcilers.get(selected)
                if external is not None:
                    return external(arguments["arguments"], context)
                if selected_effect == "read_only":
                    return ToolReconciliation("not_started")
                # Registry idempotence is useful documentation, but an absent
                # result file does not prove a materializer never started.
                return ToolReconciliation("pending")

            schema = {"type": "object", "properties": {
                "context_revision": {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"},
                "arguments": json.loads(canonical_json(descriptor["input_schema"])),
            }, "required": ["context_revision", "arguments"], "additionalProperties": False}
            result.append(AgentTool(
                tool_id, descriptor["version"],
                "Invoke the existing Blueprint " + tool_id + " capability. Re-read current run state "
                "and supply its exact context_revision. Deterministic controller observation; "
                "no scientific acceptance. Descriptor: " + descriptor["tool_digest"],
                schema, effect, invoke, reconcile,
            ))
        return tuple(result)
