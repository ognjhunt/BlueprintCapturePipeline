"""Derived operation routing for beta scene preparation, never owner task truth.

The already required Atlas review reads the entire job, separating scene setup
and operating notes from robot work. Complete operations resolve through the
current registered profiles. No provider or runtime imports here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .website_task_context_contracts import website_task_context_digest
from .task_profile_contracts import TASK_PROFILES, task_profile_for


class OperativeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: str = Field(min_length=1)
    description: str = Field(min_length=1)
    task_quote: str = Field(min_length=1)


class TaskSemantics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operative_job: str = Field(min_length=1)
    task_family: str = Field(min_length=1)
    steps: list[OperativeStep]
    covers_entire_job: bool
    unresolved_aspects: list[str]
    preparation_notes: list[str]


TASK_SEMANTICS_INSTRUCTIONS = """
The website task_family is an ambiguous text-screen hint. Also return
task_semantics: operative_job, task_family, steps, covers_entire_job,
unresolved_aspects and preparation_notes. Read the ENTIRE original task and
success requirements. Describe the actual robot job without rewriting owner
instructions. Separate authored scene setup (for example adding an absent
synthetic object), preservation, budget, source-file and execution notes from
robot operations; retain these under preparation_notes. For each operative step
give operation, description and a verbatim task_quote from the supplied task.
Pick-up plus placement is one rigid_pick_place operation; a different subsequent
operation is another step. Report every operative step, including unknown ones.
Use the registered profile IDs below when their complete operation matches the
job. rigid_pick_place lifts one rigid object and places it on its qualified
destination; articulated_open_close opens one door, drawer or lid, not closing.
The other registered families have their own independent state predicates.
For several operations propose multi_step and retain ALL ordered steps; never
collapse them into one atomic pick/place, folding or other operation. Use an
honest unregistered task_family and unknown operations when no profile matches.
Registration is routing metadata, not a finding that required native physics,
assets, reset, thresholds or runtime inputs exist. Object
movability, a synthetic primitive, and an available adapter do not establish
operation semantics. covers_entire_job is false whenever any requested work or
success requirement is unresolved. This is derived routing evidence, not owner
confirmation, robot capability, measured physics or execution qualification.
""" + "\nRegistered native profiles: " + json.dumps([
    {"profile_id": profile.profile_id, "runtime_adapter": profile.runtime_adapter,
     "required_inputs": list(profile.required_inputs)} for profile in TASK_PROFILES], sort_keys=True) + "\n"

_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def task_semantics_from_receipt(*, receipt: Mapping[str, Any], task_context: Mapping[str, Any],
                                source_video_digest: str) -> dict[str, Any] | None:
    """Validate source-bound model output; legacy completed reviews stay reusable."""
    binding = receipt.get("binding") or {}
    proposal = receipt.get("proposal") or {}
    if "task_semantics" not in proposal:
        return None
    if (task_context.get("context_digest") != website_task_context_digest(task_context)
            or not isinstance(source_video_digest, str) or not _SHA.fullmatch(source_video_digest)
            or receipt.get("status") != "completed"
            or receipt.get("binding_digest") != canonical_digest(binding)
            or receipt.get("proposal_digest") != canonical_digest(proposal)
            or binding.get("task_context_digest") != task_context["context_digest"]
            or binding.get("source_video_digest") != source_video_digest):
        raise ValueError("website_task_semantics_planner_binding_invalid")
    semantics = TaskSemantics.model_validate(proposal["task_semantics"]).model_dump(mode="json")
    def strings(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [text for child in value.values() for text in strings(child)]
        return []
    sources = strings({key: task_context.get(key) for key in (
        "description", "success_criteria", "operator_answers", "operator_task_details")})
    if (proposal.get("status") != "proposed" or any(
            not step["task_quote"].strip() or not any(step["task_quote"] in text for text in sources)
            for step in semantics["steps"])):
        raise ValueError("website_task_semantics_task_quote_invalid")
    # Retain the original completion as well as its identity. A caller cannot
    # substitute a self-hashed family projection for what the planner proposed.
    value = {"schema_version": "website_task_semantics.v1", "claim_ceiling": "development_only",
             "basis": "source_bound_planner_proposal_not_owner_confirmation",
             "binding": {"task_context_digest": task_context["context_digest"],
                         "source_video_digest": source_video_digest,
                         "planner_binding_digest": receipt["binding_digest"],
                         "planner_proposal_digest": receipt["proposal_digest"],
                         "planner_receipt_digest": canonical_digest(receipt)},
             "semantics": semantics, "planner_receipt": dict(receipt)}
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


def validate_task_semantics(value: Mapping[str, Any], *, task_context: Mapping[str, Any],
                            source_video_digest: str) -> dict[str, Any]:
    if "operation_coverage_clarification" in value:
        from .website_operation_coverage import validate_clarified_semantics
        return validate_clarified_semantics(value, task_context=task_context,
                                           source_video_digest=source_video_digest)
    expected = task_semantics_from_receipt(receipt=value.get("planner_receipt") or {},
        task_context=task_context, source_video_digest=source_video_digest)
    if expected is None or value != expected:
        raise ValueError("website_task_semantics_changed")
    return expected


def resolved_task_profile(value: Mapping[str, Any] | None, *, task_context: Mapping[str, Any],
                          source_video_digest: str | None) -> str | None:
    """Only general_task gains a derived routing hint; explicit families win."""
    if value is None or task_context.get("task_family") != "general_task":
        return None
    if source_video_digest is None:
        raise ValueError("website_task_semantics_source_video_missing")
    verified = validate_task_semantics(value, task_context=task_context, source_video_digest=source_video_digest)
    clarification = verified.get("operation_coverage_clarification")
    if clarification is not None:
        coverage = clarification["operation_coverage"]
        if not coverage["operation_coverage_complete"] or coverage["unresolved_operations"]:
            return None
        return _operation_profile(coverage["task_family"], coverage["steps"],
            source_stage_ids=_button_stage_identities(task_context, source_video_digest, coverage["steps"]))
    semantics = verified["semantics"]
    if not semantics["covers_entire_job"] or semantics["unresolved_aspects"]:
        return None
    return _operation_profile(semantics["task_family"], semantics["steps"],
        source_stage_ids=_button_stage_identities(task_context, source_video_digest, semantics["steps"]))


def _button_stage_identities(context, video, steps):
    """Only retained normal fact confirmations resolve repeated button stages.

    The original quotes must occur in their retained order, not be copied to
    manufacture repetitions. IDs/count/order must match the exact source steps.
    This describes a job; it grants no asset/runtime or execution authority.
    """
    if not 2 <= len(steps) <= 8 or any(task_profile_for(step["operation"]) is None
            or task_profile_for(step["operation"]).profile_id != "button_press" for step in steps):
        return None
    facts = (context.get("operator_answers") or {}).get("family_task_facts")
    if not isinstance(facts, Mapping):
        return None
    if "canonicalization" in facts and facts["canonicalization"] != "rfc8785":
        raise ValueError("website_button_authoring_owner_facts_canonicalization_invalid")
    digest = cross_runtime_canonical_digest if facts.get("canonicalization") == "rfc8785" else canonical_digest
    if (not isinstance(facts, Mapping) or facts.get("schema_version") != "website_family_task_facts.v1"
            or facts.get("facts_digest") != digest(facts, digest_field="facts_digest")
            or context.get("confirmed") is not True or facts.get("confirmed") is not True
            or not isinstance(facts.get("confirmation_reference"), str) or not facts["confirmation_reference"].strip()
            or facts.get("capture_id") != context.get("capture_id") or facts.get("source_video_digest") != video
            or facts.get("task_description_digest") != digest({"description": context.get("description")})
            or facts.get("task_success_criteria_digest") != digest({"success_criteria": context.get("success_criteria")})):
        return None
    expected = ["operation-" + str(index).zfill(3) + "-" + canonical_digest(step)[7:15]
                for index, step in enumerate(steps, 1)]
    answers = facts.get("operations")
    if (not isinstance(answers, list) or len(answers) != len(expected)
            or [answer.get("operation_id") if isinstance(answer, Mapping) else None for answer in answers] != expected):
        return None
    position = 0
    for step in steps:
        quote = step["task_quote"]
        found = str(context.get("description", "")).find(quote, position)
        if not quote or found < 0:
            return None
        position = found + len(quote)
    return expected


def _operation_profile(task_family, steps, *, source_stage_ids=None):
    """Routing metadata only; every requested operation must fit one native route."""
    profile = task_profile_for(task_family)
    if profile is None:
        return None
    operations = [task_profile_for(step["operation"]) for step in steps]
    if profile.profile_id == "multi_step":
        # Existing multi-step contracts score their registered family stages;
        # this does not add legacy atomic scorers to that runtime adapter.
        if len(steps) < 2 or any(operation is None or operation.profile_id == "multi_step"
                or operation.runtime_adapter != profile.runtime_adapter for operation in operations):
            return None
        # Family order alone cannot distinguish "wipe desk, then wipe counter"
        # from the reversed target order. Keep that sequence held until a
        # source-bound semantic step→native stage identity is available.
        if len({operation.profile_id for operation in operations}) != len(operations):
            expected = ["operation-" + str(index).zfill(3) + "-" + canonical_digest(step)[7:15]
                        for index, step in enumerate(steps, 1)]
            if (any(operation.profile_id != "button_press" for operation in operations)
                    or source_stage_ids != expected):
                return None
    elif len(steps) != 1 or operations[0] is None or operations[0].profile_id != profile.profile_id:
        return None
    return profile.profile_id


def effective_task_profile(*, task_context: Mapping[str, Any], task_semantics: Mapping[str, Any] | None = None,
                           source_video_digest: str | None = None) -> str | None:
    """Canonical derived routing selection, separate from unchanged owner facts."""
    declared = task_profile_for(task_context.get("task_family"))
    return (declared.profile_id if declared is not None else
            resolved_task_profile(task_semantics, task_context=task_context, source_video_digest=source_video_digest))


def require_semantic_stage_order(*, task_context: Mapping[str, Any], task_semantics: Mapping[str, Any] | None,
                                 source_video_digest: str | None, contract: Mapping[str, Any]) -> None:
    """Website multi-step contracts retain the source job's exact order.

    An explicit routing hint does not replace the retained operation reading.
    Standalone manually authored native contracts use their separate entrypoint.
    """
    if contract.get("task_family") != "multi_step":
        return
    if task_semantics is None or source_video_digest is None:
        raise ValueError("website_task_family_semantic_stages_mismatch")
    verified = validate_task_semantics(task_semantics, task_context=task_context,
                                      source_video_digest=source_video_digest)
    reading = ((verified.get("operation_coverage_clarification") or {}).get("operation_coverage")
               or verified["semantics"])
    complete = reading.get("operation_coverage_complete", reading.get("covers_entire_job"))
    unresolved = reading.get("unresolved_operations", reading.get("unresolved_aspects"))
    identities = _button_stage_identities(task_context, source_video_digest, reading["steps"])
    if not complete or unresolved or _operation_profile(reading["task_family"], reading["steps"],
            source_stage_ids=identities) != "multi_step":
        raise ValueError("website_task_family_semantic_stages_mismatch")
    proposed = [task_profile_for(step["operation"]).profile_id for step in reading["steps"]]
    if proposed != [stage["task_family"] for stage in contract["stages"]]:
        raise ValueError("website_task_family_semantic_stages_mismatch")
    if len(set(proposed)) != len(proposed):
        if identities is None or identities != [stage["stage_id"] for stage in contract["stages"]]:
            raise ValueError("website_task_family_semantic_stages_mismatch")


def retain_task_semantics(*, job_dir: Path, receipt: Mapping[str, Any], task_context: Mapping[str, Any],
                          source_video_digest: str) -> dict[str, Any] | None:
    value = task_semantics_from_receipt(receipt=receipt, task_context=task_context,
                                        source_video_digest=source_video_digest)
    if value is not None:
        from .launch_immutable_input_writer import write_exclusive_private_bytes
        path = job_dir / "task_semantics" / (value["digest"][7:] + ".json")
        write_exclusive_private_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    return value


def load_task_semantics(*, job_dir: Path, task_context: Mapping[str, Any],
                        source_video_digest: str) -> dict[str, Any] | None:
    """The selected completion anchors the artifact; never scan arbitrary plans."""
    selected = job_dir / "selection_binding.json"
    if not selected.is_file():
        return None
    if selected.is_symlink():
        raise ValueError("website_task_semantics_selection_invalid")
    receipt = json.loads(selected.read_text()).get("planner_receipt") or {}
    expected = task_semantics_from_receipt(receipt=receipt, task_context=task_context,
                                          source_video_digest=source_video_digest)
    if expected is None:
        return None
    path = job_dir / "task_semantics" / (expected["digest"][7:] + ".json")
    if path.is_symlink() or not path.is_file():
        raise ValueError("website_task_semantics_artifact_missing")
    value = json.loads(path.read_text())
    if value != expected:
        raise ValueError("website_task_semantics_changed")
    from .website_operation_coverage import load_operation_coverage
    return load_operation_coverage(job_dir=job_dir, parent_semantics=expected,
        task_context=task_context, source_video_digest=source_video_digest) or expected
