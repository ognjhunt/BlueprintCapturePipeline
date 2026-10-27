"""Derived operation routing for beta scene preparation, never owner task truth.

The already required Atlas review reads the entire job, separating scene setup
and operating notes from robot work. Only a complete single supported operation
can resolve an ambiguous text-screen hint. No provider or runtime imports here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .decision_evidence_contracts import canonical_digest
from .task_profile_contracts import task_profile_for


class OperativeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["rigid_pick_place", "articulated_open_close", "unsupported", "unknown"]
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
operation is another step. Report every operative step, including unsupported
ones. Supported operations are rigid_pick_place (one rigid object lifted and
placed ON a supported destination surface) and articulated_open_close (opening
one door, drawer or lid). Washing, folding, closing, pushing, inside-container
placement and other jobs are unsupported; never collapse them into pick/place.
Use unsupported or unknown for such steps and an honest task_family. Object
movability, a synthetic primitive, and an available adapter do not establish
operation semantics. covers_entire_job is false whenever any requested work or
success requirement is unresolved. This is derived routing evidence, not owner
confirmation, robot capability, measured physics or execution qualification.
"""

_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")


def task_semantics_from_receipt(*, receipt: Mapping[str, Any], task_context: Mapping[str, Any],
                                source_video_digest: str) -> dict[str, Any] | None:
    """Validate source-bound model output; legacy completed reviews stay reusable."""
    binding = receipt.get("binding") or {}
    proposal = receipt.get("proposal") or {}
    if "task_semantics" not in proposal:
        return None
    if (task_context.get("context_digest") != canonical_digest(task_context, digest_field="context_digest")
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
    semantics = verified["semantics"]
    profile = task_profile_for(semantics["task_family"])
    steps = semantics["steps"]
    if (not semantics["covers_entire_job"] or semantics["unresolved_aspects"] or len(steps) != 1
            or profile is None or steps[0]["operation"] != profile.profile_id):
        return None
    return profile.profile_id


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
    return expected
