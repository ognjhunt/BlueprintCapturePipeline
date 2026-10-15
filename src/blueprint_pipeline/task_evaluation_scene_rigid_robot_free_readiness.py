"""Rigid robot-free readiness: score the probe with the exact configured success contract.

Beta journey step: L5 team policy evaluation. Owner decision 2026-10-12:
robots never run in construction, and a configured scene becomes ready for
robot teams' evaluation only after robot-free checks of its success contract.
``task_evaluation_scene_robot_free_readiness`` does that for articulated
open/close; this is the rigid pick/place counterpart, scored through
``score_rigid_task_episode`` and the exact task specification teams are judged
by, one reading per control step.

Two kinds of evidence are sealed together, and only both admit evaluation:

- contract checks (no simulator): robot-free states set directly and held
  still. The untouched start, the object just outside every horizontal edge
  of the success region, tipped (when orientation matters), dropped (when the
  contract forbids drops), released for one sample less than the hold window,
  and drifting through the hold window must all fail; the authored goal must
  pass. These show the contract can tell success from failure at all.
- physics checks (a recorded simulator trace from
  ``rigid_robot_free_probe_recorder``): the scene left untouched must not
  succeed, and the object carried to the goal and let go must succeed once
  physics alone holds it there through the hold window.

Without a recorded trace bound to this revision, task specification and
replacement asset, the receipt is ``probe_unavailable`` and admits nothing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .rigid_robot_free_readiness_score import (
    REQUIRED_CHECKS,
    SCHEMA,
    TASK_KIND,
    score_rigid_robot_free_readiness,
)


def fetch_revision_documents(*, revision: Mapping[str, Any], output_root: str | Path,
                             fetcher: Callable[[Mapping[str, Any]], bytes] | None = None) -> dict[str, Path]:
    """The revision's documents, each verified against its published digest before use."""

    from . import task_evaluation_configured_controls_deferred_inputs as inputs

    root = Path(output_root) / "documents"
    root.mkdir(parents=True, exist_ok=True, mode=0o750)
    documents: dict[str, Path] = {}
    for contract_path, declared in inputs._document_references(revision).items():
        reference = inputs._reference(declared, blocker=f"robot_free_readiness_reference_invalid:{contract_path}")
        path = root / inputs._document_name(contract_path)
        if not inputs._retained_matches(path, reference=reference):
            payload = inputs._fetched(reference, fetcher=fetcher or inputs.default_reference_fetcher)
            inputs._write_immutable_bytes(path, payload, conflict="robot_free_readiness_document_conflict")
        documents[contract_path] = path
    return documents


def rigid_task_for_revision(*, revision: Mapping[str, Any], documents: Mapping[str, Path]) -> dict[str, Any]:
    """Adapt the revision's rigid task exactly as a robot team's evaluation would."""

    from . import task_evaluation_configured_controls_deferred_inputs as inputs
    from .task_evaluation_rigid_relocation_native_adapter import (
        NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
        adapt_rigid_relocation_task_template,
    )

    references = inputs._materialized_references(revision=revision, documents=documents)
    definition = adapt_rigid_relocation_task_template(
        configured_revision=revision, materialized_references=references)["native_task_definition"]
    native_import = json.loads(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH].read_text())
    return {"task_spec": definition["task_spec"], "task_object_pose_world": definition["task_object_pose_world"],
            "replacement_asset_digest": native_import.get("asset_digest"),
            # The evaluation binds a destination body on top of this spec; scoring the
            # point task instead would probe a contract no team is judged by.
            "unsupported_reason": ("destination_body_relation"
                                   if revision["task_template"].get("destination") is not None else None)}


def rigid_robot_free_readiness_for_revision(*, revision: Mapping[str, Any], output_root: str | Path,
                                            physics_probe: Mapping[str, Any] | None,
                                            fetcher: Callable[[Mapping[str, Any]], bytes] | None = None
                                            ) -> dict[str, Any]:
    """Fetch the revision's exact documents, adapt its rigid task, and score."""

    documents = fetch_revision_documents(revision=revision, output_root=output_root, fetcher=fetcher)
    task = rigid_task_for_revision(revision=revision, documents=documents)
    return score_rigid_robot_free_readiness(
        task_spec=task["task_spec"], physics_probe=physics_probe,
        configured_scene_revision_digest=revision.get("revision_digest"),
        replacement_asset_digest=task["replacement_asset_digest"], unsupported_reason=task["unsupported_reason"])


__all__ = [
    "REQUIRED_CHECKS",
    "SCHEMA",
    "TASK_KIND",
    "fetch_revision_documents",
    "rigid_robot_free_readiness_for_revision",
    "rigid_task_for_revision",
    "score_rigid_robot_free_readiness",
]
