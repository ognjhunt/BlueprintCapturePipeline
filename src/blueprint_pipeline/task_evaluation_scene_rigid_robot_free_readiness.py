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
    declared_references = inputs._document_references(revision)
    if revision["task_template"].get("destination") is not None:
        from .rigid_destination_task_binding import BUNDLE_CONTRACT, destination_references
        declared_references.update(destination_references(revision))
        declared_references[BUNDLE_CONTRACT] = revision["configured_scene_bundle"]
    for contract_path, declared in declared_references.items():
        reference = inputs._reference(declared, blocker=f"robot_free_readiness_reference_invalid:{contract_path}")
        path = root / inputs._document_name(contract_path)
        if not inputs._retained_matches(path, reference=reference):
            payload = inputs._fetched(reference, fetcher=fetcher or inputs.default_reference_fetcher)
            inputs._write_immutable_bytes(path, payload, conflict="robot_free_readiness_document_conflict")
        if contract_path == "task.destination.asset":
            header = path.read_bytes()[:16]
            suffix = (".usdz" if header.startswith(b"PK\x03\x04") else ".usdc"
                if header.startswith(b"PXR-USDC") else ".usda" if header.startswith(b"#usda") else None)
            if suffix is None:
                raise ValueError("rigid_destination_usd_format_unrecognized")
            asset_path = path.with_suffix(suffix)
            if not inputs._retained_matches(asset_path, reference=reference):
                inputs._write_immutable_bytes(asset_path, path.read_bytes(), conflict="robot_free_readiness_asset_conflict")
            path = asset_path
        documents[contract_path] = path
    return documents


def scored_rigid_task_spec(*, adapted: Mapping[str, Any], site_id: str, task_id: str,
                           team_namespace: str, task_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Apply the same runtime identity and owner criteria as the episode compiler."""
    from .native_task_subject_identity import runtime_subject_task_spec
    from .task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract

    spec = dict(task_spec if task_spec is not None else adapted["native_task_definition"]["task_spec"])
    spec["success_criteria"] = adapted["native_success_criteria"]["criteria"]
    spec = runtime_subject_task_spec(spec)
    contract = _derive_configured_owner_success_contract(
        spec, site_id=site_id, task_id=task_id, team_namespace=team_namespace)
    if contract is not None:
        spec["task_success_contract"] = contract
        spec["task_success_contract_digest"] = contract["contract_digest"]
    else:
        from .adp_articulated_task_success_contract import seal_task_success_contract
        spec["task_success_contract"] = seal_task_success_contract(
            task_kind="rigid_pick_place", task_spec=spec, site_id=site_id, task_id=task_id,
            author_source="compatibility_default", author_id="blueprint:manipulation_strategy_defaults.v1",
            confirmation_status="confirmed")
    return spec


def rigid_task_for_revision(*, revision: Mapping[str, Any], documents: Mapping[str, Path]) -> dict[str, Any]:
    """Adapt the revision's rigid task exactly as a robot team's evaluation would."""

    from . import task_evaluation_configured_controls_deferred_inputs as inputs
    from .task_evaluation_rigid_relocation_native_adapter import (
        DEFINITION_CONTRACT_PATH,
        NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH,
        adapt_rigid_relocation_task_template,
    )

    references = inputs._materialized_references(revision=revision, documents=documents)
    adapted = adapt_rigid_relocation_task_template(
        configured_revision=revision, materialized_references=references)
    definition = adapted["native_task_definition"]
    destination_probe = None
    bound_spec = definition["task_spec"]
    if revision["task_template"].get("destination") is not None:
        import tempfile
        from .rigid_destination_task_binding import (
            BUNDLE_CONTRACT, bind_destination_task_spec, probe_destination_metadata, stage_revision_destination,
        )
        from .task_evaluation_native_arena_episode_compiler import _extract_configured_assets, _sha256_and_size
        from .task_evaluation_rigid_relocation_native_adapter import SUPPORT_PLANE_CONTRACT_PATH

        # Native staging reuses its existing complete rights, geometry and
        # physics validation; scratch extraction cannot replace published bytes.
        parent = Path(documents[DEFINITION_CONTRACT_PATH]).parent
        with tempfile.TemporaryDirectory(prefix="destination-probe-", dir=parent) as temporary:
            root = Path(temporary)
            assets = _extract_configured_assets(Path(documents[BUNDLE_CONTRACT]), output_root=root)
            destination = stage_revision_destination(revision=revision, documents=documents, task_spec=bound_spec,
                collision_path=assets["collision"], output_root=root, require_placement=False)
            assert destination is not None
            support = json.loads(Path(documents[SUPPORT_PLANE_CONTRACT_PATH]).read_text())
            bound_spec = bind_destination_task_spec(bound_spec, destination, source_support=support,
                support_plane_digest=_sha256_and_size(Path(documents[SUPPORT_PLANE_CONTRACT_PATH]))[0])
            destination_probe = probe_destination_metadata(destination, documents)
    spec = scored_rigid_task_spec(adapted=adapted, site_id=revision["scene_identity"]["id"],
        task_id=revision["task_template"]["identity"]["id"], team_namespace=revision["team_namespace"],
        task_spec=bound_spec)
    native_import = json.loads(documents[NATIVE_IMPORT_QUALIFICATION_CONTRACT_PATH].read_text())
    return {"task_spec": spec, "task_object_pose_world": definition["task_object_pose_world"],
            "replacement_asset_digest": native_import.get("asset_digest"),
            "unsupported_reason": None,
            **({"destination_probe": destination_probe} if destination_probe is not None else {})}


def rigid_robot_free_readiness_for_revision(*, revision: Mapping[str, Any], output_root: str | Path,
                                            physics_probe: Mapping[str, Any] | None,
                                            fetcher: Callable[[Mapping[str, Any]], bytes] | None = None
                                            ) -> dict[str, Any]:
    """Fetch the revision's exact documents, adapt its rigid task, and score."""

    documents = fetch_revision_documents(revision=revision, output_root=output_root, fetcher=fetcher)
    task = rigid_task_for_revision(revision=revision, documents=documents)
    if physics_probe is None:
        from .rigid_robot_free_construction_probe import published_construction_probe
        physics_probe = published_construction_probe(revision=revision, documents=documents, task=task)
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
