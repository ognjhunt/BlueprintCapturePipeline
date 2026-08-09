"""Compile one resumable articulated public-scene construction run.

This is the non-agent production seam for the second-scene rehearsal.  It
joins the same frozen scene, registered components, render-derived removal
inputs, released inpainting adapter, articulated source USD, and Joint Agent
packet used during an interactive construction session.  It never converts a
prepared packet into an executed result and it stops before simulator or paid
work when any upstream gate is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .public_scene_execution_authority import (
    validate_public_scene_execution_authority,
)
from .second_scene_freeze import validate_second_scene_freeze


SCHEMA_VERSION = "articulated_public_scene_construction_run.v1"
PROGRAM_ID = "arm-decision-proof-v1"


class ArticulatedPublicScenePipelineError(ValueError):
    """Stable, sorted construction-run validation failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _clone(value: Mapping[str, Any], error: str) -> dict[str, Any]:
    try:
        cloned = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ArticulatedPublicScenePipelineError([error]) from exc
    if not isinstance(cloned, dict):
        raise ArticulatedPublicScenePipelineError([error])
    return cloned


def _require_digest(
    value: Mapping[str, Any], *, field: str, schema: str, error_prefix: str
) -> dict[str, Any]:
    payload = _clone(value, f"{error_prefix}_invalid")
    errors: list[str] = []
    if payload.get("schema_version") != schema:
        errors.append(f"{error_prefix}_schema_invalid")
    if payload.get(field) != canonical_digest(payload, digest_field=field):
        errors.append(f"{error_prefix}_digest_invalid")
    if errors:
        raise ArticulatedPublicScenePipelineError(errors)
    return payload


def _scene_target(value: Mapping[str, Any]) -> tuple[str, str]:
    scene = value.get("scene")
    if not isinstance(scene, Mapping):
        return "", ""
    return (
        str(scene.get("publisher_scene_id") or ""),
        str(scene.get("target_instance_id") or ""),
    )


def _canonical_instance_id(value: Any) -> str:
    """Normalize the dataset ID and method-level ``ins<ID>`` representation."""

    text = str(value or "")
    if text.startswith("ins"):
        text = text[3:]
    return text if text.isdigit() else ""


def _same_instance_id(left: Any, right: Any) -> bool:
    canonical_left = _canonical_instance_id(left)
    return bool(canonical_left and canonical_left == _canonical_instance_id(right))


def _component_pair(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any], *, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_payload = _require_digest(
        manifest,
        field="manifest_digest",
        schema="public_scene_component_manifest.v1",
        error_prefix=f"{role}_manifest",
    )
    receipt_payload = _require_digest(
        receipt,
        field="receipt_digest",
        schema="public_scene_component_admission_receipt.v1",
        error_prefix=f"{role}_receipt",
    )
    errors: list[str] = []
    if manifest_payload.get("role") != role:
        errors.append(f"{role}_manifest_role_invalid")
    if receipt_payload.get("role") != role:
        errors.append(f"{role}_receipt_role_invalid")
    if receipt_payload.get("status") != "admitted":
        errors.append(f"{role}_not_admitted")
    if receipt_payload.get("component_manifest_digest") != manifest_payload.get(
        "manifest_digest"
    ):
        errors.append(f"{role}_component_join_invalid")
    if errors:
        raise ArticulatedPublicScenePipelineError(errors)
    return manifest_payload, receipt_payload


def compile_articulated_public_scene_state(
    *,
    freeze: Mapping[str, Any],
    appearance_manifest: Mapping[str, Any],
    appearance_receipt: Mapping[str, Any],
    collision_manifest: Mapping[str, Any],
    collision_receipt: Mapping[str, Any],
    inpainting_input_receipt: Mapping[str, Any],
    aura_adapter_receipt: Mapping[str, Any],
    articulated_source_receipt: Mapping[str, Any],
    joint_agent_packet: Mapping[str, Any],
    repository_commit: str,
    execution_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join validated construction artifacts and choose the next finite gate."""

    frozen = _clone(freeze, "freeze_invalid")
    appearance, appearance_admission = _component_pair(
        appearance_manifest,
        appearance_receipt,
        role="interiorgs_appearance_scene",
    )
    collision, collision_admission = _component_pair(
        collision_manifest,
        collision_receipt,
        role="sage3d_collision_companion",
    )
    inpainting = _require_digest(
        inpainting_input_receipt,
        field="receipt_digest",
        schema="adp009b_interiorgs_edit_input_receipt.v1",
        error_prefix="inpainting_input_receipt",
    )
    aura = _require_digest(
        aura_adapter_receipt,
        field="receipt_digest",
        schema="adp009b_aurafusion360_adapter_receipt.v1",
        error_prefix="aura_adapter_receipt",
    )
    source = _require_digest(
        articulated_source_receipt,
        field="receipt_digest",
        schema="articulated_source_asset.v1",
        error_prefix="articulated_source_receipt",
    )
    joint = _require_digest(
        joint_agent_packet,
        field="packet_digest",
        schema="usd_content_joint_agent_packet.v1",
        error_prefix="joint_agent_packet",
    )
    authority = (
        validate_public_scene_execution_authority(execution_authority)
        if execution_authority is not None
        else None
    )

    scene_id, target_id = _scene_target(frozen)
    errors: list[str] = []
    if not scene_id or not target_id:
        errors.append("freeze_scene_target_missing")
    expected_freeze_digest = canonical_digest(frozen, digest_field="freeze_digest")
    if frozen.get("freeze_digest") != expected_freeze_digest:
        errors.append("freeze_digest_invalid")
    if frozen.get("candidate_ids") != ["pi05_droid", "groot_n17_droid"]:
        errors.append("frozen_candidate_pair_invalid")
    if frozen.get("learned_policy_outcomes_consulted") is not False:
        errors.append("learned_outcome_leakage")
    for name, component in (("appearance", appearance), ("collision", collision)):
        mapping = component.get("scene_mapping")
        target = component.get("target_binding")
        if not isinstance(mapping, Mapping) or str(
            mapping.get("publisher_scene_id") or ""
        ) != scene_id:
            errors.append(f"{name}_scene_join_invalid")
        if not isinstance(target, Mapping) or not _same_instance_id(
            target.get("interiorgs_instance_id"), target_id
        ):
            errors.append(f"{name}_target_join_invalid")
    for name, artifact in (("inpainting", inpainting), ("aura", aura)):
        observed_scene, observed_target = _scene_target(artifact)
        if observed_scene != scene_id:
            errors.append(f"{name}_scene_join_invalid")
        if not _same_instance_id(observed_target, target_id):
            errors.append(f"{name}_target_join_invalid")
    if aura.get("scene", {}).get("input_receipt_digest") != inpainting.get(
        "receipt_digest"
    ):
        errors.append("aura_inpainting_input_join_invalid")
    source_target = source.get("target")
    if not isinstance(source_target, Mapping) or not _same_instance_id(
        source_target.get("interiorgs_instance_id"), target_id
    ):
        errors.append("articulated_source_target_join_invalid")
    source_asset = source.get("output_asset")
    joint_source = joint.get("source_asset")
    if not isinstance(source_asset, Mapping) or not isinstance(joint_source, Mapping):
        errors.append("joint_agent_source_join_missing")
    else:
        if joint_source.get("source_receipt_digest") != source.get("receipt_digest"):
            errors.append("joint_agent_source_receipt_join_invalid")
        if joint_source.get("sha256") != source_asset.get("sha256"):
            errors.append("joint_agent_source_asset_join_invalid")
        if joint_source.get("connected_component_count") != source.get(
            "connected_component_count"
        ):
            errors.append("joint_agent_component_count_join_invalid")
    if len(repository_commit) != 40 or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        errors.append("repository_commit_invalid")
    if authority is not None:
        if authority.get("publisher_scene_id") != scene_id:
            errors.append("execution_authority_scene_join_invalid")
        if not _same_instance_id(authority.get("target_instance_id"), target_id):
            errors.append("execution_authority_target_join_invalid")
        if authority.get("freeze_digest") != frozen.get("freeze_digest"):
            errors.append("execution_authority_freeze_join_invalid")
        if authority.get("aura_adapter_receipt_digest") != aura.get("receipt_digest"):
            errors.append("execution_authority_aura_join_invalid")
        if authority.get("joint_agent_source_receipt_digest") != source.get(
            "receipt_digest"
        ):
            errors.append("execution_authority_joint_source_receipt_join_invalid")
        if authority.get("joint_agent_source_asset_digest") != source_asset.get(
            "sha256"
        ):
            errors.append("execution_authority_joint_source_asset_join_invalid")
    if errors:
        raise ArticulatedPublicScenePipelineError(errors)

    rights = frozen.get("rights") if isinstance(frozen.get("rights"), Mapping) else {}
    blockers: list[str] = []
    if authority is None and rights.get("external_provider_upload_authorized") is not True:
        blockers.append("external_scene_derived_byte_disclosure_authority_missing")
    joint_admission = (
        joint.get("execution_admission")
        if isinstance(joint.get("execution_admission"), Mapping)
        else {}
    )
    if authority is None and joint_admission.get("paid_execution_authorized") is not True:
        blockers.append("fresh_paid_joint_agent_execution_authority_missing")
    aura_execution = aura.get("execution") if isinstance(aura.get("execution"), Mapping) else {}
    if aura_execution.get("aurafusion360_interiorgs_executed") is not True:
        blockers.append("released_code_inpainting_execution_missing")
    if joint_admission.get("remote_execution_performed") is not True:
        blockers.append("joint_agent_topology_execution_missing")
    blockers = list(dict.fromkeys(blockers))
    upstream_ready = not blockers

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "program_id": PROGRAM_ID,
        "status": (
            "construction_inputs_admitted_for_simready_build"
            if upstream_ready
            else "typed_abstention_before_simready_build"
        ),
        "scene": {
            "publisher_scene_id": scene_id,
            "target_instance_id": target_id,
            "task_kind": frozen.get("task_spec", {}).get("task_kind"),
            "freeze_digest": frozen["freeze_digest"],
        },
        "repository": {"commit": repository_commit},
        "execution_authority": (
            {
                "authorization_digest": authority["authorization_digest"],
                "hard_total_spend_cap_usd": authority["hard_total_spend_cap_usd"],
                "maximum_single_resource_ttl_seconds": authority[
                    "maximum_single_resource_ttl_seconds"
                ],
            }
            if authority is not None
            else None
        ),
        "frozen_candidates": list(frozen["candidate_ids"]),
        "stage_receipts": {
            "appearance_component": appearance_admission["receipt_digest"],
            "collision_component": collision_admission["receipt_digest"],
            "inpainting_inputs": inpainting["receipt_digest"],
            "inpainting_adapter": aura["receipt_digest"],
            "articulated_source_asset": source["receipt_digest"],
            "joint_agent_packet": joint["packet_digest"],
        },
        "stage_status": {
            "scene_and_task_frozen": True,
            "registered_appearance_and_collision_admitted": True,
            "evaluation_authorized_removal_inputs_materialized": True,
            "released_code_inpainting_executed": bool(
                aura_execution.get("aurafusion360_interiorgs_executed")
            ),
            "joint_agent_topology_executed": bool(
                joint_admission.get("remote_execution_performed")
            ),
            "simready_replacement_materialized": False,
            "native_simulator_qualified": False,
            "controls_executed": False,
            "learned_candidates_executed": False,
        },
        "blockers": blockers,
        "smallest_blocker": blockers[0] if blockers else None,
        "next_action": (
            "obtain explicit dataset disclosure authority for the exact retained scene-derived Aura and Joint Agent inputs"
            if blockers and blockers[0] == "external_scene_derived_byte_disclosure_authority_missing"
            else "materialize and independently qualify the articulated SimReady replacement"
        ),
        "claim_boundary": {
            "same_non_agent_stage_contracts_as_interactive_rehearsal": True,
            "prepared_packet_is_not_execution": True,
            "generated_geometry_is_not_physical_truth": True,
            "simulator_evidence_is_not_physical_evidence": True,
            "partner_capture_or_deployment_claimed": False,
        },
        "run_digest": "",
    }
    result["run_digest"] = canonical_digest(result, digest_field="run_digest")
    return result


def _load(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArticulatedPublicScenePipelineError([f"json_input_invalid:{path}"]) from exc
    if not isinstance(value, dict):
        raise ArticulatedPublicScenePipelineError([f"json_input_invalid:{path}"])
    return value


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArticulatedPublicScenePipelineError(["repository_commit_unavailable"]) from exc


def compile_articulated_public_scene_run(
    *,
    freeze_path: str | Path,
    repo_root: str | Path,
    data_root: str | Path,
    components_root: str | Path,
    inpainting_input_receipt_path: str | Path,
    aura_adapter_receipt_path: str | Path,
    articulated_source_receipt_path: str | Path,
    joint_agent_packet_path: str | Path,
    execution_authority_path: str | Path | None,
    output_path: str | Path,
) -> dict[str, Any]:
    """Filesystem entrypoint used by local and GPU control planes."""

    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    components = Path(components_root).expanduser().resolve()
    frozen = validate_second_scene_freeze(
        _load(freeze_path), repo_root=repo, data_root=data, verify_files=True
    )
    result = compile_articulated_public_scene_state(
        freeze=frozen,
        appearance_manifest=_load(
            components / "interiorgs_appearance_scene.component_manifest.json"
        ),
        appearance_receipt=_load(
            components / "interiorgs_appearance_scene.component_receipt.json"
        ),
        collision_manifest=_load(
            components / "sage3d_collision_companion.component_manifest.json"
        ),
        collision_receipt=_load(
            components / "sage3d_collision_companion.component_receipt.json"
        ),
        inpainting_input_receipt=_load(inpainting_input_receipt_path),
        aura_adapter_receipt=_load(aura_adapter_receipt_path),
        articulated_source_receipt=_load(articulated_source_receipt_path),
        joint_agent_packet=_load(joint_agent_packet_path),
        repository_commit=_git_commit(repo),
        execution_authority=(
            _load(execution_authority_path)
            if execution_authority_path is not None
            else None
        ),
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--components-root", required=True)
    parser.add_argument("--inpainting-input-receipt", required=True)
    parser.add_argument("--aura-adapter-receipt", required=True)
    parser.add_argument("--articulated-source-receipt", required=True)
    parser.add_argument("--joint-agent-packet", required=True)
    parser.add_argument("--execution-authority")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = compile_articulated_public_scene_run(
        freeze_path=args.freeze,
        repo_root=args.repo_root,
        data_root=args.data_root,
        components_root=args.components_root,
        inpainting_input_receipt_path=args.inpainting_input_receipt,
        aura_adapter_receipt_path=args.aura_adapter_receipt,
        articulated_source_receipt_path=args.articulated_source_receipt,
        joint_agent_packet_path=args.joint_agent_packet,
        execution_authority_path=args.execution_authority,
        output_path=args.output,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ArticulatedPublicScenePipelineError",
    "SCHEMA_VERSION",
    "compile_articulated_public_scene_run",
    "compile_articulated_public_scene_state",
]
