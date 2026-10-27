"""Stage supplied task-family bytes for the existing native policy worker.

Beta: job-to-evaluation. This CPU producer grants no runtime, policy, provider,
or spending authority; native application and control receipts remain required.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_attempt_evidence import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file
from .task_evaluation_scene_progression_contracts import require, safe_path
from .task_family_native_contracts import (
    RUNTIME_SCHEMA, family_source_blockers, validate_task_family_native_runtime_contract,
)


def is_task_family_preparation(preparation):
    return ((preparation.get("task_evaluation_inputs") or {}).get("native_runtime_contract") or {}).get(
        "schema_version") == RUNTIME_SCHEMA


def family_source_references(*, preparation, runtime_inputs_path, task_context=None):
    """Reopen every source byte before registration, resolution and staging."""
    inputs = preparation["task_evaluation_inputs"]
    native = validate_task_family_native_runtime_contract(inputs["native_runtime_contract"])
    require(inputs.get('task_success_contract') == native['task_spec']['task_success_contract']
            and inputs.get('scope') == native['task_spec']['task_success_contract']['scope'],
            'website_task_family_native_task_binding_changed')
    require(all(preparation.get('binding', {}).get(key) == inputs['source_bindings'].get(key)
                for key in ('task_context_digest', 'source_geometry_digest', 'task_masks_digest',
                            'splat_digest', 'collision_mesh_digest')), 'website_task_family_preparation_binding_changed')
    if task_context is None:
        require(inputs.get('task_semantics') is None, 'website_task_family_semantics_context_required')
    else:
        from .task_family_scoring import canonical_task_family
        from .website_task_semantics import effective_task_profile, require_semantic_stage_order
        require(task_context.get('context_digest') == canonical_digest(task_context, digest_field='context_digest')
                and task_context.get('context_digest') == preparation['binding']['task_context_digest']
                and native['task_spec']['task_success_contract']['scope']['task_context_digest'] == task_context['context_digest'],
                'website_task_family_source_context_changed')
        family = canonical_task_family(effective_task_profile(task_context=task_context,
            task_semantics=inputs.get('task_semantics'), source_video_digest=inputs.get('source_video_digest')))
        require(family is not None and family == inputs['task_family'] == native['task_family'],
                'website_task_family_semantic_profile_changed')
        require_semantic_stage_order(task_context=task_context, task_semantics=inputs.get('task_semantics'),
            source_video_digest=inputs.get('source_video_digest'), contract=inputs['task_success_contract'])
    require(preparation.get("status") == "intake_ready" and preparation.get("blockers") == [],
            "website_task_family_preparation_not_ready")
    retained = json.loads(Path(runtime_inputs_path).read_text())
    require(retained == native, "website_task_family_runtime_contract_changed")
    require(not family_source_blockers(native=native, source_bindings=inputs["source_bindings"],
            source_scene=inputs["source_scene"]), "website_task_family_source_binding_incomplete")
    task = preparation["intake_request"]["task"]
    require(task.get("native_runtime_contract_digest") == native["contract_digest"]
            and task.get("task_success_contract_digest") == native["task_spec"]["task_success_contract"]['contract_digest']
            and task.get("task_family_runtime_selection") == inputs.get("task_family_runtime_selection")
            and task.get("policy_selection") == inputs.get("policy_selection"),
            "website_task_family_request_binding_changed")
    sources = inputs.get("asset_sources") or {}
    scene = inputs["source_scene"]
    rows = []
    paths = set()
    for asset in native["assets"]:
        asset_id = asset["asset_id"]
        relative = asset.get("bundle_relative_path") or asset.get("path")
        pure = PurePosixPath(str(relative or ""))
        require(isinstance(relative, str) and bool(relative) and not pure.is_absolute()
                and ".." not in pure.parts and pure.parts[0] == "assets"
                and len(pure.parts) > 1 and pure.as_posix() not in paths,
                "website_task_family_asset_bundle_path_invalid")
        paths.add(pure.as_posix())
        ref = sources.get(asset_id)
        if ref is None and asset.get("semantic_role") in {"scene_appearance", "scene_collision"}:
            key = "splat" if asset["semantic_role"] == "scene_appearance" else "collision_mesh"
            ref = record(scene[key + "_path"])
        require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"}
                and ref["sha256"] == asset["content_digest"], "website_task_family_asset_source_required:" + asset_id)
        source = safe_path(ref["path"])
        require(source.is_absolute(), "website_task_family_asset_source_invalid:" + asset_id)
        checked_file(source, ref)
        rows.append({"asset_id": asset_id, "semantic_role": asset.get("semantic_role", asset_id),
                     "relative_path": pure.as_posix(), "reference": ref})
    # Independently reopen the capture source even when equivalent bytes are
    # also supplied under an explicit scene asset reference.
    for key in ("splat", "collision_mesh"):
        ref = record(scene[key + "_path"])
        require(ref["sha256"] == scene[key + "_digest"], "website_task_family_source_scene_changed")
    return rows


def materialize_website_task_family_packet(*, preparation, runtime_inputs_path, output_root,
                                         intent_reference, attempt_reference, release_reference, task_context=None):
    rows = family_source_references(preparation=preparation, runtime_inputs_path=runtime_inputs_path,
                                   task_context=task_context)
    inputs = preparation["task_evaluation_inputs"]
    native = inputs["native_runtime_contract"]
    output = safe_path(output_root)
    require(output.is_absolute() and not output.exists(), "website_task_family_packet_output_exists")
    output.mkdir(parents=True, mode=0o750)
    try:
        bindings = []
        for row in rows:
            target = output / row["relative_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checked_file(row["reference"]["path"], row["reference"]), target)
            observed = record(target)
            require(observed["sha256"] == row["reference"]["sha256"]
                    and observed["size_bytes"] == row["reference"]["size_bytes"],
                    "website_task_family_asset_copy_changed")
            bindings.append({"asset_id": row["asset_id"], "semantic_role": row["semantic_role"],
                "source": row["reference"], "staged_relative_path": row["relative_path"],
                "staged_sha256": observed["sha256"], "staged_size_bytes": observed["size_bytes"]})
        request = {"schema_version": "native_task_arena_packet_request.v1",
            "producer": "website_task_family_packet.v1", "scene_id": native["scene_id"],
            "task_id": native["task_id"], "native_runtime_contract_digest": native["contract_digest"],
            "task_family_runtime_selection": inputs["task_family_runtime_selection"],
            "policy_selection": inputs.get("policy_selection"),
            "scene_intent": intent_reference, "preparation_attempt": attempt_reference,
            "release_binding": release_reference, "assets": rows, "paid_authority_granted": False}
        request["request_digest"] = canonical_digest(request, digest_field="request_digest")
        write_json(output / "native_task_arena_packet_request.v1.json", request)
        write_json(output / "native_task_runtime_contract.v1.json", native)
        from .native_task_arena_scene_plan import materialize_native_task_arena_scene_plan
        plan = materialize_native_task_arena_scene_plan(runtime_contract=native,
            provider_asset_directory=output, published_asset_directory="assets",
            physics_frequency_hz=inputs.get("physics_frequency_hz", 120),
            task_family_runtime_selection=inputs["task_family_runtime_selection"],
            policy_selection=inputs.get("policy_selection"),
            destination=output / "native_task_arena_scene_plan.v1.json")
        artifacts = []
        for name in ("native_task_arena_packet_request.v1.json", "native_task_runtime_contract.v1.json",
                     "native_task_arena_scene_plan.v1.json"):
            ref = record(output / name)
            artifacts.append({"relative_path": name, "sha256": ref["sha256"], "size_bytes": ref["size_bytes"]})
        receipt = {"schema_version": "native_task_arena_packet_receipt.v1",
            "status": "construction_packet_completed", "scene_id": native["scene_id"], "task_id": native["task_id"],
            "request_digest": request["request_digest"], "runtime_contract_digest": native["contract_digest"],
            "scenario_instance_digest": native["scenario_instance_digest"],
            "arena_scene_plan_digest": plan["plan_digest"], "source_bindings": bindings, "artifacts": artifacts,
            "source_bytes_mutated": False, "native_application_claimed": False, "policy_episode_claimed": False,
            "simulator_execution_is_not_physical_truth": True}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        write_json(output / "native_task_arena_packet_receipt.v1.json", receipt)
        family_source_references(preparation=preparation, runtime_inputs_path=runtime_inputs_path,
                                 task_context=task_context)
        return {"packet_root": str(output), "packet_receipt": record(output / "native_task_arena_packet_receipt.v1.json"),
            "scene_plan": record(output / "native_task_arena_scene_plan.v1.json"),
            "native_runtime_contract": record(output / "native_task_runtime_contract.v1.json")}
    except BaseException:
        shutil.rmtree(output)
        raise
