"""Re-review corrected media once, retaining the original learned proposal."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import task_evaluation_robot_placement_agent as agent
from . import task_evaluation_visual_review_continuation as continuation
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_supervisor.agents_sdk import OpenAIAgentsSDKInvoker
from .task_evaluation_supervisor.inference_reservations import InferenceReservationAudit
from .task_evaluation_unstarted_controls_reservations import _read
from .common import write_json


def run(*, source: Mapping[str, Any], run_id: str, scene_binding: Mapping[str, Any],
        task_binding: Mapping[str, Any], task_trajectory: Mapping[str, Any],
        inventory: Mapping[str, Any], overview_images: Sequence[Mapping[str, Any]],
        output_dir: Path, artifacts: list[dict[str, Any]], validate_candidate: Callable,
        render_candidate: Callable, placement_scene_owner: Mapping[str, Any],
        max_inference_cost_usd: float, invoker_factory: Callable = OpenAIAgentsSDKInvoker) -> dict[str, Any]:
    from .task_evaluation_scene_execution_authority import require_scene_execution_authority
    from .task_evaluation_release_identity import running_release_commit
    from .robot_placement_preview_rasterizer import RENDERER

    frozen=continuation.validate(source,expected_commit=running_release_commit())
    old=frozen['receipt']
    proposal=frozen['proposal']
    continuation.require(max_inference_cost_usd == continuation.REVIEW_CAP
        and old['scene_binding_digest'] == canonical_digest(scene_binding)
        and old['task_binding_digest'] == canonical_digest(task_binding)
        and old['task_trajectory_digest'] == task_trajectory['trajectory_digest']
        and old['candidate_inventory_digest'] == inventory['candidate_inventory_digest'], 'execution_binding_mismatch')
    gate=agent._validated_gate(agent._reject_infeasible_orientation_slew(gate=validate_candidate(proposal),
        proposal=proposal,trajectory=task_trajectory,robot_id='franka_panda',
        maximum_steps_per_phase=task_trajectory.get('maximum_steps_per_phase')))
    continuation.require(gate['status']=='passed','geometry_revalidation_failed')
    images=list(render_candidate(proposal,0))
    old_images={i['digest'] for r in old['rounds'] for i in r.get('preview_images',[])}
    continuation.require(len(images)==3 and all(i['digest'] not in old_images for i in images),'media_not_corrected')
    for image in images:
        provenance=_read(Path(image['render_provenance_path']))
        continuation.require(provenance.get('renderer')==RENDERER
            and provenance.get('render_digest') == image.get('render_provenance_digest')
            and provenance.get('render_digest') == canonical_digest(provenance,digest_field='render_digest')
            and provenance.get('image_digest')==image['digest']
            and provenance.get('robot_mesh_scope')=='default_prim_with_instance_proxies'
            and provenance.get('depth_buffer_shared_by_scene_and_robot') is True, 'preview_renderer_invalid')
    require_scene_execution_authority(placement_scene_owner,source_commit=source['execution_commit'],
        provider='openai',maximum_spend_usd=continuation.REVIEW_CAP)
    config=replace(agent.robot_placement_agents_sdk_config(max_inference_cost_usd=continuation.REVIEW_CAP,
        allow_live_invocation=True,tracing_disabled=True),max_output_tokens=continuation.MAX_OUTPUT_TOKENS,
        max_input_tokens=continuation.MAX_INPUT_TOKENS)
    invoker=invoker_factory(config)
    audit=InferenceReservationAudit(run_root=output_dir,run_id=run_id)
    invoker.configure_reservation_audit(record_reservation=audit.record_reservation,
        record_completion=audit.record_completion,restored_reserved_cost_usd=0.0)
    instructions=("You are the visual sanity reviewer for one frozen fixed-base robot placement. Review all supplied "
        "opaque, depth-tested local geometry views. Fail closed if the base appears embedded, floating, unsupported, "
        "occluded so placement cannot be judged, pointed away from the task, or the task workspace looks implausibly "
        "unreachable. The crop is explicitly labeled; trajectory, target, and facing overlays are not collision or "
        "visibility evidence. Your veto-only verdict cannot override deterministic geometry. Do not grade native "
        "physics or execution; those require the subsequent native controls. Return the declared structured verdict.")
    spec=agent.AgentsSDKAgentSpec(run_id=run_id,capability='robot_placement_visual_review',
        name='Blueprint Robot Placement Visual Reviewer',instructions=instructions,model=agent.ROBOT_PLACEMENT_AGENT_MODEL,
        max_turns=1,max_output_tokens=continuation.MAX_OUTPUT_TOKENS,max_input_tokens=continuation.MAX_INPUT_TOKENS,
        reasoning_effort=agent.ROBOT_PLACEMENT_AGENT_REASONING_EFFORT,output_type=agent.RobotPlacementVisualReviewOutput,
        stable_developer_prefix=agent._stable_prompt_prefix(capability='visual_review',instructions=instructions,
            output_type=agent.RobotPlacementVisualReviewOutput),prompt_contract_version='robot-placement-review-correction-v1',
        privacy_scope='task_evaluation_rights_admitted',processing_region='default')
    try:
        result=invoker.invoke(spec,agent._multimodal_input(prompt={'proposal':proposal,
            'geometry_gate':{k:gate[k] for k in ('status','support_passed','collision_passed','reachability_passed','facing_passed')},
            'native_checks_still_required':True},images=images))
    finally:
        audit.write_manifest()
    continuation.require(result.provider == 'openai' and result.model == agent.ROBOT_PLACEMENT_AGENT_MODEL,
                         'review_provider_or_model_mismatch')
    visual=agent.RobotPlacementVisualReviewOutput.model_validate(result.output)
    round_record={'round_index':0,'proposal':proposal,'proposal_provider':'retained_openai_proposal',
        'proposal_model':old['rounds'][0]['proposal_model'],'proposal_source':dict(source['source_placement_receipt']),
        'geometry_gate':gate,'preview_images':agent._image_metadata(images),'native_attempt':None,
        'visual_review':visual.model_dump(mode='json'),'visual_review_provider':result.provider,
        'visual_review_model':result.model,'visual_review_sdk_version':result.sdk_version,
        'visual_review_usage':dict(result.usage),'visual_review_trace_id':result.trace_id}
    receipt=agent._build_placement_receipt(run_id=run_id,scene_digest=canonical_digest(scene_binding),
        task_digest=canonical_digest(task_binding),scene_context_digest=canonical_digest({'continuation':source['continuation_digest']}),
        task_context_digest=canonical_digest(task_trajectory),overview_images=overview_images,prior_native_attempts=[],
        history=[round_record],accepted=round_record if agent._visual_passed(visual) else None,max_rounds=1,
        native_loop_enabled=False,task_trajectory_digest=task_trajectory['trajectory_digest'],
        candidate_inventory_digest=inventory['candidate_inventory_digest'],candidate_inventory_trajectory_digest=inventory['trajectory_digest'])
    receipt['visual_review_continuation']=dict(source)
    receipt['new_provider_call_count']=1
    receipt['proposal_reexecuted']=False
    receipt['receipt_digest']=canonical_digest(receipt,digest_field='receipt_digest')
    write_json(output_dir/'task_evaluation_robot_placement_receipt.v1.json',receipt)
    write_json(output_dir/'task_evaluation_robot_placement_artifact_index.v1.json',
        {'schema_version':'task_evaluation_robot_placement_artifact_index.v1','run_id':run_id,
         'receipt_digest':receipt['receipt_digest'],'artifacts':artifacts})
    return receipt
