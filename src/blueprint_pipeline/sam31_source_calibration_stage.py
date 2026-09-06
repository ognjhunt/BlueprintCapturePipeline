"""One hardware-only calibrated-view child using the canonical GPU render probe."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_json
from .source_calibration_render_packet import build_source_calibration_gpu_render_bundle, RECEIPT_NAME
from .source_calibration_render_return import (record, require, verify_source_calibration_return,
    materialize_source_calibration_closed_return, require_source_calibration_closure)
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read


def _run(argv: list[str], *, cwd: Path) -> int:
    return subprocess.run(argv,cwd=cwd,check=False).returncode


def _input(job: Mapping[str, Any], name: str) -> Path:
    ref=job['inputs'][name]
    return checked_file(ref['path'],ref)


def _write(path: Path, value: dict) -> None:
    with path.open('x',encoding='utf-8') as stream:
        stream.write(canonical_json(value)+'\n')


def _posted_charge(result: dict, output: Path) -> Path | None:
    from .vast_official_billing_extractor import extract_vast_official_instance_charge
    charge_path=output/'official_vast_instance_charge.json'
    if charge_path.exists():
        return charge_path
    ids=result.get('vast_instance_ids',[])
    require(len(ids)==1,'billing_instance_identity_invalid')
    startup_path=Path(result['provider_adapter_result_path']).parent/'vast_startup_probe_manifest.json'
    if not startup_path.is_file():
        return None
    startup=read(startup_path)
    require(startup.get('instance_id')==ids[0],'billing_startup_instance_mismatch')
    label=str(startup.get('create_request_summary',{}).get('label') or '')
    audit=Path(os.getenv('BLUEPRINT_PROVIDER_BILLING_AUDIT_ROOT') or
               '/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard/billing-audit')
    for path in sorted(audit.rglob('provider_billing_source_receipt.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
        try:
            charge=extract_vast_official_instance_charge(provider_billing_source_receipt_path=path,
                                                       instance_id=ids[0],launch_label=label)
        except (OSError,ValueError):
            continue
        require(charge['official_charge_usd']<=1.0,'official_spend_cap_exceeded')
        _write(charge_path,charge)
        return charge_path
    return None


def validate_retained_source_calibration_stage(outcome: Mapping[str, Any]) -> None:
    artifacts=outcome['artifacts']
    result=read(checked_file(artifacts['source_calibration_execution']['path'],artifacts['source_calibration_execution']))
    require(result.get('status')=='completed' and result.get('render_scope')=='source_calibration'
            and result.get('continuing_spend_from_this_run') is False
            and result.get('independent_watchdog',{}).get('provider_absence_confirmed') is True,
            'retained_execution_invalid')
    prepared=read(checked_file(artifacts['source_calibration_prepared_inputs']['path'],artifacts['source_calibration_prepared_inputs']))
    require_source_calibration_closure(prepared,checked_file(artifacts['source_calibration_return']['path'],artifacts['source_calibration_return']))
    charge=read(checked_file(artifacts['source_calibration_official_charge']['path'],artifacts['source_calibration_official_charge']),digest_field='charge_digest')
    require(charge.get('provider_instance_id')==result['vast_instance_ids'][0]
            and 0<=charge.get('official_charge_usd',-1)<=1.0,'retained_official_charge_invalid')
    if 'scene_owner_attempt' in outcome:
        from .task_evaluation_scene_execution_authority import require_scene_execution_authority
        reference=outcome['scene_owner_attempt']
        owner=read(checked_file(reference['path'],reference))
        require_scene_execution_authority(owner,
            source_commit=owner['scene_attempt_binding']['source_commit'],reopen_records=False)


def execute_source_calibration_stage(job: Mapping[str, Any], *, allocator_runner: Callable[...,int]=_run) -> dict[str, Any]:
    from .task_evaluation_sam31_preparation_cpu_stages import execute_cpu_stage
    from .public_scene_inpainting_inputs import finalize_public_scene_inpainting_inputs
    from .public_scene_inpainting_preparation import adopt_finalized_public_scene_inpainting_inputs
    from scripts.issue_retained_scene_render_paid_attempt_authority import issue_paid_attempt_authority
    profile=job['server_profile']
    settings=profile.get('calibrated_views',{})
    require(settings.get('execution_site')=='provider_gpu' and settings.get('hardware_required') is True
            and settings.get('max_spend_usd')==1.0 and settings.get('hard_ttl_seconds')==1800
            and settings.get('retry_cap')==0 and settings.get('maximum_resource_count')==1
            and settings.get('allowed_geolocation_country_codes')==['US'],'hardware_profile_invalid')
    require(isinstance(settings.get('machine_avoidlist'), Mapping),'hardware_machine_avoidlist_required')
    avoidlist=checked_file(settings['machine_avoidlist']['path'],settings['machine_avoidlist'])
    root=Path(job['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    preparation_record=root/'cpu_preparation_outcome.json'
    if not preparation_record.exists():
        require(not job.get('resume_only'),'prior_preparation_missing')
        prepared_outcome=execute_cpu_stage({**job,'output_root':str(root/'cpu')},prepare_hardware_render=True)
        _write(preparation_record,prepared_outcome)
    prepared_outcome=read(preparation_record)
    prepared_path=checked_file(prepared_outcome['prepared_inputs']['path'],prepared_outcome['prepared_inputs'])
    prepared=read(prepared_path,digest_field='preparation_digest')
    bundle_receipt=root/'bundle'/RECEIPT_NAME
    task_ref=job['plan']['host_inputs']['task_request']
    task_path=checked_file(task_ref['path'],task_ref)
    task=read(task_path)
    if not bundle_receipt.exists():
        build_source_calibration_gpu_render_bundle(prepared_inputs_path=prepared_path,repo_root=job['repo_root'],
            renderer_vendor_root=Path(job['runtime_root'])/'renderer/tools/splat_render/node_modules',
            task_request_path=task_path,conversion_receipt_path=_input(job,'standard_splat_conversion_receipt'),
            original_source_path=_input(job,'source_appearance'),job_dir=root/'bundle',
            expected_source_commit=job['expected_source_commit'],
            approved_roots=tuple(Path(p) for p in profile['approved_paid_input_roots']))
    authority_path=root/'paid_attempt_authority.json'
    rate=float(settings['max_hourly_rate_usd'])
    if not authority_path.exists():
        _write(authority_path,issue_paid_attempt_authority(bundle_receipt_path=bundle_receipt,
            authorized_by=task['human_authority']['accepted_by'],max_hourly_rate_usd=rate,hard_ttl_seconds=1800))
    result_path=root/'allocator_result.json'
    started=root/'allocator_started.json'
    if not result_path.exists():
        require(not started.exists(),'prior_allocation_requires_reconciliation')
        require(not job.get('resume_only'),'prior_allocation_requires_reconciliation')
        from .task_evaluation_scene_owner_attempt_profiles import require_fresh_task_owner
        require_fresh_task_owner(read(checked_file(task_ref['path'], task_ref)),
            source_commit=job['expected_source_commit'], maximum_spend_usd=settings['max_spend_usd'],
            output_path=root/'scene_owner_attempt.json')
        argv=[sys.executable,'-m','blueprint_pipeline.paid_resource_allocator','gpu-canary','--provider','vast',
            '--probe-kind','adp-retained-scene-gpu-render','--execute','--expected-source-commit',job['expected_source_commit'],
            '--admission-out',str(root/'admission.json'),'--adapter-output',str(result_path),
            '--adp-retained-scene-render-bundle-receipt',str(bundle_receipt),
            '--adp-retained-scene-render-attempt-authority',str(authority_path),
            '--adp-retained-scene-render-job-dir',str(root/'provider'),
            '--adp-retained-scene-render-max-hourly-rate-usd',str(rate),
            '--adp-retained-scene-render-hard-ttl-seconds','1800']
        argv.extend(['--adp-machine-avoidlist',str(avoidlist)])
        _write(started,{'source_commit':job['expected_source_commit'],'bundle_receipt':record(bundle_receipt)})
        allocator_runner(argv,cwd=Path(job['repo_root']))
    require(result_path.is_file(),'allocator_result_missing')
    result=read(result_path)
    require(result.get('status')=='completed' and result.get('render_scope')=='source_calibration','gpu_execution_not_complete')
    return_path=Path(result['source_calibration_return']['return_path'])
    verify_source_calibration_return(prepared,return_path)
    charge=_posted_charge(result,root)
    owner_path=root/'scene_owner_attempt.json'
    owner_metadata={'scene_owner_attempt':record(owner_path)} if owner_path.exists() else {}
    if charge is None:
        return {'status':'waiting_for_external_result','stage_id':'calibrated_views',
                'waiting_reason':'official_vast_billing_not_posted','candidate_policy_queried':False,
                **owner_metadata,
                'artifacts':{'source_calibration_execution':record(result_path),
                             'source_calibration_prepared_inputs':record(prepared_path),
                             'source_calibration_return':record(return_path)}}
    closed_return=root/'source_calibration_closed_return.v1.json'
    if not closed_return.exists():
        bundle=read(bundle_receipt)
        disclosure_path=root/'source_calibration_disclosure_proof.v1.json'
        if not disclosure_path.exists():
            _write(disclosure_path,bundle['source_disclosure'])
        from .vast_independent_watchdog_control import WATCHDOG_DIR_NAME, EVIDENCE_NAME
        closure={
            'source_disclosure':record(disclosure_path),
            'private_store':record(root/'provider/source_calibration_private_store_readback.v1.json'),
            'provider_execution':record(result_path),'official_billing':record(charge),
            'teardown':record(Path(result['teardown_manifest_path'])),
            'provider_zero':record(root/'provider'/WATCHDOG_DIR_NAME/EVIDENCE_NAME),
        }
        materialize_source_calibration_closed_return(prepared_inputs=prepared,returned_group_path=return_path,
            execution_closure=closure,output_path=closed_return)
    require_source_calibration_closure(prepared,closed_return)
    return_path=closed_return
    output=prepared_path.parent
    receipt_path=output/'public_scene_interiorgs_edit_input_receipt.v2.json'
    # The finalizer's terminal receipt is the checkpoint. A second checkpoint
    # could fail after that receipt was safely written and strand reentry.
    finalize = adopt_finalized_public_scene_inpainting_inputs if receipt_path.exists() else finalize_public_scene_inpainting_inputs
    receipt=finalize(preparation_path=prepared_path,returned_group_path=return_path)
    artifacts={'calibrated_view_request':prepared_outcome['calibrated_view_request'],
        'calibrated_view_receipt':record(receipt_path),
        'camera_contract':record(output/receipt['derived_artifacts']['cameras']['relative_path']),
        'source_calibration_execution':record(result_path),'source_calibration_prepared_inputs':record(prepared_path),
        'source_calibration_return':record(return_path),'source_calibration_official_charge':record(charge)}
    outcome={'status':'completed','stage_id':'calibrated_views','source_commit':job['expected_source_commit'],
             'artifacts':artifacts,'candidate_policy_queried':False,'evaluation_authorized':False,
             'provider_mutation_performed':True,**owner_metadata}
    validate_retained_source_calibration_stage(outcome)
    return outcome
