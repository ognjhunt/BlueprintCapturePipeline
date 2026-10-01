"""Explicit fixture-owner cancellation followed by original cache GC.

This does not prove automatic terminal-pin reconciliation or six-hour retention.
Failed chains, live producers/readers and unknown paths retain all their bytes.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from blueprint_pipeline.completed_replay_cache_retention import process_reference_index
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_pins import load_storage_pins, release_storage_pin
from blueprint_pipeline.control_plane_storage_roots import require_storage_class
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_scene_intake import revoke_scene_intent
from scripts.control_plane_concurrency_load_test import REQUIRED_STAGES, allocated_tree_bytes
from scripts.control_plane_concurrency_policy import deliver_policy_fixture
from scripts.control_plane_concurrency_provider import file_record


def _safe(path, root):
    path=Path(path)
    if (not path.is_absolute() or not path.is_relative_to(root)
            or any(p.is_symlink() for p in (path,*path.parents))):
        raise ValueError('retirement_path_not_owned')
    return path


def namespace_classifier(*, owned_root: Path, mapping: dict):
    """Translate only explicitly enumerated owned paths, retaining real classes."""
    owned_root=Path(owned_root).resolve(strict=True)
    roots=sorted([(_safe(real,owned_root),Path(logical)) for real,logical in mapping.items()],
                 key=lambda row:len(row[0].parts),reverse=True)
    def classify(path,*,expected,code):
        real=_safe(path,owned_root)
        for root,logical in roots:
            if real==root or real.is_relative_to(root):
                return require_storage_class(str(logical/real.relative_to(root)),expected=expected,code=code)
        raise ValueError('retirement_unclassified_owned_path')
    return classify


def retire_fixture_chains(*, control_root: Path, scenes: list, source_commit: str,
                          producer_pids: list[int], process_root: Path=Path('/proc')) -> dict:
    """All validations precede owner revocation, pin release and GC mutation."""
    for scene in scenes:
        if ([row.get('stage') for row in scene['stages']]!=list(REQUIRED_STAGES[:-1])
                or any(row.get('status')!='completed' for row in scene['stages'])):
            raise ValueError('retirement_chain_incomplete')
    if not scenes:raise ValueError('retirement_chain_incomplete')
    for pid in producer_pids:
        if type(pid) is not int or pid<=0:raise ValueError('retirement_producer_identity_invalid')
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise ValueError('retirement_producer_still_alive')
    root=Path(control_root).resolve(strict=True)
    marker=_safe(root/'.concurrency-harness-owner',root)
    if root.stat().st_uid!=os.geteuid() or marker.stat().st_uid!=os.geteuid():
        raise ValueError('retirement_root_owner_mismatch')
    if not marker.read_text().startswith('control_plane_concurrency_load_test.v1 '):
        raise ValueError('retirement_root_marker_invalid')
    reservations=_safe(root/'reservations',root)
    if list(reservations.glob('*.json')):raise ValueError('retirement_live_reservations')
    pins_root=_safe(root/'pins',root)
    prepared=_safe(root/'prepared-references',root)
    compiled_root=_safe(root/'compiled',root)
    derived=[prepared,compiled_root]
    mapping={prepared:'/var/lib/blueprint/task-evaluation-inputs/prepared-references',
             compiled_root:'/var/lib/blueprint/task-evaluation-inputs/compiled-episodes'}
    queues=[];allowed_owners=set();revocations=[]
    for scene in scenes:
        key=scene['scene_key']
        if not isinstance(key,str) or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}',key) is None:
            raise ValueError('retirement_scene_key_invalid')
        intake=scene['intake'];first=scene['preparation'];ready=scene['episode_preparation']
        compiled=scene['compiled']
        if (scene['source_commit']!=source_commit or intake['source_commit']!=source_commit
                or compiled.get('source_commit')!=source_commit
                or compiled.get('status')!='compiled_for_production_launch'
                or compiled.get('result_digest')!=canonical_digest(compiled,digest_field='result_digest')):
            raise ValueError('retirement_compiler_binding_invalid')
        path=_safe(compiled['compiled_episode_packet_path'],root)
        if file_record(path)!={'digest':compiled['compiled_episode_packet_digest'],
                               'size_bytes':compiled['compiled_episode_packet_size_bytes']}:
            raise ValueError('retirement_compiled_bytes_changed')
        for activation,preparation in ((scene['configuration_activation'],first),(scene['native_activation'],ready)):
            result=activation['worker']['results'][0]
            if (result.get('status')!='profile_authority_materialized_no_execution'
                    or result.get('preparation_result_digest')!=preparation['run']['results'][0]['result_digest']
                    or activation['preparer'].get('status')!='prepared'
                    or activation['preparer'].get('provider_allocation_performed') is not False):
                raise ValueError('retirement_preparation_binding_invalid')
            queue=_safe(activation['queue_root'],root)
            derived_root=_safe(queue.parent/'activated',root)
            derived.append(derived_root)
            mapping[derived_root]='/var/lib/blueprint/task-evaluation-inputs/launch-activations'
            queues.append(queue)
            mapping[queue]='/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-activations'
            allowed_owners.add(('activation',result['activation_id']))
        for prep in (first,ready):
            allowed_owners.add(('preparation',prep['run']['results'][0]['preparation_id']))
        allowed_owners.add(('compilation',compiled['compilation_id']))
        for queue,logical in ((intake['preparation_queue'],'task-evaluation-launch-preparations'),
            (first['construction_queue'],'task-evaluation-scene-constructions'),
            (ready['compilation_queue'],'task-evaluation-episode-compilations')):
            queue=_safe(queue,root);queues.append(queue)
            mapping[queue]='/var/lib/blueprint/pipeline-control-plane/'+logical
        if scene['delivery']['projection']!=deliver_policy_fixture(collected=scene['collected'])['projection']:
            raise ValueError('retirement_delivery_changed')
    for queue in queues:
        if list((queue/'pending').glob('*.json')) or list((queue/'processing').glob('*.json')):
            raise ValueError('retirement_queue_still_active')
    pins=load_storage_pins(pins_root)
    known={(pin['kind'],pin['owner_id']) for pin in pins}
    for pin in pins:
        if ((pin['kind'],pin['owner_id']) not in allowed_owners
                or any((dep['kind'],dep['owner_id']) not in known for dep in pin['depends_on'])):
            raise ValueError('retirement_unknown_pin_owner')
        for raw in pin['paths']:_safe(raw,root)
    reader=process_reference_index(process_root=process_root,ignored_process_ids=(os.getpid(),))
    for pin in pins:
        if any(reader(Path(path)) for path in pin['paths']):raise ValueError('retirement_process_reader')
    for path in derived:
        if reader(path):raise ValueError('retirement_process_reader')
    before=allocated_tree_bytes(root)
    # Terminate future fixture eligibility without inventing a launch, spend,
    # provider-zero or consumed standing authorization.
    for scene in scenes:
        intake=scene['intake'];intent=intake['intent']
        revocations.append(revoke_scene_intent(queue_root=Path(intake['intent_path']).parent.parent,
            intent_id=intent['intent_id'],intent_digest=intent['intent_digest'],owner=intent['request']['owner']))
    releases=[release_storage_pin(pins_root=pins_root,kind=pin['kind'],owner_id=pin['owner_id'])
              for pin in pins if pin['status']=='live']
    remaining=[pin for pin in load_storage_pins(pins_root) if pin['status']=='live']
    if remaining:raise ValueError('retirement_residual_pins')
    stores=sorted({path for directory in derived for path in directory.glob('**/content-addressed/sha256')})
    # The original GC receives the exact owned roots and original classifier.
    # Registered-experiment/live-door lanes are outside this fixture namespace.
    report=run_storage_gc(content_store_roots=stores,derived_roots=derived,queue_roots=queues,
        pins_root=pins_root,apply=True,ack=RUN_ACK,content_minimum_age_seconds=0,
        derived_minimum_age_seconds=0,running_commit=source_commit,
        _experiment_config_path=root/'no-live-operator-door.json',
        classifier=namespace_classifier(owned_root=root,mapping=mapping))
    if report.get('phase_errors') or report.get('skipped_roots'):
        raise ValueError('retirement_gc_incomplete:'+str(report.get('phase_errors')))
    receipt={'schema_version':'development_fixture_owner_retirement.v1','source_commit':source_commit,
        'claim_ceiling':'development_only','status':'completed',
        'boundary':'fixture-owner cancellation and release followed by production GC',
        'automatic_terminal_reconciliation_proven':False,'normal_six_hour_retention_proven':False,
        'zero_age_parameters_explicit':True,'producer_pids_joined':producer_pids,
        'owner_revocations':revocations,'pin_releases':releases,'gc':report,
        'before_allocated_bytes':before,'after_allocated_bytes':allocated_tree_bytes(root),
        'residual_pins':len(remaining),'residual_leases':len(list(reservations.glob('*.json'))),
        'observed_at_epoch':time.time(),'receipt_digest':''}
    receipt['receipt_digest']=canonical_digest(receipt,digest_field='receipt_digest')
    return receipt
