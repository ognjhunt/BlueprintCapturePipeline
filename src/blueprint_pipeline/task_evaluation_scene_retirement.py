"""Consent-bound scene actions; real exclusive admission lasts through completion.

A metadata KEEP plan supplies target evidence only. Protected installed policy,
current authenticated generations, actual supported lifetime closure and fresh
reference observations must independently admit an action. Missing evidence
never becomes permission. No production caller or enabled cleanup switch here.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import secrets
import stat
import time
import sys
from pathlib import Path
from contextlib import ExitStack

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _identity, _opened, _require
from .task_evaluation_scene_retirement_authority import (
    load_authority, selected_document, load_document, raw_reference,
)
from .task_evaluation_scene_retirement_generations import _write, _sealed
from .task_evaluation_scene_retirement_journal import SceneJournal
from .task_evaluation_scene_retirement_mutation import detach_and_remove, inventory_digest, removal_records
from .task_evaluation_scene_retirement_preservation import ActionAllowance, preserve_members
from .task_evaluation_scene_retirement_restore import restore_preserved_members, restore_records, _consume
from .task_evaluation_scene_retirement_metadata import retain_metadata_closure
from .task_evaluation_scene_retirement_intent_receipt import publish_pending_receipt, publish_terminal_receipt, publish_progress_receipt
from .task_evaluation_scene_lifecycle_plan import build_scene_lifecycle_plan
from . import task_evaluation_scene_retirement_recovery as recovery
from .task_evaluation_scene_retirement_declared_bytes import verify_declared_bytes as _verify_declared_bytes, verify_publication_rows
from .task_evaluation_scene_retirement_reference_transfer import validate_current_reference_transfer
from .task_evaluation_scene_lineage_budget import _Rows
from . import task_evaluation_scene_retirement_pin_mutation as pins


_LIFETIME='scene_retirement_lifetime.v1'
_TERMINAL={'completed','revoked_grace_elapsed','expired_grace_elapsed'}

# Finite actual participating consumers. Source hashes for an arbitrary subset
# do not establish closure; unknown future callsites require a reviewed update.
_COHORT_CALLS={
    'task_evaluation_scene_intake':'reserve_scene_attempt stage_scene_intent',
    'task_evaluation_scene_progression':'process_scene_intents',
    'task_evaluation_launch_preparation_worker':'materialize_preparation_references materialize_recipe_configuration_references materialize_recipe_supplemental_destination_references process_launch_preparation_queue',
    'task_evaluation_launch_activation_worker':'process_launch_activation_queue',
    'task_evaluation_episode_compilation_worker':'process_episode_compilation_queue',
    'task_evaluation_sam31_prefix_adoption':'materialize_completed_prefix_adoption publish_adoption_release_binding validate_completed_prefix_adoption',
    'task_evaluation_sam31_preparation_execution':'process_sam31_phase_queue',
    'task_evaluation_scene_configuration_sam31_preparation_driver':'advance_sam31_preparation',
    'task_evaluation_launch_dispatcher':'dispatch_launch_request process_launch_queue',
    'task_evaluation_policy_canary_dispatcher':'dispatch_policy_canary_activation process_policy_canary_activation_results process_policy_canary_dispatch_queue',
    'task_evaluation_scene_configuration_submission_publication':'publish_scene_configuration_submission',
    'website_scene_dispatch':'materialize_website_attempt register_website_preparation resolve_website_source',
    'website_native_submission':'materialize_website_submission validate_website_publication verified_submission_inputs',
    'artifixer_completed_training_reuse':'stage_completed_review stage_completed_training',
    'control_plane_storage_pins':'release_storage_pin write_storage_pin',
    'pubsub_handoff_listener':'process_handoff_payload pull_and_process stage_handoff_capture',
    'task_evaluation_terminal_scene_attempt_settlement':'budget_retained_hold retained_hold settle_retired_attempt_rows sweep_retired_attempts validate_terminal_settlement',
    'task_evaluation_scene_owner_authority':'reopen_scene_intent validate_task_scene_owner',
    'live_pipeline_result_artifact_resolution':'resolve_live_pipeline_result_artifact',
    'live_pipeline_result_artifact_response':'ResultArtifactFileResponse.__call__ result_artifact_response',
}
_REQUIRED_COHORT=frozenset('blueprint_pipeline.'+module+':'+name
    for module,names in _COHORT_CALLS.items() for name in names.split())


def _kept(reason, *, journal=None, members=()):
    result=dict(schema_version='scene_retirement_action_result.v1',status='kept',action='KEEP',
                reason=reason,mutations=0,members=list(members))
    if journal is not None:
        result.update(status='incomplete',mutations=None,journal_initial_raw_ref=journal.initial_ref,
                      last_event_raw_ref=journal.prior_ref)
    return result


def _allowance(authority, now, monotonic):
    limits=authority['policy']['limits']
    _require(type(limits) is dict and set(limits) <= {
        'logical_payload_bytes','archive_bytes','remote_bytes','elapsed_seconds'},'scene_retirement_limits_invalid')
    return ActionAllowance(expires_at=authority['consent']['expires_at'],now=now,monotonic=monotonic,
        **{('local_bytes' if key=='logical_payload_bytes' else key):value for key,value in limits.items()})


def _bind_transport(transport,allowance):
    bind=getattr(transport,'bind_allowance',None)
    if bind is not None:
        allowance.tick()
        bind(allowance)
        allowance.tick()


def _path_relation(a,b):
    first,second=Path(a),Path(b)
    return first.is_relative_to(second) or second.is_relative_to(first)


def _generation(policy, member, *, expected_states, retired_token=None):
    key=hashlib.sha256(member['canonical_path'].encode()).hexdigest()+'.json'
    value,ref=load_document(Path(policy['generation_store'])/key,maximum=65536)
    capture='capture_owner_user_id' in member
    _require(value.get('schema_version')==(
                 'scene_capture_generation.v1' if capture else 'scene_member_generation.v1')
             and value.get('state_digest')==canonical_digest(value,digest_field='state_digest')
             and value.get('canonical_path')==member['canonical_path']
             and value.get('generation_id')==member['generation_id']
             and value.get('state') in expected_states,'scene_retirement_generation_unavailable')
    if capture:
        _require(value.get('capture_owner_user_id')==member['capture_owner_user_id']
                 and value.get('owner_observation_raw_ref')==member['owner_observation_raw_ref']
                 and value.get('birth_delivery_raw_ref')==member['birth_delivery_raw_ref'],
                 'scene_retirement_generation_unavailable')
        owner=selected_document(member['owner_observation_raw_ref'],maximum=65536)
        birth=selected_document(member['birth_delivery_raw_ref'],maximum=65536)
        selected_document(member['source_membership_raw_ref'],maximum=65536)
        _require(owner.get('request_id')==member['request_id']
                 and owner.get('capture_owner',{}).get('user_id')==member['capture_owner_user_id']
                 and owner.get('completion_marker')==value.get('pinned_marker')
                 and birth.get('schema_version')=='capture_birth_delivery.v1'
                 and birth.get('source_membership_raw_ref')==member['source_membership_raw_ref']
                 and birth.get('producer_delivery')==owner.get('producer_delivery')
                 and birth.get('source_finalize')=={
                     'bucket':owner.get('bucket'),
                     'object_name':value['pinned_marker']['object_name'],
                     'generation':value['pinned_marker']['generation']},
                 'scene_retirement_generation_unavailable')
    else:
        _require(value.get('owner_intent_id')==member['owner_intent_id']
                 and value.get('owner_raw_ref')==member['owner_raw_ref'],
                 'scene_retirement_generation_unavailable')
    if retired_token is not None:
        _require(value.get('retirement_token')==retired_token,'scene_retirement_generation_unavailable')
    return value,ref


def _capture_action_current(policy, consent, member, allowance, *, expected_states):
    """Bind one original owner, accepted sponsor and current signed source under EX."""
    from .capture_original_owner_observer import (
        CaptureOwnerObservationError, load_original_owner_observation, validate_observation,
    )

    references=(member['owner_observation_raw_ref'],member['birth_delivery_raw_ref'],
                member['source_membership_raw_ref'],member['association_raw_ref'],
                member['scene_intent_raw_ref'])
    for index,reference in enumerate(references):
        allowance.charge('local_bytes',reference['size_bytes']*(2 if index<3 else 1))
    allowance.tick()
    generation,_=_generation(policy,member,expected_states=expected_states)
    _require(generation['schema_version']=='scene_capture_generation.v1'
             and consent['intent_raw_ref']==member['scene_intent_raw_ref'],
             'scene_retirement_capture_association_unproven')
    owner=selected_document(member['owner_observation_raw_ref'],maximum=65536)
    birth=selected_document(member['birth_delivery_raw_ref'],maximum=65536)
    selected_document(member['source_membership_raw_ref'],maximum=65536)
    association=selected_document(member['association_raw_ref'],maximum=65536)
    intent=selected_document(member['scene_intent_raw_ref'],maximum=65536)
    try:
        validated=validate_observation(owner,bucket=owner['bucket'],scene_id=owner['scene_id'],
            capture_id=owner['capture_id'],marker_generation=generation['pinned_marker']['generation'],
            now_epoch=owner['observed_at_epoch'])
    except (CaptureOwnerObservationError,KeyError,TypeError) as error:
        raise access.SceneRetirementAccessError('scene_retirement_capture_original_owner_unproven') from error
    _require(validated==owner and type(intent) is dict
             and intent.get('intent_id')==consent['intent_id']
             and intent.get('intent_digest')==canonical_digest(intent,digest_field='intent_digest')
             and type(intent.get('request')) is dict,
             'scene_retirement_capture_association_unproven')
    request=intent['request']
    sponsor=request.get('owner')
    selected=association.get('capture_source') if type(association) is dict else None
    context=policy.get('reference_context')
    roots=context.get('roots') if type(context) is dict else None
    _require(type(roots) is dict and type(roots.get('factory_output_root')) is str
             and type(roots.get('intent_root')) is str
             and type(roots.get('website_source_binding_root')) is str
             and Path(member['association_raw_ref']['path']).parent==(
                 Path(roots['factory_output_root'])/consent['intent_id']/'website-source')
             and Path(member['scene_intent_raw_ref']['path'])==(
                 Path(roots['intent_root'])/consent['intent_id']/'intent.json')
             and type(association) is dict
             and Path(member['association_raw_ref']['path']).name==(
                 association.get('binding_digest','')[7:]+'.json'),
             'scene_retirement_capture_association_unproven')
    _require(type(selected) is dict
             and association.get('schema_version')=='website_scene_source_binding.v1'
             and association.get('binding_digest')==canonical_digest(
                 association,digest_field='binding_digest')
             and association.get('intent_digest')==intent['intent_digest']
             and association.get('owner')==sponsor
             and selected.get('canonical_path')==member['canonical_path']
             and selected.get('generation_id')==member['generation_id']
             and selected.get('request_id')==member['request_id']
             and selected.get('capture_owner_user_id')==member['capture_owner_user_id']
             and selected.get('owner_observation_raw_ref')==member['owner_observation_raw_ref']
             and selected.get('birth_delivery_raw_ref')==member['birth_delivery_raw_ref']
             and selected.get('source_membership_raw_ref')==member['source_membership_raw_ref']
             and selected.get('source_membership_selector')==birth['source_membership_selector']
             and selected.get('delivery_key')==owner['producer_delivery']['delivery_key']
             and selected.get('capture_rights_digest')==cross_runtime_canonical_digest(
                 owner['capture_rights'])
             and selected.get('sponsoring_owner')==sponsor
             and selected.get('request_digest')==cross_runtime_canonical_digest(request),
             'scene_retirement_capture_association_unproven')
    registration_ref=raw_reference(association.get('registration'))
    _require(Path(registration_ref['path'])==(
                 Path(roots['website_source_binding_root'])/
                 (selected['request_digest'][7:]+'.json')),
             'scene_retirement_capture_association_unproven')
    allowance.charge('local_bytes',registration_ref['size_bytes'])
    registration=selected_document(registration_ref,maximum=65536)
    _require(registration.get('schema_version')=='website_scene_source_registration.v1'
             and registration.get('registration_digest')==canonical_digest(
                 registration,digest_field='registration_digest')
             and registration.get('request_digest')==selected['request_digest']
             and registration.get('capture_source')==selected,
             'scene_retirement_capture_association_unproven')
    allowance.tick()
    remaining=min(allowance.expires_at-allowance.last_wall,
                  allowance.elapsed_seconds-(allowance.last_tick-allowance.start),10)
    _require(remaining>0,'scene_retirement_deadline')
    try:
        current,response_bytes=load_original_owner_observation(
            bucket=owner['bucket'],scene_id=owner['scene_id'],capture_id=owner['capture_id'],
            marker_generation=generation['pinned_marker']['generation'],
            remaining_timeout_ms=max(1,min(10000,int(remaining*1000))),
            include_response_bytes=True)
    except CaptureOwnerObservationError as error:
        raise access.SceneRetirementAccessError('scene_retirement_capture_current_owner_unavailable') from error
    allowance.charge('remote_bytes',response_bytes)
    allowance.tick()
    _require(current['source_projection_digest']==owner['source_projection_digest']
             and current['capture_owner']==owner['capture_owner']
             and current['capture_rights']==owner['capture_rights']
             and current['producer_delivery']==owner['producer_delivery']
             and current['completion_marker']==owner['completion_marker'],
             'scene_retirement_capture_current_owner_changed')
    return generation


def _transition(policy, prior, *, state, token, journal_ref, inventory_sha256=None, identity=None):
    store=Path(policy['generation_store'])
    key=hashlib.sha256(prior['canonical_path'].encode()).hexdigest()
    current=access._read(store/(key+'.json'))
    _require(current==prior,'scene_retirement_generation_changed')
    value=dict(prior,state=state,retirement_token=token,journal_sha256=journal_ref['sha256'],
               state_sequence=prior['state_sequence']+1)
    if inventory_sha256 is not None:
        value['inventory_sha256']=inventory_sha256
    if identity is not None:
        value.update(zip(('dev','ino','mode'),identity))
    value=_sealed(value)
    with _opened(store,directory=True) as (fd,info):
        _require(stat.S_IMODE(info.st_mode)==0o700
                 and (info.st_uid,info.st_gid)==access._service_identity(),
                 'scene_retirement_service_identity_unproven')
        # Global EX excludes every enrolled SH birth/publisher; immutable history
        # and the exact prior value are checked before updating this projection.
        _write(fd,key+'.'+value['generation_id']+'.'+str(value['state_sequence'])+'.'+state+'.json',
               value,parent_identity=_identity(info))
        _write(fd,key+'.json',value,parent_identity=_identity(info),replace=True)
    return value


def _installed_cohort(policy, allowance):
    rows=policy['consumer_cohort']
    _require(type(rows) is list and 1<=len(rows)<=256,'scene_retirement_cohort_unproven')
    names=set()
    for row in rows:
        allowance.tick()
        _require(type(row) is dict and type(row.get('entrypoint')) is str
                 and len(row['entrypoint'])<=256 and row['entrypoint'] not in names,
                 'scene_retirement_cohort_unproven')
        names.add(row['entrypoint'])
    _require(names==_REQUIRED_COHORT,'scene_retirement_cohort_unproven')
    for row in rows:
        allowance.tick()
        entrypoint=row['entrypoint']
        _require(type(entrypoint) is str and entrypoint.startswith('blueprint_pipeline.')
                 and entrypoint.count(':')==1,'scene_retirement_cohort_unproven')
        module_name,name=entrypoint.split(':')
        _require(module_name.replace('_','').replace('.','').isalnum()
                 and name.replace('_','').replace('.','').isalnum(),'scene_retirement_cohort_unproven')
        module=importlib.import_module(module_name)
        path=_canonical(module.__file__)
        with _opened(path) as (fd,info):
            _require(info.st_size<=1024*1024,'scene_retirement_cohort_unproven')
            digest=hashlib.sha256()
            remaining=info.st_size
            while remaining:
                allowance.tick()
                _require(_identity(os.fstat(fd))==_identity(info),'scene_retirement_cohort_unproven')
                raw=os.read(fd,min(65536,remaining))
                _require(raw,'scene_retirement_cohort_unproven')
                digest.update(raw)
                remaining-=len(raw)
            _require(_identity(os.fstat(fd))==_identity(info),'scene_retirement_cohort_unproven')
        _require('sha256:'+digest.hexdigest()==row['installed_source_sha'],'scene_retirement_cohort_unproven')
        target=module
        for part in name.split('.'):
            target=getattr(target,part,None)
        _require(getattr(target,'__scene_retirement_lifetime__',None)==_LIFETIME,
                 'scene_retirement_cohort_unproven')
    # This verifies actual installed/admitted callsites, never absence of an
    # unknown older process. Such lifetimes require their own retained evidence.


def _plan_members(plan, consent, allowance=None, *, cache_inventory=None):
    rows=plan.get('measured_members')
    _require(type(rows) in (list,_Rows) and len(rows)<=10000,'scene_retirement_members_unproven')
    selected=consent['members']
    _require(selected,'scene_retirement_members_unproven')
    roots=[Path(member['canonical_path']) for member in selected]
    _require(not any(a!=b and a.is_relative_to(b) for a in roots for b in roots),
             'scene_retirement_members_unproven')
    aliases={}
    if cache_inventory is not None:
        _require(type(cache_inventory) is dict and set(cache_inventory)=={
            'members','files','directories','cache_aliases','unique_allocated_bytes'},'scene_retirement_members_unproven')
        cache_rows=cache_inventory['cache_aliases']
        targets=consent.get('cache_objects',[])
        _require(type(cache_rows) is list and len(cache_rows)<=256 and type(targets) is list
                 and len(targets)==len(cache_rows),'scene_retirement_members_unproven')
        targets={row['canonical_path']:(row['digest'],row['size_bytes']) for row in targets}
        _require(len(targets)==len(cache_rows)
                 and [row['path'] for row in cache_inventory['members']]==[str(root) for root in roots],
                 'scene_retirement_members_unproven')
        for alias in cache_rows:
            _require(type(alias) is dict and targets.get(alias['path'])==(alias['digest'],alias['size_bytes'])
                     and alias['path'] not in aliases,'scene_retirement_members_unproven')
            aliases[alias['path']]=alias
    cache_matched=set()
    matched=set()
    shared_keeps=[]
    observed={row['path'] for row in rows if type(row) is dict and row.get('status')=='observed_scoped_metadata'}
    for row in rows:
        if allowance is not None:
            allowance.tick()
        _require(type(row) is dict and type(row.get('path')) is str,'scene_retirement_members_unproven')
        path=_canonical(row['path'])
        owners=[index for index,root in enumerate(roots) if path.is_relative_to(root)]
        if not owners and str(path) in aliases:
            _require(row.get('status')=='observed_scoped_metadata'
                     and row.get('storage_class') in {'cache','host'}
                     and type(row.get('kinds')) in (list,_Rows)
                     and 'prepared_cache_object' in row['kinds']
                     and type(row.get('keeps')) in (list,_Rows)
                     and set(row['keeps'])<= {'shared_content_object_not_exclusive',
                         'external_hardlink_or_unobserved_alias'},'scene_retirement_shared_or_unresolved_member')
            cache_matched.add(str(path))
            continue
        reasons=row.get('keeps')
        observed_shared=(row.get('status')=='observed_scoped_metadata'
                         and reasons==['shared_content_object_not_exclusive'])
        absent_shared=(row.get('status')=='incomplete_scoped_metadata'
                       and type(reasons) in (list,_Rows) and len(reasons)==2
                       and set(reasons)=={'shared_content_object_not_exclusive',
                                          'member_or_child_unavailable_or_changed'}
                       and row.get('physical_identity') is None)
        if (not owners and str(path) not in aliases
                and type(row.get('kinds')) in (list,_Rows)
                and 'prepared_cache_object' in row['kinds']
                and (observed_shared or absent_shared)):
            # A shared object outside every requested directory is not in
            # this removal union. An absent prepared
            # cache object retains both its unavailable and shared KEEP.
            # No alias, changed child, or additional reason is cleared.
            shared_keeps.append({'canonical_path':str(path),'action':'KEEP',
                'observation_status':row['status'],'reasons':list(reasons)})
            continue
        _require(len(owners)==1,'scene_retirement_members_unproven')
        if row.get('status')=='coalesced_descendant_member':
            attributed=_canonical(row.get('attributed_root'))
            _require(str(attributed) in observed and path!=attributed and path.is_relative_to(attributed)
                     and attributed.is_relative_to(roots[owners[0]]) and not row.get('keeps'),
                     'scene_retirement_members_unproven')
            continue
        _require(row.get('status')=='observed_scoped_metadata','scene_retirement_members_unproven')
        # Shared scratch/release or an unselected external alias is not owned by
        # a consented parent. Positive keep reasons remain independent blockers.
        keeps=row.get('keeps',[])
        _require(type(keeps) in (list,_Rows) and len(keeps)<=10000
                 and all(type(reason) is str for reason in keeps),
                 'scene_retirement_shared_or_unresolved_member')
        # The actual authenticated owner scope supplies the explicit retention
        # decision; private preservation is still proved before any mutation.
        # No metadata flag is cleared, and no other keep reason is waived.
        retention={'sam_evidence_retention_policy_required'} if (
            selected[owners[0]]['class'] in consent['private_archive_classes']
            and row.get('storage_class')!='cache') else set()
        if cache_inventory is not None:
            # Every regular inode in this exact folder union was observed with
            # nlink == inventoried projections + separately selected aliases.
            # Logical/current other-owner references still independently KEEP.
            retention.add('external_hardlink_or_unobserved_alias')
        _require(set(keeps)<=retention,'scene_retirement_shared_or_unresolved_member')
        matched.add(owners[0])
    _require(len(matched)==len(selected) and cache_matched==set(aliases),'scene_retirement_members_unproven')
    return shared_keeps


def _current_plan(policy, consent, retained, allowance, now, monotonic):
    context=retained.get('planner_context')
    _require(type(context) is dict and policy.get('reference_context')==context,
             'scene_retirement_installed_context_unproven')
    _require(retained.get('schema_version')=='task_evaluation_scene_lifecycle_plan.v1'
             and retained.get('intent_id')==consent['intent_id'],'scene_retirement_plan_invalid')
    allowance.tick()
    fresh=build_scene_lifecycle_plan(intent_id=consent['intent_id'],context=context,
                                    observed_at_epoch=now(),monotonic=monotonic)
    allowance.tick()
    _require(fresh.get('finished_observation',{}).get('status') in _TERMINAL
             and 'historical_lineage' in fresh,'scene_retirement_not_finished')
    provenance=fresh.get('selected_intent_provenance')
    _require(type(provenance) is dict and provenance.get('role')=='intent'
             and {key:provenance.get(key) for key in ('path','sha256','size_bytes')}==consent['intent_raw_ref'],
             'scene_retirement_owner_changed')
    cache_inventory=None
    if consent.get('cache_objects'):
        from .task_evaluation_scene_retirement_cache import validate_cache_objects
        from .task_evaluation_scene_retirement_preservation import _inventory_members
        cache_targets,recipe_proofs=validate_cache_objects(policy,consent,allowance,fresh=fresh)
        fresh['recipe_stage_authority']=recipe_proofs
        cache_inventory=_inventory_members([row['canonical_path'] for row in consent['members']],allowance,
            cache_aliases=[{key:row[key] for key in ('canonical_path','digest','size_bytes')} for row in cache_targets])
    fresh['unselected_shared_content_keeps']=_plan_members(
        fresh,consent,allowance,cache_inventory=cache_inventory)
    _installed_cohort(policy,allowance)
    _require(not fresh.get('other_owner_capture_keeps'),
             'scene_retirement_reference_protected')
    # Every intersecting local keep must be retained by an exact selected proof;
    # this is pending preservation, never global reference/reader clearance.
    fresh['reference_transfer']=validate_current_reference_transfer(fresh,allowance,policy=policy,consent=consent)
    _current_readers(policy,allowance)
    return fresh


def _current_readers(policy,allowance):
    from . import task_evaluation_scene_retirement_supervisor as native
    allowance.tick()
    callback=getattr(native,'require_current_reader_closure',None)
    _require(callable(callback),'scene_retirement_reader_closure_unproven')
    # Native admission must raise for every unknown/manual/old/HTTP gap.
    # No source catalogue, policy flag or diagnostic observation can replace it.
    callback(policy,allowance)
    allowance.tick()


def _resume_current_references(policy,consent,retained,allowance,now,monotonic,*,preserved,pin_journal=None,
                               recipe_stage_authority=None):
    from .control_plane_reference_budget import ReferenceCollectionBudget
    from .task_evaluation_scene_lineage_budget import RetainedEmissionBudget
    from .task_evaluation_scene_lifecycle_plan import _context
    from .task_evaluation_scene_lifecycle_references import observe
    context=retained.get('planner_context')
    _require(type(context) is dict and policy.get('reference_context')==context,
             'scene_retirement_installed_context_unproven')
    _require(retained.get('schema_version')=='task_evaluation_scene_lifecycle_plan.v1'
             and retained.get('intent_id')==consent['intent_id'], 'scene_retirement_plan_invalid')
    _installed_cohort(policy,allowance)
    budget=ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=monotonic,time_budget_seconds=30)
    sink=RetainedEmissionBudget(max_bytes=16*1024*1024,max_rows=10000,max_references=10000,work_budget=budget)
    try:
        _context(context,budget)
        observation=observe(context,now(),budget,sink)
        allowance.tick()
        current=dict(retained,reference_observation=observation)
        if recipe_stage_authority is not None:
            current['recipe_stage_authority']=recipe_stage_authority
        validate_current_reference_transfer(current,allowance,
                                            preserved=preserved,policy=policy,consent=consent,pin_journal=pin_journal)
        _current_readers(policy,allowance)
    finally:
        budget.close()


def _partial_result(reason,journal,outcomes,policy,consent,pending,allowance,restore_context=None,preparation=None):
    result=_kept(reason,journal=journal,members=outcomes)
    if preparation is not None and journal is None:
        result.update(status='incomplete',mutations=None,token=preparation['claim']['token'],
            preparation_claim_raw_ref=preparation['claim_raw_ref'],
            last_preparation_escrow_raw_ref=preparation['prior_raw_ref'],
            preparation_budget_counts=dict(allowance.counts),
            preparation_budget_method='conservative_escrow_plus_observed_physical_work')
    if journal is None or pending is None:
        return result
    # The action context has unwound. Reacquire actual EX before advancing a
    # readable projection; a refused clock or a live reader leaves the already
    # durable private chain and prior public version untouched.
    try:
        with access.exclusive_scene_access() as locked:
            _require(locked==policy,'scene_retirement_policy_changed')
            progress=dict(status='incomplete',
                token=journal.token,intent_id=consent['intent_id'],members=outcomes,
                last_event_raw_ref=journal.prior_ref)
            if restore_context is not None:
                progress.update(restore_context)
            reference=publish_progress_receipt(policy,consent,pending,progress,allowance)
        result['intent_receipt_raw_ref']=reference
    except (ValueError,OSError,KeyError,TypeError,AttributeError):
        result['receipt_finalization']='unavailable_existing_durable_evidence_retained'
    return result


def _cache_generations(policy,initial,journal,allowance):
    targets=initial.get('cache_objects',[])
    original=initial.get('cache_generations',[])
    _require(type(targets) is list and type(original) is list and len(targets)==len(original)<=256,
             'scene_retirement_cache_journal_unproven')
    current=[]
    for index,(target,prior) in enumerate(zip(targets,original)):
        allowance.tick()
        key=hashlib.sha256(target['canonical_path'].encode()).hexdigest()+'.json'
        value,reference=load_document(Path(policy['generation_store'])/key,maximum=65536)
        _require(value.get('schema_version')=='scene_content_generation.v1'
                 and value.get('state_digest')==canonical_digest(value,digest_field='state_digest')
                 and all(value.get(field)==target[field] for field in
                     ('canonical_path','generation_id','digest','size_bytes'))
                 and value.get('source_publication_raw_ref')==target['source_raw_ref'],
                 'scene_retirement_generation_changed')
        if value.get('state') in {'active','restored-active'}:
            _require(value==prior and reference==target['generation_raw_ref'],
                     'scene_retirement_generation_changed')
        else:
            _require(value.get('state') in {'retiring','retired'} and value.get('retirement_token')==journal.token,
                     'scene_retirement_generation_changed')
            events=[event for event in journal.events if event['member_key']=='cache-'+str(index)
                and event['raw_ref']['sha256']==value.get('journal_sha256')]
            _require(len(events)==1 and events[0]['event']==(
                'retiring' if value['state']=='retiring' else 'cache_unlinked'),
                'scene_retirement_generation_changed')
            proof=events[0]['evidence']
            if value['state']=='retiring':
                _require(proof.get('generation_id')==value['generation_id'],'scene_retirement_generation_changed')
            else:
                _require(all(proof.get(field)==value[field] for field in
                    ('canonical_path','digest','size_bytes')),'scene_retirement_generation_changed')
        current.append(value)
    return current


def _cache_remove_records(initial):
    preserved=initial['preserved']
    targets=initial.get('cache_objects',[])
    aliases=preserved.get('cache_aliases',[])
    _require(len(targets)==len(aliases)<=256,'scene_retirement_cache_journal_unproven')
    for index,(target,alias) in enumerate(zip(targets,aliases)):
        _require(target['canonical_path']==alias['path'] and target['digest']==alias['digest']
                 and target['size_bytes']==alias['size_bytes'],'scene_retirement_cache_journal_unproven')
        key='cache-'+str(index)
        yield 'retiring',key,{'generation_id':target['generation_id']}
        outcome=dict(outcome='removed',canonical_path=alias['path'],digest=alias['digest'],
            size_bytes=alias['size_bytes'],removed_allocated_bytes=2**63-1,
            allocation_method='observed_file_st_blocks_512_last_union_link_unlinked')
        snapshot=list(alias['snapshot'])
        snapshot[-2],snapshot[-1]=2**63-1,1
        yield 'cache_unlink_planned',key,dict(canonical_path=alias['path'],
            original_identity=alias['physical_identity'],parent_identity=alias['parent_identity'],
            snapshot=snapshot,outcome=outcome)
        yield 'cache_unlinked',key,outcome


def _finish_retirement(policy,consent,initial,journal,pending,generations,outcomes,allowance,*,resumed=False):
    preserved=initial['preserved']
    closure=initial['metadata_closure_raw_ref']
    token=journal.token
    removed=recovery.removed_inode_counts(journal) if resumed else {}
    cache_generations=_cache_generations(policy,initial,journal,allowance)
    # Preflight ONE remaining folder+cache suffix, never separate allowances
    # that each fit while their combined retained events exceed the journal.
    def remaining_records():
        existing=set()
        for event in journal.events:
            allowance.tick()
            existing.add((event['event'],event['member_key'],event['evidence'].get('relative_path')))
        def records():
            for index,generation in enumerate(generations):
                yield 'retiring',str(index),{'generation_id':generation['generation_id'],
                    'inventory_sha256':consent['members'][index]['inventory_sha256']}
                with _opened(Path(consent['members'][index]['canonical_path']).parent,directory=True) as (_,info):
                    yield from removal_records(preserved,index,generation['generation_id'],journal,_identity(info))
            yield from _cache_remove_records(initial)
            yield from pins.pin_records(initial.get('terminal_pin_release_rows',[]))
        for event,key,evidence in records():
            allowance.tick()
            if (event,key,evidence.get('relative_path')) not in existing:
                yield event,key,evidence
    journal.preflight(remaining_records())
    pin_rows=initial.get('terminal_pin_release_rows',[])
    pin_outcomes=pins.release_terminal_pins(policy,consent,pin_rows,journal=journal,pending_raw_ref=pending) if pin_rows else []
    for index,generation in enumerate(cache_generations):
        if generation['state'] in {'active','restored-active'}:
            event=journal.append('retiring',member_key='cache-'+str(index),
                evidence={'generation_id':generation['generation_id']})
            cache_generations[index]=_transition(policy,generation,state='retiring',token=token,journal_ref=event)
    for index,generation in enumerate(generations):
        allowance.tick()
        outcome=detach_and_remove(preserved,member_index=index,generation_id=generation['generation_id'],
                  journal=journal,removed_inodes=removed)
        outcomes.append(outcome)
        if generation['state']!='retired':
            generations[index]=_transition(policy,generation,state='retired',token=token,
                    journal_ref=outcome['event_raw_ref'])
        # The member event and generation transition are already durable. A
        # failure projects their exact journal prefix through _partial_result;
        # a successful run publishes the terminal receipt once. Revalidating
        # the full protected journal/cache for every member is quadratic.
    from .task_evaluation_scene_retirement_cache import remove_preserved_cache_aliases
    cache_outcomes=remove_preserved_cache_aliases(preserved,journal=journal,removed_inodes=removed)
    _require(len(cache_outcomes)==len(cache_generations),'scene_retirement_cache_journal_unproven')
    for index,(generation,outcome) in enumerate(zip(cache_generations,cache_outcomes)):
        if generation['state']!='retired':
            cache_generations[index]=_transition(policy,generation,state='retired',token=token,
                journal_ref=outcome['event_raw_ref'])
    extra=dict(cache_outcomes=cache_outcomes,cache_generations=cache_generations) if cache_generations else {}
    if pin_rows:
        extra['terminal_pin_outcomes']=pin_outcomes
    snapshot=journal.retired_snapshot(dict(initial,status='retired',members=consent['members'],
                   outcomes=outcomes,generations=generations,**extra))
    receipt=dict(schema_version='scene_retirement_receipt.v1',status='retired',intent_id=consent['intent_id'],
        token=token,members=outcomes,retired_journal_raw_ref=snapshot,fresh_remote_readback_verified=True,
        removed_allocated_bytes=sum(row['removed_allocated_bytes'] for row in outcomes+cache_outcomes),
        logical_bytes=sum(row['logical_bytes'] for row in outcomes),
        planned_unique_allocated_bytes=preserved['unique_allocated_bytes'],
        metadata_closure_raw_ref=closure,
        unselected_shared_content_keeps=initial.get('unselected_shared_content_keeps',[]),**extra)
    receipt['intent_receipt_raw_ref']=publish_terminal_receipt(policy,consent,pending,receipt,allowance)
    receipt['intent_receipt_path']=receipt['intent_receipt_raw_ref']['path']
    return receipt


def retire_scene(plan_path, consent_path, *, transport, now=time.time, monotonic=time.monotonic):
    preparation=None
    journal=None
    pending=None
    policy=consent=allowance=None
    outcomes=[]
    try:
        authority=load_authority(consent_path,action='retire',now=now)
        policy,consent=authority['policy'],authority['consent']
        allowance=_allowance(authority,now,monotonic)
        _bind_transport(transport,allowance)
        _require(str(_canonical(str(plan_path)))==consent['plan_raw_ref']['path'],
                 'scene_retirement_raw_reference_changed')
        with access.exclusive_scene_access() as locked, pins.terminal_pin_guard(policy,consent,allowance):
            _require(locked==policy,'scene_retirement_policy_changed')
            # Reload both authorities after EX; none of the earlier observation
            # can grant action if the installed records changed while waiting.
            current=load_authority(consent_path,action='retire',now=now)
            _require(current==authority,'scene_retirement_policy_changed')
            retained=selected_document(consent['plan_raw_ref'],maximum=16*1024*1024)
            for member in consent['members']:
                if 'capture_owner_user_id' in member:
                    _require(retained.get('selected_intent_provenance') is not None
                             and {key:retained['selected_intent_provenance'].get(key)
                                  for key in ('path','sha256','size_bytes')}==consent['intent_raw_ref'],
                             'scene_retirement_capture_association_unproven')
                    _capture_action_current(policy,consent,member,allowance,expected_states={
                        'active','restored-active','retiring','retired'})
            resumed=recovery.select_retirement(policy,authority,allowance)
            if resumed is not None and type(resumed) is tuple:
                journal,pending,initial=resumed
                _resume_current_references(policy,consent,retained,allowance,now,monotonic,
                                           preserved=initial['preserved'],pin_journal=journal,
                                           recipe_stage_authority=initial['reference_transfer'].get('recipe_stage_authority',[]))
                generations=recovery.resumed_generations(sys.modules[__name__],policy,consent,journal,initial)
                published_objects=initial.get('declared_byte_verification',{}).get('published_objects',[])
                recovery.reserve_phase(journal,initial['preserved'],readback=True,published_objects=published_objects)
                _consume(initial['preserved'],transport,allowance)
                verify_publication_rows(published_objects,transport,allowance)
                recovery.reserve_phase(journal,initial['preserved'],pin_rows=initial.get('terminal_pin_release_rows',[]))
                for index,generation in enumerate(generations):
                    if generation['state'] in {'active','restored-active'}:
                        event=journal.append('retiring',member_key=str(index),evidence={
                            'generation_id':generation['generation_id'],
                            'inventory_sha256':consent['members'][index]['inventory_sha256']})
                        generations[index]=_transition(policy,generation,state='retiring',token=journal.token,
                            journal_ref=event,inventory_sha256=consent['members'][index]['inventory_sha256'])
                return _finish_retirement(policy,consent,initial,journal,pending,generations,outcomes,allowance,resumed=True)
            fresh=_current_plan(policy,consent,retained,allowance,now,monotonic)
            cache_generations=[]
            cache_targets=consent.get('cache_objects',[])
            for target in cache_targets:
                allowance.tick()
                allowance.charge('local_bytes',target['generation_raw_ref']['size_bytes'])
                cache_generations.append(selected_document(target['generation_raw_ref'],maximum=65536))
            generations=[]
            for member in consent['members']:
                allowance.tick()
                generation,_=_generation(policy,member,expected_states={'active','restored-active'})
                with _opened(member['canonical_path'],directory=True) as (_,info):
                    _require(_identity(info)==(member['dev'],member['ino'],member['mode'])
                             ==(generation['dev'],generation['ino'],generation['mode']),
                             'scene_retirement_generation_changed')
                generations.append(generation)
            if resumed is None:
                token=secrets.token_hex(32)[:32]
                preparation=recovery.claim_retirement(policy,authority,token,allowance)
            else:
                preparation=resumed
                token=preparation['claim']['token']
            if preparation['ready'] is None:
                name=token+'.'+str(preparation['archive_index']+1)+'.tar'
                def escrow(files,directories,archive_bytes):
                    selected=recovery.preparation_escrow(policy,preparation,allowance,phase='archive',
                        local_bytes=2*sum(row['size_bytes'] for row in files),archive_bytes=archive_bytes,
                        remote_bytes=2*archive_bytes+1)
                    _require(selected==name,'scene_retirement_preparation_resume_unproven')
                def before_upload(inventory):
                    for index,member in enumerate(consent['members']):
                        _require(inventory_digest(inventory,index)==member['inventory_sha256'],
                                 'scene_retirement_inventory_changed')
                    _verify_declared_bytes(fresh,inventory,transport,allowance,verify_remote=False)
                preserved=preserve_members([member['canonical_path'] for member in consent['members']],
                    transport=transport,allowance=allowance,token=token,before_payload=escrow,
                    before_upload=before_upload,archive_name=name,cache_aliases=(
                        [{key:row[key] for key in ('canonical_path','digest','size_bytes')} for row in cache_targets]
                        if cache_targets else None))
                recovery.complete_preparation(policy,preparation,preserved,allowance)
            else:
                preserved=preparation['ready']
                _require([row['path'] for row in preserved['members']]==[
                    member['canonical_path'] for member in consent['members']],
                    'scene_retirement_preparation_resume_unproven')
                recovery.preparation_escrow(policy,preparation,allowance,phase='ready-revalidation',
                    remote_bytes=preserved['archive']['size_bytes']+1)
                _consume(preserved,transport,allowance)
            verified_bytes=_verify_declared_bytes(fresh,preserved,transport,allowance,verify_remote=False)
            verified_references=validate_current_reference_transfer(fresh,allowance,preserved=preserved,policy=policy,consent=consent)
            _require(verified_references['archive_inventory_verified'] is True,
                     'scene_retirement_reference_closure_unproven')
            recovery.preparation_escrow(policy,preparation,allowance,phase='finalize',
                local_bytes=2*sum(row['size_bytes'] for row in preserved['files']),
                remote_bytes=sum(row['size_bytes']+1 for row in verified_bytes['published_objects']))
            verify_publication_rows(verified_bytes['published_objects'],transport,allowance)
            for index,member in enumerate(consent['members']):
                _require(inventory_digest(preserved,index)==member['inventory_sha256'],
                         'scene_retirement_inventory_changed')
                _require(member['class'] in consent['private_archive_classes'],
                         'scene_retirement_private_archive_denied')
            closure=retain_metadata_closure(preserved,policy,generations,token,allowance)
            initial=dict(schema_version='scene_retirement_journal.v1',intent_id=consent['intent_id'],
                intent_raw_ref=consent['intent_raw_ref'],plan_raw_ref=consent['plan_raw_ref'],
                consent_raw_ref=authority['consent_raw_ref'],policy_sha256=consent['policy_sha256'],
                cohort_sha256=consent['cohort_sha256'],status='pending',members=consent['members'],
                generations=generations,preserved=preserved,metadata_closure_raw_ref=closure,
                action_allowance=allowance.checkpoint(),declared_byte_verification=verified_bytes,
                reference_transfer=verified_references,
                unselected_shared_content_keeps=fresh['unselected_shared_content_keeps'])
            if cache_targets:
                initial.update(cache_objects=cache_targets,cache_generations=cache_generations)
            if verified_references.get('terminal_pin_release_rows'):
                initial['terminal_pin_release_rows']=verified_references['terminal_pin_release_rows']
            journal=SceneJournal.create(policy['journal_store'],token=token,initial=initial,allowance=allowance)
            pending=publish_pending_receipt(policy,consent,journal,preserved,allowance)
            def complete_records():
                for index,generation in enumerate(generations):
                    yield 'retiring',str(index),{
                        'generation_id':generation['generation_id'],
                        'inventory_sha256':consent['members'][index]['inventory_sha256']}
                    with _opened(consent['members'][index]['canonical_path'],directory=True) as (_,info):
                        _require(_identity(info)==tuple(preserved['members'][index]['physical_identity']),
                                 'scene_retirement_generation_changed')
                    with _opened(Path(consent['members'][index]['canonical_path']).parent,directory=True) as (_,info):
                        yield from removal_records(preserved,index,generation['generation_id'],journal,_identity(info))
                yield from _cache_remove_records(initial)
                yield from pins.pin_records(initial.get('terminal_pin_release_rows',[]))
            journal.preflight(complete_records())
            recovery.reserve_phase(journal,preserved,pin_rows=initial.get('terminal_pin_release_rows',[]))
            for index,generation in enumerate(generations):
                event=journal.append('retiring',member_key=str(index),evidence={
                    'generation_id':generation['generation_id'],'inventory_sha256':consent['members'][index]['inventory_sha256']})
                generations[index]=_transition(policy,generation,state='retiring',token=token,journal_ref=event,
                    inventory_sha256=consent['members'][index]['inventory_sha256'])
            return _finish_retirement(policy,consent,initial,journal,pending,generations,outcomes,allowance)
    except access.SceneRetirementAccessError as error:
        code=str(error) if str(error).startswith('scene_retirement_') and len(str(error))<=128 else 'scene_retirement_action_unproven'
        return _partial_result(code,journal,outcomes,policy,consent,pending,allowance,preparation=preparation)
    except (ValueError,OSError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
        return _partial_result('scene_retirement_action_unproven',journal,outcomes,policy,consent,pending,allowance,preparation=preparation)


def _cache_restore_records(retired):
    identity=[2**64-1]*3
    aliases=retired['preserved'].get('cache_aliases',[])
    generations=retired.get('cache_generations',[])
    _require(len(aliases)==len(generations)<=256,'scene_retirement_cache_journal_unproven')
    for index,(alias,generation) in enumerate(zip(aliases,generations)):
        key='cache-'+str(index)
        yield 'restoring',key,{'generation_id':generation['generation_id']}
        evidence=dict(canonical_path=alias['path'],restore_identity=identity,
            source_path=str(Path(retired['preserved']['members'][alias['member_index']]['path'])/alias['relative_path']),
            parent_identity=identity,digest=alias['digest'],size_bytes=alias['size_bytes'])
        yield 'cache_restore_planned',key,evidence
        yield 'cache_alias_restored',key,evidence
        yield 'restored-active',key,evidence


def _cache_restore_generations(policy,retired,journal,allowance):
    targets=retired.get('cache_objects',[])
    prior=retired.get('cache_generations',[])
    aliases=retired['preserved'].get('cache_aliases',[])
    _require(type(targets) is list and type(prior) is list and type(aliases) is list
             and len(targets)==len(prior)==len(aliases)<=256,'scene_retirement_cache_journal_unproven')
    proofs={}
    for event in journal.events:
        allowance.tick()
        proofs[event['raw_ref']['sha256']]=event
    result=[]
    variable={'state','dev','ino','mode','state_sequence','journal_sha256','state_digest'}
    for index,(target,original,alias) in enumerate(zip(targets,prior,aliases)):
        allowance.tick()
        path=_canonical(target['canonical_path'])
        _require(any(path.is_relative_to(Path(row['root'])) for row in policy['roots'])
            and original.get('schema_version')=='scene_content_generation.v1' and original.get('state')=='retired'
            and original.get('retirement_token')==retired['token']
            and original.get('state_digest')==canonical_digest(original,digest_field='state_digest')
            and all(target[field]==original[field] for field in ('canonical_path','generation_id','digest','size_bytes'))
            and target['source_raw_ref']==original['source_publication_raw_ref']
            and alias['path']==str(path) and alias['digest']==target['digest']
            and alias['size_bytes']==target['size_bytes'],'scene_retirement_generation_changed')
        key=hashlib.sha256(str(path).encode()).hexdigest()+'.json'
        allowance.charge('local_bytes',65536)  # Conservative bound before protected generation acquisition.
        current,_=load_document(Path(policy['generation_store'])/key,maximum=65536)
        _require(type(current) is dict and set(current)==set(original)
            and current.get('state_digest')==canonical_digest(current,digest_field='state_digest')
            and all(current[field]==original[field] for field in set(original)-variable),
            'scene_retirement_generation_changed')
        if current['state']=='retired':
            _require(current==original,'scene_retirement_generation_changed')
        else:
            event=proofs.get(current.get('journal_sha256'))
            _require(current['state'] in {'restoring','restored-active'} and event is not None
                and event['member_key']=='cache-'+str(index) and event['event']==current['state'],
                'scene_retirement_generation_changed')
            if current['state']=='restoring':
                _require(event['evidence']=={'generation_id':current['generation_id']},
                    'scene_retirement_generation_changed')
            else:
                _require(all(event['evidence'].get(field)==current[field] for field in
                    ('canonical_path','digest','size_bytes'))
                    and event['evidence'].get('restore_identity')==[current[field] for field in ('dev','ino','mode')],
                    'scene_retirement_generation_changed')
        result.append(current)
    return result


def _finish_restore(policy,consent,retired,reference,journal,pending,restore_context,generations,outcomes,allowance,transport,*,was_restored=False):
    token=journal.token
    cache_generations=_cache_restore_generations(policy,retired,journal,allowance)
    existing=set()
    for event in journal.events:
        allowance.tick()
        existing.add((event['event'],event['member_key'],event['evidence'].get('relative_path')))
    def remaining():
        for event,key,evidence in pins.pin_records(retired.get('terminal_pin_release_rows',[]),
                restoring=True,outcomes=retired.get('terminal_pin_outcomes',[])):
            if (event,key,evidence.get('relative_path')) not in existing:
                yield event,key,evidence
        for event,key,evidence in _cache_restore_records(retired):
            allowance.tick()
            if (event,key,evidence.get('relative_path')) not in existing:
                yield event,key,evidence
        for event,key,evidence in restore_records(retired['preserved'],journal):
            allowance.tick()
            if (event,key,evidence.get('relative_path')) not in existing:
                yield event,key,evidence
        for index,generation in enumerate(generations):
            if generation['state']!='restored-active':
                yield 'restored-active',str(index),dict(canonical_path=consent['members'][index]['canonical_path'],
                    outcome='restored',restore_identity=[2**64-1]*3)
    journal.preflight(remaining())
    for index,generation in enumerate(cache_generations):
        if generation['state']=='retired':
            event=journal.append('restoring',member_key='cache-'+str(index),
                evidence={'generation_id':generation['generation_id']})
            cache_generations[index]=_transition(policy,generation,state='restoring',token=retired['token'],journal_ref=event)
    restored=restore_preserved_members(retired['preserved'],transport=transport,journal=journal)
    for index,(generation,outcome) in enumerate(zip(generations,restored)):
        if generation['state']!='restored-active':
            event=journal.append('restored-active',member_key=str(index),evidence=outcome)
            _transition(policy,generation,state='restored-active',token=retired['token'],journal_ref=event,
                identity=outcome['restore_identity'])
        else:
            _require(outcome['restore_identity']==[generation['dev'],generation['ino'],generation['mode']],
                     'scene_retirement_generation_changed')
        if index<len(outcomes):
            _require(outcomes[index]==outcome,'scene_retirement_restore_journal_unproven')
        else:
            outcomes.append(outcome)
        if not was_restored:
            pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=token,
                intent_id=consent['intent_id'],members=list(outcomes),last_event_raw_ref=journal.prior_ref,
                **restore_context),allowance)
    aliases=retired['preserved'].get('cache_aliases',[])
    alias_events={}
    for event in journal.events:
        allowance.tick()
        if event['event']=='cache_alias_restored':
            _require(event['member_key'] not in alias_events,'scene_retirement_cache_journal_unproven')
            alias_events[event['member_key']]=event
    cache_outcomes=[]
    for index,(generation,alias) in enumerate(zip(cache_generations,aliases)):
        allowance.tick()
        event=alias_events.get('cache-'+str(index))
        _require(event is not None,'scene_retirement_cache_journal_unproven')
        evidence=event['evidence']
        with _opened(alias['path']) as (_,info):
            _require(evidence['restore_identity']==list(_identity(info)) and info.st_size==alias['size_bytes']
                and (info.st_uid,info.st_gid)==(alias['uid'],alias['gid'])
                and stat.S_IMODE(info.st_mode)==alias['mode'],'scene_retirement_generation_changed')
        if generation['state']!='restored-active':
            event=journal.append('restored-active',member_key='cache-'+str(index),evidence=evidence)
            _transition(policy,generation,state='restored-active',token=retired['token'],journal_ref=event,
                identity=evidence['restore_identity'])
        cache_outcomes.append(dict(evidence,outcome='restored'))
    pin_rows=retired.get('terminal_pin_release_rows',[])
    pin_outcomes=pins.restore_terminal_pins(policy,pin_rows,retired.get('terminal_pin_outcomes',[]),journal=journal) if pin_rows else []
    receipt=dict(schema_version='scene_restore_receipt.v1',status='restored',intent_id=consent['intent_id'],
         token=token,members=outcomes,retired_journal_raw_ref=reference,
         unselected_shared_content_keeps=retired.get('unselected_shared_content_keeps',[]))
    if cache_outcomes:
        receipt['cache_outcomes']=cache_outcomes
    if pin_rows:
        receipt['terminal_pin_outcomes']=pin_outcomes
    receipt['intent_receipt_raw_ref']=publish_progress_receipt(policy,consent,pending,dict(status='restored',
        token=token,intent_id=consent['intent_id'],members=outcomes,last_event_raw_ref=journal.prior_ref,
        **restore_context),allowance)
    receipt['intent_receipt_path']=receipt['intent_receipt_raw_ref']['path']
    return receipt


def restore_scene(retired_journal_path, consent_path, *, transport, now=time.time, monotonic=time.monotonic):
    journal=None
    pending=restore_context=None
    policy=consent=allowance=None
    outcomes=[]
    try:
        authority=load_authority(consent_path,action='restore',now=now)
        policy,consent=authority['policy'],authority['consent']
        allowance=_allowance(authority,now,monotonic)
        _bind_transport(transport,allowance)
        reference=consent['retired_journal_raw_ref']
        _require(str(_canonical(str(retired_journal_path)))==reference['path'],'scene_retirement_raw_reference_changed')
        with access.exclusive_scene_access() as locked, ExitStack() as lifetimes:
            _require(locked==policy and load_authority(consent_path,action='restore',now=now)==authority,
                     'scene_retirement_policy_changed')
            retired=selected_document(reference,maximum=16*1024*1024,protected=True)
            _require(retired.get('schema_version')=='scene_retirement_journal.v1' and retired.get('status')=='retired'
                     and retired.get('journal_digest')==canonical_digest(retired,digest_field='journal_digest')
                     and retired.get('intent_id')==consent['intent_id'] and retired.get('members')==consent['members'],
                     'scene_retirement_restore_snapshot_invalid')
            for member in consent['members']:
                if 'capture_owner_user_id' in member:
                    _capture_action_current(policy,consent,member,allowance,expected_states={
                        'retired','restoring','restored-active'})
            pin_rows=retired.get('terminal_pin_release_rows',[])
            lifetimes.enter_context(pins.terminal_pin_guard(policy,dict(consent,terminal_pin_refs=[
                row['original_raw_ref'] for row in pin_rows]),allowance))
            resumed=recovery.select_restore(policy,authority,allowance,reference)
            if resumed is not None:
                journal,pending,initial,projection=resumed
                recovery.bind_original_allowance(journal,initial,allowance,restoring=True)
                _installed_cohort(policy,allowance)
                generations=recovery.resumed_restore_generations(sys.modules[__name__],policy,consent,journal,initial)
                outcomes.extend(recovery.restored_prefix(journal,generations))
                recovery.reserve_phase(journal,retired['preserved'],restoring=True,pin_rows=retired.get('terminal_pin_release_rows',[]))
                restore_context=dict(original_retirement_token=retired['token'],restore_journal_initial_raw_ref=journal.initial_ref)
                was_restored=projection['status']=='restored'
                if not was_restored:
                    pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=journal.token,
                        intent_id=consent['intent_id'],members=list(outcomes),last_event_raw_ref=journal.prior_ref,**restore_context),allowance)
                for index,generation in enumerate(generations):
                    if generation['state']=='retired':
                        event=journal.append('restoring',member_key=str(index),evidence={'generation_id':generation['generation_id']})
                        generations[index]=_transition(policy,generation,state='restoring',token=retired['token'],journal_ref=event)
                return _finish_restore(policy,consent,retired,reference,journal,pending,restore_context,
                                       generations,outcomes,allowance,transport,was_restored=was_restored)
            generations=[]
            for member in consent['members']:
                allowance.tick()
                generation,_=_generation(policy,member,expected_states={'retired'},retired_token=retired['token'])
                generations.append(generation)
                try:
                    os.lstat(member['canonical_path'])
                except FileNotFoundError:
                    pass
                else:
                    _require(False,'scene_retirement_restore_conflict')
            _installed_cohort(policy,allowance)
            token=secrets.token_hex(32)[:32]
            initial=dict(schema_version='scene_restore_journal.v1',status='restoring',intent_id=consent['intent_id'],
                intent_raw_ref=consent['intent_raw_ref'],members=consent['members'],generations=generations,
                original_retirement_token=retired['token'],retired_journal_raw_ref=reference,
                consent_raw_ref=authority['consent_raw_ref'],action_allowance=allowance.checkpoint())
            if retired.get('cache_objects'):
                initial.update(cache_objects=retired['cache_objects'],cache_generations=retired['cache_generations'])
            if pin_rows:
                initial.update(terminal_pin_release_rows=pin_rows,terminal_pin_outcomes=retired['terminal_pin_outcomes'])
            journal=SceneJournal.create(policy['journal_store'],token=token,initial=initial,allowance=allowance)
            def complete_restore_records():
                for index,generation in enumerate(generations):
                    yield 'restoring',str(index),{'generation_id':generation['generation_id']}
                    yield 'restored-active',str(index),dict(canonical_path=consent['members'][index]['canonical_path'],
                        outcome='restored',restore_identity=[2**64-1]*3)
                yield from restore_records(retired['preserved'],journal)
                yield from _cache_restore_records(retired)
                yield from pins.pin_records(pin_rows,restoring=True,outcomes=retired.get('terminal_pin_outcomes',[]))
            journal.preflight(complete_restore_records())
            recovery.reserve_phase(journal,retired['preserved'],restoring=True,pin_rows=retired.get('terminal_pin_release_rows',[]))
            receipt_path=Path(policy['reference_context']['roots']['intent_root'])/consent['intent_id']/'scene-retired.v1.json'
            allowance.tick()
            _,pending=load_document(receipt_path,maximum=16*1024*1024)
            restore_context=dict(original_retirement_token=retired['token'],restore_journal_initial_raw_ref=journal.initial_ref)
            pending=publish_progress_receipt(policy,consent,pending,dict(status='restoring',token=token,
                intent_id=consent['intent_id'],members=[],last_event_raw_ref=journal.prior_ref,**restore_context),allowance)
            for index,generation in enumerate(generations):
                event=journal.append('restoring',member_key=str(index),evidence={'generation_id':generation['generation_id']})
                generations[index]=_transition(policy,generation,state='restoring',token=retired['token'],journal_ref=event)
            return _finish_restore(policy,consent,retired,reference,journal,pending,restore_context,
                                   generations,outcomes,allowance,transport)
    except access.SceneRetirementAccessError as error:
        code=str(error) if str(error).startswith('scene_retirement_') and len(str(error))<=128 else 'scene_retirement_action_unproven'
        return _partial_result(code,journal,outcomes,policy,consent,pending,allowance,restore_context)
    except (ValueError,OSError,KeyError,TypeError,AttributeError,OverflowError,RecursionError):
        return _partial_result('scene_retirement_action_unproven',journal,outcomes,policy,consent,pending,allowance,restore_context)
