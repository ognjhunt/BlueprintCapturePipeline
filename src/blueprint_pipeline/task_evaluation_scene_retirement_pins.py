"""Engine-internal terminal pin selection; never process or action authority."""
import math
import hashlib
import os
import stat
import re

from .task_evaluation_scene_retirement_access import _canonical, _opened, _identity, _require
from .task_evaluation_scene_retirement_authority import CAPTURE_MEMBER_KEYS, selected_document
from .task_evaluation_scene_retirement_declared_bytes import _selector
from .task_evaluation_scene_downstream_contracts import bounded_size
from .task_evaluation_scene_lineage_budget import _Rows

_FIELDS={'schema_version','kind','owner_id','paths','depends_on','created_at_epoch','expires_at_epoch','released_at_epoch'}
_OBSERVED=_FIELDS-{'schema_version'}|{'status','row_path','raw_sha256','raw_size_bytes','row_identity'}
_ID=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,191}')
_REASON='scene_retirement_terminal_pin_unproven'
_SUCCESS={
    'preparation':('task_evaluation_launch_preparation_result.v1','preparation_id','preparation_input_root',{
        'inputs_materialized_awaiting_construction_adapter','native_arena_inputs_verified_awaiting_profile_authority',
        'queued_for_production_episode_compilation','queued_for_production_scene_configuration'}),
    'compilation':('task_evaluation_episode_compilation_result.v1','compilation_id','compilation_output_root',{
        'compiled_for_production_launch'}),
    'activation':('task_evaluation_launch_activation_result.v1','activation_id','activation_output_root',{
        'profile_authority_materialized_no_execution','policy_campaign_queue_materialized_no_execution'}),
}


def _rows(value,maximum):
    _require(type(value) in (list,_Rows) and len(value)<=maximum,_REASON)
    return value


def _finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def identity(observation):
    _require(type(observation) is dict and set(observation)==_OBSERVED,_REASON)
    _require(observation['kind'] in _SUCCESS and type(observation['owner_id']) is str
             and _ID.fullmatch(observation['owner_id']),_REASON)
    return _selector(observation,'row_path','raw_sha256','raw_size_bytes')


def select_terminal_pins(fresh,policy,consent,documents,allowance,*,history=None):
    """Read the already bounded native observation; never scan another ledger.

    Current reader admission and authenticated member generations remain engine
    requirements. These rows merely bind explicit pin releases to selected
    successful producer output and exactly owned paths before preservation.
    """
    selected=_rows(consent.get('terminal_pin_refs',[]),256)
    if not selected:
        return []
    context=fresh.get('planner_context')
    _require(type(context) is dict and policy.get('reference_context')==context
        and fresh.get('finished_observation',{}).get('status') in {
            'completed','revoked_grace_elapsed','expired_grace_elapsed'},_REASON)
    root=_canonical(context.get('pins_root'))
    roots=context.get('roots')
    _require(type(roots) is dict,_REASON)
    owner=fresh.get('selected_intent_provenance')
    _require(type(owner) is dict and _selector(owner)==_selector(consent['intent_raw_ref']),_REASON)
    members=_rows(consent['members'],256)
    for member in members:
        allowance.tick()
        if 'capture_owner_user_id' in member:
            # The engine separately authenticates the original capture owner.
            # Pin closure binds its existing scene sponsor without transferring ownership.
            _require(set(member)==CAPTURE_MEMBER_KEYS
                     and member.get('sponsoring_intent_id')==consent['intent_id']
                     and member.get('scene_intent_raw_ref')==consent['intent_raw_ref'],_REASON)
        else:
            _require(member.get('owner_intent_id')==consent['intent_id']
                     and member.get('owner_raw_ref')==consent['intent_raw_ref'],_REASON)
    values={}
    for protection in _rows(fresh['reference_observation']['protections'],10000):
        allowance.tick()
        if protection.get('kind')!='pin_observation':
            continue
        observed=protection['observation']
        raw=identity(observed)
        key=(observed['kind'],observed['owner_id'])
        _require(key not in values,_REASON)
        values[key]=(observed,raw)
    result=[]
    pending=set()
    emitted=0
    for reference in selected:
        allowance.tick()
        raw=_selector(reference)
        path=_canonical(raw[0])
        _require(path.is_relative_to(root) and len(path.relative_to(root).parts)==2
            and path.parent.name in _SUCCESS and path.suffix=='.json' and raw[2]<=16384,_REASON)
        key=(path.parent.name,path.stem)
        prior=None if history is None else history.get(raw)
        _require(key in values and key not in pending and (values[key][1]==raw or
            prior is not None and _selector(prior['observed_raw_ref'])==values[key][1]),_REASON)
        observed=values[key][0]
        current_ref=reference if prior is None else prior['observed_raw_ref']
        _require((observed['status'] in {'live','expired'} and observed['released_at_epoch'] is None
                or prior is not None and observed['status']=='released')
            and all(_finite(observed[field]) for field in ('created_at_epoch','expires_at_epoch'))
            and 0<=observed['created_at_epoch']<observed['expires_at_epoch']
            and allowance.last_wall-observed['created_at_epoch']>=6*60*60,_REASON)
        schema,field,root_key,statuses=_SUCCESS[key[0]]
        primary=str(_canonical(roots.get(root_key))/key[1])
        _require(any(value.get('schema_version')==schema and value.get(field)==key[1]
            and value.get('status') in statuses for value in documents.values()),_REASON)
        paths=_rows(observed['paths'],256)
        _require(paths and primary in paths and len(set(paths))==len(paths),_REASON)
        for pinned in paths:
            allowance.tick()
            target=_canonical(pinned)
            _require(any(target.is_relative_to(_canonical(member['canonical_path'])) for member in members),_REASON)
        _require(emitted+2*raw[2]+8192<=512*1024,_REASON)
        allowance.charge('local_bytes',raw[2])
        value=selected_document(current_ref,maximum=16384)
        _require(type(value) is dict and set(value)==_FIELDS and value['schema_version']=='control_plane_storage_pin.v1'
            and all(value[field]==observed[field] for field in _FIELDS-{'schema_version'}),_REASON)
        dependencies=_rows(value['depends_on'],256)
        _require(all(type(row) is dict and set(row)=={'kind','owner_id'} and row['kind'] in _SUCCESS
            and type(row['owner_id']) is str and _ID.fullmatch(row['owner_id']) for row in dependencies),_REASON)
        allowance.charge('local_bytes',raw[2])
        with _opened(path) as (fd,info):
            _require(stat.S_IMODE(info.st_mode)==0o640 and info.st_nlink==1 and tuple(observed['row_identity'])==(
                info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns),_REASON)
            before=(info.st_size,info.st_mtime_ns,info.st_ctime_ns,_identity(info))
            allowance.tick()
            contents=os.read(fd,info.st_size)
            after=os.fstat(fd)
            _require((after.st_size,after.st_mtime_ns,after.st_ctime_ns,_identity(after))==before
                and len(contents)==current_ref['size_bytes'] and 'sha256:'+hashlib.sha256(contents).hexdigest()==current_ref['sha256'],_REASON)
            row=dict(original_raw_ref=reference,original_value=value,original_raw_hex=contents.hex(),physical_identity=list(_identity(info)),
                snapshot=[info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_uid,info.st_gid,
                          info.st_mtime_ns,info.st_ctime_ns,info.st_nlink])
        if prior is not None:
            _require(row['snapshot']==prior['observed_snapshot'],_REASON)
            row=dict(prior)
        emitted+=bounded_size(row,512*1024-emitted)
        _require(emitted<=512*1024,_REASON)
        result.append(row)
        pending.add(key)
    for key,(observed,_) in values.items():
        allowance.tick()
        dependencies=_rows(observed['depends_on'],256)
        for dependency in dependencies:
            allowance.tick()
            _require(type(dependency) is dict and set(dependency)=={'kind','owner_id'},_REASON)
            parent=(dependency['kind'],dependency['owner_id'])
            # Expiry does not erase a dependent pin. Only a recorded release or
            # this exact owner-bound closure can stop its positive protection.
            if parent in pending:
                _require(key in pending or observed['status']=='released',_REASON)
            if key in pending and parent in values:
                _require(parent in pending or values[parent][0]['status']=='released',_REASON)
    return result


def covers(protection,rows):
    """Exact raw/path coverage only; caller clear flags are not accepted."""
    if not rows or type(protection) is not dict:
        return False
    if protection.get('kind')=='pin_observation':
        observed=protection.get('observation')
        raw=identity(observed)
        return any(_selector(row.get('observed_raw_ref',row['original_raw_ref']))==raw for row in rows)
    if protection.get('kind')!='positive_pin_path' or set(protection)!={'kind','path','source','binding_status','action'}:
        return False
    source=protection['source']
    _require(type(source) is dict and set(source)=={'row_path','raw_sha256','raw_size_bytes','row_identity'},_REASON)
    raw=_selector(source,'row_path','raw_sha256','raw_size_bytes')
    return protection['action']=='KEEP' and protection['binding_status']=='historical_positive_only' and any(
        _selector(row.get('observed_raw_ref',row['original_raw_ref']))==raw and protection['path'] in row['original_value']['paths']
        and tuple(source['row_identity'])==tuple(row.get('observed_snapshot',row['snapshot'])[index] for index in (0,1,3,6,7)) for row in rows)
