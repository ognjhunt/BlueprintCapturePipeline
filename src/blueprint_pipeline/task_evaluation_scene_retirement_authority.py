"""Protected installed retirement scopes and exact action-specific raw selectors."""
from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from . import task_evaluation_scene_retirement_access as access
from .task_evaluation_scene_retirement_access import _canonical, _document, _identity, _opened, _require
from .task_evaluation_scene_retirement_generations import _guard

SHA = re.compile(r'sha256:[0-9a-f]{64}')
TOKEN = re.compile(r'[0-9a-f]{32}')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}')
CONSENT_KEYS = {'schema_version','consent_id','principal_id','intent_id','intent_raw_ref',
    'plan_raw_ref','retired_journal_raw_ref','policy_sha256','cohort_sha256','action',
    'created_at','expires_at','members','private_archive_classes','consent_digest'}
CACHE_KEYS={'canonical_path','digest','size_bytes','generation_id','generation_raw_ref','source_raw_ref'}
OPTIONAL_CONSENT_KEYS={'cache_objects','terminal_pin_refs'}
MEMBER_KEYS = {'canonical_path','class','owner_intent_id','owner_raw_ref','generation_id',
               'dev','ino','mode','inventory_sha256'}


def raw_digest(raw):
    return 'sha256:'+hashlib.sha256(raw).hexdigest()


def load_document(path, *, maximum, protected=False):
    path = _canonical(str(path))
    with _opened(path,protected=protected) as (fd,info):
        _require(0 < info.st_size <= maximum,'scene_retirement_document_limit')
        if protected:
            _require(info.st_uid == access._POLICY_UID and stat.S_IMODE(info.st_mode) == 0o600,
                     'scene_retirement_authority_permissions')
        expected = _identity(info)
        chunks, remaining = [], info.st_size
        while remaining:
            _guard(fd,expected)
            raw = os.read(fd,min(1024*1024,remaining))
            _guard(fd,expected)
            _require(raw,'scene_retirement_document_changed')
            chunks.append(raw)
            remaining -= len(raw)
        _guard(fd,expected)
        after = os.fstat(fd)
        _require((after.st_size,after.st_mtime_ns,after.st_ctime_ns)==(info.st_size,info.st_mtime_ns,info.st_ctime_ns),
                 'scene_retirement_document_changed')
    raw = b''.join(chunks)
    return _document(raw),dict(path=str(path),sha256=raw_digest(raw),size_bytes=len(raw))


def raw_reference(value):
    _require(type(value) is dict and set(value)=={'path','sha256','size_bytes'},'scene_retirement_raw_reference_invalid')
    _canonical(value['path'])
    _require(type(value['sha256']) is str and SHA.fullmatch(value['sha256'])
             and type(value['size_bytes']) is int and 0 < value['size_bytes'] <= 32*1024*1024,
             'scene_retirement_raw_reference_invalid')
    return value


def selected_document(reference, *, maximum, protected=False):
    reference=raw_reference(reference)
    value, observed=load_document(reference['path'],maximum=maximum,protected=protected)
    _require(observed==reference,'scene_retirement_raw_reference_changed')
    return value


def cohort_digest(rows):
    return canonical_digest({'consumer_cohort':rows})


def _strings(value, *, maximum, grammar=ID):
    _require(type(value) is list and len(value)<=maximum)
    _require(all(type(item) is str and grammar.fullmatch(item) for item in value))
    _require(len(set(value))==len(value))
    return set(value)


def _scopes(policy):
    principals=policy['principals']
    _require(type(principals) is list and len(principals)<=64)
    rows, owner_count = {},0
    for row in principals:
        _require(type(row) is dict and {'principal_id','actions','owner_intent_ids',
                 'private_archive_classes'} <= set(row) <= {'principal_id','actions',
                 'owner_intent_ids','private_archive_classes','capture_owner_scopes'})
        _require(type(row['principal_id']) is str and ID.fullmatch(row['principal_id']) and row['principal_id'] not in rows)
        actions=_strings(row['actions'],maximum=2)
        _require(actions and actions <= {'retire','restore'})
        owners=_strings(row['owner_intent_ids'],maximum=256)
        capture_rows=row.get('capture_owner_scopes',[])
        _require(type(capture_rows) is list and len(capture_rows)<=256)
        captures=set()
        for capture in capture_rows:
            _require(type(capture) is dict and set(capture)=={'user_id','request_id'}
                     and all(type(capture[key]) is str and ID.fullmatch(capture[key])
                             for key in ('user_id','request_id')))
            pair=(capture['user_id'],capture['request_id'])
            _require(pair not in captures)
            captures.add(pair)
        owner_count += len(owners)+len(captures)
        _require(owner_count<=256)
        private=_strings(row['private_archive_classes'],maximum=64)
        rows[row['principal_id']] = dict(actions=actions,owners=owners,
                                         captures=captures,private=private)
    return rows


def load_authority(consent_path, *, action, now):
    """Load scope only; callers must still prove current references and members."""
    _require(action in {'retire','restore'},'scene_retirement_action_invalid')
    policy=access._policy()
    _require(policy is not None,'scene_retirement_disabled')
    policy_path=access._policy_path()
    # _policy has already proved installed UID/ancestry and 0644 reader format;
    # bind precisely those raw bytes and refuse a changed installed document.
    raw=access._bytes(policy_path,protected=True,required_mode=0o644)
    _require(_document(raw)==policy,'scene_retirement_policy_changed')
    scopes=_scopes(policy)
    allowed_private=_strings(policy['private_archive_allowed_classes'],maximum=64)
    cohort=policy['consumer_cohort']
    _require(type(cohort) is list and len(cohort)<=256)
    names=set()
    for row in cohort:
        _require(type(row) is dict and set(row)=={'entrypoint','installed_source_sha','lifetime_contract_version'})
        _require(type(row['entrypoint']) is str and len(row['entrypoint'])<=256 and row['entrypoint'] not in names)
        _require(type(row['installed_source_sha']) is str and SHA.fullmatch(row['installed_source_sha'])
                 and row['lifetime_contract_version']=='scene_retirement_lifetime.v1')
        names.add(row['entrypoint'])
    consent,consent_ref=load_document(consent_path,maximum=512*1024,protected=True)
    _require(CONSENT_KEYS<=set(consent)<=CONSENT_KEYS|OPTIONAL_CONSENT_KEYS
             and consent['schema_version']=='scene_retirement_consent.v1')
    _require(type(consent['consent_id']) is str and TOKEN.fullmatch(consent['consent_id']))
    _require(consent['consent_digest']==canonical_digest(consent,digest_field='consent_digest'))
    _require(consent['action']==action,'scene_retirement_consent_action_invalid')
    scope=scopes.get(consent['principal_id'])
    _require(scope is not None and action in scope['actions'],'scene_retirement_principal_untrusted')
    _require(type(consent['intent_id']) is str and ID.fullmatch(consent['intent_id'])
             and consent['intent_id'] in scope['owners'],'scene_retirement_owner_scope_denied')
    raw_reference(consent['intent_raw_ref'])
    if action=='retire':
        _require(consent['plan_raw_ref'] is not None and consent['retired_journal_raw_ref'] is None,
                 'scene_retirement_consent_selector_invalid')
        raw_reference(consent['plan_raw_ref'])
    else:
        _require(consent['retired_journal_raw_ref'] is not None and consent['plan_raw_ref'] is None,
                 'scene_retirement_consent_selector_invalid')
        raw_reference(consent['retired_journal_raw_ref'])
        journal=Path(consent['retired_journal_raw_ref']['path'])
        _require(journal.parent==Path(policy['journal_store'])/'retired'
                 and journal.name==consent['retired_journal_raw_ref']['sha256'][7:]+'.json',
                 'scene_retirement_restore_snapshot_invalid')
    _require(consent['policy_sha256']==raw_digest(raw) and consent['cohort_sha256']==cohort_digest(cohort),
             'scene_retirement_policy_changed')
    _require(type(consent['created_at']) is int and type(consent['expires_at']) is int
             and 0 <= consent['created_at'] < consent['expires_at'])
    try:
        observed=now()
        _require(type(observed) in (int,float) and math.isfinite(observed)
                 and consent['created_at']<=observed<consent['expires_at'],
                 'scene_retirement_consent_expired')
    except (OSError,OverflowError,TypeError) as error:
        raise access.SceneRetirementAccessError('scene_retirement_clock_unproven') from error
    private=_strings(consent['private_archive_classes'],maximum=64)
    _require(private <= scope['private'] & allowed_private,'scene_retirement_private_archive_denied')
    members=consent['members']
    _require(type(members) is list and len(members)<=256)
    paths=set()
    for member in members:
        _require(type(member) is dict and set(member)==MEMBER_KEYS)
        path=_canonical(member['canonical_path'])
        _require(str(path) not in paths and member['owner_intent_id'] in scope['owners'])
        paths.add(str(path))
        _require(type(member['class']) is str and ID.fullmatch(member['class'])
                 and type(member['generation_id']) is str and TOKEN.fullmatch(member['generation_id']))
        raw_reference(member['owner_raw_ref'])
        _require(all(type(member[key]) is int and member[key]>=0 for key in ('dev','ino','mode'))
                 and type(member['inventory_sha256']) is str and SHA.fullmatch(member['inventory_sha256']))
        _require(any(path.is_relative_to(Path(row['root'])) and member['dev']==row['device'] for row in policy['roots']))
        _require(not any(path==Path(policy[key]) or path.is_relative_to(Path(policy[key]))
                         or Path(policy[key]).is_relative_to(path) for key in ('coordinator_path','generation_store','journal_store')))
    caches=consent.get('cache_objects',[])
    pins=consent.get('terminal_pin_refs',[])
    _require(type(caches) is list and len(caches)<=256 and type(pins) is list and len(pins)<=256)
    for row in caches:
        _require(type(row) is dict and set(row)==CACHE_KEYS)
        path=_canonical(row['canonical_path'])
        _require(str(path) not in paths and type(row['digest']) is str and SHA.fullmatch(row['digest'])
                 and path.name==row['digest'][7:] and type(row['size_bytes']) is int and row['size_bytes']>=0
                 and type(row['generation_id']) is str and TOKEN.fullmatch(row['generation_id'])
                 and any(path.is_relative_to(Path(root['root'])) for root in policy['roots']))
        paths.add(str(path))
        raw_reference(row['generation_raw_ref'])
        raw_reference(row['source_raw_ref'])
        _require(row['generation_raw_ref']['path']==str(Path(policy['generation_store'])/(
            hashlib.sha256(str(path).encode()).hexdigest()+'.json')))
    pin_paths=set()
    for row in pins:
        raw_reference(row)
        _require(row['path'] not in pin_paths)
        pin_paths.add(row['path'])
    return dict(policy=policy,consent=consent,consent_raw_ref=consent_ref,scope=scope)
