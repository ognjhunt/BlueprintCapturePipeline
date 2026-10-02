"""Bind declared scene byte promises to preservation and fresh publication GETs."""
import hashlib
import re
import sys
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_retirement_access import SceneRetirementAccessError, _canonical, _require
from .task_evaluation_scene_retirement_authority import selected_document
from .task_evaluation_scene_retirement_preservation import CHUNK
from .task_evaluation_scene_lineage_budget import _Rows

_SHA=re.compile(r'sha256:[0-9a-f]{64}\Z')
_NAMESPACE=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


def _objects(value,allowance):
    """Bound every occurrence before entering it; no bulk stack expansion."""
    stack=[iter((value,))]
    count=0
    while stack:
        allowance.tick()
        try:
            item=next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        count+=1
        _require(count<=100000,'scene_retirement_declared_reference_limit')
        if type(item) is dict:
            _require(len(stack)<64 and len(item)<=10000,'scene_retirement_declared_reference_limit')
            yield item
            stack.append(iter(item.values()))
        elif type(item) in (list,_Rows):
            _require(len(stack)<64 and len(item)<=10000,'scene_retirement_declared_reference_limit')
            stack.append(iter(item))
        elif isinstance(item,(dict,list,tuple,set)):
            # Only the exact native retained-row container is an accepted
            # collection extension. Foreign callbacks cannot hide obligations.
            _require(False,'scene_retirement_declared_reference_invalid')


def _selector(value,path_field='path',digest_field='sha256',size_field='size_bytes'):
    path=value[path_field]
    digest=value[digest_field]
    size=value[size_field]
    _require(type(path) is str and len(path)<=4096 and type(digest) is str and _SHA.fullmatch(digest)
             and type(size) is int and size>=0,'scene_retirement_declared_reference_invalid')
    return str(_canonical(path)),digest,size


def verify_publication_rows(rows,transport,allowance):
    _require(type(rows) is list and len(rows)<=10000,'scene_retirement_declared_reference_limit')
    read=getattr(transport,'read_published_object_charged',None)
    _require(not rows or callable(read),'scene_retirement_published_readback_unproven')
    for row in rows:
        allowance.tick()
        _require(row['size_bytes']<=allowance.limits['remote_bytes']-allowance.counts['remote_bytes'],
                 'scene_retirement_byte_limit')
        source=iter(read(row['uri'],allowance,expected_size_bytes=row['size_bytes']))
        digest=hashlib.sha256()
        count=0
        try:
            for chunk in source:
                allowance.tick()
                _require(type(chunk) is bytes and 0<len(chunk)<=CHUNK and count+len(chunk)<=row['size_bytes'],
                         'scene_retirement_published_readback_unproven')
                digest.update(chunk)
                count+=len(chunk)
            _require(count==row['size_bytes'] and 'sha256:'+digest.hexdigest()==row['digest'],
                     'scene_retirement_published_readback_unproven')
        finally:
            incoming=sys.exception()
            close=getattr(source,'close',None)
            if close is not None:
                try:
                    close()
                except Exception:
                    if incoming is None:
                        raise SceneRetirementAccessError('scene_retirement_remote_cleanup_unproven') from None
                    incoming.add_note('scene_retirement_remote_cleanup_unproven')
        allowance.tick()


def verify_declared_bytes(fresh,preserved,transport,allowance,*,verify_remote=True):
    physical={str(Path(preserved['members'][row['member_index']]['path'])/row['relative_path']):row
              for row in preserved['files']}
    local,publications={},{}
    for item in _objects(fresh.get('historical_lineage',{}),allowance):
        fields=None
        if {'path','sha256','size_bytes'}<=item.keys():
            fields=('path','sha256','size_bytes')
        elif {'path','raw_sha256','raw_size_bytes'}<=item.keys():
            fields=('path','raw_sha256','raw_size_bytes')
        if fields is not None:
            path,digest,size=_selector(item,*fields)
            if path in physical:
                row=physical[path]
                _require(row['sha256']==digest and row['size_bytes']==size,
                         'scene_retirement_declared_payload_changed')
                _require(len(local)<10000 or path in local,'scene_retirement_declared_reference_limit')
                local[path]=(digest,size)
        if 'publication_observations' not in item:
            continue
        _require(item.get('schema_version')=='task_evaluation_scene_source_family_inventory.v1',
                 'scene_retirement_publication_unproven')
        for observation in item['publication_observations']:
            allowance.tick()
            _require(observation.get('historical_publication_binding_verified') is True
                     and len(observation.get('source_provenance',[]))==1,
                     'scene_retirement_publication_unproven')
            proof=observation['source_provenance'][0]
            _require(proof['role']=='submission_publications','scene_retirement_publication_unproven')
            path,digest,size=_selector(proof)
            receipt=selected_document(dict(path=path,sha256=digest,size_bytes=size),maximum=4*1024*1024)
            allowance.tick()
            namespace=receipt.get('input_namespace')
            _require(receipt.get('schema_version')=='task_evaluation_scene_configuration_submission_publication.v1'
                     and receipt.get('status')=='published_and_read_back'
                     and receipt.get('receipt_digest')==canonical_digest(receipt,digest_field='receipt_digest')
                     and type(namespace) is str and _NAMESPACE.fullmatch(namespace)
                     and receipt.get('raw_source_uploaded') is False,
                     'scene_retirement_publication_unproven')
            prefix='s3://blueprint/task-evaluation/production-inputs/'+namespace+'/'
            rows=receipt['published_objects']
            _require(type(rows) is list and 1<=len(rows)<=1025,'scene_retirement_publication_unproven')
            for row in rows:
                allowance.tick()
                relative=row['relative_path']
                _require(type(relative) is str and len(relative)<=4096 and not relative.startswith('source/')
                         and str(_canonical('/'+relative))=='/'+relative
                         and row['uri']==prefix+relative and type(row['digest']) is str and _SHA.fullmatch(row['digest'])
                         and type(row['size_bytes']) is int and row['size_bytes']>0,
                         'scene_retirement_publication_unproven')
                selected={key:row[key] for key in ('uri','digest','size_bytes')}
                _require(len(publications)<10000 or row['uri'] in publications,'scene_retirement_declared_reference_limit')
                _require(row['uri'] not in publications or publications[row['uri']]==selected,
                         'scene_retirement_publication_unproven')
                publications[row['uri']]=selected
    rows=list(publications.values())
    if verify_remote:
        verify_publication_rows(rows,transport,allowance)
    return dict(local_objects_verified=len(local),published_objects_verified=len(rows),published_objects=rows)
