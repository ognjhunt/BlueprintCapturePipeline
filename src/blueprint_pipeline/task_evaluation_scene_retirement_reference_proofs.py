"""Selected terminal obligations, never global reader or ownership clearance."""
from pathlib import Path

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_scene_retirement_access import _require, SceneRetirementAccessError
from .task_evaluation_scene_retirement_authority import selected_document
from .task_evaluation_scene_retirement_declared_bytes import _selector
from .task_evaluation_scene_downstream_contracts import bounded_size
from .task_evaluation_scene_lineage_budget import _Rows

_REASON='scene_retirement_reference_closure_unproven'
_FACT_FIELDS={'source','contract_path','binding_status','reason','digest_meaning','digest','path','uri',
              'size_bytes','related_sources'}
MAX_BYTES=1024*1024
MAX_OCCURRENCES=10000
_SOURCE_FIELDS={'family','queue_root','role','row_path','raw_sha256','raw_size_bytes','observed_identity'}


def _source(value):
    _require(type(value) is dict and set(value)==_SOURCE_FIELDS,_REASON)
    identity=_selector(value,'row_path','raw_sha256','raw_size_bytes')
    _require(identity[2]>0,_REASON)
    _require(all(type(value[k]) is str and len(value[k])<=4096 for k in ('family','queue_root','role')),_REASON)
    observed=value['observed_identity']
    _require(observed is None or type(observed) in (list,tuple) and len(observed)<=16
             and all(type(v) is int for v in observed),_REASON)
    return identity


def _measure(value,remaining):
    try:
        return bounded_size(value,remaining)
    except (ValueError,TypeError,OverflowError,RecursionError):
        raise SceneRetirementAccessError('scene_retirement_reference_limit') from None


def _at(value,path):
    _require(type(path) is str and 0<len(path)<=1024 and len(path.split('.'))<=64,_REASON)
    for key in path.split('.'):
        if type(value) is dict:
            value=value.get(key)
        elif type(value) is list and key.isdigit() and len(key)<=5 and int(key)<len(value):
            value=value[int(key)]
        else:
            return None
    return value


class TerminalProofs:
    """Finite exact indexes; raw_versions and caller clear flags are not inputs."""
    def __init__(self,fresh,selected,records,allowance,preserved):
        self.allowance=allowance
        self.fresh=fresh
        self.selected=selected
        self.documents={}
        self.canonical={}
        self.raw={}
        self.physical={}
        self.has_inventory=preserved is not None
        self.emitted=0
        self.count=0
        self.transferred=[]
        self.locals={}
        self.covered={}
        for row in records:
            allowance.tick()
            identity=_source(row['source'])
            value=self._read(identity)
            self.documents[identity]=value
            self._index(identity,value,selected[identity])
        intent=fresh.get('selected_intent_provenance')
        if intent is not None:
            identity=_selector(intent)
            value=self._read(identity)
            _require(value.get('schema_version')=='task_evaluation_scene_intent.v1'
                and value.get('intent_digest') in {canonical_digest(value,digest_field='intent_digest'),
                    cross_runtime_canonical_digest(value,digest_field='intent_digest')}
                and intent.get('seal_digest')==value['intent_digest'],_REASON)
            self._index(identity,value,[intent])
        for identity,proofs in selected.items():
            allowance.tick()
            if identity in self.documents or not any(p.get('seal_field') for p in proofs):
                continue
            value=self._read(identity)
            self._index(identity,value,proofs)
        if preserved is not None:
            _require(type(preserved) is dict and type(preserved.get('files')) is list
                and len(preserved['files'])<=MAX_OCCURRENCES and type(preserved.get('members')) is list
                and len(preserved['members'])<=256,_REASON)
            for row in preserved['files']:
                allowance.tick()
                _require(type(row) is dict and type(row.get('member_index')) is int
                    and 0<=row['member_index']<len(preserved['members']),_REASON)
                root=Path(preserved['members'][row['member_index']]['path'])
                relative=row['relative_path']
                path=str(root/relative)
                identity=_selector(dict(path=path,sha256=row['sha256'],size_bytes=row['size_bytes']))
                _require(Path(path).is_relative_to(root) and path not in self.physical,_REASON)
                self.physical[path]=identity
            aliases=preserved.get('cache_aliases',[])
            _require(type(aliases) is list and len(aliases)<=256
                and len(preserved['files'])+len(aliases)<=MAX_OCCURRENCES,_REASON)
            files={}
            for row in preserved['files']:
                allowance.tick()
                files[(row['member_index'],row['relative_path'])]=row
            for alias in aliases:
                allowance.tick()
                _require(type(alias) is dict and type(alias.get('member_index')) is int
                    and type(alias.get('relative_path')) is str,_REASON)
                source=files.get((alias['member_index'],alias['relative_path']))
                _require(source is not None and alias.get('physical_identity')==source.get('physical_identity')
                    and alias.get('digest')==source.get('sha256')
                    and alias.get('size_bytes')==source.get('size_bytes')
                    and all(alias.get(key)==source.get(key) for key in ('mode','uid','gid')),_REASON)
                identity=_selector(alias,'path','digest','size_bytes')
                path=identity[0]
                _require(path not in self.physical and Path(path).name==identity[1][7:]
                    and not any(Path(path).is_relative_to(Path(member['path']))
                                for member in preserved['members']),_REASON)
                self.physical[path]=identity

    def _read(self,identity):
        self.allowance.tick()
        _require(0<identity[2]<=4*1024*1024,_REASON)
        self.allowance.charge('local_bytes',identity[2])
        try:
            value=selected_document(dict(zip(('path','sha256','size_bytes'),identity)),maximum=4*1024*1024)
        except (OSError,ValueError,TypeError,UnicodeError):
            raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
        self.allowance.tick()
        _require(type(value) is dict,_REASON)
        return value

    def _index(self,identity,value,proofs):
        _require(len(self.raw)<MAX_OCCURRENCES or identity in self.raw,'scene_retirement_reference_limit')
        self.raw[identity]=value
        for proof in proofs:
            self.allowance.tick()
            field,digest=proof.get('seal_field'),proof.get('seal_digest')
            if field is None and digest is None:
                continue
            sealed=value
            pointer=proof.get('json_pointer')
            if pointer is not None and value.get(field)!=digest:
                _require(type(pointer) is str and pointer.startswith('/') and len(pointer)<=1024,_REASON)
                sealed=_at(value,'.'.join(part.replace('~1','/').replace('~0','~') for part in pointer[1:].split('/')))
            _require(type(sealed) is dict and type(field) is str and len(field)<=128 and sealed.get(field)==digest
                and digest in {canonical_digest(sealed,digest_field=field),cross_runtime_canonical_digest(sealed,digest_field=field)},_REASON)
            self.canonical.setdefault(digest,set()).add(identity)
        if value.get('schema_version') in {'task_evaluation_launch_preparation_envelope.v1',
            'task_evaluation_launch_activation_envelope.v1'}:
            request=value.get('request')
            _require(type(request) is dict and value.get('request_digest')==canonical_digest(request),_REASON)
            self.canonical.setdefault(value['request_digest'],set()).add(identity)

    def _fact(self,protection):
        _require(type(protection) is dict and set(protection)=={'kind','observation','action'}
            and protection['action']=='KEEP',_REASON)
        fact=protection['observation']
        _require(type(fact) is dict and set(fact)==_FACT_FIELDS,_REASON)
        _require(all(type(fact[k]) is str and len(fact[k])<=1024 for k in
            ('contract_path','binding_status','reason','digest_meaning')),_REASON)
        _require(all(fact[k] is None or type(fact[k]) is str and len(fact[k])<=4096
            for k in ('digest','path','uri')) and (fact['size_bytes'] is None or
            type(fact['size_bytes']) is int and fact['size_bytes']>=0),_REASON)
        source=_source(fact['source'])
        _require(source in self.documents,_REASON)
        _require(type(fact['related_sources']) in (list,tuple,_Rows) and len(fact['related_sources'])<=MAX_OCCURRENCES,_REASON)
        for related in fact['related_sources']:
            self.allowance.tick()
            _require(_source(related) in self.documents,_REASON)
        return fact,source,self.documents[source]

    def _declared(self,value,path):
        direct=_at(value,path)
        request=_at(value.get('request'),path)
        _require(direct is None or request is None or direct==request,_REASON)
        return direct if direct is not None else request

    def _local(self,fact,source,value):
        _require(fact['binding_status']=='request_bound' and fact['digest_meaning']=='declared_materialized_raw_bytes'
            and fact['reason']=='exact_supplied_request_binding' and fact['related_sources'],_REASON)
        identity=_selector(fact,'path','digest','size_bytes')
        _require(identity[2]>0,_REASON)
        matches=[]
        references=value.get('references')
        _require(type(references) is list and len(references)<=MAX_OCCURRENCES,_REASON)
        for row in references:
            self.allowance.tick()
            if row.get('contract_path')==fact['contract_path']:
                _require(_selector(row,'materialized_path','digest','size_bytes')==identity,_REASON)
                matches.append(row)
        _require(len(matches)==1,_REASON)
        remote={k:matches[0][k] for k in ('uri','digest','size_bytes')}
        for related in fact['related_sources']:
            parent=self.documents[_source(related)]
            _require(self._declared(parent,fact['contract_path'])==remote,_REASON)
        if self.has_inventory:
            _require(self.physical.get(identity[0])==identity,_REASON)
        self.locals.setdefault(identity[0],[]).append((identity,source))
        return dict(kind='exact_request_bound_materialized_tuple',path=identity[0],sha256=identity[1],
                    size_bytes=identity[2],archive_inventory_verified=self.has_inventory)

    def transfer(self,protection):
        self.allowance.tick()
        self.count+=1
        _require(self.count<=MAX_OCCURRENCES,'scene_retirement_reference_limit')
        fact,source,value=self._fact(protection)
        kind=protection['kind']
        if kind=='local_path_protections':
            proof=self._local(fact,source,value)
        elif kind=='remote_raw_references':
            _require(fact['binding_status']=='declared_remote_raw' and fact['digest_meaning']=='remote_raw_bytes'
                and type(fact['size_bytes']) is int and fact['size_bytes']>0,_REASON)
            remote={k:fact[k] for k in ('uri','digest','size_bytes')}
            declared=self._declared(value,fact['contract_path'])
            # Result reference rows are keyed by their actual contract_path,
            # rather than pretending the result contains the request object.
            if declared is None:
                matches=[r for r in value.get('references',[]) if r.get('contract_path')==fact['contract_path']]
                _require(len(matches)==1,_REASON)
                declared={k:matches[0][k] for k in remote}
            _require(declared==remote,_REASON)
            proof=dict(kind='retained_exact_remote_metadata',remote_reference=remote,
                       remote_availability_verified=False,remote_mutations=0)
        elif kind=='canonical_document_selector_obligations':
            _require(fact['binding_status']=='selector_only' and fact['digest_meaning']=='canonical_document_seal'
                and self._declared(value,fact['contract_path'])==fact['digest'],_REASON)
            targets=self.canonical.get(fact['digest'],set())
            _require(targets,_REASON)
            proof=dict(kind='selected_canonical_seal_with_original_raw_proof',canonical_digest=fact['digest'],
                targets=self._targets(targets,protection))
        elif kind=='raw_digest_selector_obligations':
            _require(fact['binding_status']=='selector_only' and fact['digest_meaning']=='raw_digest_only'
                and self._declared(value,fact['contract_path'])==fact['digest'],_REASON)
            targets=[target for target in self.raw if target[1]==fact['digest']]
            _require(targets,_REASON)
            proof=dict(kind='selected_original_raw_selector',targets=self._targets(targets,protection))
        else:
            _require(False,_REASON)
        self.allowance.tick()
        size=_measure(protection,MAX_BYTES-self.emitted)+_measure(proof,MAX_BYTES-self.emitted)
        _require(size<=MAX_BYTES-self.emitted,'scene_retirement_reference_limit')
        self.emitted+=size
        self.transferred.append(dict(original_obligation=protection,preservation_proof=proof,action='KEEP'))
        if kind=='local_path_protections':
            key=(source,fact['contract_path'],fact['path'],fact['digest'],fact['size_bytes'])
            self.covered[key]=protection

    def _targets(self,targets,protection):
        # Reserve bounded framing before growing any variable target collection.
        remaining=MAX_BYTES-self.emitted-_measure(protection,MAX_BYTES-self.emitted)-256
        _require(remaining>=0,'scene_retirement_reference_limit')
        result=[]
        _require(len(targets)<=MAX_OCCURRENCES,'scene_retirement_reference_limit')
        for target in sorted(targets):
            self.allowance.tick()
            row=dict(zip(('path','sha256','size_bytes'),target))
            size=_measure(row,remaining)
            _require(size<=remaining,'scene_retirement_reference_limit')
            remaining-=size
            result.append(row)
        return result

    def covered_keeps(self):
        result=[]
        keeps=self.fresh.get('reference_keeps',[])
        _require(type(keeps) in (list,_Rows) and len(keeps)<=MAX_OCCURRENCES,_REASON)
        for keep in keeps:
            self.allowance.tick()
            protection=keep.get('observation')
            _require(type(protection) is dict and protection.get('kind')=='local_path_protections',_REASON)
            fact,source,_=self._fact(protection)
            key=(source,fact['contract_path'],fact['path'],fact['digest'],fact['size_bytes'])
            _require(self.covered.get(key)==protection,_REASON)
            path=fact['path']
            _require(path==keep.get('protected_path') and path in self.locals,_REASON)
            result.append(keep)
        return result
