"""Selected terminal obligations, never global reader or ownership clearance."""
import re
from pathlib import Path

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_scene_retirement_access import _require, SceneRetirementAccessError
from .task_evaluation_scene_retirement_authority import selected_document
from .task_evaluation_scene_retirement_declared_bytes import _selector
from .task_evaluation_scene_downstream_contracts import bounded_size
from .task_evaluation_scene_lineage_budget import _Rows
from .task_evaluation_surface_target import validate_surface_target

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
        # Measured member provenance can select raw JSON receipts without a
        # canonical seal. Their acquired hash is usable for a raw-digest
        # selector, but they are not parsed as arbitrary semantic documents.
        self.selected_raw=set(selected)
        self.documents={}
        self.canonical={}
        self.raw={}
        self.physical={}
        self.has_inventory=preserved is not None
        self.member_roots=[]
        self.emitted=0
        self.count=0
        self.transferred=[]
        self.locals={}
        self.covered={}
        self.bundle_parts={}
        self.scene_handoffs={}
        self.recipe_stage_seen=set()
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
            self.member_roots=[Path(row['path']) for row in preserved['members']]
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
        self._index_preparation_render_inputs()

    def _index_preparation_render_inputs(self):
        """Bind a selected worker result to its owned, byte-inventoried output."""
        from .task_evaluation_scene_configuration_disclosure import (
            RENDER_INPUT_STATUSES, render_inputs_disclosure_is_coherent)

        roots=self.fresh.get('planner_context',{}).get('roots',{})
        queue_root=roots.get('preparation_queue_root')
        input_root=roots.get('preparation_input_root')
        if type(queue_root) is not str or type(input_root) is not str:
            return
        queue=Path(queue_root)
        inputs=Path(input_root)
        for source,value in self.documents.items():
            self.allowance.tick()
            path=Path(source[0])
            if (path.parent!=queue/'results'
                    or value.get('schema_version')!='task_evaluation_launch_preparation_result.v1'
                    or value.get('status')!='queued_for_production_scene_configuration'):
                continue
            preparation_id=value.get('preparation_id')
            _require(type(preparation_id) is str and re.fullmatch(
                r'[A-Za-z0-9][A-Za-z0-9._-]{0,191}',preparation_id),_REASON)
            paired=[envelope for identity,envelope in self.documents.items()
                    if Path(identity[0]) in {queue/'materialized'/path.name,
                                             queue/'completed'/path.name}]
            _require(len(paired)==1,_REASON)
            request=paired[0].get('request',{})
            _require(type(request) is dict
                and request.get('preparation_id')==preparation_id
                and request.get('run_id')==value.get('run_id')
                and request.get('team_namespace')==value.get('team_namespace')
                and request.get('expected_production_commit')==value.get('source_commit'),_REASON)
            digest=value.get('configuration_render_inputs_result_digest')
            _require(type(digest) is str and re.fullmatch(r'sha256:[0-9a-f]{64}',digest),_REASON)
            output=inputs/preparation_id/'configuration-render-inputs'/(
                'task_evaluation_scene_configuration_render_inputs.v1.json')
            from .task_evaluation_scene_retirement_authority import load_document
            try:
                render,observed=load_document(output,maximum=4*1024*1024)
            except (OSError,ValueError,TypeError,UnicodeError):
                raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
            identity=_selector(observed)
            self.allowance.charge('local_bytes',identity[2])
            if self.has_inventory:
                _require(self.physical.get(str(output))==identity,_REASON)
            _require(render.get('schema_version')=='task_evaluation_scene_configuration_render_inputs.v1'
                and render.get('status') in RENDER_INPUT_STATUSES
                and render.get('run_id')==value['run_id']
                and render.get('derived_frame_count')==value.get('configuration_render_input_count')
                and render.get('raw_interiorgs_bytes_in_provider_packet')
                    ==value.get('raw_interiorgs_bytes_in_provider_packet')
                and render.get('provider_mutation_performed') is False
                and render.get('paid_execution_requested') is False
                and render_inputs_disclosure_is_coherent(render)
                and render.get('result_digest')==digest
                and digest==canonical_digest(render,digest_field='result_digest'),_REASON)
            self.canonical.setdefault(digest,set()).add(identity)
            construction=self._index_scene_construction_handoff(value,request,render)
            if construction is not None:
                self.scene_handoffs[source]={
                    'configuration_render_inputs_result_digest':identity,
                    'construction_recipe_digest':construction,
                    'construction_queue_envelope_digest':construction,
                    'construction_queue_receipt_digest':construction}

    def _index_scene_construction_handoff(self,result,request,render):
        """Bind one terminal construction queue item to its retained producer bytes."""
        from .task_evaluation_scene_compilation_owner_contracts import PREPARATION_BASE_FIELDS
        from .task_evaluation_scene_retirement_authority import load_document

        root=self.fresh.get('planner_context',{}).get('scene_construction_queue_root')
        if root is None:
            return
        finished=self.fresh.get('finished_observation',{}).get('status')
        _require(type(root) is str and finished in {
            'completed','revoked_grace_elapsed','expired_grace_elapsed'},_REASON)
        recipe_digest=result.get('construction_recipe_digest')
        _require(type(recipe_digest) is str and re.fullmatch(r'sha256:[0-9a-f]{64}',recipe_digest),_REASON)
        name=result['preparation_id']+'-'+recipe_digest[7:]+'.json'
        queue=Path(root)
        terminal=queue/('completed' if finished=='completed' else 'blocked')/name
        final_path=queue/'results'/name
        try:
            envelope,envelope_raw=load_document(terminal,maximum=4*1024*1024)
            final,final_raw=load_document(final_path,maximum=65536)
        except (OSError,ValueError,TypeError,UnicodeError):
            raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
        self.allowance.charge('local_bytes',envelope_raw['size_bytes']+final_raw['size_bytes'])
        _require(not self.has_inventory or all(not terminal.is_relative_to(member)
                 and not final_path.is_relative_to(member) for member in self.member_roots),_REASON)
        pre={key:result[key] for key in PREPARATION_BASE_FIELDS if key in result}
        _require(set(pre)==PREPARATION_BASE_FIELDS,_REASON)
        pre['status']='inputs_materialized_awaiting_construction_adapter'
        pre['result_digest']=canonical_digest(pre,digest_field='result_digest')
        recipe=envelope.get('recipe')
        seal=envelope.get('envelope_digest')
        _require(envelope.get('schema_version')=='task_evaluation_scene_construction_envelope.v1'
                 and envelope.get('orchestration_id')==result.get('construction_orchestration_id')
                 ==result['preparation_id']
                 and envelope.get('preparation_id')==result['preparation_id']
                 and envelope.get('run_id')==result['run_id']
                 and envelope.get('team_namespace')==result['team_namespace']
                 and envelope.get('expected_production_commit')==result['source_commit']
                 and envelope.get('request')==request,_REASON)
        _require(envelope.get('preparation_result_digest')==pre['result_digest']
                 and envelope.get('materialized_references')==result['references']
                 and envelope.get('render_inputs_result')==render,_REASON)
        _require(type(recipe) is dict and recipe.get('recipe_digest')==recipe_digest
                 and canonical_digest(recipe,digest_field='recipe_digest')==recipe_digest
                 and envelope.get('recipe_digest')==recipe_digest
                 and envelope.get('construction_output_identity')==recipe.get('output_identity'),_REASON)
        _require(envelope.get('automatic_progression_required') is True
                 and envelope.get('canonical_allocator_required_for_gpu_stages') is True
                 and envelope.get('provider_mutation_performed') is False
                 and envelope.get('paid_execution_requested') is False
                 and seal==result.get('construction_queue_envelope_digest')
                 and seal==canonical_digest(envelope,digest_field='envelope_digest'),_REASON)
        _require(final.get('schema_version')=='task_evaluation_scene_construction_finalization.v1'
                 and final.get('status')==final.get('queue_state')==terminal.parent.name
                 and final.get('orchestration_id')==result['preparation_id']
                 and final.get('run_id')==result['run_id']
                 and final.get('source_commit')==result['source_commit']
                 and final.get('recipe_digest')==recipe_digest
                 and final.get('construction_envelope_digest')==seal
                 and final.get('queue_path')==str(terminal)
                 and final.get('result_path')==str(final_path)
                 and final.get('continuing_spend_from_this_run') is False
                 and final.get('finalization_performed') is True
                 and final.get('result_digest')==canonical_digest(final,digest_field='result_digest'),_REASON)
        if finished=='completed':
            _require(final.get('configuration_completed') is True
                     and final.get('configured_scene_published') is True
                     and final.get('full_byte_service_account_readback_passed') is True
                     and final.get('blockers')==[],_REASON)
            self._index_completed_construction_publication(final,request)
        else:
            authority_blocker = ('scene_owner_revoked' if finished == 'revoked_grace_elapsed'
                                 else 'scene_execution_owner_expired')
            _require(final.get('configuration_completed') is False
                     and final.get('configured_scene_published') is False
                     and final.get('blockers')==[authority_blocker],_REASON)
        original_path=queue/'pending'/name
        receipts=[]
        for created in (False,True):
            receipt={'schema_version':'task_evaluation_scene_construction_intake_receipt.v1',
                     'status':'queued_for_production_construction',
                     'orchestration_id':result['preparation_id'],'run_id':result['run_id'],
                     'recipe_digest':recipe_digest,'envelope_digest':seal,
                     'queue_path':str(original_path),'created':created,
                     'automatic_progression_required':True,
                     'provider_mutation_performed':False,'paid_execution_requested':False,
                     'receipt_digest':''}
            receipt['receipt_digest']=canonical_digest(receipt,digest_field='receipt_digest')
            if receipt['receipt_digest']==result.get('construction_queue_receipt_digest'):
                receipts.append(receipt)
        _require(len(receipts)==1,_REASON)
        identity=_selector(envelope_raw)
        for digest in (recipe_digest,seal,receipts[0]['receipt_digest']):
            self.canonical.setdefault(digest,set()).add(identity)
        return identity

    def _index_completed_construction_publication(self,final,request):
        """Reread the one archived result and revision selected by finalization."""
        from .task_evaluation_configured_scene_revision import validate_configured_scene_revision
        from .task_evaluation_scene_retirement_authority import selected_document

        _require(self.has_inventory,_REASON)
        name='task_evaluation_scene_configuration_publication.v1.json'
        candidates=[identity for path,identity in self.physical.items()
                    if Path(path).name==name and any(Path(path).is_relative_to(root)
                    for root in self.member_roots)]
        _require(len(candidates)<=256,_REASON)
        selected=[]
        for identity in candidates:
            self.allowance.tick()
            self.allowance.charge('local_bytes',identity[2])
            try:
                value=selected_document(dict(zip(('path','sha256','size_bytes'),identity)),maximum=4*1024*1024)
            except (OSError,ValueError,TypeError,UnicodeError):
                raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
            _require(type(value) is dict,_REASON)
            if value.get('result_digest')!=final.get('publication_result_digest'):
                continue
            _require(value.get('schema_version')=='task_evaluation_scene_configuration_publication.v1'
                     and value.get('status')=='configured_scene_published'
                     and value.get('configuration_run_id')==final['run_id']
                     and value.get('configured_scene_revision_digest')==final.get('configured_scene_revision_digest')
                     and value.get('full_byte_service_account_readback_passed') is True
                     and value.get('provider_mutation_performed') is False
                     and value.get('paid_execution_requested') is False
                     and value['result_digest']==canonical_digest(value,digest_field='result_digest'),_REASON)
            selected.append((identity,value))
        _require(len(selected)==1,_REASON)
        publication_identity,publication=selected[0]
        revision_row=publication.get('configured_scene_revision')
        _require(type(revision_row) is dict and revision_row.get('role')=='configured_scene_revision',_REASON)
        revision_identity=_selector(revision_row,'path','digest','size_bytes')
        revision_path=Path(revision_identity[0])
        _require(revision_path==Path(publication_identity[0]).parent/'configured_scene_revision.v1.json'
                 and self.physical.get(str(revision_path))==revision_identity,_REASON)
        self.allowance.charge('local_bytes',revision_identity[2])
        try:
            revision=selected_document(dict(zip(('path','sha256','size_bytes'),revision_identity)),maximum=4*1024*1024)
            _require(type(revision) is dict,_REASON)
            validate_configured_scene_revision(revision)
        except (OSError,ValueError,TypeError,UnicodeError):
            raise SceneRetirementAccessError('scene_retirement_reference_changed') from None
        _require(revision.get('revision_digest')==final['configured_scene_revision_digest']
                 and revision.get('configuration_run_id')==final['run_id']
                 and revision.get('source_commit')==final['source_commit']
                 and revision.get('team_namespace')==request['team_namespace']
                 and revision.get('scene_identity')==request['scene']['identity']
                 and revision.get('source',{}).get('manifest')==request['scene']['source_manifest']
                 and revision.get('source',{}).get('rights_admission')==request['scene']['rights']['admission']
                 and revision.get('source',{}).get('rights_evidence')==request['scene']['rights']['evidence']
                 and revision.get('appearance',{}).get('observed_source')==request['scene']['appearance']['representation']
                 and revision.get('geometry',{}).get('candidate_collision_source')==request['scene']['geometry']['collision']
                 and revision.get('task_template',{}).get('identity')==request['task']['identity']
                 and revision.get('task_template',{}).get('definition')==request['task']['definition']
                 and revision.get('task_template',{}).get('success_criteria')==request['task']['success_criteria']
                 and revision.get('replacement',{}).get('identity')==request['task']['subject']['identity']
                 and revision.get('configured_scene_bundle')==publication.get('configured_scene_bundle_reference')
                 and revision['revision_digest']==canonical_digest(revision,digest_field='revision_digest'),_REASON)
        remote=publication.get('configured_scene_revision_reference')
        _require(type(remote) is dict and remote.get('digest')==revision_identity[1]
                 and remote.get('size_bytes')==revision_identity[2],_REASON)
        self.canonical.setdefault(publication['result_digest'],set()).add(publication_identity)
        self.canonical.setdefault(revision['revision_digest'],set()).add(revision_identity)

    def transfer_scene_handoff(self,protection):
        """Retain an exact selected preparation's blocked construction outputs."""
        self.allowance.tick()
        self.count+=1
        _require(self.count<=MAX_OCCURRENCES,'scene_retirement_reference_limit')
        fact,source,value=self._fact(protection)
        targets=self.scene_handoffs.get(source,{})
        target=targets.get(fact['contract_path'])
        _require(protection['kind']=='missing_edge_obligations'
                 and fact['binding_status']=='unresolved'
                 and fact['reason']=='deferred_downstream_document'
                 and fact['digest_meaning']=='no_inferred_raw_identity'
                 and all(fact[key] is None for key in ('digest','path','uri','size_bytes'))
                 and not fact['related_sources']
                 and fact['source']['family']=='preparation'
                 and fact['source']['role']=='result'
                 and target is not None
                 and value.get(fact['contract_path']) in self.canonical
                 and target in self.canonical[value[fact['contract_path']]],_REASON)
        proof={'kind':'selected_scene_construction_handoff',
               'canonical_digest':value[fact['contract_path']],
               'original_target':dict(zip(('path','sha256','size_bytes'),target)),
               'archive_inventory_verified':self.has_inventory and self.physical.get(target[0])==target,
               'retained_outside_removal_union':self.has_inventory and self.physical.get(target[0]) is None}
        size=_measure(protection,MAX_BYTES-self.emitted)+_measure(proof,MAX_BYTES-self.emitted)
        _require(size<=MAX_BYTES-self.emitted,'scene_retirement_reference_limit')
        self.emitted+=size
        self.transferred.append({'original_obligation':protection,'preservation_proof':proof,'action':'KEEP'})

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
            task=request.get('task')
            target=task.get('surface_target') if type(task) is dict else None
            if type(target) is dict:
                try:
                    validated=validate_surface_target(target)
                except ValueError:
                    pass
                else:
                    self.canonical.setdefault(validated['target_digest'],set()).add(identity)

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
            if (fact['contract_path']=='scene.configured_revision.configured_scene_bundle'
                    and fact['binding_status']=='receipt_only'):
                self.transfer_native_bundle(protection,counted=True)
                return
            if (re.fullmatch(r'construction\.recipe\.stage_sequence\.[0-5]\.configuration',
                             fact['contract_path']) and fact['binding_status']=='receipt_only'):
                self.transfer_recipe_stage(protection,self.fresh.get('recipe_stage_authority',[]),counted=True)
                return
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
            targets=[target for target in self.selected_raw | self.raw.keys()
                     if target[1]==fact['digest']]
            _require(targets,_REASON)
            if self.has_inventory:
                for target in targets:
                    if any(Path(target[0]).is_relative_to(root) for root in self.member_roots):
                        _require(self.physical.get(target[0])==target,_REASON)
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

    def transfer_inline_owner(self,protection):
        """Keep one original sealed native owner, with no inferred live path."""
        self.allowance.tick()
        self.count+=1
        _require(self.count<=MAX_OCCURRENCES,'scene_retirement_reference_limit')
        fact,source,value=self._fact(protection)
        _require(protection['kind']=='missing_edge_obligations'
            and fact['binding_status']=='unresolved'
            and fact['reason']=='deferred_semantic_object'
            and fact['digest_meaning']=='no_inferred_raw_identity'
            and fact['contract_path']=='authorization.scene_owner_attempt'
            and all(fact[key] is None for key in ('digest','path','uri','size_bytes'))
            and not fact['related_sources']
            and fact['source']['family']=='activation' and fact['source']['role']=='envelope',_REASON)
        from .task_evaluation_scene_compilation_native_owners import RECORD_FIELDS, BINDING_FIELDS
        owner=_at(value.get('request'),'authorization.scene_owner_attempt')
        _require(type(owner) is dict and set(owner)==RECORD_FIELDS
            and owner.get('schema_version')=='task_evaluation_scene_owner_attempt.v1'
            and type(owner.get('scene_attempt_binding')) is dict
            and set(owner['scene_attempt_binding'])==BINDING_FIELDS
            and owner.get('owner_attempt_digest')==canonical_digest(owner,digest_field='owner_attempt_digest'),_REASON)
        digest=owner['owner_attempt_digest']
        matching=[]
        for proof in self.selected[source]:
            self.allowance.tick()
            if (proof.get('json_pointer')=='/request/authorization/scene_owner_attempt'
                    and proof.get('seal_field')=='owner_attempt_digest'
                    and proof.get('seal_digest')==digest):
                matching.append(proof)
        _require(len(matching)==1 and source in self.canonical.get(digest,set()),_REASON)
        lineage=self.fresh.get('historical_lineage',{})
        native=lineage.get('compilation_native_owner_inventory',lineage)
        rows=native.get('compilation_native_owner_observations',[])
        _require(type(rows) in (list,_Rows) and len(rows)<=MAX_OCCURRENCES,_REASON)
        matched=[]
        for row in rows:
            self.allowance.tick()
            if (type(row) is dict and row.get('kind')=='native_owner'
                    and row.get('owner_metadata_binding_verified') is True
                    and row.get('profile_metadata_binding_verified') is True):
                proofs=row.get('source_provenance',[])
                _require(type(proofs) in (list,_Rows) and len(proofs)<=MAX_OCCURRENCES,_REASON)
                bound=False
                for proof in proofs:
                    self.allowance.tick()
                    if (type(proof) is dict and _selector(proof)==source
                            and proof.get('json_pointer')=='/request/authorization/scene_owner_attempt'
                            and proof.get('seal_digest')==digest):
                        bound=True
                if bound:
                    matched.append(row)
        _require(len(matched)==1,_REASON)
        proof={'kind':'selected_inline_native_owner_attempt','canonical_digest':digest,
               'original_envelope':dict(zip(('path','sha256','size_bytes'),source)),
               'active_path_inferred':False}
        size=_measure(protection,MAX_BYTES-self.emitted)+_measure(proof,MAX_BYTES-self.emitted)
        _require(size<=MAX_BYTES-self.emitted,'scene_retirement_reference_limit')
        self.emitted+=size
        self.transferred.append({'original_obligation':protection,'preservation_proof':proof,'action':'KEEP'})

    def transfer_recipe_stage(self,protection,authorities,*,counted=False):
        """Retain a deferred stage only through its selected parent recipe proof."""
        self.allowance.tick()
        if not counted:
            self.count+=1
        _require(self.count<=MAX_OCCURRENCES and type(authorities) is list and len(authorities)<=16,_REASON)
        fact,source,value=self._fact(protection)
        missing=protection['kind']=='missing_edge_obligations'
        local=protection['kind']=='local_path_protections'
        _require((missing or local) and fact['reason']=='deferred_parent_reference_proof'
            and not fact['related_sources']
            and fact['source']['family']=='preparation' and fact['source']['role']=='result'
            and (not missing or (fact['binding_status']=='unresolved'
                and fact['digest_meaning']=='no_inferred_raw_identity'
                and all(fact[key] is None for key in ('digest','path','uri','size_bytes'))))
            and (not local or (fact['binding_status']=='receipt_only'
                and fact['digest_meaning']=='declared_materialized_raw_bytes'
                and fact['uri'] is None and type(fact['path']) is str
                and type(fact['digest']) is str and type(fact['size_bytes']) is int
                and fact['size_bytes']>0)),_REASON)
        matches=[]
        for authority in authorities:
            self.allowance.tick()
            _require(type(authority) is dict and authority.get('kind')=='selected_recipe_stage_authority.v1'
                and type(authority.get('stages')) is list and len(authority['stages'])==6,_REASON)
            result=authority.get('result_raw_ref')
            _require(type(result) is dict and set(result)=={'path','sha256','size_bytes'},_REASON)
            if _selector(result)==source:
                matches.extend((authority,stage) for stage in authority['stages']
                    if type(stage) is dict and stage.get('contract_path')==fact['contract_path'])
        _require(len(matches)==1,_REASON)
        authority,stage=matches[0]
        reference=stage.get('reference')
        _require(type(reference) is dict and set(reference)=={'uri','digest','size_bytes'},_REASON)
        row=[row for row in value.get('references',[]) if type(row) is dict
             and row.get('contract_path')==fact['contract_path']]
        _require(len(row)==1 and all(row[0].get(key)==reference[key] for key in reference)
            and row[0].get('materialized_path')==stage.get('projected_path')
            and row[0].get('full_byte_service_account_readback_passed') is True,_REASON)
        recipe=authority.get('recipe_raw_ref')
        _require(type(recipe) is dict and set(recipe)=={'path','sha256','size_bytes'},_REASON)
        projection=_selector(dict(path=stage['projected_path'],sha256=reference['digest'],
                                  size_bytes=reference['size_bytes']))
        alias=_selector(dict(path=stage['cache_path'],sha256=reference['digest'],
                             size_bytes=reference['size_bytes']))
        _require(not local or _selector(fact,'path','digest','size_bytes')==projection,_REASON)
        if self.has_inventory:
            _require(self.physical.get(recipe['path'])==_selector(recipe)
                and self.physical.get(projection[0])==projection
                and self.physical.get(alias[0])==alias,_REASON)
        self.recipe_stage_seen.add((source,fact['contract_path']))
        proof={'kind':'selected_recipe_stage_with_original_raw_and_archive',
               'recipe_raw_ref':recipe,'result_raw_ref':authority['result_raw_ref'],
               'indexed_contract_path':fact['contract_path'],'stage_index':stage['index'],
               'projected_raw_ref':dict(zip(('path','sha256','size_bytes'),projection)),
               'cache_raw_ref':dict(zip(('path','sha256','size_bytes'),alias)),
               'archive_inventory_verified':self.has_inventory}
        size=_measure(protection,MAX_BYTES-self.emitted)+_measure(proof,MAX_BYTES-self.emitted)
        _require(size<=MAX_BYTES-self.emitted,'scene_retirement_reference_limit')
        self.emitted+=size
        self.transferred.append({'original_obligation':protection,'preservation_proof':proof,'action':'KEEP'})
        if local:
            key=(source,fact['contract_path'],fact['path'],fact['digest'],fact['size_bytes'])
            self.covered[key]=protection
            self.locals.setdefault(projection[0],[]).append((projection,source))

    def transfer_native_downstream(self,protection):
        """Retain one native handoff's exact selected sealed downstream target."""
        self.allowance.tick()
        self.count+=1
        _require(self.count<=MAX_OCCURRENCES,'scene_retirement_reference_limit')
        fact,source,value=self._fact(protection)
        roles={'configured_scene_revision_digest':('configured_revisions','revision_digest',
                'task_evaluation_configured_scene_revision.v1'),
               'episode_compilation_queue_envelope_digest':('compilation_envelopes','envelope_digest',
                'task_evaluation_episode_compilation_envelope.v1'),
               'episode_compilation_queue_receipt_digest':('compilation_intake_receipts','receipt_digest',
                'task_evaluation_episode_compilation_intake_receipt.v1')}
        _require(protection['kind']=='missing_edge_obligations'
            and fact['binding_status']=='unresolved' and fact['reason']=='deferred_downstream_document'
            and fact['digest_meaning']=='no_inferred_raw_identity'
            and fact['contract_path'] in roles
            and all(fact[key] is None for key in ('digest','path','uri','size_bytes'))
            and not fact['related_sources']
            and fact['source']['family']=='preparation' and fact['source']['role']=='result',_REASON)
        digest=value.get(fact['contract_path'])
        role,field,schema=roles[fact['contract_path']]
        _require(type(digest) is str and digest.startswith('sha256:') and len(digest)==71,_REASON)
        targets=[]
        for target in self.canonical.get(digest,set()):
            self.allowance.tick()
            proofs=self.selected[target]
            if any(proof.get('role')==role and proof.get('seal_field')==field
                   and proof.get('seal_digest')==digest for proof in proofs):
                targets.append(target)
        _require(len(targets)==1,_REASON)
        target=targets[0]
        selected_value=self.raw[target]
        _require(selected_value.get('schema_version')==schema and selected_value.get(field)==digest,_REASON)
        if role=='configured_revisions':
            _require(selected_value.get('status')=='configured',_REASON)
        elif role=='compilation_intake_receipts':
            _require(selected_value.get('status')=='queued_for_production_episode_compilation',_REASON)
        else:
            root=self.fresh.get('planner_context',{}).get('roots',{}).get('compilation_queue_root')
            _require(type(root) is str and Path(target[0]).parent==Path(root)/'completed',_REASON)
        lineage=self.fresh.get('historical_lineage',{})
        native=lineage.get('compilation_native_owner_inventory',lineage)
        rows=native.get('preparation_handoff_observations',[])
        _require(type(rows) in (list,_Rows) and len(rows)<=MAX_OCCURRENCES,_REASON)
        matches=[]
        for row in rows:
            self.allowance.tick()
            if type(row) is not dict or row.get('pre_handoff_binding_verified') is not True:
                continue
            proofs=row.get('source_provenance',[])
            _require(type(proofs) in (list,_Rows) and len(proofs)<=MAX_OCCURRENCES,_REASON)
            selected_source=selected_target=False
            for item in proofs:
                self.allowance.tick()
                if type(item) is dict:
                    identity=_selector(item)
                    selected_source|=(identity==source and item.get('role') in {
                        'native_preparation_results','preparation_results'})
                    selected_target|=(identity==target and item.get('role')==role
                        and item.get('seal_field')==field and item.get('seal_digest')==digest)
            if selected_source and selected_target:
                matches.append(row)
        _require(len(matches)==1,_REASON)
        inside=any(Path(target[0]).is_relative_to(root) for root in self.member_roots)
        _require(not inside or self.physical.get(target[0])==target,_REASON)
        proof={'kind':'selected_native_downstream_document','canonical_digest':digest,
               'original_target':dict(zip(('path','sha256','size_bytes'),target)),
               'archive_inventory_verified':self.has_inventory and inside,
               'retained_outside_removal_union':self.has_inventory and not inside}
        size=_measure(protection,MAX_BYTES-self.emitted)+_measure(proof,MAX_BYTES-self.emitted)
        _require(size<=MAX_BYTES-self.emitted,'scene_retirement_reference_limit')
        self.emitted+=size
        self.transferred.append({'original_obligation':protection,'preservation_proof':proof,'action':'KEEP'})

    def transfer_native_bundle(self,protection,*,counted=False):
        """Keep the exact nested bundle row and its result digest together."""
        self.allowance.tick()
        if not counted:
            self.count+=1
        _require(self.count<=MAX_OCCURRENCES,'scene_retirement_reference_limit')
        fact,source,value=self._fact(protection)
        pair={('configured_scene_bundle_digest','deferred_downstream_document'),
              ('scene.configured_revision.configured_scene_bundle','deferred_parent_reference_proof')}
        local=protection['kind']=='local_path_protections'
        _require(fact['source']['family']=='preparation' and fact['source']['role']=='result'
            and not fact['related_sources'],_REASON)
        if local:
            _require(fact['contract_path']=='scene.configured_revision.configured_scene_bundle'
                and fact['binding_status']=='receipt_only'
                and fact['reason']=='deferred_parent_reference_proof'
                and fact['digest_meaning']=='declared_materialized_raw_bytes'
                and fact['uri'] is None
                and any(proof.get('role')=='native_preparation_results'
                    and proof.get('seal_field')=='result_digest'
                    and proof.get('seal_digest')==value.get('result_digest')
                    and proof.get('json_pointer') is None
                    for proof in self.selected.get(source,[]))
                and value.get('result_digest')==canonical_digest(value,digest_field='result_digest'),_REASON)
        else:
            _require(protection['kind']=='missing_edge_obligations'
                and (fact['contract_path'],fact['reason']) in pair
                and fact['binding_status']=='unresolved'
                and fact['digest_meaning']=='no_inferred_raw_identity'
                and all(fact[key] is None for key in ('digest','path','uri','size_bytes')),_REASON)
        references=value.get('references')
        _require(type(references) is list and len(references)<=MAX_OCCURRENCES,_REASON)
        matches=[row for row in references if type(row) is dict and
                 row.get('contract_path')=='scene.configured_revision.configured_scene_bundle']
        _require(len(matches)==1,_REASON)
        row=matches[0]
        path,digest,size=_selector(row,'materialized_path','digest','size_bytes')
        if local:
            _require((fact['path'],fact['digest'],fact['size_bytes'])==(path,digest,size),_REASON)
        _require(value.get('configured_scene_bundle_digest')==digest
                 and row.get('full_byte_service_account_readback_passed') is True
                 and type(row.get('content_addressed_reuse')) is bool
                 and type(row.get('uri')) is str and 0<len(row['uri'])<=4096,_REASON)
        root=self.fresh.get('planner_context',{}).get('roots',{}).get('preparation_input_root')
        prep=value.get('preparation_id')
        _require(type(root) is str and type(prep) is str
                 and Path(path).is_relative_to(Path(root)/prep),_REASON)
        revision_digest=value.get('configured_scene_revision_digest')
        _require(type(revision_digest) is str,_REASON)
        revisions=[]
        for identity in self.canonical.get(revision_digest,set()):
            self.allowance.tick()
            if any(proof.get('role')=='configured_revisions' and proof.get('seal_field')=='revision_digest'
                   and proof.get('seal_digest')==revision_digest for proof in self.selected[identity]):
                revisions.append(identity)
        _require(len(revisions)==1,_REASON)
        revision=revisions[0]
        revision_value=self.raw[revision]
        _require(revision_value.get('schema_version')=='task_evaluation_configured_scene_revision.v1'
                 and revision_value.get('status')=='configured'
                 and revision_value.get('configured_scene_bundle')=={
                     'uri':row['uri'],'digest':digest,'size_bytes':size},_REASON)
        lineage=self.fresh.get('historical_lineage',{})
        native=lineage.get('compilation_native_owner_inventory',lineage)
        observations=native.get('preparation_handoff_observations',[])
        _require(type(observations) in (list,_Rows) and len(observations)<=MAX_OCCURRENCES,_REASON)
        bound=[]
        for observation in observations:
            self.allowance.tick()
            if type(observation) is not dict or observation.get('pre_handoff_binding_verified') is not True:
                continue
            proofs=observation.get('source_provenance',[])
            _require(type(proofs) in (list,_Rows) and len(proofs)<=MAX_OCCURRENCES,_REASON)
            source_seen=revision_seen=False
            for item in proofs:
                self.allowance.tick()
                if type(item) is dict:
                    identity=_selector(item)
                    source_seen|=(identity==source and item.get('role') in {
                        'native_preparation_results','preparation_results'})
                    revision_seen|=(identity==revision and item.get('role')=='configured_revisions')
            if source_seen and revision_seen:
                bound.append(observation)
        _require(len(bound)==1,_REASON)
        inside=any(Path(path).is_relative_to(root) for root in self.member_roots)
        if self.has_inventory:
            _require(inside and self.physical.get(path)==(path,digest,size),_REASON)
        else:
            _require(not self.physical,_REASON)
        if not local:
            parts=self.bundle_parts.setdefault(source,set())
            _require(fact['reason'] not in parts,_REASON)
            parts.add(fact['reason'])
        proof={'kind':'selected_native_bundle_local_bytes' if local else 'selected_native_bundle',
               'raw_digest':digest,
               'materialized_path':path,'size_bytes':size,
               'revision_target':dict(zip(('path','sha256','size_bytes'),revision)),
               'archive_inventory_verified':self.has_inventory and inside}
        amount=_measure(protection,MAX_BYTES-self.emitted)+_measure(proof,MAX_BYTES-self.emitted)
        _require(amount<=MAX_BYTES-self.emitted,'scene_retirement_reference_limit')
        self.emitted+=amount
        self.transferred.append({'original_obligation':protection,'preservation_proof':proof,'action':'KEEP'})
        if local:
            identity=(path,digest,size)
            self.locals.setdefault(path,[]).append((identity,source))
            key=(source,fact['contract_path'],path,digest,size)
            _require(key not in self.covered,_REASON)
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

    def covered_keeps(self,terminal_pins=()):
        result=[]
        keeps=self.fresh.get('reference_keeps',[])
        _require(type(keeps) in (list,_Rows) and len(keeps)<=MAX_OCCURRENCES,_REASON)
        for keep in keeps:
            self.allowance.tick()
            protection=keep.get('observation')
            if type(protection) is dict and protection.get('kind')=='positive_pin_path':
                from .task_evaluation_scene_retirement_pins import covers
                _require(covers(protection,terminal_pins) and keep.get('protected_path')==protection['path'],_REASON)
                result.append(keep)
                continue
            _require(type(protection) is dict and protection.get('kind')=='local_path_protections',_REASON)
            fact,source,_=self._fact(protection)
            key=(source,fact['contract_path'],fact['path'],fact['digest'],fact['size_bytes'])
            _require(self.covered.get(key)==protection,_REASON)
            path=fact['path']
            _require(path==keep.get('protected_path') and path in self.locals,_REASON)
            result.append(keep)
        return result
