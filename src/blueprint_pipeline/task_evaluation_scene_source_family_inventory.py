"""Pure supplied website/SAM source-family inventory; ADP-009D/day-28.

Historical lexical dependency membership never establishes cleanup, current
provider state, presence, scientific validity, rights or original-owner transfer.
"""
from __future__ import annotations

import json
from pathlib import PurePosixPath

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_items, _work_kwargs

from . import task_evaluation_scene_downstream_inventory as downstream
from . import task_evaluation_scene_source_family_contracts as contracts
from . import task_evaluation_scene_source_family_website as website
from . import task_evaluation_scene_source_family_sam as sam
from . import task_evaluation_scene_source_family_adoption as adoption
from .task_evaluation_scene_lineage_budget import RetainedEmissionBudget, RetainedEmissionBudgetError
from .task_evaluation_scene_source_family_contracts import SceneSourceFamilyInventoryError

def _downstream_join(*, intent_id, seed_records, downstream_records, roots, emission_budget, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    return downstream._join(intent_id, seed_records, downstream_records, roots, emission_budget=emission_budget, **_work_kwargs(work_budget))


MAX_RECORD_BYTES = MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RECORDS = MAX_REFERENCES = MAX_ROWS = 10_000
MAX_NODES, MAX_DEPTH = 100_000, 64
MAX_ADOPTION_DEPTH, MAX_ADOPTION_NODES = 16, 1024
EXTRA_ROOTS = {'pubsub_root', 'website_source_binding_root', 'sam_queue_root', 'sam_execution_root', 'host_input_root'}
ROLES = {'website_registrations', 'website_bindings', 'website_handoffs', 'website_preparations',
         'website_runtime_inputs', 'website_task_contexts', 'submission_publications', 'sam_parent_envelopes',
         'sam_parent_results',
         'source_progress', 'source_resume_signals', 'sam_plans', 'sam_profiles', 'sam_recipes',
         'sam_stage_configurations', 'sam_jobs', 'sam_results', 'sam_execution_receipts', 'sam_execution_progress',
         'sam_adoptions', 'sam_prefix_selections', 'sam_host_tasks', 'sam_host_evidence', 'sam_artifact_metadata', 'opaque_evidence'}
FALSE_FLAGS = downstream.FALSE_FLAGS + ('source_family_complete', 'original_owner_transfer_authorized',
    'scientific_validity_checked', 'current_rights_checked', 'current_billing_settled', 'capture_acknowledged',
    'current_queue_ownership_clear', 'sam_cache_retirement_policy_resolved')


def _activation_raw_artifacts(context, old, opaque_records, *, work_budget=None):
    """Bind exact activation-owned raw and sealed outputs without cleanup authority."""
    if work_budget is not None:
        _work(work_budget)
    def rows(values):
        return _work_items(values, work_budget) if work_budget is not None else values
    bound = {row['path'] for row in rows(old['lexical_members'])
             if row['kind'] == 'activation_workspace'}
    opaque = [proof for _, proof in rows(context.decoded['opaque_evidence'])]
    def sealed_document(path, field, digest, raw_reference=None):
        proofs = [proof for proof in rows(opaque) if proof['path'] == path]
        supplied = [raw for name, raw in rows(opaque_records) if name == path]
        if len(proofs) != 1 or len(supplied) != 1 or not 0 < len(supplied[0]) <= 65536:
            return None
        proof, raw = proofs[0], supplied[0]
        if (proof['size_bytes'] != len(raw)
                or proof['sha256'] != contracts.c.raw_digest(raw, **_work_kwargs(work_budget))
                or raw_reference is not None and (
                    proof['sha256'] != raw_reference.get('digest')
                    or proof['size_bytes'] != raw_reference.get('size_bytes'))):
            return None
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            return None
        if (type(value) is not dict or value.get(field) != digest
                or digest != contracts.c.canonical_digest(value, digest_field=field)):
            return None
        return value, dict(proof, seal_field=field, seal_digest=digest)

    for result, result_proof in rows(context.decoded['activation_results']):
        if result.get('status') != 'profile_authority_materialized_no_execution':
            continue
        root = contracts.c.child(context.roots['activation_output_root'], result['activation_id'],
                                 **_work_kwargs(work_budget))
        if root not in bound:
            continue
        profile_path = contracts.c.child(root, 'profiles', result['profile_id'] + '.json',
                                         **_work_kwargs(work_budget))
        matched_profile = sealed_document(profile_path, 'profile_digest', result['profile_digest'])
        if matched_profile is not None:
            profile, sealed = matched_profile
            if (profile.get('schema_version') == 'task_evaluation_launch_profile.v1'
                    and profile.get('profile_id') == result['profile_id']
                    and profile.get('source_commit') == result['source_commit']):
                context.member(root, 'activation_workspace', {
                    'activation_id': result['activation_id'], 'profile_document_bound': True},
                    context.provenance((result_proof, sealed)))
        envelopes = [(envelope, proof) for envelope, proof in rows(context.decoded['activation_envelopes'])
                     if envelope.get('request', {}).get('activation_id') == result['activation_id']
                     and envelope.get('request', {}).get('expected_production_commit') == result['source_commit']]
        if len(envelopes) == 1:
            envelope, envelope_proof = envelopes[0]
            reference = envelope['request'].get('release_window')
            if (type(reference) is dict and type(reference.get('digest')) is str
                    and len(reference['digest']) == 71 and reference['digest'].startswith('sha256:')):
                window_path = contracts.c.child(root, 'references', reference['digest'][7:],
                                                **_work_kwargs(work_budget))
                matched_window = sealed_document(window_path, 'window_digest',
                                                 result['release_window_digest'], reference)
                if matched_window is not None:
                    window, sealed = matched_window
                    if (window.get('schema_version') == 'task_evaluation_shared_mutation_window.v1'
                            and window.get('activation_id') == result['activation_id']
                            and window.get('team_namespace') == result['team_namespace']
                            and window.get('expected_production_commit') == result['source_commit']):
                        context.member(root, 'activation_workspace', {
                            'activation_id': result['activation_id'], 'release_window_document_bound': True},
                            context.provenance((result_proof, envelope_proof, sealed)))
        publication = root + '/launch-set/profile_publication_receipt.v1.json'
        authorizations = PurePosixPath(root) / 'standing-authorizations'
        matches = [proof for proof in rows(opaque)
                   if proof['path'] == publication
                   and proof['sha256'] == result['profile_publication_receipt_digest']]
        approvals = [proof for proof in rows(opaque)
                     if proof['sha256'] == result['standing_authorization_digest']
                     and PurePosixPath(proof['path']).parent == authorizations
                     and PurePosixPath(proof['path']).suffix == '.json']
        if len(matches) != 1 or len(approvals) != 1:
            continue
        context.member(root, 'activation_workspace', {'activation_id': result['activation_id'],
            'raw_artifacts_bound': True}, context.provenance((result_proof, matches[0], approvals[0])))


def join_retained_scene_source_family_inventory(*, intent_id, seed_records, downstream_records, source_records,
                                               roots, parent_routes, retained_metadata_roots):
    """Only supplied bytes are checked; private values never enter failure text."""
    try:
        return _join(intent_id, seed_records, downstream_records, source_records, roots, parent_routes,
                     retained_metadata_roots)
    except SceneSourceFamilyInventoryError:
        raise
    except RetainedEmissionBudgetError:
        raise SceneSourceFamilyInventoryError('scene_source_family_output_limit') from None
    except contracts.c.SceneDownstreamInventoryError as exc:
        code = str(exc).removeprefix('scene_downstream_')
        raise SceneSourceFamilyInventoryError('scene_source_family_' + code) from None
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError, AttributeError):
        raise SceneSourceFamilyInventoryError('scene_source_family_input_invalid') from None


def _join(intent_id, seed_records, downstream_records, source_records, roots, parent_routes, metadata_roots, *, emission_budget=None, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    c = contracts
    c.require(c.matches(intent_id, downstream.contracts.ID, **_work_kwargs(work_budget)) and isinstance(seed_records, dict)
        and (_work_collect(work_budget, set, seed_records) if work_budget is not None else set(seed_records)) == downstream.seed_module._ROLES | {'intent', 'projection'}
        and all(isinstance(seed_records[r], (list, tuple)) for r in (_work_items(downstream.seed_module._ROLES, work_budget) if work_budget is not None else downstream.seed_module._ROLES))
        and isinstance(downstream_records, dict) and (_work_collect(work_budget, set, downstream_records) if work_budget is not None else set(downstream_records)) == downstream.ROLES
        and isinstance(source_records, dict) and (_work_collect(work_budget, set, source_records) if work_budget is not None else set(source_records)) == ROLES
        and all(isinstance(rows, (list, tuple)) for rows in (_work_items([*downstream_records.values(), *source_records.values()], work_budget) if work_budget is not None else [*downstream_records.values(), *source_records.values()]))
        and isinstance(roots, dict) and (_work_collect(work_budget, set, roots) if work_budget is not None else set(roots)) == downstream.seed_module._ROOTS | downstream.EXTRA_ROOTS | EXTRA_ROOTS,
        'parameters_invalid', **_work_kwargs(work_budget))
    roots = {k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items())}
    c.require(isinstance(parent_routes, (list, tuple)) and 1 <= len(parent_routes) <= 4
        and isinstance(metadata_roots, (list, tuple)) and 1 <= len(metadata_roots) <= 8, 'routes_invalid', **_work_kwargs(work_budget))
    routes = []
    for route in (_work_items(parent_routes, work_budget) if work_budget is not None else parent_routes):
        c.require(isinstance(route, dict) and (_work_collect(work_budget, set, route) if work_budget is not None else set(route)) == {'queue_root', 'input_root'}, 'routes_invalid', **_work_kwargs(work_budget))
        routes.append({k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(route.items(), work_budget) if work_budget is not None else route.items())})
    metadata_roots = [c.path(v, **_work_kwargs(work_budget)) for v in (_work_items(metadata_roots, work_budget) if work_budget is not None else metadata_roots)]
    c.require(len({r['queue_root'] for r in (_work_items(routes, work_budget) if work_budget is not None else routes)}) == len(routes)
        and len((_work_collect(work_budget, set, metadata_roots) if work_budget is not None else set(metadata_roots))) == len(metadata_roots)
        and {'queue_root': roots['preparation_queue_root'], 'input_root': roots['preparation_input_root']} in routes,
        'routes_invalid', **_work_kwargs(work_budget))
    groups = {'intent': [seed_records['intent']], 'projection': [] if seed_records['projection'] is None else [seed_records['projection']],
        **{r: seed_records[r] for r in (_work_items(downstream.seed_module._ROLES, work_budget) if work_budget is not None else downstream.seed_module._ROLES)}, **downstream_records, **source_records}
    limits = {k: globals()[k] for k in (_work_items(('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
        'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH', 'MAX_ADOPTION_DEPTH', 'MAX_ADOPTION_NODES'), work_budget) if work_budget is not None else ('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
        'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH', 'MAX_ADOPTION_DEPTH', 'MAX_ADOPTION_NODES'))}
    decoded = c.decode(groups, limits, **_work_kwargs(work_budget))
    if emission_budget is None:
        emission_budget = RetainedEmissionBudget(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_ROWS, max_references=MAX_REFERENCES, **_work_kwargs(work_budget))
    else:
        c.require(isinstance(emission_budget, RetainedEmissionBudget), 'parameters_invalid', **_work_kwargs(work_budget))
        emission_budget = emission_budget.scope(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_ROWS, max_references=MAX_REFERENCES)
    context = c.Context(decoded, roots, limits, intent_id, ROLES, emission_budget=emission_budget, **_work_kwargs(work_budget))
    context.routes, context.metadata_roots = routes, metadata_roots
    # The nested downstream reader already observes the predecessor roles.
    # The private shared-budget composition scans source roles once here;
    # the public standalone API retains its original complete projection.
    context.references(roles=frozenset(ROLES) if work_budget is not None else None)
    old = _downstream_join(intent_id=intent_id, seed_records=seed_records, downstream_records=downstream_records,
                           roots={k: v for k, v in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items()) if k not in EXTRA_ROOTS}, emission_budget=emission_budget, **_work_kwargs(work_budget))
    _activation_raw_artifacts(context, old, source_records['opaque_evidence'],
                              **_work_kwargs(work_budget))
    if work_budget is not None:
        context.predecessor_remote_identities(old['remote_reference_obligations'],
            raw_rows=old['raw_reference_obligations'])
    websites = website.capture(context, old, **_work_kwargs(work_budget))
    publications = website.publication(context, old, **_work_kwargs(work_budget))
    context.source_owner_workspaces = {m['path'] for m in (_work_items(old['seed']['members'], work_budget) if work_budget is not None else old['seed']['members']) if m['kind'] == 'administrative_source_workspace'}
    sam_rows = sam.inventory(context, old, **_work_kwargs(work_budget))
    adoption_rows, original_phases = adoption.inventory(context, **_work_kwargs(work_budget))
    result = {'schema_version': 'task_evaluation_scene_source_family_inventory.v1',
        'scope': 'supplied_retained_source_family_records', 'status': 'kept_unresolved', 'intent_id': intent_id,
        'downstream_inventory': old, 'website_observations': websites, 'publication_observations': publications,
        'sam_observations': sam_rows, 'adoption_observations': adoption_rows, 'original_phase_observations': original_phases,
        'raw_versions': context.rows(p for p in (_work_items(context.raw, work_budget) if work_budget is not None else context.raw) if p['role'] in ROLES),
        'raw_reference_obligations': context.obligations, 'remote_reference_obligations': context.remote,
        'structural_join_obligations': context.structural, 'lexical_members': context.members,
        'mutations': 0, **{flag: False for flag in (_work_items(FALSE_FLAGS, work_budget) if work_budget is not None else FALSE_FLAGS)}}
    emission_budget.check_document(result)
    # The private sink already checked the complete compact document. Preserve
    # the legacy public native check without traversing it twice in a plan.
    if work_budget is None:
        contracts.c.bounded_size(result, MAX_OUTPUT_BYTES)
    for key, rows in (_work_items(result.items(), work_budget) if work_budget is not None else result.items()):
        if isinstance(rows, list):
            result[key] = contracts.c.unique(rows, MAX_OUTPUT_BYTES, **_work_kwargs(work_budget))
    return result
