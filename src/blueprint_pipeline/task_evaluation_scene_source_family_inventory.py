"""Pure supplied website/SAM source-family inventory; ADP-009D/day-28.

Historical lexical dependency membership never establishes cleanup, current
provider state, presence, scientific validity, rights or original-owner transfer.
"""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work, _work_items, _work_kwargs

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
         'source_progress', 'source_resume_signals', 'sam_plans', 'sam_profiles', 'sam_recipes',
         'sam_stage_configurations', 'sam_jobs', 'sam_results', 'sam_execution_receipts', 'sam_execution_progress',
         'sam_adoptions', 'sam_prefix_selections', 'sam_host_tasks', 'sam_host_evidence', 'sam_artifact_metadata', 'opaque_evidence'}
FALSE_FLAGS = downstream.FALSE_FLAGS + ('source_family_complete', 'original_owner_transfer_authorized',
    'scientific_validity_checked', 'current_rights_checked', 'current_billing_settled', 'capture_acknowledged',
    'current_queue_ownership_clear', 'sam_cache_retirement_policy_resolved')


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
        and set(seed_records) == downstream.seed_module._ROLES | {'intent', 'projection'}
        and all(isinstance(seed_records[r], (list, tuple)) for r in (_work_items(downstream.seed_module._ROLES, work_budget) if work_budget is not None else downstream.seed_module._ROLES))
        and isinstance(downstream_records, dict) and set(downstream_records) == downstream.ROLES
        and isinstance(source_records, dict) and set(source_records) == ROLES
        and all(isinstance(rows, (list, tuple)) for rows in (_work_items([*downstream_records.values(), *source_records.values()], work_budget) if work_budget is not None else [*downstream_records.values(), *source_records.values()]))
        and isinstance(roots, dict) and set(roots) == downstream.seed_module._ROOTS | downstream.EXTRA_ROOTS | EXTRA_ROOTS,
        'parameters_invalid', **_work_kwargs(work_budget))
    roots = {k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items())}
    c.require(isinstance(parent_routes, (list, tuple)) and 1 <= len(parent_routes) <= 4
        and isinstance(metadata_roots, (list, tuple)) and 1 <= len(metadata_roots) <= 8, 'routes_invalid', **_work_kwargs(work_budget))
    routes = []
    for route in (_work_items(parent_routes, work_budget) if work_budget is not None else parent_routes):
        c.require(isinstance(route, dict) and set(route) == {'queue_root', 'input_root'}, 'routes_invalid', **_work_kwargs(work_budget))
        routes.append({k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(route.items(), work_budget) if work_budget is not None else route.items())})
    metadata_roots = [c.path(v, **_work_kwargs(work_budget)) for v in (_work_items(metadata_roots, work_budget) if work_budget is not None else metadata_roots)]
    c.require(len({r['queue_root'] for r in (_work_items(routes, work_budget) if work_budget is not None else routes)}) == len(routes)
        and len(set(metadata_roots)) == len(metadata_roots)
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
    context.references()
    old = _downstream_join(intent_id=intent_id, seed_records=seed_records, downstream_records=downstream_records,
                           roots={k: v for k, v in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items()) if k not in EXTRA_ROOTS}, emission_budget=emission_budget, **_work_kwargs(work_budget))
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
    contracts.c.bounded_size(result, MAX_OUTPUT_BYTES, **_work_kwargs(work_budget))
    for key, rows in (_work_items(result.items(), work_budget) if work_budget is not None else result.items()):
        if isinstance(rows, list):
            result[key] = contracts.c.unique(rows, MAX_OUTPUT_BYTES, **_work_kwargs(work_budget))
    return result
