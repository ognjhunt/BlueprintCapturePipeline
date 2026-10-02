"""Unused supplied native metadata inventory; no runtime or action authority."""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_items, _work_kwargs

from . import task_evaluation_scene_source_family_inventory as prior
from . import task_evaluation_scene_compilation_owner_contracts as c
from . import task_evaluation_scene_compilation_owner_preparations as preparation
from . import task_evaluation_scene_compilation_owner_outputs as outputs
from . import task_evaluation_scene_compilation_native_owners as owners
from .task_evaluation_scene_lineage_budget import RetainedEmissionBudget, RetainedEmissionBudgetError
from .task_evaluation_scene_compilation_owner_contracts import SceneCompilationOwnerInventoryError

MAX_RECORD_BYTES = MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 16*1024*1024
MAX_RECORDS = MAX_REFERENCES = MAX_ROWS = 10_000
MAX_NODES, MAX_DEPTH = 100_000, 64
ROLES = c.ROLES


def join_retained_scene_compilation_native_owner_inventory(*, intent_id, seed_records, downstream_records,
        source_records, bridge_records, roots, parent_routes, retained_metadata_roots):
    try:
        return _join(intent_id, seed_records, downstream_records, source_records, bridge_records,
                     roots, parent_routes, retained_metadata_roots)
    except SceneCompilationOwnerInventoryError:
        raise
    except (RetainedEmissionBudgetError, c.retained.SceneSourceFamilyInventoryError, c.retained.c.SceneDownstreamInventoryError,
            ValueError, TypeError, KeyError, OverflowError, RecursionError, UnicodeError, AttributeError) as exc:
        raise c.translate(exc) from None


def _join(intent_id, seed, downstream, source, bridge, roots, routes, metadata, *, emission_budget=None, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
    d = prior.downstream
    c.require(c.matches(intent_id, c.OWNER_ID, **_work_kwargs(work_budget)) and isinstance(seed, dict)
        and (_work_collect(work_budget, set, seed) if work_budget is not None else set(seed)) == d.seed_module._ROLES | {'intent', 'projection'}
        and isinstance(downstream, dict) and (_work_collect(work_budget, set, downstream) if work_budget is not None else set(downstream)) == d.ROLES
        and isinstance(source, dict) and (_work_collect(work_budget, set, source) if work_budget is not None else set(source)) == prior.ROLES
        and isinstance(bridge, dict) and (_work_collect(work_budget, set, bridge) if work_budget is not None else set(bridge)) == ROLES
        and all(isinstance(rows, (list, tuple)) for mapping in (_work_items((downstream, source, bridge), work_budget) if work_budget is not None else (downstream, source, bridge)) for rows in (_work_items(mapping.values(), work_budget) if work_budget is not None else mapping.values()))
        and all(isinstance(seed[role], (list, tuple)) for role in (_work_items(d.seed_module._ROLES, work_budget) if work_budget is not None else d.seed_module._ROLES))
        and isinstance(roots, dict) and (_work_collect(work_budget, set, roots) if work_budget is not None else set(roots)) == d.seed_module._ROOTS | d.EXTRA_ROOTS | prior.EXTRA_ROOTS,
        'parameters_invalid', **_work_kwargs(work_budget))
    roots = {k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items())}
    c.require(isinstance(routes, (list, tuple)) and 1 <= len(routes) <= 4
        and isinstance(metadata, (list, tuple)) and 1 <= len(metadata) <= 8, 'routes_invalid', **_work_kwargs(work_budget))
    c.require(all(isinstance(route, dict) and (_work_collect(work_budget, set, route) if work_budget is not None else set(route)) == {'queue_root', 'input_root'} for route in (_work_items(routes, work_budget) if work_budget is not None else routes)), 'routes_invalid', **_work_kwargs(work_budget))
    routes = [{k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(route.items(), work_budget) if work_budget is not None else route.items())} for route in (_work_items(routes, work_budget) if work_budget is not None else routes)]
    metadata = [c.path(v, **_work_kwargs(work_budget)) for v in (_work_items(metadata, work_budget) if work_budget is not None else metadata)]
    c.require(len({r['queue_root'] for r in (_work_items(routes, work_budget) if work_budget is not None else routes)}) == len(routes) and len((_work_collect(work_budget, set, metadata) if work_budget is not None else set(metadata))) == len(metadata)
        and {'queue_root': roots['preparation_queue_root'], 'input_root': roots['preparation_input_root']} in routes, 'routes_invalid', **_work_kwargs(work_budget))
    groups = {'intent': [seed['intent']], 'projection': [] if seed['projection'] is None else [seed['projection']],
              **{role: seed[role] for role in (_work_items(d.seed_module._ROLES, work_budget) if work_budget is not None else d.seed_module._ROLES)}, **downstream, **source, **bridge}
    limits = {k: globals()[k] for k in (_work_items(('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
        'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH'), work_budget) if work_budget is not None else ('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
        'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH'))}
    decoded = c.retained.decode(groups, limits, **_work_kwargs(work_budget))  # ALL decoded bounds before hashes/any child.
    public_join = emission_budget is None
    if public_join:
        sink = RetainedEmissionBudget(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_ROWS,
                                     max_references=MAX_REFERENCES, **_work_kwargs(work_budget))
    else:
        c.require(type(emission_budget) is RetainedEmissionBudget
                  and emission_budget.work_budget is work_budget, 'parameters_invalid', **_work_kwargs(work_budget))
        sink = emission_budget.scope(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_ROWS, max_references=MAX_REFERENCES)
    context = c.Context(decoded, roots, limits, intent_id, ROLES, emission_budget=sink, **_work_kwargs(work_budget))
    context.routes, context.metadata_roots = routes, metadata
    # The nested source-family reader owns every predecessor role under this
    # same budget. Scan native bridge records once in private composition.
    context.references(roles=frozenset(ROLES) if work_budget is not None else None)
    old = prior._join(intent_id, seed, downstream, source, roots, routes, metadata, emission_budget=sink, **_work_kwargs(work_budget))
    if work_budget is not None:
        context.predecessor_remote_identities(old['remote_reference_obligations'])
        context.predecessor_remote_identities(old['downstream_inventory']['remote_reference_obligations'])
    observations = preparation.inventory(context, **_work_kwargs(work_budget))
    output_observations = outputs.inventory(context, **_work_kwargs(work_budget))
    output_observations.extend(owners.inventory(context, **_work_kwargs(work_budget)))
    result = {'schema_version': 'task_evaluation_scene_compilation_native_owner_inventory.v1',
        'scope': 'supplied_retained_compilation_native_owner_metadata', 'status': 'kept_unresolved', 'intent_id': intent_id,
        'source_family_inventory': old, 'preparation_handoff_observations': observations,
        'compilation_native_owner_observations': output_observations,
        'raw_versions': context.rows(p for p in (_work_items(context.raw, work_budget) if work_budget is not None else context.raw) if p['role'] in ROLES),
        'declared_lexical_members': context.members, 'raw_reference_obligations': context.obligations,
        'remote_reference_obligations': context.remote, 'structural_join_obligations': context.structural,
        'mutations': 0, **{flag: False for flag in (_work_items(prior.FALSE_FLAGS, work_budget) if work_budget is not None else prior.FALSE_FLAGS)}}
    sink.check_document(result)
    # A supplied private sink has already bounded the exact complete document.
    if public_join:
        c.retained.c.bounded_size(result, MAX_OUTPUT_BYTES)
    for key, rows in (_work_items(result.items(), work_budget) if work_budget is not None else result.items()):
        if isinstance(rows, list):
            result[key] = c.retained.c.unique(rows, MAX_OUTPUT_BYTES, **_work_kwargs(work_budget))
    return result
