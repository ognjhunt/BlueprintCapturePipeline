"""Unused supplied native metadata inventory; no runtime or action authority."""
from __future__ import annotations

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


def _join(intent_id, seed, downstream, source, bridge, roots, routes, metadata):
    d = prior.downstream
    c.require(c.matches(intent_id, c.OWNER_ID) and isinstance(seed, dict)
        and set(seed) == d.seed_module._ROLES | {'intent', 'projection'}
        and isinstance(downstream, dict) and set(downstream) == d.ROLES
        and isinstance(source, dict) and set(source) == prior.ROLES
        and isinstance(bridge, dict) and set(bridge) == ROLES
        and all(isinstance(rows, (list, tuple)) for mapping in (downstream, source, bridge) for rows in mapping.values())
        and all(isinstance(seed[role], (list, tuple)) for role in d.seed_module._ROLES)
        and isinstance(roots, dict) and set(roots) == d.seed_module._ROOTS | d.EXTRA_ROOTS | prior.EXTRA_ROOTS,
        'parameters_invalid')
    roots = {k: c.path(v) for k, v in roots.items()}
    c.require(isinstance(routes, (list, tuple)) and 1 <= len(routes) <= 4
        and isinstance(metadata, (list, tuple)) and 1 <= len(metadata) <= 8, 'routes_invalid')
    c.require(all(isinstance(route, dict) and set(route) == {'queue_root', 'input_root'} for route in routes), 'routes_invalid')
    routes = [{k: c.path(v) for k, v in route.items()} for route in routes]
    metadata = [c.path(v) for v in metadata]
    c.require(len({r['queue_root'] for r in routes}) == len(routes) and len(set(metadata)) == len(metadata)
        and {'queue_root': roots['preparation_queue_root'], 'input_root': roots['preparation_input_root']} in routes, 'routes_invalid')
    groups = {'intent': [seed['intent']], 'projection': [] if seed['projection'] is None else [seed['projection']],
              **{role: seed[role] for role in d.seed_module._ROLES}, **downstream, **source, **bridge}
    limits = {k: globals()[k] for k in ('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
        'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH')}
    decoded = c.retained.decode(groups, limits)  # ALL decoded bounds before hashes/any child.
    sink = RetainedEmissionBudget(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_ROWS, max_references=MAX_REFERENCES)
    context = c.Context(decoded, roots, limits, intent_id, ROLES, emission_budget=sink)
    context.routes, context.metadata_roots = routes, metadata
    context.references()
    old = prior._join(intent_id, seed, downstream, source, roots, routes, metadata, emission_budget=sink)
    observations = preparation.inventory(context)
    output_observations = outputs.inventory(context)
    output_observations.extend(owners.inventory(context))
    result = {'schema_version': 'task_evaluation_scene_compilation_native_owner_inventory.v1',
        'scope': 'supplied_retained_compilation_native_owner_metadata', 'status': 'kept_unresolved', 'intent_id': intent_id,
        'source_family_inventory': old, 'preparation_handoff_observations': observations,
        'compilation_native_owner_observations': output_observations,
        'raw_versions': context.rows(p for p in context.raw if p['role'] in ROLES),
        'declared_lexical_members': context.members, 'raw_reference_obligations': context.obligations,
        'remote_reference_obligations': context.remote, 'structural_join_obligations': context.structural,
        'mutations': 0, **{flag: False for flag in prior.FALSE_FLAGS}}
    sink.check_document(result)
    c.retained.c.bounded_size(result, MAX_OUTPUT_BYTES)
    for key, rows in result.items():
        if isinstance(rows, list):
            result[key] = c.retained.c.unique(rows, MAX_OUTPUT_BYTES)
    return result
