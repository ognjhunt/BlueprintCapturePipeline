"""Pure supplied execution-family lineage; no filesystem completeness/authority.

ADP-009D/day-28: every retained raw version remains discoverable. Recorded
identity and lexical membership never become paid proof or retirement approval.
"""
from __future__ import annotations

from .task_evaluation_scene_lineage_budget import _work_collect, _work, _work_items, _work_kwargs

from . import task_evaluation_scene_inventory_seed as seed_module
from . import task_evaluation_scene_downstream_contracts as contracts
from . import task_evaluation_scene_downstream_execution as execution
from . import task_evaluation_scene_downstream_terminal as terminal
from .task_evaluation_scene_downstream_contracts import SceneDownstreamInventoryError

MAX_RECORD_BYTES = MAX_TOTAL_BYTES = MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_RECORDS = MAX_REFERENCES = MAX_ROWS = 10_000
MAX_NODES, MAX_DEPTH = 100_000, 64
EXTRA_ROOTS = {'activation_output_root', 'launch_execution_root', 'terminal_result_root', 'policy_canary_root',
               'compilation_queue_root', 'compilation_output_root'}
ROLES = {'activation_results', 'launch_progressions', 'launch_profiles', 'launch_requests', 'launch_receipts',
         'terminal_states', 'canary_dispatches', 'canary_projections', 'canary_syncs', 'provider_zero_receipts',
         'allocator_results', 'canary_offload_pointers', 'terminal_publications', 'compilation_envelopes', 'compilation_results'}
FALSE_FLAGS = ('complete_scene_inventory', 'host_history_complete', 'filesystem_inventory_complete', 'references_clear',
               'process_fences_held', 'scene_finished', 'paid_scope_verified', 'current_provider_zero_verified',
               'fresh_remote_readback_verified', 'retirement_eligible', 'cleanup_authorized', 'execution_authorized')


def join_retained_scene_downstream_inventory(*, intent_id, seed_records, downstream_records, roots):
    """Join exact supplied bytes only; every refusal is fixed and bounded."""
    try:
        return _join(intent_id, seed_records, downstream_records, roots)
    except SceneDownstreamInventoryError:
        raise
    except (seed_module.SceneInventoryError, seed_module.retained.SceneLineageError, ValueError, TypeError,
            KeyError, OverflowError, RecursionError, UnicodeError):
        raise SceneDownstreamInventoryError('scene_downstream_input_invalid') from None


def _join(intent_id, seed_records, downstream_records, roots, *, emission_budget=None, work_budget=None):
    if work_budget is not None:
        _work(work_budget)
        contracts.require(emission_budget is not None and getattr(emission_budget, 'work_budget', None) is work_budget,
                 'parameters_invalid', **_work_kwargs(work_budget))
    c = contracts
    if emission_budget is not None:
        emission_budget = emission_budget.scope(max_bytes=MAX_OUTPUT_BYTES, max_rows=MAX_ROWS, max_references=MAX_REFERENCES)
    c.require(c.matches(intent_id, c.ID, **_work_kwargs(work_budget)) and isinstance(seed_records, dict)
              and (_work_collect(work_budget, set, seed_records) if work_budget is not None else set(seed_records)) == seed_module._ROLES | {'intent', 'projection'}
              and all(isinstance(seed_records[r], (list, tuple)) for r in (_work_items(seed_module._ROLES, work_budget) if work_budget is not None else seed_module._ROLES))
              and isinstance(downstream_records, dict) and (_work_collect(work_budget, set, downstream_records) if work_budget is not None else set(downstream_records)) == ROLES
              and all(isinstance(rows, (list, tuple)) for rows in (_work_items(downstream_records.values(), work_budget) if work_budget is not None else downstream_records.values()))
              and isinstance(roots, dict) and (_work_collect(work_budget, set, roots) if work_budget is not None else set(roots)) == seed_module._ROOTS | EXTRA_ROOTS, 'parameters_invalid', **_work_kwargs(work_budget))
    roots = {k: c.path(v, **_work_kwargs(work_budget)) for k, v in (_work_items(roots.items(), work_budget) if work_budget is not None else roots.items())}
    groups = {'intent': [seed_records['intent']], 'projection': [] if seed_records['projection'] is None else [seed_records['projection']],
              **{r: seed_records[r] for r in (_work_items(seed_module._ROLES, work_budget) if work_budget is not None else seed_module._ROLES)}, **downstream_records}
    limits = {name: globals()[name] for name in (_work_items(('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
                                               'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH'), work_budget) if work_budget is not None else ('MAX_RECORD_BYTES', 'MAX_TOTAL_BYTES', 'MAX_OUTPUT_BYTES',
                                               'MAX_RECORDS', 'MAX_REFERENCES', 'MAX_ROWS', 'MAX_NODES', 'MAX_DEPTH'))}
    decoded = c.decode(groups, limits, **_work_kwargs(work_budget))
    context = c.Context(decoded, roots, limits, intent_id, emission_budget=emission_budget, **_work_kwargs(work_budget))
    context.references()
    if emission_budget is not None:
        seed = seed_module._join(intent_id, seed_records, {k: roots[k] for k in (_work_items(seed_module._ROOTS, work_budget) if work_budget is not None else seed_module._ROOTS)}, emission_budget=emission_budget, **_work_kwargs(work_budget))
    else:
        seed = seed_module.join_retained_scene_inventory_seed(intent_id=intent_id, records=seed_records,
                                                              roots={k: roots[k] for k in (_work_items(seed_module._ROOTS, work_budget) if work_budget is not None else seed_module._ROOTS)})
    context.seed_budget(seed)
    activations, matched = execution.activation(context, seed, **_work_kwargs(work_budget))
    launches, bound = execution.launches(context, matched, **_work_kwargs(work_budget))
    terminals = terminal.terminal(context, bound, **_work_kwargs(work_budget))
    compilations = terminal.compilations(context, **_work_kwargs(work_budget))
    result = {'schema_version': 'task_evaluation_scene_downstream_inventory.v1', 'scope': 'supplied_retained_execution_records',
              'status': 'kept_unresolved', 'intent_id': intent_id, 'seed': seed,
              'activation_observations': activations, 'launch_observations': launches, 'terminal_observations': terminals,
              'compilation_observations': compilations, 'raw_versions': context.raw,
              'raw_reference_obligations': context.obligations, 'remote_reference_obligations': context.remote,
              'structural_join_obligations': context.structural, 'lexical_members': context.members,
              'mutations': 0, **{flag: False for flag in (_work_items(FALSE_FLAGS, work_budget) if work_budget is not None else FALSE_FLAGS)}}
    rows = sum(len(value) for value in (_work_items(result.values(), work_budget) if work_budget is not None else result.values()) if isinstance(value, list))
    rows += sum(len(value) for value in (_work_items(seed.values(), work_budget) if work_budget is not None else seed.values()) if isinstance(value, list))
    c.require(rows <= MAX_ROWS, 'rows_limit', **_work_kwargs(work_budget))
    if emission_budget is not None:
        emission_budget.check_document(result)
    c.bounded_size(result, MAX_OUTPUT_BYTES, **_work_kwargs(work_budget))  # Refuse BEFORE bulk row-key/document serialization.
    for key, value in (_work_items(result.items(), work_budget) if work_budget is not None else result.items()):
        if isinstance(value, list):
            result[key] = c.unique(value, MAX_OUTPUT_BYTES, **_work_kwargs(work_budget))
    return result
