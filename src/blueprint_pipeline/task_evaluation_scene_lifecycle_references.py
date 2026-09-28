"""Historical scoped observers share one allowance; no reference clearance."""
from __future__ import annotations

from dataclasses import fields, is_dataclass

from .control_plane_storage_pin_observation import observe_storage_pins
from .control_plane_queue_observation import QueueRootContract, observe_queue_states
from .control_plane_queue_auxiliary_observation import AuxiliaryQueueContract, observe_preparation_sam_auxiliaries
from .control_plane_preparation_activation_references import (
    ReferenceFamilyContract, RetainedReferenceRecord, interpret_preparation_activation_references,
)
from .task_evaluation_scene_lineage_budget import _work_items


def _copy(value, budget):
    """Bounded conversion only after a child bounded its native output."""
    budget.charge('values')
    if is_dataclass(value):
        result = {}
        for field in _work_items(fields(value), budget):
            budget.charge('facts')
            result[field.name] = _copy(getattr(value, field.name), budget)
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in _work_items(value, budget):
            budget.charge('facts')
            result.append(_copy(item, budget))
        return result
    if isinstance(value, dict):
        result = {}
        for key, item in _work_items(value.items(), budget):
            budget.charge('facts')
            result[key] = _copy(item, budget)
        return result
    return value


def observe(context, now, budget, sink):
    reader_bytes = {'pins': 0, 'primary_queues': 0, 'auxiliary_queues': 0}
    supplied_work = 0
    blockers, child_scopes, protections = set(), sink.rows(), sink.rows()
    dispositions = sink.rows()
    records = []
    family_roots = {}
    for contract in _work_items(context['reference_family_contracts'], budget):
        budget.charge('facts')
        family_roots[contract['queue_root']] = contract['family']
    try:
        for name, callback in (
            ('pins', lambda: observe_storage_pins(context['pins_root'], observed_at_epoch=now, budget=budget)),
            ('primary_queues', lambda: observe_queue_states(
                [QueueRootContract(row['root_path'], tuple(row['states']))
                 for row in _work_items(context['primary_queue_contracts'], budget)], observed_at_epoch=now, budget=budget)),
            ('auxiliary_queues', lambda: observe_preparation_sam_auxiliaries(
                [AuxiliaryQueueContract(**row) for row in _work_items(context['auxiliary_queue_contracts'], budget)],
                observed_at_epoch=now, budget=budget)),
        ):
            before = budget.counts['raw_bytes']
            try:
                observed = callback()
            finally:
                reader_bytes[name] = budget.counts['raw_bytes'] - before
            budget.tick()
            for blocker in _work_items(observed.blockers, budget):
                if len(blockers) < 32:
                    blockers.add(blocker)
            budget.available('rows', 1)
            sink.available_occurrence()
            child_scopes.append({'child': name, 'scope': observed.scope, 'complete': observed.complete,
                                 'historical_only': True, 'consumer_fence_checked': False, 'action': 'KEEP'})
            if name == 'pins':
                for row in _work_items(observed.rows, budget):
                    budget.available('rows', 1)
                    sink.available_occurrence()
                    budget.measure(row, cap=sink.remaining_bytes)
                    protections.append({'kind': 'pin_observation', 'observation': _copy(row, budget), 'action': 'KEEP'})
                continue
            for row in _work_items(observed.rows, budget):
                # SAM rows remain raw scoped evidence; this interpreter has no
                # supported SAM contract. Never relabel them as preparation.
                family = family_roots.get(row.root_path)
                role = 'envelope' if name == 'primary_queues' else row.layout_role
                if family is None or (name == 'auxiliary_queues' and row.family != family) or role not in {
                    'envelope', 'identity', 'result', 'result_conflict'
                }:
                    budget.available('rows', 1)
                    sink.available_occurrence()
                    protections.append({'kind': 'unsupported_queue_observation', 'path': row.row_path,
                                        'raw_sha256': row.raw_sha256, 'raw_size_bytes': row.raw_size_bytes,
                                        'scope': observed.scope, 'action': 'KEEP'})
                    continue
                budget.charge('facts')
                budget.measure(row.raw_text)
                budget.tick()
                raw = row.raw_text.encode('utf-8')
                budget.tick()
                records.append(RetainedReferenceRecord(family, row.root_path, role, row.row_path, raw, row.row_identity))
        before = budget.counts['raw_bytes']
        try:
            interpreted = interpret_preparation_activation_references(
                [ReferenceFamilyContract(**row) for row in _work_items(context['reference_family_contracts'], budget)],
                records, budget=budget)
        finally:
            # Existing supplied-byte work charge is deliberately conservative,
            # distinct from physical reads and retained even on later refusal.
            supplied_work = budget.counts['raw_bytes'] - before
        budget.tick()
        for blocker in _work_items(interpreted.blockers, budget):
            if len(blockers) < 32:
                blockers.add(blocker)
        for row in _work_items(interpreted.records, budget):
            budget.available('rows', 1)
            sink.available_occurrence()
            budget.measure(row, cap=sink.remaining_bytes)
            dispositions.append(_copy(row, budget))
        for key in ('local_path_protections', 'remote_raw_references', 'raw_digest_selector_obligations',
                    'canonical_document_selector_obligations', 'missing_edge_obligations'):
            for fact in _work_items(getattr(interpreted, key), budget):
                budget.available('rows', 1)
                sink.available_occurrence()
                budget.measure(fact, cap=sink.remaining_bytes)
                protections.append({'kind': key, 'observation': _copy(fact, budget), 'action': 'KEEP'})
    except (ValueError, TypeError, OSError, UnicodeError, OverflowError, RecursionError):
        blockers.add(budget.failure or 'scene_lifecycle_reference_observation_unavailable')
    result = {'status': 'incomplete', 'historical_only': True, 'references_clear': False,
              'consumer_fence_checked': False, 'action': 'KEEP', 'mutations': 0,
              'child_scopes': child_scopes, 'record_dispositions': dispositions, 'protections': protections, 'blockers': sorted(blockers),
              'raw_accounting': {'reader_raw_bytes_by_child': reader_bytes,
                  'supplied_reference_input_work_bytes': supplied_work,
                  'cumulative_raw_allowance_bytes': budget.counts['raw_bytes'],
                  'conservative_accounting': True, 'scope': 'historical_observers_and_supplied_reference_input_work',
                  'reason': 'unchanged_interpreter_charges_supplied_input_occurrences_without_physical_reread'}}
    # Resource failure cannot authorize a fresh sink or another traversal. This
    # fixed framing retains already bounded evidence and actual charged deltas.
    if not budget.failure:
        sink.check_document(result)
    return result
