"""Bounded exact-scene metadata/size plan; always KEEP, never an apply path."""
from __future__ import annotations

import math
import json
import time
from pathlib import PurePosixPath

from .control_plane_reference_budget import ReferenceCollectionBudget
from . import task_evaluation_scene_compilation_native_owner_inventory as native
from . import task_evaluation_scene_lifecycle_acquisition as acquisition
from . import task_evaluation_scene_lifecycle_pool as pool_module
from .task_evaluation_scene_lineage_budget import RetainedEmissionBudget, _work_items

MAX_OUTPUT_BYTES = 16 * 1024 * 1024
FIELDS = {'roots', 'parent_routes', 'retained_metadata_roots', 'acquisition_anchors', 'retained_metadata_files',
          'pins_root', 'primary_queue_contracts', 'auxiliary_queue_contracts', 'reference_family_contracts', 'progression_config'}
FALSE_FLAGS = ('scene_inventory_complete', 'references_clear', 'process_fences_held', 'retirement_eligible',
               'cleanup_authorized', 'restore_verified', 'fresh_remote_readback_verified', 'execution_authorized',
               'current_provider_zero_verified', 'current_rights_checked', 'settlement_reopen_clear', 'owner_consent_verified',
               'complete_scene_inventory', 'consumer_fence_checked', 'filesystem_inventory_complete', 'host_history_complete',
               'scene_finished', 'paid_scope_verified')
FAMILIES = ('capture_pipeline', 'administrative_source_workspace', 'preparation_workspace',
            'configuration_progression_workspace', 'activation_workspace', 'prepared_objects',
            'compilation_workspace', 'sam_current_child', 'sam_original_child', 'launch_canary_workspace')
# Union of the actual preparation, activation and SAM producer states. The
# observers remain separate; a family unsupported by a child is not forwarded.
PRIMARY_STATES = set(pool_module.STATES) | {'prepared', 'waiting_external', 'failed'}
CONTRACT_FAMILIES = {'auxiliary_queue_contracts': {'preparation', 'sam'},
                     'reference_family_contracts': {'preparation', 'activation'}}


def fallback(code):
    return {'schema_version': 'task_evaluation_scene_lifecycle_plan.v1', 'status': 'incomplete', 'action': 'KEEP',
            'scope': 'selected_exact_scene_metadata_and_measured_paths', 'blockers': [code], 'mutations': 0,
            **{flag: False for flag in FALSE_FLAGS}}


def _context(value, budget):
    budget.tick()
    budget.measure(value)
    acquisition.require(isinstance(value, dict) and set(value) == FIELDS, 'context_invalid')
    roots = value['roots']
    expected = native.prior.downstream.seed_module._ROOTS | native.prior.downstream.EXTRA_ROOTS | native.prior.EXTRA_ROOTS
    acquisition.require(isinstance(roots, dict) and set(roots) == expected, 'context_roots_invalid')
    anchors = value['acquisition_anchors']
    acquisition.require(isinstance(anchors, list) and 1 <= len(anchors) <= 4, 'context_anchors_invalid')
    anchors = [acquisition.path(path, budget) for path in _work_items(anchors, budget)]
    acquisition.require(len(set(anchors)) == len(anchors), 'context_anchors_invalid')
    routes, metadata = value['parent_routes'], value['retained_metadata_roots']
    acquisition.require(isinstance(routes, list) and 1 <= len(routes) <= 4 and isinstance(metadata, list)
                        and 1 <= len(metadata) <= 8, 'context_routes_invalid')
    for path in _work_items([*roots.values(), *metadata, value['pins_root']], budget):
        acquisition.path(path, budget)
        acquisition.require(any(PurePosixPath(path).is_relative_to(PurePosixPath(anchor)) for anchor in anchors),
                            'context_path_outside_anchor')
    for route in _work_items(routes, budget):
        acquisition.require(isinstance(route, dict) and set(route) == {'queue_root', 'input_root'}, 'context_routes_invalid')
        for path in _work_items(route.values(), budget):
            acquisition.path(path, budget)
            acquisition.require(any(PurePosixPath(path).is_relative_to(PurePosixPath(anchor)) for anchor in anchors),
                                'context_path_outside_anchor')
    acquisition.require({'queue_root': roots['preparation_queue_root'], 'input_root': roots['preparation_input_root']} in routes,
                        'context_routes_invalid')
    selectors = value['retained_metadata_files']
    acquisition.require(isinstance(selectors, list) and len(selectors) <= 1024, 'context_selectors_invalid')
    selected = set()
    for row in _work_items(selectors, budget):
        acquisition.require(isinstance(row, dict) and set(row) == {'role', 'path'} and isinstance(row['role'], str)
                            and row['role'] in pool_module.SELECTOR_ROLES, 'context_selectors_invalid')
        path = acquisition.path(row['path'], budget)
        digest_revision=(row['role']=='configured_revisions'
            and acquisition.configured_revision_projection(path,roots['preparation_input_root']))
        acquisition.require((path.endswith('.json') or digest_revision)
                            and any(PurePosixPath(path).is_relative_to(PurePosixPath(root))
                            and path != root for root in metadata), 'context_selectors_invalid')
        acquisition.require(path not in selected, 'context_selectors_invalid')
        budget.charge('facts')
        selected.add(path)
    for field, keys in [('primary_queue_contracts', {'root_path', 'states'}),
                        ('auxiliary_queue_contracts', {'family', 'root_path'}),
                        ('reference_family_contracts', {'family', 'queue_root'})]:
        contracts = value[field]
        acquisition.require(isinstance(contracts, list) and 1 <= len(contracts) <= 16, 'context_contracts_invalid')
        for row in _work_items(contracts, budget):
            acquisition.require(isinstance(row, dict) and set(row) == keys, 'context_contracts_invalid')
            path = row.get('root_path', row.get('queue_root'))
            acquisition.path(path, budget)
            acquisition.require(any(PurePosixPath(path).is_relative_to(PurePosixPath(root)) for root in anchors),
                                'context_path_outside_anchor')
            if 'family' in row:
                acquisition.require(isinstance(row['family'], str) and row['family'] in CONTRACT_FAMILIES[field],
                                    'context_contracts_invalid')
            if 'states' in row:
                acquisition.require(isinstance(row['states'], list) and 1 <= len(row['states']) <= 16
                                    and all(isinstance(state, str) and state in PRIMARY_STATES
                                            for state in row['states']), 'context_contracts_invalid')
    config = value['progression_config']
    if config is not None:
        acquisition.path(config, budget)
        acquisition.require(config.endswith('.json') and any(PurePosixPath(config).is_relative_to(PurePosixPath(root))
                            and config != root for root in metadata), 'context_config_invalid')
    return value


def _finished(seed, decoded, context, intent_id, now, budget, scopes):
    from .task_evaluation_scene_lifecycle_finished import finished
    extension_root = context['roots']['intent_root'] + '/' + intent_id + '/execution-window-extensions'
    observed = any(row['role'] == 'extensions' and row['path'] == extension_root
                   and row['status'] == 'observed_selected_layout' for row in _work_items(scopes, budget))
    observed = observed and not any(row['role'] == 'extensions' and row['path'] == extension_root
            and row['status'] == 'unsupported_layout' for row in _work_items(scopes, budget))
    return finished(seed['history'], decoded, context['roots'], intent_id, now, budget,
                    extensions_observed=observed)


def _families(sink):
    return list({'family': family, 'status': 'not_observed_within_scope', 'action': 'KEEP',
                      'measured_allocated_bytes': None, 'member_count': 0, 'restore_verified': False}
                     for family in FAMILIES)


def build_scene_lifecycle_plan(*, intent_id, context, observed_at_epoch, monotonic=time.monotonic, time_budget_seconds=30.0):
    try:
        budget = ReferenceCollectionBudget._for_scene_lifecycle_plan(monotonic=monotonic, time_budget_seconds=time_budget_seconds)
    except ValueError:
        return fallback('scene_lifecycle_budget_invalid')
    return _build_scene_lifecycle_plan(intent_id=intent_id, context=context, observed_at_epoch=observed_at_epoch, budget=budget)


def _build_scene_lifecycle_plan(*, intent_id, context, observed_at_epoch, budget, context_anchor=None):
    # Admission precedes every candidate callback and cleanup. A constructor
    # marker alone does not prove initialization survived its allocations.
    if type(budget) is not ReferenceCollectionBudget:
        return fallback('scene_lifecycle_budget_invalid')
    state = vars(budget)
    fields = ('_monotonic', '_duration', '_deadline', '_last', '_closed',
              '_failure', '_limits', '_counts', 'blockers')
    if (state.get('_initialization_started') is not True or not all(name in state for name in fields)
            or not callable(state['_monotonic']) or type(state['_duration']) is not float
            or not math.isfinite(state['_duration']) or not 0 < state['_duration'] <= 30
            or type(state['_closed']) is not bool or type(state['_limits']) is not dict
            or type(state['_counts']) is not dict or type(state['blockers']) is not set
            or len(state['_limits']) != 8 or state['_limits'].keys() != state['_counts'].keys()):
        return fallback('scene_lifecycle_budget_invalid')
    reader = None
    result, reference_result, sink, measured_by_path = None, None, None, None
    try:
        budget.tick()
        acquisition.require(type(budget) is ReferenceCollectionBudget and isinstance(intent_id, str)
                            and native.c.matches(intent_id, native.c.OWNER_ID)
                            and type(observed_at_epoch) in (int, float) and math.isfinite(observed_at_epoch)
                            and observed_at_epoch >= 0, 'parameters_invalid')
        context = _context(context, budget)
        if context_anchor is None:
            reader = acquisition.Acquisition(budget, context['acquisition_anchors'])
        else:
            acquisition.require(type(context_anchor) is acquisition.ContextAnchor
                                and context_anchor.reader.budget is budget, 'context_anchor_invalid')
            reader = context_anchor.reader
            context_anchor.coalesced = reader.add_planner_anchors(context['acquisition_anchors'])
        pool = pool_module.Pool(reader, context, intent_id)
        pool.discovery()
        decoded = pool.decode()
        seed, downstream, source, bridge, protected = pool_module.select(decoded, context, intent_id, budget)
        from .task_evaluation_scene_lifecycle_metadata import configuration, other_capture_references
        config_observation = configuration(decoded, context, budget)
        sink = RetainedEmissionBudget(max_bytes=MAX_OUTPUT_BYTES, max_rows=10_000, max_references=10_000, work_budget=budget)
        strict_roles = native.prior.downstream.seed_module._ROLES | {'intent', 'projection', 'parent_envelopes', 'parent_results'}
        strict_unknown = seed['intent'] is None or any(
            row['status'] == 'kept_unsupported_schema' and row['role'] in strict_roles
            for row in _work_items(protected, budget))
        if strict_unknown:
            result = fallback('strict_lineage_join_unavailable')
            result.update(intent_id=intent_id, observed_at_epoch=observed_at_epoch,
                          finished_observation={'status': 'unknown', 'reason': 'strict_lineage_join_unavailable',
                                                'finished_for_cleanup_authority': False},
                          configuration_observation=config_observation,
                          acquisition_scopes=sink.rows(pool.scopes),
                          unselected_metadata_protections=sink.rows(protected),
                          acquired_metadata_protections=sink.rows(
                              {'role': row['role'], 'path': row['path'], 'sha256': row['sha256'],
                               'size_bytes': len(row['raw']), 'status': 'kept_join_unavailable',
                               'current_owner_binding_verified': False}
                              for row in _work_items(decoded, budget)),
                          family_obligations=sink.rows(_families(sink)), measured_members=sink.rows(),
                          sharing=sink.rows(), reference_keeps=sink.rows(), unique_observed_allocated_bytes=None,
                          supported_family_validation_available=False)
            sink.reserve_row(config_observation)
        else:
            historical = native._join(intent_id, seed, downstream, source, bridge, context['roots'], context['parent_routes'],
                                      context['retained_metadata_roots'], emission_budget=sink, work_budget=budget)
            seed_result = historical['source_family_inventory']['downstream_inventory']['seed']
            result = {'schema_version': 'task_evaluation_scene_lifecycle_plan.v1', 'status': 'incomplete', 'action': 'KEEP',
                      'scope': 'selected_exact_scene_metadata_and_measured_paths', 'observed_at_epoch': observed_at_epoch,
                      'intent_id': intent_id, 'selected_intent_provenance': seed_result['intent_provenance'],
                      'finished_observation': _finished(seed_result, decoded, context, intent_id, observed_at_epoch, budget, pool.scopes),
                      'historical_lineage': historical, 'configuration_observation': config_observation, 'acquisition_scopes': sink.rows(pool.scopes),
                      'unselected_metadata_protections': sink.rows(protected), 'family_obligations': _families(sink),
                      'measured_members': sink.rows(), 'sharing': sink.rows(), 'reference_keeps': sink.rows(),
                      'unique_observed_allocated_bytes': None, 'blockers': ['reference_and_consumer_lifetime_unproven'],
                      'mutations': 0, **{flag: False for flag in FALSE_FLAGS}}
            sink.reserve_row(config_observation)
            from .task_evaluation_scene_lifecycle_measurement import measure
            result['measured_members'], result['sharing'], result['unique_observed_allocated_bytes'], _, measured_by_path = measure(
                reader, historical, sink, result['family_obligations'], context['roots'])
            result['family_obligations'] = sink.rows(result['family_obligations'])
        # Target evidence only: action admission additionally binds the exact
        # contracts to protected installed policy and repeats them under EX.
        sink.reserve_row(context)
        result['planner_context'] = context
        if context_anchor is not None:
            sink.reserve_row({'metadata_only': True, 'anchor_coalesced': context_anchor.coalesced})
            result['context_acquisition'] = {'metadata_only': True, 'anchor_coalesced': context_anchor.coalesced}
        from .task_evaluation_scene_lifecycle_references import observe, intersect
        reference_result = observe(context, observed_at_epoch, budget, sink)
        other_keeps = other_capture_references(decoded, context, intent_id, budget, sink)
        reference_result['protections'].extend(other_keeps)
        result['other_owner_capture_keeps'] = other_keeps
        result['reference_observation'] = reference_result
        result['reference_keeps'] = intersect(result['measured_members'], reference_result, budget, sink,
                                              measured_by_path=measured_by_path)
        result['planner_acquired_raw_bytes'] = reader.physical_read_bytes
        reader.verify()
        _screen_output(result, budget)
        sink.check_document(result)
        native.c.retained.c.bounded_size(result, MAX_OUTPUT_BYTES, work_budget=budget)
        budget.tick()
    except (ValueError, OSError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as error:
        ordinary_drift = (isinstance(error, OSError) or isinstance(error, acquisition.AcquisitionError)
            and str(error) in {'scene_lifecycle_metadata_changed', 'scene_lifecycle_metadata_unavailable'}) and not budget.failure
        if result is not None and sink is not None and ordinary_drift:
            try:
                # Accepted positives stay historical KEEP evidence while the
                # same B permits bounded framing. Current measured totals become
                # unknown after a later metadata verification failure.
                code = 'metadata_changed_after_observation'
                sink.reserve_row({'reason': code})
                result['blockers'].append(code)
                for row in _work_items(result['measured_members'], budget):
                    sink.reserve_row({'reason': code})
                    row['status'] = 'incomplete_scoped_metadata'
                    row['measured_allocated_bytes'] = None
                    if 'measured_logical_bytes' in row:
                        row['measured_logical_bytes'] = row['measured_apparent_bytes'] = None
                    row.setdefault('keeps', []).append(code)
                for family in _work_items(result['family_obligations'], budget):
                    family['measured_allocated_bytes'] = None
                _screen_output(result, budget)
                sink.check_document(result)
            except (ValueError, TypeError, OverflowError, RecursionError):
                result = fallback(budget.failure or 'scene_lifecycle_output_unproven')
        else:
            result = fallback(budget.failure or 'scene_lifecycle_metadata_or_context_unproven')
        if reference_result is not None and 'historical_lineage' not in result:
            result['raw_accounting'] = reference_result['raw_accounting']
        if reader is not None:
            result['planner_acquired_raw_bytes'] = reader.physical_read_bytes
    finally:
        if reader is not None:
            try:
                reader.close()
            except ValueError:
                result = fallback('scene_lifecycle_descriptor_cleanup_unproven')
        if context_anchor is not None and type(context_anchor) is acquisition.ContextAnchor:
            try:
                budget.tick()
                _screen_output(result, budget)
                budget.measure(result, cap=MAX_OUTPUT_BYTES)
                budget.tick()
                context_anchor.serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
                budget.tick()
            except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError):
                result = fallback(budget.failure or 'scene_lifecycle_publication_unproven')
                context_anchor.serialized = json.dumps(result, sort_keys=True, separators=(',', ':'))
        budget.close()
    return result


def _screen_output(value, budget):
    """Refuse secret-shaped output identities rather than editing their meaning."""
    pending = [iter(((None, value),))]
    while pending:
        budget.available('values', 1)
        try:
            key, item = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        budget.charge('values')
        if isinstance(item, dict):
            pending.append(iter(item.items()))
        elif isinstance(item, list):
            pending.append(iter((key, entry) for entry in item))
        elif isinstance(item, str) and isinstance(key, str) and (
                key in {'path', 'paths', 'uri', 'uris', 'directory'}
                or key.endswith(('_path', '_paths', '_root', '_roots', '_uri', '_uris'))):
            acquisition.require(len(item) <= 4096, 'output_identity_invalid')
            from .control_plane_disk_usage import _CREDENTIAL_SHAPED_NAME
            acquisition.require(_CREDENTIAL_SHAPED_NAME.search(item) is None, 'credential_shaped_output')
            budget.tick()
            lowered = item.lower()
            acquisition.require(not any(marker in lowered for marker in
                ('x-amz-', 'signature=', 'token=', 'credential=', 'password=', 'api_key=', 'secret=', 'access_key=')),
                'credential_shaped_output')
            parts = PurePosixPath(item).parts
            acquisition.require(not any(part.lower() in {'.env', 'credentials.json', 'secrets.json', 'private_key.pem'}
                                        for part in _work_items(parts, budget)), 'credential_shaped_output')


if __name__ == '__main__':
    from .task_evaluation_scene_lifecycle_cli import main
    raise SystemExit(main())
