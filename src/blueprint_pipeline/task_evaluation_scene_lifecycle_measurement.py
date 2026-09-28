"""Metadata-only allocated inode accounting for exact retained member paths."""
from __future__ import annotations

import stat
from pathlib import PurePosixPath

from .task_evaluation_scene_lifecycle_acquisition import AcquisitionError, require
from .control_plane_disk_usage import allocated_bytes
from .task_evaluation_scene_lineage_budget import _work_items, _work_order

KINDS = {
    'capture_dependency': 'capture_pipeline', 'administrative_source_workspace': 'administrative_source_workspace',
    'preparation_workspace': 'preparation_workspace', 'configuration_progression_workspace': 'configuration_progression_workspace',
    'activation_workspace': 'activation_workspace', 'native_activation_workspace': 'activation_workspace',
    'preparation_projected_file': 'prepared_objects', 'prepared_cache_object': 'prepared_objects',
    'compiled_episode_packet': 'compilation_workspace', 'adapter_packet_root': 'compilation_workspace',
    'adapter_runtime_source_receipt': 'compilation_workspace',
    'sam_execution_dependency': 'sam_current_child', 'launch_workspace': 'launch_canary_workspace',
    'terminal_index_workspace': 'launch_canary_workspace', 'canary_evidence_workspace': 'launch_canary_workspace',
}


def members(historical, budget, sink):
    source = historical['source_family_inventory']
    downstream = source['downstream_inventory']
    layers = [(historical, 'declared_lexical_members'), (source, 'lexical_members'),
              (downstream, 'lexical_members'), (downstream['seed'], 'members')]
    indexed = {}
    for layer, key in _work_items(layers, budget):
        for row in _work_items(layer[key], budget):
            path = row['path']
            budget.charge('facts')
            indexed.setdefault(path, []).append(row)
    receipts = {}
    for row in _work_items(downstream['seed']['members'], budget):
        if row.get('kind') == 'preparation_projected_file':
            budget.charge('facts')
            receipts.setdefault((row['receipt_digest'], row['receipt_size_bytes']), []).append(row)
    for cache in _work_items(downstream['seed']['shared_cache_references'], budget):
        rows = receipts.get((cache['digest'], cache['size_bytes']), ())
        if not rows:
            continue
        sink.available_occurrence()
        proof = sink.reserve_provenance(p for row in _work_items(rows, budget) for p in row['source_provenance'])
        member = {'path': cache['path'], 'kind': 'prepared_cache_object',
                  'binding_strength': 'shared_cache_reference', 'source_provenance': proof}
        sink.reserve_row(member)
        budget.charge('facts')
        indexed.setdefault(cache['path'], []).append(member)
    return indexed


def allocated(info):
    blocks = getattr(info, 'st_blocks', None)
    require(blocks is None or type(blocks) is int and blocks >= 0, 'allocated_metadata_invalid')
    return allocated_bytes(info)


def measure(reader, historical, sink, families):
    budget = reader.budget
    declared = members(historical, budget, sink)
    ordered = _work_order(budget, sorted, declared)
    roots, parent_of = [], {}
    for path in _work_items(ordered, budget):
        ancestor = next((root for root in _work_items(roots, budget) if PurePosixPath(path).is_relative_to(PurePosixPath(root))), None)
        budget.charge('facts')
        if ancestor is None:
            roots.append(path)
        else:
            parent_of[path] = ancestor
    inode_index, rows, sharing, regular_keys = {}, sink.rows(), sink.rows(), {}
    complete_all = bool(roots)
    for root in _work_items(roots, budget):
        budget.available('rows', 1)
        sink.available_occurrence()
        budget.charge('facts')
        proofs = sink.reserve_provenance(proof for row in declared[root] for proof in row['source_provenance'])
        row = {'path': root, 'kinds': _work_order(budget, sorted, {r['kind'] for r in declared[root]}),
               'status': 'observed_scoped_metadata', 'measured_allocated_bytes': None,
               'observed_allocated_bytes': 0, 'measured_logical_bytes': None, 'observed_logical_bytes': 0,
               'measured_apparent_bytes': None, 'observed_apparent_bytes': 0,
               'logical_method': 'unique_regular_inode_stat_size_first_member_attribution',
               'apparent_method': 'regular_name_stat_size_including_hardlink_names',
               'allocated_method': 'unique_inode_stat_blocks_512_else_stat_size_first_member_attribution',
               'observed_regular_names': 0, 'observed_unique_regular_inodes': 0,
               'exclusive_ownership_proven': False, 'payload_bytes_verified': False, 'action': 'KEEP',
               'keeps': ['shared_content_object_not_exclusive'] if any(r['kind'] == 'prepared_cache_object' for r in declared[root]) else [], 'source_provenance': proofs}
        # Reserve conservative final numeric framing before any subtree access.
        # No allowance is refunded; emitted rows retain their ordinary charge.
        numeric_frame = dict(row)
        for key in ('measured_allocated_bytes', 'observed_allocated_bytes', 'measured_logical_bytes',
                    'observed_logical_bytes', 'measured_apparent_bytes', 'observed_apparent_bytes',
                    'observed_regular_names', 'observed_unique_regular_inodes'):
            numeric_frame[key] = (1 << 128) - 1
        sink.reserve_row(numeric_frame)
        row['keeps'] = sink.rows(row['keeps'])
        stack, owned, regular, total, complete = [root], set(), {}, 0, True
        root_device, logical, apparent = None, 0, 0
        while stack:
            budget.charge('values')
            path = stack.pop()
            try:
                info = reader.stat(path)
            except (OSError, AcquisitionError):
                if budget.failure:
                    raise
                complete = False
                row['keeps'].append('member_or_child_unavailable_or_changed')
                continue
            if root_device is None:
                root_device = info.st_dev
            if info.st_dev != root_device or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                complete = False
                row['keeps'].append('cross_device_link_or_special_member')
                continue
            inode = info.st_dev, info.st_ino
            require(type(info.st_size) is int and info.st_size >= 0, 'allocated_metadata_invalid')
            amount = allocated(info)
            if inode not in inode_index:
                require(len(inode_index) < 20_000, 'inode_index_limit')
                budget.charge('facts')
                inode_index[inode] = {'first_path': path, 'first_root': root, 'allocated_bytes': amount,
                                      'regular': stat.S_ISREG(info.st_mode), 'names': set(), 'nlink': info.st_nlink}
                total += amount
                if stat.S_ISREG(info.st_mode):
                    logical += info.st_size
            else:
                previous = inode_index[inode]
                require(previous['allocated_bytes'] == amount and previous['nlink'] == info.st_nlink,
                        'inode_metadata_changed')
                sharing.append({'first_path': previous['first_path'], 'also_observed_path': path,
                                'first_member_root': previous['first_root'], 'also_member_root': root,
                                'status': 'kept_shared_inode', 'action': 'KEEP', 'exclusive_ownership_proven': False})
            if stat.S_ISREG(info.st_mode):
                row['observed_regular_names'] += 1
                apparent += info.st_size
                budget.charge('facts')
                inode_index[inode]['names'].add(path)
                regular[inode] = info.st_nlink
            elif inode not in owned:
                owned.add(inode)
                try:
                    for name in _work_items(reader.entries(path), budget):
                        budget.charge('facts')
                        stack.append(path + '/' + name)
                except (OSError, AcquisitionError):
                    if budget.failure:
                        raise
                    complete = False
                    row['keeps'].append('member_directory_unavailable_or_changed')
        row['observed_unique_regular_inodes'] = len(regular)
        row['observed_logical_bytes'], row['observed_apparent_bytes'] = logical, apparent
        row['measured_logical_bytes'] = logical if complete else None
        row['measured_apparent_bytes'] = apparent if complete else None
        row['observed_allocated_bytes'] = total
        row['measured_allocated_bytes'] = total if complete else None
        if not complete:
            row['status'] = 'incomplete_scoped_metadata'
        regular_keys[root] = _work_order(budget, tuple, regular)
        if any(inode_index[inode]['nlink'] > len(inode_index[inode]['names'])
               for inode in _work_items(regular, budget)):
            row['keeps'].append('external_hardlink_or_unobserved_alias')
        row['keeps'] = _work_order(budget, sorted, set(row['keeps']))
        rows.append(row)
        complete_all = complete_all and complete
    # A later member may reveal another name for an already charged inode.
    # Only remove the conservative keep after all names are observed; emitted
    # rows never grow after their reservation and no allowance is refunded.
    for row in _work_items(rows, budget):
        if all(inode_index[inode]['nlink'] <= len(inode_index[inode]['names'])
               for inode in _work_items(regular_keys[row['path']], budget)):
            row['keeps'] = [reason for reason in _work_items(row['keeps'], budget)
                            if reason != 'external_hardlink_or_unobserved_alias']
    for path, ancestor in _work_items(parent_of.items(), budget):
        budget.available('rows', 1)
        sink.available_occurrence()
        rows.append({'path': path, 'status': 'coalesced_descendant_member', 'attributed_root': ancestor,
                     'measured_allocated_bytes': None, 'action': 'KEEP', 'exclusive_ownership_proven': False,
                     'source_provenance': sink.reserve_provenance(p for r in declared[path] for p in r['source_provenance'])})
    by_path = {}
    for row in _work_items(rows, budget):
        budget.charge('facts')
        by_path[row['path']] = row
    for family in _work_items(families, budget):
        selected = [path for path in _work_items(declared, budget)
                    if any(KINDS.get(row['kind']) == family['family'] for row in declared[path])]
        if selected:
            family['status'] = 'resolved_members'
            family['member_count'] = len(selected)
            own = [by_path[path]['measured_allocated_bytes'] for path in selected]
            family['measured_allocated_bytes'] = sum(own) if all(value is not None for value in own) else None
    total = sum(row['allocated_bytes'] for row in inode_index.values()) if inode_index else None
    require(total is None or type(total) is int and total >= 0, 'allocated_metadata_invalid')
    return rows, sharing, total, complete_all
