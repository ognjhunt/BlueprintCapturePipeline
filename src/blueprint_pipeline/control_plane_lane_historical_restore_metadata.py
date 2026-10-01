"""ADP-009D/day28: allocation eligibility before private restore effects.

Original metadata is a shape rehearsal only. It never represents a new inode,
restored snapshot, producer completion or execution authorization.
"""
from . import control_plane_lane_historical_generation as generation
from . import control_plane_lane_owner_consents as owners
from . import control_plane_lane_scratch_decisions as retained
from .control_plane_lane_historical_journal import MAX_EVENTS, event_byte_limit
from .control_plane_lane_experiment_publication import _CAPS


def _object_width(fields):
    """Counter-only JSON shapes; no values represent observed kernel facts."""
    return (1 + len(fields) + sum(nodes for nodes, _ in fields.values()),
            len(fields) + sum(keys for _, keys in fields.values()))


_SCALAR = (1, 0)
_VERSION = (11, 0)  # list plus ten native integer fields
_SELECTOR = _object_width(dict(sha256=_SCALAR, size_bytes=_SCALAR))
_RESUME = _object_width(dict(decision_id=_SCALAR, decision=_SELECTOR))
_BODY = {key: _SCALAR for key in (
    'phase path stage_path parent_path sha256 size_bytes uncertain uid gid mode '
    'role token expected_bytes device owner_pid archive_sha256 archive_size_bytes '
    'restored_files restored_logical_bytes status action action_id owner generation_digest '
    'original_final_event_digest fresh_disk_reservation root_directory_retained '
    'owner_access_reopened decision_id original_head_event_digest '
    'credited_removed_allocated_bytes boot_id started_at_epoch started_monotonic'
).split()}
_BODY.update({key: _VERSION for key in (
    'version parent_version stage_version target_version member_version protected_root_version'
).split()})
_BODY.update({key: _SELECTOR for key in ('original_manifest', 'restored_snapshot', 'decision')})
_BODY.update(dict(delete_resume=_RESUME, observation_resume=_RESUME))
_BODY_WIDTH = _object_width(_BODY)
# Ten envelope fields: nine scalars and one body. The body union bounds every
# field emitted by RestoreTree, restore final and reconciliation producers.
_EVENT_WIDTH = _object_width(dict(body=_BODY_WIDTH,
    **{key: _SCALAR for key in (
        'schema_version action_id scope_digest sequence kind previous_event_digest '
        'observed_at_epoch execution_authorized event_digest').split()}))
RESTORE_EVENT_VALUE_WORK = 2 * _EVENT_WIDTH[0] + _EVENT_WIDTH[1]


def private_snapshot_byte_upper(original, raw_size):
    # Names/kinds/hash/payload sizes stay exact. Mutable member/root versions
    # have ten native64 fields; two identities have six numeric fields each.
    # The aggregate allocation bound includes all4096 rows and512-byte blocks.
    aggregate_width = len(str((2**64 - 1) * generation.MAX_MEMBERS * 512))
    return raw_size + (len(original['members']) + 2) * 10 * 21 + 12 * 21 + aggregate_width


def _eligible(counts, limits, **reserve):
    generation._require(all(0 <= counts[key] <= limits[key] - value
                           for key, value in reserve.items()), 'restore_metadata_ineligible')


def preflight_restore_metadata(worker):
    """Rehearse snapshot allocation inside genuine pre/post selection gates.

    The selected original and eventual private snapshot have the same JSON
    containers, keys and version-list lengths. Exercise both publication
    encodings and the strict readback allocation without creating any snapshot.
    The acquisition's original counters and five-second deadline include its
    actual final current-owner selection. No counter is reset in this phase.
    """
    with worker.checkpoint(journal=True) as (files, _, journal):
        raw = owners._encoded(worker.selected[2], files.budget,
                             cap=generation.MAX_MANIFEST_BYTES)
        retained._document(raw, generation.MAX_MANIFEST_BYTES,
                           _work_budget=files.budget)
        before_exit = dict(files.budget.counts)
        budget = files.budget
        # Only diagnostic workload is normalized; actual B is never modified.
        read_values = getattr(journal, 'read_value_work', 0)
        read_raw = getattr(journal, 'read_raw_work', 0)
        scan_entries = getattr(journal, 'scan_entry_work', 0)
        read_entries = getattr(journal, 'read_entry_work', 0)
        current_events = getattr(journal, '_count', 0)
        head_fds = getattr(journal, 'diagnostic_head_fds', set())
        metadata_fds = len(files.owned) - len(head_fds.intersection(files.owned))
        probe_fds = len(files.probe_owned)
    original = worker.selected[2]
    upper = private_snapshot_byte_upper(original, len(raw))
    generation._require(upper <= generation.MAX_MANIFEST_BYTES, 'restore_metadata_ineligible')
    paths = [row['path'] for row in original['members']]
    ordinary_events = 3 * len(paths) + 2 * sum(bool(path) and '/' not in path for path in paths) + 7
    # Existing authenticated recovery/reservation records plus a complete
    # remaining ordinary transcript. Count the future snapshot entry separately.
    future_events = current_events + ordinary_events + 1
    generation._require(future_events - 1 <= MAX_EVENTS, 'restore_metadata_ineligible')
    adjusted = dict(budget.counts)
    adjusted['values'] -= read_values
    adjusted['raw_bytes'] -= read_raw
    adjusted['entries'] -= scan_entries + read_entries
    event_bytes = event_byte_limit(worker.selected)
    raw_overhead = max(12 * (event_bytes + 1) + 80,
                       9 * (event_bytes + 1) + 6 * _CAPS['event'] + 80)
    _eligible(adjusted, budget.limits,
        values=12 * RESTORE_EVENT_VALUE_WORK,
        raw_bytes=3 * upper + raw_overhead + 3,
        entries=future_events + 256 + 12 + 4,
        output_bytes=2 * upper - len(raw))
    generation._require(adjusted['raw_bytes'] + 3 * upper + raw_overhead + 3
                        <= files.raw_cap, 'restore_metadata_ineligible')
    generation._require(metadata_fds + 4 <= 104 and metadata_fds + probe_fds + 4 <= 128,
                        'restore_metadata_ineligible')
    worker.metadata_post_selection = {
        key: budget.counts[key] - before_exit[key] for key in ('values', 'raw_bytes')}


def strict_document_value_work(value, tick):
    """Count intrinsic strict-parser work on an already bounded JSON graph."""
    def visit(item):
        tick()
        if type(item) is dict:
            nodes, keys = 1 + len(item), len(item)
            for child in item.values():
                child_nodes, child_keys = visit(child)
                nodes += child_nodes
                keys += child_keys
            return nodes, keys
        if type(item) is list:
            nodes, keys = 1, 0
            for child in item:
                child_nodes, child_keys = visit(child)
                nodes += child_nodes
                keys += child_keys
            return nodes, keys
        return _SCALAR
    nodes, keys = visit(value)
    return 2 * nodes + keys
