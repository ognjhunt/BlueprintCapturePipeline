"""The connected action fixture must retain genuine current producer records."""
import json

from tests.test_scene_lifecycle_connected_acquisition import full_connected_finished_scene
from tests.test_scene_retirement_connected_acceptance import _native_fixture_records


def retained_envelopes():
    args=_native_fixture_records(full_connected_finished_scene())
    for group in ('seed_records','source_records','bridge_records'):
        for role,rows in args[group].items():
            if role not in {'preparation_envelopes','sam_parent_envelopes',
                'activation_envelopes','native_preparation_envelopes','native_activation_envelopes'}:
                continue
            yield from rows


def test_finished_fixture_uses_terminal_activation_route_not_pending_live_record():
    paths=[path for path,raw in retained_envelopes()
        if json.loads(raw)['schema_version']=='task_evaluation_launch_activation_envelope.v1']
    assert paths and all('/prepared/' in path for path in paths)


def test_current_fixture_requests_preserve_known_shape_without_invented_local_revision_reference():
    from blueprint_pipeline.control_plane_preparation_activation_references import (
        ReferenceFamilyContract,RetainedReferenceRecord,interpret_preparation_activation_references,
    )
    records=[]
    for path,raw in retained_envelopes():
        family='activation' if json.loads(raw)['schema_version']=='task_evaluation_launch_activation_envelope.v1' else 'preparation'
        root=path.rsplit('/',2)[0]
        records.append(RetainedReferenceRecord(family,root,'envelope',path,raw))
    contracts=[ReferenceFamilyContract(family,root) for family,root in sorted({(r.family,r.queue_root) for r in records})]
    observed=interpret_preparation_activation_references(contracts,records)
    invalid={'supported_request_invalid','supported_reference_invalid','unsupported_metadata_shape'}
    assert not (set(observed.blockers)&invalid), observed.blockers
    # Missing results/identity/remote evidence remain actual obligations; this
    # grammar check is not a producer authorization or reader-closure grant.
    assert not observed.references_clear and not observed.execution_authorized
