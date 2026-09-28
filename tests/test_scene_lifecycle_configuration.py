# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_scene_lifecycle_metadata.py
"""Sealed installed configuration metadata is not live service authority."""
import hashlib
import json

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
from tests.test_scene_lifecycle_plan import context_fixture
from tests.test_scene_inventory_history import seal


def row(context, **changes):
    roots = context['roots']
    config = {'schema_version': 'task_evaluation_scene_progression_config.v1',
        'intent_root': roots['intent_root'], 'factory_output_root': roots['factory_output_root'],
        'capture_store_root': roots['pubsub_root'], 'preparation_queue_root': roots['preparation_queue_root'],
        'preparation_worker': {'input_root': roots['preparation_input_root']},
        'child_queue_root': roots['sam_queue_root'], 'child_execution_root': roots['sam_execution_root'],
        'launch_execution_root': roots['launch_execution_root'], 'terminal_result_root': roots['terminal_result_root'], **changes}
    value = seal(config, 'config_digest')
    raw = json.dumps(value).encode()
    return {'role': 'progression_config', 'path': context['retained_metadata_roots'][0]+'/progression.json',
            'sha256': 'sha256:'+hashlib.sha256(raw).hexdigest(), 'raw': raw, 'value': value}


def test_retained_config_matches_exact_declared_roots_without_live_config_claim(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_metadata import configuration
    context, _ = context_fixture(tmp_path)
    observed = configuration([row(context)], context, ReferenceCollectionBudget(monotonic=lambda: 0))
    assert observed['status'] == 'retained_configuration_roots_matched'
    assert observed['running_configuration_verified'] is False and observed['service_fences_checked'] is False
    assert observed['source_provenance'][0]['seal_field'] == 'config_digest'


def test_available_config_root_contradiction_refuses_independently_of_owner_history(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_metadata import configuration
    context, _ = context_fixture(tmp_path)
    with pytest.raises(ValueError, match='configuration_root_mismatch'):
        configuration([row(context, child_execution_root='/foreign/sam')], context,
                      ReferenceCollectionBudget(monotonic=lambda: 0))


def test_missing_and_future_config_are_explicit_unknown(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_lifecycle_metadata import configuration
    context, _ = context_fixture(tmp_path)
    for rows in ([], [row(context, schema_version='future_config.v99')]):
        observed = configuration(rows, context, ReferenceCollectionBudget(monotonic=lambda: 0))
        assert observed['status'] == 'configuration_unproven'
        assert observed['running_configuration_verified'] is False


def test_real_acquired_config_enters_report_before_planning(tmp_path):
    from tests.test_scene_lifecycle_plan import run
    from pathlib import Path
    context, intent = context_fixture(tmp_path)
    config = row(context)
    target = Path(config['path'])
    target.parent.mkdir(parents=True)
    target.write_bytes(config['raw'])
    context['progression_config'] = config['path']
    report = run(context, intent)
    assert report['configuration_observation']['status'] == 'retained_configuration_roots_matched'
