"""Capture action proof must select the original native generation and delivery."""

import json
from pathlib import Path

import pytest

from tests.test_capture_generation_birth import _fixture


def test_action_generation_selects_original_capture_proofs(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_scene_retirement import _generation
    from blueprint_pipeline.task_evaluation_scene_retirement_generations import birth_capture_member

    _, policy, target, owner, selector, raw = _fixture(tmp_path, monkeypatch)
    born = birth_capture_member(target, observation=owner, membership_selector=selector,
                                membership_raw=raw)
    info = target.stat()
    member = dict(canonical_path=str(target), **{'class': 'site_capture'},
                  generation_id=born['generation_id'], dev=info.st_dev, ino=info.st_ino,
                  mode=info.st_mode, inventory_sha256='sha256:' + 'b' * 64,
                  capture_owner_user_id=owner['capture_owner']['user_id'],
                  request_id=owner['request_id'], sponsoring_intent_id='scene-1',
                  owner_observation_raw_ref=born['owner_observation_raw_ref'],
                  birth_delivery_raw_ref=born['birth_delivery_raw_ref'],
                  source_membership_raw_ref=json.loads(
                      Path(born['birth_delivery_raw_ref']['path']).read_bytes())[
                          'source_membership_raw_ref'],
                  association_raw_ref=born['birth_delivery_raw_ref'],
                  scene_intent_raw_ref=born['owner_observation_raw_ref'])
    assert _generation(policy, member, expected_states={'active'})[0] == born
    for changed in (dict(capture_owner_user_id='sponsor-2'),
                    dict(owner_observation_raw_ref=member['birth_delivery_raw_ref']),
                    dict(source_membership_raw_ref=member['owner_observation_raw_ref'])):
        with pytest.raises(ValueError, match='scene_retirement_generation_unavailable'):
            _generation(policy, dict(member, **changed), expected_states={'active'})
