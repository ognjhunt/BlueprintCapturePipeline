import pytest

from blueprint_pipeline.artifixer_operator_training_selection import apply_user_exclusions
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _review():
    value = {'schema_version': 'task_evaluation_artifixer_ai_visual_review_execution.v1',
             'status': 'completed', 'provider_called': True, 'review_phase': 'pre_training_semantic_targets',
             'decision': 'accepted', 'reviewer': {'kind': 'ai'}, 'frames': [
                 {'camera_id': camera, 'frame_sha256': 'sha256:' + token * 64,
                  'decision': 'accepted', 'preserves_non_target_content': True}
                 for camera, token in [('source-01', 'a'), ('source-07', 'b')]]}
    value['execution_digest'] = canonical_digest(value, digest_field='execution_digest')
    return value


def test_veto_is_negative_only_and_preserves_original_provider_evidence():
    original = _review()
    result = apply_user_exclusions(review=original, excluded_frames={'source-07': 'sha256:'+'b'*64},
                                   authorization_reference='User requests correct 14 targets excluding 07 and 11')
    assert result['source_provider_review'] == original
    assert original['frames'][1]['decision'] == 'accepted'
    assert result['frames'][0] == original['frames'][0]
    assert result['frames'][1]['decision'] == 'rejected'
    assert result['reviewer']['kind'] == 'ai_review_with_explicit_user_exclusions'
    assert result['operator_selection']['new_provider_call_performed'] is False


@pytest.mark.parametrize('exclusions', [{'source-07': 'sha256:'+'c'*64}, {'unknown': 'sha256:'+'b'*64}])
def test_veto_refuses_changed_or_unknown_frame(exclusions):
    with pytest.raises(ValueError, match='frame_binding'):
        apply_user_exclusions(review=_review(), excluded_frames=exclusions, authorization_reference='user')


def test_veto_refuses_changed_source_receipt():
    original = _review(); original['decision'] = 'rejected'
    with pytest.raises(ValueError, match='source_review'):
        apply_user_exclusions(review=original, excluded_frames={'source-07':'sha256:'+'b'*64}, authorization_reference='user')
