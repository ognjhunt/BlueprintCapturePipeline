"""A successor attempt starts with no attempt-scoped state, including the activation link."""
from blueprint_pipeline.task_evaluation_scene_progression import ATTEMPT_STATE_KEYS, _clear_attempt


def test_successor_clears_every_attempt_scoped_key_including_the_activation_link():
    """Scene 840938, 2026-09-12: the release successor kept the previous attempt's
    activation link, so the new attempt activated with the old request digest and
    commit and the registry refused it (intent_registry_same_release_conflict)."""
    lineage = {"binding_digest": "sha256:" + "b" * 64, "source_binding": {"path": "/x"},
               "source_analysis": {"path": "/y"}, "release_predecessors": [{"basis": "x"}],
               "recovery_predecessors": [], "capacity_admission": {"status": "admitted"}}
    state = {**lineage, **{key: f"old-{key}" for key in ATTEMPT_STATE_KEYS}}
    assert {"activation_link", "activation", "preparation_link", "factory"} <= set(ATTEMPT_STATE_KEYS)
    _clear_attempt(state)
    assert state == lineage
