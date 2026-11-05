"""One admission state and unchanged producer birth through the public facade."""

import importlib
import inspect

import pytest


def test_facade_and_pin_lifetime_share_policy_and_acquisition_hooks(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    lifetime = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_lifetime')
    assert access.SceneRetirementAccessError is lifetime.SceneRetirementAccessError
    assert access.scene_access is lifetime.scene_access
    marker = object()
    monkeypatch.setattr(access, '_INSTALLED_POLICY', marker)
    assert lifetime._INSTALLED_POLICY is marker
    replacement = object()
    monkeypatch.setattr(lifetime, '_POLICY_UID', replacement)
    assert access._POLICY_UID is replacement
    calls = []
    def refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise access.SceneRetirementAccessError('fixture acquisition refused')
    monkeypatch.setattr(access, '_open_owned', refuse)
    with pytest.raises(access.SceneRetirementAccessError, match='fixture acquisition refused'):
        with lifetime._opened('/bounded-local-fixture'):
            pytest.fail('the original admission hook did not refuse')
    assert calls == [(('/', lifetime.os.O_RDONLY | lifetime.os.O_DIRECTORY), {})]
    monkeypatch.delattr(access, '_open_owned')
    assert not hasattr(lifetime, '_open_owned')
    monkeypatch.setattr(access, '_open_owned', refuse, raising=False)
    assert lifetime._open_owned is refuse


def test_a_facade_override_is_undone_to_the_lifetime_value():
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    lifetime = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_lifetime')
    # pytest 9.2's monkeypatch records an old value from the target's own __dict__ unless the target's
    # type holds a data descriptor for the name; finding none here, its undo deleted the leaf's value
    # (every later access fixture then failed: no attribute '_POLICY_UID').
    for name in sorted(access._LIFETIME_NAMES - set(vars(access))):
        descriptor = inspect.getattr_static(type(access), name, None)
        assert hasattr(type(descriptor), '__set__') and hasattr(type(descriptor), '__delete__'), name
    originals = {name: getattr(lifetime, name) for name in ('_POLICY_UID', '_INSTALLED_POLICY', '_open_owned')}
    with pytest.MonkeyPatch.context() as patch:
        for name in originals:
            patch.setattr(access, name, object())
        patch.delattr(access, '_open_owned')
        assert not hasattr(lifetime, '_open_owned')
    assert {name: getattr(lifetime, name) for name in originals} == originals
    assert access._POLICY_UID is lifetime._POLICY_UID and '_POLICY_UID' not in vars(access)


def test_facade_preserves_actual_producer_birth_and_own_source_identity(monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations
    lifetime = importlib.import_module('blueprint_pipeline.task_evaluation_scene_retirement_lifetime')
    assert access.__file__ != lifetime.__file__
    assert access.__spec__.name.endswith('task_evaluation_scene_retirement_access')
    calls = []
    def birth(*args, **kwargs):
        calls.append((args, kwargs))
        return 'original producer delegated'
    monkeypatch.setattr(generations, 'birth_member', birth)
    selectors = dict(owner_intent_id='owner', owner_raw_ref={'path': 'owner'},
                     birth_request_raw_ref={'path': 'birth'}, now=123)
    assert access.birth_scene_member('/member', **selectors) == 'original producer delegated'
    assert calls == [(('/member',), selectors)]
