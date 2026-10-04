"""Actual listener, owner and billing readers must hold the coarse scene fence."""
import threading
import pytest
from tests.test_scene_retirement_real_participants import access_fixture, run_paused, StopFixture


@pytest.mark.parametrize('role',['listener','owner','settlement'])
def test_real_special_reader_holds_scene_fence_through_its_body(tmp_path,monkeypatch,role):
    access,_,member=access_fixture(tmp_path,monkeypatch)
    entered,finish,release=threading.Event(),threading.Event(),threading.Event()
    errors=[]
    def paused(*args,**kwargs):
        entered.set()
        assert release.wait(3)
        raise StopFixture()
    if role=='listener':
        from blueprint_pipeline import pubsub_handoff_listener as module
        monkeypatch.setattr(module,'parse_handoff_payload',paused)
        def operation():
            return module.process_handoff_payload({},storage_root=member,provider='fixture')
    elif role=='owner':
        from blueprint_pipeline import task_evaluation_scene_owner_authority as module
        root=tmp_path/'owners'
        (root/'scene-1').mkdir(parents=True)
        monkeypatch.setenv('BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT',str(root))
        monkeypatch.setattr(module,'checked_file',paused)
        def operation():
            return module.reopen_scene_intent({'path':str(root/'scene-1'/'intent.json')},now=100)
    else:
        from blueprint_pipeline import task_evaluation_terminal_scene_attempt_settlement as module
        monkeypatch.setattr(module,'_derive_settled_spend',paused)
        def operation():
            return module.retained_hold({'maximum_spend_usd':1})
    worker=threading.Thread(target=run_paused,args=(operation,entered,finish,errors))
    worker.start()
    try:
        assert entered.wait(3),errors
        with pytest.raises(access.SceneRetirementAccessError,match='scene_retirement_reader_active'):
            with access.exclusive_scene_access():
                pytest.fail('entered retirement while actual reader remained active')
    finally:
        release.set()
        worker.join(4)
    assert errors==[] and not worker.is_alive() and finish.is_set()
