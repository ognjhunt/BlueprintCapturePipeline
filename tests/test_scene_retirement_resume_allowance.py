"""Native resumed work narrows the original private action, never resets it."""
import pytest
from blueprint_pipeline.task_evaluation_scene_retirement_preservation import ActionAllowance


def test_resume_keeps_original_deadline_and_charged_prefix_before_next_physical_work():
    clock=[10.0];wall=[200]
    first=ActionAllowance(expires_at=1000,now=lambda:wall[0],monotonic=lambda:clock[0],local_bytes=10,elapsed_seconds=20)
    first.charge('local_bytes',7)
    checkpoint=first.checkpoint()
    clock[0]=15;wall[0]=205
    resumed=ActionAllowance(expires_at=1000,now=lambda:wall[0],monotonic=lambda:clock[0],local_bytes=10,elapsed_seconds=20)
    resumed.bind_resume(checkpoint)
    assert resumed.counts['local_bytes']==7
    with pytest.raises(ValueError,match='scene_retirement_byte_limit'):
        resumed.charge('local_bytes',4)
    clock[0]=31;wall[0]=221
    expired=ActionAllowance(expires_at=1000,now=lambda:wall[0],monotonic=lambda:clock[0],local_bytes=10,elapsed_seconds=20)
    with pytest.raises(ValueError,match='scene_retirement_deadline'):
        expired.bind_resume(checkpoint)


@pytest.mark.parametrize('mutation',['future_origin','clock_backwards','bool_counts','different_limits','rebind','consumed'])
def test_resume_refuses_contradictory_or_reusable_native_origin(mutation):
    first=ActionAllowance(expires_at=1000,now=lambda:200,monotonic=lambda:10,local_bytes=10)
    checkpoint=first.checkpoint()
    resumed=ActionAllowance(expires_at=1000,now=lambda:205,monotonic=lambda:15,local_bytes=10)
    if mutation=='future_origin':checkpoint['start_monotonic']=100
    elif mutation=='clock_backwards':checkpoint['last_monotonic']=100
    elif mutation=='bool_counts':checkpoint['counts']['local_bytes']=True
    elif mutation=='different_limits':checkpoint['limits']['local_bytes']=100
    elif mutation=='rebind':resumed.bind_resume(checkpoint)
    elif mutation=='consumed':resumed.charge('local_bytes',1)
    with pytest.raises(ValueError):
        resumed.bind_resume(checkpoint)


def test_resume_wall_clock_failure_never_extends_origin_after_process_restart():
    first=ActionAllowance(expires_at=1000,now=lambda:200,monotonic=lambda:10,elapsed_seconds=20)
    checkpoint=first.checkpoint()
    resumed=ActionAllowance(expires_at=1000,now=lambda:230,monotonic=lambda:15,elapsed_seconds=20)
    with pytest.raises(ValueError,match='scene_retirement_deadline'):
        resumed.bind_resume(checkpoint)
