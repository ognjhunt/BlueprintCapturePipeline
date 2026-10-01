"""A failed child cannot be retired or transformed into a successful summary."""
from pathlib import Path
import pytest


def test_child_failure_and_timeout_are_terminal_without_retirement(tmp_path):
    from scripts.control_plane_concurrency_orchestrator import child_failure
    class Process:
        pid=123
        def __init__(self,code):self.code=code
        def poll(self):return self.code
    assert child_failure([{'process':Process(1),'started':0,'scene_key':'one'}],now=1,
                         child_timeout=10,global_deadline=20)=='child_failed:one:1'
    assert child_failure([{'process':Process(None),'started':0,'scene_key':'one'}],now=11,
                         child_timeout=10,global_deadline=20)=='child_timeout:one'
    assert child_failure([{'process':Process(None),'started':0,'scene_key':'one'}],now=21,
                         child_timeout=30,global_deadline=20)=='global_timeout'


def test_live_acceptance_requires_kernel_confinement_before_creating_roots(tmp_path):
    from scripts.control_plane_concurrency_orchestrator import require_kernel_confinement
    with pytest.raises(ValueError,match='harness_kernel_confinement_required'):
        require_kernel_confinement(process_root=tmp_path/'unavailable-proc')
    assert list(tmp_path.iterdir())==[]
