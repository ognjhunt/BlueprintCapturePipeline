"""Real child queue replay after process death, with a test-owned provider edge."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_execution as execution
from tests.test_task_evaluation_sam31_preparation_execution import setup as setup, _complete


class ProcessDeath(BaseException):
    pass


def test_lost_ack_restart_retains_one_create_and_resumes_exact_child(setup):
    root, args, process = setup
    queued = execution.enqueue_sam31_phase(**args)
    instance = root / "synthetic-started-instance.txt"
    calls = []
    def edge(context):
        calls.append((context["child_id"], context["resume_only"]))
        if not context["resume_only"]:
            instance.write_text("synthetic-instance-1")
            raise ProcessDeath("create acknowledgement lost before progress publication")
        assert instance.read_text() == "synthetic-instance-1"
        assert context["previous_progress"] is None
        return _complete(context)
    with pytest.raises(ProcessDeath):
        execution.process_sam31_phase_queue(**process, phase_executor=edge)
    queue = Path(args["queue_root"])
    name = queued["child_id"] + ".json"
    job_bytes = (queue / "processing" / name).read_bytes()
    assert (queue / "started" / name).is_file()
    assert not Path(queued["result_path"]).exists()
    resumed = execution.process_sam31_phase_queue(**process, phase_executor=edge)
    assert resumed["results"][0]["status"] == "completed"
    assert calls == [(queued["child_id"], False), (queued["child_id"], True)]
    assert (queue / "completed" / name).read_bytes() == job_bytes
    result_bytes = Path(queued["result_path"]).read_bytes()
    assert json.loads(result_bytes)["status"] == "completed"
    assert execution.process_sam31_phase_queue(**process,
        phase_executor=lambda _context: pytest.fail("completed child was relaunched"))["results"] == []
    assert Path(queued["result_path"]).read_bytes() == result_bytes


def test_two_workers_observe_one_child_and_execute_once(setup):
    from concurrent.futures import ThreadPoolExecutor
    _, args, process = setup
    queued = execution.enqueue_sam31_phase(**args)
    calls = []
    def run(_index):
        return execution.process_sam31_phase_queue(**process,
            phase_executor=lambda context: calls.append(context["child_id"]) or _complete(context))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, [0, 1]))
    assert calls == [queued["child_id"]]
    assert sum(len(result["results"]) for result in results) == 1
