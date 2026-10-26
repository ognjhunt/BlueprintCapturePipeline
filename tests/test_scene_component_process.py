from __future__ import annotations

import subprocess
import sys
import time

import pytest

from blueprint_pipeline.scene_component_process import run_component_process
from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
    STAGE_DEADLINE_EPOCH_ENV,
    artifixer_training_timeout_seconds,
)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "bad", ""])
def test_invalid_stage_deadline_refuses_training(raw):
    with pytest.raises(ValueError, match="stage_deadline_invalid"):
        artifixer_training_timeout_seconds({STAGE_DEADLINE_EPOCH_ENV: raw}, now_epoch=1_000)


@pytest.mark.parametrize("deadline", ["1000", "1599", "1600"])
def test_exhausted_stage_retains_closeout_time(deadline):
    with pytest.raises(ValueError, match="training_time_exhausted"):
        artifixer_training_timeout_seconds({STAGE_DEADLINE_EPOCH_ENV: deadline}, now_epoch=1_000)


def test_legacy_caller_keeps_its_original_bound():
    assert artifixer_training_timeout_seconds({}, now_epoch=1_000) == 7_000


@pytest.mark.slow
def test_component_timeout_closes_background_children_and_retains_output(tmp_path):
    marker = tmp_path / "orphan-finished"
    # The child must outlive the 2 s bound asserted below: with a shorter delay a
    # slow runner returning inside that bound let the child write before the kill.
    child = "import pathlib,time;time.sleep(3);pathlib.Path(%r).write_text('orphan')" % str(marker)
    wrapper = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c',%r]);"
        "print('started',flush=True);time.sleep(20)" % child
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        run_component_process([sys.executable, "-c", wrapper], timeout=.3, text=True)
    assert time.monotonic() - started < 2
    assert "started" in failure.value.stdout
    # The child started before "started" printed, so a survivor writes within 3 s of now.
    time.sleep(3.2)
    assert not marker.exists()


@pytest.mark.slow
def test_component_process_preserves_exit_and_streams():
    result = run_component_process(
        [sys.executable, "-c", "import sys;print('out');print('err',file=sys.stderr);sys.exit(3)"],
        timeout=2, text=True,
    )
    assert (result.returncode, result.stdout, result.stderr) == (3, "out\n", "err\n")
