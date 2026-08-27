import subprocess
import sys
import time
from pathlib import Path

from blueprint_pipeline.task_evaluation_release_reference_lock import (
    release_reference_lock,
)


def test_exclusive_reaper_lock_blocks_a_reference_publisher(tmp_path: Path) -> None:
    marker = tmp_path / "publisher-acquired"
    source = """
import pathlib
import sys
from blueprint_pipeline.task_evaluation_release_reference_lock import release_reference_lock
with release_reference_lock(pathlib.Path(sys.argv[1]), exclusive=False):
    pathlib.Path(sys.argv[2]).write_text("acquired", encoding="utf-8")
"""
    with release_reference_lock(tmp_path, exclusive=True):
        process = subprocess.Popen(
            [sys.executable, "-c", source, str(tmp_path), str(marker)]
        )
        time.sleep(0.2)
        assert process.poll() is None
        assert not marker.exists()
    process.wait(timeout=5)
    assert process.returncode == 0
    assert marker.read_text(encoding="utf-8") == "acquired"


def test_queue_publication_waits_for_exclusive_reaper_lock(tmp_path: Path) -> None:
    destination = tmp_path / "queue" / "pending" / "record.json"
    destination.parent.mkdir(parents=True)
    source = """
import json
import pathlib
import sys
from blueprint_pipeline.task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive
write_launch_preparation_record_exclusive(
    pathlib.Path(sys.argv[1]), {"source_commit": "a" * 40}
)
"""
    with release_reference_lock(tmp_path, exclusive=True):
        process = subprocess.Popen(
            [sys.executable, "-c", source, str(destination)]
        )
        time.sleep(0.2)
        assert process.poll() is None
        assert not destination.exists()
    process.wait(timeout=5)
    assert process.returncode == 0
    assert destination.is_file()
