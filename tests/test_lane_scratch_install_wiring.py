"""The deployment installer provisions both lane scratch roots before door use."""

from pathlib import Path


def test_installer_provisions_lane_roots_on_the_work_volume() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "install_live_pipeline_control_plane.sh")
    source = script.read_text(encoding="utf-8")
    assert 'WORK_VOLUME_ROOT="${WORK_VOLUME_ROOT:-/mnt/blueprint-work}"' in source
    assert '"${TASK_EVALUATION_INPUT_ROOT}/lanes"' in source
    assert 'mountpoint -q -- "${WORK_VOLUME_ROOT}"' in source
    assert '"${WORK_VOLUME_ROOT}/lanes"' in source
    assert 'install -d -m 0750 -o "${SERVICE_USER}" -g "${SERVICE_GROUP}"' in source
