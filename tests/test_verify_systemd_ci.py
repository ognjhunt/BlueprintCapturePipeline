from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.verify_systemd_ci import PIPELINE_PYTHON, stage_root, verify_root


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("systemd-analyze") is None,
    reason="requires native Linux systemd unit verifier",
)


@pytest.fixture
def staged(tmp_path):
    root = tmp_path / "root"
    units = stage_root(REPO_ROOT / "deploy/systemd", root, Path(sys.executable))
    return root, units


def test_all_repository_units_verify_and_meet_existing_security_threshold(staged):
    root, units = staged
    assert len(units) == len(list((REPO_ROOT / "deploy/systemd").glob("blueprint-*.service")))
    for unit in units:
        assert unit.read_bytes() == (REPO_ROOT / "deploy/systemd" / unit.name).read_bytes()
    verify_root(root, units)


def test_missing_staged_runtime_interpreter_still_fails(staged):
    root, units = staged
    (root / PIPELINE_PYTHON).unlink()
    with pytest.raises(subprocess.CalledProcessError):
        verify_root(root, units)


def test_typo_in_unit_executable_is_not_automatically_stubbed(staged):
    root, units = staged
    unit = root / "etc/systemd/system/blueprint-agent-stage-replay.service"
    original = unit.read_text()
    changed = original.replace(str(PIPELINE_PYTHON), "opt/blueprint/unknown-python")
    assert changed != original
    unit.write_text(changed)
    with pytest.raises(subprocess.CalledProcessError):
        verify_root(root, units)


def test_insecure_unit_still_fails_existing_security_threshold(staged):
    root, _ = staged
    unit = root / "etc/systemd/system/blueprint-insecure.service"
    unit.write_text("[Service]\nExecStart=/bin/bash -c true\n")
    with pytest.raises(subprocess.CalledProcessError):
        verify_root(root, [unit])


def test_the_batch_slice_is_staged_and_a_mistyped_reference_or_limit_fails(staged):
    """systemd-analyze accepts a Slice= that names no file and a limit it cannot parse."""
    root, units = staged
    installed = root / "etc/systemd/system"
    slice_unit = installed / "system-blueprintbatch.slice"
    assert slice_unit.read_bytes() == (REPO_ROOT / "deploy/systemd" / slice_unit.name).read_bytes()
    assert slice_unit not in units, "a slice runs nothing, so it has no security score"
    original = slice_unit.read_text()

    slice_unit.write_text(original.replace("MemoryMax=5.5G", "MemoryMax=plenty"))
    with pytest.raises(ValueError, match="slice_unit_invalid:system-blueprintbatch.slice"):
        verify_root(root, units)
    slice_unit.write_text(original.replace("MemoryMax=5.5G\n", ""))
    with pytest.raises(ValueError, match="slice_unit_unbounded:system-blueprintbatch.slice"):
        verify_root(root, units)
    slice_unit.write_text(original)

    member = installed / "blueprint-team-scene-packet.service"
    text = member.read_text()
    member.write_text(text.replace("Slice=system-blueprintbatch.slice", "Slice=system-blueprintbach.slice"))
    with pytest.raises(ValueError, match="referenced_unit_missing:blueprint-team-scene-packet.service"):
        verify_root(root, units)
    member.write_text(text)

    (installed / "blueprint-unit-failure@.service").unlink()
    with pytest.raises(ValueError, match="referenced_unit_missing:.*:blueprint-unit-failure@.service"):
        verify_root(root, [unit for unit in units if unit.name != "blueprint-unit-failure@.service"])


def test_missing_real_interpreter_cannot_be_replaced_with_a_stub(tmp_path):
    with pytest.raises(ValueError, match="interpreter_missing_or_not_executable"):
        stage_root(REPO_ROOT / "deploy/systemd", tmp_path / "root", tmp_path / "missing-python")
    assert not (tmp_path / "root").exists()


def test_empty_unit_directory_cannot_produce_success(tmp_path):
    with pytest.raises(ValueError, match="no_blueprint_service_units"):
        stage_root(tmp_path, tmp_path / "root", Path(sys.executable))
