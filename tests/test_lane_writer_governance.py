"""A reviewed-source gate, not whole-program filesystem enforcement."""

# Covers (for impacted-test selection):
#   scripts/verify_lane_writer_governance.py
#   docs/architecture/lane-scratch-writer-manifest.json

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


WRITER = "src/blueprint_pipeline/writer.py"


def _manifest(tmp_path, source="from pathlib import Path\nPath('/mnt/blueprint-work/lanes/g1/a').mkdir()\n"):
    path = tmp_path / WRITER
    path.parent.mkdir(parents=True)
    path.write_text(source)
    test = tmp_path / "tests/test_writer.py"
    test.parent.mkdir()
    test.write_text("def test_writer(): pass\n")
    return {"schema_version": "lane_scratch_writer_manifest.v1", "sources": [{
        "path": WRITER, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "responsibility": "payload_consumer", "output_class": "lane_scratch",
        "admission_method": "LeasedScratchDirectory", "characterization_tests": ["tests/test_writer.py"],
        "review_reference": "independent fixture review", "historical_exceptions": [],
    }]}


def _validate(tmp_path, manifest):
    from scripts.verify_lane_writer_governance import validate_lane_writer_manifest

    return validate_lane_writer_manifest(root=tmp_path, manifest=manifest, required_sources=(WRITER,))


def test_valid_manifest_and_unchanged_writer_pass(tmp_path):
    assert _validate(tmp_path, _manifest(tmp_path)) == []


@pytest.mark.parametrize("replacement", [
    "from pathlib import Path\nPath('/mnt/blueprint-work/lanes/g1/b').mkdir()\n",
    "# changed shell launch/default\n",
])
def test_changed_whole_file_requires_new_review(tmp_path, replacement):
    manifest = _manifest(tmp_path)
    (tmp_path / WRITER).write_text(replacement)
    assert any("digest_changed" in blocker for blocker in _validate(tmp_path, manifest))


@pytest.mark.parametrize("change", ["missing", "duplicate", "test_missing", "review_missing", "empty_tests", "bad_hash"])
def test_manifest_fails_closed_for_missing_reviewed_identity(tmp_path, change):
    manifest = _manifest(tmp_path)
    entry = manifest["sources"][0]
    if change == "missing":
        manifest["sources"] = []
    elif change == "duplicate":
        manifest["sources"].append(dict(entry))
    elif change == "test_missing":
        (tmp_path / "tests/test_writer.py").unlink()
    elif change == "review_missing":
        entry["review_reference"] = ""
    elif change == "empty_tests":
        entry["characterization_tests"] = []
    else:
        entry["sha256"] = "not-a-digest"
    assert _validate(tmp_path, manifest)


@pytest.mark.parametrize("field", ["path", "characterization_tests"])
@pytest.mark.parametrize("unsafe", ["/tmp/outside.py", "../outside.py", "src/../outside.py", "src//writer.py", "."])
def test_manifest_source_and_test_paths_cannot_escape_repo(tmp_path, field, unsafe):
    manifest = _manifest(tmp_path)
    manifest["sources"][0][field] = [unsafe] if field == "characterization_tests" else unsafe
    assert any("path_unsafe" in blocker for blocker in _validate(tmp_path, manifest))


@pytest.mark.parametrize("field", ["path", "characterization_tests"])
@pytest.mark.parametrize("ancestor", [False, True])
def test_manifest_refuses_symlink_files_and_ancestors(tmp_path, field, ancestor):
    manifest = _manifest(tmp_path)
    relative = WRITER if field == "path" else "tests/test_writer.py"
    path = tmp_path / relative
    target = path.parent if ancestor else path
    saved = target.with_name(target.name + "-saved")
    target.rename(saved)
    target.symlink_to(saved, target_is_directory=ancestor)
    assert any("path_unsafe" in blocker for blocker in _validate(tmp_path, manifest))


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nPath('/mnt/blueprint-work/lanes/g1/new').mkdir()\n",
    "from pathlib import Path\nROOT = Path('/var/lib/blueprint/task-evaluation-inputs')\nOUT = ROOT / 'lanes' / 'arena' / 'new'\nOUT.mkdir()\n",
    "from pathlib import Path as P\nROOT: str = '/mnt/blueprint-work'\nOUT = P(ROOT + '/lanes/g1/new')\nOUT.mkdir()\n",
    "from pathlib import Path\nPath('/mnt/blueprint-work/' + 'lane' + 's/g1/new').mkdir()\n",
    "from pathlib import Path\nROOT = '/mnt/blueprint-work'\nPath(f'{ROOT}/lanes/g1/new').mkdir()\n",
    "from blueprint_pipeline.control_plane_lane_scratch import create_lane_scratch as make\nmake('g1','new')\n",
    "from .control_plane_lane_scratch import create_lane_scratch\nfactory = create_lane_scratch\nfactory('g1','new')\n",
    "import blueprint_pipeline.control_plane_lane_scratch as leases\nleases.create_lane_scratch('g1','new')\n",
    "import blueprint_pipeline.control_plane_lane_scratch\nblueprint_pipeline.control_plane_lane_scratch.create_lane_scratch('g1','new')\n",
    "from blueprint_pipeline import control_plane_lane_scratch as leases\nleases.create_lane_scratch('g1','new')\n",
    "from blueprint_pipeline.control_plane_leased_scratch import create_leased_lane_scratch as make\nmake('g1','new')\n",
])
def test_new_supported_python_lane_writer_requires_registration(tmp_path, source):
    manifest = _manifest(tmp_path)
    (tmp_path / "src/blueprint_pipeline/new_writer.py").write_text(source)
    assert "lane_writer_unregistered:src/blueprint_pipeline/new_writer.py" in _validate(tmp_path, manifest)


@pytest.mark.parametrize("source", [
    "mkdir -p /mnt/blueprint-work/lanes/g1/new\n",
    "mkdir -p \"${WORK_VOLUME_ROOT}/lanes/g1/new\"\n",
    "mkdir -p $WORK_VOLUME_ROOT/lanes/g1/new\n",
    "install -d \"${TASK_EVALUATION_INPUT_ROOT}/lanes/arena\"\n",
    "install -d $TASK_EVALUATION_INPUT_ROOT/lanes/arena\n",
    "mkdir -p /var/lib/blueprint/task-evaluation-inputs/lanes/arena/new\n",
    "python -m blueprint_pipeline.control_plane_arena_scratch prepare --tag r33\n",
])
def test_new_supported_shell_lane_writer_requires_registration(tmp_path, source):
    manifest = _manifest(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/new_writer.sh").write_text(source)
    assert "lane_writer_unregistered:scripts/new_writer.sh" in _validate(tmp_path, manifest)


@pytest.mark.parametrize("source", [
    "from pathlib import Path\nprint(Path('/mnt/blueprint-work/lanes/g1/new'))\n",
    "from pathlib import Path\nPath(args.output).mkdir()\n",
    "getattr(module, 'create_' + 'lane_scratch')('g1','new')\n",
    "from elsewhere import OUTPUT\nOUTPUT.mkdir()\n",
])
def test_read_only_reference_and_documented_dynamic_noncoverage(tmp_path, source):
    manifest = _manifest(tmp_path)
    (tmp_path / "src/blueprint_pipeline/unresolved.py").write_text(source)
    assert _validate(tmp_path, manifest) == []


def test_unreadable_manifest_source_fails_closed(tmp_path, monkeypatch):
    from scripts import verify_lane_writer_governance as guard

    manifest = _manifest(tmp_path)
    original = guard._read_file

    def refuse(root, relative):
        if relative == WRITER:
            raise PermissionError("injected unreadable source")
        return original(root, relative)

    monkeypatch.setattr(guard, "_read_file", refuse)
    assert any("source_path_unsafe" in blocker for blocker in _validate(tmp_path, manifest))


def test_reviewed_lane_writer_manifest_is_satisfied():
    from scripts.verify_lane_writer_governance import validate_lane_writer_manifest

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/architecture/lane-scratch-writer-manifest.json").read_text())
    assert validate_lane_writer_manifest(root=root, manifest=manifest) == []


def test_reviewed_lane_writer_guard_is_an_always_on_sentinel():
    from blueprint_pipeline.impacted_test_selection import SENTINEL_TESTS

    assert "tests/test_lane_writer_governance.py::test_reviewed_lane_writer_manifest_is_satisfied" in SENTINEL_TESTS
