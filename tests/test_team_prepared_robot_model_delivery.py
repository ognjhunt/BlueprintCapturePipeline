"""Closed USD dependencies survive relocation and cannot escape operator staging."""
from copy import deepcopy
import hashlib
import os

import pytest

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.team_robot_model_delivery import RobotModelDeliveryError, copy_prepared_closure


def setup_closure(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    (source / "textures").mkdir()
    (source / "robot.usda").write_bytes(b'#usda 1.0\ndef Xform "Robot" {}\n')
    (source / "textures" / "body.png").write_bytes(b"retained-texture")
    files = [{"path": name, "size_bytes": (source / name).stat().st_size,
              "sha256": "sha256:" + hashlib.sha256((source / name).read_bytes()).hexdigest()}
             for name in ("robot.usda", "textures/body.png")]
    closure = {"schema_version": "embodiment_asset_dependency_closure.v1", "model_path": "robot.usda", "files": files}
    closure["closure_digest"] = cross_runtime_canonical_digest(closure, digest_field="closure_digest")
    return source, closure


def test_full_relative_closure_survives_two_stage_relocation(tmp_path):
    source, closure = setup_closure(tmp_path)
    staged, bundled = tmp_path / "staged", tmp_path / "bundle"
    copy_prepared_closure(source=source, destination=staged, closure=closure)
    copy_prepared_closure(source=staged, destination=bundled, closure=closure)
    for row in closure["files"]:
        assert (bundled / row["path"]).read_bytes() == (source / row["path"]).read_bytes()
    with pytest.raises(RobotModelDeliveryError, match="copy_unavailable"):
        copy_prepared_closure(source=source, destination=bundled, closure=closure)


@pytest.mark.parametrize("mutation", ["escape", "absolute", "duplicate", "missing_root", "digest", "bound"])
def test_malformed_closure_refused_before_copy(tmp_path, mutation):
    source, closure = setup_closure(tmp_path)
    if mutation == "escape":
        closure["files"][0]["path"] = "../outside"
    elif mutation == "absolute":
        closure["files"][0]["path"] = "/outside"
    elif mutation == "duplicate":
        closure["files"].append(deepcopy(closure["files"][0]))
    elif mutation == "missing_root":
        closure["model_path"] = "missing.usda"
    elif mutation == "bound":
        closure["files"][0]["size_bytes"] = 2**31
    if mutation != "digest":
        closure["closure_digest"] = cross_runtime_canonical_digest(closure, digest_field="closure_digest")
    else:
        closure["closure_digest"] = "sha256:" + "0" * 64
    with pytest.raises(RobotModelDeliveryError, match="closure_invalid"):
        copy_prepared_closure(source=source, destination=tmp_path / "target", closure=closure)
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("mutation", ["changed", "symlink", "missing"])
def test_dependency_mutation_is_never_sealed(tmp_path, mutation):
    source, closure = setup_closure(tmp_path)
    texture = source / "textures" / "body.png"
    if mutation == "changed":
        texture.write_bytes(b"changed-texture!")
    else:
        texture.unlink()
        if mutation == "symlink":
            texture.symlink_to(source / "robot.usda")
    with pytest.raises(RobotModelDeliveryError):
        copy_prepared_closure(source=source, destination=tmp_path / "target", closure=closure)


def test_prepared_paths_are_operator_only():
    from blueprint_pipeline.team_candidate_quick10_registry import DERIVED_FIELDS, OPERATOR_FIELDS

    assert "prepared_robot_models" in OPERATOR_FIELDS
    assert "prepared_robot_models" not in DERIVED_FIELDS


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("level", ["root", "nested"])
def test_directory_swapped_before_leaf_open_stays_on_original_inode(tmp_path, monkeypatch, side, level):
    source, closure = setup_closure(tmp_path)
    destination = tmp_path / "target"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "body.png").write_bytes(b"outside-is-not-source")
    original_open = os.open
    swapped = False
    held = tmp_path / "held"

    def racing_open(path, flags, *args, **kwargs):
        nonlocal swapped
        writing = bool(flags & os.O_WRONLY)
        if path == "body.png" and writing == (side == "destination") and not swapped:
            selected = source if side == "source" else destination
            selected = selected if level == "root" else selected / "textures"
            selected.rename(held)
            selected.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("blueprint_pipeline.team_robot_model_delivery.os.open", racing_open)
    copy_prepared_closure(source=source, destination=destination, closure=closure)
    assert swapped
    assert (outside / "body.png").read_bytes() == b"outside-is-not-source"
    assert set(outside.iterdir()) == {outside / "body.png"}
    texture = (held / "textures" / "body.png" if level == "root" else held / "body.png")
    if side == "destination":
        assert texture.read_bytes() == b"retained-texture"
    else:
        assert (destination / "textures" / "body.png").read_bytes() == b"retained-texture"


@pytest.mark.parametrize("side", ["source", "destination"])
def test_directory_swapped_before_directory_open_is_refused(tmp_path, monkeypatch, side):
    source, closure = setup_closure(tmp_path)
    destination = tmp_path / "target"
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open
    directory_calls = 0
    swapped = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal directory_calls, swapped
        if path == "textures" and flags & os.O_DIRECTORY:
            directory_calls += 1
            if directory_calls == (1 if side == "source" else 2):
                selected = (source if side == "source" else destination) / "textures"
                selected.rename(tmp_path / "held")
                selected.symlink_to(outside, target_is_directory=True)
                swapped = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr("blueprint_pipeline.team_robot_model_delivery.os.open", racing_open)
    with pytest.raises(RobotModelDeliveryError, match="copy_unavailable"):
        copy_prepared_closure(source=source, destination=destination, closure=closure)
    assert swapped
    assert not list(outside.iterdir())


@pytest.mark.parametrize("side", ["source", "destination"])
@pytest.mark.parametrize("level", ["root", "nested"])
def test_existing_directory_symlinks_are_refused(tmp_path, side, level):
    source, closure = setup_closure(tmp_path)
    destination = tmp_path / "target"
    outside = tmp_path / "outside"
    outside.mkdir()
    selected = source if side == "source" else destination
    if side == "destination":
        selected.mkdir()
        (selected / "textures").mkdir()
    selected = selected if level == "root" else selected / "textures"
    selected.rename(tmp_path / "held")
    selected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RobotModelDeliveryError, match="copy_unavailable"):
        copy_prepared_closure(source=source, destination=destination, closure=closure)
    assert not list(outside.iterdir())
