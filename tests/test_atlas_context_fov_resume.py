"""Reuse an exact completed FOV check without accepting changed source or returned pixels."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import atlas_context_fov as fov
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.website_atlas_pose import ATLAS_POSE_ENDPOINT


@pytest.fixture
def audit(tmp_path):
    frames, sources = [], []
    for index in range(2):
        rgb = np.full((48, 32, 3), 80 + 20 * index, np.uint8)
        image = tmp_path / f"source-{index}.png"
        Image.fromarray(rgb).save(image)
        sources.append({"frame_id": str(index), "image_path": str(image), "image_digest": _sha256_file(image)})
        frames.append({"frame_id": str(index), "rgb": rgb})
    inputs = {"schema_version": "website_atlas_pose_inputs.v1", "status": "pending",
              "endpoint": ATLAS_POSE_ENDPOINT, "heldout_pixels_included": False,
              "claim_ceiling": "development_only", "metric_measurement_proven": False,
              "frames": sources}
    inputs["digest"] = canonical_digest(inputs, digest_field="digest")
    (tmp_path / "atlas_pose_inputs.json").write_text(json.dumps(inputs))
    result = fov.verify_pose_context_fov(pose_root=tmp_path, frames=frames, output_root=tmp_path)
    assert result["status"] == "passed"
    return tmp_path, frames, inputs, result


def test_exact_passed_audit_reuses_pixels_without_matching_or_writes(audit, monkeypatch):
    root, frames, _, expected = audit
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("completed audit recomputed or rewritten")

    monkeypatch.setattr(fov, "image_coverage", forbidden)
    monkeypatch.setattr(fov, "write_json", forbidden)
    assert fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root) == expected
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    assert not (root / "context-fov-audit-history").exists()


def test_canonical_loader_still_refuses_changed_source_bytes(audit, monkeypatch):
    root, frames, _, _ = audit
    before = (root / "context-fov-audit.json").read_bytes()
    (root / "source-0.png").write_bytes(b"changed")
    monkeypatch.setattr(fov, "image_coverage", lambda *_: pytest.fail("changed source reached matching"))
    with pytest.raises(ValueError, match="website_atlas_pose_input_changed"):
        fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root)
    assert (root / "context-fov-audit.json").read_bytes() == before


@pytest.mark.parametrize("changed", ["pixels", "grid", "frame_order"])
def test_changed_returned_pixels_or_grid_cannot_inherit_a_pass(audit, changed):
    root, frames, _, _ = audit
    previous = (root / "context-fov-audit.json").read_bytes()
    if changed == "pixels":
        frames[0]["rgb"] = np.full_like(frames[0]["rgb"], 255)
    elif changed == "grid":
        frames[0]["rgb"] = np.zeros((24, 32, 3), np.uint8)
    else:
        frames.reverse()
    with pytest.raises(ValueError, match="atlas_context_field_of_view_lost"):
        fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root)
    report = json.loads((root / "context-fov-audit.json").read_text())
    assert report["status"] == "refused"
    assert (root / "context-fov-audit-history" / (hashlib.sha256(previous).hexdigest() + ".json")).read_bytes() == previous


@pytest.mark.parametrize("changed", ["threshold", "dependency", "implementation"])
def test_changed_method_or_library_requires_recomputation(audit, monkeypatch, changed):
    import cv2

    root, frames, _, old = audit
    calls, original = [], fov.image_coverage
    monkeypatch.setattr(fov, "image_coverage", lambda *args: (calls.append(1), original(*args))[1])
    if changed == "threshold":
        monkeypatch.setattr(fov, "MIN_SOURCE_AREA_RETAINED", .995)
    elif changed == "dependency":
        monkeypatch.setattr(cv2, "__version__", cv2.__version__ + "-changed")
    else:
        implementation = root / "changed-module.py"
        implementation.write_bytes(Path(fov.__file__).read_bytes() + b"\n")
        monkeypatch.setattr(fov, "__file__", str(implementation))
    result = fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root)
    assert calls == [1, 1] and result["status"] == "passed"
    assert result["input_binding"] != old["input_binding"]


def test_legacy_pass_is_recomputed_once_and_its_original_bytes_are_preserved(audit, monkeypatch):
    root, frames, _, legacy = audit
    legacy.pop("input_binding")
    legacy["digest"] = canonical_digest(legacy, digest_field="digest")
    path = root / "context-fov-audit.json"
    previous = json.dumps(legacy).encode()
    path.write_bytes(previous)
    calls, original = [], fov.image_coverage
    monkeypatch.setattr(fov, "image_coverage", lambda *args: (calls.append(1), original(*args))[1])
    result = fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root)
    assert result["status"] == "passed" and calls == [1, 1]
    assert result["input_binding"]
    assert (root / "context-fov-audit-history" / (hashlib.sha256(previous).hexdigest() + ".json")).read_bytes() == previous


def test_corrupt_cached_audit_is_preserved_and_refused(audit, monkeypatch):
    root, frames, _, report = audit
    report["frames"][0]["retained_source_area_fraction"] = .1
    path = root / "context-fov-audit.json"
    bad = json.dumps(report).encode()
    path.write_bytes(bad)
    monkeypatch.setattr(fov, "image_coverage", lambda *_: pytest.fail("corrupt audit reached matching"))
    with pytest.raises(ValueError, match="atlas_context_audit_invalid"):
        fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root)
    assert path.read_bytes() == bad


def test_self_hashed_bad_coverage_is_not_promoted_by_the_cache(audit, monkeypatch):
    root, frames, _, report = audit
    report["frames"][0]["retained_source_area_fraction"] = .1
    report["digest"] = canonical_digest(report, digest_field="digest")
    (root / "context-fov-audit.json").write_text(json.dumps(report))
    monkeypatch.setattr(fov, "image_coverage", lambda *_: pytest.fail("bad cached coverage reached matching"))
    with pytest.raises(ValueError, match="atlas_context_field_of_view_lost"):
        fov.verify_pose_context_fov(pose_root=root, frames=frames, output_root=root)
    assert json.loads((root / "context-fov-audit.json").read_text())["status"] == "refused"
