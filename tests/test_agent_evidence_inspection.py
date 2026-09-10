"""Verify source pixels, source identity, disclosure scope and derived crops."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json

from PIL import Image
import pytest

from blueprint_pipeline.agent_execution.contracts import AgentExecutionError
from blueprint_pipeline.agent_execution.evidence import ImageEvidence, ImageEvidenceCatalog


def catalog(tmp_path):
    path = tmp_path / "frame.png"
    image = Image.new("RGB", (8, 6), (40, 50, 60))
    image.putpixel((3, 2), (255, 0, 0))
    image.save(path)
    sha = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    record = ImageEvidence("frame_1", path, sha, "wrist", "policy_input", 1.2, 12)
    return ImageEvidenceCatalog(root=tmp_path, images=[record], admitted_digests=frozenset({sha})), path


def test_full_frame_returns_the_exact_source_bytes_without_a_path(tmp_path):
    evidence, path = catalog(tmp_path)
    content = evidence.inspect("frame_1")
    metadata = json.loads(content[0]["text"])
    assert base64.b64decode(content[1]["image_url"].split(",", 1)[1]) == path.read_bytes()
    assert metadata["view_sha256"] == metadata["source_sha256"]
    assert metadata["new_source_observation"] is False
    assert str(tmp_path) not in content[0]["text"]


def test_crop_preserves_pixels_and_binds_to_original_frame(tmp_path):
    evidence, _ = catalog(tmp_path)
    content = evidence.inspect("frame_1", crop=[2, 1, 5, 4])
    metadata = json.loads(content[0]["text"])
    pixels = Image.open(BytesIO(base64.b64decode(content[1]["image_url"].split(",", 1)[1])))
    assert pixels.size == (3, 3)
    assert pixels.getpixel((1, 1)) == (255, 0, 0)
    assert metadata["derived_crop"] is True
    assert metadata["view_sha256"] != metadata["evidence_reference_digest"]
    assert metadata["pixel_rectangle"] == [2, 1, 5, 4]
    assert metadata["source_rectangle_pixels_sha256"] == metadata["view_pixels_sha256"]


@pytest.mark.parametrize("crop", [[0, 0, 99, 5], [3, 2, 1, 4], [True, 0, 4, 4], [1, 2]])
def test_invalid_crop_is_refused(tmp_path, crop):
    evidence, _ = catalog(tmp_path)
    with pytest.raises(AgentExecutionError, match="crop_invalid"):
        evidence.inspect("frame_1", crop=crop)


def test_changed_source_and_model_supplied_path_are_refused(tmp_path):
    evidence, path = catalog(tmp_path)
    with pytest.raises(AgentExecutionError, match="not_in_inventory"):
        evidence.inspect("../../secret")
    path.write_bytes(b"changed")
    with pytest.raises(AgentExecutionError, match="changed_before_disclosure"):
        evidence.inspect("frame_1")


def test_disclosure_not_granted_by_having_a_local_file(tmp_path):
    evidence, _ = catalog(tmp_path)
    with pytest.raises(AgentExecutionError, match="disclosure_not_admitted"):
        ImageEvidenceCatalog(root=tmp_path, images=list(evidence.images.values()),
                             admitted_digests=frozenset())


def test_symlink_is_not_a_new_admitted_artifact(tmp_path):
    evidence, path = catalog(tmp_path)
    link = tmp_path / "link.png"
    link.symlink_to(path)
    original = evidence.images["frame_1"]
    with pytest.raises(AgentExecutionError, match="symlink"):
        ImageEvidenceCatalog(root=tmp_path, images=[ImageEvidence(
            "linked", link, original.sha256, "wrist", "policy_input",
        )], admitted_digests=frozenset({original.sha256}))


@pytest.mark.parametrize("kind", ["orientation", "animation", "palette"])
def test_unqualified_pixel_interpretation_is_refused(tmp_path, kind):
    path = tmp_path / "unqualified.png"
    image = Image.new("RGB", (8, 6), (30, 50, 80))
    if kind == "orientation":
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif)
    elif kind == "animation":
        image.save(path, save_all=True, append_images=[Image.new("RGB", (8, 6), "red")])
    else:
        image.convert("P").save(path)
    sha = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = ImageEvidenceCatalog(root=tmp_path, images=[
        ImageEvidence("unqualified", path, sha, "wrist", "policy_input"),
    ], admitted_digests=frozenset({sha}))
    with pytest.raises(AgentExecutionError, match="pixel_interpretation_not_qualified"):
        evidence.inspect("unqualified", crop=[0, 0, 4, 3])


def test_crop_preserves_source_color_metadata(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    import struct

    path = tmp_path / "color.png"
    image = Image.new("RGB", (8, 6), (30, 50, 80))
    info = PngInfo()
    info.add(b"gAMA", struct.pack(">I", 45455))
    info.add(b"sRGB", b"\x00")
    image.save(path, pnginfo=info)
    sha = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = ImageEvidenceCatalog(root=tmp_path, images=[
        ImageEvidence("color", path, sha, "wrist", "policy_input"),
    ], admitted_digests=frozenset({sha}))
    content = evidence.inspect("color", crop=[0, 0, 4, 3])
    emitted = Image.open(BytesIO(base64.b64decode(content[1]["image_url"].split(",", 1)[1])))
    assert emitted.info["gamma"] == Image.open(path).info["gamma"]
    assert emitted.info["srgb"] == 0
