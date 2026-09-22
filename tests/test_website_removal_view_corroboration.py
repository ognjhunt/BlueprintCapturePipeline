"""A tracked removal mask that has left the target must not reach the editor."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import website_removal_view_corroboration as corroboration
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file


def _runs(mask: np.ndarray) -> list[dict[str, int]]:
    edges = np.diff(np.pad(mask.reshape(-1).astype(np.int8), (1, 1)))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [{"start": int(start), "length": int(end - start)} for start, end in zip(starts, ends)]


def _observation(frame_id: str, rows: int, *, size: int = 40) -> dict:
    mask = np.zeros((size, size), dtype=bool)
    mask[:rows, :size // 2] = True
    return {"source_frame_id": frame_id, "width": size, "height": size, "runs": _runs(mask)}


def _scene(tmp_path, shares: dict[str, int]):
    frames = []
    for index, frame_id in enumerate(shares):
        path = tmp_path / f"{frame_id}.png"
        Image.new("RGB", (40, 40), (200, 190, 170)).save(path)
        frames.append({"frame_id": frame_id, "source_image_path": str(path),
                       "source_image_digest": _sha256_file(path),
                       "display_rotation_degrees": 0.0, "timestamp_seconds": index * 0.5})
    task_masks = {"targets": [{
        "target_id": "pedestal_cabinet", "task_effect": "manipulated", "disposition": "remove",
        "segmentation_prompt": "under-desk cabinet",
        "source_track": {"track_id": "t", "observations": [
            _observation(frame_id, rows) for frame_id, rows in shares.items()]},
    }]}
    return frames, task_masks


def test_a_view_whose_mask_has_left_the_target_is_dropped(tmp_path, monkeypatch):
    # Four views the tracker agrees on, one where it claims four times as much.
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 8, "c": 32, "d": 8, "e": 8})
    asked = []

    def hosted(**kwargs):
        asked.append(kwargs["frame_registry"][0]["source_frame_id"])
        # An independent look at that view finds only the real target.
        return {"tracks": [{"track_id": "independent", "label": "pedestal_cabinet",
                            "observations": [_observation("x", 8)]}]}

    monkeypatch.setattr(corroboration, "run_meta_sam31", hosted)
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")

    assert [frame["frame_id"] for frame in kept] == ["a", "b", "d", "e"]
    # Only the outlier was worth paying to adjudicate.
    assert asked == ["c"]
    assert receipt["status"] == "passed"
    assert receipt["refuted_frame_ids"] == ["c"]
    verdicts = {row["frame_id"]: row["verdict"] for row in receipt["observations"]}
    assert verdicts == {"a": "within_track_scale", "b": "within_track_scale", "c": "refuted",
                        "d": "within_track_scale", "e": "within_track_scale"}
    written = json.loads((tmp_path / "out" / "removal_view_corroboration.json").read_text())
    assert written["digest"] == receipt["digest"]


def test_an_outlier_the_second_look_agrees_with_is_kept(tmp_path, monkeypatch):
    """Apparent size grows when the camera comes closer; that is not drift."""
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 8, "c": 16, "d": 8, "e": 8})
    monkeypatch.setattr(corroboration, "run_meta_sam31", lambda **kwargs: {"tracks": [
        {"track_id": "independent", "label": "pedestal_cabinet",
         "observations": [_observation("x", 14)]}]})
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    assert [frame["frame_id"] for frame in kept] == ["a", "b", "c", "d", "e"]
    assert receipt["refuted_frame_ids"] == []
    assert [row for row in receipt["observations"] if row["frame_id"] == "c"][0]["verdict"] == "corroborated"


def test_finding_nothing_is_not_a_refutation(tmp_path, monkeypatch):
    """A target clipped by the frame edge can be real and still unnamed."""
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 8, "c": 32, "d": 8, "e": 8})
    monkeypatch.setattr(corroboration, "run_meta_sam31", lambda **kwargs: {"tracks": []})
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    assert len(kept) == 5
    assert [row for row in receipt["observations"]
            if row["frame_id"] == "c"][0]["verdict"] == "no_independent_instance"


def test_too_few_trustworthy_views_refuses_rather_than_reconstructing(tmp_path, monkeypatch):
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 32})
    monkeypatch.setattr(corroboration, "run_meta_sam31", lambda **kwargs: {"tracks": [
        {"track_id": "independent", "label": "pedestal_cabinet",
         "observations": [_observation("x", 8)]}]})
    with pytest.raises(ValueError, match="website_removal_views_uncorroborated"):
        corroboration.corroborate_removal_views(
            frames=frames, task_masks=task_masks, task_context={"confirmed": True},
            output_root=tmp_path / "out")
    written = json.loads((tmp_path / "out" / "removal_view_corroboration.json").read_text())
    assert written["status"] == "blocked" and written["retained_view_count"] == 1
    assert written["refuted_frame_ids"] == ["b"]


def test_a_track_that_drifted_on_most_views_is_left_to_the_background_review(tmp_path, monkeypatch):
    """The median is the drifted scale there, so the honest view is the outlier."""
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 32, "c": 32, "d": 32})
    monkeypatch.setattr(corroboration, "run_meta_sam31", lambda **kwargs: {"tracks": [
        {"track_id": "independent", "label": "pedestal_cabinet",
         "observations": [_observation("x", 8)]}]})
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    assert len(kept) == 4 and receipt["refuted_frame_ids"] == []
    assert {row["verdict"] for row in receipt["observations"]} == {"within_track_scale"}


def test_a_kept_target_is_never_second_guessed(tmp_path, monkeypatch):
    """Only an object being erased can erase the wrong thing."""
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 8, "c": 32})
    task_masks["targets"][0].update(task_effect="static_obstacle", disposition="keep")
    monkeypatch.setattr(corroboration, "run_meta_sam31",
                        lambda **kwargs: pytest.fail("a kept target must not be adjudicated"))
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    assert len(kept) == 3 and receipt["observations"] == []
