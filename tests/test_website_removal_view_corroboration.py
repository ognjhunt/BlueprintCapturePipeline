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


def test_the_largest_views_are_adjudicated_until_one_is_corroborated(tmp_path, monkeypatch):
    """Drift only adds area, so the biggest claims are the ones worth buying."""
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 36, "c": 32, "d": 20, "e": 8})
    asked, independent = [], {"b": 8, "c": 8, "d": 18}

    def hosted(**kwargs):
        frame_id = kwargs["frame_registry"][0]["source_frame_id"]
        asked.append(frame_id)
        return {"tracks": [{"track_id": "independent", "label": "pedestal_cabinet",
                            "observations": [_observation("x", independent[frame_id])]}]}

    monkeypatch.setattr(corroboration, "run_meta_sam31", hosted)
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")

    # Biggest first, and the moment one agrees the smaller ones are accepted.
    assert asked == ["b", "c", "d"]
    assert [frame["frame_id"] for frame in kept] == ["a", "d", "e"]
    assert receipt["refuted_frame_ids"] == ["b", "c"]
    verdicts = {row["frame_id"]: row["verdict"] for row in receipt["observations"]}
    assert verdicts == {"b": "refuted", "c": "refuted", "d": "corroborated",
                        "a": "below_corroborated_view", "e": "below_corroborated_view"}
    written = json.loads((tmp_path / "out" / "removal_view_corroboration.json").read_text())
    assert written["digest"] == receipt["digest"]


def test_a_healthy_scene_costs_one_second_opinion(tmp_path, monkeypatch):
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 20, "c": 12, "d": 8})
    asked = []

    def hosted(**kwargs):
        asked.append(kwargs["frame_registry"][0]["source_frame_id"])
        return {"tracks": [{"track_id": "independent", "label": "pedestal_cabinet",
                            "observations": [_observation("x", 18)]}]}

    monkeypatch.setattr(corroboration, "run_meta_sam31", hosted)
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    assert asked == ["b"] and len(kept) == 4 and receipt["refuted_frame_ids"] == []


def test_finding_nothing_settles_nothing_and_the_next_view_is_still_checked(tmp_path, monkeypatch):
    """A target clipped by the frame edge can be real and still unnamed."""
    frames, task_masks = _scene(tmp_path, {"a": 8, "b": 36, "c": 32})
    asked, independent = [], {"b": None, "c": 8, "a": 8}

    def hosted(**kwargs):
        frame_id = kwargs["frame_registry"][0]["source_frame_id"]
        asked.append(frame_id)
        rows = independent[frame_id]
        return {"tracks": [] if rows is None else [
            {"track_id": "independent", "label": "pedestal_cabinet",
             "observations": [_observation("x", rows)]}]}

    monkeypatch.setattr(corroboration, "run_meta_sam31", hosted)
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    # Neither "nothing found" nor a refutation settles the question, so the
    # search keeps going until a view is actually corroborated.
    assert asked == ["b", "c", "a"]
    assert [frame["frame_id"] for frame in kept] == ["a", "b"]
    verdicts = {row["frame_id"]: row["verdict"] for row in receipt["observations"]}
    assert verdicts == {"b": "no_independent_instance", "c": "refuted", "a": "corroborated"}


def test_a_pathological_track_cannot_buy_an_unbounded_number_of_looks(tmp_path, monkeypatch):
    frames, task_masks = _scene(tmp_path, {name: 36 - index for index, name in enumerate("abcdef")})
    asked = []

    def hosted(**kwargs):
        asked.append(kwargs["frame_registry"][0]["source_frame_id"])
        return {"tracks": []}

    monkeypatch.setattr(corroboration, "run_meta_sam31", hosted)
    kept, receipt = corroboration.corroborate_removal_views(
        frames=frames, task_masks=task_masks, task_context={"confirmed": True},
        output_root=tmp_path / "out")
    assert len(asked) == corroboration.MAXIMUM_SECOND_OPINIONS
    assert len(kept) == 6
    assert [row["verdict"] for row in receipt["observations"]][-2:] == [
        "second_opinion_budget_spent", "second_opinion_budget_spent"]


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
