import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline import website_image_repair_agent as repair
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file

CONTEXT = {"context_digest": "sha256:" + "c" * 64, "capture_id": "cap", "request_id": "req", "scene_id": "scene"}


def _image(path: Path, color: str) -> dict:
    Image.new("RGB", (40, 20), color).save(path)
    return {"image_path": str(path), "image_digest": _sha256_file(path)}


def _views(tmp_path):
    selected, sources = [], []
    for index, color in enumerate(("red", "green", "blue")):
        original = _image(tmp_path / f"original-{index}.png", color)
        edited = _image(tmp_path / f"edited-{index}.png", "white")
        frame_id = f"f{index}"
        sources.append({"frame_id": frame_id, **original, "original_image_path": original["image_path"],
                        "original_image_digest": original["image_digest"], "remaining_pixel_count": 5})
        selected.append({"frame_id": frame_id, **edited, "original_image_path": original["image_path"],
                         "original_image_digest": original["image_digest"], "generated_pixels_present": True,
                         "generated_pixel_count": 100 * (index + 1), "remaining_pixel_count": 0})
    return selected, sources


class _Invoker:
    def __init__(self, output):
        self.output, self.calls = output, []

    def configure_reservation_audit(self, **_kwargs):
        pass

    def invoke(self, spec, input_value):
        self.calls.append((spec, input_value))
        return SimpleNamespace(output=self.output, model=repair.MODEL, usage={"input_tokens": 9000},
                               cost_usd=0.04)


@pytest.fixture
def website(monkeypatch):
    calls = {"reserve": [], "settle": [], "edit": [], "review": []}

    def reserve(**kwargs):
        calls["reserve"].append(kwargs)
        return {"status": "admitted", "allocation_binding_digest": kwargs["binding_digest"]}, object()

    def webapp(**kwargs):
        calls["settle"].append(kwargs["payload"]["settlement"])
        return {**kwargs["payload"]["settlement"], "status": "settled"}

    def edit(**kwargs):
        calls["edit"].append(kwargs)
        return [{**kwargs["frames"][0], "image_path": "repaired.png", "image_digest": "sha256:repaired",
                 "generated_pixels_present": True, "remaining_pixel_count": 0}]

    def review(**kwargs):
        calls["review"].append(kwargs)
        return {"status": "passed", "review": {"consistent_background": True}}

    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserve)
    monkeypatch.setattr(control, "website_webapp_request", webapp)
    monkeypatch.setattr(completion, "complete_background_images", edit)
    monkeypatch.setattr(completion, "verify_completed_background", review)
    return calls


FAILED = {"status": "blocked", "request_digest": "sha256:" + "a" * 64, "review": {
    "consistent_background": True, "task_objects_removed": True, "unrelated_objects_preserved": False,
    "reason": "paper towels in f0 were removed", "remaining_task_object_frame_ids": []}}
PLAN = {"targets": [{"target_id": "dishwasher", "task_effect": "manipulated", "disposition": "remove"}],
        "task_context_sha256": "task"}


def _run(tmp_path, invoker, **overrides):
    selected, sources = _views(tmp_path)
    args = dict(selected=selected, object_removal_frames=sources, original_frames=[], plan=PLAN,
                failed_review=FAILED, output_root=tmp_path / "image_completion", task_context=CONTEXT,
                invoker=invoker)
    args.update(overrides)
    return selected, repair.repair_rejected_views(**args)


def test_planned_repair_reedits_from_original_and_the_independent_review_decides(tmp_path, website):
    invoker = _Invoker({"repairs": [{"frame_id": "f0", "instruction": "Keep the paper towels on the island"}],
                        "summary": "towels erased in f0"})
    selected, (frames, review) = _run(tmp_path, invoker)
    # One agent call, sees each original and edited view, never approves.
    assert len(invoker.calls) == 1
    spec, input_value = invoker.calls[0]
    assert spec.max_turns == 1 and spec.output_type is repair.RepairPlan
    images = [part for part in input_value[0]["content"] if part["type"] == "input_image"]
    assert len(images) == 2 * len(selected)
    # Its own tokens are reserved against the scene cap, then settled to the charge.
    assert website["reserve"][0]["resource_class"] == "openai_api_candidate"
    assert website["reserve"][0]["maximum_cost_usd"] == repair.MAX_COST_USD
    assert website["settle"][0]["provider_charge_amount_usd"] == 0.04
    # The re-edit starts from the ORIGINAL through the paid lane, with the instruction
    # as data and the largest accepted edit as the fixed reference.
    (edit,) = website["edit"]
    assert edit["frames"][0]["image_path"].endswith("original-0.png")
    assert edit["repair_instruction"] == "Keep the paper towels on the island"
    assert Path(edit["repair_reference_path"]).name == "edited-2.png"
    assert [frame["frame_id"] for frame in frames] == ["f0", "f1", "f2"]
    assert frames[0]["image_path"] == "repaired.png" and frames[1] == selected[1]
    # Only the independent review's verdict counts.
    assert review["status"] == "passed" and review["repaired_frame_ids"] == ["f0"]
    assert review["prior_failed_review"] == FAILED
    receipt = json.loads(next((tmp_path / "image_completion" / "repair_agent").glob("plan-*[0-9a-f].json")).read_text())
    assert receipt["status"] == "completed" and receipt["approves_views"] is False


def test_restart_reuses_the_retained_plan_without_buying_again(tmp_path, website):
    invoker = _Invoker({"repairs": [{"frame_id": "f1", "instruction": "Keep the floor mat"}], "summary": "mat"})
    _run(tmp_path, invoker)
    _run(tmp_path, invoker)
    assert len(invoker.calls) == 1 and len(website["reserve"]) == 1 and len(website["settle"]) == 1


def test_uncertain_plan_is_never_bought_again(tmp_path, website):
    invoker = _Invoker({"repairs": [], "summary": ""})
    _run(tmp_path, invoker)
    receipt_path = next((tmp_path / "image_completion" / "repair_agent").glob("plan-*[0-9a-f].json"))
    receipt_path.write_text(json.dumps({"status": "submitting"}))
    with pytest.raises(ValueError, match="requires_reconciliation"):
        _run(tmp_path, invoker)
    assert len(invoker.calls) == 1


def test_empty_plan_keeps_the_rejection_and_edits_nothing(tmp_path, website):
    _, (frames, review) = _run(tmp_path, _Invoker({"repairs": [], "summary": "nothing fixable"}))
    assert review["status"] == "blocked" and not website["edit"] and not website["review"]


@pytest.mark.parametrize("repairs,error", [
    ([{"frame_id": f"f{i % 3}", "instruction": "x"} for i in range(4)], ValidationError),
    ([{"frame_id": "unknown", "instruction": "x"}], ValueError),
    ([{"frame_id": "f0", "instruction": "x"}, {"frame_id": "f0", "instruction": "y"}], ValueError),
])
def test_plan_beyond_limits_or_outside_the_set_is_refused(tmp_path, website, repairs, error):
    with pytest.raises(error):
        _run(tmp_path, _Invoker({"repairs": repairs, "summary": ""}))
    assert not website["edit"]


def test_repair_is_off_unless_explicitly_enabled(monkeypatch):
    monkeypatch.delenv(repair.ENABLE_ENV, raising=False)
    assert repair.image_repair_enabled() is False
    monkeypatch.setenv(repair.ENABLE_ENV, "1")
    assert repair.image_repair_enabled() is True


def test_repair_edit_binding_adds_to_the_rules_and_leaves_normal_edits_unchanged(tmp_path):
    frame = {"frame_id": "f0", "image_digest": "sha256:a", "remaining_mask_digest": "sha256:b"}
    normal = completion.completion_binding([frame], task_digest="t", backend_digest="b")
    repaired = completion.completion_binding([frame], task_digest="t", backend_digest="b",
                                             repair_instruction="Keep the mat", reference_digest="sha256:r")
    assert "repair_reference_digest" not in normal and normal["reference_policy"] == "edited_anchor_largest_removal"
    assert repaired["prompt"].startswith(normal["prompt"]) and '"Keep the mat"' in repaired["prompt"]
    assert repaired["reference_policy"] == "fixed_repair_reference"
    assert repaired["repair_reference_digest"] == "sha256:r"
