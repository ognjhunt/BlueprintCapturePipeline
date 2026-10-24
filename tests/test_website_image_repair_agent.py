import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from blueprint_pipeline import website_image_completion as completion
from blueprint_pipeline import website_image_repair_agent as repair
from blueprint_pipeline import website_task_context as control
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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
        mask = tmp_path / f"mask-{index}.png"
        pixels = Image.new("L", (40, 20))
        for x in range(5):
            pixels.putpixel((x, 0), 255)
        pixels.save(mask)
        frame_id = f"f{index}"
        sources.append({"frame_id": frame_id, **original, "original_image_path": original["image_path"],
                        "original_image_digest": original["image_digest"], "remaining_pixel_count": 5,
                        "remaining_mask_path": str(mask), "remaining_mask_digest": _sha256_file(mask)})
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


class _AuditedInvoker(_Invoker):
    def configure_reservation_audit(self, **kwargs):
        self.audit = kwargs

    def invoke(self, spec, input_value):
        reservation = {"schema_version": "task_evaluation_inference_reservation.v1", "run_id": spec.run_id,
            "capability": repair.CAPABILITY, "model": spec.model,
            "input_digest": canonical_digest({"input": input_value}), "max_turns": 1,
            "max_output_tokens": spec.max_output_tokens, "projected_max_cost_usd": .5,
            "cache_policy_digest": "policy", "cache_policy": {"policy_digest": "policy"}, "breakpoint_digests": []}
        reservation["reservation_id"] = canonical_digest({k: reservation[k] for k in (
            "run_id", "capability", "model", "input_digest", "max_turns", "max_output_tokens", "cache_policy_digest")})
        reservation["inference_reservation_digest"] = canonical_digest(reservation, digest_field="inference_reservation_digest")
        self.audit["record_reservation"](reservation)
        result = super().invoke(spec, input_value)
        completion = {"schema_version": "task_evaluation_inference_completion.v1",
            **{k: reservation[k] for k in ("reservation_id", "run_id", "capability", "model", "cache_policy", "breakpoint_digests")},
            "provider": "openai", "usage": {"provider_response_id": "response-fixture"},
            "structured_output_digest": canonical_digest(result.output), "projected_max_cost_usd": .5,
            "reconciled_actual_cost_usd": .04, "released_reservation_usd": .46}
        completion["inference_completion_digest"] = canonical_digest(completion, digest_field="inference_completion_digest")
        self.audit["record_completion"](completion)
        return result


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


def test_completed_removal_marker_reedits_original_through_real_completion_and_reuses_receipts(tmp_path, monkeypatch):
    from hashlib import sha256
    from io import BytesIO
    from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission
    from blueprint_pipeline.semantic_teacher_image_edit_worker import _normalized_usage

    selected, sources = _views(tmp_path)
    # Exact retained-data failure: the edited image is finished, but a later
    # review requests a new edit of its original using the still nonempty mask.
    sources[0]["remaining_pixel_count"] = 0
    sources[0].update(image_path=selected[0]["image_path"], image_digest=selected[0]["image_digest"])
    original_mask = Path(sources[0]["remaining_mask_path"]).read_bytes()
    invoker = _Invoker({"repairs": [{"frame_id": "f0", "instruction": "Preserve the original camera"}],
                        "summary": "The reused edit changed viewpoint"})
    reservations, edits, reviews = [], [], []

    def reserve(**kwargs):
        reservations.append(kwargs)
        admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
            "resource_class": kwargs["resource_class"], "external_disclosure_allowed": True,
            "allocation_binding_digest": kwargs["binding_digest"], "maximum_cost_usd": kwargs["maximum_cost_usd"]}
        grant = require_paid_resource_admission(admission, resource_class=kwargs["resource_class"],
                                                expected_schema_version="paid_lane_admission.v1")
        return admission, grant

    def edit(**kwargs):
        edits.append(kwargs)
        stream = BytesIO()
        Image.new("RGB", kwargs["expected_size"], "blue").save(stream, format="PNG")
        return {"succeeded": True, "generated": stream.getvalue(),
                "usage": _normalized_usage({"output_tokens_details": {"image_tokens": 100}})}

    def review(**kwargs):
        reviews.append(kwargs)
        image = Image.open(kwargs["frames"][0]["image_path"])
        assert image.getpixel((20, 10)) == (0, 0, 255)
        assert kwargs["frames"][0]["image_digest"] != sources[0]["original_image_digest"]
        return {"status": "passed"}

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserve)
    monkeypatch.setattr(control, "website_webapp_request", lambda **kw:
                        {**kw["payload"]["settlement"], "status": "settled"})
    monkeypatch.setattr(completion, "_execute_frame_request", edit)
    monkeypatch.setattr(completion, "verify_completed_background", review)
    task_digest = sha256(json.dumps(CONTEXT, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args = dict(selected=selected, object_removal_frames=sources, original_frames=sources,
        plan={**PLAN, "task_context_sha256": task_digest}, failed_review=FAILED,
        output_root=tmp_path / "repair", task_context=CONTEXT, invoker=invoker)
    frames, reviewed = repair.repair_rejected_views(**args)
    assert reviewed["status"] == "passed" and len(edits) == 1
    assert frames[1:] == selected[1:] and sources[0]["remaining_pixel_count"] == 0
    assert Path(sources[0]["remaining_mask_path"]).read_bytes() == original_mask
    assert len(reservations) == 2  # One plan and one image, both canonical.
    assert repair.repair_rejected_views(**args)[0] == frames
    assert len(edits) == len(invoker.calls) == 1 and len(reservations) == 2 and len(reviews) == 2


@pytest.mark.parametrize("fault", ["digest", "grid", "nonbinary"])
def test_invalid_reused_source_mask_refuses_before_repair_plan(tmp_path, website, fault):
    selected, sources = _views(tmp_path)
    mask = Path(sources[0]["remaining_mask_path"])
    if fault == "digest":
        sources[0]["remaining_mask_digest"] = "changed"
    else:
        Image.new("L", (20, 40) if fault == "grid" else (40, 20), 255 if fault == "grid" else 127).save(mask)
        sources[0]["remaining_mask_digest"] = _sha256_file(mask)
    invoker = _Invoker({"repairs": [], "summary": ""})
    with pytest.raises(ValueError, match="website_image_repair_(source_changed|mask_invalid)"):
        repair.repair_rejected_views(selected=selected, object_removal_frames=sources, original_frames=sources,
            plan=PLAN, failed_review=FAILED, output_root=tmp_path / "repair", task_context=CONTEXT, invoker=invoker)
    assert not invoker.calls and not website["reserve"] and not website["edit"]


def test_atlas_repairs_use_scene_size_same_source_cap_and_fixed_clean_reference(tmp_path, website):
    selected, sources = _views(tmp_path)
    selected.append({**selected[-1], "frame_id": "f3"})
    sources.append({**sources[-1], "frame_id": "f3"})
    reference = Path(selected[2]["image_path"])
    invoker = _Invoker({"repairs": [{"frame_id": f"f{i}", "instruction": "Keep the bay open"}
                                    for i in range(4)], "summary": "four inconsistent views"})
    frames, reviewed = repair.repair_rejected_views(
        selected=selected, object_removal_frames=sources, original_frames=sources,
        plan=PLAN, failed_review=FAILED, output_root=tmp_path / "repair",
        task_context=CONTEXT, invoker=invoker, max_repairs=len(selected),
        source_video_digest="sha256:video", repair_reference_path=reference)
    assert len(frames) == 4 and reviewed["status"] == "passed"
    assert website["reserve"][0]["source_video_digest"] == "sha256:video"
    assert all(call["source_video_digest"] == "sha256:video" for call in website["edit"])
    assert all(call["repair_reference_path"] == reference for call in website["edit"])
    assert website["review"][0]["source_video_digest"] == "sha256:video"


def test_uncertain_plan_is_never_bought_again(tmp_path, website):
    invoker = _Invoker({"repairs": [], "summary": ""})
    _run(tmp_path, invoker)
    receipt_path = next((tmp_path / "image_completion" / "repair_agent").glob("plan-*[0-9a-f].json"))
    receipt = json.loads(receipt_path.read_text())
    receipt["status"] = "submitting"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="requires_reconciliation"):
        _run(tmp_path, invoker)
    assert len(invoker.calls) == 1


def test_different_reviews_keep_independent_inference_audits(tmp_path, website):
    invoker = _AuditedInvoker({"repairs": [], "summary": "nothing fixable"})
    _run(tmp_path, invoker)
    _run(tmp_path, invoker, failed_review={**FAILED, "review": {**FAILED["review"], "reason": "different failure"}})
    manifests = list((tmp_path / "image_completion/repair_agent").glob("inference-*/inference_reservations/manifest.json"))
    assert len(manifests) == len(invoker.calls) == 2
    assert len({json.loads(p.read_text())["run_id"] for p in manifests}) == 2
    assert all(json.loads(p.read_text())["in_flight_unknown_count"] == 0 for p in manifests)


def test_completed_plan_is_saved_before_audit_finalization(tmp_path, website, monkeypatch):
    from blueprint_pipeline.task_evaluation_supervisor.inference_reservations import InferenceReservationAudit

    def fail(_self):
        raise ValueError("audit_write_fixture")
    monkeypatch.setattr(InferenceReservationAudit, "write_manifest", fail)
    invoker = _AuditedInvoker({"repairs": [], "summary": "nothing fixable"})
    with pytest.raises(ValueError, match="audit_write_fixture"):
        _run(tmp_path, invoker)
    (path,) = (tmp_path / "image_completion/repair_agent").glob("plan-*[0-9a-f].json")
    assert json.loads(path.read_text())["status"] == "completed"
    _run(tmp_path, invoker)
    assert len(invoker.calls) == len(website["reserve"]) == 1


@pytest.mark.parametrize("fault", [None, "missing-completion", "changed-completion"])
def test_lost_output_reconciles_only_completed_exact_input_and_preserves_its_charge(tmp_path, website, fault):
    invoker = _AuditedInvoker({"repairs": [], "summary": "nothing fixable"})
    _run(tmp_path, invoker)
    root = tmp_path / "image_completion/repair_agent"
    (path,) = root.glob("plan-*[0-9a-f].json")
    saved = json.loads(path.read_text())
    pending = {k: saved[k] for k in ("binding_digest", "binding", "admission")}
    pending["status"] = "submitting"
    path.write_text(json.dumps(pending))
    original = path.read_bytes()
    (root / f"{path.stem}.settlement.json").unlink()
    website["settle"].clear()
    # Recreate the legacy shared audit directory without altering its records.
    audit_parent = root / ("inference-" + saved["binding_digest"][7:])
    (audit_parent / "inference_reservations").rename(root / "inference_reservations")
    (completion_path,) = (root / "inference_reservations/completed").glob("*.json")
    if fault == "missing-completion":
        completion_path.unlink()
    elif fault == "changed-completion":
        value = json.loads(completion_path.read_text())
        value["structured_output_digest"] = "changed"
        completion_path.write_text(json.dumps(value))
    # The old call was charged and settled; repeat settlement is unnecessary.
    if fault:
        with pytest.raises(ValueError):
            _run(tmp_path, invoker)
        assert len(invoker.calls) == len(website["reserve"]) == 1
    else:
        _run(tmp_path, invoker)
        assert len(invoker.calls) == len(website["reserve"]) == 2
        assert all(s["provider_charge_amount_usd"] == .04 for s in website["settle"])
        (lost,) = root.glob("lost-output-*.json")
        assert json.loads(lost.read_text())["cost_usd"] == .04
        _run(tmp_path, invoker)
        assert len(invoker.calls) == len(website["reserve"]) == 2
    assert path.read_bytes() == original


def test_empty_plan_keeps_the_rejection_and_edits_nothing(tmp_path, website):
    _, (frames, review) = _run(tmp_path, _Invoker({"repairs": [], "summary": "nothing fixable"}))
    assert review["status"] == "blocked" and not website["edit"] and not website["review"]


def test_unchanged_repair_preserves_rejection_without_buying_the_same_review(tmp_path, website, monkeypatch):
    selected, sources = _views(tmp_path)
    monkeypatch.setattr(completion, "complete_background_images", lambda **kw:
                        [dict(next(f for f in selected if f["frame_id"] == kw["frames"][0]["frame_id"]))])
    invoker = _Invoker({"repairs": [{"frame_id": "f0", "instruction": "Preserve the original viewpoint"}],
                        "summary": "camera mismatch"})
    frames, reviewed = repair.repair_rejected_views(selected=selected, object_removal_frames=sources,
        original_frames=sources, plan=PLAN, failed_review=FAILED,
        output_root=tmp_path / "repair", task_context=CONTEXT, invoker=invoker)
    assert frames[0]["image_digest"] == selected[0]["image_digest"]
    assert reviewed["status"] == "blocked" and reviewed["request_digest"] == FAILED["request_digest"]
    assert not website["review"] and reviewed["repaired_frame_ids"] == ["f0"]


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
