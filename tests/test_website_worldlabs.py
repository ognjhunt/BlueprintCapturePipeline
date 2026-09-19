import json
from hashlib import sha256

from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission



import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.provider_preview import WorldLabsPreviewProvider
from blueprint_pipeline import provider_preview
from blueprint_pipeline.website_worldlabs import submit_website_prepared_views


def _grant(admission):
    return require_paid_resource_admission(admission, resource_class=admission["resource_class"],
                                           expected_schema_version="paid_lane_admission.v1")


def _descriptor(tmp_path):
    task = {"description": "Move the blue box to the tray", "confirmed": True}
    frames = []
    (tmp_path / "pipeline").mkdir()
    for index in range(2):
        path = tmp_path / "pipeline" / f"prepared-{index}.png"
        Image.new("RGB", (14, 14), "white").save(path)
        frames.append({"image_path": str(path), "image_digest": _sha256_file(path)})
    preparation = {"status": "ready", "frames": frames,
                   "task_context_sha256": sha256(json.dumps(task, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    binding = {"prepared_views_digest": preparation["digest"], "model": "marble-1.1-plus",
               "scene_id": "site-test", "capture_id": "walkthrough-test"}
    return {"scene_id": binding["scene_id"], "capture_id": binding["capture_id"],
            "metadata": {"capture_entry_source": "browser_self_capture", "site_task_context": task,
                         "clean_plate": {"status": "objects_removed", "privacy_verified": True, "prepared_views": preparation},
                         "website_reconstruction_admission": {"schema_version": "paid_lane_admission.v1", "status": "admitted",
                                                              "resource_class": "provider_reconstruction_api", "blockers": [],
                                                              "maximum_cost_usd": 2.48, "external_disclosure_allowed": True,
                                                              "allocation_binding_digest": canonical_digest(binding)}}}


def test_website_submits_only_prepared_images_and_reuses_the_operation(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    requests, uploads = [], []

    def api(path, **kwargs):
        requests.append((path, kwargs))
        if path.endswith("prepare_upload"):
            return {"media_asset": {"media_asset_id": f"asset-{len(requests)}"},
                    "upload_info": {"upload_url": "https://upload.example/image"}}
        assert (tmp_path / "pipeline/website_reconstruction/submission.json").is_file()
        prompt = kwargs["body"]["world_prompt"]
        assert prompt["type"] == "multi-image"
        assert prompt["reconstruct_images"] is True
        assert "video_prompt" not in prompt
        assert all("azimuth" not in frame for frame in prompt["multi_image_prompt"])
        return {"operation_id": "operation-1", "done": False}

    monkeypatch.setattr(provider_preview, "_worldlabs_api_request", api)
    monkeypatch.setattr(provider_preview, "_presigned_upload", lambda *a, **k: uploads.append(k["data"]))
    provider = WorldLabsPreviewProvider()
    first = provider.submit(descriptor=descriptor, capture_root=tmp_path,
                            provider_adapter_input={"paid_resource_admission_grant": _grant(descriptor["metadata"]["website_reconstruction_admission"])})
    second = provider.submit(descriptor=descriptor, capture_root=tmp_path,
                            provider_adapter_input={"paid_resource_admission_grant": _grant(descriptor["metadata"]["website_reconstruction_admission"])})
    assert first["provider_run_id"] == second["provider_run_id"] == "operation-1"
    assert len(requests) == 3 and len(uploads) == 2


@pytest.mark.parametrize("bad", ["raw_input", "changed_image", "changed_task", "no_admission", "no_privacy"])
def test_invalid_preparation_cannot_call_worldlabs(tmp_path, bad):
    descriptor = _descriptor(tmp_path)
    metadata = descriptor["metadata"]
    preparation = metadata["clean_plate"]["prepared_views"]
    if bad == "raw_input":
        metadata["clean_plate"] = {}
        descriptor["raw_video_uri"] = "gs://bucket/raw.mov"
    elif bad == "changed_image":
        from pathlib import Path
        Path(preparation["frames"][0]["image_path"]).write_bytes(b"raw substituted")
    elif bad == "changed_task":
        metadata["site_task_context"]["description"] = "Different task"
    elif bad == "no_admission":
        metadata.pop("website_reconstruction_admission")
    else:
        metadata["clean_plate"]["privacy_verified"] = False
    with pytest.raises((ValueError, RuntimeError)):
        submit_website_prepared_views(descriptor=descriptor, capture_root=tmp_path,
                                      api_request=lambda *_a, **_k: pytest.fail("must not call provider"),
                                      upload=lambda *_a, **_k: pytest.fail("must not upload"))


def test_uncertain_purchase_is_not_repeated(tmp_path):
    descriptor = _descriptor(tmp_path)
    calls = []

    def api(path, **kwargs):
        calls.append(path)
        if path.endswith("prepare_upload"):
            return {"media_asset": {"media_asset_id": "image"}, "upload_info": {"upload_url": "https://upload.example"}}
        raise TimeoutError("response lost after provider accepted the request")

    with pytest.raises(TimeoutError):
        submit_website_prepared_views(descriptor=descriptor, capture_root=tmp_path, api_request=api,
                                      admission_grant=_grant(descriptor["metadata"]["website_reconstruction_admission"]), upload=lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="requires_reconciliation"):
        submit_website_prepared_views(descriptor=descriptor, capture_root=tmp_path,
                                      api_request=lambda *_a, **_k: pytest.fail("must not purchase twice"), upload=lambda *_a, **_k: None)
    assert calls.count("/marble/v1/worlds:generate") == 1


def test_admission_dictionary_cannot_authorize_marble_purchase(tmp_path):
    with pytest.raises(RuntimeError, match="grant_missing"):
        submit_website_prepared_views(descriptor=_descriptor(tmp_path), capture_root=tmp_path,
                                     api_request=lambda *_a, **_k: pytest.fail("must not call provider"),
                                     upload=lambda *_a, **_k: pytest.fail("must not upload"))


@pytest.mark.parametrize("budget", [None, True, -1, 2.47, float("nan"), float("inf")])
def test_insufficient_or_invalid_budget_cannot_upload_or_purchase(tmp_path, budget):
    descriptor = _descriptor(tmp_path)
    admission = descriptor["metadata"]["website_reconstruction_admission"]
    admission["maximum_cost_usd"] = budget
    with pytest.raises(ValueError, match="budget_insufficient"):
        submit_website_prepared_views(descriptor=descriptor, capture_root=tmp_path,
            admission_grant=_grant(admission), api_request=lambda *_a, **_k: pytest.fail("must not spend"),
            upload=lambda *_a, **_k: pytest.fail("must not upload"))


def test_canonical_allocator_preflights_then_submits_through_the_same_grant(tmp_path, monkeypatch, capsys):
    from blueprint_pipeline import paid_resource_allocator as allocator
    descriptor = _descriptor(tmp_path)
    descriptor["metadata"]["website_reconstruction_admission"]["source_commit"] = "1" * 40
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(descriptor))
    checks, calls = [], []
    def source(commit, **kwargs):
        checks.append((commit, kwargs))
        return [], commit
    monkeypatch.setattr(allocator, "_source_checkout_blockers", source)
    def submit(self, **kwargs):
        from blueprint_pipeline.paid_resource_admission import require_paid_resource_admission_grant
        require_paid_resource_admission_grant(kwargs["provider_adapter_input"]["paid_resource_admission_grant"],
            resource_class="provider_reconstruction_api", require_allocation_binding=True,
            allocation_binding_digest=descriptor["metadata"]["website_reconstruction_admission"]["allocation_binding_digest"])
        calls.append(kwargs)
        return {"provider_run_id": "operation", "status": "processing"}
    monkeypatch.setattr(WorldLabsPreviewProvider, "submit", submit)
    args = ["provider-reconstruction", "--provider", "world_labs", "--descriptor", str(path),
            "--capture-root", str(tmp_path), "--output-dir", str(tmp_path / "allocator"), "--experimental-branch-diagnostic"]
    assert allocator.main(args) == 0
    assert not calls
    assert allocator.main([*args, "--execute"]) == 0
    assert len(calls) == 1
    assert checks == [("1" * 40, {"allow_pushed_branch_diagnostic": True})] * 2
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_a, **_k: (["dirty_checkout"], "1" * 40))
    assert allocator.main([*args, "--execute"]) == 2
    assert len(calls) == 1


def test_disclosure_denial_blocks_the_provider(tmp_path):
    descriptor = _descriptor(tmp_path)
    admission = descriptor["metadata"]["website_reconstruction_admission"]
    admission["external_disclosure_allowed"] = False
    with pytest.raises(ValueError, match="disclosure_not_authorized"):
        submit_website_prepared_views(descriptor=descriptor, capture_root=tmp_path,
            admission_grant=_grant(admission), api_request=lambda *_a, **_k: pytest.fail("must not spend"),
            upload=lambda *_a, **_k: pytest.fail("must not upload"))
