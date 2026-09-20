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


@pytest.mark.parametrize("credits", [1600, 3100, None, True, -1])
def test_marble_poll_reports_settled_provider_cost_without_inventing_zero(monkeypatch, credits):
    monkeypatch.setattr(provider_preview, "_worldlabs_api_request", lambda *_a, **_k: {
        "done": True, "response": {"world_id": "world", "world_marble_url": "https://marble.worldlabs.ai/world/world"},
        "cost": {"total_credits": credits}})
    result = WorldLabsPreviewProvider().poll(run_id="operation")
    assert result["status"] == "ready"
    if credits in (1600, 3100):
        assert result["billing_status"] == "settled"
        assert result["cost_credits"] == credits
        assert result["cost_usd"] == credits / 1250
    else:
        assert result["billing_status"] == "unreported"
        assert "cost_usd" not in result


def test_controller_allocator_reserves_once_and_reuses_retained_marble_operation(tmp_path, monkeypatch):
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_task_context as control
    descriptor = _descriptor(tmp_path)
    descriptor["metadata"].pop("website_reconstruction_admission")
    reservations, calls, checks = [], [], []
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("1" * 40, True, True))
    def source(commit, **kwargs):
        checks.append(kwargs)
        return [], commit
    monkeypatch.setattr(allocator, "_source_checkout_blockers", source)
    monkeypatch.setattr(provider_preview, "_worldlabs_api_key", lambda: "test")
    def reserve(**kwargs):
        reservations.append(kwargs)
        admission = {"schema_version": "paid_lane_admission.v1", "status": "admitted", "blockers": [],
                     "resource_class": kwargs["resource_class"], "external_disclosure_allowed": True,
                     "maximum_cost_usd": kwargs["maximum_cost_usd"], "allocation_binding_digest": kwargs["binding_digest"]}
        return admission, _grant(admission)
    monkeypatch.setattr(control, "reserve_website_preparation_spend", reserve)
    def api(path, **kwargs):
        calls.append(path)
        if path.endswith("prepare_upload"):
            return {"media_asset": {"media_asset_id": "image"}, "upload_info": {"upload_url": "https://example.com/upload"}}
        return {"operation_id": "retained-operation"}
    monkeypatch.setattr(provider_preview, "_worldlabs_api_request", api)
    monkeypatch.setattr(provider_preview, "_presigned_upload", lambda *a, **kw: None)
    first = allocator.submit_sponsored_website_reconstruction(descriptor=descriptor, capture_root=tmp_path)
    second = allocator.submit_sponsored_website_reconstruction(descriptor=descriptor, capture_root=tmp_path)
    assert first["provider_run_id"] == second["provider_run_id"] == "retained-operation"
    assert len(reservations) == 1 and reservations[0]["provider"] == "world_labs"
    assert reservations[0]["maximum_cost_usd"] == 2.48
    assert calls.count("/marble/v1/worlds:generate") == 1
    assert checks == [{}]  # Automatic path cannot opt into a diagnostic branch.


def test_controller_does_not_reserve_marble_on_unmerged_code(tmp_path, monkeypatch):
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_task_context as control
    monkeypatch.setattr(allocator, "_current_checkout_source_state", lambda: ("1" * 40, True, True))
    monkeypatch.setattr(allocator, "_source_checkout_blockers", lambda *_: (["checkout_not_main"], "1" * 40))
    monkeypatch.setattr(control, "reserve_website_preparation_spend", lambda **_: pytest.fail("must not reserve"))
    with pytest.raises(ValueError, match="release_not_admitted"):
        allocator.submit_sponsored_website_reconstruction(descriptor=_descriptor(tmp_path), capture_root=tmp_path)


def test_controller_retains_original_reservation_after_upload_auth_failure(tmp_path, monkeypatch):
    from blueprint_pipeline import paid_resource_allocator as allocator
    from blueprint_pipeline import website_task_context as control
    descriptor = _descriptor(tmp_path)
    descriptor['metadata'].pop('website_reconstruction_admission')
    monkeypatch.setattr(allocator, '_current_checkout_source_state', lambda: ('1'*40, True, True))
    monkeypatch.setattr(allocator, '_source_checkout_blockers', lambda *_: ([], '1'*40))
    monkeypatch.setattr(provider_preview, '_worldlabs_api_key', lambda: 'test')
    reservations, calls = [], []
    def reserve(**kwargs):
        reservations.append(kwargs)
        if len(reservations)>1:
            assert kwargs['retained_admission']['status'] == 'admitted'
            assert kwargs['retained_admission']['allocation_binding_digest'] == kwargs['binding_digest']
        admission = {'schema_version':'paid_lane_admission.v1', 'status':'admitted','blockers':[],
                     'resource_class':kwargs['resource_class'], 'external_disclosure_allowed':True,
                     'maximum_cost_usd':kwargs['maximum_cost_usd'], 'allocation_binding_digest':kwargs['binding_digest']}
        return admission, _grant(admission)
    monkeypatch.setattr(control, 'reserve_website_preparation_spend', reserve)
    def api(path, **kwargs):
        calls.append(path)
        if len(calls)==1:
            raise RuntimeError('worldlabs_api_401')
        if path.endswith('prepare_upload'):
            return {'media_asset':{'media_asset_id':'image'}, 'upload_info':{'upload_url':'https://example.com/upload'}}
        return {'operation_id':'one-operation'}
    monkeypatch.setattr(provider_preview, '_worldlabs_api_request', api)
    monkeypatch.setattr(provider_preview, '_presigned_upload', lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match='worldlabs_api_401'):
        allocator.submit_sponsored_website_reconstruction(descriptor=descriptor, capture_root=tmp_path)
    second = allocator.submit_sponsored_website_reconstruction(descriptor=descriptor, capture_root=tmp_path)
    assert second['provider_run_id'] == 'one-operation'
    assert calls.count('/marble/v1/worlds:generate') == 1


@pytest.mark.parametrize('fault', [None, 'pending', 'unreported', 'wrong-operation', 'changed-admission', 'bad-receipt'])
def test_settlement_uses_only_bound_terminal_provider_billing(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    from blueprint_pipeline import website_task_context as control
    from blueprint_pipeline.website_worldlabs import settle_website_reconstruction
    from blueprint_pipeline.common import write_json
    root = tmp_path / 'pipeline/website_reconstruction'
    root.mkdir(parents=True)
    operation = {'operation_id':'op-one', 'done':True, 'error':None, 'cost':{'total_credits':1600}}
    if fault == 'pending':
        operation['done'] = False
    if fault == 'unreported':
        operation['cost'] = None
    if fault == 'wrong-operation':
        operation['operation_id'] = 'op-other'
    write_json(root/'submission.json', {'operation_id':'op-one', 'request_digest':'sha256:'+'a'*64})
    write_json(root/'controller_admission.json', {'allocation_binding_digest':'sha256:'+'a'*64,
        'task_context_digest':'sha256:'+('b' if fault != 'changed-admission' else 'c')*64})
    write_json(root/'operation.json', operation)
    context = {'capture_id':'walkthrough-req1','request_id':'req1','scene_id':'site-req1','context_digest':'sha256:'+'b'*64}
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        value = {**json.loads(kwargs['data'])['settlement'], 'status':'settled',
                 'actual_cost_usd':0 if fault=='bad-receipt' else 1.28}
        return SimpleNamespace(body=json.dumps(value).encode())
    monkeypatch.setenv('PIPELINE_SYNC_WEBAPP_URL', 'https://tryblueprint.io/api/internal/pipeline/sync')
    monkeypatch.setattr(control, 'load_pipeline_sync_token', lambda: 'test-secret')
    monkeypatch.setattr(control, 'safe_request', post)
    kwargs = dict(provider_run={'provider_run_id':'op-one','worldlabs_operation_manifest_uri':str(root/'operation.json')},
                  capture_root=tmp_path,task_context=context)
    if fault in ('wrong-operation','changed-admission','bad-receipt'):
        with pytest.raises(ValueError,match='billing_binding|settlement_receipt'):
            settle_website_reconstruction(**kwargs)
        assert not (root/'settlement.json').exists()
    elif fault:
        assert settle_website_reconstruction(**kwargs) is None and not calls
    else:
        receipt = settle_website_reconstruction(**kwargs)
        assert receipt['actual_cost_usd'] == 1.28
        url, request = calls[0]
        assert url.endswith('/api/internal/pipeline/creator-captures/walkthrough-req1/preparation-settlement')
        assert request['headers']['X-Blueprint-Pipeline-Signature'].startswith('sha256=')
        assert request['method'] == 'POST' and request['timeout_seconds'] == 10
        assert json.loads(request['data'])['request_id'] == 'req1'
        assert json.loads((root/'settlement.json').read_text()) == receipt
