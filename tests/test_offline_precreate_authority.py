"""Exercise owner expiry at the adapter's actual pre-create hook, after offer search."""
import pytest

from blueprint_pipeline import native_task_arena_vast as native
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import vast_provider_adapter as adapter
from tests.test_native_task_arena_paired_witness_staging import context as context
from tests.test_task_evaluation_scene_intake import stage, attempt, request
from tests.test_vast_provider_adapter import _configure_live_gates, _paid_grant


@pytest.mark.parametrize("ending", ["expiry", "revocation"])
def test_bound_session_hook_refuses_new_create_after_offer_search(context, tmp_path, monkeypatch, ending):
    _, bundle_path, binding = context
    owner_root = tmp_path / "owner"
    intent = stage(owner_root)
    reserved = attempt(owner_root, intent, commit="c", cost=4)
    owner = {**authority.bind_scene_attempt(reserved), "source_commit": reserved["source_commit"]}
    monkeypatch.setenv(intake.ROOT_ENV, str(owner_root))
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    clock = [102]
    monkeypatch.setattr(authority.time, "time", lambda: clock[0])
    session = {**binding, "hard_cap_usd": 4., "hard_ttl_seconds": 9000,
               "resource_name": "blueprint-native-task-policy-canary-test", "scene_execution_owner": owner}
    bundle = {"implementation_commit": binding["implementation_commit"],
        "bundle_sha256": binding["provider_bundle_sha256"], "container_image": "immutable-image",
        "bundle_path": str(bundle_path), "bundle_size_bytes": bundle_path.stat().st_size}
    monkeypatch.setattr(native, "validate_policy_canary_session_authority", lambda value: value)
    monkeypatch.setattr(native, "validate_policy_canary_provider_bundle", lambda *_a, **_kw: bundle)
    monkeypatch.setattr(native, "_policy_provider_transfer_byte_budget", lambda _candidate: (100, 20))
    monkeypatch.setattr(native, "run_arena_native_control_vast", lambda **kwargs: kwargs)
    forwarded = native.run_native_task_arena_policy_canary_session_vast(job_dir=tmp_path,
        prepared_bundle=bundle, session_authority=session, paid_resource_admission_grant=None,
        execute=False, hard_ttl_seconds=9000,
        provider_runtime_environment={"BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "640x360"})
    hook = forwarded["pre_provider_mutation_hook"]
    assert hook()["scene_attempt_id"] == reserved["attempt_id"]
    _configure_live_gates(tmp_path, monkeypatch)
    monkeypatch.setenv(adapter.VAST_LAUNCH_LOCK_FILE_ENV, str(tmp_path / "launch.lock"))
    requests = []
    def transport(*, method, path, **_kwargs):
        requests.append((method, path))
        if method == "GET" and path == "/instances/":
            return 200, {"instances": []}
        if method == "POST" and path == "/bundles/":
            if ending == "revocation":
                intake.revoke_scene_intent(queue_root=owner_root, intent_id=intent["intent_id"],
                    intent_digest=intent["intent_digest"], owner=request()["owner"], now=103)
                clock[0] = 104
            else:
                clock[0] = 1001
            return 200, {"offers": [{"id": 303, "ask_contract_id": 303, "gpu_name": "RTX A6000",
                "gpu_ram": 49152, "dph_total": 0.25, "driver_version": "580.159.03",
                "machine_id": 9303, "num_gpus": 1, "rentable": True}]}
        pytest.fail(f"external mutation edge reached: {method} {path}")
    monkeypatch.setattr(adapter, "_api_json", transport)
    result = adapter.run_vast_provider_adapter(job_dir=tmp_path / "adapter",
        mode="live-startup-probe", paid_resource_admission_grant=_paid_grant(),
        allow_vast_api_call=True, allow_instance_launch=True, poll_interval_seconds=0,
        startup_timeout_seconds=20, session_max_live_minutes=None, pre_provider_mutation_hook=hook,
        stale_offer_create_retry_limit=0)
    assert ("POST", "/bundles/") in requests
    assert not any(method in {"PUT", "DELETE"} for method, _path in requests)
    assert result["provider_create_attempted"] is False
    assert result["vast_instance_ids"] == []
    assert f"scene_execution_owner_{'expired' if ending == 'expiry' else 'revoked'}" in str(result)
