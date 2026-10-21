"""No-provider regressions for bounded FLORA packaging and paid admission."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import flora_allocator_lane as lane
from blueprint_pipeline import flora_provider_bundle as bundler
from blueprint_pipeline import flora_provider_runtime as worker
from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline.gpu_selection_policy import (
    gpu_allowed_by_policy, policy_manifest, required_gpu_count, resolve_gpu_selection_policy,
)
from blueprint_pipeline.vast_provider_adapter import _search_payload


def _bundle(tmp_path, monkeypatch):
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as archive:
        data = Path(worker.__file__).read_bytes()
        info = tarfile.TarInfo("src/blueprint_pipeline/flora_provider_runtime.py")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(bundler.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=tar_bytes.getvalue()))
    monkeypatch.setattr(bundler, "validate_flora_benchmark_inputs", lambda *a: {
        "status":"inputs_verified", "eligible_episode_count":1, "selected_episode_count":1})
    root = tmp_path / "benchmark"
    root.mkdir()
    (root / "manifest.json").write_text("{}")
    (root / "profile.json").write_text("{}")
    (root / "rights").mkdir()
    (root / "rights/episode.json").write_text("{}")
    bundle = bundler.build_flora_provider_bundle(benchmark_root=root, output_bundle=tmp_path / "bundle.zip",
        source_commit="a" * 40, container_image="docker.io/test/flora@sha256:" + "b" * 64)
    return bundle


def _authority(bundle):
    value = {"schema_version":"flora_paid_attempt_authority.v1", "status":"authorized", "provider":"vast",
        "probe_kind":lane.PROBE_KIND, "model_id":bundler.MODEL_ID, "model_revision":bundler.MODEL_REVISION,
        "claim_class":bundler.CLAIM_CLASS, "maximum_provider_allocations":1, "automatic_paid_retry_allowed":False,
        "provider_zero_required_after_return":True, "tensor_parallel_size":2, "model_terms_accepted":True,
        "external_video_processing_authorized":True, "research_only":True, "allowed_geolocation_country_codes":["US"],
        "authorization_id":"fixture-only", "authorized_by":"fixture", "user_authorization_evidence":"hermetic-fixture",
        "model_terms_evidence":"hermetic-fixture", "video_rights_evidence":"hermetic-fixture",
        "bundle_sha256":bundle["bundle_sha256"], "source_commit":bundle["source_commit"],
        "container_image":bundle["container_image"], "hard_cap_usd":20.0, "max_hourly_rate_usd":9.0,
        "hard_ttl_seconds":7200, "expected_provider_download_bytes":75_000_000_000,
        "provider_output_archive_max_bytes":1_000_000_000,
        "expected_provider_upload_bytes":3_000_000_000,
        "expires_at":(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
    value["authority_digest"] = bundler.digest(value)
    return value


def test_bundle_has_immutable_source_and_rejects_tampering(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    path = Path(bundle["bundle_path"])
    checked = bundler.validate_flora_provider_bundle(bundle, bundle_path=path)
    assert checked["source_commit"] == "a" * 40
    with zipfile.ZipFile(path) as archive:
        assert "benchmark/rights/episode.json" in archive.namelist()
        assert archive.read("provider_runtime/evaluator_provider_runtime_runner.py") == Path(worker.__file__).read_bytes()
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="receipt_binding_invalid"):
        bundler.validate_flora_provider_bundle(bundle, bundle_path=path)


@pytest.mark.parametrize("field,value", [("hard_cap_usd", None), ("hard_cap_usd", float("nan")),
    ("model_terms_accepted", False), ("external_video_processing_authorized", False),
    ("maximum_provider_allocations", 2), ("automatic_paid_retry_allowed", True),
    ("allowed_geolocation_country_codes", ["US", "GB"]), ("source_commit", "c" * 40),
    ("expected_provider_download_bytes", 1), ("expires_at", "2020-01-01T00:00:00Z")])
def test_authority_rejects_missing_caps_rights_retries_and_identity(tmp_path, monkeypatch, field, value):
    bundle = _bundle(tmp_path, monkeypatch)
    authority = _authority(bundle)
    authority[field] = value
    authority["authority_digest"] = bundler.digest({k:v for k,v in authority.items() if k != "authority_digest"})
    with pytest.raises(ValueError):
        lane.validate_flora_attempt_authority(authority, bundle)


def test_single_use_consumption_is_global_across_job_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_SPEND_AUTHORITY_ROOT", str(tmp_path / "authorities"))
    authority = {"authorization_id":"single-fixture", "authority_digest":"sha256:" + "1" * 64}
    assert lane._consume_once(authority)["status"] == "consumed"
    with pytest.raises(ValueError, match="already_consumed"):
        lane._consume_once(authority)


@pytest.mark.parametrize("field,value", [
    ("provider_output_archive_max_bytes", None), ("provider_output_archive_max_bytes", True),
    ("provider_output_archive_max_bytes", 0), ("provider_output_archive_max_bytes", 1_000_000_001),
    ("provider_output_archive_max_bytes", 1_000_000_000.0),
    ("expected_provider_upload_bytes", 2_999_999_999), ("expected_provider_upload_bytes", True),
])
def test_output_archive_bound_and_all_three_puts_require_explicit_reserve(tmp_path, monkeypatch, field, value):
    bundle = _bundle(tmp_path, monkeypatch)
    authority = _authority(bundle)
    authority[field] = value
    authority["authority_digest"] = bundler.digest({k:v for k,v in authority.items() if k != "authority_digest"})
    with pytest.raises(ValueError, match="output_archive_limit_invalid|transfer_budget_invalid"):
        lane.validate_flora_attempt_authority(authority, bundle)


def test_exact_device_count_reaches_search_and_rejects_provider_mismatch():
    assert _search_payload(limit=10, max_hourly_rate=9, gpu_count=2)["num_gpus"] == {"eq":2}
    assert _search_payload(limit=10, max_hourly_rate=9)["num_gpus"] == {"eq":1}
    policy = resolve_gpu_selection_policy(lane.GPU_POLICY)
    assert gpu_allowed_by_policy({"gpu_name":"H200", "num_gpus":2, "cuda_max_good":13.0}, policy)
    assert not gpu_allowed_by_policy("H200", policy, cuda_max_good=13.0)
    for count in (1, 4, None, True):
        assert not gpu_allowed_by_policy({"gpu_name":"H200", "num_gpus":count, "cuda_max_good":13.0}, policy)
    for count in (0, -1, 9, True, 2.0, "2"):
        with pytest.raises(ValueError, match="gpu_count_invalid"):
            _search_payload(limit=10, max_hourly_rate=9, gpu_count=count)


def test_legacy_inline_policy_resolves_without_changing_offer_count_semantics():
    resolved = resolve_gpu_selection_policy({"policy_id":"existing", "allowed_gpu_keywords":["H200"]})
    assert "required_gpu_count" not in resolved
    assert "required_gpu_count" not in policy_manifest(resolved)
    assert gpu_allowed_by_policy({"gpu_name":"H200"}, resolved)
    assert gpu_allowed_by_policy("H200", resolved)
    assert _search_payload(limit=10, max_hourly_rate=9, gpu_count=required_gpu_count(resolved))["num_gpus"] == {"eq":1}


def test_missing_authority_cannot_reach_provider_or_consume(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("provider or consumption called without valid authority")
    monkeypatch.setattr(lane, "get_render_provider", forbidden)
    monkeypatch.setattr(lane, "_consume_once", forbidden)
    monkeypatch.setattr(lane, "run_arena_native_control_vast", forbidden)
    args = SimpleNamespace(provider="vast", flora_job_dir=str(tmp_path), flora_bundle=None,
        flora_bundle_receipt=None, flora_attempt_authority=None, execute=True,
        admission_out=str(tmp_path / "admission.json"), adapter_output=str(tmp_path / "result.json"))
    assert lane.run_flora_allocator_lane(args, ([], "a" * 40)) == 2
    assert "flora_attempt_authority_missing" in json.loads((tmp_path / "result.json").read_text())["blockers"]


def test_dry_and_execute_share_preflight_and_transport_bounds(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    authority = _authority(bundle)
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority))
    args = SimpleNamespace(provider="vast", flora_job_dir=str(tmp_path / "job"), flora_bundle=bundle["bundle_path"],
        flora_bundle_receipt=str(Path(bundle["bundle_path"]).with_suffix(".receipt.json")),
        flora_attempt_authority=str(authority_path), execute=False,
        admission_out=str(tmp_path / "admission.json"), adapter_output=str(tmp_path / "result.json"))
    calls = []
    monkeypatch.setattr(lane, "_environment_preflight", lambda *a: calls.append("preflight") or [])
    monkeypatch.setattr(lane, "_consume_once", lambda *a: calls.append("consume") or {"status":"consumed"})
    transport = []
    monkeypatch.setattr(lane, "run_arena_native_control_vast", lambda **kw: transport.append(kw) or {"status":"completed"})
    assert lane.run_flora_allocator_lane(args, ([], "a" * 40)) == 0
    assert calls == ["preflight"] and not transport
    args.execute = True
    assert lane.run_flora_allocator_lane(args, ([], "a" * 40)) == 0
    assert calls == ["preflight", "preflight", "consume"]
    assert transport[0]["require_independent_watchdog"] is True
    assert transport[0]["wait_for_fresh_spend_lock"] is True
    assert transport[0]["require_direct_port"] is True
    assert transport[0]["stale_offer_create_retry_limit"] == 0
    assert transport[0]["gpu_selection_policy"]["required_gpu_count"] == 2
    assert transport[0]["candidate_policy_query_expected"] is False
    assert transport[0]["ngc_image_login_mode"] == "never"
    assert transport[0]["expected_provider_upload_bytes"] == 3_000_000_000
    assert transport[0]["provider_runtime_environment"] == {
        "BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES": "1000000000",
    }


def _preflight_fixture(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    authority = _authority(bundle)
    observed = {"inventory": 0, "capacity": 0, "clock": 0.0, "sleeps": [], "preflights": 0}
    def inventory(**kwargs):
        observed["inventory"] += 1
        return {"status": "observed", "api_confirmed": True, "live_resource_count": 0}
    def capacity(request):
        observed["capacity"] += 1
        return {"status": "available", "observation": observed["capacity"]}
    def sleep(seconds):
        observed["sleeps"].append(seconds)
        observed["clock"] += seconds
    monkeypatch.setattr(lane, "get_render_provider", lambda _: SimpleNamespace(
        billable_inventory=inventory, capacity_preflight=capacity))
    monkeypatch.setattr(lane, "_caller_exit_survival_contract", lambda: {})
    monkeypatch.setattr(lane, "_caller_exit_survival_proven", lambda _: True)
    monkeypatch.setattr(lane, "_vast_credential_file_present", lambda: True)
    monkeypatch.setattr(lane, "consumption_root", lambda: tmp_path / "consumption")
    monkeypatch.setattr(arena.time, "monotonic", lambda: observed["clock"])
    monkeypatch.setattr(arena.time, "sleep", sleep)
    monkeypatch.setattr(lane, "_consume_once", lambda *_: pytest.fail("authority consumed by preflight"))
    monkeypatch.setattr(lane, "run_arena_native_control_vast", lambda **_: pytest.fail("provider launch during preflight"))
    return bundle, authority, observed


def test_stale_lock_wait_refreshes_inventory_capacity_and_retains_each_attempt(tmp_path, monkeypatch):
    bundle, authority, observed = _preflight_fixture(tmp_path, monkeypatch)
    def preflight(**kwargs):
        observed["preflights"] += 1
        result = {"status": "PASS", "blockers": [], "capacity_evidence": kwargs["capacity_evidence"]}
        if observed["preflights"] == 1:
            result.update(status="FAIL", blockers=["spend_admission:spend_admission_lock_stale"])
            raise arena.PreSpendPreflightBlocked(result)
        return result
    monkeypatch.setattr(arena, "require_pre_spend_preflight", preflight)
    assert lane._environment_preflight(authority, bundle, tmp_path / "job") == []
    assert observed == {"inventory": 2, "capacity": 2, "clock": 15.0, "sleeps": [15], "preflights": 2}
    receipts = sorted((tmp_path / "job/preflight/spend_lock_wait").glob("*/*.json"))
    assert [json.loads(path.read_text())["status"] for path in receipts] == ["FAIL", "PASS"]
    assert len(list((tmp_path / "job/preflight/environment_observations").glob("*.json"))) == 2
    assert json.loads((tmp_path / "job/preflight/provider_capacity.json").read_text())["observation"] == 2


def test_persistent_stale_lock_stops_at_monotonic_deadline_without_consuming(tmp_path, monkeypatch):
    bundle, authority, observed = _preflight_fixture(tmp_path, monkeypatch)
    def preflight(**kwargs):
        observed["preflights"] += 1
        # Request latency also consumes the wait budget.
        observed["clock"] += 50
        raise arena.PreSpendPreflightBlocked({"blockers": ["spend_admission:spend_admission_lock_stale"]})
    monkeypatch.setattr(arena, "require_pre_spend_preflight", preflight)
    assert lane._environment_preflight(authority, bundle, tmp_path / "job") == ["spend_admission:spend_admission_lock_stale"]
    assert observed["preflights"] == 4
    assert observed["clock"] == 245
    assert observed["sleeps"] == [15, 15, 15]


def test_slow_environment_refresh_cannot_start_spend_check_after_deadline(tmp_path, monkeypatch):
    bundle, authority, observed = _preflight_fixture(tmp_path, monkeypatch)
    arguments = lane._environment_preflight_arguments
    def slow_refresh(*args):
        result = arguments(*args)
        if observed["preflights"]:
            observed["clock"] += 240
        return result
    monkeypatch.setattr(lane, "_environment_preflight_arguments", slow_refresh)
    def preflight(**kwargs):
        observed["preflights"] += 1
        if observed["preflights"] > 1:
            pytest.fail("canonical spend check started after refresh exhausted deadline")
        raise arena.PreSpendPreflightBlocked({"blockers": ["spend_admission:spend_admission_lock_stale"]})
    monkeypatch.setattr(arena, "require_pre_spend_preflight", preflight)
    assert lane._environment_preflight(authority, bundle, tmp_path / "job") == ["spend_admission:spend_admission_lock_stale"]
    assert observed["preflights"] == 1 and observed["sleeps"] == [15]


@pytest.mark.parametrize("blockers", [
    ["spend_admission:vast_billing_missing"],
    ["spend_admission:spend_admission_lock_stale", "spend_admission:vast_billing_missing"],
])
def test_stale_lock_wait_never_retries_other_spend_blockers(tmp_path, monkeypatch, blockers):
    bundle, authority, observed = _preflight_fixture(tmp_path, monkeypatch)
    def preflight(**kwargs):
        observed["preflights"] += 1
        raise arena.PreSpendPreflightBlocked({"blockers": blockers})
    monkeypatch.setattr(arena, "require_pre_spend_preflight", preflight)
    assert lane._environment_preflight(authority, bundle, tmp_path / "job") == sorted(blockers)
    assert observed["sleeps"] == [] and observed["preflights"] == 1


@pytest.mark.parametrize("changed_fact", ["authority", "capacity"])
def test_stale_lock_retry_stops_when_refreshed_environment_changes(tmp_path, monkeypatch, changed_fact):
    bundle, authority, observed = _preflight_fixture(tmp_path, monkeypatch)
    validate = lane.validate_flora_attempt_authority
    def validate_again(*args):
        if changed_fact == "authority" and observed["preflights"]:
            raise ValueError("flora_attempt_authority_expired_or_invalid")
        return validate(*args)
    monkeypatch.setattr(lane, "validate_flora_attempt_authority", validate_again)
    def preflight(**kwargs):
        observed["preflights"] += 1
        if changed_fact == "capacity":
            monkeypatch.setattr(lane, "get_render_provider", lambda _: SimpleNamespace(
                billable_inventory=lambda **_: {"status": "observed", "api_confirmed": True, "live_resource_count": 0},
                capacity_preflight=lambda _: {"status": "blocked", "blockers": ["new_capacity_blocker"]}))
        raise arena.PreSpendPreflightBlocked({"blockers": ["spend_admission:spend_admission_lock_stale"]})
    monkeypatch.setattr(arena, "require_pre_spend_preflight", preflight)
    expected = "flora_attempt_authority_expired_or_invalid" if changed_fact == "authority" else "new_capacity_blocker"
    assert lane._environment_preflight(authority, bundle, tmp_path / "job") == [expected]
    assert observed["sleeps"] == [15] and observed["preflights"] == 1


def test_worker_checks_members_before_model_download(tmp_path):
    (tmp_path / "member").write_text("bad")
    with pytest.raises(ValueError, match="member_digest_mismatch"):
        worker._verify(tmp_path, {"files":{"member":"sha256:" + "0" * 64}})


def test_worker_stops_before_model_import_or_download_when_bootstrap_fails(tmp_path, monkeypatch):
    from blueprint_pipeline import flora_dependency_bootstrap as bootstrap
    root, output = tmp_path / "bundle", tmp_path / "output"
    runtime = root / "provider_runtime"
    runtime.mkdir(parents=True)
    (runtime / "evaluator_provider_runtime_manifest.json").write_text(json.dumps({"source_commit":"a" * 40}))
    (runtime / "evaluator_input_manifest.json").write_text(json.dumps({
        "model_id":bundler.MODEL_ID, "model_revision":bundler.MODEL_REVISION}))
    monkeypatch.setenv("BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR", str(root))
    monkeypatch.setenv("BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR", str(output))
    monkeypatch.setattr(worker.sys, "path", list(worker.sys.path))
    monkeypatch.setattr(worker, "_verify", lambda *a: None)
    receipt = {"status":"blocked", "blockers":["fixture_bootstrap_refused"]}
    def refuse(**kwargs):
        assert kwargs["output_root"] == output / "dependency_bootstrap"
        raise bootstrap.FloraDependencyBootstrapError("fixture_bootstrap_refused", receipt=receipt)
    monkeypatch.setattr(bootstrap, "ensure_flora_runtime_dependencies", refuse)
    monkeypatch.setattr(worker.importlib, "import_module", lambda *a: pytest.fail("model dependency import before bootstrap"))
    assert worker.main() == 2
    result = json.loads((output / "evaluator_runtime_result.json").read_text())
    assert result["dependency_bootstrap"] == receipt
    assert result["blockers"] == ["fixture_bootstrap_refused"]
    assert result["candidate_policy_queried"] is False


def test_bundle_admission_reopens_rights_before_returning_ready(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    def refuse(*args):
        raise ValueError("flora_benchmark_inputs_or_rights_blocked")
    monkeypatch.setattr(bundler, "validate_flora_benchmark_inputs", refuse)
    with pytest.raises(ValueError, match="inputs_or_rights_blocked"):
        bundler.validate_flora_provider_bundle(bundle, bundle_path=bundle["bundle_path"])


def test_provider_report_requires_live_inference_and_preserves_partial_status():
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    report = {"status":"partial", "runtime_execution_verified":True, "inference_performed":True,
              "live_episode_count":20, "fixture_episode_count":0, "unknown_episode_count":0}
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    summary = worker._benchmark_execution_summary(report)
    assert summary["runtime_execution_verified"] is True
    assert summary["benchmark_grades_complete"] is False
    report["inference_performed"] = False
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    with pytest.raises(ValueError, match="real_model_execution_unverified"):
        worker._benchmark_execution_summary(report)


def test_capacity_preflight_matches_actual_ssh_transport_requirements(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path, monkeypatch)
    authority = _authority(bundle)
    requests = []
    provider = SimpleNamespace(
        billable_inventory=lambda **kw: {"status":"observed", "api_confirmed":True, "live_resource_count":0},
        capacity_preflight=lambda request: requests.append(request) or {"status":"available", "blockers":[]},
    )
    monkeypatch.setattr(lane, "get_render_provider", lambda name: provider)
    monkeypatch.setattr(lane, "_caller_exit_survival_proven", lambda *a: True)
    monkeypatch.setattr(lane, "_caller_exit_survival_contract", lambda: {})
    monkeypatch.setattr(lane, "_remaining_session_live_minutes", lambda **kw: 120)
    monkeypatch.setattr(lane, "_vast_credential_file_present", lambda: True)
    monkeypatch.setattr(arena, "require_pre_spend_preflight", lambda **kw: {"status": "PASS", "blockers": []})
    monkeypatch.setattr(lane, "consumption_root", lambda: tmp_path / "consumption")
    assert lane._environment_preflight(authority, bundle, tmp_path / "job") == []
    assert requests[0]["require_direct_port"] is True
    assert requests[0]["minimum_driver_version"] == lane.MINIMUM_DRIVER_VERSION
    assert requests[0]["gpu_selection_policy"]["required_gpu_count"] == 2


def test_bundle_rejects_known_input_gaps_before_paid_execution(tmp_path, monkeypatch):
    from blueprint_pipeline import flora_episode_benchmark as benchmark
    profile = {"execution_site":"external_provider", "compute_provider_id":"vast", "runtime_config":{
        "model_path":"/opt/blueprint/flora/model", "source_root":"/opt/blueprint/flora/source", "tensor_parallel_size":2}}
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    monkeypatch.setattr(benchmark, "run_flora_benchmark", lambda **kw: {
        "status":"inputs_verified", "eligible_episode_count":1, "selected_episode_count":2, "blockers":[]})
    with pytest.raises(ValueError, match="inputs_or_rights_blocked"):
        bundler.validate_flora_benchmark_inputs(tmp_path, {"profile_relative_path":"profile.json",
            "manifest_relative_path":"manifest.json", "rights_relative_path":"rights"})


@pytest.mark.slow
def test_candidate_watchdog_arms_in_real_subprocess_without_service_deploy(tmp_path, monkeypatch):
    """The paid lane's child imports this candidate, before any provider access."""
    from blueprint_pipeline import vast_independent_watchdog_control as control

    source_root = Path(control.__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", str(source_root))
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    # The process is stopped immediately after its armed receipt. Provider reads
    # and mutations are deferred until cancellation or the two-minute deadline.
    exact_name = "blueprint-flora-" + "a" * 32
    handoff, handle = control.arm_independent_vast_watchdog(
        job_dir=tmp_path, max_live_minutes=2,
        generated_at=datetime.now(timezone.utc).isoformat(),
        pod_name_prefix="blueprint-flora-", resource_name_exact=exact_name,
    )
    try:
        assert handle is not None, handoff
        evidence = json.loads((handle.out_dir / control.EVIDENCE_NAME).read_text())
        assert handoff["status"] == "armed"
        assert handoff["caller_exit_survival_contract"] == "detached_posix_session"
        assert evidence["pod_name_prefix"] == exact_name
        assert evidence["provider"] == "vast"
        assert evidence["provider_mutations_performed"] == 0
        assert handle.process.poll() is None
    finally:
        if handle is not None:
            handle.process.terminate()
            handle.process.wait(timeout=5)
