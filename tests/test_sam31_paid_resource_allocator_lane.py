from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import blueprint_pipeline.paid_resource_allocator as allocator
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_paid_resource_allocator_lane import (
    run_sam31_paid_resource_allocator_lane,
)
from blueprint_pipeline.sam31_vast_source_track_canary import (
    PRELAUNCH_INVENTORY_RECEIPT_NAME,
    PRELAUNCH_INVENTORY_SCHEMA_VERSION,
    Sam31VastCanaryError,
)


def _args(tmp_path: Path, *, execute: bool) -> Namespace:
    return Namespace(
        provider_launch_request=str(tmp_path / "request.json"),
        preflight_bundle=str(tmp_path / "preflight.json"),
        admission_out=str(tmp_path / "admission.json"),
        bound_request_out=str(tmp_path / "bound.json"),
        adapter_output=str(tmp_path / "adapter.json"),
        provider="vast",
        expected_source_commit="c" * 40,
        sam31_max_spend_usd=1.0,
        sam31_max_hourly_rate_usd=0.5,
        sam31_hard_ttl_seconds=300,
        sam31_retry_cap=0,
        sam31_authority_id="fixture-authority",
        sam31_input_bundle=str(tmp_path / "input.zip"),
        sam31_input_bundle_receipt=str(tmp_path / "input-receipt.json"),
        sam31_attempt_authority=str(tmp_path / "authority.json"),
        sam31_allowed_active_vast_instance_id=[],
        sam31_hf_token_file=str(tmp_path / "hf-token.txt"),
        provider_bundle_url_file=str(tmp_path / "input-url.txt"),
        provider_output_put_url_file=str(tmp_path / "put-url.txt"),
        provider_output_get_url_file=str(tmp_path / "get-url.txt"),
        execute=execute,
    )


def _write_private(path: Path, value: str, *, mode: int = 0o600) -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(mode)


class _ReadOnlyProvider:
    def capacity_preflight(self, _request):
        return {
            "status": "available",
            "selected_offer": {
                "gpu_name": "L40S",
                "gpu_ram_mb": 48_000,
                "on_demand_price_usd_per_hour": 0.5,
            },
        }

    def billable_inventory(self, *, name_prefix: str):
        return {
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
            "name_prefix": name_prefix,
        }


def test_sam31_allocator_lane_routes_exact_private_inputs(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.provider_launch_request), {"request": True})
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    write_json(Path(args.sam31_attempt_authority), {"request_authority_id": "fixture-authority"})
    write_json(Path(args.sam31_input_bundle_receipt), {"receipt": True})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    # Canonical production secret integration: root-owned and readable by the
    # private ``blueprint`` service group, with no group write or world bits.
    _write_private(Path(args.sam31_hf_token_file), "hf-secret", mode=0o640)
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.validate_sam31_paid_attempt_authority",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.consume_sam31_paid_attempt_authority_once",
        lambda *_args, **_kwargs: {
            "status": "consumed",
            "authorization_digest": "sha256:" + "a" * 64,
        },
    )
    observed: dict[str, object] = {}

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        write_json(Path(kwargs["bound_request_out"]), {"bound": True})
        return {"status": "execute_ready", "blockers": []}

    def execute(**kwargs):
        observed["execute"] = kwargs
        canary_root = Path(kwargs["job_dir"])
        canary_root.mkdir(parents=True, exist_ok=True)
        write_json(canary_root / "provider_runtime_result.json", {"status": "passed"})
        write_json(
            canary_root / "semantic_source_track_import_result.v1.json",
            {
                "schema_version": "semantic_source_track_import_result.v1",
                "status": "completed",
                "result_digest": "sha256:" + "e" * 64,
            },
        )
        write_json(
            canary_root / "teardown_receipt.json",
            {
                "status": "PASS",
                "instance_id": "123",
                "provider_zero_verified": True,
                "teardown_receipt_digest": "sha256:" + "d" * 64,
            },
        )
        return {
            "status": "completed",
            "instance_id": "123",
            "provider_mutations_performed": 2,
            "provider_zero_verified": True,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        }

    provider = _ReadOnlyProvider()
    started = tmp_path / "watchdog" / "started.txt"

    def arm(**_kwargs):
        observed["arm"] = _kwargs
        return (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            SimpleNamespace(started_instance_id_path=started),
        )

    def stage(**kwargs):
        root = Path(kwargs["job_dir"])
        root.mkdir(parents=True)
        for name, value in (
            ("provider_bundle_url.txt", "https://objects.example/input"),
            ("provider_output_put_url.txt", "https://objects.example/put"),
            ("provider_output_get_url.txt", "https://objects.example/get"),
        ):
            _write_private(root / name, value)
        return {"status": "completed", "blockers": []}

    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=prepare,
        provider_factory=lambda _name: provider,
        execute_canary=execute,
        stage_bundle=stage,
        cleanup_bundle=lambda *_args, **_kwargs: {
            "all_objects_absent": True,
            "signed_url_files_removed": True,
        },
        arm_watchdog=arm,
        close_watchdog=lambda **kwargs: (
            write_json(
                Path(kwargs["job_dir"])
                / "independent_vast_watchdog"
                / "groot_oscar_runpod_canary_watchdog.json",
                {"status": "provider_terminal"},
            )
            or {"status": "provider_terminal"}
        ),
    )
    assert result["status"] == "completed"
    assert observed["prepare"]["execution_adapter_qualified"] is True
    assert observed["arm"]["pod_name_prefix"] == "blueprint-sam31-source-tracks-"
    assert observed["execute"]["provider"] is provider
    assert observed["execute"]["hf_token"] == "hf-secret"
    assert observed["execute"]["input_bundle_get_url"].endswith("/input")
    assert observed["execute"]["paid_resource_admission_grant"].resource_class == "gpu_render"
    assert result["all_staged_objects_absent"] is True
    assert result["continuing_spend_from_this_run"] is False
    assert result["authorization_consumption"]["status"] == "consumed"
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()
    assert Path(result["source_track_import_result_path"]).is_file()
    teardown = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert teardown["continuing_spend_from_this_run"] is False
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text())
    assert manifest["status"] == "completed"
    assert manifest["binding"]["allocator_lane"] == "semantic_sam31_source_tracks"
    assert {
        "allocator_adapter_result",
        "sam31_runtime_result",
        "sam31_normalized_source_tracks",
        "sam31_source_teardown_receipt",
        "sam31_watchdog_receipt",
        "teardown_manifest",
    }.issubset(set(manifest["observed_roles"]))
    assert result["execution_result_digest"] == canonical_digest(
        result, digest_field="execution_result_digest"
    )


def test_sam31_allocator_exception_preserves_terminal_provider_zero(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.provider_launch_request), {"request": True})
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    write_json(
        Path(args.sam31_attempt_authority),
        {"request_authority_id": "fixture-authority"},
    )
    write_json(Path(args.sam31_input_bundle_receipt), {"receipt": True})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    _write_private(Path(args.sam31_hf_token_file), "hf-secret", mode=0o640)
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.validate_sam31_paid_attempt_authority",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.consume_sam31_paid_attempt_authority_once",
        lambda *_args, **_kwargs: {"status": "consumed"},
    )
    started = tmp_path / "watchdog" / "started.txt"

    def execute(**kwargs):
        canary_root = Path(kwargs["job_dir"])
        canary_root.mkdir(parents=True)
        started.parent.mkdir(parents=True)
        started.write_text("123", encoding="utf-8")
        write_json(
            canary_root / "teardown_receipt.json",
            {
                "status": "PASS",
                "instance_id": "123",
                "provider_zero_verified": True,
                "teardown_receipt_digest": "sha256:" + "d" * 64,
            },
        )
        raise urllib.error.HTTPError("https://objects.example/output", 404, "Not Found", None, None)

    def stage(**kwargs):
        root = Path(kwargs["job_dir"])
        root.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            _write_private(root / name, "https://objects.example/value")
        return {"status": "completed", "blockers": []}

    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **kwargs: (
            write_json(Path(kwargs["bound_request_out"]), {"bound": True})
            or {"status": "execute_ready", "blockers": []}
        ),
        provider_factory=lambda _name: _ReadOnlyProvider(),
        execute_canary=execute,
        stage_bundle=stage,
        cleanup_bundle=lambda *_args, **_kwargs: {"all_objects_absent": True},
        arm_watchdog=lambda **_kwargs: (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            SimpleNamespace(started_instance_id_path=started),
        ),
        close_watchdog=lambda **_kwargs: {"status": "provider_terminal"},
    )

    assert result["status"] == "failed"
    assert result["provider_zero_verified"] is True
    assert result["continuing_spend_from_this_run"] is False
    teardown = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert teardown["status"] == "completed"
    assert teardown["continuing_spend_from_this_run"] is False


def test_sam31_allocator_jit_nonzero_is_terminal_without_run_spend(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.provider_launch_request), {"request": True})
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    write_json(
        Path(args.sam31_attempt_authority),
        {"request_authority_id": "fixture-authority"},
    )
    write_json(Path(args.sam31_input_bundle_receipt), {"receipt": True})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    _write_private(Path(args.sam31_hf_token_file), "hf-secret", mode=0o640)
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.validate_sam31_paid_attempt_authority",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.consume_sam31_paid_attempt_authority_once",
        lambda *_args, **_kwargs: {"status": "consumed"},
    )
    started = tmp_path / "watchdog" / "started.txt"

    def execute(**kwargs):
        canary_root = Path(kwargs["job_dir"])
        canary_root.mkdir(parents=True)
        snapshots = [
            {
                "label": "scoped",
                "name_prefix": "blueprint-sam31-source-tracks-",
                "inventory": {
                    "api_confirmed": True,
                    "live_resource_count": 0,
                    "resources": [],
                },
            },
            {
                "label": "global",
                "name_prefix": "",
                "inventory": {
                    "api_confirmed": True,
                    "live_resource_count": 1,
                    "resources": [{"id": 991, "label": "provider-returned-row"}],
                },
            },
        ]
        receipt = {
            "schema_version": PRELAUNCH_INVENTORY_SCHEMA_VERSION,
            "status": "blocked",
            "blocker": "sam31_provider_not_zero_before_launch",
            "provider": "vast",
            "request_digest": "sha256:" + "1" * 64,
            "bound_request_digest": "sha256:" + "2" * 64,
            "provider_mutations_performed": 0,
            "initial_provider_zero_status": "nonzero",
            "provider_zero_status": "nonzero",
            "inventory_snapshots": snapshots,
            "postfailure_inventory_snapshots": snapshots,
            "postfailure_inventory_digest": canonical_digest(
                {"inventory_snapshots": snapshots},
                digest_field="postfailure_inventory_digest",
            ),
            "captured_at": "2026-08-15T10:54:12Z",
            "raw_secret_values_recorded": False,
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        write_json(canary_root / PRELAUNCH_INVENTORY_RECEIPT_NAME, receipt)
        provider_zero = {
            "provider_zero_status": "nonzero",
            "provider_zero_verified": False,
        }
        provider_zero["provider_zero_receipt_digest"] = canonical_digest(
            provider_zero, digest_field="provider_zero_receipt_digest"
        )
        write_json(canary_root / "provider_zero_verification.json", provider_zero)
        teardown = {
            "status": "FAIL",
            "instance_id": None,
            "provider_mutations_performed": 0,
            "continuing_spend_from_this_run": False,
            "provider_zero_verified": False,
            "provider_zero_status": "nonzero",
            "provider_zero_receipt_digest": provider_zero["provider_zero_receipt_digest"],
        }
        teardown["teardown_receipt_digest"] = canonical_digest(
            teardown, digest_field="teardown_receipt_digest"
        )
        write_json(canary_root / "teardown_receipt.json", teardown)
        raise Sam31VastCanaryError("sam31_provider_not_zero_before_launch")

    def stage(**kwargs):
        root = Path(kwargs["job_dir"])
        root.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            _write_private(root / name, "https://objects.example/value")
        return {"status": "completed", "blockers": []}

    def close_watchdog(**kwargs):
        receipt_path = (
            Path(kwargs["job_dir"])
            / "independent_vast_watchdog"
            / "groot_oscar_runpod_canary_watchdog.json"
        )
        write_json(receipt_path, {"status": "cancelled_no_allocation"})
        return {"status": "cancelled_no_allocation"}

    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **kwargs: (
            write_json(Path(kwargs["bound_request_out"]), {"bound": True})
            or {"status": "execute_ready", "blockers": []}
        ),
        provider_factory=lambda _name: _ReadOnlyProvider(),
        execute_canary=execute,
        stage_bundle=stage,
        cleanup_bundle=lambda *_args, **_kwargs: {"all_objects_absent": True},
        arm_watchdog=lambda **_kwargs: (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            SimpleNamespace(started_instance_id_path=started),
        ),
        close_watchdog=close_watchdog,
    )

    assert result["status"] == "failed"
    assert result["provider_mutations_performed"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert result["provider_zero_verified"] is False
    assert result["provider_zero_status"] == "nonzero"
    assert result["provider_zero_receipt_digest"]
    assert result["independent_watchdog"]["status"] == "cancelled_no_allocation"
    assert "sam31_provider_not_zero_before_launch" in result["blockers"][0]
    assert Path(result["prelaunch_provider_inventory_receipt_path"]).is_file()
    teardown = json.loads(Path(result["teardown_manifest_path"]).read_text())
    assert teardown["status"] == "not_required_provider_adapter_never_invoked"
    assert teardown["continuing_spend_from_this_run"] is False
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text())
    assert "sam31_prelaunch_provider_inventory" in manifest["observed_roles"]
    assert "sam31_provider_zero_verification" in manifest["observed_roles"]


def test_sam31_allocator_lane_dry_run_never_reads_secrets(tmp_path: Path) -> None:
    args = _args(tmp_path, execute=False)
    args.sam31_attempt_authority = None
    Path(args.provider_launch_request).write_text("{}", encoding="utf-8")
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **_kwargs: {"status": "dry_run_ready", "blockers": []},
        provider_factory=lambda _name: (_ for _ in ()).throw(AssertionError("provider accessed")),
        execute_canary=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("canary executed")),
    )
    assert result["status"] == "dry_run_ready"


def test_sam31_allocator_lane_dry_run_types_missing_live_preflight(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, execute=False)
    args.sam31_attempt_authority = None
    Path(args.provider_launch_request).write_text("{}", encoding="utf-8")

    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["sam31_dry_run_preflight_missing_or_unsafe"]
    assert result["provider_mutations_performed"] == 0
    assert result["continuing_spend_from_this_run"] is False
    assert Path(args.adapter_output).is_file()
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()


def test_sam31_allocator_lane_refuses_nonprivate_token_before_provider(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.provider_launch_request), {"request": True})
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    write_json(Path(args.sam31_attempt_authority), {"request_authority_id": "fixture-authority"})
    write_json(Path(args.sam31_input_bundle_receipt), {"receipt": True})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    Path(args.sam31_hf_token_file).write_text("hf-secret", encoding="utf-8")
    Path(args.sam31_hf_token_file).chmod(0o644)
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.validate_sam31_paid_attempt_authority",
        lambda *_args, **_kwargs: {},
    )
    handle = SimpleNamespace(started_instance_id_path=tmp_path / "started.txt")
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **kwargs: (
            write_json(Path(kwargs["bound_request_out"]), {"bound": True})
            or {"status": "execute_ready", "blockers": []}
        ),
        provider_factory=lambda _name: _ReadOnlyProvider(),
        execute_canary=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("canary executed")),
        arm_watchdog=lambda **_kwargs: (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            handle,
        ),
        close_watchdog_without_allocation=lambda **_kwargs: {"status": "provider_terminal"},
    )
    assert result["status"] == "blocked"
    assert "sam31_hf_token_file_permissions_not_0600" in result["blockers"]
    assert result["provider_mutations_performed"] == 0
    assert Path(result["artifact_manifest_path"]).is_file()
    assert Path(result["teardown_manifest_path"]).is_file()
    assert result["continuing_spend_from_this_run"] is False


def test_sam31_allocator_closes_watchdog_when_live_capacity_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.provider_launch_request), {"request": True})
    write_json(Path(args.sam31_attempt_authority), {"request_authority_id": "fixture-authority"})
    write_json(Path(args.sam31_input_bundle_receipt), {"receipt": True})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.validate_sam31_paid_attempt_authority",
        lambda *_args, **_kwargs: {},
    )
    closed: list[bool] = []
    handle = SimpleNamespace(started_instance_id_path=tmp_path / "started.txt")

    class NoCapacity(_ReadOnlyProvider):
        def capacity_preflight(self, _request):
            return {"status": "unavailable", "selected_offer": None}

    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **_kwargs: {
            "status": "blocked",
            "blockers": ["sam31_gpu_provider_api_not_verified"],
        },
        provider_factory=lambda _name: NoCapacity(),
        arm_watchdog=lambda **_kwargs: (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            handle,
        ),
        close_watchdog_without_allocation=lambda **_kwargs: (
            closed.append(True) or {"status": "provider_terminal"}
        ),
    )
    assert result["status"] == "blocked"
    assert "sam31_gpu_provider_api_not_verified" in result["blockers"]
    assert closed == [True]


def test_sam31_allocator_closes_watchdog_when_live_preflight_raises(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.provider_launch_request), {"request": True})
    write_json(Path(args.sam31_attempt_authority), {"request_authority_id": "fixture-authority"})
    write_json(Path(args.sam31_input_bundle_receipt), {"receipt": True})
    Path(args.sam31_input_bundle).write_bytes(b"bundle")
    monkeypatch.setattr(
        "blueprint_pipeline.sam31_paid_resource_allocator_lane.validate_sam31_paid_attempt_authority",
        lambda *_args, **_kwargs: {},
    )
    closed: list[bool] = []
    handle = SimpleNamespace(started_instance_id_path=tmp_path / "started.txt")
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        provider_factory=lambda _name: (_ for _ in ()).throw(RuntimeError("offline")),
        arm_watchdog=lambda **_kwargs: (
            {
                "watchdog_pid": 123,
                "watchdog_started_epoch": 1_000,
                "watchdog_deadline_epoch": 9_999_999_999,
                "pod_name_prefix": "blueprint-sam31-source-tracks-fixture-",
            },
            handle,
        ),
        close_watchdog_without_allocation=lambda **_kwargs: (
            closed.append(True) or {"status": "provider_terminal"}
        ),
    )
    assert result["blockers"] == ["sam31_live_preflight_collection_failed"]
    assert closed == [True]


def test_canonical_allocator_dispatches_sam31_dry_run_without_provider(
    tmp_path: Path, monkeypatch
) -> None:
    observed: dict[str, object] = {}

    def run(args, *, checkout_commit: str):
        observed["args"] = args
        observed["checkout_commit"] = checkout_commit
        return {"status": "dry_run_ready", "blockers": []}

    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: ([], "c" * 40),
    )
    monkeypatch.setattr(allocator, "run_sam31_paid_resource_allocator_lane", run)
    result = allocator.main(
        [
            "gpu-canary",
            "--probe-kind",
            "semantic-sam31-source-tracks",
            "--provider",
            "vast",
            "--provider-launch-request",
            str(tmp_path / "request.json"),
            "--preflight-bundle",
            str(tmp_path / "preflight.json"),
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--expected-source-commit",
            "c" * 40,
            "--sam31-max-spend-usd",
            "1.0",
            "--sam31-hard-ttl-seconds",
            "600",
            "--sam31-retry-cap",
            "0",
            "--sam31-authority-id",
            "fixture-authority",
        ]
    )
    assert result == 0
    assert observed["checkout_commit"] == "c" * 40
    assert observed["args"].execute is False


def test_canonical_allocator_refuses_sam31_execute_without_private_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: ([], "c" * 40),
    )
    monkeypatch.setattr(
        allocator,
        "run_sam31_paid_resource_allocator_lane",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lane reached with missing secret files")
        ),
    )
    result = allocator.main(
        [
            "gpu-canary",
            "--probe-kind",
            "semantic-sam31-source-tracks",
            "--provider-launch-request",
            str(tmp_path / "request.json"),
            "--preflight-bundle",
            str(tmp_path / "preflight.json"),
            "--admission-out",
            str(tmp_path / "admission.json"),
            "--bound-request-out",
            str(tmp_path / "bound.json"),
            "--adapter-output",
            str(tmp_path / "adapter.json"),
            "--expected-source-commit",
            "c" * 40,
            "--sam31-max-spend-usd",
            "1.0",
            "--sam31-hard-ttl-seconds",
            "600",
            "--sam31-authority-id",
            "fixture-authority",
            "--execute",
        ]
    )
    assert result == 2
    blocked = json.loads((tmp_path / "adapter.json").read_text())
    assert "sam31_attempt_authority_missing" in blocked["blockers"]
    assert blocked["provider_mutations_performed"] == 0
