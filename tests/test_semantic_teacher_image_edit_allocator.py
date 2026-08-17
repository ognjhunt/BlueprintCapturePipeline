from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import blueprint_pipeline.paid_resource_allocator as allocator


SOURCE_COMMIT = "c" * 40
IMAGE = "registry.example/semantic-teacher@sha256:" + "d" * 64


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _base_args(tmp_path: Path) -> list[str]:
    authority = tmp_path / "authority.json"
    receipt = tmp_path / "bundle-receipt.json"
    bundle = tmp_path / "bundle.zip"
    _write(
        authority,
        {
            "authorization_digest": "sha256:" + "a" * 64,
            "backend_entry_digest": "sha256:" + "b" * 64,
            "task_count": 2,
            "camera_count": 16,
            "maximum_hourly_rate_usd": 0.5,
            "hard_ttl_seconds": 600,
        },
    )
    _write(
        receipt,
        {
            "bundle": {"sha256": "sha256:" + "e" * 64, "size_bytes": 6},
        },
    )
    bundle.write_bytes(b"bundle")
    return [
        "gpu-canary",
        "--probe-kind",
        "semantic-teacher-image-edit",
        "--provider",
        "vast",
        "--expected-source-commit",
        SOURCE_COMMIT,
        "--semantic-teacher-bundle",
        str(bundle),
        "--semantic-teacher-bundle-receipt",
        str(receipt),
        "--semantic-teacher-attempt-authority",
        str(authority),
        "--semantic-teacher-runtime-image-identity",
        IMAGE,
        "--semantic-teacher-job-dir",
        str(tmp_path / "job"),
    ]


class _InventoryProvider:
    def billable_inventory(self, *, name_prefix: str) -> dict:
        return {
            "api_confirmed": True,
            "live_resource_count": 0,
            "resources": [],
            "name_prefix": name_prefix,
        }


def test_allocator_routes_semantic_teacher_dry_run_without_watchdog_or_token(
    tmp_path: Path, monkeypatch
) -> None:
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: ([], SOURCE_COMMIT),
    )
    monkeypatch.setattr(allocator, "get_render_provider", lambda _name: _InventoryProvider())

    def prepare(**kwargs):
        observed.update(kwargs)
        return {"status": "dry_run_ready", "provider_mutations_performed": 0}

    monkeypatch.setattr(
        allocator, "prepare_semantic_teacher_image_edit_allocator_dry_run", prepare
    )
    monkeypatch.setattr(
        allocator,
        "arm_independent_vast_watchdog",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run armed a watchdog")
        ),
    )
    result = allocator.main(
        [
            *_base_args(tmp_path),
            "--semantic-teacher-dry-run-output",
            str(tmp_path / "dry-run.json"),
        ]
    )

    assert result == 0
    assert observed["checkout_source_commit"] == SOURCE_COMMIT
    assert observed["live_inventory"]["live_resource_count"] == 0


def test_allocator_refuses_execute_without_bound_dry_run_and_token(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: ([], SOURCE_COMMIT),
    )
    monkeypatch.setattr(
        allocator,
        "run_semantic_teacher_image_edit_vast",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("paid adapter reached without private gates")
        ),
    )
    output = tmp_path / "adapter.json"
    result = allocator.main(
        [*_base_args(tmp_path), "--adapter-output", str(output), "--execute"]
    )

    assert result == 2
    blocked = json.loads(output.read_text(encoding="utf-8"))
    assert "semantic_teacher_token_file_missing" in blocked["blockers"]
    assert "semantic_teacher_dry_run_receipt_missing" in blocked["blockers"]
    assert blocked["provider_mutations_performed"] == 0


def test_semantic_capacity_preflight_uses_authority_and_openai_geography(
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    class Provider:
        def build_request(self, spec, job_dir):
            observed["spec"] = spec
            observed["job_dir"] = job_dir
            return {
                "max_hourly_rate_usd": spec.max_hourly_rate_usd,
                "min_gpu_ram_mb": spec.min_gpu_ram_mb,
                "allowed_geolocation_country_codes": list(
                    spec.allowed_geolocation_country_codes
                ),
            }

        def capacity_preflight(self, request):
            observed["request"] = request
            return {
                "status": "available",
                "selected_offer": {
                    "hourly_rate_usd": 0.24,
                    "gpu_ram_mb": 48_000,
                },
            }

    result = allocator._semantic_teacher_capacity_preflight(
        provider=Provider(),
        authority={"maximum_hourly_rate_usd": 0.40},
        runtime_image_identity="registry.example/teacher@sha256:" + "a" * 64,
        job_dir=tmp_path,
        watchdog={"status": "armed"},
        excluded_machine_ids=(76546,),
    )

    spec = observed["spec"]
    assert spec.max_hourly_rate_usd == pytest.approx(0.40)
    assert spec.min_gpu_ram_mb == 16_000
    assert spec.excluded_machine_ids == (76546,)
    assert "us" in spec.allowed_geolocation_country_codes
    assert "cn" not in spec.allowed_geolocation_country_codes
    assert observed["request"]["allowed_geolocation_country_codes"]
    assert result["status"] == "ready"
    assert result["on_demand_price_usd_per_hour"] == pytest.approx(0.24)
    assert result["gpu_memory_bytes"] == 48_000_000_000


def test_allocator_execute_arms_watchdog_then_routes_exact_adapter(
    tmp_path: Path, monkeypatch
) -> None:
    args = _base_args(tmp_path)
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    token.chmod(0o600)
    admin_key = tmp_path / "admin-key"
    admin_key.write_text("sk-admin-fixture", encoding="utf-8")
    admin_key.chmod(0o600)
    cost_scope = tmp_path / "cost-scope.json"
    _write(cost_scope, {"fixture": True})
    dry_run = tmp_path / "dry-run.json"
    _write(dry_run, {"fixture": True})
    preflight_output = tmp_path / "preflight.json"
    adapter_output = tmp_path / "adapter.json"
    observed: dict[str, object] = {}
    handle = SimpleNamespace(started_instance_id_path=tmp_path / "started-instance")
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: ([], SOURCE_COMMIT),
    )
    monkeypatch.setattr(
        allocator, "_semantic_teacher_dry_run_binding_valid", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(allocator, "get_render_provider", lambda _name: _InventoryProvider())
    def arm(**kwargs):
        observed["arm"] = kwargs
        return (
            {
                "status": "armed",
                "watchdog_pid": 123,
                "watchdog_deadline_epoch": 9999999999,
                "pod_name_prefix": "blueprint-semantic-teacher-fixture-",
            },
            handle,
        )

    monkeypatch.setattr(allocator, "arm_independent_vast_watchdog", arm)

    def capacity(**kwargs):
        observed["capacity"] = kwargs
        return {
            "status": "ready",
            "watchdog": kwargs["watchdog"],
            "on_demand_price_usd_per_hour": 0.25,
            "gpu_memory_bytes": 16_000_000_000,
            "container_disk_bytes": 32 * 1024**3,
        }

    monkeypatch.setattr(allocator, "_semantic_teacher_capacity_preflight", capacity)
    def close(**kwargs):
        observed["close"] = kwargs
        return {"status": "provider_terminal"}

    monkeypatch.setattr(allocator, "close_independent_vast_watchdog", close)

    def run_adapter(namespace, **kwargs):
        observed["namespace"] = namespace
        observed["adapter"] = kwargs
        kwargs["watchdog_closer"](
            instance_ids=[],
            provider_teardown_completed=True,
            provider_allocation_impossible=True,
        )
        return {"status": "completed", "provider_mutations_performed": 2}

    monkeypatch.setattr(allocator, "run_semantic_teacher_image_edit_vast", run_adapter)
    result = allocator.main(
        [
            *args,
            "--semantic-teacher-excluded-machine-id",
            "76546",
            "--semantic-teacher-token-file",
            str(token),
            "--semantic-teacher-openai-cost-scope-attestation",
            str(cost_scope),
            "--semantic-teacher-openai-admin-api-key-file",
            str(admin_key),
            "--semantic-teacher-openai-project-id",
            "proj_fixture",
            "--semantic-teacher-openai-api-key-id",
            "key_fixture",
            "--semantic-teacher-dry-run-receipt",
            str(dry_run),
            "--semantic-teacher-preflight-output",
            str(preflight_output),
            "--adapter-output",
            str(adapter_output),
            "--execute",
        ]
    )

    assert result == 0
    assert observed["arm"]["pod_name_prefix"] == "blueprint-semantic-teacher-"
    assert observed["capacity"]["excluded_machine_ids"] == [76546]
    assert observed["adapter"]["checkout_commit"] == SOURCE_COMMIT
    assert observed["close"]["provider_allocation_impossible"] is True
    assert json.loads(adapter_output.read_text(encoding="utf-8"))["status"] == "completed"


def test_allocator_refuses_invalid_semantic_teacher_machine_exclusion_before_provider(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        allocator,
        "_source_checkout_blockers",
        lambda *_args, **_kwargs: ([], SOURCE_COMMIT),
    )
    monkeypatch.setattr(
        allocator,
        "get_render_provider",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("invalid exclusion reached provider")
        ),
    )
    output = tmp_path / "adapter.json"

    result = allocator.main(
        [
            *_base_args(tmp_path),
            "--semantic-teacher-excluded-machine-id",
            "0",
            "--semantic-teacher-dry-run-output",
            str(tmp_path / "dry-run.json"),
            "--adapter-output",
            str(output),
        ]
    )

    assert result == 2
    blocked = json.loads(output.read_text(encoding="utf-8"))
    assert "semantic_teacher_excluded_machine_id_invalid" in blocked["blockers"]
    assert blocked["provider_mutations_performed"] == 0
