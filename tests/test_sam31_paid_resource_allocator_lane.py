from argparse import Namespace
from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.sam31_paid_resource_allocator_lane import (
    run_sam31_paid_resource_allocator_lane,
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
        sam31_hard_ttl_seconds=300,
        sam31_retry_cap=0,
        sam31_authority_id="fixture-authority",
        sam31_hf_token_file=str(tmp_path / "hf-token.txt"),
        provider_bundle_url_file=str(tmp_path / "input-url.txt"),
        provider_output_put_url_file=str(tmp_path / "put-url.txt"),
        provider_output_get_url_file=str(tmp_path / "get-url.txt"),
        execute=execute,
    )


def _write_private(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def test_sam31_allocator_lane_routes_exact_private_inputs(tmp_path: Path) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    _write_private(Path(args.sam31_hf_token_file), "hf-secret")
    _write_private(Path(args.provider_bundle_url_file), "https://objects.example/input")
    _write_private(Path(args.provider_output_put_url_file), "https://objects.example/put")
    _write_private(Path(args.provider_output_get_url_file), "https://objects.example/get")
    observed: dict[str, object] = {}

    def prepare(**kwargs):
        observed["prepare"] = kwargs
        write_json(Path(kwargs["bound_request_out"]), {"bound": True})
        return {"status": "execute_ready", "blockers": []}

    def execute(**kwargs):
        observed["execute"] = kwargs
        return {
            "status": "completed",
            "provider_mutations_performed": 2,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        }

    provider = object()
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=prepare,
        provider_factory=lambda _name: provider,
        execute_canary=execute,
    )
    assert result["status"] == "completed"
    assert observed["prepare"]["execution_adapter_qualified"] is True
    assert observed["execute"]["provider"] is provider
    assert observed["execute"]["hf_token"] == "hf-secret"
    assert observed["execute"]["input_bundle_get_url"].endswith("/input")
    assert observed["execute"]["paid_resource_admission_grant"].resource_class == "gpu_render"


def test_sam31_allocator_lane_dry_run_never_reads_secrets(tmp_path: Path) -> None:
    args = _args(tmp_path, execute=False)
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **_kwargs: {"status": "dry_run_ready", "blockers": []},
        provider_factory=lambda _name: (_ for _ in ()).throw(AssertionError("provider accessed")),
        execute_canary=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("canary executed")),
    )
    assert result["status"] == "dry_run_ready"


def test_sam31_allocator_lane_refuses_nonprivate_token_before_provider(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, execute=True)
    write_json(Path(args.preflight_bundle), {"provider": "vast"})
    Path(args.sam31_hf_token_file).write_text("hf-secret", encoding="utf-8")
    Path(args.sam31_hf_token_file).chmod(0o644)
    for path, value in (
        (Path(args.provider_bundle_url_file), "https://objects.example/input"),
        (Path(args.provider_output_put_url_file), "https://objects.example/put"),
        (Path(args.provider_output_get_url_file), "https://objects.example/get"),
    ):
        _write_private(path, value)
    result = run_sam31_paid_resource_allocator_lane(
        args,
        checkout_commit="c" * 40,
        prepare=lambda **kwargs: (
            write_json(Path(kwargs["bound_request_out"]), {"bound": True})
            or {"status": "execute_ready", "blockers": []}
        ),
        provider_factory=lambda _name: (_ for _ in ()).throw(AssertionError("provider accessed")),
        execute_canary=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("canary executed")),
    )
    assert result["status"] == "blocked"
    assert "sam31_hf_token_file_permissions_not_0600" in result["blockers"]
    assert result["provider_mutations_performed"] == 0
