from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.external_provider_nurec import ExternalProviderNuRecError
from blueprint_pipeline.provider_nurec_isaac_preparation import (
    main,
    materialize_provider_nurec_isaac_request,
)


D = ["sha256:" + character * 64 for character in "abcdef"]


def _template() -> dict:
    return {
        "stable_run_identity": "provider-nurec-preparation-test",
        "package_digest": D[0],
        "package_artifact_reference": "source/ethel_sim.usdz",
        "external_import_receipt_digest": D[1],
        "qualification_report_digest": D[2],
        "fixed_camera_spec_digest": D[3],
        "fixed_camera_ids": ["probe-near"],
        "runtime_implementation_digest": D[4],
        "runtime_container_image_digest": "registry.test/isaac@" + D[5],
        "expected_prim_paths": {
            "appearance": "/World/gauss/gauss",
            "collision": "/World/gauss/mesh",
        },
        "physics_probe_request": {
            "ground_collider_prim": "/World/gauss/mesh",
            "ground_height_m": 0.0,
            "probe_xy_m": [1.0, 2.0],
            "selection_status": "cpu_geometry_candidate_unverified_in_isaac",
            "manufacture_ground_plane": False,
            "require_contact_event": True,
            "steps": 240,
        },
        "timeout_seconds": 3600,
        "spend_controls": {
            "authorized": False,
            "estimated_max_spend_usd": 2.0,
            "hard_ttl_seconds": 3600,
            "teardown_required": True,
            "provider_zero_required_before_and_after": True,
        },
        "provider_authored_package": True,
        "exact_package_required": True,
        "headless": True,
        "display_attached": False,
        "execution_status": "awaiting_explicit_paid_runtime_authorization",
        "provider_allocation_performed": False,
        "expected_runtime_schema": "provider_nurec_isaac_runtime_result.v1",
        "proof_effect": "none",
        "claim_ceiling": "request_only",
    }


def test_preparation_cli_uses_real_checkout_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[1]
    template_path = tmp_path / "template.json"
    template_path.write_text(json.dumps(_template()), encoding="utf-8")
    output = tmp_path / "request.json"
    monkeypatch.setattr(
        "blueprint_pipeline.external_provider_nurec.subprocess.run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {
                "stdout": (
                    str(checkout)
                    if "--show-toplevel" in args[0]
                    else "1" * 40
                    if "HEAD" in args[0]
                    else ""
                )
            },
        )(),
    )
    assert (
        main(
            [
                "--request-template",
                str(template_path),
                "--source-checkout",
                str(checkout),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    request = json.loads(output.read_text(encoding="utf-8"))
    assert request["source_commit_sha"] == "1" * 40
    assert request["provider_allocation_performed"] is False


def test_preparation_cli_blocks_invalid_template(tmp_path: Path) -> None:
    template_path = tmp_path / "template.json"
    template_path.write_text("[]", encoding="utf-8")
    output = tmp_path / "request.json"
    assert (
        main(
            [
                "--request-template",
                str(template_path),
                "--source-checkout",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_preparation_refuses_symlink_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    output = tmp_path / "request.json"
    output.symlink_to(target)
    monkeypatch.setattr(
        "blueprint_pipeline.provider_nurec_isaac_preparation.build_provider_nurec_isaac_request_from_checkout",
        lambda *args, **kwargs: {"source_commit_sha": "1" * 40},
    )
    with pytest.raises(
        ExternalProviderNuRecError,
        match="provider_isaac_request_output_symlink_forbidden",
    ):
        materialize_provider_nurec_isaac_request(
            request_template=_template(),
            source_checkout=checkout,
            output_path=output,
        )
