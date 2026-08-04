#!/usr/bin/env python3
"""Prepare immutable Lightwheel sink USD, bundle, receipt, and guarded GPU request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.lightwheel_sink_isaac_bundle import (
    compile_lightwheel_sink_isaac_input_bundle,
    derivative_wrapper_usda,
)
from blueprint_pipeline.measurement_isaac_runtime_release import (
    RUNTIME_IMAGE,
    build_measurement_isaac_runtime_release,
)
from blueprint_pipeline.reconstruction_gpu_admission import (
    build_reconstruction_gpu_canary_request,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--textures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-spend-usd", type=float, required=True)
    parser.add_argument("--hard-ttl-seconds", type=int, required=True)
    parser.add_argument("--authority-id", required=True)
    parser.add_argument("--vast-gpu-keyword", action="append", default=[])
    parser.add_argument("--authorize-remote-upload", action="store_true")
    parser.add_argument("--authorize-paid-compute", action="store_true")
    args = parser.parse_args()
    if not args.authorize_remote_upload or not args.authorize_paid_compute:
        parser.error("both --authorize-remote-upload and --authorize-paid-compute are required")
    if args.max_spend_usd <= 0 or args.hard_ttl_seconds <= 0:
        parser.error("positive spend and TTL bounds are required")

    root = args.repo_root.expanduser().resolve()
    model = args.model.expanduser().resolve()
    textures = args.textures.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_before = _sha256(model)
    texture_before = {
        path.relative_to(textures).as_posix(): _sha256(path)
        for path in sorted(textures.rglob("*"))
        if path.is_file()
    }
    bundle_path = output / "lightwheel_sink_isaac_input.zip"
    receipt = compile_lightwheel_sink_isaac_input_bundle(
        repo_root=root,
        model_path=model,
        textures_path=textures,
        output_path=bundle_path,
    )
    wrapper_path = output / "lightwheel_sink_test.usda"
    wrapper_path.write_text(derivative_wrapper_usda(), encoding="utf-8")
    runtime_release = build_measurement_isaac_runtime_release()
    request = build_reconstruction_gpu_canary_request(
        {
            "schema_version": "reconstruction_gpu_canary_request.v1",
            "operation": "measurement_isaac_canary",
            "capture_profile": "external_generated_asset",
            "source_commit_sha": receipt["source_commit_sha"],
            "worker_image_digest": RUNTIME_IMAGE,
            "worker_stack_manifest_digest": runtime_release["runtime_release_digest"],
            "deterministic_configuration_digest": receipt["test_configuration_digest"],
            "operation_request_digest": receipt["bundle_manifest_digest"],
            "operation_input_bundle_digest": receipt["input_bundle_digest"],
            "source_model_digest": receipt["source_model_digest"],
            "texture_manifest_digest": receipt["texture_manifest_digest"],
            "wrapper_digest": receipt["wrapper_digest"],
            "test_configuration_digest": receipt["test_configuration_digest"],
            "expected_runtime_result_schema": "lightwheel_sink_isaac_runtime_result.v1",
            "source_relationship_to_blueprint_raw_capture": "none",
            "external_derived_support_asset": True,
            "blueprint_raw_capture_truth": False,
            "remote_upload_authorized": True,
            "paid_compute_authorized": True,
            "candidate_may_read_hidden_heldout": False,
            "trainer_may_grade_heldout": False,
            "max_spend_usd": args.max_spend_usd,
            "hard_ttl_seconds": args.hard_ttl_seconds,
            "retry_cap": 0,
            "authority_id": args.authority_id,
            "vast_preferred_gpu_keywords": args.vast_gpu_keyword or ["L40", "RTX 4090"],
            "proof_effect": "none",
        }
    )
    model_after = _sha256(model)
    texture_after = {
        path.relative_to(textures).as_posix(): _sha256(path)
        for path in sorted(textures.rglob("*"))
        if path.is_file()
    }
    digest_receipt = {
        "schema_version": "lightwheel_sink_source_digest_preservation.v1",
        "source_model_path": str(model),
        "source_model_digest_before": model_before,
        "source_model_digest_after": model_after,
        "texture_root": str(textures),
        "texture_digests_before": texture_before,
        "texture_digests_after": texture_after,
        "source_assets_preserved": model_before == model_after and texture_before == texture_after,
        "source_asset_modified": False,
    }
    write_json(output / "lightwheel_sink_bundle_receipt.json", receipt)
    write_json(output / "measurement_isaac_runtime_release.json", runtime_release)
    write_json(output / "lightwheel_sink_gpu_request.json", request)
    write_json(output / "source_digest_preservation.json", digest_receipt)
    print(
        json.dumps(
            {
                "status": "prepared",
                "output_dir": str(output),
                "input_bundle_digest": receipt["input_bundle_digest"],
                "source_assets_preserved": digest_receipt["source_assets_preserved"],
                "request_digest": request["request_digest"],
                "provider_mutations_performed": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
