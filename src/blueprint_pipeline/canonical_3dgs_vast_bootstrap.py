"""Execute one canonical Splatfacto arm inside an admitted Vast worker."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .canonical_3dgs_admission import build_canonical_3dgs_worker_admission
from .canonical_3dgs_transport import extract_canonical_3dgs_transport_bundle
from .canonical_3dgs_vast_output import compile_canonical_3dgs_vast_output_bundle
from .canonical_3dgs_worker import main as run_worker_main
from .common import utc_now_iso, write_json
from .safe_outbound_http import presigned_transfer_policy, upload_file


SPLATFACTO_RUNTIME_DIGEST = (
    "sha256:913d5afd190a9bed736f6a978d472b58654f650d3bc173a07d8a5375d95703c6"
)


class Canonical3DGSVastBootstrapError(ValueError):
    pass


def _required(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name) or "")
    if not value:
        raise Canonical3DGSVastBootstrapError(f"canonical_vast_env_missing:{name}")
    return value


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise Canonical3DGSVastBootstrapError("canonical_vast_json_not_object")
    return dict(value)


def run_canonical_3dgs_vast_bootstrap(
    *, environment: Mapping[str, str], work_root: str | Path
) -> dict[str, Any]:
    """Revalidate transport/authority, train, package, and upload one candidate."""

    root = Path(work_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(_required(environment, "BLUEPRINT_CANONICAL_BUNDLE_PATH"))
    receipt_path = Path(_required(environment, "BLUEPRINT_CANONICAL_RECEIPT_PATH"))
    output_url = _required(environment, "BLUEPRINT_RECONSTRUCTION_OUTPUT_BUNDLE_PUT_URL")
    image = _required(environment, "BLUEPRINT_CONTAINER_IMAGE_DIGEST")
    authority = _required(environment, "BLUEPRINT_CANONICAL_AUTHORITY_ID")
    transport = _json(receipt_path)
    allocator = json.loads(
        base64.b64decode(
            _required(environment, "BLUEPRINT_CANONICAL_ALLOCATOR_ADMISSION_B64"),
            validate=True,
        )
    )
    if not isinstance(allocator, Mapping):
        raise Canonical3DGSVastBootstrapError(
            "canonical_vast_allocator_admission_not_object"
        )
    max_spend = float(_required(environment, "BLUEPRINT_CANONICAL_MAX_SPEND_USD"))
    hard_ttl = int(_required(environment, "BLUEPRINT_CANONICAL_HARD_TTL_SECONDS"))
    admission = build_canonical_3dgs_worker_admission(
        transport_receipt=transport,
        arm_id="splatfacto-comparison",
        worker_platform="linux",
        paid_allocator_admission=allocator,
        worker_image_digest=image,
        trainer_runtime_digest=SPLATFACTO_RUNTIME_DIGEST,
        trainer_runtime_version="nerfstudio-1.1.5+gsplat-1.4.0",
        authority_id=authority,
        max_spend_usd=max_spend,
        hard_ttl_seconds=hard_ttl,
        provider_upload_authorized=True,
        paid_compute_authorized=True,
        watchdog_armed=True,
        provider_zero_before_allocation=True,
        timestamp=utc_now_iso(),
    )
    admission_path = root / "canonical_3dgs_worker_admission.json"
    write_json(admission_path, admission)
    extraction = extract_canonical_3dgs_transport_bundle(
        bundle_path=bundle_path,
        receipt=transport,
        output_root=root / "materialized",
    )
    materialized = (
        root
        / "materialized"
        / extraction["transport_bundle_digest"].removeprefix("sha256:")
    )
    result_root = root / "results" / "splatfacto-comparison"
    result_root.mkdir(parents=True, exist_ok=True)
    write_json(result_root / "paid_allocator_admission.json", allocator)
    worker_receipt_path = result_root / "worker_receipt.json"
    os.environ["BLUEPRINT_WORKER_IMAGE_DIGEST"] = image
    exit_code = run_worker_main(
        [
            "--arm",
            "splatfacto-comparison",
            "--plan",
            str(materialized / "campaign/canonical_3dgs_execution_plan.json"),
            "--dataset-root",
            str(materialized / "campaign/dataset"),
            "--output-root",
            str(result_root),
            "--receipt",
            str(worker_receipt_path),
            "--transport-receipt",
            str(receipt_path),
            "--admission",
            str(admission_path),
        ]
    )
    if exit_code != 0:
        raise Canonical3DGSVastBootstrapError(
            "canonical_vast_splatfacto_execution_failed"
        )
    worker_receipt = _json(worker_receipt_path)
    output_path = root / "canonical_3dgs_vast_output.zip"
    output_receipt = compile_canonical_3dgs_vast_output_bundle(
        result_root=result_root,
        worker_receipt=worker_receipt,
        output_path=output_path,
        worker_image_digest=image,
        source_commit_sha=_required(environment, "BLUEPRINT_SOURCE_COMMIT"),
    )
    transfer = upload_file(
        output_url,
        input_path=output_path,
        expected_sha256=output_receipt["operation_output_bundle_digest"],
        max_bytes=96 * 1024**3,
        timeout_seconds=600,
        policy=presigned_transfer_policy(output_url),
        content_type="application/zip",
    )
    result = {
        "schema_version": "canonical_3dgs_vast_bootstrap.v1",
        "status": "uploaded",
        "canonical_3dgs_worker_receipt_digest": worker_receipt[
            "canonical_3dgs_worker_receipt_digest"
        ],
        "operation_output_bundle_digest": transfer.sha256,
        "provider_zero_verified": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "appearance_asset_candidate_only",
    }
    write_json(root / "canonical_3dgs_vast_bootstrap.json", result)
    return result


def main() -> int:
    run_canonical_3dgs_vast_bootstrap(
        environment=os.environ,
        work_root=os.environ.get("BLUEPRINT_CANONICAL_WORK_ROOT", "/workspace/canonical"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Canonical3DGSVastBootstrapError",
    "SPLATFACTO_RUNTIME_DIGEST",
    "run_canonical_3dgs_vast_bootstrap",
]
