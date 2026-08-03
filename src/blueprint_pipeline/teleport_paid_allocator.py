"""Canonical paid-allocator seam for the Teleport provider adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .reconstruction_provider_contracts import (
    require_reconstruction_provider_execution_authority,
)
from .teleport_reconstruction_adapter import (
    CLIENT_ID_FILE_ENV as TELEPORT_CLIENT_ID_FILE_ENV,
    CLIENT_SECRET_FILE_ENV as TELEPORT_CLIENT_SECRET_FILE_ENV,
    PUBLIC_SPEND_CAP_ENV as TELEPORT_PUBLIC_SPEND_CAP_ENV,
    PUBLIC_UPLOAD_AUTH_ENV as TELEPORT_PUBLIC_UPLOAD_AUTH_ENV,
    TELEPORT_RESOURCE_CLASS,
    TeleportAdapterError,
    build_teleport_sealed_evaluation_runner,
    load_teleport_credentials,
    run_teleport_reconstruction,
    validate_teleport_terms_review,
    validate_teleport_upload_packet,
)


JsonLoader = Callable[[Path], dict[str, Any]]
SourceCheckoutValidator = Callable[..., tuple[list[str], str]]
CredentialLoader = Callable[[Mapping[str, str]], Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def add_teleport_provider_arguments(commands: Any, *, root: Path) -> None:
    provider = commands.add_parser("provider-reconstruction")
    provider.add_argument("--provider", choices=("teleport",), default="teleport")
    provider.add_argument("--upload-packet", required=True)
    provider.add_argument("--execution-request", required=True)
    provider.add_argument("--candidate-observations", required=True)
    provider.add_argument("--sealed-evaluation-request")
    provider.add_argument("--output-dir", required=True)
    provider.add_argument(
        "--terms-review",
        default=str(root / "docs/evidence/teleport_provider_terms_review_2026-08-03.json"),
    )
    provider.add_argument(
        "--teleport-client-id-file",
        default="~/.blueprint-secrets/teleport_client_id",
    )
    provider.add_argument(
        "--teleport-client-secret-file",
        default="~/.blueprint-secrets/teleport_client_secret",
    )
    provider.add_argument("--maximum-ply-bytes", type=int, default=4_000_000_000)
    provider.add_argument("--poll-interval-seconds", type=float, default=30.0)
    provider.add_argument("--observed-at", help=argparse.SUPPRESS)
    provider.add_argument("--experimental-branch-diagnostic", action="store_true")
    provider.add_argument("--execute", action="store_true")


def run_teleport_provider(
    args: argparse.Namespace,
    *,
    load_json: JsonLoader,
    source_checkout_blockers: SourceCheckoutValidator,
    credential_loader: CredentialLoader = load_teleport_credentials,
) -> dict[str, Any]:
    """Run the fail-closed Teleport route; dry-run never loads credentials."""

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    request: dict[str, Any] = {}
    packet_path = Path(args.upload_packet).expanduser().resolve()
    try:
        request = load_json(Path(args.execution_request).expanduser().resolve())
        request = require_reconstruction_provider_execution_authority(
            request, at_time=args.observed_at or _utc_now()
        )
    except Exception as exc:
        blockers.append("teleport_execution_request_invalid:" + type(exc).__name__)
    try:
        packet = validate_teleport_upload_packet(
            load_json(packet_path), packet_root=packet_path.parent
        )
    except Exception as exc:
        blockers.append("teleport_upload_packet_invalid:" + type(exc).__name__)
        packet = {}
    try:
        terms = validate_teleport_terms_review(
            load_json(Path(args.terms_review).expanduser().resolve())
        )
    except Exception as exc:
        blockers.append("teleport_terms_review_invalid:" + type(exc).__name__)
        terms = {}
    try:
        observations_value = load_json(
            Path(args.candidate_observations).expanduser().resolve()
        )
        observations = (
            observations_value.get("observations")
            if isinstance(observations_value, Mapping)
            else observations_value
        )
        if not isinstance(observations, list) or not observations:
            raise ValueError("candidate observations missing")
    except Exception as exc:
        blockers.append("teleport_candidate_observations_invalid:" + type(exc).__name__)
        observations = []
    if request:
        source_blockers, _checkout_commit = source_checkout_blockers(
            str(request.get("source_commit_sha") or ""),
            allow_pushed_branch_diagnostic=args.experimental_branch_diagnostic,
        )
        blockers.extend("teleport_" + item for item in source_blockers)
        if request.get("provider_identity") != "teleport":
            blockers.append("teleport_execution_provider_mismatch")
        if terms and request.get("provider_admission", {}).get("terms_digest") != terms.get(
            "teleport_provider_terms_review_digest"
        ):
            blockers.append("teleport_terms_review_not_bound_to_provider_admission")
    if packet and terms:
        recorded_terms = packet.get("teleport_provider_terms_review_digest")
        if recorded_terms is not None and recorded_terms != terms.get(
            "teleport_provider_terms_review_digest"
        ):
            blockers.append("teleport_packet_terms_review_mismatch")
    if args.maximum_ply_bytes < 1024 * 1024 or args.maximum_ply_bytes > 16_000_000_000:
        blockers.append("teleport_maximum_ply_bytes_out_of_bounds")
    if args.poll_interval_seconds < 5.0 or args.poll_interval_seconds > 300.0:
        blockers.append("teleport_poll_interval_out_of_bounds")
    execute_blockers: list[str] = []
    sealed_evaluation_runner = None
    if args.execute:
        if os.environ.get(TELEPORT_PUBLIC_UPLOAD_AUTH_ENV, "").strip().lower() != "true":
            execute_blockers.append("teleport_public_data_upload_interlock_missing")
        raw_cap = os.environ.get(TELEPORT_PUBLIC_SPEND_CAP_ENV, "").strip()
        try:
            environment_cap = float(raw_cap)
        except ValueError:
            environment_cap = -1.0
        if not request or environment_cap != float(request.get("max_cost_usd") or -2.0):
            execute_blockers.append("teleport_public_data_spend_cap_interlock_mismatch")
        if not args.sealed_evaluation_request:
            execute_blockers.append("teleport_sealed_evaluation_request_missing")
        else:
            try:
                evaluation_config = load_json(
                    Path(args.sealed_evaluation_request).expanduser().resolve()
                )
                sealed_evaluation_runner = build_teleport_sealed_evaluation_runner(
                    evaluation_config
                )
            except Exception as exc:
                execute_blockers.append(
                    "teleport_sealed_evaluation_request_invalid:" + type(exc).__name__
                )
    paid_admission = build_paid_lane_admission(
        resource_class=TELEPORT_RESOURCE_CLASS,
        blockers=[*blockers, *execute_blockers],
    )
    write_json(output / "teleport_paid_lane_admission.json", paid_admission)
    preflight = {
        "schema_version": "teleport_provider_preflight.v1",
        "status": "blocked" if blockers or execute_blockers else "ready",
        "execute_requested": bool(args.execute),
        "provider_identity": "teleport",
        "provider_execution_request_digest": request.get(
            "provider_execution_request_digest"
        ),
        "upload_packet_digest": packet.get("teleport_t1_upload_packet_digest")
        or packet.get("teleport_ready_to_upload_packet_digest"),
        "terms_review_digest": terms.get("teleport_provider_terms_review_digest"),
        "blockers": sorted(set([*blockers, *execute_blockers])),
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "proof_effect": "pre_provider_mutation_admission_only",
        "claim_ceiling": "none",
    }
    preflight["teleport_provider_preflight_digest"] = canonical_digest(
        preflight, digest_field="teleport_provider_preflight_digest"
    )
    write_json(output / "teleport_provider_preflight.v1.json", preflight)
    if not args.execute:
        return {
            **preflight,
            "status": "dry_run_ready" if not blockers else "blocked",
        }
    if blockers or execute_blockers:
        return preflight
    try:
        grant = require_paid_resource_admission(
            paid_admission,
            resource_class=TELEPORT_RESOURCE_CLASS,
            expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
        )
        credentials = credential_loader(
            {
                TELEPORT_CLIENT_ID_FILE_ENV: args.teleport_client_id_file,
                TELEPORT_CLIENT_SECRET_FILE_ENV: args.teleport_client_secret_file,
            }
        )
        assert sealed_evaluation_runner is not None
        return run_teleport_reconstruction(
            upload_packet_path=packet_path,
            execution_request=request,
            candidate_observations=observations,
            output_root=output,
            paid_resource_admission_grant=grant,
            credentials=credentials,
            sealed_evaluation_runner=sealed_evaluation_runner,
            maximum_ply_bytes=args.maximum_ply_bytes,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except TeleportAdapterError as exc:
        return exc.result or {
            "schema_version": "teleport_provider_run_receipt.v1",
            "status": "failed",
            "failure_codes": list(exc.codes),
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        }
