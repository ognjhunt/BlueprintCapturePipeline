"""No-data Codex advisory reviews for a Content Agents candidate matrix.

Codex can review exact input identity, declared material/physics scope, and a
prepared bundle without replacing NVIDIA's released material/texture/physics
execution.  This module deliberately sends only digest-bound metadata in its
prompt: never source images, InteriorGS bytes, STEP/USD/mesh bytes, secrets, or
provider credentials.  The model response remains advisory and cannot set any
simulation or physical-evidence claim true.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .adp_content_agents_bundle_matrix import (
    validate_agent_cad_content_agents_bundle_matrix,
)
from .agent_operator_runtime import (
    OperatorRunConfig,
    blocked_operator_ledger,
    completed_operator_ledger,
    run_codex_cli_operator,
)
from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest
from .openai_successor_models import OPENAI_REASONING_EFFORT, OPENAI_TEXT_MODEL


SCHEMA_VERSION = "adp_content_agents_codex_advisory_review.v1"
MATRIX_SCHEMA_VERSION = "adp_content_agents_codex_advisory_matrix.v1"
DEFAULT_CODEX_COMMAND_PREFIX = ("npx", "--yes", "@openai/codex@0.147.0")
_MAX_OUTPUT_CHARACTERS = 16_000

OperatorRunner = Callable[[OperatorRunConfig], Mapping[str, Any]]
VersionProbe = Callable[[Sequence[str]], str]


class ContentAgentsCodexAdvisoryReviewError(ValueError):
    """Fail-closed advisory review request/receipt error."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _expected_backend(value: Any) -> str:
    backend = _text(value)
    if not backend:
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_review_backend_invalid"
        )
    return backend


def _expected_slot(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_review_slot_invalid"
        )
    return value


def _select_item(
    *,
    bundle_matrix: Mapping[str, Any],
    replacement_slot: int,
    cad_agent_backend_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = validate_agent_cad_content_agents_bundle_matrix(bundle_matrix)
    matches = [
        row
        for row in matrix["items"]
        if row["replacement_slot"] == replacement_slot
        and row["cad_agent_backend_id"] == cad_agent_backend_id
    ]
    if len(matches) != 1:
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_review_candidate_not_found"
        )
    return matrix, dict(matches[0])


def build_content_agents_codex_advisory_request(
    *,
    bundle_matrix: Mapping[str, Any],
    replacement_slot: int,
    cad_agent_backend_id: str,
) -> dict[str, Any]:
    """Build one metadata-only review request from a verified matrix row."""

    slot = _expected_slot(replacement_slot)
    backend = _expected_backend(cad_agent_backend_id)
    matrix, item = _select_item(
        bundle_matrix=bundle_matrix,
        replacement_slot=slot,
        cad_agent_backend_id=backend,
    )
    route = item["content_agents_execution_route"]
    request = {
        "schema_version": "adp_content_agents_codex_advisory_request.v1",
        "purpose": "content_agents_codex_first_metadata_only_advisory_review",
        "bundle_matrix_digest": matrix["receipt_digest"],
        "candidate": {
            key: item[key]
            for key in (
                "replacement_slot",
                "task_id",
                "asset_id",
                "cad_agent_backend_id",
                "cad_agent_output_receipt_digest",
                "cad_agent_request_digest",
                "cad_agent_reference_manifest_object_digest",
                "mesh_projection_receipt_digest",
                "mesh_packet_digest",
                "candidate_step_sha256",
                "mesh_count",
            )
        },
        "route": {
            "route_digest": route["route_digest"],
            "codex_local_capabilities": list(route["codex_local_capabilities"]),
            "nvidia_content_agents_capabilities": list(
                route["nvidia_content_agents_capabilities"]
            ),
        },
        "disclosure": {
            "prompt_contains_only": "digest_bound_metadata",
            "raw_interiorgs_bytes_disclosed": False,
            "scene_derived_image_bytes_disclosed": False,
            "cad_step_or_usd_bytes_disclosed": False,
            "mesh_bytes_disclosed": False,
            "secrets_or_api_keys_disclosed": False,
            "provider_upload_performed": False,
        },
        "required_review_rules": [
            "Review only the declared Codex-local capabilities.",
            "Do not infer geometry, appearance, physics, contact, or source truth from digests.",
            "Do not treat Codex as an NVIDIA Content Agents execution.",
            "Do not claim SimReady, native import, dynamics, physical equivalence, or policy success.",
            "Identify the residual released NVIDIA capability without proposing an unbound provider call.",
        ],
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def _prompt(request: Mapping[str, Any]) -> str:
    return (
        "You are a read-only Blueprint Content Agents advisory reviewer. "
        "Do not call tools. Review only the digest-bound metadata below. "
        "Return concise JSON with keys `summary`, `codex_local_work`, "
        "`residual_nvidia_work`, and `claim_boundary`. Each value must be a "
        "short string or list of short strings. Do not claim any simulator, "
        "physics, visual, source-truth, or physical qualification.\n\n"
        + json.dumps(request, sort_keys=True, separators=(",", ":"))
    )


def _probe_codex_cli_version(command_prefix: Sequence[str]) -> str:
    command = [*command_prefix, "--version"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_review_cli_version_probe_failed"
        ) from exc
    version = _text(completed.stdout or completed.stderr)
    if completed.returncode != 0 or not version.startswith("codex-cli "):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_review_cli_version_probe_failed"
        )
    return version.splitlines()[0]


def _receipt(
    *,
    request: Mapping[str, Any],
    generated_at: str,
    command_prefix: Sequence[str],
    cli_version: str,
    status: str,
    operator_ledger: Mapping[str, Any],
    output: str,
    blockers: Sequence[str],
    live_codex_host_oauth_authorized: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "request": dict(request),
        "codex_operator": {
            "transport": "codex_cli_host_oauth",
            "model": OPENAI_TEXT_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "command_prefix": list(command_prefix),
            "cli_version": cli_version,
            "sandbox": "read-only",
            "ephemeral": True,
            "user_config_ignored": True,
            "api_key_forwarded": False,
            "workspace_contains_disclosed_artifacts": False,
            "live_codex_host_oauth_authorized": live_codex_host_oauth_authorized,
        },
        "operator_ledger": dict(operator_ledger),
        "model_output": {
            "text": output,
            "sha256": _sha256_text(output),
            "interpreted_as_deterministic_evidence": False,
        },
        "blockers": sorted(set(_text(blocker) for blocker in blockers if _text(blocker))),
        "claim_boundary": {
            "codex_advisory_review_completed": status
            == "completed_metadata_only_advisory_review",
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "appearance_materially_qualified": False,
            "physical_equivalence": False,
        },
        "receipt_digest": "",
    }


def run_content_agents_codex_advisory_review(
    *,
    bundle_matrix: Mapping[str, Any],
    replacement_slot: int,
    cad_agent_backend_id: str,
    output_path: str | Path,
    generated_at: str | None = None,
    timeout_seconds: int = 120,
    live_codex_host_oauth_authorized: bool = False,
    codex_command_prefix: Sequence[str] = DEFAULT_CODEX_COMMAND_PREFIX,
    runner: OperatorRunner = run_codex_cli_operator,
    version_probe: VersionProbe = _probe_codex_cli_version,
) -> dict[str, Any]:
    """Run a host-OAuth Codex review without sending source artifact bytes."""

    prefix = tuple(_text(item) for item in codex_command_prefix)
    if not prefix or any(not item for item in prefix):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_review_command_invalid"
        )
    request = build_content_agents_codex_advisory_request(
        bundle_matrix=bundle_matrix,
        replacement_slot=replacement_slot,
        cad_agent_backend_id=cad_agent_backend_id,
    )
    timestamp = _text(generated_at) or utc_now_iso()
    cli_version = "not_probed"
    output = ""
    blockers: list[str] = []
    try:
        if not live_codex_host_oauth_authorized:
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_host_oauth_authorization_missing"
            )
        cli_version = version_probe(prefix)
        with tempfile.TemporaryDirectory(prefix="blueprint-content-agents-codex-") as cwd:
            operator_output = dict(
                runner(
                    OperatorRunConfig(
                        adapter="content_agents_codex_metadata_advisory_reviewer",
                        model=OPENAI_TEXT_MODEL,
                        prompt=_prompt(request),
                        plan_context=request,
                        reasoning_effort=OPENAI_REASONING_EFFORT,
                        sandbox="read-only",
                        cwd=cwd,
                        timeout_seconds=timeout_seconds,
                        codex_command_prefix=prefix,
                        codex_ephemeral=True,
                        codex_ignore_user_config=True,
                    )
                )
            )
        output = _text(operator_output.get("final_output"))
        if len(output) > _MAX_OUTPUT_CHARACTERS:
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_review_output_too_large"
            )
        ledger = completed_operator_ledger(
            adapter="content_agents_codex_metadata_advisory_reviewer",
            output=operator_output,
            default_command="codex exec",
            proof_artifacts_required=("deterministically accepted Content Agents artifacts",),
        )
        status = "completed_metadata_only_advisory_review"
    except (ContentAgentsCodexAdvisoryReviewError, RuntimeError, TypeError) as exc:
        code = _text(exc) or "content_agents_codex_advisory_review_operator_failed"
        blockers.append(code)
        ledger = blocked_operator_ledger(
            adapter="content_agents_codex_metadata_advisory_reviewer",
            blockers=blockers,
            command_chosen="codex exec",
            proof_artifacts_required=("deterministically accepted Content Agents artifacts",),
        )
        status = "blocked_before_metadata_only_advisory_review"
    receipt = _receipt(
        request=request,
        generated_at=timestamp,
        command_prefix=prefix,
        cli_version=cli_version,
        status=status,
        operator_ledger=ledger,
        output=output,
        blockers=blockers,
        live_codex_host_oauth_authorized=live_codex_host_oauth_authorized,
    )
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(output_path, receipt)
    return receipt


def _review_file_record(path_value: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_review_file_invalid"
        )
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_review_file_invalid"
        ) from exc
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "completed_metadata_only_advisory_review"
        or receipt.get("receipt_digest")
        != canonical_digest(receipt, digest_field="receipt_digest")
    ):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_review_receipt_invalid"
        )
    return (
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "receipt_digest": receipt["receipt_digest"],
        },
        dict(receipt),
    )


def _review_matrix_expected(
    *,
    bundle_matrix: Mapping[str, Any],
    review_receipt_paths: Sequence[str | Path],
    generated_at: str,
) -> dict[str, Any]:
    matrix = validate_agent_cad_content_agents_bundle_matrix(bundle_matrix)
    if (
        not isinstance(review_receipt_paths, Sequence)
        or isinstance(review_receipt_paths, (str, bytes))
        or len(review_receipt_paths) != matrix["candidate_count"]
    ):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_review_count_invalid"
        )
    expected_by_key = {
        (item["replacement_slot"], item["cad_agent_backend_id"]): item
        for item in matrix["items"]
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for path_value in review_receipt_paths:
        record, review = _review_file_record(path_value)
        request = review.get("request")
        candidate = request.get("candidate") if isinstance(request, Mapping) else None
        route = request.get("route") if isinstance(request, Mapping) else None
        if not isinstance(candidate, Mapping) or not isinstance(route, Mapping):
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_review_binding_invalid"
            )
        key = (candidate.get("replacement_slot"), candidate.get("cad_agent_backend_id"))
        if not isinstance(key[0], int) or isinstance(key[0], bool) or key in seen:
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_review_binding_invalid"
            )
        expected = expected_by_key.get(key)
        if expected is None:
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_review_binding_invalid"
            )
        candidate_fields = (
            "replacement_slot",
            "task_id",
            "asset_id",
            "cad_agent_backend_id",
            "cad_agent_output_receipt_digest",
            "cad_agent_request_digest",
            "cad_agent_reference_manifest_object_digest",
            "mesh_projection_receipt_digest",
            "mesh_packet_digest",
            "candidate_step_sha256",
            "mesh_count",
        )
        if any(candidate.get(field) != expected[field] for field in candidate_fields) or (
            request.get("bundle_matrix_digest") != matrix["receipt_digest"]
            or route.get("route_digest")
            != expected["content_agents_execution_route"]["route_digest"]
            or review.get("claim_boundary")
            != {
                "codex_advisory_review_completed": True,
                "content_agents_executed": False,
                "simready_qualified": False,
                "native_simulator_import_qualified": False,
                "appearance_materially_qualified": False,
                "physical_equivalence": False,
            }
        ):
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_review_binding_invalid"
            )
        seen.add(key)
        rows.append(
            {
                key: expected[key]
                for key in (
                    "replacement_slot",
                    "task_id",
                    "asset_id",
                    "cad_agent_backend_id",
                )
            }
            | {"review_receipt": record}
        )
    if set(seen) != set(expected_by_key):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_review_coverage_invalid"
        )
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "all_candidates_metadata_only_advisory_reviewed",
        "bundle_matrix_digest": matrix["receipt_digest"],
        "candidate_count": matrix["candidate_count"],
        "replacement_object_capacity": dict(matrix["replacement_object_capacity"]),
        "items": sorted(
            rows,
            key=lambda row: (row["replacement_slot"], row["cad_agent_backend_id"]),
        ),
        "claim_boundary": {
            "codex_advisory_reviews_completed": True,
            "content_agents_executed": False,
            "simready_qualified": False,
            "native_simulator_import_qualified": False,
            "appearance_materially_qualified": False,
            "physical_equivalence": False,
        },
    }


def validate_content_agents_codex_advisory_matrix(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-open every review receipt and reject relabelled receipt drift."""

    if not isinstance(value, Mapping):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_not_mapping"
        )
    matrix = json.loads(json.dumps(value))
    if (
        matrix.get("schema_version") != MATRIX_SCHEMA_VERSION
        or not _text(matrix.get("generated_at"))
        or matrix.get("receipt_digest")
        != canonical_digest(matrix, digest_field="receipt_digest")
    ):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_schema_invalid"
        )
    capacity = matrix.get("replacement_object_capacity")
    items = matrix.get("items")
    if (
        not isinstance(capacity, Mapping)
        or capacity.get("minimum") != 1
        or not isinstance(capacity.get("maximum"), int)
        or not isinstance(capacity.get("sealed_slots"), int)
        or not isinstance(items, list)
        or matrix.get("candidate_count") != len(items)
    ):
        raise ContentAgentsCodexAdvisoryReviewError(
            "content_agents_codex_advisory_matrix_content_invalid"
        )
    seen: set[tuple[int, str]] = set()
    for row in items:
        if not isinstance(row, Mapping):
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_content_invalid"
            )
        slot = row.get("replacement_slot")
        backend = _text(row.get("cad_agent_backend_id"))
        review_record = row.get("review_receipt")
        if (
            not isinstance(slot, int)
            or isinstance(slot, bool)
            or not backend
            or (slot, backend) in seen
            or not isinstance(review_record, Mapping)
        ):
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_content_invalid"
            )
        review_file_record, review = _review_file_record(
            _text(review_record.get("path"))
        )
        if (
            dict(review_record) != review_file_record
            or not isinstance(review.get("request"), Mapping)
            or not isinstance((review["request"]).get("candidate"), Mapping)
            or any(
                (review["request"]["candidate"]).get(key) != row.get(key)
                for key in (
                    "replacement_slot",
                    "task_id",
                    "asset_id",
                    "cad_agent_backend_id",
                )
            )
        ):
            raise ContentAgentsCodexAdvisoryReviewError(
                "content_agents_codex_advisory_matrix_content_invalid"
            )
        seen.add((slot, backend))
    return matrix


def materialize_content_agents_codex_advisory_matrix(
    *,
    bundle_matrix: Mapping[str, Any],
    review_receipt_paths: Sequence[str | Path],
    output_path: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Seal complete 1--5-slot Codex advisory coverage from review receipts."""

    timestamp = _text(generated_at) or utc_now_iso()
    matrix = _review_matrix_expected(
        bundle_matrix=bundle_matrix,
        review_receipt_paths=review_receipt_paths,
        generated_at=timestamp,
    )
    matrix["receipt_digest"] = canonical_digest(matrix, digest_field="receipt_digest")
    write_json(output_path, matrix)
    return matrix


__all__ = [
    "ContentAgentsCodexAdvisoryReviewError",
    "DEFAULT_CODEX_COMMAND_PREFIX",
    "MATRIX_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_content_agents_codex_advisory_request",
    "materialize_content_agents_codex_advisory_matrix",
    "run_content_agents_codex_advisory_review",
    "validate_content_agents_codex_advisory_matrix",
]
