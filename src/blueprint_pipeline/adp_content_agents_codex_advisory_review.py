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


__all__ = [
    "ContentAgentsCodexAdvisoryReviewError",
    "DEFAULT_CODEX_COMMAND_PREFIX",
    "SCHEMA_VERSION",
    "build_content_agents_codex_advisory_request",
    "run_content_agents_codex_advisory_review",
]
