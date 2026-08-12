"""Release one terminal Vast record recovered from a website-owned launch.

The original launch watchdog normally retains its exact provider id.  If the
control-plane host itself is lost after the provider workload exits, this
release-only queue closes that narrow gap.  It cannot launch, resume, retry, or
score an evaluation: it permits one exact ``DELETE`` only after an authenticated
WebApp request binds a terminal-blocked launch to a stopped Vast record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider
from .task_evaluation_launch_dispatcher import canonical_digest
from .vast_provider_adapter import VAST_TERMINAL_INSTANCE_STATUSES


REQUEST_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_request.v1"
QUEUE_RECEIPT_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_queue_receipt.v1"
RECEIPT_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_receipt.v1"
QUEUE_RUN_SCHEMA_VERSION = "task_evaluation_terminal_resource_release_queue_run.v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
_INSTANCE_ID_RE = re.compile(r"^[1-9][0-9]{0,18}$")
_LABEL_RE = re.compile(r"^blueprint-adp009d-[1-9][0-9]{9,}$")
_TERMINAL_STATUSES = frozenset(str(value).lower() for value in VAST_TERMINAL_INSTANCE_STATUSES)
_REQUIRED_PROVIDERS = ("runpod", "vast", "digitalocean")


class TerminalResourceReleaseError(ValueError):
    """A release-only request failed a fail-closed contract."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _is_digest(value: Any) -> bool:
    return bool(_DIGEST_RE.fullmatch(_string(value)))


def _is_identifier(value: Any) -> bool:
    return bool(_IDENTIFIER_RE.fullmatch(_string(value)))


def validate_terminal_resource_release_request(value: Mapping[str, Any]) -> list[str]:
    """Validate the immutable, zero-spend Website recovery capability."""

    request = _mapping(value)
    blockers: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        blockers.append("terminal_resource_release_schema_version_mismatch")
    for field in ("release_id", "launch_id", "run_id"):
        if not _is_identifier(request.get(field)):
            blockers.append(f"terminal_resource_release_{field}_invalid")
    if not _is_digest(request.get("request_digest")):
        blockers.append("terminal_resource_release_launch_request_digest_invalid")
    if request.get("provider") != "vast":
        blockers.append("terminal_resource_release_provider_must_be_vast")
    if not _INSTANCE_ID_RE.fullmatch(_string(request.get("instance_id"))):
        blockers.append("terminal_resource_release_instance_id_invalid")
    if not _LABEL_RE.fullmatch(_string(request.get("expected_label"))):
        blockers.append("terminal_resource_release_expected_label_invalid")
    blocker = _mapping(request.get("control_plane_terminal_blocker"))
    expected_blocker = {
        "schema_version": "task_evaluation_launch_control_plane_blocker.v1",
        "status": "blocked",
        "code": "control_plane_terminal_receipt_missing_after_spend_authority_expiry",
        "pipeline_terminal_receipt_observed": False,
        "provider_mutation_performed_by_webapp": False,
        "paid_execution_retry_performed": False,
        "execution_result": "not_observed",
        "scripted_positive_controls_result": "not_observed",
        "learned_policy_result": "not_observed",
    }
    for field, expected in expected_blocker.items():
        if blocker.get(field) != expected:
            blockers.append(f"terminal_resource_release_blocker_mismatch:{field}")
    for field in ("launch_id", "run_id", "request_digest"):
        if blocker.get(field) != request.get(field):
            blockers.append(f"terminal_resource_release_blocker_binding_mismatch:{field}")
    authorization = _mapping(request.get("authorization"))
    actor = _mapping(authorization.get("actor"))
    if authorization.get("action") != "terminal_provider_record_release":
        blockers.append("terminal_resource_release_action_invalid")
    if authorization.get("approved") is not True:
        blockers.append("terminal_resource_release_authority_missing")
    if authorization.get("max_additional_spend_usd") != 0:
        blockers.append("terminal_resource_release_spend_must_be_zero")
    if authorization.get("retry_cap") != 0:
        blockers.append("terminal_resource_release_retry_cap_must_be_zero")
    if actor.get("role") not in {"admin", "ops"} or not _is_identifier(actor.get("id")):
        blockers.append("terminal_resource_release_actor_invalid")
    if not _string(authorization.get("authorized_at")):
        blockers.append("terminal_resource_release_authorized_at_missing")
    if request.get("provider_mutation_performed_inside_web_request") is not False:
        blockers.append("terminal_resource_release_webapp_mutation_forbidden")
    if request.get("automatic_retry_performed") is not False:
        blockers.append("terminal_resource_release_automatic_retry_forbidden")
    if request.get("claim_ceiling") != "operational_resource_release_only":
        blockers.append("terminal_resource_release_claim_ceiling_invalid")
    if request.get("terminal_resource_release_digest") != canonical_digest(
        request, digest_field="terminal_resource_release_digest"
    ):
        blockers.append("terminal_resource_release_digest_mismatch")
    return sorted(set(blockers))


def _write_immutable(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
        return True
    except FileExistsError:
        if path.read_bytes() != payload:
            raise TerminalResourceReleaseError(f"immutable_terminal_resource_release_conflict:{path.name}")
        return False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TerminalResourceReleaseError(f"terminal_resource_release_json_object_required:{path.name}")
    return dict(value)


def _artifact(path: Path) -> dict[str, Any]:
    digest = None
    if path.is_file():
        digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "exists": path.is_file(), "digest": digest}


def stage_terminal_resource_release_request(
    *, value: Mapping[str, Any], queue_root: str | Path
) -> dict[str, Any]:
    request = dict(value)
    blockers = validate_terminal_resource_release_request(request)
    if blockers:
        raise TerminalResourceReleaseError(",".join(blockers))
    queue = Path(queue_root).expanduser().resolve()
    digest = _string(request["terminal_resource_release_digest"])
    filename = f"{request['release_id']}-{digest[7:23]}.json"
    existing: Path | None = None
    for state in ("pending", "processing", "completed", "blocked"):
        candidate = queue / state / filename
        if candidate.exists():
            if existing is not None:
                raise TerminalResourceReleaseError(f"duplicate_terminal_resource_release_queue_state:{filename}")
            existing = candidate
    path = existing or queue / "pending" / filename
    created = _write_immutable(path, request)
    return {
        "schema_version": QUEUE_RECEIPT_SCHEMA_VERSION,
        "status": "queued" if created else path.parent.name,
        "already_exists": not created,
        "release_id": request["release_id"],
        "launch_id": request["launch_id"],
        "terminal_resource_release_digest": digest,
        "provider_mutation_performed": False,
    }


def _safe_inspect(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: value.get(field)
        for field in (
            "status", "provider", "http", "instance_id", "desiredStatus", "actual_status",
            "cur_state", "intended_status", "name", "api_confirmed",
            "provider_absence_confirmed", "blockers", "error_type",
        )
        if field in value
    }


def _provider_zero_guard(*, output_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    argv = [
        sys.executable,
        str(root / "scripts" / "gpu_spend_guard.py"),
        "--max-live-instances", "0",
        "--max-burn-usd-per-hour", "0",
        "--json-report", str(output_path),
    ]
    for provider in _REQUIRED_PROVIDERS:
        argv.extend(("--require-provider", provider))
    completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=90)
    try:
        report = _read_json(output_path)
    except (OSError, ValueError, json.JSONDecodeError, TerminalResourceReleaseError):
        report = {}
    return {
        "exit_code": completed.returncode,
        "report": report,
        "raw_process_output_recorded": False,
    }


def dispatch_terminal_resource_release(
    *,
    request_path: str | Path,
    state_root: str | Path,
    provider_factory: Callable[[str], Any] = get_render_provider,
    provider_zero_guard: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect/delete exactly one stopped Vast record and retain global zero evidence."""

    request = _read_json(Path(request_path).expanduser().resolve())
    blockers = validate_terminal_resource_release_request(request)
    release_id = _string(request.get("release_id")) or Path(request_path).stem
    state_dir = Path(state_root).expanduser().resolve() / release_id
    state_dir.mkdir(parents=True, exist_ok=True)
    zero_path = state_dir / "provider_zero.json"
    initial: dict[str, Any] = {}
    final: dict[str, Any] = {}
    termination: dict[str, Any] | None = None
    exact_absence = False
    provider_mutations = 0
    if not blockers:
        provider = provider_factory("vast")
        try:
            initial = _safe_inspect(_mapping(provider.inspect(_string(request["instance_id"]))))
        except Exception as exc:  # noqa: BLE001 - retain a typed, secret-safe blocker
            initial = {"status": "unavailable", "error_type": type(exc).__name__}
        if initial.get("status") == "absent" and initial.get("provider_absence_confirmed") is True:
            exact_absence = True
        elif (
            initial.get("status") != "observed"
            or initial.get("api_confirmed") is not True
            or initial.get("instance_id") != request["instance_id"]
        ):
            blockers.append("terminal_resource_release_exact_instance_unverified")
        elif initial.get("name") != request["expected_label"]:
            blockers.append("terminal_resource_release_expected_label_mismatch")
        else:
            observed_status = _string(
                initial.get("desiredStatus")
                or initial.get("actual_status")
                or initial.get("cur_state")
                or initial.get("intended_status")
            ).lower()
            if observed_status not in _TERMINAL_STATUSES:
                blockers.append("terminal_resource_release_instance_not_terminal")
            else:
                try:
                    termination = _mapping(provider.terminate(_string(request["instance_id"])))
                    provider_mutations = 1
                except Exception as exc:  # noqa: BLE001
                    termination = {"status": "teardown_unverified", "error_type": type(exc).__name__}
                try:
                    final = _safe_inspect(_mapping(provider.inspect(_string(request["instance_id"]))))
                except Exception as exc:  # noqa: BLE001
                    final = {"status": "unavailable", "error_type": type(exc).__name__}
                exact_absence = (
                    final.get("status") == "absent"
                    and final.get("provider_absence_confirmed") is True
                    and final.get("instance_id") == request["instance_id"]
                )
                if not exact_absence:
                    blockers.append("terminal_resource_release_exact_absence_unverified")
    guard = (provider_zero_guard or (lambda path: _provider_zero_guard(output_path=path)))(zero_path)
    report = _mapping(_mapping(guard).get("report"))
    provider_zero_verified = report.get("provider_zero_verified") is True
    if not provider_zero_verified:
        blockers.append("terminal_resource_release_global_provider_zero_unverified")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "completed" if exact_absence and provider_zero_verified and not blockers else "blocked",
        "release_id": release_id,
        "launch_id": request.get("launch_id"),
        "run_id": request.get("run_id"),
        "request_digest": request.get("request_digest"),
        "terminal_resource_release_digest": request.get("terminal_resource_release_digest"),
        "provider": "vast",
        "instance_id": request.get("instance_id"),
        "expected_label": request.get("expected_label"),
        "initial_exact_inspect": initial,
        "termination": termination,
        "final_exact_inspect": final,
        "exact_provider_absence_confirmed": exact_absence,
        "provider_zero": _artifact(zero_path),
        "provider_zero_verified": provider_zero_verified,
        "provider_mutation_attempted": provider_mutations > 0,
        "provider_mutations_performed": provider_mutations,
        "automatic_retry_performed": False,
        "execution_result": "not_observed",
        "scripted_positive_controls_result": "not_observed",
        "learned_policy_result": "not_observed",
        "claim_ceiling": "operational_resource_release_only",
        "raw_secret_values_recorded": False,
        "completed_at_iso": utc_now_iso(),
        "blockers": sorted(set(blockers)),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    write_json(state_dir / "terminal_resource_release_receipt.json", receipt)
    return receipt


def _dispatch_through_canonical_allocator(*, request_path: Path, state_root: Path) -> dict[str, Any]:
    """Invoke the only provider-mutation entrypoint with a fixed argv shape."""

    request = _read_json(request_path)
    release_id = _string(request.get("release_id")) or request_path.stem
    output = state_root / release_id / "canonical_allocator_release_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable, "-m", "blueprint_pipeline.paid_resource_allocator", "gpu-canary",
            "--terminal-resource-release", str(request_path),
            "--terminal-resource-release-output", str(output),
            "--execute",
        ],
        check=False, capture_output=True, text=True, timeout=120,
    )
    try:
        receipt = _read_json(output)
    except (OSError, ValueError, json.JSONDecodeError, TerminalResourceReleaseError):
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "blocked",
            "release_id": release_id,
            "provider_mutation_attempted": None,
            "provider_mutations_performed": None,
            "automatic_retry_performed": False,
            "blockers": ["terminal_resource_release_allocator_receipt_missing"],
            "allocator_exit_code": completed.returncode,
            "raw_secret_values_recorded": False,
        }
    receipt["canonical_allocator"] = "python -m blueprint_pipeline.paid_resource_allocator gpu-canary"
    receipt["allocator_exit_code"] = completed.returncode
    receipt["allocator_raw_process_output_recorded"] = False
    return receipt


def process_terminal_resource_release_queue(
    *, queue_root: str | Path, state_root: str | Path, max_messages: int = 1,
    dispatcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    queue = Path(queue_root).expanduser().resolve()
    pending, processing = queue / "pending", queue / "processing"
    pending.mkdir(parents=True, exist_ok=True)
    processing.mkdir(parents=True, exist_ok=True)
    dispatch = dispatcher or _dispatch_through_canonical_allocator
    receipts: list[dict[str, Any]] = []
    for source in sorted(pending.glob("*.json"))[: max(0, max_messages)]:
        claimed = processing / source.name
        source.replace(claimed)
        try:
            receipt = dispatch(request_path=claimed, state_root=Path(state_root).expanduser().resolve())
        except Exception as exc:  # noqa: BLE001 - retain a terminal typed blocker
            receipt = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "status": "blocked",
                "release_id": claimed.stem,
                "provider_mutation_attempted": None,
                "provider_mutations_performed": None,
                "automatic_retry_performed": False,
                "blockers": ["terminal_resource_release_dispatcher_unhandled_error"],
                "error_type": type(exc).__name__,
                "raw_secret_values_recorded": False,
            }
        destination = queue / ("completed" if receipt.get("status") == "completed" else "blocked")
        destination.mkdir(parents=True, exist_ok=True)
        claimed.replace(destination / claimed.name)
        receipts.append(receipt)
    return {
        "schema_version": QUEUE_RUN_SCHEMA_VERSION,
        "status": "completed" if all(row.get("status") == "completed" for row in receipts) else "blocked",
        "processed_count": len(receipts),
        "receipts": receipts,
        "automatic_retry_performed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--max-messages", type=int, default=1)
    args = parser.parse_args(argv)
    result = process_terminal_resource_release_queue(
        queue_root=args.queue_root, state_root=args.state_root, max_messages=args.max_messages,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("schema_version") == QUEUE_RUN_SCHEMA_VERSION else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
