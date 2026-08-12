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
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .gpu_render_providers import get_render_provider
from .task_evaluation_terminal_resource_release_contract import (
    QUEUE_RUN_SCHEMA_VERSION,
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA_VERSION,
    TerminalResourceReleaseError,
    canonical_digest,
    validate_terminal_resource_release_request,
)

# Compatibility export for the provider-worker module; the HTTP intake imports
# this pure contract directly and therefore never reaches the hot provider lane.
from .task_evaluation_terminal_resource_release_contract import (  # noqa: F401
    stage_terminal_resource_release_request,
)


_TERMINAL_STATUSES = frozenset(
    {"stopped", "exited", "failed", "destroyed", "deleted", "inactive", "completed"}
)
_REQUIRED_PROVIDERS = ("runpod", "vast", "digitalocean")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


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


def _retain_release_receipt(
    *, state_root: Path, release_id: str, receipt: Mapping[str, Any]
) -> None:
    """Persist the outcome receipt beside the release's other evidence.

    Retention must never mask the outcome it records, so a write failure is
    swallowed rather than replacing a real terminal receipt with an unrelated
    I/O error.
    """
    try:
        destination = state_root / release_id
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / RECEIPT_FILENAME, dict(receipt))
    except OSError:
        return


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
        # Retain the outcome for every terminal state, not just success. A
        # blocked release that keeps no receipt can never be shown to have
        # stopped before touching the provider, so it can never be safely
        # re-armed and the resource it names stays stranded.
        _retain_release_receipt(
            state_root=Path(state_root).expanduser().resolve(),
            release_id=_string(receipt.get("release_id")) or claimed.stem,
            receipt=receipt,
        )
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
