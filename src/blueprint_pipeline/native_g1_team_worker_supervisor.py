"""Supervise a selected G1 episode without owning paid provider allocation.

Isaac may exit its interpreter natively during close. Recovery requires the
actual zero exit receipt, bound preclose and scored child receipts. The caller
must still verify all score/media bytes, provider teardown and posted billing.
Raw child output is retained only in a private log, never copied into receipts.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_g1_team_policy_worker import (
    FILENAME, PRECLOSE_FILENAME, PRECLOSE_SCHEMA, SCHEMA as WORKER_SCHEMA,
    _execution_packet, _verify_packet,
    _argument_parser,
)
from .native_g1_team_supervised_episode import FILENAME as SUPERVISED_FILENAME


EXIT_SCHEMA = "native_g1_team_worker_process_exit.v1"
EXIT_FILENAME = "worker.exit.json"
MAX_TIMEOUT_SECONDS = 45 * 60
RESULT_SCHEMA = "native_g1_team_supervised_worker_result.v1"
RESULT_FILENAME = RESULT_SCHEMA + ".json"
_ARGUMENT_FLAGS = {
    "execution_packet_path": "execution-packet",
    "expected_implementation_commit": "expected-implementation-commit",
    "scene_packet_root": "scene-packet-root",
    "runtime_provisioning_receipt_path": "runtime-provisioning-receipt",
    "sonic_provider_source": "sonic-provider-source",
    "sonic_encoder": "sonic-encoder",
    "sonic_encoder_sha256": "sonic-encoder-sha256",
    "sonic_decoder": "sonic-decoder",
    "sonic_decoder_sha256": "sonic-decoder-sha256",
    "max_steps": "max-steps",
    "credential_file_path": "credential-file",
    "policy_relay_config_path": "policy-relay-config",
}


def _read(path: Path, digest_field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("g1_team_worker_receipt_unavailable")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get(digest_field) != canonical_digest(value, digest_field=digest_field)
    ):
        raise ValueError("g1_team_worker_receipt_digest_invalid")
    return value


def run_g1_team_worker_process(
    *, command: Sequence[str], diagnostics_dir: Path, timeout_seconds: float,
) -> dict[str, Any]:
    """Retain one real child handle and exit; never restart after observation."""

    root = Path(diagnostics_dir)
    if (
        not root.is_absolute() or root.is_symlink() or root.exists()
        or not isinstance(command, (list, tuple)) or not command
        or any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in command)
        or type(timeout_seconds) not in (int, float)
        or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS
    ):
        raise ValueError("g1_team_worker_supervision_input_invalid")
    root.mkdir(mode=0o700, parents=True)
    process = None
    result: dict[str, Any] = {
        "schema_version": EXIT_SCHEMA,
        "status": "launch_failed",
        "returncode": None,
        "child_pid": None,
        "command_digest": canonical_digest({"argv": list(command)}),
        "timeout_seconds": timeout_seconds,
        "process_group_kill_requested": False,
        "error_type": None,
        "claim_ceiling": "development_only",
    }
    fd = os.open(root / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(
                list(command), stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            result["child_pid"] = process.pid
            try:
                result["returncode"] = process.wait(timeout=timeout_seconds)
                result["status"] = "exited"
            except subprocess.TimeoutExpired:
                result["status"] = "timed_out"
                # The worker and any children in its session must stop. A zero
                # return from SIGTERM is still a timeout, never episode success.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                result["process_group_kill_requested"] = True
                result["returncode"] = process.wait()
        except OSError as exc:
            result["error_type"] = type(exc).__name__
        finally:
            # Also reap on a supervisor exception, without claiming it was a
            # normal completion. Independent provider watchdogs remain outside.
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                result["process_group_kill_requested"] = True
                result["returncode"] = process.wait()
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    with (root / EXIT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the same worker configuration to provider entrypoints."""

    parser = _argument_parser()
    parser.description = __doc__
    parser.add_argument("--worker-launcher", type=Path, default=Path("/isaac-sim/python.sh"))
    parser.add_argument("--timeout-seconds", type=float, default=MAX_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    result = run_supervised_g1_team_worker(
        worker_arguments={
            "execution_packet_path": args.execution_packet,
            "expected_implementation_commit": args.expected_implementation_commit,
            "scene_packet_root": args.scene_packet_root,
            "runtime_provisioning_receipt_path": args.runtime_provisioning_receipt,
            "sonic_provider_source": args.sonic_provider_source,
            "sonic_encoder": args.sonic_encoder,
            "sonic_encoder_sha256": args.sonic_encoder_sha256,
            "sonic_decoder": args.sonic_decoder,
            "sonic_decoder_sha256": args.sonic_decoder_sha256,
            "credential_file_path": args.credential_file,
            "max_steps": args.max_steps,
        },
        output_dir=args.output_dir, worker_launcher=args.worker_launcher,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"status": result["status"], "result_digest": result["result_digest"]}))
    return 0 if result["status"] == "completed_development_only" else 2


def valid_g1_team_worker_exit(
    *, output_dir: Path, worker: Mapping[str, Any], preclose: Mapping[str, Any],
) -> bool:
    """A process-exit teardown must bind retained zero-exit and preclose bytes."""

    if worker.get("teardown") == {"environment": "closed", "simulator": "closed"}:
        return worker.get("process_exit_evidence") is None
    if worker.get("teardown") != {
        "environment": "closed", "simulator": "process_exited_after_close_request",
    }:
        return False
    exit_receipt = _read(Path(output_dir) / EXIT_FILENAME, "receipt_digest")
    evidence = worker.get("process_exit_evidence")
    return (
        exit_receipt.get("schema_version") == EXIT_SCHEMA
        and exit_receipt.get("status") == "exited"
        and type(exit_receipt.get("returncode")) is int and exit_receipt["returncode"] == 0
        and exit_receipt.get("process_group_kill_requested") is False
        and exit_receipt.get("error_type") is None
        and evidence == {
            "returncode": 0,
            "receipt_digest": exit_receipt["receipt_digest"],
            "preclose_digest": preclose.get("preclose_digest"),
        }
    )


def recover_g1_team_worker_result(
    *, worker_output_dir: Path, execution_packet: Mapping[str, Any], returncode: int,
) -> dict[str, Any]:
    """Seal a native close exit, retaining process-exit wording in teardown."""

    root = Path(worker_output_dir)
    if type(returncode) is not int or returncode != 0:
        raise ValueError("g1_team_worker_child_exit_invalid")
    exit_receipt = _read(root / EXIT_FILENAME, "receipt_digest")
    if (
        exit_receipt.get("schema_version") != EXIT_SCHEMA
        or exit_receipt.get("status") != "exited"
        or type(exit_receipt.get("returncode")) is not int
        or exit_receipt["returncode"] != returncode
        or exit_receipt.get("process_group_kill_requested") is not False
        or exit_receipt.get("error_type") is not None
    ):
        raise ValueError("g1_team_worker_child_exit_invalid")
    preclose = _read(root / PRECLOSE_FILENAME, "preclose_digest")
    supervised = _read(root / "episode" / SUPERVISED_FILENAME, "result_digest")
    queries = preclose.get("policy_query_count")
    request = execution_packet["request"]
    if (
        preclose.get("schema_version") != PRECLOSE_SCHEMA
        or preclose.get("status") != "awaiting_simulator_close"
        or preclose.get("execution_packet_digest") != execution_packet["packet_digest"]
        or preclose.get("operator_approval_digest")
        != execution_packet["operator_approval"]["approval_digest"]
        or preclose.get("profile_digest") != request["policy_profile"]["profile_digest"]
        or preclose.get("objective_id") != request["objective_id"]
        or preclose.get("delivery_mode") != request["policy_profile"]["delivery"]["mode"]
        or preclose.get("teardown") != {"environment": "closed", "simulator": "close_requested"}
        or preclose.get("blocker_type") is not None
        or type(queries) is not int or queries <= 0
        or supervised.get("status") != "completed_development_only"
        or supervised.get("profile_digest") != preclose.get("profile_digest")
        or supervised.get("operator_approval_digest") != preclose.get("operator_approval_digest")
        or supervised.get("policy_query_count") != queries
        or preclose.get("supervised_episode_result_digest") != supervised["result_digest"]
    ):
        raise ValueError("g1_team_worker_preclose_binding_invalid")
    result = {
        **{key: value for key, value in preclose.items() if key != "preclose_digest"},
        "schema_version": WORKER_SCHEMA,
        "status": "completed_development_only",
        "teardown": {"environment": "closed", "simulator": "process_exited_after_close_request"},
        "process_exit_evidence": {
            "returncode": returncode,
            "receipt_digest": exit_receipt["receipt_digest"],
            "preclose_digest": preclose["preclose_digest"],
        },
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with (root / FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def run_supervised_g1_team_worker(
    *, worker_arguments: Mapping[str, Any], output_dir: Path,
    worker_launcher: Path, timeout_seconds: float = MAX_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Run exactly one approved selected-policy child and verify its score/media.

    This is a worker entry point, not a provider launcher. A paid controller
    must recheck current registry/approval and spend authority before invoking
    it on allocated compute. Delivery-mode runtime capability remains a
    separate pre-allocation gate; this function cannot grant those capabilities.
    """

    from .native_g1_team_paid_output import verify_g1_team_paid_output

    root = Path(output_dir)
    launcher = Path(worker_launcher)
    if (
        not root.is_absolute() or root.exists() or root.is_symlink()
        or not launcher.is_absolute() or not launcher.is_file()
        or set(worker_arguments) - set(_ARGUMENT_FLAGS)
        or not set(_ARGUMENT_FLAGS) - {"credential_file_path", "policy_relay_config_path"} <= set(worker_arguments)
        or re.fullmatch(r"[a-f0-9]{40}", str(worker_arguments.get("expected_implementation_commit"))) is None
        or type(worker_arguments.get("max_steps")) is not int
        or not 1 <= worker_arguments["max_steps"] <= 3000
        or any(
            not isinstance(worker_arguments[field], Path)
            or not worker_arguments[field].is_absolute()
            for field in _ARGUMENT_FLAGS
            if field in worker_arguments and field not in {
                "max_steps", "expected_implementation_commit",
                "sonic_encoder_sha256", "sonic_decoder_sha256",
            } and worker_arguments[field] is not None
        )
    ):
        raise ValueError("g1_team_supervised_worker_inputs_invalid")
    root.mkdir(parents=True, mode=0o700)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "blocked", "claim_ceiling": "development_only",
        "stage_reached": "input_verification", "blocker_type": None, "blocker_code": None,
        "execution_packet_digest": None, "child_exit_receipt_digest": None,
        "recovered_native_close": False, "verified_output": None,
        "provider_teardown_verified": False, "official_billing_reconciled": False,
        "public_redistribution_authorized": False,
    }
    try:
        packet = _execution_packet(
            worker_arguments["execution_packet_path"],
            worker_arguments["expected_implementation_commit"],
        )
        scene = _verify_packet(worker_arguments["scene_packet_root"])
        result["execution_packet_digest"] = packet["packet_digest"]
        command = [str(launcher), "-m", "blueprint_pipeline.native_g1_team_policy_worker"]
        for field, flag in _ARGUMENT_FLAGS.items():
            if worker_arguments.get(field) is not None:
                command.extend(["--" + flag, str(worker_arguments[field])])
        worker_root = root / "worker"
        command.extend(["--output-dir", str(worker_root)])
        result["stage_reached"] = "worker_process"
        exited = run_g1_team_worker_process(
            command=command, diagnostics_dir=root / "private_diagnostics",
            timeout_seconds=timeout_seconds,
        )
        result["child_exit_receipt_digest"] = exited["receipt_digest"]
        result["stage_reached"] = "worker_" + exited["status"]
        if exited["status"] != "exited" or exited["returncode"] != 0:
            raise ValueError("g1_team_supervised_worker_child_failed")
        if not worker_root.is_dir() or worker_root.is_symlink():
            raise ValueError("g1_team_supervised_worker_output_missing")
        with (worker_root / EXIT_FILENAME).open("x", encoding="utf-8") as stream:
            json.dump(exited, stream, sort_keys=True, indent=2)
            stream.write("\n")
        if not (worker_root / FILENAME).exists():
            result["stage_reached"] = "native_close_recovery"
            recover_g1_team_worker_result(
                worker_output_dir=worker_root, execution_packet=packet,
                returncode=exited["returncode"],
            )
            result["recovered_native_close"] = True
        result["stage_reached"] = "score_media_verification"
        result["verified_output"] = verify_g1_team_paid_output(
            output_dir=worker_root, execution_packet=packet,
            scene_plan_digest=scene["arena_scene_plan_digest"],
            scene_packet_receipt_digest=scene["receipt_digest"],
        )
        result["status"] = "completed_development_only"
    except Exception as exc:  # noqa: BLE001 - retain terminal typed evidence, never raw details
        result["blocker_type"] = type(exc).__name__
        # Only fixed codes created here may be exposed. Validation/provider
        # exceptions can contain endpoint responses or a secret-bearing URL.
        code = str(exc)
        result["blocker_code"] = code if code in {
            "g1_team_supervised_worker_child_failed",
            "g1_team_supervised_worker_output_missing",
            "g1_team_worker_child_exit_invalid",
            "g1_team_worker_preclose_binding_invalid",
            "g1_team_worker_receipt_unavailable",
            "g1_team_worker_receipt_digest_invalid",
        } else "g1_team_supervised_worker_validation_failed"
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with (root / RESULT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    sys.exit(main())
