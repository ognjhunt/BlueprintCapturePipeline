"""Execute one digest-bound Isaac reconstruction verification bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_isaac_output_bundle import (
    IsaacVerificationOutputBundleError,
    compile_isaac_verification_output_bundle,
)
from .reconstruction_isaac_worker_bundle import (
    IsaacWorkerBundleError,
    extract_isaac_verification_worker_bundle,
    validate_isaac_verification_worker_bundle_receipt,
)
from .safe_outbound_http import (
    SafeOutboundHttpError,
    download_file,
    presigned_transfer_policy,
    upload_file,
)


SCHEMA_VERSION = "reconstruction_isaac_bootstrap.v1"
MAX_INPUT_BUNDLE_BYTES = 5_000_000_000
MAX_RECEIPT_BYTES = 16 * 1024**2
MAX_OUTPUT_BUNDLE_BYTES = 5_000_000_000
MAX_LOG_BYTES = 16 * 1024**2
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class ReconstructionIsaacBootstrapError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _required(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name) or "")
    if not value:
        raise ReconstructionIsaacBootstrapError(
            [f"reconstruction_isaac_bootstrap_env_missing:{name}"]
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(
    path: Path, *, code: str = "reconstruction_isaac_bootstrap_receipt_invalid"
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionIsaacBootstrapError([code]) from exc
    if not isinstance(value, Mapping):
        raise ReconstructionIsaacBootstrapError([code])
    return dict(value)


def _default_process_runner(
    command: Sequence[str], root: Path, log_path: Path, timeout_seconds: int
) -> int:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "BLUEPRINT_ISAAC_INPUT_BUNDLE_GET_URL",
            "BLUEPRINT_ISAAC_INPUT_RECEIPT_GET_URL",
            "BLUEPRINT_ISAAC_OUTPUT_BUNDLE_PUT_URL",
        }
    }
    process = subprocess.Popen(
        list(command),
        cwd=root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=True,
        bufsize=0,
    )
    assert process.stdout is not None

    def signal_process(signum: signal.Signals) -> None:
        try:
            os.killpg(process.pid, signum)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            # Some macOS process states refuse process-group signaling even
            # though the direct child remains controllable. Linux workers
            # retain group-first teardown so descendants cannot outlive TTL.
            pass
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            pass

    def stop_process() -> None:
        if process.poll() is not None:
            return
        signal_process(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            signal_process(signal.SIGKILL)
            process.wait(timeout=5)

    deadline = time.monotonic() + timeout_seconds
    total = 0
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        with log_path.open("xb") as log:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_process()
                    raise ReconstructionIsaacBootstrapError(
                        ["reconstruction_isaac_bootstrap_runtime_timeout"]
                    )
                events = selector.select(timeout=min(1.0, remaining))
                if not events and process.poll() is not None:
                    # Drain the final pipe bytes before accepting process exit.
                    events = [(selector.get_key(process.stdout), selectors.EVENT_READ)]
                for key, _mask in events:
                    chunk = os.read(key.fd, 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > MAX_LOG_BYTES:
                        remaining_bytes = MAX_LOG_BYTES - (total - len(chunk))
                        if remaining_bytes > 0:
                            log.write(chunk[:remaining_bytes])
                        stop_process()
                        raise ReconstructionIsaacBootstrapError(
                            ["reconstruction_isaac_bootstrap_log_oversized"]
                        )
                    log.write(chunk)
        return int(process.wait(timeout=5))
    except BaseException:
        stop_process()
        raise
    finally:
        selector.close()
        process.stdout.close()


def _rebase_command(command: Sequence[str], root: Path) -> list[str]:
    replacements = {
        "/workspace/bundle/run_isaac_splat_nurec_render.py": str(
            root / "bundle/run_isaac_splat_nurec_render.py"
        ),
        "/workspace/bundle/reconstruction.usdz": str(
            root / "bundle/reconstruction.usdz"
        ),
        "/workspace/bundle/fixed_cameras.json": str(
            root / "bundle/fixed_cameras.json"
        ),
        "/workspace/out": str(root / "out"),
    }
    return [replacements.get(item, item) for item in command]


def run_reconstruction_isaac_bootstrap(
    *,
    environment: Mapping[str, str],
    work_root: str | Path,
    process_runner: Callable[[Sequence[str], Path, Path, int], int] = (
        _default_process_runner
    ),
) -> dict[str, Any]:
    """Download, run, package, and upload one typed Isaac compatibility probe."""

    input_url = _required(environment, "BLUEPRINT_ISAAC_INPUT_BUNDLE_GET_URL")
    receipt_url = _required(environment, "BLUEPRINT_ISAAC_INPUT_RECEIPT_GET_URL")
    output_url = _required(environment, "BLUEPRINT_ISAAC_OUTPUT_BUNDLE_PUT_URL")
    input_digest = _required(environment, "BLUEPRINT_ISAAC_INPUT_BUNDLE_DIGEST")
    receipt_file_digest = _required(
        environment, "BLUEPRINT_ISAAC_INPUT_RECEIPT_FILE_DIGEST"
    )
    verification_request_digest = _required(
        environment, "BLUEPRINT_ISAAC_VERIFICATION_REQUEST_DIGEST"
    )
    worker_image = _required(environment, "BLUEPRINT_CONTAINER_IMAGE_DIGEST")
    source_commit = _required(environment, "BLUEPRINT_SOURCE_COMMIT")
    try:
        hard_ttl_seconds = int(
            _required(environment, "BLUEPRINT_RECONSTRUCTION_HARD_TTL_SECONDS")
        )
    except ValueError as exc:
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_ttl_invalid"]
        ) from exc
    if (
        _DIGEST.fullmatch(input_digest) is None
        or _DIGEST.fullmatch(receipt_file_digest) is None
        or _DIGEST.fullmatch(verification_request_digest) is None
        or _IMAGE.fullmatch(worker_image) is None
        or _COMMIT.fullmatch(source_commit) is None
    ):
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_binding_invalid"]
        )
    if not 300 <= hard_ttl_seconds <= 14_400:
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_ttl_invalid"]
        )
    root = Path(work_root)
    if root.is_symlink():
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_root_symlink_forbidden"]
        )
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    input_path = root / "isaac_verification_input.zip"
    receipt_path = root / "isaac_verification_input_receipt.json"
    output_path = root / "isaac_verification_output.zip"
    try:
        receipt_transfer = download_file(
            receipt_url,
            output_path=receipt_path,
            expected_sha256=receipt_file_digest,
            max_bytes=MAX_RECEIPT_BYTES,
            timeout_seconds=300,
            policy=presigned_transfer_policy(receipt_url),
        )
        receipt = validate_isaac_verification_worker_bundle_receipt(
            _load(receipt_path)
        )
    except (SafeOutboundHttpError, IsaacWorkerBundleError) as exc:
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_receipt_invalid"]
        ) from exc
    if (
        receipt.get("bundle_digest") != input_digest
        or receipt.get("isaac_verification_request_digest")
        != verification_request_digest
        or receipt.get("runtime_container_image_digest") != worker_image
        or receipt.get("source_commit_sha") != source_commit
    ):
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_receipt_binding_mismatch"]
        )
    try:
        input_transfer = download_file(
            input_url,
            output_path=input_path,
            expected_sha256=input_digest,
            max_bytes=MAX_INPUT_BUNDLE_BYTES,
            timeout_seconds=3600,
            policy=presigned_transfer_policy(input_url),
        )
        extraction = extract_isaac_verification_worker_bundle(
            bundle_path=input_path,
            bundle_receipt=receipt,
            output_root=root / "materialized",
        )
    except (SafeOutboundHttpError, IsaacWorkerBundleError) as exc:
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_input_invalid"]
        ) from exc
    materialized = root / "materialized" / input_digest[7:]
    bundle_root = root / "bundle"
    if bundle_root.exists() or not materialized.is_dir():
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_materialization_invalid"]
        )
    os.replace(materialized, bundle_root)
    out_root = root / "out"
    out_root.mkdir()
    command = _rebase_command(receipt["command"], root)
    exit_code = process_runner(
        command,
        root,
        root / "isaac_execution.log",
        max(60, hard_ttl_seconds - 120),
    )
    result_path = out_root / "isaac_runtime_result.json"
    if not result_path.is_file():
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_runtime_result_missing"]
        )
    runtime_status = _load(
        result_path, code="reconstruction_isaac_bootstrap_runtime_result_invalid"
    ).get("status")
    if (exit_code, runtime_status) not in {(0, "completed"), (2, "blocked")}:
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_runtime_exit_status_mismatch"]
        )
    try:
        output_receipt = compile_isaac_verification_output_bundle(
            bundle_receipt=receipt,
            runtime_output_root=out_root,
            output_path=output_path,
        )
        output_transfer = upload_file(
            output_url,
            input_path=output_path,
            expected_sha256=output_receipt["output_bundle_digest"],
            max_bytes=MAX_OUTPUT_BUNDLE_BYTES,
            timeout_seconds=3600,
            policy=presigned_transfer_policy(output_url, max_response_bytes=1024 * 1024),
            content_type="application/zip",
        )
    except (IsaacVerificationOutputBundleError, SafeOutboundHttpError) as exc:
        raise ReconstructionIsaacBootstrapError(
            ["reconstruction_isaac_bootstrap_output_invalid"]
        ) from exc
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "output_uploaded",
        "isaac_runtime_exit_code": exit_code,
        "isaac_verification_request_digest": verification_request_digest,
        "input_bundle_digest": input_transfer.sha256,
        "input_bundle_bytes": input_transfer.transferred_bytes,
        "input_receipt_file_digest": receipt_transfer.sha256,
        "input_extraction_receipt_digest": extraction["extraction_receipt_digest"],
        "output_bundle_digest": output_transfer.sha256,
        "output_bundle_bytes": output_transfer.transferred_bytes,
        "worker_image_digest": worker_image,
        "source_commit_sha": source_commit,
        "raw_secret_values_recorded": False,
        "scientific_qualification_inferred": False,
        "simulator_task_success_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
        "proof_effect": "none",
        "claim_ceiling": "isaac_runtime_transport_only",
    }
    result["bootstrap_receipt_digest"] = canonical_digest(
        result, digest_field="bootstrap_receipt_digest"
    )
    write_json(root / "reconstruction_isaac_bootstrap.v1.json", result)
    return result


def main() -> int:
    result = run_reconstruction_isaac_bootstrap(
        environment=os.environ,
        work_root="/workspace",
    )
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ReconstructionIsaacBootstrapError",
    "run_reconstruction_isaac_bootstrap",
]


if __name__ == "__main__":
    raise SystemExit(main())
