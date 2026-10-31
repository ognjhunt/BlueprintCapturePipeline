"""Exercise the actual capture malware scanner as the intake service account.

Deployment invokes this before activation. These are infrastructure controls,
never owner footage, reconstruction evidence, or robot evaluation episodes.
"""
from __future__ import annotations

import hashlib
import base64
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

# One scan loads the whole signature database: 25-50 s on the 4-core production host,
# and past a 60 s limit with its load average at 8 (2026-10-31), which blocked a deploy.
# This bounds a hung scanner; it is not a speed requirement.
SCANNER_CONTROL_TIMEOUT_SECONDS = 240
# What a caller of this module allows: both controls at their limit, and its own start.
SCANNER_CONTROLS_TIMEOUT_SECONDS = 2 * SCANNER_CONTROL_TIMEOUT_SECONDS + 30


class CaptureUploadRuntimeError(ValueError):
    """A secret-free, fail-closed runtime refusal."""

    def __init__(self, reason: str, diagnostic: dict | None = None):
        super().__init__(reason)
        self.diagnostic = diagnostic


def verify_capture_upload_runtime(*, store_root: str, scanner_argv: list[str]) -> dict:
    root = Path(store_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise CaptureUploadRuntimeError("capture_transfer_store_unavailable")
    info = root.stat()
    if info.st_uid not in {0, os.geteuid()} or stat.S_IMODE(info.st_mode) & 0o002:
        raise CaptureUploadRuntimeError("capture_transfer_store_permissions_unsafe")
    if (not isinstance(scanner_argv, list) or not scanner_argv
            or not all(isinstance(value, str) and value and "\x00" not in value for value in scanner_argv)
            or not Path(scanner_argv[0]).is_absolute()):
        raise CaptureUploadRuntimeError("capture_transfer_scanner_configuration_invalid")
    program = Path(scanner_argv[0])
    if not program.is_file() or not os.access(program, os.X_OK):
        raise CaptureUploadRuntimeError("capture_transfer_scanner_executable_missing")
    program_digest = hashlib.sha256(program.read_bytes()).hexdigest()
    evidence_root = root / "runtime-preflight"
    if evidence_root.is_symlink():
        raise CaptureUploadRuntimeError("capture_transfer_evidence_root_invalid")
    evidence_root.mkdir(mode=0o700, exist_ok=True)
    evidence_info = evidence_root.stat()
    if evidence_info.st_uid != os.geteuid() or stat.S_IMODE(evidence_info.st_mode) & 0o077:
        raise CaptureUploadRuntimeError("capture_transfer_evidence_root_invalid")
    # The standard benign EICAR signature is an independent negative control.
    # Split its literal so repository antivirus tools do not mistake this source
    # file for a downloaded test-signature artifact.
    controls = (("clean", b"Blueprint capture scanner control.\n", 0),
                ("known_test_signature", b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
                 b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*", 1))
    outcomes = {}
    try:
        with tempfile.TemporaryDirectory(prefix=".scanner-preflight-", dir=root) as temporary:
            for name, content, expected in controls:
                path = Path(temporary) / (name + ".txt")
                path.write_bytes(content)
                result = subprocess.run([*scanner_argv, str(path)], shell=False, check=False,
                                        capture_output=True, timeout=SCANNER_CONTROL_TIMEOUT_SECONDS)
                record = {"schema_version": "capture_scanner_control.v1", "control": name,
                          "input_sha256": hashlib.sha256(content).hexdigest(),
                          "source_module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "scanner_program_sha256": program_digest, "exit_code": result.returncode,
                          "stdout_base64": base64.b64encode(result.stdout).decode("ascii"),
                          "stderr_base64": base64.b64encode(result.stderr).decode("ascii")}
                payload = (json.dumps(record, sort_keys=True) + "\n").encode()
                evidence = evidence_root / (uuid.uuid4().hex + ".json")
                descriptor = os.open(evidence, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                diagnostic = {"control": name, "exit_code": result.returncode,
                              "evidence_relative_path": str(evidence.relative_to(root)),
                              "evidence_sha256": hashlib.sha256(payload).hexdigest()}
                if result.returncode != expected:
                    raise CaptureUploadRuntimeError("capture_transfer_scanner_control_failed:" + name, diagnostic)
                outcomes[name] = {"exit_code": result.returncode,
                                  "input_sha256": hashlib.sha256(content).hexdigest(), **diagnostic}
    except subprocess.TimeoutExpired as exc:
        raise CaptureUploadRuntimeError("capture_transfer_scanner_timed_out") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise CaptureUploadRuntimeError("capture_transfer_scanner_execution_failed") from exc
    return {"status": "ready", "store_root": str(root), "service_uid": os.geteuid(),
            "scanner_program": program.name,
            "scanner_program_sha256": program_digest,
            "controls": outcomes, "provider_mutation_performed": False,
            "source_footage_processed": False, "robot_episode_executed": False}


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(16385)
        if len(raw) > 16384:
            raise CaptureUploadRuntimeError("capture_transfer_runtime_configuration_too_large")
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"store_root", "scanner_argv"}:
            raise CaptureUploadRuntimeError("capture_transfer_runtime_configuration_invalid")
        result = verify_capture_upload_runtime(**value)
    except (CaptureUploadRuntimeError, TypeError, OSError, json.JSONDecodeError) as exc:
        reason = (str(exc) if isinstance(exc, CaptureUploadRuntimeError)
                  else "capture_transfer_runtime_io_failed" if isinstance(exc, OSError)
                  else "capture_transfer_runtime_configuration_invalid")
        print(json.dumps({"status": "blocked", "reason": reason,
                          **({"diagnostic": exc.diagnostic} if isinstance(exc, CaptureUploadRuntimeError)
                             and exc.diagnostic else {})}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
