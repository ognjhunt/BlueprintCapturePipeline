"""Run a reviewed, digest-pinned team JSONL policy in an isolated container.

The team image receives observations only over stdin. It has no network or
scene/evidence mounts. Provider allocation, model rights, and site-observation
authorization remain separate controller gates. This launcher first sends
synthetic G1 input; a successful probe is not a scored site episode.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_team_policy_conformance import run_g1_team_policy_synthetic_conformance
from .native_g1_team_policy_jsonl_client import NativeG1TeamPolicyJsonlClient
from .team_policy_delivery_profile import validate_team_policy_delivery_profile


_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,400}@sha256:[0-9a-f]{64}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NAME = re.compile(r"blueprint-team-policy-[0-9a-f]{32}\Z")
TEARDOWN_SCHEMA = "native_g1_team_container_teardown.v1"
CONFORMANCE_FILENAME = "native_g1_team_policy_synthetic_conformance.v1.json"


def isolated_container_command(
    *, image_ref: str, container_name: str, gpu_device: int | None
) -> list[str]:
    """Build the fixed Docker security profile; the team supplies no flags."""

    if (
        not isinstance(image_ref, str)
        or _IMAGE.fullmatch(image_ref) is None
        or not isinstance(container_name, str)
        or _NAME.fullmatch(container_name) is None
        or (gpu_device is not None and (type(gpu_device) is not int or gpu_device < 0))
    ):
        raise ValueError("g1_team_container_configuration_invalid")
    command = [
        "docker", "run", "--pull", "never", "--name", container_name,
        "--init", "--interactive", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534", "--pids-limit", "128", "--cpus", "4",
        "--memory", "16g", "--shm-size", "1g",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=1024m",
        "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp",
    ]
    if gpu_device is not None:
        command.extend(["--gpus", f"device={gpu_device}"])
    command.append(image_ref)
    return command


def _verified_local_image(image_ref: str) -> str:
    """Require the exact repository digest already provisioned on this host."""

    if not isinstance(image_ref, str) or _IMAGE.fullmatch(image_ref) is None:
        raise ValueError("g1_team_container_image_invalid")
    try:
        digests = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image_ref],
            capture_output=True, text=True, timeout=20, check=False,
        )
        image_id = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image_ref],
            capture_output=True, text=True, timeout=20, check=False,
        )
        values = json.loads(digests.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError) as exc:
        raise ValueError("g1_team_container_image_inspection_failed") from exc
    if (
        digests.returncode != 0
        or image_id.returncode != 0
        or not isinstance(values, list)
        or image_ref not in values
        or _DIGEST.fullmatch(image_id.stdout.strip()) is None
    ):
        raise ValueError("g1_team_container_local_image_identity_invalid")
    return image_id.stdout.strip()


class NativeG1TeamContainerLease:
    """Own one policy process and record force-removal on every close path."""

    def __init__(
        self,
        *, process: subprocess.Popen[bytes],
        client: NativeG1TeamPolicyJsonlClient,
        container_name: str,
        image_ref: str,
        image_id: str,
        profile_digest: str,
        output_dir: Path,
        stderr_file: Any,
    ) -> None:
        self.process = process
        self.client = client
        self.container_name = container_name
        self.image_ref = image_ref
        self.image_id = image_id
        self.profile_digest = profile_digest
        self.output_dir = output_dir
        self._stderr_file = stderr_file
        self._closed: dict[str, Any] | None = None

    def close(self) -> dict[str, Any]:
        if self._closed is not None:
            return self._closed
        process_error: str | None = None
        try:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            process_error = type(exc).__name__
        finally:
            self._stderr_file.close()
        try:
            removal = subprocess.run(
                ["docker", "rm", "--force", self.container_name],
                capture_output=True, text=True, timeout=20, check=False,
            )
            inspect = subprocess.run(
                ["docker", "container", "inspect", self.container_name],
                capture_output=True, text=True, timeout=20, check=False,
            )
            absent = inspect.returncode != 0 and (
                "No such object:" in inspect.stderr
                or "No such container:" in inspect.stderr
            )
            remove_exit = removal.returncode
        except (OSError, subprocess.TimeoutExpired):
            absent = False
            remove_exit = None
        receipt = {
            "schema_version": TEARDOWN_SCHEMA,
            "status": "container_removed" if absent else "teardown_blocked",
            "container_name": self.container_name,
            "image_ref": self.image_ref,
            "local_image_id": self.image_id,
            "profile_digest": self.profile_digest,
            "docker_remove_exit_code": remove_exit,
            "container_absent_verified": absent,
            "process_exit_code": self.process.poll(),
            "process_close_error": process_error,
            "raw_stderr_quarantined": True,
            "provider_teardown_verified": False,
            "claim_ceiling": "development_only",
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        path = self.output_dir / (TEARDOWN_SCHEMA + ".json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self._closed = receipt
        return receipt


def launch_g1_team_container_synthetic_probe(
    *,
    profile: Mapping[str, Any],
    trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
    operator_approved_profile_digest: str,
    operator_approved_image_ref: str,
    output_dir: Path,
    gpu_device: int | None,
) -> tuple[NativeG1TeamContainerLease, dict[str, Any]]:
    """Start an approved local image and prove its bound synthetic G1 wire."""

    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=trusted_setup, authenticated_owner=authenticated_owner
    )
    delivery = bound["delivery"]
    if (
        sys.platform != "linux"
        or delivery["mode"] != "container"
        or bound["profile_digest"] != operator_approved_profile_digest
        or delivery["image_ref"] != operator_approved_image_ref
        or not isinstance(output_dir, Path)
        or not output_dir.is_absolute()
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise ValueError("g1_team_container_admission_invalid")
    image_ref = delivery["image_ref"]
    image_id = _verified_local_image(image_ref)
    name = "blueprint-team-policy-" + uuid.uuid4().hex
    command = isolated_container_command(
        image_ref=image_ref, container_name=name, gpu_device=gpu_device
    )
    output_dir.mkdir(mode=0o700)
    stderr_path = output_dir / "team_policy_raw_stderr.quarantined.log"
    descriptor = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    stderr_file = os.fdopen(descriptor, "wb")
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_file,
            bufsize=0,
        )
        client = NativeG1TeamPolicyJsonlClient(
            process, timeout_seconds=30.0, profile_digest=bound["profile_digest"]
        )
    except BaseException:
        stderr_file.close()
        # A Docker client can fail after the daemon created the named child.
        try:
            subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True, timeout=20,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        raise
    lease = NativeG1TeamContainerLease(
        process=process, client=client, container_name=name, image_ref=image_ref,
        image_id=image_id, profile_digest=bound["profile_digest"],
        output_dir=output_dir, stderr_file=stderr_file,
    )
    try:
        conformance = run_g1_team_policy_synthetic_conformance(
            profile=bound,
            trusted_setup=trusted_setup,
            authenticated_owner=authenticated_owner,
            policy_client=client,
        )
        with (output_dir / CONFORMANCE_FILENAME).open("x", encoding="utf-8") as stream:
            json.dump(conformance, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        lease.close()
        raise
    return lease, conformance
