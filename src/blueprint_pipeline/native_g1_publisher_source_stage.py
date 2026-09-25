"""Stage the exact HumanoidArena source on a controller or GPU host.

The controller may run this on its own disk before buying GPU time. The
checkout contains publisher source only; model weights are staged separately
on the GPU. A completed receipt requires exact commit, clean tree, and pinned
policy-server/SONIC source bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_official_sonic_target_bridge import PINNED_ACTION_PROVIDER_SHA256
from .native_g1_policy_runtime_build import PYPROJECT_SHA256
from .native_g1_policy_server_supervisor import PINNED_SOURCE_REVISION, _source_revision
from .native_g1_run_preflight import PINNED_POLICY_SERVER_SHA256


SCHEMA = "native_g1_publisher_source_stage.v1"
SOURCE_REPOSITORY = "https://github.com/William-wAng618/HumanoidArena"
INVENTORY = (
    Path(__file__).resolve().parents[2] / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
)


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_publisher_source_file_missing_or_symlink")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git(*args: str, cwd: Path, timeout: int = 60) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("g1_publisher_source_git_command_failed") from exc
    return completed.stdout.strip()


def _inventory_identity() -> str:
    if INVENTORY.is_symlink() or not INVENTORY.is_file():
        raise ValueError("g1_publisher_source_inventory_missing")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if (
        inventory.get("schema_version") != "g1_humanoidarena_checkpoint_inventory.v1"
        or inventory.get("source_repository") != SOURCE_REPOSITORY
        or inventory.get("source_revision") != PINNED_SOURCE_REVISION
    ):
        raise ValueError("g1_publisher_source_inventory_identity_mismatch")
    return _sha256(INVENTORY)


def verify_g1_publisher_source(output_dir: Path) -> dict[str, Any]:
    """Verify an existing checkout without network access or mutation."""

    output = Path(output_dir)
    source = output / "source"
    if (
        not output.is_absolute()
        or output.is_symlink()
        or not output.is_dir()
        or source.is_symlink()
        or not source.is_dir()
    ):
        raise ValueError("g1_publisher_source_output_invalid")
    server = source / "lerobot/scripts/serve_lerobot_vla_http.py"
    sonic = source / "isaaclab_twist2_g1/action_provider/action_provider_sonic.py"
    pyproject = source / "lerobot/pyproject.toml"
    if (
        _git("remote", "get-url", "origin", cwd=source) != SOURCE_REPOSITORY
        or _source_revision(server) != PINNED_SOURCE_REVISION
        or _source_revision(sonic, expected_parent="action_provider") != PINNED_SOURCE_REVISION
        or _sha256(server) != "sha256:" + PINNED_POLICY_SERVER_SHA256
        or _sha256(sonic) != "sha256:" + PINNED_ACTION_PROVIDER_SHA256
        or _sha256(pyproject) != "sha256:" + PYPROJECT_SHA256
    ):
        raise ValueError("g1_publisher_source_identity_mismatch")
    receipt = {
        "schema_version": SCHEMA,
        "status": "ready",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": PINNED_SOURCE_REVISION,
        "source_root": str(source),
        "inventory_file_sha256": _inventory_identity(),
        "policy_server_sha256": "sha256:" + PINNED_POLICY_SERVER_SHA256,
        "sonic_provider_sha256": "sha256:" + PINNED_ACTION_PROVIDER_SHA256,
        "lerobot_pyproject_sha256": "sha256:" + PYPROJECT_SHA256,
        "model_weights_included": False,
        "gpu_allocated": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def stage_g1_publisher_source(output_dir: Path, *, verify_only: bool = False) -> dict[str, Any]:
    """Fetch a pinned clean checkout atomically, or verify one already staged."""

    output = Path(output_dir)
    if (
        not output.is_absolute()
        or output.is_symlink()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise ValueError("g1_publisher_source_output_invalid")
    _inventory_identity()
    if output.exists():
        return verify_g1_publisher_source(output)
    if verify_only:
        raise ValueError("g1_publisher_source_checkout_missing")
    with tempfile.TemporaryDirectory(prefix=".g1-publisher-source-", dir=output.parent) as temp:
        temporary = Path(temp)
        source = temporary / "source"
        source.mkdir()
        _git("init", "-q", cwd=source)
        _git("remote", "add", "origin", SOURCE_REPOSITORY, cwd=source)
        _git(
            "-c",
            "protocol.version=2",
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            PINNED_SOURCE_REVISION,
            cwd=source,
            timeout=1800,
        )
        _git("checkout", "--detach", "FETCH_HEAD", cwd=source, timeout=1800)
        # Verification before promotion prevents a partial or substituted
        # checkout from ever becoming the controller's reusable source cache.
        verify_g1_publisher_source(temporary)
        temporary.rename(output)
    receipt = verify_g1_publisher_source(output)
    (output / (SCHEMA + ".json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    receipt = stage_g1_publisher_source(args.output_dir, verify_only=args.verify_only)
    print(json.dumps({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
