"""Bounded, atomic CPU creation of the pinned native appearance cache."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import sha256_file
from .decision_evidence_contracts import canonical_digest
from .nvidia_3dgrut_particlefield_transcode import UPSTREAM_SOURCE_REVISION
from .particlefield_runtime_asset_cache import (
    DEFAULT_CACHE_ROOT,
    cache_entry_root,
    materialize_cached_particlefield,
    publish_particlefield_runtime_asset,
)

MAXIMUM_AUTOMATIC_SOURCE_BYTES = 128 * 1024**2
CONVERSION_TIMEOUT_SECONDS = 180
RUNTIME_SCHEMA = "native_appearance_transcode_runtime.v1"
DEFAULT_RUNTIME_ROOT = Path(
    "/var/lib/blueprint/task-evaluation-inputs/system-runtimes/"
    f"native-appearance-transcode/{UPSTREAM_SOURCE_REVISION}"
)


def _runtime(root: Path) -> dict:
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("particlefield_transcode_runtime_unsafe")
    try:
        receipt = json.loads((root / "runtime.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("particlefield_transcode_runtime_missing_or_unreadable") from exc
    if (
        receipt.get("schema_version") != RUNTIME_SCHEMA
        or receipt.get("upstream_revision") != UPSTREAM_SOURCE_REVISION
        or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
        or not (root / "source" / ".git").is_dir()
        or not (root / "python-packages").is_dir()
    ):
        raise ValueError("particlefield_transcode_runtime_invalid")
    for row in receipt.get("files", []):
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("particlefield_transcode_runtime_member_invalid")
        member = root / relative
        if member.is_symlink() or sha256_file(member) != row["sha256"]:
            raise ValueError("particlefield_transcode_runtime_member_invalid")
    if not receipt.get("files"):
        raise ValueError("particlefield_transcode_runtime_files_missing")
    return receipt


def _convert(*, source: Path, source_digest: str, output: Path, runtime_root: Path) -> None:
    _runtime(runtime_root)
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
        PYTHONPATH=os.pathsep.join(
            (
                str(Path(__file__).resolve().parents[1]),
                str(runtime_root / "python-packages"),
                str(runtime_root / "source"),
            )
        ),
    )
    with (output / "conversion.log").open("wb") as log:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    __name__,
                    "--convert",
                    "--source",
                    str(source),
                    "--source-digest",
                    source_digest,
                    "--output",
                    str(output),
                    "--runtime-root",
                    str(runtime_root),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=CONVERSION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("particlefield_transcode_timeout") from exc
    if completed.returncode:
        raise ValueError(f"particlefield_transcode_failed:exit_{completed.returncode}")


def materialize_automatic_particlefield(
    *,
    source_path: Path,
    source_digest: str,
    output_root: Path,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    runtime_root: Path | None = None,
    converter=None,
) -> dict:
    """Build once from sealed source; failures cannot publish a cache entry."""
    source = Path(source_path)
    if (
        source.is_symlink()
        or not source.is_file()
        or source.stat().st_size > MAXIMUM_AUTOMATIC_SOURCE_BYTES
        or "sha256:" + sha256_file(source) != source_digest
    ):
        raise ValueError("particlefield_automatic_source_invalid_or_over_limit")
    cache_root = Path(cache_root)
    if cache_root.is_symlink() or any(p.is_symlink() for p in cache_root.parents):
        raise ValueError("particlefield_automatic_cache_unsafe")
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_entry_root(source_digest, cache_root=cache_root)
    lock = cache_root / (destination.name + ".build.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o640)
    with os.fdopen(fd, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        cached = materialize_cached_particlefield(
            source_digest=source_digest, output_root=output_root, cache_root=cache_root
        )
        if cached is not None:
            return cached
        if destination.exists() or destination.is_symlink():
            raise ValueError("particlefield_automatic_cache_incomplete")
        runtime = runtime_root or Path(
            os.environ.get("BLUEPRINT_NATIVE_APPEARANCE_TRANSCODE_ROOT", str(DEFAULT_RUNTIME_ROOT))
        )
        with tempfile.TemporaryDirectory(prefix=".building-", dir=cache_root) as temporary:
            scratch = Path(temporary)
            try:
                (converter or _convert)(
                    source=source,
                    source_digest=source_digest,
                    output=scratch,
                    runtime_root=Path(runtime),
                )
                published = publish_particlefield_runtime_asset(
                    source_digest=source_digest,
                    particlefield_path=scratch / "scene_appearance.usdc",
                    authoring_receipt_path=scratch / "particlefield_authoring_receipt.v1.json",
                    cache_root=scratch / "validated-cache",
                )
            except Exception as exc:
                failure = cache_root / "failed-builds" / scratch.name
                failure.mkdir(parents=True)
                if (scratch / "conversion.log").is_file():
                    shutil.copyfile(scratch / "conversion.log", failure / "conversion.log")
                (failure / "failure.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "particlefield_runtime_cache_build_failure.v1",
                            "source_digest": source_digest,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "cache_published": False,
                        }
                    )
                )
                raise
            os.rename(published["root"], destination)
        cached = materialize_cached_particlefield(
            source_digest=source_digest, output_root=output_root, cache_root=cache_root
        )
        if cached is None:
            raise ValueError("particlefield_automatic_cache_publication_missing")
        return cached


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert", action="store_true", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args(argv)
    _runtime(args.runtime_root)
    from .nvidia_3dgrut_particlefield_transcode import write_direct_particlefield_from_nurec

    result = write_direct_particlefield_from_nurec(
        args.source,
        args.output / "scene_appearance.usdc",
        expected_source_sha256=args.source_digest,
        receipt_path=args.output / "particlefield_authoring_receipt.v1.json",
        source_root=args.runtime_root / "source",
    )
    print(json.dumps({"status": result.get("status"), "blockers": result.get("blockers", [])}))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
