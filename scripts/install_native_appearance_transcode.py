#!/usr/bin/env python3
"""Install the pinned CPU transcode import closure without changing the base venv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.nvidia_3dgrut_particlefield_transcode import (
    UPSTREAM_REPOSITORY,
    UPSTREAM_SOURCE_REVISION,
)
from blueprint_pipeline.particlefield_runtime_cache_build import (
    DEFAULT_RUNTIME_ROOT,
    RUNTIME_SCHEMA,
    _runtime,
)


def install(root: Path) -> dict:
    if root.exists():
        return _runtime(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements-native-appearance-transcode.txt"
    )
    with tempfile.TemporaryDirectory(prefix=".installing-", dir=root.parent) as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        packages = scratch / "python-packages"
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                UPSTREAM_REPOSITORY,
                str(source),
            ],
            check=True,
            timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(source), "checkout", UPSTREAM_SOURCE_REVISION],
            check=True,
            timeout=120,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(packages),
                "-r",
                str(requirements),
            ],
            check=True,
            timeout=300,
        )
        env = dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            WANDB_MODE="disabled",
            PYTHONPATH=os.pathsep.join((str(packages), str(source))),
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from threedgrut.export.scripts.transcode import transcode; "
                "from pxr import Usd; import torch; print('transcode_import_closure_passed')",
            ],
            env=env,
            check=True,
            timeout=60,
        )
        rows = []
        for member in sorted(packages.rglob("*")):
            if member.is_file():
                if member.is_symlink():
                    raise ValueError("transcode_install_symlink_refused")
                rows.append(
                    {
                        "path": str(member.relative_to(scratch)),
                        "sha256": hashlib.file_digest(member.open("rb"), "sha256").hexdigest(),
                    }
                )
        receipt = {
            "schema_version": RUNTIME_SCHEMA,
            "upstream_revision": UPSTREAM_SOURCE_REVISION,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
            "interpreter": sys.executable,
            "files": rows,
        }
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        (scratch / "runtime.json").write_text(json.dumps(receipt, indent=2) + "\n")
        _runtime(scratch)
        # The temporary context owns an empty container after the atomic rename.
        staged = scratch / "installed"
        staged.mkdir()
        for name in ("source", "python-packages", "runtime.json"):
            os.rename(scratch / name, staged / name)
        os.rename(staged, root)
    return _runtime(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    args = parser.parse_args()
    result = install(args.root)
    print(
        json.dumps(
            {
                "status": "installed",
                "root": str(args.root),
                "receipt_digest": result["receipt_digest"],
            }
        )
    )
