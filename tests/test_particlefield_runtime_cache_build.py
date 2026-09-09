import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.particlefield_runtime_cache_build import (
    RUNTIME_SCHEMA, UPSTREAM_SOURCE_REVISION, _runtime,
)


@pytest.mark.parametrize("tamper", [None, "bytes", "revision", "receipt", "path", "symlink"])
def test_pinned_runtime_refuses_tampered_dependencies(tmp_path: Path, tamper) -> None:
    root = tmp_path / "runtime"
    (root / "source" / ".git").mkdir(parents=True)
    packages = root / "python-packages"
    packages.mkdir()
    member = packages / "dependency.py"
    member.write_bytes(b"pinned bytes")
    receipt = {
        "schema_version": RUNTIME_SCHEMA, "upstream_revision": UPSTREAM_SOURCE_REVISION,
        "files": [{"path": "python-packages/dependency.py",
                   "sha256": hashlib.sha256(member.read_bytes()).hexdigest()}],
    }
    if tamper == "revision":
        receipt["upstream_revision"] = "a" * 40
    if tamper == "path":
        receipt["files"][0]["path"] = "../dependency.py"
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if tamper == "receipt":
        receipt["receipt_digest"] = "sha256:" + "b" * 64
    if tamper == "bytes":
        member.write_bytes(b"changed")
    if tamper == "symlink":
        original = packages / "original.py"
        member.rename(original)
        member.symlink_to(original)
    (root / "runtime.json").write_text(json.dumps(receipt))
    if tamper is None:
        assert _runtime(root)["receipt_digest"] == receipt["receipt_digest"]
    else:
        with pytest.raises(ValueError, match="particlefield_transcode_runtime_"):
            _runtime(root)
