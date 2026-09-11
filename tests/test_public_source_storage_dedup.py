import json
import os

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_source_storage_dedup import (
    ACK,
    plan_public_source_dedup,
    apply_public_source_dedup,
)
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha


def source(tmp_path, name, body=b"fixture immutable publisher bytes"):
    root = tmp_path / name
    root.mkdir()
    p = root / "source.ply"
    p.write_bytes(body)
    p.chmod(0o440)
    os.utime(p, (1, 1))
    receipt = {
        "schema_version": "public_scene_host_input_installation_receipt.v1",
        "status": "installed",
        "destination_root": str(root),
        "files": [
            {
                "role": "appearance_3dgs",
                "relative_path": "source.ply",
                "sha256": sha(p),
                "size_bytes": p.stat().st_size,
            }
        ],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    r = root / "public_scene_host_input_installation_receipt.v1.json"
    r.write_text(json.dumps(receipt))
    os.utime(r, (1, 1))
    return p, r


def test_duplicate_bytes_share_storage_but_every_path_and_receipt_survives(tmp_path):
    a, ra = source(tmp_path, "a")
    b, rb = source(tmp_path, "b")
    c, rc = source(tmp_path, "c", b"different publisher bytes")
    original = {p: p.read_bytes() for p in (a, b, c, ra, rb, rc)}
    kwargs = dict(
        inputs_root=tmp_path, minimum_size_bytes=1, now=300000, reference_checker=lambda _: False
    )
    plan = plan_public_source_dedup(**kwargs)
    assert len(plan["pairs"]) == 1 and plan["potential_reclaimed_bytes"] == a.stat().st_size
    result = apply_public_source_dedup(plan, ack=ACK, reference_checker=lambda _: False)
    assert len(result["applied"]) == 1 and a.stat().st_ino == b.stat().st_ino
    assert (
        all(p.read_bytes() == v for p, v in original.items()) and c.stat().st_ino != a.stat().st_ino
    )
    assert not plan_public_source_dedup(**kwargs)["pairs"]


@pytest.mark.parametrize("change", ["writable", "active", "tampered", "symlink"])
def test_unsafe_or_changed_sources_are_never_replaced(tmp_path, change):
    a, _ = source(tmp_path, "a")
    b, _ = source(tmp_path, "b")
    kwargs = dict(
        inputs_root=tmp_path, minimum_size_bytes=1, now=300000, reference_checker=lambda _: False
    )
    plan = plan_public_source_dedup(**kwargs)
    if change == "writable":
        b.chmod(0o640)
    elif change == "tampered":
        b.chmod(0o640)
        b.write_bytes(b"changed")
        b.chmod(0o440)
    elif change == "symlink":
        b.unlink()
        b.symlink_to(a)
    check = (lambda _: True) if change == "active" else (lambda _: False)
    with pytest.raises(ValueError):
        apply_public_source_dedup(plan, ack=ACK, reference_checker=check)
    assert a.read_bytes() == b"fixture immutable publisher bytes"
