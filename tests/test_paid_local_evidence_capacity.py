from __future__ import annotations

from types import SimpleNamespace

import pytest

from blueprint_pipeline.paid_local_evidence_capacity import (
    materialize_local_closeout_reserve,
    measure_paid_local_evidence_capacity,
    release_local_closeout_reserve,
)


def test_capacity_accounts_for_input_replicas_and_closeout_reserve(tmp_path) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"x" * 100)
    usage = SimpleNamespace(total=10_000, used=9_000, free=999)

    result = measure_paid_local_evidence_capacity(
        evidence_root=tmp_path,
        immutable_input_paths=(bundle,),
        blocker="fixture_headroom_insufficient",
        minimum_free_bytes=500,
        input_replica_multiplier=2,
        closeout_reserve_bytes=500,
        disk_usage=lambda path: usage,
    )

    assert result["status"] == "blocked"
    assert result["minimum_free_bytes"] == 1_000
    assert result["immutable_input_bytes"] == 100
    assert result["blockers"] == ["fixture_headroom_insufficient"]


def test_capacity_rejects_unresolved_inputs_instead_of_underestimating(tmp_path) -> None:
    with pytest.raises(ValueError, match="adp_paid_local_evidence_input_missing"):
        measure_paid_local_evidence_capacity(
            evidence_root=tmp_path,
            immutable_input_paths=(tmp_path / "missing.zip",),
            blocker="fixture",
        )


def test_closeout_reserve_materializes_real_bytes_and_releases_them(tmp_path) -> None:
    reserve = tmp_path / "closeout.reserve"

    materialize_local_closeout_reserve(reserve, size_bytes=4097)

    assert reserve.is_file()
    assert reserve.stat().st_size == 4097
    assert reserve.read_bytes() == b"\0" * 4097
    release_local_closeout_reserve(reserve)
    assert not reserve.exists()
