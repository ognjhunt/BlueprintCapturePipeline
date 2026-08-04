from __future__ import annotations

from pathlib import Path

import pytest

import blueprint_pipeline.adp_isaac_lab_arena_materialization as materialization
from blueprint_pipeline.adp_founder_sim_protocol import (
    APPROVAL_SCHEMA_VERSION,
    PROTOCOL_ID,
    admit_founder_sim_execution,
    build_founder_sim_protocol,
)
from blueprint_pipeline.adp_isaac_lab_arena_materialization import (
    REQUIRED_BYTE_GROUPS,
    SOURCE_REVISIONS,
    ArenaMaterializationError,
    admit_materialized_worker,
    build_materialization_receipt,
    verify_materialization_receipt,
)


def _inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    sources: dict[str, Path] = {}
    for name in SOURCE_REVISIONS:
        checkout = tmp_path / name
        checkout.mkdir()
        (checkout / "source.txt").write_text(name, encoding="utf-8")
        if name == "arena_source":
            (checkout / "uv.lock").write_text("frozen Arena lock", encoding="utf-8")
        sources[name] = checkout

    groups: dict[str, Path] = {}
    for name in REQUIRED_BYTE_GROUPS:
        root = tmp_path / name
        root.mkdir()
        (root / "bytes.bin").write_bytes(f"{name}-bytes".encode())
        groups[name] = root
    return sources, groups


@pytest.fixture
def pinned_source_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_git_output(checkout: Path, *arguments: str) -> str:
        name = checkout.name
        if arguments == ("rev-parse", "HEAD"):
            return SOURCE_REVISIONS[name]
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "1" * 40
        if arguments == ("submodule", "status", "--recursive"):
            return ""
        raise AssertionError(arguments)

    original_sha256_file = materialization._sha256_file
    expected_lock = build_founder_sim_protocol()["scene"]["simulator_stack"][
        "isaac_lab_arena"
    ]["uv_lock_sha256"]

    def fake_sha256_file(path: Path) -> str:
        if path.name == "uv.lock":
            return expected_lock
        return original_sha256_file(path)

    monkeypatch.setattr(materialization, "_git_output", fake_git_output)
    monkeypatch.setattr(materialization, "_sha256_file", fake_sha256_file)


def _founder_execution_admission() -> dict[str, object]:
    protocol = build_founder_sim_protocol()
    return admit_founder_sim_execution(
        protocol,
        {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "approved": True,
            "approver_role": "blueprint_founder_sim_owner",
            "protocol_id": PROTOCOL_ID,
            "protocol_digest": protocol["protocol_digest"],
        },
    )


def test_receipt_binds_every_source_asset_checkpoint_and_runtime_group(
    tmp_path: Path, pinned_source_probes: None
) -> None:
    sources, groups = _inputs(tmp_path)
    receipt = build_materialization_receipt(
        source_checkouts=sources,
        byte_group_roots=groups,
        isaac_sim_version="6.0.0.1",
    )

    assert receipt["status"] == "verified_from_local_worker_bytes"
    assert set(receipt["sources"]) == set(SOURCE_REVISIONS)
    assert set(receipt["byte_groups"]) == set(REQUIRED_BYTE_GROUPS)
    assert receipt["candidate_bindings"]["baseline"]["candidate_id"] == (
        "pi05_droid_jointpos_polaris"
    )
    assert receipt["candidate_bindings"]["alternative"]["candidate_id"] == (
        "nvidia/GR00T-N1.7-DROID"
    )
    assert receipt["candidate_jobs_authorized"] is False
    assert receipt["paid_compute_authorized"] is False
    assert verify_materialization_receipt(
        receipt,
        source_checkouts=sources,
        byte_group_roots=groups,
    ) == receipt


def test_changed_worker_byte_invalidates_existing_receipt(
    tmp_path: Path, pinned_source_probes: None
) -> None:
    sources, groups = _inputs(tmp_path)
    receipt = build_materialization_receipt(
        source_checkouts=sources,
        byte_group_roots=groups,
        isaac_sim_version="6.0.0.1",
    )
    (groups["groot_checkpoint"] / "bytes.bin").write_bytes(b"changed checkpoint")

    with pytest.raises(
        ArenaMaterializationError, match="materialization_receipt_not_current_for_worker"
    ):
        verify_materialization_receipt(
            receipt,
            source_checkouts=sources,
            byte_group_roots=groups,
        )


def test_materialized_worker_admits_controls_but_never_candidate_jobs_or_spend(
    tmp_path: Path, pinned_source_probes: None
) -> None:
    sources, groups = _inputs(tmp_path)
    receipt = build_materialization_receipt(
        source_checkouts=sources,
        byte_group_roots=groups,
        isaac_sim_version="6.0.0.1",
    )
    admission = admit_materialized_worker(
        founder_execution_admission=_founder_execution_admission(),
        receipt=receipt,
        source_checkouts=sources,
        byte_group_roots=groups,
    )

    assert admission["status"] == "materialized_pending_native_controls"
    assert admission["native_control_canaries_authorized"] is True
    assert admission["candidate_jobs_authorized"] is False
    assert admission["paid_compute_authorized"] is False


@pytest.mark.parametrize(
    ("sources_to_remove", "groups_to_remove", "expected"),
    [
        ({"openpi_source"}, set(), "materialization_source_groups_not_exact"),
        (set(), {"arena_registry_hdr"}, "materialization_byte_groups_not_exact"),
    ],
)
def test_materialization_rejects_any_missing_required_group(
    tmp_path: Path,
    pinned_source_probes: None,
    sources_to_remove: set[str],
    groups_to_remove: set[str],
    expected: str,
) -> None:
    sources, groups = _inputs(tmp_path)
    for name in sources_to_remove:
        sources.pop(name)
    for name in groups_to_remove:
        groups.pop(name)

    with pytest.raises(ArenaMaterializationError, match=expected):
        build_materialization_receipt(
            source_checkouts=sources,
            byte_group_roots=groups,
            isaac_sim_version="6.0.0.1",
        )


def test_materialization_rejects_runtime_version_or_founder_digest_mismatch(
    tmp_path: Path, pinned_source_probes: None
) -> None:
    sources, groups = _inputs(tmp_path)
    with pytest.raises(
        ArenaMaterializationError, match="materialization_isaac_sim_version_mismatch"
    ):
        build_materialization_receipt(
            source_checkouts=sources,
            byte_group_roots=groups,
            isaac_sim_version="6.0.1",
        )

    receipt = build_materialization_receipt(
        source_checkouts=sources,
        byte_group_roots=groups,
        isaac_sim_version="6.0.0.1",
    )
    admission = _founder_execution_admission()
    admission["protocol_digest"] = "sha256:" + "0" * 64
    with pytest.raises(
        ArenaMaterializationError, match="materialization_founder_admission_not_canonical"
    ):
        admit_materialized_worker(
            founder_execution_admission=admission,
            receipt=receipt,
            source_checkouts=sources,
            byte_group_roots=groups,
        )

    admission = _founder_execution_admission()
    admission["execution_admission_digest"] = "sha256:" + "1" * 64
    with pytest.raises(
        ArenaMaterializationError, match="materialization_founder_admission_not_canonical"
    ):
        admit_materialized_worker(
            founder_execution_admission=admission,
            receipt=receipt,
            source_checkouts=sources,
            byte_group_roots=groups,
        )
