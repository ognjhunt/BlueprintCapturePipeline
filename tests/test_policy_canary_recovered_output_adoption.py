# Covers (for impacted-test selection):
#   src/blueprint_pipeline/policy_canary_recovered_output_adoption.py
#   src/blueprint_pipeline/task_evaluation_policy_canary_dispatcher.py
#   src/blueprint_pipeline/native_task_arena_policy_canary_worker.py
#   src/blueprint_pipeline/policy_canary_worker_evidence.py
#   src/blueprint_pipeline/provider_output_member_view.py
"""A complete SSH-recovered canary is adopted by its member index when no local archive remains.

In stream mode the recovered ZIP is published to B2 and removed behind its
pointer; only the contract's JSON is on the host. Adoption then checks the
archive by its member index and aggregates the ten cells into the same result
download mode produces from the whole extracted tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import PROVIDER_RESULT_FILENAME
from blueprint_pipeline.provider_output_member_view import open_member_view
from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import (
    _recovered_complete_policy_canary_result,
)
from tests.provider_output_fixtures import serve_member_views, stream_evidence_tree, zip_tree
from tests.test_task_evaluation_policy_canary_setup import _setup as public_setup

CANDIDATES = ("pi05_droid", "groot_n17_droid")
CAMERAS = ("external", "wrist", "overview")
LINEAGE = "compiled_configured_scene_diagnostic"


def _write(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"size_bytes": len(data), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}


def _recovered_tree(root: Path, *, mp4_count: int = 120) -> dict:
    """A Quick-10 output whose top-level result was never written: ten sealed cells and their media."""
    setup = public_setup()
    cells = [{"cell_id": f"cell-{index}", "seed": 3100 + index} for index in range(10)]
    written = 0
    for index, cell in enumerate(cells):
        cell_root = root / "cell_runs" / f"{index:02d}"
        episodes = []
        for candidate in CANDIDATES:
            media = f"episodes/media/{cell['cell_id']}-{candidate}"
            videos = {}
            for camera in CAMERAS:
                if written < mp4_count:
                    videos[camera] = {"relative_path": f"{media}/{camera}.mp4",
                                      **_write(cell_root / media / f"{camera}.mp4", f"{index}{candidate}{camera}".encode() * 50)}
                    written += 1
            _write(cell_root / media / "frames/external/000000.png", bytes(range(256)) * (index + 1))
            _write(cell_root / media / "policy-requests/0000.json", b'{"observation": [0.0]}' * 20)
            score = {"relative_path": f"episodes/{cell['cell_id']}-{candidate}.score_receipt.json",
                     **_write(cell_root / f"episodes/{cell['cell_id']}-{candidate}.score_receipt.json",
                              json.dumps({"task_succeeded": index % 2 == 0}).encode())}
            episodes.append({"candidate_id": candidate, "cell_id": cell["cell_id"], "seed": cell["seed"],
                             "status": "completed", "candidate_policy_queried": True,
                             "telemetry": {"completed_at_unix_ns": 1_700_000_000_000_000_000 + index},
                             "evidence_artifacts": {"review_video": videos.get("external"), "score_receipt": score}})
        for control in ("zero_action_negative", "deterministic_scripted_positive"):
            for camera in CAMERAS:
                if written < mp4_count:
                    _write(root / "control_runs" / f"{index:02d}" / f"media/{control}/{camera}.mp4",
                           f"control{index}{control}{camera}".encode() * 40)
                    written += 1
        _write(cell_root / "worker_console.log", b"cell started\n" * 10)
        _write(cell_root / "cell_progress.log", f"cell {index} done\n".encode())
        child = {"selected_cell_index": index, "status": "runtime_selected_cell_completed_pending_aggregation",
                 "construction_lineage_mode": LINEAGE, "task_success_contract": setup["task_success_contract"],
                 "task_success_contract_digest": setup["task_success_contract_digest"],
                 "episodes": episodes, "result_digest": ""}
        child["result_digest"] = canonical_digest(child, digest_field="result_digest")
        _write(cell_root / PROVIDER_RESULT_FILENAME, json.dumps(child, sort_keys=True).encode())
    inputs = {"candidate_ids": list(CANDIDATES), "cells": cells, "matrix_digest": "sha256:" + "a" * 64,
              "task_success_contract": setup["task_success_contract"],
              "task_success_contract_digest": setup["task_success_contract_digest"]}
    return inputs


def _command(attempt: Path, archive_bytes: bytes, *, local_archive: bool, recovered_sha256: str | None = None,
             recovered_size: int | None = None) -> Path:
    run = attempt / "vast_provider_run"
    run.mkdir(parents=True, exist_ok=True)
    archive = run / "vast_provider_runtime_output.zip"
    if local_archive:
        archive.write_bytes(archive_bytes)
    command = {
        "provider_bundle_kind": "native_task_arena_policy_canary_session",
        "provider_runtime_output_zip_received": True,
        "provider_runtime_output_zip_path": str(archive),
        "provider_output_download_manifest": {"ssh_recovery": {
            "status": "completed", "strict_host_key_checking": True, "streamed_to_disk": True,
            "recovered_size_bytes": recovered_size if recovered_size is not None else len(archive_bytes),
            "recovered_sha256": recovered_sha256 or "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
            "known_hosts_sha256": "a" * 64}},
        "provider_runtime_output_zip_inspection": {"zip_present": True, "mp4_count": 120},
    }
    (run / "vast_provider_command_result.json").write_text(json.dumps(command), encoding="utf-8")
    return archive


def _adopt(root: Path, attempt: Path, inputs: dict):
    return _recovered_complete_policy_canary_result(
        root=root, native_path=(attempt / "immutable_execution" / PROVIDER_RESULT_FILENAME).resolve(),
        adapter={"attempt_root": str(attempt)}, authority={"run_id": "scene-839873-recovered"},
        runtime_inputs=inputs)


def _listing(root: Path) -> list[tuple[str, int]]:
    return sorted((path.relative_to(root).as_posix(), path.stat().st_size)
                  for path in root.rglob("*") if path.is_file())


def _cases(tmp_path: Path, monkeypatch, *, mp4_count: int = 120):
    source = tmp_path / "provider_output"
    inputs = _recovered_tree(source, mp4_count=mp4_count)
    archive_bytes = zip_tree(source).to_bytes()
    download_root = tmp_path / "download"
    download_attempt = download_root / "allocator/attempts/attempt_001"
    shutil.copytree(source, download_attempt / "immutable_execution")
    _command(download_attempt, archive_bytes, local_archive=True)
    stream_root = tmp_path / "streamed"
    streamed = stream_evidence_tree(source, stream_root / "allocator/attempts/attempt_001")
    serve_member_views(monkeypatch, streamed.store)
    return inputs, archive_bytes, (download_root, download_attempt), (stream_root, streamed)


def test_streamed_ssh_recovery_adopts_ten_cells_by_index_digest(tmp_path, monkeypatch):
    inputs, archive_bytes, (download_root, download_attempt), (stream_root, streamed) = _cases(tmp_path, monkeypatch)
    _command(streamed.attempt, archive_bytes, local_archive=False)  # removed after its verified promotion
    evidence_before = _listing(streamed.evidence)

    downloaded, _ = _adopt(download_root, download_attempt, inputs)
    adopted, adopted_path = _adopt(stream_root, streamed.attempt, inputs)

    # The same aggregate, down to every artifact row and the result digest.
    assert adopted == downloaded
    assert adopted["status"] == "runtime_completed_unqualified_pending_closeout" and len(adopted["episodes"]) == 20
    rows = {row["relative_path"]: row for row in adopted["artifact_inventory"]}
    video = "cell_runs/00/episodes/media/cell-0-pi05_droid/external.mp4"
    assert rows[video]["sha256"] == streamed.rows[video]["sha256"]
    adoption = stream_root / "recovered_provider_output_adoption"
    assert adopted_path == adoption / PROVIDER_RESULT_FILENAME
    assert not any(path.suffix in {".mp4", ".png"} for path in adoption.rglob("*"))
    # Adopted JSON is a writable copy, never a hard link to the 0440 member (review I8).
    for root, attempt_evidence in ((adoption, streamed.evidence),
                                   (download_root / "recovered_provider_output_adoption",
                                    download_attempt / "immutable_execution")):
        child = root / "cell_runs/00" / PROVIDER_RESULT_FILENAME
        original = attempt_evidence / "cell_runs/00" / PROVIDER_RESULT_FILENAME
        assert child.read_bytes() == original.read_bytes()
        assert os.stat(child).st_ino != os.stat(original).st_ino and os.stat(child).st_nlink == 1
        assert stat.S_IMODE(os.stat(child).st_mode) & stat.S_IWUSR
    # Readers of the adoption root find its members through a sibling descriptor.
    view = open_member_view(adoption / video)
    assert view.evidence_root == adoption.resolve() and view.digest(video) == streamed.rows[video]["sha256"]
    receipt = json.loads((stream_root / "recovered_provider_output_adoption.json").read_text())
    assert receipt["status"] == "adopted_complete_provider_output" and receipt["mp4_count"] == 120
    assert receipt["archive"] == {
        "location": "durable_archive", "sha256": streamed.index["archive"]["sha256"],
        "size_bytes": streamed.index["archive"]["size"],
        "durable_reference": streamed.index["archive"]["durable_reference"],
        "member_index_digest": streamed.index["index_digest"]}
    assert receipt["adoption_digest"] == canonical_digest(receipt, digest_field="adoption_digest")
    # Children were compared by digest: no member byte came from B2, nothing was written to the evidence root.
    assert streamed.data_ranges() == []
    assert _listing(streamed.evidence) == evidence_before
    # A second pass returns the sealed aggregate.
    assert _adopt(stream_root, streamed.attempt, inputs) == (adopted, adopted_path)


@pytest.mark.parametrize("mismatch", ["archive_sha256", "archive_size", "mp4_count", "child_digest", "no_view"])
def test_reference_adoption_refuses_a_mismatched_archive_or_mp4_count(tmp_path, monkeypatch, mismatch):
    inputs, archive_bytes, _, (stream_root, streamed) = _cases(
        tmp_path, monkeypatch, mp4_count=119 if mismatch == "mp4_count" else 120)
    _command(streamed.attempt, archive_bytes, local_archive=False,
             recovered_sha256="sha256:" + "0" * 64 if mismatch == "archive_sha256" else None,
             recovered_size=len(archive_bytes) + 1 if mismatch == "archive_size" else None)
    if mismatch == "child_digest":
        child = streamed.evidence / "cell_runs/03" / PROVIDER_RESULT_FILENAME
        os.chmod(child, 0o640)
        child.write_bytes(child.read_bytes() + b" ")
    if mismatch == "no_view":
        (streamed.attempt / "immutable_execution.member_view.v1.json").unlink()

    assert _adopt(stream_root, streamed.attempt, inputs) is None
    assert not (stream_root / "recovered_provider_output_adoption").exists()
    assert not (stream_root / "recovered_provider_output_adoption.json").exists()


def test_download_adoption_links_bulk_and_policy_requests_and_fails_closed_on_a_failed_link(
        tmp_path, monkeypatch):
    """Adoption copies only what the aggregator could rewrite. Bulk media and the policy
    requests (bulk to the GC, possibly hundreds of MB) stay hard links, and a link that fails
    is the adoption's copy failure, never a silent full copy."""
    from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import TaskEvaluationPolicyCanaryDispatchError

    inputs, _, (download_root, download_attempt), _ = _cases(tmp_path, monkeypatch)
    evidence = download_attempt / "immutable_execution"
    request = next(evidence.glob("cell_runs/00/episodes/media/*/policy-requests/0000.json"))
    video = next(evidence.glob("cell_runs/00/episodes/media/*/external.mp4"))
    child = evidence / "cell_runs/00" / PROVIDER_RESULT_FILENAME

    assert _adopt(download_root, download_attempt, inputs) is not None

    adopted = download_root / "recovered_provider_output_adoption"
    for original in (request, video):
        assert os.stat(adopted / original.relative_to(evidence)).st_ino == os.stat(original).st_ino
    assert os.stat(adopted / child.relative_to(evidence)).st_ino != os.stat(child).st_ino

    shutil.rmtree(adopted)
    (download_root / "recovered_provider_output_adoption.json").unlink()
    real_link = os.link

    def refuse(source, destination, **kwargs):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", refuse)
    with pytest.raises(TaskEvaluationPolicyCanaryDispatchError,
                       match="^policy_canary_recovered_output_adoption_copy_failed$"):
        _adopt(download_root, download_attempt, inputs)
    monkeypatch.setattr(os, "link", real_link)
    assert not (download_root / "recovered_provider_output_adoption.json").exists()
