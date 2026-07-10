from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.output_run_transaction import (
    OutputRunTransaction,
    verify_output_run_commit,
)


def test_output_run_commit_binds_exact_inventory_and_detects_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "package"
    with OutputRunTransaction(
        output,
        lane="test-package",
        request_fingerprint="request-sha-1",
    ) as transaction:
        write_json(output / "manifest.json", {"status": "completed"})
        (output / "artifact.bin").write_bytes(b"artifact")
        commit = transaction.commit()

    assert commit["status"] == "committed"
    assert verify_output_run_commit(
        output,
        expected_request_fingerprint="request-sha-1",
    )["status"] == "passed"

    (output / "artifact.bin").write_bytes(b"tampered")
    verification = verify_output_run_commit(output)
    assert verification["status"] == "blocked"
    assert "output_run_inventory_digest_mismatch" in verification["blockers"]


def test_failed_run_has_no_commit_and_resets_previous_outputs(tmp_path: Path) -> None:
    output = tmp_path / "package"
    output.mkdir()
    write_json(output / "old-complete.json", {"status": "old"})
    write_json(output / "run_commit.json", {"status": "committed"})

    with pytest.raises(RuntimeError, match="fault-injected"):
        with OutputRunTransaction(
            output,
            lane="test-package",
            request_fingerprint="request-sha-2",
        ):
            write_json(output / "partial.json", {"status": "partial"})
            raise RuntimeError("fault-injected")

    assert not (output / "run_commit.json").exists()
    assert not (output / "old-complete.json").exists()
    lease = json.loads((tmp_path / ".package.run-lease.json").read_text())
    assert lease["status"] == "failed_uncommitted"


def test_concurrent_runs_are_serialized_and_never_mix(tmp_path: Path) -> None:
    output = tmp_path / "package"
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first() -> None:
        with OutputRunTransaction(
            output,
            lane="test-package",
            request_fingerprint="first",
        ) as transaction:
            write_json(output / "first.json", {"run": "first"})
            first_entered.set()
            assert release_first.wait(timeout=5)
            transaction.commit()

    def second() -> None:
        assert first_entered.wait(timeout=5)
        with OutputRunTransaction(
            output,
            lane="test-package",
            request_fingerprint="second",
        ) as transaction:
            second_entered.set()
            write_json(output / "second.json", {"run": "second"})
            transaction.commit()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_entered.wait(timeout=5)
    assert not second_entered.wait(timeout=0.1)
    release_first.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    assert second_entered.is_set()
    assert not (output / "first.json").exists()
    assert (output / "second.json").is_file()
    verification = verify_output_run_commit(
        output,
        expected_request_fingerprint="second",
    )
    assert verification["status"] == "passed"
