"""A run that never rented anything still has to be able to close.

Five launches on the live control plane sat at `provider_zero_pending` waiting
on a teardown manifest for a resource that was never obtained, and the
reconciler unit failed on every sweep after that. A provider-zero signal that
is permanently red is not a strict one; it is one nobody can read, on the day
something really does leak.

The answer is evidence, not an exception: the lane records that nothing was
allocated, and the reconciler accepts that only when the record proves it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_artifact_manifest import (
    TEARDOWN_MANIFEST_NAME,
    UNALLOCATED_TEARDOWN_STATUS,
    seal_unallocated_provider_teardown,
)
from blueprint_pipeline.task_evaluation_launch_reconciler import (
    _terminal_teardown_evidence,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"


def _teardown_receipt(tmp_path: Path, manifest: dict) -> dict:
    """A terminal launch receipt pointing at a digest-bound teardown manifest."""

    import hashlib

    path = tmp_path / TEARDOWN_MANIFEST_NAME
    path.write_text(json.dumps(manifest), encoding="utf-8")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "terminal_evidence": {
            "artifacts": {
                "teardown_manifest_path": {
                    "path": str(path),
                    "digest": digest,
                    "exists": True,
                }
            }
        }
    }


def _unallocated_manifest(**overrides) -> dict:
    manifest = {
        "schema_version": "vast_teardown_manifest.v1",
        "generated_at": "2026-08-13T05:07:13.880175+00:00",
        "status": UNALLOCATED_TEARDOWN_STATUS,
        "vast_instance_ids": [],
        "teardown_actions_performed": [],
        "continuing_spend_from_this_run": False,
        "zero_continuing_spend_scope": "no provider resource was ever requested",
    }
    manifest.update(overrides)
    return manifest


def test_a_proven_unallocated_teardown_closes_the_run(tmp_path) -> None:
    receipt = _teardown_receipt(tmp_path, _unallocated_manifest())

    teardown, blockers = _terminal_teardown_evidence(receipt=receipt)

    assert blockers == []
    assert teardown is not None
    # The distinction survives into the closure receipt: this run was never
    # allocated, it was not torn down, and a reader must be able to tell.
    assert teardown["provider_resource_allocated"] is False
    assert teardown["status"] == UNALLOCATED_TEARDOWN_STATUS


def test_a_completed_teardown_still_reads_as_allocated(tmp_path) -> None:
    receipt = _teardown_receipt(
        tmp_path,
        _unallocated_manifest(
            status="completed",
            vast_instance_ids=[47593142],
            teardown_actions_performed=["destroy"],
        ),
    )

    teardown, blockers = _terminal_teardown_evidence(receipt=receipt)

    assert blockers == []
    assert teardown["provider_resource_allocated"] is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"vast_instance_ids": [47593142]},
        {"teardown_actions_performed": ["destroy"]},
        {"status": "blocked"},
    ],
    ids=["instance-existed", "teardown-was-attempted", "status-not-not-required"],
)
def test_anything_short_of_proven_zero_allocation_stays_pending(
    tmp_path, overrides
) -> None:
    """Wherever a resource may have existed, `completed` is the only answer."""

    receipt = _teardown_receipt(tmp_path, _unallocated_manifest(**overrides))

    teardown, blockers = _terminal_teardown_evidence(receipt=receipt)

    assert teardown is None
    assert "terminal_teardown_manifest_not_completed" in blockers


def test_continuing_spend_is_still_refused_regardless_of_allocation(tmp_path) -> None:
    receipt = _teardown_receipt(
        tmp_path, _unallocated_manifest(continuing_spend_from_this_run=True)
    )

    teardown, blockers = _terminal_teardown_evidence(receipt=receipt)

    assert teardown is None
    assert "terminal_teardown_continuing_spend_not_false" in blockers


def test_the_sealer_records_the_absence_of_any_allocation(tmp_path) -> None:
    run = tmp_path / "vast_provider_run"

    path = seal_unallocated_provider_teardown(run, reason="secret_missing")

    assert path is not None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == UNALLOCATED_TEARDOWN_STATUS
    assert manifest["vast_instance_ids"] == []
    assert manifest["continuing_spend_from_this_run"] is False
    assert "secret_missing" in manifest["zero_continuing_spend_scope"]


def test_the_sealer_never_overwrites_a_real_teardown(tmp_path) -> None:
    run = tmp_path / "vast_provider_run"
    run.mkdir()
    existing = run / TEARDOWN_MANIFEST_NAME
    existing.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

    assert seal_unallocated_provider_teardown(run, reason="late_failure") is None
    assert json.loads(existing.read_text(encoding="utf-8"))["status"] == "completed"


def test_the_sealer_refuses_when_an_instance_existed(tmp_path) -> None:
    """The case that would make this a lie: the adapter ran and launched."""

    run = tmp_path / "vast_provider_run"
    run.mkdir()
    (run / "vast_provider_adapter_result.json").write_text(
        json.dumps({"status": "blocked", "vast_instance_ids": [47593142]}),
        encoding="utf-8",
    )

    assert seal_unallocated_provider_teardown(run, reason="crashed_after_launch") is None
    assert not (run / TEARDOWN_MANIFEST_NAME).exists()


def test_the_sealer_never_raises_on_an_unusable_run_directory(tmp_path) -> None:
    """It is called from an except handler; it must not become the failure."""

    blocker = tmp_path / "vast_provider_run"
    blocker.write_text("not a directory", encoding="utf-8")

    assert seal_unallocated_provider_teardown(blocker, reason="whatever") is None


def test_every_lane_that_catches_an_adapter_failure_seals_the_absence() -> None:
    """Rediscovered from source, so a lane added tomorrow is covered too.

    Fixing this lane by lane leaves the next lane to rediscover it -- which is
    exactly how eight lanes came to share the same missing terminal artifact.
    """

    missing: list[str] = []
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "run_vast_provider_adapter(" not in source:
            continue
        tree = ast.parse(source)
        for handler in ast.walk(tree):
            if not isinstance(handler, ast.ExceptHandler):
                continue
            body = ast.dump(handler)
            if "_adapter_failed" not in body and "vast_adapter_failed" not in body:
                continue
            if "seal_unallocated_provider_teardown" not in body:
                missing.append(f"{path.name}:{handler.lineno}")

    assert not missing, (
        "these catch a provider adapter failure and leave the run unable to "
        f"ever close: {missing}"
    )
