"""A settled scene attempt's provider bundle zip is reclaimed; everything else stays.

capture-447ae4f4, 2026-11-01: seven pinned activation directories held ~40 GB
of bundles for replaced, settled attempts and the launch was refused for disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_settled_activation_bundles as reaper
from blueprint_pipeline import task_evaluation_retained_controls_evidence as evidence
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

INTENT = "scene-" + "a" * 64


def _seal(value: dict, field: str) -> dict:
    # Intake records use the cross-runtime digest (task_evaluation_scene_intent_contracts).
    value[field] = cross_runtime_canonical_digest(value, digest_field=field)
    return value


def _activation(root: Path, activation_id: str, payload: bytes = b"zip-bytes") -> Path:
    bundle = root / activation_id / "launch-set/bundle"
    bundle.mkdir(parents=True)
    (bundle / reaper.BUNDLE_RELATIVE.name).write_bytes(payload)
    (bundle / reaper.BUNDLE_RECEIPT_NAME).write_text(json.dumps({"bundle_sha256": "sha256:" + "b" * 64}))
    return bundle


def _attempt(intents: Path, attempt_id: str, launch_id: str, *, settled: bool) -> dict:
    directory = intents / INTENT
    (directory / "attempts").mkdir(parents=True, exist_ok=True)
    row = _seal({"attempt_id": attempt_id, "intent_id": INTENT, "maximum_spend_usd": 20.0}, "attempt_digest")
    (directory / "attempts" / f"{attempt_id}.json").write_text(json.dumps(row))
    if settled:
        (directory / reaper.SETTLEMENT_DIRECTORY).mkdir(exist_ok=True)
        receipt = {"schema_version": reaper.SETTLEMENT_SCHEMA, "attempt_id": attempt_id,
                   "status": "settled_after_terminal_attempt",
                   "execution_terminal": {"launch_id": launch_id, "launch_never_queued": True}}
        (directory / reaper.SETTLEMENT_DIRECTORY / f"{attempt_id}.json").write_text(json.dumps(receipt))
    return row


@pytest.fixture
def layout(tmp_path, monkeypatch):
    activations, intents, queues = tmp_path / "launch-activations", tmp_path / "intents", tmp_path / "queues"
    for path in (activations, intents, queues / "task-evaluation-launches/pending"):
        path.mkdir(parents=True)
    # Stand in for the intake's full settlement proof: a settlement file means settled.
    def validated(directory, attempt):
        path = Path(directory) / reaper.SETTLEMENT_DIRECTORY / (attempt["attempt_id"] + ".json")
        return json.loads(path.read_text()) if path.exists() else None
    monkeypatch.setattr(evidence, "validated_cancellation", validated)
    return activations, intents, queues


def _plan(activations, intents, queues):
    return reaper.build_settled_activation_bundle_manifest(
        activation_roots=[activations], intent_root=intents,
        queue_roots=[queues / "task-evaluation-launches"], now=lambda: 1000.0)


def test_only_a_settled_attempts_bundle_zip_is_removed_and_its_receipts_stay(layout):
    activations, intents, queues = layout
    settled = _activation(activations, "website-x-a00c7494-activation-auto")
    live = _activation(activations, "website-x-f35b93e3-activation-auto")
    _attempt(intents, "scene-configuration-old", "website-x-a00c7494-activation-auto-launch", settled=True)
    _attempt(intents, "scene-configuration-live", "website-x-f35b93e3-activation-auto-launch", settled=False)

    plan = _plan(activations, intents, queues)
    assert [row["activation_id"] for row in plan["candidates"]] == ["website-x-a00c7494-activation-auto"]
    assert (settled / reaper.BUNDLE_RELATIVE.name).exists()  # planning removes nothing

    result = reaper.apply_settled_activation_bundle_manifest(plan, ack=reaper.ACK, now=lambda: 1001.0)
    assert result["removed_count"] == 1 and result["removed_bytes"] == len(b"zip-bytes")
    assert not (settled / reaper.BUNDLE_RELATIVE.name).exists()
    assert (settled / reaper.BUNDLE_RECEIPT_NAME).exists()
    marker = json.loads((settled / reaper.MARKER_NAME).read_text())
    assert marker["attempt_id"] == "scene-configuration-old"
    assert marker["bundle_sha256"] == "sha256:" + "b" * 64 and marker["receipts_retained"] is True
    assert marker["marker_digest"] == canonical_digest(marker, digest_field="marker_digest")
    # The attempt that is not settled keeps every byte.
    assert (live / reaper.BUNDLE_RELATIVE.name).exists()


def test_an_activation_a_live_queue_message_names_keeps_its_bundle(layout):
    activations, intents, queues = layout
    bundle = _activation(activations, "website-x-95d87185-activation-auto")
    _attempt(intents, "scene-configuration-old", "website-x-95d87185-activation-auto-launch", settled=True)
    (queues / "task-evaluation-launches/pending/launch.json").write_text(
        json.dumps({"launch_id": "website-x-95d87185-activation-auto-launch"}))

    plan = _plan(activations, intents, queues)
    assert plan["candidates"] == [] and plan["retained_counts"]["queue_referenced"] == 1
    assert (bundle / reaper.BUNDLE_RELATIVE.name).exists()


def test_a_bundle_or_settlement_changed_after_planning_is_skipped(layout):
    activations, intents, queues = layout
    bundle = _activation(activations, "website-x-d63579c5-activation-auto")
    _attempt(intents, "scene-configuration-old", "website-x-d63579c5-activation-auto-launch", settled=True)
    plan = _plan(activations, intents, queues)
    (bundle / reaper.BUNDLE_RELATIVE.name).write_bytes(b"rebuilt-bundle-of-another-size")

    result = reaper.apply_settled_activation_bundle_manifest(plan, ack=reaper.ACK)
    assert result["removed_count"] == 0
    assert result["skipped"] == [{"activation_id": "website-x-d63579c5-activation-auto", "reason": "candidate_changed"}]
    assert (bundle / reaper.BUNDLE_RELATIVE.name).exists()


def test_a_manifest_without_its_acknowledgement_or_seal_removes_nothing(layout):
    activations, intents, queues = layout
    bundle = _activation(activations, "website-x-237194fe-activation-auto")
    _attempt(intents, "scene-configuration-old", "website-x-237194fe-activation-auto-launch", settled=True)
    plan = _plan(activations, intents, queues)
    with pytest.raises(reaper.SettledActivationBundleError):
        reaper.apply_settled_activation_bundle_manifest(plan, ack="wrong")
    with pytest.raises(reaper.SettledActivationBundleError):
        reaper.apply_settled_activation_bundle_manifest({**plan, "candidate_count": 9}, ack=reaper.ACK)
    assert (bundle / reaper.BUNDLE_RELATIVE.name).exists()


def test_a_symlinked_bundle_is_never_followed(layout, tmp_path):
    activations, intents, queues = layout
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"not ours")
    bundle = (activations / "website-x-7d27dc64-activation-auto/launch-set/bundle")
    bundle.mkdir(parents=True)
    (bundle / reaper.BUNDLE_RELATIVE.name).symlink_to(outside)
    _attempt(intents, "scene-configuration-old", "website-x-7d27dc64-activation-auto-launch", settled=True)
    assert _plan(activations, intents, queues)["candidates"] == []
    assert outside.exists()


def test_the_gc_unit_reclaims_settled_bundles_under_launch_activations():
    unit = (Path(__file__).resolve().parents[1]
            / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text()
    assert ("Environment=BLUEPRINT_CONTROL_PLANE_GC_SETTLED_ACTIVATION_BUNDLE_ROOTS="
            "/var/lib/blueprint/task-evaluation-inputs/launch-activations\n") in unit
    assert "/var/lib/blueprint/task-evaluation-inputs/launch-activations" in next(
        line for line in unit.splitlines() if line.startswith("ReadWritePaths=/var/lib/blueprint/task-evaluation-inputs/prepared-references"))
    assert "Environment=BLUEPRINT_CONTROL_PLANE_GC_SCENE_INTENT_ROOT=" in unit
