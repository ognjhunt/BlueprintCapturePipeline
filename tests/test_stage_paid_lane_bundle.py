"""Staging is a transfer with a receipt, or it is the thing that broke before.

Lane-neutral on purpose: every paid lane has the same problem and the same
shape of answer, so a per-lane copy would be a per-lane chance to omit a digest
check.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

RECEIPT_NAME = "adp_retained_scene_gpu_render_bundle_receipt.json"

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "stage_paid_lane_bundle",
    REPO_ROOT / "scripts" / "stage_paid_lane_bundle.py",
)
stager = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stager)


class _LocalTransport:
    """A control-plane host rooted in a temporary directory."""

    def __init__(self, root: Path, *, corrupt: str | None = None) -> None:
        self.root = root
        self.corrupt = corrupt
        self.placed: list[str] = []

    def _local(self, remote: str) -> Path:
        return self.root / remote.lstrip("/")

    def mkdir(self, remote_dir: str) -> None:
        self._local(remote_dir).mkdir(parents=True, exist_ok=True)

    def put(self, local: Path, remote: str) -> None:
        target = self._local(remote)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
        if self.corrupt and remote.endswith(self.corrupt):
            target.write_bytes(b"truncated-in-flight")
        self.placed.append(remote)

    def digest(self, remote: str) -> str:
        target = self._local(remote)
        if not target.is_file():
            return ""
        return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _job(tmp_path: Path, *, portable: bool = True) -> Path:
    job = tmp_path / "job"
    runtime = job / "provider_runtime"
    runtime.mkdir(parents=True)
    archive = job / "adp_retained_scene_gpu_render_bundle.zip"
    archive.write_bytes(b"bundle-archive-bytes")
    authority = runtime / "execution_authority.json"
    authority.write_bytes(b'{"schema_version": "third_scene_dual_task_execution_authority.v1"}')
    request = runtime / "source_request_manifest.json"
    request.write_bytes(b'{"schema_version": "adp009d_retained_scene_gpu_render_request.v1"}')
    (job / "adp_retained_scene_gpu_render_exact_bundle_rehearsal.json").write_bytes(
        b'{"status": "passed"}'
    )
    receipt = {
        "status": "ready",
        "blueprint_commit": "a" * 40,
        "bundle_path": "/Users/author/job/bundle.zip",
        "bundle_sha256": _digest(archive.read_bytes()),
        "execution_authority": {
            "path": "/private/tmp/checkout/execution_authority.json",
            "sha256": _digest(authority.read_bytes()),
        },
        "request": {
            "path": "/private/tmp/checkout/request.v1.json",
            "sha256": _digest(request.read_bytes()),
        },
    }
    if not portable:
        # The legacy shape: absolute authoring paths whose basenames are the
        # filenames that actually travel, which is what the resolver looks for
        # beside the receipt.
        receipt["bundle_path"] = "/Users/author/job/adp_retained_scene_gpu_render_bundle.zip"
        receipt["execution_authority"]["path"] = "/private/tmp/checkout/execution_authority.json"
        receipt.pop("request")
    if portable:
        receipt["bundle_relative_path"] = "adp_retained_scene_gpu_render_bundle.zip"
        receipt["execution_authority"]["relative_path"] = (
            "provider_runtime/execution_authority.json"
        )
        receipt["request"]["relative_path"] = "provider_runtime/source_request_manifest.json"
    (job / RECEIPT_NAME).write_text(json.dumps(receipt), encoding="utf-8")
    return job


def test_staging_places_the_referenced_files_and_verifies_them_on_the_host(tmp_path):
    job = _job(tmp_path)
    transport = _LocalTransport(tmp_path / "host")

    receipt = stager.stage_paid_lane_bundle(
        receipt_path=job / RECEIPT_NAME,
        lane_id="retained-scene-render-840920",
        remote_root="/var/lib/blueprint/task-evaluation-inputs",
        transport=transport,
    )

    assert receipt["status"] == "staged"
    assert receipt["remote_dir"] == (
        "/var/lib/blueprint/task-evaluation-inputs/retained-scene-render-840920"
    )
    assert receipt["provider_mutation_performed"] is False
    assert {row["relative_path"] for row in receipt["staged_files"]} == {
        RECEIPT_NAME,
        "adp_retained_scene_gpu_render_exact_bundle_rehearsal.json",
        "adp_retained_scene_gpu_render_bundle.zip",
        "provider_runtime/execution_authority.json",
        "provider_runtime/source_request_manifest.json",
    }


def test_a_receipt_read_on_the_host_resolves_against_what_was_staged(tmp_path):
    """The point of staging: the same receipt reports ready over there."""

    from blueprint_pipeline.host_resident_launch_inputs import (
        resolve_host_resident_bundle_receipt,
    )

    job = _job(tmp_path)
    host_root = tmp_path / "host"
    transport = _LocalTransport(host_root)
    staged = stager.stage_paid_lane_bundle(
        receipt_path=job / RECEIPT_NAME,
        lane_id="retained-scene-render-840920",
        remote_root="/var/lib/blueprint/task-evaluation-inputs",
        transport=transport,
    )

    staged_dir = host_root / staged["remote_dir"].lstrip("/")
    resolution = resolve_host_resident_bundle_receipt(
        staged_dir / RECEIPT_NAME, roots=[host_root]
    )

    assert resolution["status"] == "ready"
    assert resolution["receipt"]["status"] == "ready"
    assert resolution["receipt"]["bundle_path"].startswith(str(staged_dir))


def test_a_destination_outside_a_control_plane_root_is_refused(tmp_path):
    job = _job(tmp_path)

    with pytest.raises(stager.StagingError, match="remote_root_outside_control_plane"):
        stager.stage_paid_lane_bundle(
            receipt_path=job / RECEIPT_NAME,
            lane_id="lane",
            remote_root="/Users/author/workspace/BlueprintValidation/data",
            transport=_LocalTransport(tmp_path / "host"),
        )


def test_a_receipt_that_does_not_resolve_here_is_not_staged(tmp_path):
    job = _job(tmp_path)
    (job / "adp_retained_scene_gpu_render_bundle.zip").unlink()

    with pytest.raises(stager.StagingError, match="staging_receipt_not_self_resolving"):
        stager.stage_paid_lane_bundle(
            receipt_path=job / RECEIPT_NAME,
            lane_id="lane",
            transport=_LocalTransport(tmp_path / "host"),
        )


def test_a_receipt_predating_portable_references_still_stages(tmp_path):
    """Refusing these would strand every bundle built before the receipt format
    carried relative paths, for no gain in safety -- the archive is taken from
    the basename of its recorded path, which is where the resolver looks first
    anyway, and the digest is checked either way."""

    job = _job(tmp_path, portable=False)

    receipt = stager.stage_paid_lane_bundle(
        receipt_path=job / RECEIPT_NAME,
        lane_id="lane",
        transport=_LocalTransport(tmp_path / "host"),
    )

    assert receipt["status"] == "staged"
    assert any(
        row["relative_path"] == "adp_retained_scene_gpu_render_bundle.zip"
        for row in receipt["staged_files"]
    )


def test_bytes_that_did_not_survive_the_transfer_are_reported(tmp_path):
    job = _job(tmp_path)
    transport = _LocalTransport(tmp_path / "host", corrupt="bundle.zip")

    with pytest.raises(stager.StagingError, match="staging_remote_digest_mismatch"):
        stager.stage_paid_lane_bundle(
            receipt_path=job / RECEIPT_NAME,
            lane_id="lane",
            transport=transport,
        )


def test_a_lane_id_cannot_escape_the_remote_root(tmp_path):
    job = _job(tmp_path)

    with pytest.raises(stager.StagingError, match="staging_lane_id_invalid"):
        stager.stage_paid_lane_bundle(
            receipt_path=job / RECEIPT_NAME,
            lane_id="../../etc",
            transport=_LocalTransport(tmp_path / "host"),
        )
