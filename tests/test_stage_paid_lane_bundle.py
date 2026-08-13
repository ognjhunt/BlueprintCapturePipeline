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

import pytest

RECEIPT_NAME = "adp_retained_scene_gpu_render_bundle_receipt.json"
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

    def __init__(
        self, root: Path, *, corrupt: str | None = None, unreadable_for: tuple = ()
    ) -> None:
        self.root = root
        self.corrupt = corrupt
        self.unreadable_for = unreadable_for
        self.placed: list[str] = []
        self.finalized: tuple | None = None

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

    def finalize(self, remote_dir: str, owner: str) -> None:
        self.finalized = (remote_dir, owner)

    def digest(self, remote: str, *, as_user: str | None = None) -> str:
        target = self._local(remote)
        if not target.is_file():
            return ""
        if as_user and as_user in self.unreadable_for:
            # The shape that broke staging in production: bytes present, digest
            # correct for the transfer user, unopenable by the consumer.
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


def test_staged_bytes_the_control_plane_cannot_read_are_reported(tmp_path):
    """`scp` preserves the source mode, so a receipt that happened to be 0600
    on the authoring machine lands 0600 and root-owned. Everything looks
    correct -- bytes present, digests matching -- while the service account
    cannot open one of them. Verifying as the transfer user proves nothing
    about the consumer."""

    job = _job(tmp_path)
    transport = _LocalTransport(tmp_path / "host", unreadable_for=("blueprint",))

    with pytest.raises(stager.StagingError, match="staging_consumer_cannot_read"):
        stager.stage_paid_lane_bundle(
            receipt_path=job / RECEIPT_NAME,
            lane_id="lane",
            transport=transport,
        )


def test_the_staged_tree_is_handed_to_the_consuming_account(tmp_path):
    job = _job(tmp_path)
    transport = _LocalTransport(tmp_path / "host")

    receipt = stager.stage_paid_lane_bundle(
        receipt_path=job / RECEIPT_NAME, lane_id="lane", transport=transport
    )

    assert transport.finalized == (receipt["remote_dir"], "blueprint")
    assert receipt["verified_readable_as"] == "blueprint"


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


def test_a_receipt_the_lane_binds_from_outside_the_bundle_directory_is_staged(tmp_path):
    """The allocator refuses to run a content-agents attempt without its config
    preflight, and that preflight lives in a sibling directory of the bundle."""

    job = _job(tmp_path)
    preflight_dir = tmp_path / "local_config_preflight_v1"
    preflight_dir.mkdir()
    preflight = preflight_dir / "adp_content_agents_bundle_config_preflight.json"
    preflight.write_text('{"bundle_receipt_sha256": "sha256:x"}', encoding="utf-8")
    transport = _LocalTransport(tmp_path / "host")

    receipt = stager.stage_paid_lane_bundle(
        receipt_path=job / RECEIPT_NAME,
        lane_id="lane",
        transport=transport,
        extra_paths=[preflight],
    )

    staged = {row["relative_path"] for row in receipt["staged_files"]}
    # Flat by name, so the destination stays a directory of files rather than a
    # copy of somebody's tree.
    assert "adp_content_agents_bundle_config_preflight.json" in staged


def test_an_extra_that_would_overwrite_a_bundle_file_is_refused(tmp_path):
    job = _job(tmp_path)
    collision = tmp_path / "elsewhere" / RECEIPT_NAME
    collision.parent.mkdir()
    collision.write_text("{}", encoding="utf-8")

    with pytest.raises(stager.StagingError, match="staging_extra_file_name_collides"):
        stager.stage_paid_lane_bundle(
            receipt_path=job / RECEIPT_NAME,
            lane_id="lane",
            transport=_LocalTransport(tmp_path / "host"),
            extra_paths=[collision],
        )
