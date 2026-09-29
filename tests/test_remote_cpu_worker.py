# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_worker.py
#   src/blueprint_pipeline/remote_cpu_job_contract.py
#   src/blueprint_pipeline/remote_cpu_output_archive.py
#   src/blueprint_pipeline/remote_cpu_environment.py
#   tests/remote_cpu_worker_support.py
"""ADP-009D/day-28, plan 14 PR 3: one Cloud Run execution runs one stage from its pinned transport and release."""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from blueprint_pipeline import cloud_run_jobs_client
from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_worker as worker
from tests.remote_cpu_worker_support import COMMIT, EXECUTION, WorkerWorld, digest_of, release_archive

SOURCE = "remote-cpu-source/"


def _refused(world: WorkerWorld, argv: list[str], environ: dict[str, str]) -> int:
    launched: list[dict] = []
    code = worker.bootstrap(argv, world.runtime(environ=environ, launch=lambda handoff: launched.append(handoff) or 0))
    assert launched == []
    return code


def test_bootstrap_refuses_extra_argv_task_count_and_urls_outside_the_prefix(tmp_path: Path) -> None:
    world = WorkerWorld(tmp_path)
    # The worker pins exactly what the dispatcher sets and mints (PR 2), without importing either.
    assert worker.OVERRIDE_VARIABLES == cloud_run_jobs_client.OVERRIDE_VARIABLES
    assert worker.STAGING_OBJECTS == allocator.STAGING_OBJECTS
    assert list(cloud_run_jobs_client.BOOTSTRAP_COMMAND[2:]) == [worker.MODULE, "bootstrap"]

    environ = world.environ
    generation = int(environ["BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION"])
    refusals = [
        (["bootstrap", "--overridden"], {}, "remote_cpu_worker_argv_invalid"),
        ([], {}, "remote_cpu_worker_argv_invalid"),
        (["bootstrap"], {"CLOUD_RUN_TASK_COUNT": "2"}, "remote_cpu_worker_task_count_invalid"),
        (["bootstrap"], {"CLOUD_RUN_TASK_COUNT": None}, "remote_cpu_worker_task_count_invalid"),
        (["bootstrap"], {"CLOUD_RUN_EXECUTION": "Not/A-Name"}, "remote_cpu_worker_execution_unnamed"),
        (["bootstrap"], {"BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION": str(generation + 1)},
         "remote_cpu_worker_transport_unavailable"),
        (["bootstrap"], {"BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION": "0"}, "remote_cpu_worker_transport_unpinned"),
        (["bootstrap"], {"BLUEPRINT_REMOTE_CPU_ATTEMPT_ID": world.descriptor["attempt_id"][:-1] + "0"},
         "remote_cpu_worker_transport_unpinned"),
        (["bootstrap"], {"BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256": "sha256:" + "0" * 64},
         "remote_cpu_worker_descriptor_mismatch"),
        (["bootstrap"], {"CLOUD_RUN_JOB": "blueprint-remote-cpu-other"}, "remote_cpu_worker_job_mismatch"),
        (["bootstrap"], {"BLUEPRINT_REMOTE_CPU_STAGE": "other-stage"}, "remote_cpu_worker_job_mismatch"),
        (["bootstrap"], {"BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX": None}, "remote_cpu_worker_object_prefix_invalid"),
        # Another B2 host, another bucket, or a narrower prefix than the objects the transport names.
        *[(["bootstrap"], {"BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX": environ["BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX"].replace(
            old, new)}, "remote_cpu_worker_url_outside_prefix") for old, new in (
            ("s3.us-west-004.backblazeb2.test", "s3.eu-central-003.backblazeb2.test"),
            ("/b2-bucket/", "/other-bucket/"), ("configured-scenes/", "configured-scenes/remote-cpu/staging/"))],
    ]
    for argv, changes, code in refusals:
        changed = {**environ, **changes}
        for name, value in changes.items():
            if value is None:
                del changed[name]
        assert _refused(world, argv, changed) == 2, (argv, changes)
        assert json.loads(world.logs[-1]) == {"mode": "bootstrap", "status": "refused", "code": code}, (argv, changes)

    # A replaced transport whose URLs leave the pinned prefix, or name another object, is refused whole.
    def swap(transport: dict, first: tuple, second: tuple) -> None:
        def slot(path: tuple) -> tuple[dict, str]:
            container = transport
            for step in path[:-1]:
                container = container[step]
            return container, path[-1]

        (a, key_a), (b, key_b) = slot(first), slot(second)
        a[key_a], b[key_b] = b[key_b], a[key_a]

    def rehost(transport: dict, url: str) -> None:
        transport["outputs"]["heartbeat.json"] = url

    heartbeat = world.transport()["outputs"]["heartbeat.json"]
    outside, invalid = "remote_cpu_worker_url_outside_prefix", "remote_cpu_worker_transport_invalid"
    tampered = [
        (lambda transport: swap(transport, ("inputs", 0, "url"), ("source_archive", "url")), outside),
        (lambda transport: swap(transport, ("outputs", "blobs.tar"), ("outputs", "receipt.json")), outside),
        (lambda transport: swap(transport, ("receipt_url",), ("inputs", 1, "url")), outside),
        (lambda transport: rehost(transport, heartbeat.replace("https://", "https://evil@")), outside),
        (lambda transport: rehost(transport, heartbeat.replace("https://", "http://")), outside),
        (lambda transport: rehost(transport, heartbeat.replace("s3.us-west-004.backblazeb2.test", "evil.example")),
         outside),
        (lambda transport: rehost(transport, heartbeat.replace("heartbeat.json", "heartbeat.json/../index.json")),
         outside),
        (lambda transport: rehost(transport, heartbeat.replace("?", "#fragment?")), outside),
        (lambda transport: transport["outputs"].update({"extra.json": heartbeat}), invalid),
        (lambda transport: transport["inputs"].pop(), invalid),
        (lambda transport: transport["inputs"][0].update({"digest": "sha256:" + "0" * 64}), invalid),
        (lambda transport: transport.update({"schema_version": "remote_cpu_job_transport.v0"}), invalid),
    ]
    for index, (mutate, code) in enumerate(tampered):
        replaced = world.mint(nonce=f"{index + 1:032x}", mutate=mutate)
        assert _refused(world, ["bootstrap"], replaced) == 2, index
        assert json.loads(world.logs[-1])["code"] == code, index

    # A descriptor altered after sealing is refused even when its digest is the one the dispatcher named.
    def unsealed(transport: dict) -> None:
        transport["descriptor"]["outputs"]["declared_scratch"] = []

    assert _refused(world, ["bootstrap"], world.mint(nonce="f" * 32, mutate=unsealed)) == 2
    assert json.loads(world.logs[-1])["code"] == "remote_cpu_worker_descriptor_invalid"
    # Nothing trusted, so nothing was fetched, uploaded or written.
    assert world.http.requests == [] and not world.fs.exists() and world.receipt() is None


def test_bootstrap_extracts_the_release_under_tmp_and_verifies_digests(tmp_path: Path) -> None:
    world = WorkerWorld(tmp_path / "pinned")
    handoffs: list[dict] = []
    assert worker.bootstrap(["bootstrap"], world.runtime(launch=lambda handoff: handoffs.append(handoff) or 0)) == 0
    [handoff] = handoffs
    release = world.fs / "tmp" / "blueprint-release" / COMMIT
    assert handoff["release_root"] == str(release) and handoff["filesystem_root"] == str(world.fs)
    assert handoff["transport"] == world.transport() and handoff["execution_name"] == EXECUTION
    with tarfile.open(fileobj=io.BytesIO(world.archive)) as archive:
        expected = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    assert {str(path.relative_to(release)): path.read_bytes() for path in release.rglob("*") if path.is_file()} == expected
    # Only the extracted tree remains: the downloaded archive is gone, and nothing reached the path root.
    assert sorted(path.name for path in release.parent.iterdir()) == [COMMIT]
    assert sorted(str(path.relative_to(world.fs)) for path in world.fs.rglob("*") if path.is_file()) == sorted(
        f"tmp/blueprint-release/{COMMIT}/{name}" for name in expected)
    source_key = world.transport()["source_archive"]["url"].split("?")[0].split("/b2-bucket/")[1]
    assert world.http.requests == [("GET", world.key("receipt.json")), ("PUT", world.key("heartbeat.json")),
                                   ("GET", source_key)]
    [heartbeat] = world.heartbeats()
    assert contract.validate_heartbeat(heartbeat, attempt_id=world.descriptor["attempt_id"], execution_name=EXECUTION)
    assert (heartbeat["sequence"], heartbeat["phase"]) == (1, "bootstrap") and world.receipt() is None
    assert (handoff["heartbeat_sequence"], handoff["bytes_fetched"]) == (1, len(world.archive))

    # The archive must be the descriptor's bytes, of the descriptor's commit, holding only the recipe's paths.
    for label, change, code in [
        ("tampered", "bytes", "remote_cpu_worker_source_digest_mismatch"),
        ("other-commit", release_archive("b" * 40), "remote_cpu_worker_source_commit_mismatch"),
        ("outside-recipe", release_archive(files={"src/x.py": b"", "tests/test_x.py": b""}),
         "remote_cpu_worker_source_member_invalid"),
        ("escaping", release_archive(files={"src/x.py": b"", "src/../../etc/x": b""}),
         "remote_cpu_worker_source_member_invalid"),
        ("link", release_archive(extra=_symlink), "remote_cpu_worker_source_member_invalid"),
    ]:
        other = WorkerWorld(tmp_path / label, archive=None if change == "bytes" else change)
        if change == "bytes":
            key = world.transport()["source_archive"]["url"].split("?")[0].split("/b2-bucket/")[1]
            other.store.put_object(Bucket="b2-bucket", Key=key, Body=other.archive.replace(b"example", b"EXAMPLE"))
        launched: list[dict] = []
        assert worker.bootstrap(["bootstrap"], other.runtime(launch=lambda handoff: launched.append(handoff) or 0)) == 0
        receipt = other.receipt()
        assert launched == [] and receipt["status"] == "infrastructure_failed", label
        assert receipt["infrastructure_failures"] == [f"infrastructure_failed:{code}"], label
        verdict = contract.validate_receipt(receipt, descriptor=other.descriptor, execution_name=EXECUTION)
        assert (verdict["outcome"], verdict["terminal"]) == ("infrastructure_failed", False)
        release_parent = other.fs / "tmp" / "blueprint-release"
        assert list(release_parent.iterdir()) == [], label


def _symlink(tar: tarfile.TarFile) -> None:
    info = tarfile.TarInfo("src/link.py")
    info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
    tar.addfile(info)


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a 0555 directory")
def test_unwritable_path_root_refuses_before_fetching(tmp_path: Path) -> None:
    world = WorkerWorld(tmp_path)
    root = world.local("/var/lib/blueprint")
    root.mkdir(parents=True)
    root.chmod(0o555)
    launched: list[dict] = []
    try:
        code = worker.bootstrap(["bootstrap"], world.runtime(launch=lambda handoff: launched.append(handoff) or 0))
    finally:
        root.chmod(0o755)
    assert code == 0 and launched == []
    receipt = world.receipt()
    assert receipt["status"] == "infrastructure_failed" and receipt["result"] is None and receipt["output"] is None
    assert receipt["infrastructure_failures"] == ["infrastructure_failed:remote_cpu_worker_path_root_unwritable"]
    assert receipt["phases"]["fetch"] is None and receipt["bytes_fetched"] == 0
    verdict = contract.validate_receipt(receipt, descriptor=world.descriptor, execution_name=EXECUTION)
    assert (verdict["outcome"], verdict["terminal"]) == ("infrastructure_failed", False)
    # The receipt check, a heartbeat and the receipt: no source or input was fetched, and nothing was written.
    assert world.http.requests == [("GET", world.key("receipt.json")), ("PUT", world.key("heartbeat.json")),
                                   ("PUT", world.key("receipt.json"))]
    assert list(root.iterdir()) == [] and not (world.fs / "tmp" / "blueprint-release" / COMMIT).exists()
    assert world.descriptor["code"]["source_archive"]["digest"] == digest_of(world.archive)
