# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_worker.py
#   src/blueprint_pipeline/remote_cpu_worker_stage.py
#   src/blueprint_pipeline/remote_cpu_job_contract.py
#   src/blueprint_pipeline/remote_cpu_output_archive.py
#   src/blueprint_pipeline/remote_cpu_environment.py
#   tests/remote_cpu_worker_support.py
"""ADP-009D/day-28, plan 14 PR 3: one Cloud Run execution runs one stage from its pinned transport and release."""

from __future__ import annotations

import io
import json
import logging
import os
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
from pathlib import Path

import pytest

from blueprint_pipeline import cloud_run_jobs_client
from blueprint_pipeline import remote_cpu_job_allocator as allocator
from blueprint_pipeline import remote_cpu_environment as census
from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_output_archive as archive
from blueprint_pipeline import remote_cpu_worker as worker
from blueprint_pipeline import remote_cpu_worker_stage as stage_child
from tests.remote_cpu_allocator_fakes import B2_BUCKET, JOB, RemoteCpuWorld
from tests.remote_cpu_worker_stages import NEW_BYTES, sealed_result
from tests.remote_cpu_worker_support import (
    BUNDLE_MEMBERS,
    COMMIT,
    EXECUTION,
    PREFIX_URL,
    REFERENCES,
    WORKER_RECORD,
    RecordingHttp,
    WorkerWorld,
    digest_of,
    release_archive,
    report,
    worker_release_archive,
)

SUCCESS_SCHEMA = "docs/schemas/rigid_task_success_contract.v1.schema.json"


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
    # The receipt check, a heartbeat, and the receipt once none is up: no source or input was fetched.
    assert world.http.requests == [("GET", world.key("receipt.json")), ("PUT", world.key("heartbeat.json")),
                                   ("GET", world.key("receipt.json")), ("PUT", world.key("receipt.json"))]
    assert list(root.iterdir()) == [] and not (world.fs / "tmp" / "blueprint-release" / COMMIT).exists()
    assert world.descriptor["code"]["source_archive"]["digest"] == digest_of(world.archive)


def _key(uri: str) -> str:
    return uri.split("/", 3)[3]


def _gets(world: WorkerWorld) -> list[str]:
    return [key for method, key in world.http.requests if method == "GET"]


def _puts(world: WorkerWorld) -> list[str]:
    return [key.rsplit("/", 1)[1] for method, key in world.http.requests if method == "PUT"]


def _verdict(world: WorkerWorld) -> tuple[str, bool]:
    verdict = contract.validate_receipt(world.receipt(), descriptor=world.descriptor, execution_name=EXECUTION)
    return verdict["outcome"], verdict["terminal"]


def _outputs(world: WorkerWorld, descriptor: dict) -> dict:
    """A clean stage: new bytes, an input's bytes and a bundle member's bytes, and a declared-scratch member."""

    output = world.local(descriptor["outputs"]["output_root"])
    (output / "native-arena-adapter").mkdir(parents=True)
    (output / "native-arena-adapter" / "result.json").write_bytes(NEW_BYTES)
    (output / "native-arena-adapter" / "model.bin").write_bytes(BUNDLE_MEMBERS["runtime/model.bin"])
    (output / "robot.json").write_bytes(world.local(f"{REFERENCES}/prep-1/robot.json").read_bytes())
    scratch = world.local(descriptor["outputs"]["declared_scratch"][0]) / "adapter-members"
    scratch.mkdir(parents=True)
    (scratch / "model.bin").write_bytes(BUNDLE_MEMBERS["runtime/model.bin"])
    return report(sealed_result(descriptor, blockers=[]))


def test_inputs_materialize_at_declared_paths_with_digest_checks(tmp_path: Path) -> None:
    world = WorkerWorld(tmp_path / "fetched")
    seen: list[tuple[str, bytes, int]] = []

    def stage(*, descriptor: dict, **_: object) -> dict:
        for item in descriptor["inputs"]:
            local = world.local(item["materialize_at"])
            seen.append((item["materialize_at"], local.read_bytes(), stat.S_IMODE(local.stat().st_mode)))
        return report(sealed_result(descriptor, blockers=["episode_compilation_envelope_invalid"]))

    assert world.run(stage) == 0
    # Every input at its declared path with its declared bytes and mode, fetched once, and no partial left.
    assert seen == [(path, data, 0o440) for path, (data, _, _) in world.inputs.items()]
    assert _gets(world) == [world.key("receipt.json"), _key(world.descriptor["code"]["source_archive"]["uri"]),
                            *(_key(item["uri"]) for item in world.descriptor["inputs"]), world.key("receipt.json")]
    assert not [path for path in world.fs.rglob("*") if path.name.endswith(".partial")]
    receipt = world.receipt()
    assert receipt["status"] == "blocked" and _verdict(world) == ("blocked", True)
    assert receipt["bytes_fetched"] == len(world.archive) + sum(len(data) for data, _, _ in world.inputs.values())

    # Bytes that are not the declared digest never reach the declared path, and the stage never runs.
    tampered = WorkerWorld(tmp_path / "tampered")
    path, (data, _, _) = list(tampered.inputs.items())[1]
    tampered.store.put_object(Bucket=B2_BUCKET, Key=_key(tampered.descriptor["inputs"][1]["uri"]), Body=data[:-1] + b"X")
    ran: list[int] = []
    assert tampered.run(lambda **_: ran.append(1) or report()) == 0
    assert ran == [] and not tampered.local(path).exists()
    assert not [item for item in tampered.fs.rglob("*") if item.name.endswith(".partial")]
    assert tampered.receipt()["infrastructure_failures"] == [f"infrastructure_failed:input_digest_mismatch:{path}"]
    assert _verdict(tampered) == ("infrastructure_failed", False)

    # Only a same-machine run finds an input already in place: an identical one is kept without a fetch,
    # and any other is refused.
    present = WorkerWorld(tmp_path / "present")
    first, second = list(present.inputs)[:2]
    for path, data in ((first, present.inputs[first][0]), (second, b"another file")):
        present.local(path).parent.mkdir(parents=True, exist_ok=True)
        present.local(path).write_bytes(data)
        present.local(path).chmod(0o440)
    assert present.run(lambda **_: ran.append(1) or report()) == 0
    assert ran == [] and present.receipt()["infrastructure_failures"] == [
        f"infrastructure_failed:input_conflict:{second}"]
    assert _key(present.descriptor["inputs"][0]["uri"]) not in _gets(present)


def test_duplicate_execution_with_an_existing_receipt_is_a_noop(tmp_path: Path) -> None:
    earlier = b'{"receipt": "an earlier execution committed this attempt"}\n'
    world = WorkerWorld(tmp_path / "earlier")
    world.store.put_object(Bucket=B2_BUCKET, Key=world.key("receipt.json"), Body=earlier)
    ran: list[int] = []
    assert world.run(lambda **_: ran.append(1) or report()) == 0
    assert world.http.requests == [("GET", world.key("receipt.json"))] and ran == []
    assert not world.fs.exists() and world.staged("receipt.json") == earlier
    assert json.loads(world.logs[-1]) == {"mode": "bootstrap", "status": "duplicate_execution"}

    # A duplicate that commits while this one runs wins: this one then uploads no output and no receipt.
    racing = WorkerWorld(tmp_path / "racing")

    def stage(*, descriptor: dict, **_: object) -> dict:
        outputs = _outputs(racing, descriptor)
        racing.store.put_object(Bucket=B2_BUCKET, Key=racing.key("receipt.json"), Body=earlier)
        return outputs

    assert racing.run(stage) == 0
    assert set(_puts(racing)) == {"heartbeat.json"} and racing.staged("receipt.json") == earlier
    assert racing.staged("blobs.tar") is None and racing.staged("index.json") is None
    assert json.loads(racing.logs[-1]) == {"mode": "execute", "status": "receipt_already_committed"}


def test_a_committed_receipt_is_never_overwritten(tmp_path: Path) -> None:
    """Review I1: a receipt another execution committed stays, however long this one runs, and when whether one
    is up cannot be read, nothing is written at all."""

    earlier = b'{"receipt": "committed by another execution of this attempt"}\n'
    # Past the data GETs' fetch window (1020 s): a duplicate commits while this stage runs 1100 s.
    late = WorkerWorld(tmp_path / "late")

    def long_stage(*, descriptor: dict, **_: object) -> dict:
        outputs = _outputs(late, descriptor)
        late.store.put_object(Bucket=B2_BUCKET, Key=late.key("receipt.json"), Body=earlier)
        late.clock.advance(1100)
        return outputs

    assert late.run(long_stage) == 0
    assert (late.versions("receipt.json"), late.staged("blobs.tar"), late.staged("index.json")) == ([earlier], None, None)
    assert json.loads(late.logs[-1]) == {"mode": "execute", "status": "receipt_already_committed"}

    # A failure receipt checks too: another execution committed before this one's stage hit an undeclared write.
    failing = WorkerWorld(tmp_path / "failing")

    def stray(*, descriptor: dict, **_: object) -> dict:
        outputs = _outputs(failing, descriptor)
        failing.local("/var/lib/blueprint/stray.json").write_bytes(b"{}")
        failing.store.put_object(Bucket=B2_BUCKET, Key=failing.key("receipt.json"), Body=earlier)
        return outputs

    assert failing.run(stray) == 0 and failing.versions("receipt.json") == [earlier]

    # Execute committed, then died before exiting 0: bootstrap keeps that receipt rather than failing over it.
    killed = WorkerWorld(tmp_path / "killed")
    execute = killed.runtime(run_stage=lambda *, descriptor, **_: _outputs(killed, descriptor),
                             handlers={killed.descriptor["stage"]: "x:y"})

    def launch(handoff: dict) -> int:
        assert worker.execute_attempt(handoff, execute) == 0
        return -9

    assert worker.bootstrap(["bootstrap"], killed.runtime(launch=launch)) == 0
    [committed] = [json.loads(data) for data in killed.versions("receipt.json")]
    assert committed["status"] == "succeeded" and _verdict(killed) == ("succeeded", True)
    assert json.loads(killed.logs[-1]) == {"mode": "bootstrap", "status": "receipt_already_committed"}

    # Whether a receipt is up cannot be read at seal: nothing is uploaded, and both processes exit non-zero.
    unknown = WorkerWorld(tmp_path / "unknown")
    sealing: list[bool] = []

    def refuse_receipt_reads(method: str, key: str) -> None:
        if sealing and method == "GET" and key == unknown.key("receipt.json"):
            raise ConnectionResetError("the receipt read failed")

    unknown.http.before = refuse_receipt_reads

    def compiled(*, descriptor: dict, **_: object) -> dict:
        sealing.append(True)
        return _outputs(unknown, descriptor)

    assert unknown.run(compiled) == worker.EXIT_REFUSED
    assert set(_puts(unknown)) == {"heartbeat.json"}
    assert [json.loads(line) for line in unknown.logs[-2:]] == [
        {"mode": mode, "status": "refused", "code": "remote_cpu_worker_receipt_unreadable"}
        for mode in ("execute", "bootstrap")]


def test_undeclared_write_or_changed_input_is_an_infrastructure_failure(tmp_path: Path) -> None:
    clean = WorkerWorld(tmp_path / "clean")
    assert clean.run(lambda *, descriptor, **_: _outputs(clean, descriptor)) == 0
    receipt = clean.receipt()
    assert (receipt["status"], receipt["infrastructure_failures"]) == ("succeeded", [])
    assert _verdict(clean) == ("succeeded", True)
    # Sealed: host-known bytes are indexed by origin, only new bytes are archived, and the receipt goes last.
    runtime_zip, robot = (clean.descriptor["inputs"][index]["digest"] for index in (1, 2))
    from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import MANIFEST_NAME

    assert worker.BUNDLE_MANIFEST == MANIFEST_NAME  # the runtime bundle's own manifest names its members
    index = json.loads(clean.staged("index.json"))
    assert {entry["path"]: entry["origin"] for entry in index["entries"]} == {
        "native-arena-adapter/model.bin": {"input_member": {"input": runtime_zip, "member": "runtime/model.bin"}},
        "native-arena-adapter/result.json": "archive", "robot.json": {"input": robot}}
    blobs = clean.staged("blobs.tar")
    output = receipt["output"]
    assert archive.verify_blobs_stream(io.BytesIO(blobs), index, expected_digest=output["archive"]["digest"]) == {
        "digest": digest_of(blobs), "size_bytes": len(blobs), "blob_count": 1}
    assert output == {"format": "remote_cpu_output.v1", "paths_total": 3, "bytes_total": index["bytes_total"],
                      "index": {"digest": digest_of(clean.staged("index.json")),
                                "size_bytes": len(clean.staged("index.json"))},
                      "archive": {"digest": digest_of(blobs), "size_bytes": len(blobs)},
                      "host_known": {"count": 2, "bytes": index["bytes_total"] - len(NEW_BYTES)}}
    assert _puts(clean)[-3:] == ["blobs.tar", "index.json", "receipt.json"]
    assert receipt["bytes_uploaded"] == sum(
        len(data) for name in ("blobs.tar", "index.json", "heartbeat.json") for data in clean.versions(name))

    compiled = "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes"
    envelope, runtime_path, robot_path = list(clean.inputs)

    def write(path: str, data: bytes = b"{}") -> None:
        clean_path = world.local(path)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_path.write_bytes(data)

    def rewrite(path: str) -> None:
        world.local(path).chmod(0o640)
        world.local(path).write_bytes(b"x" * len(world.inputs[path][0]))
        world.local(path).chmod(0o440)

    cases = {
        "stray": (lambda: write("/var/lib/blueprint/stray/leftover.json"),
                  ["undeclared_write:/var/lib/blueprint/stray",
                   "undeclared_write:/var/lib/blueprint/stray/leftover.json"]),
        "beside-the-output": (lambda: write(f"{compiled}/leftover.json"), [f"undeclared_write:{compiled}/leftover.json"]),
        "link": (lambda: os.symlink("/etc", world.local("/var/lib/blueprint/link")),
                 ["undeclared_write:/var/lib/blueprint/link"]),
        "rewritten": (lambda: rewrite(envelope), [f"input_changed:{envelope}"]),
        "chmod": (lambda: world.local(robot_path).chmod(0o640), [f"input_changed:{robot_path}"]),
        "removed": (lambda: world.local(runtime_path).unlink(), [f"input_changed:{runtime_path}"]),
    }
    for label, (act, expected) in cases.items():
        world = WorkerWorld(tmp_path / label)

        def stage(*, descriptor: dict, **_: object) -> dict:
            outputs = _outputs(world, descriptor)
            act()
            return outputs

        assert world.run(stage) == 0, label
        receipt = world.receipt()
        assert receipt["status"] == "infrastructure_failed", label
        assert receipt["infrastructure_failures"] == sorted(f"infrastructure_failed:{code}" for code in expected), label
        assert _verdict(world) == ("infrastructure_failed", False)
        assert world.staged("blobs.tar") is None and world.staged("index.json") is None, label


def test_release_audit_resolves_relative_reads_against_the_release_root(tmp_path: Path) -> None:
    """Review I2: the host compiles from its checkout, so a read relative to the working directory is a release
    read.  Only an ``os.open`` (no mode in its audit event) may be relative to a directory descriptor."""

    release = tmp_path / "release"
    roots = (os.path.normpath(release) + os.sep,)

    def read(path: object, mode: object = "r", flags: int = os.O_RDONLY) -> str | None:
        found = stage_child.release_read(path, mode, flags, roots=roots, cwd=str(release))
        return None if found is None else found[1]

    assert read(str(release / "docs/schemas/a.json")) == "docs/schemas/a.json"
    assert read("docs/schemas/b.json") == read("./docs/schemas/b.json") == "docs/schemas/b.json"
    assert read("docs/./schemas/../schemas/c.json") == "docs/schemas/c.json"
    assert read("assets/d.json") == "assets/d.json" and read(b"configs/e.json") == "configs/e.json"
    assert read("docs/f.json", mode=None) == read("./docs/f.json", mode=None) == "docs/f.json"
    assert read("schemas/g.json", mode=None) is None  # an os.open that may be relative to a directory descriptor
    assert stage_child.release_read("docs/h.json", "r", 0, roots=roots, cwd=str(tmp_path)) is None
    for path, mode, flags in ((str(release / "docs/i.json"), "w", 0), (str(release / "docs/j.json"), "r+", 0),
                              (str(release / "docs/k.json"), None, os.O_WRONLY | os.O_CREAT),
                              (str(release / "src/__pycache__/l.pyc"), "r", 0), ("../outside.json", "r", 0),
                              (str(tmp_path / "elsewhere.json"), "r", 0), (7, "r", 0)):
        assert read(path, mode, flags) is None, (path, mode, flags)


def _gone(pid: int) -> bool:
    """Killed: no such process, or (on Linux) a zombie its new parent has yet to reap."""
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except OSError:
            pass  # no /proc here, or the process has just gone: the next signal says which
        time.sleep(0.05)
    return False


@pytest.mark.slow
def test_missing_release_path_is_an_infrastructure_failure_not_a_blocked_result(tmp_path: Path) -> None:
    entries = ("blueprint_pipeline.remote_cpu_worker", "blueprint_pipeline.rigid_task_success_contract_schema")
    missing = WorkerWorld(tmp_path / "missing", archive=worker_release_archive(entries=entries, omit=(SUCCESS_SCHEMA,)))
    assert missing.run(stage="reads_success_schema") == 0
    receipt = missing.receipt()
    # The stage turned the missing schema into a blocked result, as the host would; the audit hook saw the miss.
    assert receipt["result"]["status"] == "blocked"
    assert receipt["result"]["blockers"] == ["rigid_task_success_contract_schema_unavailable"]
    assert (receipt["status"], receipt["release_path_misses"]) == ("infrastructure_failed", [SUCCESS_SCHEMA])
    assert receipt["infrastructure_failures"] == [f"infrastructure_failed:release_path_missing:{SUCCESS_SCHEMA}"]
    assert _verdict(missing) == ("infrastructure_failed", False)
    assert missing.staged("blobs.tar") is None

    # With the schema in the release, the same stage's blocked result is deterministic, and terminal.
    present = WorkerWorld(tmp_path / "present", archive=worker_release_archive(entries=entries))
    assert present.run(stage="reads_success_schema") == 0
    receipt = present.receipt()
    assert (receipt["status"], receipt["release_path_misses"], receipt["infrastructure_failures"]) == (
        "blocked", [], [])
    assert receipt["result"]["blockers"] == ["episode_compilation_envelope_invalid"]
    assert _verdict(present) == ("blocked", True)

    # Reads relative to the working directory, the release root as the checkout is the host's, are audited too;
    # an os.open through a directory descriptor, like a stat or a listdir, cannot be (4.3 and shadow mode are
    # the backstop).
    relative = WorkerWorld(tmp_path / "relative", archive=worker_release_archive())
    assert relative.run(stage="reads_relative") == 0
    receipt = relative.receipt()
    assert (receipt["status"], receipt["release_path_misses"]) == ("infrastructure_failed", [
        "assets/missing.json", "configs/missing.json", "docs/schemas/missing-dot.json"])
    assert _verdict(relative) == ("infrastructure_failed", False)


@pytest.mark.slow
def test_stage_child_is_spawned_not_forked_and_heartbeats_advance(tmp_path: Path, monkeypatch) -> None:
    world = WorkerWorld(tmp_path, archive=worker_release_archive(), limits={"heartbeat_interval_seconds": 1})
    spawned: list[tuple[list[str], object, dict]] = []
    popen = subprocess.Popen

    def recording(args, **kwargs):
        spawned.append((list(args), kwargs.get("cwd"), dict(kwargs.get("env") or {})))
        return popen(args, **kwargs)

    def forbidden(*_: object) -> None:
        raise AssertionError("the worker forked a Python process")

    monkeypatch.setattr(worker.subprocess, "Popen", recording)
    monkeypatch.setattr(os, "fork", forbidden)
    assert world.run(stage="slow_compiled") == 0
    receipt = world.receipt()
    assert receipt["status"] == "succeeded", receipt["infrastructure_failures"]

    # One fresh interpreter from the extracted release: not a copy of this process, and no transport in sight.
    release = world.fs / "tmp" / "blueprint-release" / COMMIT
    [(args, cwd, env)] = spawned
    assert args == [sys.executable, "-P", "-m", "blueprint_pipeline.remote_cpu_worker", "stage"]
    assert (Path(cwd), env["PYTHONPATH"]) == (release, str(release / "src"))
    identity = receipt["result"]["identity"]
    assert identity["ppid"] == os.getpid() and identity["pid"] != os.getpid()
    assert (identity["argv"], identity["pytest_loaded"], identity["presigned_in_environment"]) == (["stage"], False, False)
    assert Path(identity["package_file"]).resolve().is_relative_to(release.resolve())
    assert Path(identity["working_directory"]).resolve() == release.resolve()
    assert not {"BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT", "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION"} & set(
        identity["environment_names"])

    # Heartbeats: bootstrap's, execute's at once, then one per interval while the child runs, all before the receipt.
    heartbeats = world.heartbeats()
    for heartbeat in heartbeats:
        contract.validate_heartbeat(heartbeat, attempt_id=world.descriptor["attempt_id"], execution_name=EXECUTION)
    assert [heartbeat["sequence"] for heartbeat in heartbeats] == list(range(1, len(heartbeats) + 1))
    phases = [heartbeat["phase"] for heartbeat in heartbeats]
    assert phases[:2] == ["bootstrap", "fetch"] and phases.count("stage") >= 2, phases
    assert phases == sorted(phases, key=list(contract.PHASES).index)
    assert _puts(world)[-1] == "receipt.json" and "heartbeat.json" not in _puts(world)[-3:]


@pytest.mark.slow
def test_phase_deadlines_still_upload_a_timeout_receipt(tmp_path: Path) -> None:
    # Fetch: the budget runs out while inputs arrive; the rest are never fetched, and the stage never runs.
    fetch = WorkerWorld(tmp_path / "fetch")
    first, second = (_key(item["uri"]) for item in fetch.descriptor["inputs"][:2])
    fetch.http.before = lambda method, key: fetch.clock.advance(400) if key == first else None
    ran: list[int] = []
    assert fetch.run(lambda **_: ran.append(1) or report()) == 0
    receipt = fetch.receipt()
    assert receipt["infrastructure_failures"] == ["infrastructure_failed:phase_deadline:fetch"] and ran == []
    assert receipt["phases"]["fetch"] >= 300 and receipt["phases"]["stage"] is None
    assert first in _gets(fetch) and second not in _gets(fetch)

    # Stage: the child and everything in its session are killed at the deadline; the receipt still uploads.
    stage = WorkerWorld(tmp_path / "stage", archive=worker_release_archive(),
                        limits={"phase_seconds": {"fetch": 300, "stage": 6, "seal_upload": 420}})
    pids = tmp_path / "pids"
    stage.environ["REMOTE_CPU_TEST_PIDS"] = str(pids)
    started = time.monotonic()
    assert stage.run(stage="hangs", clock=time.monotonic) == 0
    assert time.monotonic() - started < 60
    receipt = stage.receipt()
    assert receipt["infrastructure_failures"] == ["infrastructure_failed:phase_deadline:stage"]
    assert 6 <= receipt["phases"]["stage"] < 30 and receipt["phases"]["seal_upload"] is None
    assert all(_gone(int(pid)) for pid in pids.read_text(encoding="utf-8").split())
    assert _verdict(stage) == ("infrastructure_failed", False)

    # Seal and upload: the budget runs out during the archive; the index never goes, the receipt still does.
    seal = WorkerWorld(tmp_path / "seal")
    seal.http.before = lambda method, key: seal.clock.advance(500) if key == seal.key("blobs.tar") else None
    assert seal.run(lambda *, descriptor, **_: _outputs(seal, descriptor)) == 0
    receipt = seal.receipt()
    assert receipt["infrastructure_failures"] == ["infrastructure_failed:phase_deadline:seal_upload"]
    assert (receipt["output"], seal.staged("index.json")) == (None, None) and receipt["phases"]["seal_upload"] >= 420
    assert _puts(seal)[-2:] == ["blobs.tar", "receipt.json"]


class _Response:
    def __init__(self, body: bytes = b"") -> None:
        self.body = body


@pytest.mark.slow
def test_worker_never_logs_or_persists_a_presigned_url(tmp_path: Path, monkeypatch, capfd, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    succeeded = WorkerWorld(tmp_path / "succeeded", archive=worker_release_archive())
    assert succeeded.run() == 0 and succeeded.receipt()["status"] == "succeeded"
    # An authority that expires mid-upload: every later request is refused, and the refusals are typed.
    expired = WorkerWorld(tmp_path / "expired")
    expired.http.before = lambda method, key: expired.clock.advance(3000) if key == expired.key("blobs.tar") else None
    assert expired.run(lambda *, descriptor, **_: _outputs(expired, descriptor)) == worker.EXIT_REFUSED
    assert [json.loads(line) for line in expired.logs[-2:]] == [
        {"mode": mode, "status": "refused", "code": "remote_cpu_worker_receipt_unreadable"}
        for mode in ("execute", "bootstrap")]

    # The launcher hands execute the transport on its stdin: never in argv, the environment or a file.
    handoffs: list[dict] = []
    assert worker.bootstrap(["bootstrap"], WorkerWorld(tmp_path / "launch").runtime(
        launch=lambda handoff: handoffs.append(handoff) or 0)) == 0
    written = io.BytesIO()

    class Process:
        pid, stdin = 0, written

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            return 0

    launched: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(written, "close", lambda: None)
    monkeypatch.setattr(worker.subprocess, "Popen", lambda args, **kwargs: launched.append((args, kwargs)) or Process())
    assert worker._launch_execute(succeeded.runtime(), handoffs[0]) == 0
    [(args, kwargs)] = launched
    assert "X-Amz-Signature=" in written.getvalue().decode()
    assert not any("X-Amz-" in text for text in [*args, *kwargs["env"].values()])
    # The pinned prefix is configuration, not authority, and the transport's name is gone.
    assert [name for name, value in kwargs["env"].items() if "backblazeb2" in value] == [worker.PREFIX_VARIABLE]
    assert not {worker.TRANSPORT_OBJECT_VARIABLE, worker.TRANSPORT_GENERATION_VARIABLE} & set(kwargs["env"])

    # The production transfers type every failure by status or type; the URL is in no message.
    from blueprint_pipeline import safe_outbound_http

    url = handoffs[0]["transport"]["outputs"]["heartbeat.json"]
    for error in (urllib.error.HTTPError(url, 403, f"denied {url}", {}, io.BytesIO(b"")),
                  urllib.error.URLError(f"unreachable {url}"), safe_outbound_http.SafeOutboundHttpError(url)):
        def refuse(*_: object, error=error, **__: object) -> None:
            raise error

        monkeypatch.setattr(safe_outbound_http, "open_request", refuse)
        monkeypatch.setattr(safe_outbound_http, "download_file_observed", refuse)
        for call in (lambda: worker.PresignedTransfers().upload(url, size=2, body=b"{}", content_type="x", timeout=1),
                     lambda: worker.PresignedTransfers().read(url, max_bytes=10, timeout=1),
                     lambda: worker.PresignedTransfers().download(url, tmp_path / "d", max_bytes=10, timeout=1)):
            with pytest.raises(worker.TransferError) as failure:
                call()
            assert "X-Amz-" not in str(failure.value) and failure.value.__cause__ is None
    streamed: list[bytes] = []

    def accept(request, **_: object) -> _Response:
        assert request.get_header("Content-length") == "70000" and request.get_method() == "PUT"
        streamed.append(request.data.read())
        return _Response()

    monkeypatch.setattr(safe_outbound_http, "open_request", accept)
    worker.PresignedTransfers().upload(url, size=70_000, body=lambda sink: sink.write(b"b" * 70_000),
                                       content_type="application/x-tar", timeout=1)
    assert streamed == [b"b" * 70_000]

    # Nothing written, logged or uploaded by the worker carries a presigned URL or its host.
    out, err = capfd.readouterr()
    texts = {"stdout": out, "stderr": err, "caplog": caplog.text,
             "logs": "\n".join(succeeded.logs + expired.logs)}
    texts.update({str(path): path.read_bytes().decode("utf-8", "replace")
                  for path in tmp_path.rglob("*") if path.is_file() and not path.is_symlink()})
    for world in (succeeded, expired):
        for name in ("heartbeat.json", "receipt.json", "index.json", "blobs.tar"):
            texts.update({f"{world.tmp_path.name}:{name}:{index}": data.decode("utf-8", "replace")
                          for index, data in enumerate(world.versions(name))})
    assert any(name.endswith("receipt.json:0") for name in texts)
    for name, text in texts.items():
        assert "X-Amz-Signature" not in text and "backblazeb2" not in text, name


def _registered(*, runtime, descriptor: dict, release: Path, handler: str, seconds: float) -> dict:
    """The registered handler, run in this process with the roots a stage child would get."""

    module, _, name = handler.partition(":")
    function = getattr(__import__(module, fromlist=[name]), name)
    return report(function(descriptor, worker.StageRoots(runtime.filesystem_root, release)))


def test_worker_environment_matches_the_host_census_schema_and_cpu_class_gate(tmp_path: Path, capsys) -> None:
    umask = os.umask(0o022)
    os.umask(umask)
    # ``environment`` prints the host census record (plan 14 PR 1): its schema, and a behaviour digest that
    # leaves build strings out.  On one machine the worker and the host measure the same environment.
    assert worker.main(["environment"]) == 0
    printed = json.loads(capsys.readouterr().out)
    host = census.environment_record()
    assert printed == host and printed["environment_digest"] == census.environment_digest(printed)
    assert set(printed) == {"schema_version", *census.DIGESTED_FIELDS, "informational", "environment_digest"}
    assert worker.main(["environment", "--extra"]) == worker.EXIT_REFUSED
    assert os.umask(umask) == umask  # a mode other than bootstrap leaves the process's umask alone

    # The environment probe, the stage the allocator's preflight runs, is this PR's one registered stage; its
    # receipt carries that record.
    assert worker.STAGE_HANDLERS == {
        "environment_probe": "blueprint_pipeline.remote_cpu_worker_stage:run_environment_probe"}
    probe = WorkerWorld(tmp_path / "probe", stage=contract.PROBE_STAGE, environment_digest=host["environment_digest"])
    runtime = probe.runtime(measure=census.environment_record, run_stage=_registered)
    assert worker.bootstrap(["bootstrap"], probe.runtime(
        measure=census.environment_record, launch=lambda handoff: worker.execute_attempt(handoff, runtime))) == 0
    receipt = probe.receipt()
    assert (receipt["status"], receipt["result"]["status"]) == ("succeeded", "environment_recorded")
    recorded = receipt["result"]["environment"]
    assert recorded["environment_digest"] == receipt["environment"]["environment_digest"] == host["environment_digest"]
    assert {name: recorded[name] == host[name] for name in allocator.DIGESTED_FIELDS} == dict.fromkeys(
        census.DIGESTED_FIELDS, True)
    assert contract.validate_receipt(receipt, descriptor=probe.descriptor, execution_name=EXECUTION)["terminal"]
    index = json.loads(probe.staged("index.json"))
    assert [entry["path"] for entry in index["entries"]] == ["environment.json"]
    assert json.loads(archive.land_subset(
        index=index, reader=lambda offset, length: io.BytesIO(probe.staged("blobs.tar")[offset:offset + length]),
        host_sources={}, destination_root=tmp_path / "landed", selectors=["**"], member_store=None) and (
        tmp_path / "landed" / "environment.json").read_bytes()) == recorded

    # Inline NuRec float math needs a qualified CPU class: bootstrap refuses any other before it fetches.
    inline = {"class": "absent_inline_only", "source_appearance_digest": "sha256:" + "8" * 64}
    qualified = WORKER_RECORD["cpu_class"]
    for label, allowed, measured, refusal in [
        ("unqualified", ["sha256:" + "e" * 64], WORKER_RECORD, "cpu_class_unqualified"),
        ("unmeasured", [qualified], {**WORKER_RECORD, "cpu_class": None}, "cpu_class_unqualified"),
        ("qualified", [qualified], WORKER_RECORD, None),
        # Any stage but the probe must run on the environment its descriptor was dispatched for.
        ("elsewhere", [qualified], {**WORKER_RECORD, "environment_digest": "sha256:" + "9" * 64},
         "environment_mismatch"),
    ]:
        world = WorkerWorld(tmp_path / label, closure=inline, limits={"allowed_cpu_classes": allowed})
        ran: list[int] = []
        runtime = world.runtime(measure=lambda measured=measured: measured, handlers={"episode_compilation": "x:y"},
                                run_stage=lambda **_: ran.append(1) or report(
                                    sealed_result(world.descriptor, blockers=["episode_compilation_envelope_invalid"])))
        code = worker.bootstrap(["bootstrap"], world.runtime(
            measure=lambda measured=measured: measured, launch=lambda handoff: worker.execute_attempt(handoff, runtime)))
        receipt = world.receipt()
        assert code == 0 and receipt["environment"]["cpu_class"] == measured["cpu_class"], label
        if refusal is None:
            assert (receipt["status"], ran) == ("blocked", [1]), label
            continue
        assert receipt["infrastructure_failures"] == [f"infrastructure_failed:{refusal}"] and ran == [], label
        assert _gets(world) == [world.key("receipt.json")] * 2, label
        verdict = contract.validate_receipt(receipt, descriptor=world.descriptor, execution_name=EXECUTION)
        assert f"infrastructure_failed:{refusal}" in verdict["infrastructure_failures"] and not verdict["terminal"]


def test_the_allocators_preflight_completes_against_the_real_worker(tmp_path: Path, monkeypatch) -> None:
    """PR 2's preflight, with this worker where PR 2's tests had a fake: the transport the allocator mints is
    one the worker trusts, its heartbeats renew the lease, and its receipt records the worker environment."""

    from blueprint_pipeline.remote_cpu_transport import SOURCE_FILENAME, SOURCE_KIND, cas_key

    world = RemoteCpuWorld(tmp_path / "host", monkeypatch)
    source = release_archive(COMMIT)
    key = cas_key(SOURCE_KIND, digest_of(source), SOURCE_FILENAME)

    def stage_release(object_store: tuple) -> dict:
        object_store[0].put_object(Bucket=B2_BUCKET, Key=key, Body=source, Metadata={"sha256": digest_of(source)[7:]})
        return {"source_commit": COMMIT, "digest": digest_of(source), "size_bytes": len(source),
                "uri": f"s3://{B2_BUCKET}/{key}"}

    started: list[str] = []

    def run_workers() -> None:  # what Cloud Run starts for each execution: the job's command, once
        for execution in world.jobs.executions[JOB]:
            view, name = world.jobs.get_execution(execution["name"]), execution["name"].rsplit("/", 1)[1]
            if view["runningCount"] != 1 or name in started:
                continue
            started.append(name)
            environ = {**{row["name"]: row["value"] for row in view["template"]["containers"][0]["env"]},
                       "CLOUD_RUN_JOB": view["job"], "CLOUD_RUN_EXECUTION": name, "CLOUD_RUN_TASK_COUNT": "1",
                       "BLUEPRINT_REMOTE_CPU_STAGE": "episode-compilation", worker.PREFIX_VARIABLE: PREFIX_URL}
            common = {"environ": environ, "http": RecordingHttp(world.store), "filesystem_root": tmp_path / name,
                      "clock": world.clock, "measure": census.environment_record, "log": lambda line: None}
            execute = worker.WorkerRuntime(**common, run_stage=_registered)
            assert worker.bootstrap(["bootstrap"], worker.WorkerRuntime(
                **common, reader=world.bucket.reader(),
                launch=lambda handoff, execute=execute: worker.execute_attempt(handoff, execute))) == 0

    world.runtime.stage_release_source, world.on_sleep = stage_release, run_workers
    result = world.run("preflight")
    assert (result["status"], result["outcome"], result["blockers"]) == ("completed", "environment_recorded", [])
    assert len(started) == 1 and result["teardown"]["provider_zero_proven"]
    lease = json.loads((world.root / "leases" / f"{result['probe']['job_id']}.json").read_text(encoding="utf-8"))
    assert lease["state"] == "completed" and lease["heartbeat"]["sequence"] >= 2
    recorded = json.loads((world.root / "environment" / "episode_compilation.json").read_text(encoding="utf-8"))
    assert recorded["worker_environment"] == census.environment_record()
    assert recorded["environment_digest"] == census.environment_record()["environment_digest"]


def test_the_worker_holds_no_allocation_authority() -> None:
    """The worker runs inside the paid execution: nothing in its import closure can admit, mint or dispatch."""

    import ast

    from blueprint_pipeline.task_evaluation_production_chain_preflight import import_closure

    root = Path(worker.__file__).resolve().parents[1]
    closure = import_closure(root.parent, "blueprint_pipeline.remote_cpu_worker")
    assert not {"blueprint_pipeline.paid_resource_admission", "blueprint_pipeline.paid_resource_allocator",
                "blueprint_pipeline.remote_cpu_job_allocator", "blueprint_pipeline.cloud_run_jobs_client",
                "blueprint_pipeline.task_evaluation_configured_scene_object_store",
                "blueprint_pipeline.remote_cpu_transport"} & set(closure), sorted(closure)
    source = Path(worker.__file__).read_text(encoding="utf-8")
    called = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
              for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)}
    assert not called & {"require_paid_resource_admission", "require_paid_resource_admission_grant",
                         "build_paid_lane_admission", "generate_presigned_url", "client", "run_job"}
    assert "boto3" not in source and "service_account" not in source


def test_an_unexpected_error_is_typed_never_printed(tmp_path: Path, monkeypatch, capsys) -> None:
    """A traceback prints its exception's message, which could carry a URL: the worker logs only types."""

    world = WorkerWorld(tmp_path)
    url = world.transport()["outputs"]["heartbeat.json"]

    def launch(handoff: dict) -> int:
        raise OSError(f"could not start execute for {url}")

    assert worker.bootstrap(["bootstrap"], world.runtime(launch=launch)) == 0
    assert world.receipt()["infrastructure_failures"] == ["infrastructure_failed:remote_cpu_worker_raised:OSError"]

    def explode(argv, runtime) -> int:
        raise ValueError(url)

    monkeypatch.setattr(worker, "bootstrap", explode)
    umask = os.umask(0o022)
    try:
        assert worker.main(["bootstrap"]) == worker.EXIT_REFUSED
    finally:
        os.umask(umask)  # bootstrap mode sets the worker's umask on its process: this one is pytest's
    out, err = capsys.readouterr()
    assert json.loads(err) == {"mode": "worker", "status": "failed", "code": "ValueError"} and out == ""
    assert "X-Amz-" not in "\n".join(world.logs)
