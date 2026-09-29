# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_transport.py
#   src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py
#   src/blueprint_pipeline/remote_cpu_job_contract.py
"""ADP-009D/day-28, plan 14 PR 3: the host stages a remote job's inputs and release source in B2 CAS by digest."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_transport as transport
from blueprint_pipeline.task_evaluation_configured_scene_object_store import LARGE_ARTIFACT_KEY_PREFIX
from tests.remote_cpu_fakes import FakeArtifactStore

ROOT = Path(__file__).resolve().parents[1]
B2 = "b2-bucket"
COMMIT = "c" * 40
GIT_IDENTITY = ("-c", "user.name=Plan 14", "-c", "user.email=plan14@example.invalid", "-c", "commit.gpgsign=false")


class RecordingStore(FakeArtifactStore):
    """The B2 fake, also recording the order of HEADs and started uploads."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.events: list[tuple[str, str]] = []

    def head_object(self, **kwargs):
        self.events.append(("head", kwargs["Key"]))
        return super().head_object(**kwargs)

    def create_multipart_upload(self, **kwargs):
        self.events.append(("upload", kwargs["Key"]))
        return super().create_multipart_upload(**kwargs)


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _host_file(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "digest": _digest(data), "size_bytes": len(data)}


def _key(kind: str, digest: str, filename: str) -> str:
    return f"{LARGE_ARTIFACT_KEY_PREFIX}/{kind}/sha256/{digest.removeprefix('sha256:')}/{filename}"


def _objects(store: FakeArtifactStore, kind: str) -> dict[str, bytes]:
    return {key: versions[-1].data for key, versions in store.buckets[B2].items()
            if f"/{kind}/" in key and versions and versions[-1].data is not None}


def _snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    return {str(path.relative_to(root)): (path.lstat().st_size, path.lstat().st_mtime_ns, path.lstat().st_mode)
            for path in sorted(root.rglob("*"))}


def _release_tar(commit: str = COMMIT) -> bytes:
    """A recipe-v2 shaped archive, as ``git archive --format=tar`` writes one: a pax commit comment first."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT, pax_headers={"comment": commit}) as archive:
        for name, data in (("pyproject.toml", b"[project]\nname = 'x'\n"),
                           ("src/blueprint_pipeline/__init__.py", b""),
                           ("docs/schemas/example.v1.schema.json", b'{"type": "object"}\n')):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o664, 1_790_000_000
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_inputs_are_staged_to_cas_once_by_digest(tmp_path: Path) -> None:
    store = RecordingStore(bucket=B2)
    host = tmp_path / "host"
    runtime = hashlib.sha256(b"runtime").digest() * 4096
    envelope = _host_file(host / "processing" / "prep-1.json", b'{"envelope": 1}\n')
    bundle = _host_file(host / "prepared-references" / "prep-1" / "runtime.zip", runtime)
    same_bytes = _host_file(host / "prepared-references" / "prep-2" / "runtime.zip", runtime)

    staged = transport.stage_inputs([envelope, bundle, same_bytes], client=store, bucket=B2)

    # One content-addressed object per digest, whatever the host path; the duplicate moves no byte.
    assert [row["uploaded"] for row in staged] == [True, True, False]
    for row, item in zip(staged, (envelope, bundle, same_bytes)):
        key = _key("remote-cpu-input", item["digest"], "input.bin")
        assert row == {"digest": item["digest"], "size_bytes": item["size_bytes"], "uri": f"s3://{B2}/{key}",
                       "uploaded": row["uploaded"]}
        assert contract._cas_uri_reasons(row["uri"], row["digest"], "uri") == []
        # HEAD by digest comes before any upload of that digest.
        assert store.events.index(("head", key)) < store.events.index(("upload", key))
    assert _objects(store, "remote-cpu-input") == {
        _key("remote-cpu-input", envelope["digest"], "input.bin"): b'{"envelope": 1}\n',
        _key("remote-cpu-input", bundle["digest"], "input.bin"): runtime,
    }
    assert [event for event in store.events if event[0] == "upload"] == [
        ("upload", _key("remote-cpu-input", envelope["digest"], "input.bin")),
        ("upload", _key("remote-cpu-input", bundle["digest"], "input.bin"))]
    for key in _objects(store, "remote-cpu-input"):
        assert store.head_object(Bucket=B2, Key=key)["Metadata"] == {"sha256": key.split("/")[-2]}

    # A later attempt that stages the same inputs sends a HEAD each and moves no byte in either direction.
    moved = (store.bytes_received_from_client, store.bytes_sent_to_client, len(store.operations))
    again = transport.stage_inputs([bundle, envelope], client=store, bucket=B2)
    assert [row["uploaded"] for row in again] == [False, False]
    assert [row["uri"] for row in again] == [staged[1]["uri"], staged[0]["uri"]]
    assert (store.bytes_received_from_client, store.bytes_sent_to_client, len(store.operations)) == moved

    # Bytes that are not the declared digest never reach CAS; the upload is abandoned, not completed.
    declared = _host_file(host / "changed.json", b"declared bytes")
    Path(declared["path"]).write_bytes(b"replaced bytes")
    with pytest.raises(transport.RemoteCpuTransportError) as changed:
        transport.stage_inputs([declared], client=store, bucket=B2)
    assert changed.value.code.startswith("remote_cpu_input_publication_failed:")
    assert _key("remote-cpu-input", declared["digest"], "input.bin") not in store.buckets[B2]
    assert store.uploads == {}
    shorter = dict(declared, size_bytes=declared["size_bytes"] + 1)
    with pytest.raises(transport.RemoteCpuTransportError) as resized:
        transport.stage_inputs([shorter], client=store, bucket=B2)
    assert resized.value.code == "remote_cpu_input_size_mismatch"

    # An object already at that key with another identity is refused, never overwritten.
    imposter = _host_file(host / "imposter.bin", b"honest bytes")
    key = _key("remote-cpu-input", imposter["digest"], "input.bin")
    store.put_object(Bucket=B2, Key=key, Body=b"other bytes!", Metadata={"sha256": "0" * 64})
    with pytest.raises(transport.RemoteCpuTransportError) as mismatch:
        transport.stage_inputs([imposter], client=store, bucket=B2)
    assert mismatch.value.code == "remote_cpu_cas_identity_mismatch"
    assert store.get_object(Bucket=B2, Key=key)["Body"].read() == b"other bytes!"

    # A symlink or an empty input is refused before anything is sent.
    link = host / "link.json"
    link.symlink_to(host / "processing" / "prep-1.json")
    before = len(store.operations)
    for row, code in (({**envelope, "path": str(link)}, "remote_cpu_input_source_invalid"),
                      (_host_file(host / "empty.bin", b""), "remote_cpu_input_empty"),
                      ({**envelope, "digest": "sha256:xyz"}, "remote_cpu_input_invalid")):
        with pytest.raises(transport.RemoteCpuTransportError) as refused:
            transport.stage_inputs([row], client=store, bucket=B2)
        assert refused.value.code == code
    assert len(store.operations) == before


def test_staging_writes_nothing_to_host_disk(tmp_path: Path, monkeypatch) -> None:
    store = RecordingStore(bucket=B2)
    host = tmp_path / "host"
    files = [_host_file(host / "processing" / "prep-1.json", b'{"envelope": 1}\n'),
             _host_file(host / "prepared-references" / "runtime.zip", b"PK" * 70_000)]
    checkout = host / "checkout"
    checkout.mkdir()
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    archive_bytes = _release_tar()
    streamed: list[tuple[Path, str]] = []

    def archive(repository: Path, source_commit: str, sink) -> None:
        streamed.append((repository, source_commit))
        for offset in range(0, len(archive_bytes), 7000):
            sink.write(archive_bytes[offset:offset + 7000])

    writes: list[str] = []
    real_open, real_os_open = io.open, os.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            writes.append(str(path))
        return real_os_open(path, flags, *args, **kwargs)

    before = _snapshot(tmp_path)
    with monkeypatch.context() as patched:
        patched.setenv("TMPDIR", str(temporary))
        patched.setattr(tempfile, "tempdir", str(temporary))
        for module in (builtins, io):
            patched.setattr(module, "open", guarded_open)
        patched.setattr(os, "open", guarded_os_open)
        staged = transport.stage_inputs(files, client=store, bucket=B2)
        first = transport.publish_release_source(repository=checkout, source_commit=COMMIT, client=store, bucket=B2,
                                                 archive=archive)
        second = transport.publish_release_source(repository=checkout, source_commit=COMMIT, client=store,
                                                  bucket=B2, archive=archive)
    assert writes == []
    assert _snapshot(tmp_path) == before and list(temporary.iterdir()) == []
    assert [row["uploaded"] for row in staged] == [True, True]

    # Recipe v2, streamed twice (hash, then upload) only when the HEAD by digest misses; never re-uploaded.
    digest = _digest(archive_bytes)
    key = _key("remote-cpu-source", digest, "source.tar")
    assert first == {"source_commit": COMMIT, "recipe": "git_archive_tar.v2",
                     "paths": ["src", "docs/schemas", "pyproject.toml"], "digest": digest,
                     "size_bytes": len(archive_bytes), "uri": f"s3://{B2}/{key}", "uploaded": True}
    assert second == {**first, "uploaded": False}
    assert streamed == [(checkout, COMMIT)] * 3
    assert _objects(store, "remote-cpu-source") == {key: archive_bytes}
    assert store.events.index(("head", key)) < store.events.index(("upload", key))
    assert contract._cas_uri_reasons(first["uri"], digest, "uri", kind="remote-cpu-source", filename="source.tar") == []

    # A commit that is not 40 hex (an option, say) never reaches git.
    for commit in ("--output=/tmp/x", "HEAD", "C" * 40):
        with pytest.raises(transport.RemoteCpuTransportError) as refused:
            transport.publish_release_source(repository=checkout, source_commit=commit, client=store, bucket=B2,
                                             archive=archive)
        assert refused.value.code == "remote_cpu_release_commit_invalid"
    assert len(streamed) == 3


def test_the_allocator_preflight_stages_the_release_it_runs_from(monkeypatch) -> None:
    from blueprint_pipeline import remote_cpu_job_allocator as allocator

    store, calls = RecordingStore(bucket=B2), []
    monkeypatch.setattr(transport, "release_commit", lambda repository: calls.append(("commit", repository)) or COMMIT)
    monkeypatch.setattr(transport, "publish_release_source", lambda **kwargs: calls.append(("publish", kwargs)) or {})
    runtime = allocator.RemoteCpuRuntime()
    assert runtime.stage_release_source is transport.publish_running_release
    assert runtime.stage_release_source((store, B2, "us-west-004")) == {}
    assert calls == [("commit", ROOT), ("publish", {"repository": ROOT, "source_commit": COMMIT, "client": store,
                                                    "bucket": B2})]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", *GIT_IDENTITY, "-C", str(repository), *arguments], check=True, capture_output=True,
                          text=True, env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()


@pytest.mark.slow
def test_release_commit_is_the_clean_head_and_git_archive_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "release"
    for name, data in (("src/blueprint_pipeline/__init__.py", b""), ("src/AGENTS.md", b"agents\n"),
                       ("docs/schemas/a.v1.schema.json", b"{}\n"), ("docs/other.md", b"not shipped\n"),
                       ("tests/test_x.py", b"not shipped\n"), ("pyproject.toml", b"[project]\n")):
        (repository / name).parent.mkdir(parents=True, exist_ok=True)
        (repository / name).write_bytes(data)
    _git(repository, "init", "-q")
    _git(repository, "add", "-A")
    _git(repository, "commit", "-q", "-m", "release")
    head = _git(repository, "rev-parse", "HEAD")
    (repository / "untracked.txt").write_text("the unit ignores untracked files\n", encoding="utf-8")
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    store = FakeArtifactStore(bucket=B2)

    before = _snapshot(repository)
    assert transport.release_commit(repository) == head
    reference = transport.publish_release_source(repository=repository, source_commit=head, client=store, bucket=B2)
    assert _snapshot(repository) == before and list(temporary.iterdir()) == []
    data = store.get_object(Bucket=B2, Key=reference["uri"].removeprefix(f"s3://{B2}/"))["Body"].read()
    assert (_digest(data), len(data)) == (reference["digest"], reference["size_bytes"])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        names = sorted(member.name for member in archive.getmembers())
        assert archive.pax_headers["comment"] == head
    assert names == ["docs", "docs/schemas", "docs/schemas/a.v1.schema.json", "pyproject.toml", "src",
                     "src/AGENTS.md", "src/blueprint_pipeline", "src/blueprint_pipeline/__init__.py"]

    (repository / "src" / "AGENTS.md").write_text("edited\n", encoding="utf-8")
    with pytest.raises(transport.RemoteCpuTransportError) as dirty:
        transport.release_commit(repository)
    assert dirty.value.code == "remote_cpu_release_checkout_dirty"


_AUDIT = textwrap.dedent('''
    import atexit, json, os, sys

    release, out = sys.argv[1], sys.argv[2]
    import blueprint_pipeline  # the release's package, before tests/conftest.py puts the checkout's src first
    from blueprint_pipeline.remote_cpu_worker import install_release_audit

    roots = tuple({os.path.normpath(release) + os.sep, os.path.realpath(release) + os.sep})
    found = set()

    def record(event, args):
        # The reads that did find their file, for the report; the worker's own audit records the misses.
        if event == "open" and len(args) >= 3 and isinstance(args[0], str) and os.path.isabs(args[0]):
            root = next((root for root in roots if args[0].startswith(root)), None)
            if root is not None and "__pycache__" not in args[0] and os.path.isfile(args[0]):
                found.add(os.path.normpath(args[0])[len(root):])

    def dump():
        with open(out, "w", encoding="utf-8") as stream:
            json.dump({"found": sorted(found), "missing": sorted(missing), "package": blueprint_pipeline.__file__},
                      stream)

    missing = install_release_audit(release)
    sys.addaudithook(record)
    atexit.register(dump)
    import pytest

    sys.exit(pytest.main(sys.argv[3:]))
''')
COMPILE_TESTS = ("tests/test_task_evaluation_native_arena_episode_compiler.py",
                 "tests/test_task_evaluation_episode_compilation_worker.py")


def _compile_from(release: Path, tmp_path: Path) -> dict[str, list[str]]:
    """Run the compiler and compile-worker tests with the release tree as the code, from inside it.

    The tests and their fixture inputs come from the checkout; every ``blueprint_pipeline`` module and
    every path the code derives from its own location (``parents[2]``) comes from ``release``.
    """

    out = tmp_path / "audit.json"
    run = subprocess.run(
        [sys.executable, "-c", _AUDIT, str(release), str(out), "-q", "-p", "no:cacheprovider", "--rootdir",
         str(ROOT), "-c", str(ROOT / "pyproject.toml"), *(str(ROOT / name) for name in COMPILE_TESTS)],
        cwd=release, capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": f"{release / 'src'}{os.pathsep}{ROOT}", "PYTHONDONTWRITEBYTECODE": "1",
             "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache")})
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_release_source_archive_contains_every_repo_root_read_of_the_compile(tmp_path: Path, monkeypatch) -> None:
    head = _git(ROOT, "rev-parse", "HEAD")

    def status() -> tuple[str, int]:
        index = _git(ROOT, "rev-parse", "--path-format=absolute", "--git-path", "index")
        return _git(ROOT, "status", "--porcelain"), Path(index).stat().st_mtime_ns

    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    store = FakeArtifactStore(bucket=B2)
    before = status()
    reference = transport.publish_release_source(repository=ROOT, source_commit=head, client=store, bucket=B2)
    assert status() == before and list(temporary.iterdir()) == []
    data = store.get_object(Bucket=B2, Key=reference["uri"].removeprefix(f"s3://{B2}/"))["Body"].read()
    assert (_digest(data), len(data)) == (reference["digest"], reference["size_bytes"])
    release = tmp_path / "release"
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.pax_headers["comment"] == head
        members = archive.getmembers()
        assert all(member.name in {"src", "docs", "docs/schemas", "pyproject.toml"}
                   or member.name.startswith(("src/", "docs/schemas/")) for member in members)
        archive.extractall(release, filter="data")

    # The compile, run with the extracted archive as its code: the compiler and compile-worker tests, under the
    # worker's own release audit (``remote_cpu_worker.install_release_audit``).  No read under the release misses.
    audit = _compile_from(release, tmp_path)
    assert audit["missing"] == []
    assert Path(audit["package"]).is_relative_to(release)
    assert "src/blueprint_pipeline/task_evaluation_native_arena_episode_compiler.py" in audit["found"]
    # The reads outside src are schemas found through parents[2]: recipe v1 (src and pyproject.toml alone)
    # would have shipped a tree that cannot compile (plan 14 C1).
    outside_src = [path for path in audit["found"] if not path.startswith("src/")]
    assert "docs/schemas/task_evaluation_launch_preparation_request.v1.schema.json" in outside_src
    assert all(path.startswith("docs/schemas/") or path == "pyproject.toml" for path in outside_src), outside_src
