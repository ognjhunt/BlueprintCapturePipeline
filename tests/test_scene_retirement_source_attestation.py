"""Admitted SHA256 authority and public proof transport, using synthetic data."""

import ast
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import urllib.request

import pytest

import fcntl
import tempfile


@pytest.fixture
def tmp_path():
    fd = os.open(
        Path.home() / ".blueprint-scene-fixture.lock",
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    with os.fdopen(fd, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(
            prefix=".blueprint-source-proof-", dir=Path.home()
        ) as name:
            yield Path(name).resolve()


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
COMMIT = "a" * 40
PREDICATE = (
    "https://github.com/ognjhunt/BlueprintCapturePipeline/attestations/source-sha256-manifest/v1"
)


def installer():
    spec = importlib.util.spec_from_file_location(
        "admitted_installer_tests", SCRIPTS / "install_scene_retirement_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deployer(tmp_path):
    tree = ast.parse((SCRIPTS / "deploy_control_plane_commit.py").read_bytes())
    names = {
        "_scene_source_delivery",
        "_scene_source_cache_publish",
        "_scene_source_selected_proof",
        "_scene_source_attestation",
        "_bootstrap_scene_retirement_installer",
    }
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "Path": Path,
        "Any": object,
        "os": os,
        "hashlib": hashlib,
        "json": json,
        "re": re,
        "stat": stat,
        "subprocess": subprocess,
        "signal": signal,
        "time": time,
        "base64": base64,
        "urllib": __import__("urllib"),
        "fcntl": __import__("fcntl"),
        "math": __import__("math"),
        "ControlPlaneDeployError": ValueError,
        "_SCENE_RUNTIME_OWNER": os.getuid(),
        "_SCENE_RUNTIME_BOOT_ROOT": tmp_path / "runtime",
        "_SCENE_SOURCE_ATTESTATIONS": tmp_path / "proofs",
        "_SCENE_SOURCE_GH": tmp_path / "gh",
    }
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(SCRIPTS / "deploy_control_plane_commit.py"),
            "exec",
        ),
        namespace,
    )
    return namespace


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def row(path, body=b"data"):
    return {
        "path": path,
        "git_blob_oid": "b" * 40,
        "mode": "100644",
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def manifest():
    pipeline = [
        row(path)
        for path in (
            "deploy/systemd/a.service",
            "pyproject.toml",
            "scripts/deploy_control_plane_commit.py",
            "scripts/install_scene_retirement_runtime.py",
            "scripts/release_source_manifest.py",
            "src/blueprint_pipeline/__init__.py",
            "uv.lock",
        )
    ]
    return {
        "schema_version": "blueprint.source_sha256_manifest.v1",
        "sources": [
            {
                "repository": "ognjhunt/BlueprintCapturePipeline",
                "commit": COMMIT,
                "tree": "c" * 40,
                "roots": [
                    "src/blueprint_pipeline",
                    "scripts",
                    "deploy/systemd",
                    "uv.lock",
                    "pyproject.toml",
                ],
                "files": pipeline,
            },
            {
                "repository": "ognjhunt/BlueprintContracts",
                "commit": "7708a4e4c5dedeeb39cc73d3f6869304de295b81",
                "tree": "d" * 40,
                "roots": ["src/blueprint_contracts", "blueprint_contracts"],
                "files": [row("src/blueprint_contracts/__init__.py")],
            },
        ],
    }


def statement(value):
    return {
        "_type": "https://in-toto.io/Statement/v1",
        "predicateType": PREDICATE,
        "predicate": value,
        "subject": [
            {
                "name": "source-sha256-manifest.json",
                "digest": {"sha256": hashlib.sha256(canonical(value)).hexdigest()},
            }
        ],
    }


@pytest.mark.parametrize(
    "change", ["missing", "wrong-repo", "wrong-commit", "unsafe-path", "wrong-sha256", "wrong-size"]
)
def test_installer_refuses_unadmitted_or_invalid_inventory_before_git(
    tmp_path, monkeypatch, change
):
    module = installer()
    value = manifest()
    if change == "missing":
        value = None
    elif change == "wrong-repo":
        value["sources"][0]["repository"] = "other/repository"
    elif change == "wrong-commit":
        value["sources"][0]["commit"] = "e" * 40
    elif change == "unsafe-path":
        value["sources"][0]["files"][0]["path"] = "../foreign"
    elif change == "wrong-sha256":
        value["sources"][0]["files"][0]["sha256"] = "not-sha256"
    else:
        value["sources"][0]["files"][0]["size"] = True
    monkeypatch.setattr(
        module, "_sdk_git_command", lambda *a, **kw: pytest.fail("admission must precede Git reads")
    )
    with pytest.raises(ValueError, match="scene_retirement_runtime_unproven"):
        module._authenticated_git_entries(
            tmp_path, COMMIT, ("scripts",), time.monotonic() + 5, source_manifest=value
        )


def test_signed_blob_uses_sha256_authority_even_when_git_selector_matches(tmp_path, monkeypatch):
    module = installer()
    trusted = b"trusted bytes"
    entry = ("100644", "b" * 40, "scripts/a.py", len(trusted), hashlib.sha256(trusted).hexdigest())
    header = ("b" * 40 + " blob " + str(len(trusted)) + "\n").encode()
    output = header + b"hostile bytes" + b"\n"
    monkeypatch.setattr(
        module,
        "_sdk_git_command",
        lambda path, args, *a, **kw: header if "--batch-check" in args else output,
    )
    with pytest.raises(ValueError, match="scene_retirement_runtime_unproven"):
        module._signed_release_blobs(tmp_path, [entry], time.monotonic() + 5)


class Response:
    status = 200

    def __init__(self, url, value):
        self.url = url
        self.body = io.BytesIO(canonical(value))
        self.fp = type(
            "FP",
            (),
            {
                "raw": type(
                    "Raw", (), {"_sock": type("Socket", (), {"settimeout": lambda *a: None})()}
                )()
            },
        )()

    def read1(self, size):
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def public_transport(value, *, change=None, copies=1, reverse=False):
    digest = hashlib.sha256(canonical(value)).hexdigest()
    signed = statement(value)
    if change == "subject":
        signed["subject"][0]["digest"] = {"sha256": "0" * 64}
    bundle = {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "dsseEnvelope": {
            "payloadType": "application/vnd.in-toto+json",
            "payload": base64.b64encode(canonical(signed)).decode(),
        },
    }
    run = {
        "id": 123,
        "path": ".github/workflows/ci.yml",
        "head_sha": COMMIT,
        "head_branch": "main",
        "event": "push",
        "status": "completed",
        "conclusion": "success",
        "head_repository": {"full_name": "ognjhunt/BlueprintCapturePipeline"},
    }
    if change == "ci":
        run["conclusion"] = "failure"
    requests = []

    def open_request(request, timeout):
        url = request.full_url
        assert 0 < timeout <= 30 and not request.has_header("Authorization")
        assert url.startswith("https://api.github.com/repos/ognjhunt/BlueprintCapturePipeline/")
        requests.append(url)
        if "/actions/workflows/" in url:
            result = {"workflow_runs": [run]}
        elif "/artifacts?" in url:
            result = {
                "total_count": 1,
                "artifacts": [{"expired": False, "name": "source-sha256-" + COMMIT + "-" + digest}],
            }
        else:
            result = {
                "attestations": [
                    {"bundle": bundle | {"synthetic_signature": index}} for index in (reversed(range(copies)) if reverse else range(copies))
                ]
            }
        return Response(url, result)

    return open_request, requests


def test_public_delivery_stages_only_data_and_reuses_exact_protected_bytes(tmp_path, monkeypatch):
    namespace = deployer(tmp_path)
    value = manifest()
    open_request, requests = public_transport(value)
    namespace["_scene_source_attestation"] = lambda *a, **kw: (
        value,
        canonical(value),
        b"{}",
    )  # mocked crypto boundary
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(open_request)})(),
    )
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    path = namespace["_SCENE_SOURCE_ATTESTATIONS"] / COMMIT / "source-sha256-manifest.json"
    original = path.stat()
    assert path.read_bytes() == canonical(value) and stat.S_IMODE(original.st_mode) == 0o444
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    assert path.stat().st_ino == original.st_ino and len(requests) == 6
    assert not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


@pytest.mark.parametrize("change", ["ci", "subject"])
def test_public_metadata_or_subject_failure_never_publishes(tmp_path, monkeypatch, change):
    namespace = deployer(tmp_path)
    open_request, _ = public_transport(manifest(), change=change)
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(open_request)})(),
    )
    with pytest.raises(ValueError, match="deploy_scene_retirement_runtime_unproven"):
        namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    assert not namespace["_SCENE_SOURCE_ATTESTATIONS"].exists()


def test_missing_host_verifier_refuses_before_network_or_candidate_execution(tmp_path):
    namespace = deployer(tmp_path)
    namespace["_scene_source_delivery"] = lambda *a, **kw: pytest.fail(
        "missing verifier precedes network"
    )
    with pytest.raises(FileNotFoundError):
        namespace["_scene_source_attestation"](COMMIT, deadline=time.monotonic() + 5)
    assert not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


@pytest.mark.parametrize(
    "change", ["none", "certificate", "timestamp", "subject", "predicate", "wrong-contract"]
)
def test_direct_bootstrap_crypto_and_policy_refusal_preserves_no_installer(
    tmp_path, monkeypatch, change
):
    namespace = deployer(tmp_path)
    value = manifest()
    if change == "wrong-contract":
        value["sources"][1]["commit"] = "f" * 40
    root = namespace["_SCENE_SOURCE_ATTESTATIONS"] / COMMIT
    root.mkdir(parents=True)
    (root / "source-sha256-manifest.json").write_bytes(canonical(value))
    (root / "source-provenance.sigstore.json").write_bytes(b"{}")
    tool = namespace["_SCENE_SOURCE_GH"]
    tool.write_bytes(b"# synthetic protected command boundary\n")
    tool.chmod(0o755)
    verified = {
        "signature": {"certificate": {"issuer": "synthetic"}},
        "verifiedTimestamps": [{"timestamp": "synthetic"}],
        "statement": statement(value),
    }
    if change == "certificate":
        verified["signature"]["certificate"] = {}
    elif change == "timestamp":
        verified["verifiedTimestamps"] = []
    elif change == "subject":
        verified["statement"]["subject"][0]["digest"] = {"sha256": "0" * 64}
    elif change == "predicate":
        verified["statement"]["predicate"] = {}
    output = canonical([{"verificationResult": verified}])
    actual = subprocess.Popen
    calls = []

    def crypto_boundary(command, **kwargs):
        assert command[1:3] == ["attestation", "verify"]
        assert "--bundle" in command and "--deny-self-hosted-runners" in command
        assert command[command.index("--source-digest") + 1] == COMMIT
        assert command[command.index("--signer-digest") + 1] == COMMIT
        assert command[command.index("--source-ref") + 1] == "refs/heads/main"
        assert command[command.index("--cert-identity") + 1].endswith(
            "/.github/workflows/ci.yml@refs/heads/main"
        )
        assert (
            command[command.index("--cert-oidc-issuer") + 1]
            == "https://token.actions.githubusercontent.com"
        )
        assert command[command.index("--predicate-type") + 1] == PREDICATE
        assert kwargs["env"]["HOME"] == "/nonexistent"
        calls.append(command)
        return actual(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(" + repr(output) + ")"],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", crypto_boundary)
    if change == "none":
        admitted, raw, _ = namespace["_scene_source_attestation"](
            COMMIT, deadline=time.monotonic() + 10
        )
        assert admitted == value and raw == canonical(value)
    else:
        with pytest.raises(ValueError, match="deploy_scene_retirement_runtime_unproven"):
            namespace["_scene_source_attestation"](COMMIT, deadline=time.monotonic() + 10)
    assert len(calls) == 1 and not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


def test_wheel_redirect_is_refused_before_following_another_origin():
    module = installer()
    handler = module._SdkNoRedirect()
    with pytest.raises(ValueError, match="scene_retirement_runtime_unproven"):
        handler.redirect_request(None, None, 302, "redirect", {}, "https://other.invalid/file.whl")


def test_wrapper_embeds_the_same_admission_before_first_privileged_execution():
    source = (SCRIPTS / "deploy_control_plane_commit.py").read_text()
    shell = (SCRIPTS / "install_live_pipeline_control_plane.sh").read_text()
    embedded = shell.split("<<'PY_RUNTIME'\n", 1)[1].split("\nPY_RUNTIME", 1)[0]
    original = ast.parse(source)
    wrapper = ast.parse(embedded)
    names = {
        "_scene_source_delivery",
        "_scene_source_cache_publish",
        "_scene_source_selected_proof",
        "_scene_source_attestation",
        "_bootstrap_scene_retirement_installer",
        "_scene_runtime_diagnostic",
        "_prepare_scene_retirement_runtime",
    }
    left = {
        node.name: ast.dump(node, include_attributes=False)
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    right = {
        node.name: ast.dump(node, include_attributes=False)
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert left == right
    assert "sha1(" not in embedded and "sha1(" not in source
    bootstrap = ast.get_source_segment(
        embedded,
        next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_bootstrap_scene_retirement_installer"
        ),
    )
    assert bootstrap.index("_scene_source_attestation(") < bootstrap.index("body = admitted")
    assert "_deadline=deadline" in embedded
    assert shell.index("PY_RUNTIME\nfi") < shell.index("run chown -R")


@pytest.mark.parametrize("deadline", [0.0, float("nan"), float("inf"), -float("inf")])
def test_deployer_earlier_or_nonfinite_origin_refuses_before_any_proof(
    tmp_path, monkeypatch, deadline
):
    namespace = deployer(tmp_path)
    source = (SCRIPTS / "deploy_control_plane_commit.py").read_bytes()
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_prepare_scene_retirement_runtime", "_scene_runtime_diagnostic"}
    ]
    namespace.update({"math": __import__("math"), "_SCENE_RUNTIME_INSTALL_SECONDS": 900})
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(SCRIPTS / "deploy_control_plane_commit.py"),
            "exec",
        ),
        namespace,
    )
    namespace["_bootstrap_scene_retirement_installer"] = lambda *a, **kw: pytest.fail(
        "bad deadline must refuse before proof"
    )
    with pytest.raises(ValueError, match="deploy_scene_retirement_runtime_unproven"):
        namespace["_prepare_scene_retirement_runtime"](
            source_repo=tmp_path, source_commit=COMMIT, _deadline=deadline
        )
    assert not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


def test_interrupted_public_cache_write_resumes_exact_prefix_without_replacement(
    tmp_path, monkeypatch
):
    namespace = deployer(tmp_path)
    value = manifest()
    request, _ = public_transport(value)
    namespace["_scene_source_attestation"] = lambda *a, **kw: (
        value,
        canonical(value),
        b"{}",
    )  # mocked crypto boundary
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(request)})(),
    )
    write = os.write
    interrupted = [False]

    def interrupted_write(fd, body):
        if not interrupted[0] and bytes(body) == canonical(value):
            interrupted[0] = True
            write(fd, body[:20])
            raise OSError("synthetic interrupted proof cache")
        return write(fd, body)

    monkeypatch.setattr(os, "write", interrupted_write)
    with pytest.raises(OSError, match="synthetic interrupted"):
        namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    pending = next(
        namespace["_SCENE_SOURCE_ATTESTATIONS"].rglob("source-sha256-manifest.json.public-pending")
    )
    inode = pending.stat().st_ino
    assert pending.read_bytes() == canonical(value)[:20]
    monkeypatch.setattr(os, "write", write)
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    target = pending.with_name("source-sha256-manifest.json")
    assert target.stat().st_ino == inode and target.read_bytes() == canonical(value)
    assert not pending.exists()


@pytest.mark.parametrize("invalid_first", [False, True])
def test_same_sha_ci_rerun_proofs_require_crypto_admission_before_selection(
    tmp_path, monkeypatch, invalid_first
):
    namespace = deployer(tmp_path)
    value = manifest()
    request, _ = public_transport(value, copies=2)
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(request)})(),
    )
    admitted = []

    def crypto_boundary(commit, *, deadline, _proof_root):
        assert commit == COMMIT and deadline > time.monotonic()
        assert (_proof_root / "source-sha256-manifest.json").read_bytes() == canonical(value)
        proof = json.loads((_proof_root / "source-provenance.sigstore.json").read_bytes())
        admitted.append(proof["synthetic_signature"])
        if invalid_first and proof["synthetic_signature"] == 0:
            raise ValueError("synthetic wrong certificate policy")
        return value, canonical(value), canonical(proof)

    namespace["_scene_source_attestation"] = crypto_boundary
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    assert admitted == ([0, 1] if invalid_first else [0])
    root = namespace["_SCENE_SOURCE_ATTESTATIONS"] / COMMIT
    selected = json.loads((root / "source-provenance.sigstore.json").read_bytes())
    assert selected["synthetic_signature"] == (1 if invalid_first else 0)
    assert not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


def test_reversed_rerun_delivery_reverifies_and_preserves_existing_selected_proof(
    tmp_path, monkeypatch
):
    namespace = deployer(tmp_path)
    value = manifest()
    request, _ = public_transport(value, copies=2)
    transport = [request]
    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *a: type(
            "Opener", (), {"open": staticmethod(lambda *args, **kw: transport[0](*args, **kw))}
        )(),
    )
    admitted = []

    def crypto_boundary(commit, *, deadline, _proof_root):
        raw = (_proof_root / "source-sha256-manifest.json").read_bytes()
        proof = (_proof_root / "source-provenance.sigstore.json").read_bytes()
        assert raw == canonical(value)
        admitted.append(json.loads(proof)["synthetic_signature"])
        return value, raw, proof

    namespace["_scene_source_attestation"] = crypto_boundary
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    path = namespace["_SCENE_SOURCE_ATTESTATIONS"] / COMMIT / "source-provenance.sigstore.json"
    original, body = path.stat(), path.read_bytes()
    transport[0], _ = public_transport(value, copies=2, reverse=True)
    response = transport[0](urllib.request.Request(
        'https://api.github.com/repos/ognjhunt/BlueprintCapturePipeline/attestations/check'
    ),timeout=10)
    assert [row['bundle']['synthetic_signature'] for row in json.loads(response.body.getvalue())['attestations']] == [1,0]
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    assert admitted == [0, 0]
    assert path.stat().st_ino == original.st_ino and path.read_bytes() == body


def test_interrupted_selected_proof_retry_preserves_partial_bytes_when_rerun_order_changes(
    tmp_path,monkeypatch
):
    namespace = deployer(tmp_path)
    value = manifest()
    request,_ = public_transport(value,copies=2)
    transport = [request]
    monkeypatch.setattr(urllib.request,'build_opener',lambda *a:type(
        'Opener',(),{'open':staticmethod(lambda *a,**kw:transport[0](*a,**kw))})())
    admitted = []
    def crypto_boundary(commit,*,deadline,_proof_root):
        assert commit == COMMIT and time.monotonic() < deadline
        raw = (_proof_root/'source-sha256-manifest.json').read_bytes()
        proof = (_proof_root/'source-provenance.sigstore.json').read_bytes()
        assert raw == canonical(value)
        admitted.append(json.loads(proof)['synthetic_signature'])
        return value,raw,proof
    namespace['_scene_source_attestation'] = crypto_boundary
    root = namespace['_SCENE_SOURCE_ATTESTATIONS']/COMMIT
    publish = namespace['_scene_source_cache_publish']
    in_fixed = [False]
    def publication(path,*args,**kwargs):
        in_fixed[0] = path == root
        try:
            return publish(path,*args,**kwargs)
        finally:
            in_fixed[0] = False
    namespace['_scene_source_cache_publish'] = publication
    write = os.write
    def interruption(fd,body):
        if in_fixed[0] and bytes(body).startswith(b'{"dsseEnvelope"'):
            write(fd,body[:-7])
            raise OSError('synthetic selected proof interruption')
        return write(fd,body)
    monkeypatch.setattr(os,'write',interruption)
    with pytest.raises(OSError,match='synthetic selected proof interruption'):
        namespace['_scene_source_delivery'](COMMIT,deadline=time.monotonic()+10)
    pending = root/'source-provenance.sigstore.json.public-pending'
    inode,prefix = pending.stat().st_ino,pending.read_bytes()
    assert (root/'source-sha256-manifest.json').read_bytes() == canonical(value)
    selected = (root/'.public-selected-proof.sha256').read_bytes()
    assert len(selected) == 65 and len(prefix) > 100
    monkeypatch.setattr(os,'write',write)
    transport[0],_ = public_transport(value,copies=2,reverse=True)
    response = transport[0](urllib.request.Request(
        'https://api.github.com/repos/ognjhunt/BlueprintCapturePipeline/attestations/check'
    ),timeout=10)
    assert [row['bundle']['synthetic_signature'] for row in json.loads(response.body.getvalue())['attestations']] == [1,0]
    namespace['_scene_source_delivery'](COMMIT,deadline=time.monotonic()+10)
    proof = root/'source-provenance.sigstore.json'
    assert proof.stat().st_ino == inode and proof.read_bytes().startswith(prefix)
    assert hashlib.sha256(proof.read_bytes()).hexdigest().encode()+b'\n' == selected
    assert json.loads(proof.read_bytes())['synthetic_signature'] == 0
    assert admitted == [0,0] and not pending.exists()
    assert not namespace['_SCENE_RUNTIME_BOOT_ROOT'].exists()


@pytest.mark.parametrize('selection',[b'not-a-digest',b'0'*64+b'\n'])
def test_existing_selected_proof_never_switches_or_admits_an_unknown_proof(
    tmp_path,monkeypatch,selection
):
    namespace = deployer(tmp_path)
    value = manifest()
    request,_ = public_transport(value,copies=2)
    monkeypatch.setattr(urllib.request,'build_opener',lambda *a:type(
        'Opener',(),{'open':staticmethod(request)})())
    root = namespace['_SCENE_SOURCE_ATTESTATIONS']/COMMIT
    root.mkdir(parents=True)
    claim = root/'.public-selected-proof.sha256'
    claim.write_bytes(selection)
    claim.chmod(0o444)
    namespace['_scene_source_attestation'] = lambda *a,**kw:pytest.fail('unknown selected proof may never reach crypto')
    with pytest.raises(ValueError,match='deploy_scene_retirement_runtime_unproven'):
        namespace['_scene_source_delivery'](COMMIT,deadline=time.monotonic()+10)
    assert claim.read_bytes() == selection and not namespace['_SCENE_RUNTIME_BOOT_ROOT'].exists()


def framed_http_response(url, body, framing):
    """Actual HTTPResponse framing/EOF, with an in-memory socket only."""
    import http.client

    class Socket:
        def settimeout(self, value):
            assert 0 < value <= 30

        def makefile(self, mode):
            return io.BufferedReader(Raw(wire, self))

    class Raw(io.RawIOBase):
        def __init__(self, data, sock):
            self.data = io.BytesIO(data)
            self._sock = sock

        def readable(self):
            return True

        def readinto(self, target):
            data = self.data.read(len(target))
            target[: len(data)] = data
            return len(data)

    if framing == "content-length":
        wire = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    elif framing == "chunked":
        wire = (
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            + hex(len(body))[2:].encode()
            + b"\r\n"
            + body
            + b"\r\n0\r\n\r\n"
        )
    else:
        wire = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + body
    response = http.client.HTTPResponse(Socket())
    response.begin()
    response.url = url
    return response


@pytest.mark.parametrize("framing", ["content-length", "chunked", "eof"])
def test_public_delivery_real_http_framing_survives_closed_fp(tmp_path, monkeypatch, framing):
    namespace = deployer(tmp_path)
    value = manifest()
    base_request, _ = public_transport(value)

    def request(req, timeout):
        synthetic = base_request(req, timeout)
        return framed_http_response(req.full_url, synthetic.body.getvalue(), framing)

    monkeypatch.setattr(
        urllib.request,
        "build_opener",
        lambda *a: type("Opener", (), {"open": staticmethod(request)})(),
    )
    namespace["_scene_source_attestation"] = lambda *a, **kw: (value, canonical(value), b"{}")
    namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    assert (
        namespace["_SCENE_SOURCE_ATTESTATIONS"] / COMMIT / "source-sha256-manifest.json"
    ).read_bytes() == canonical(value)


@pytest.mark.parametrize("framing", ["content-length", "chunked", "eof"])
def test_sdk_artifact_real_http_framing_survives_closed_fp(tmp_path, monkeypatch, framing):
    module = installer()
    monkeypatch.setattr(module, "_OWNER", os.getuid())
    monkeypatch.setattr(module, "_RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(module, "_BOOT_ROOT", tmp_path / "boot")
    monkeypatch.setattr(module, "_FREE_FLOOR", 0)
    raw = b"synthetic wheel bytes"
    url = "https://files.pythonhosted.org/synthetic-py3-none-any.whl"
    row = {"url": url, "size": len(raw), "hash": "sha256:" + hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda *a: type(
            "Opener",
            (),
            {
                "open": staticmethod(
                    lambda requested, timeout: framed_http_response(requested, raw, framing)
                )
            },
        )(),
    )
    result = module._sdk_artifact(row, None, time.monotonic() + 10)
    assert result.read_bytes() == raw
