"""Admitted SHA256 authority and public proof transport, using synthetic data."""

import ast
import base64
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import socket
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
        "_SourceDeadlineSocket",
        "_SourceDeadlineHTTPSConnection",
        "_SourceDeadlineHTTPSHandler",
        "_SourceClosingHTTPErrorProcessor",
        "_SourceNoRedirect",
        "_source_response_opener",
        "_scene_source_snappy",
        "_scene_source_selector_bytes",
        "_scene_source_delivery",
        "_scene_source_cache_publish",
        "_scene_source_selected_proof",
        "_scene_source_attestation",
        "_bootstrap_scene_retirement_installer",
    }
    definitions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef,ast.ClassDef)) and node.name in names
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
        "http": __import__("http"),
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


def selector_bytes(commit=COMMIT):
    return canonical({"schema_version": "blueprint.source_commit_selector.v1",
                      "repository": "ognjhunt/BlueprintCapturePipeline", "source_commit": commit})


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
        self.headers = {"Content-Type": "application/json"}
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


def snappy_literal(raw, width=None):
    """Independent literal-only fixtures from Google's public wire format."""
    declared, preamble = len(raw), bytearray()
    while declared >= 128:
        preamble.append((declared & 127) | 128)
        declared >>= 7
    preamble.append(declared)
    if len(raw) <= 60 and width is None:
        return bytes(preamble) + bytes([(len(raw)-1) << 2]) + raw
    width = width or ((len(raw)-1).bit_length()+7)//8
    return bytes(preamble) + bytes([(59+width) << 2]) + (len(raw)-1).to_bytes(width,'little') + raw


@pytest.mark.parametrize("width", [1,2,3,4])
def test_public_snappy_literal_lengths_are_bounded_and_exact(tmp_path, width):
    namespace = deployer(tmp_path)
    raw = b"public bundle bytes " * 5
    assert namespace["_scene_source_snappy"](snappy_literal(raw,width),deadline=time.monotonic()+5,cap=1024) == raw


@pytest.mark.parametrize("copy", [b"\x01\x02",b"\x0e\x02\x00",b"\x0f\x02\x00\x00\x00"])
def test_public_snappy_overlapping_copy_supports_all_offset_encodings(tmp_path, copy):
    namespace = deployer(tmp_path)
    assert namespace["_scene_source_snappy"](b"\x07\x08xab"+copy,deadline=time.monotonic()+5,cap=1024) == b"xababab"


@pytest.mark.parametrize("raw", [
    b"", b"\x80", b"\x80"*5, b"\x80\x08",  # Missing/overlarge declared lengths.
    b"\x05\x10abc", b"\x01\xf0", b"\x01\xf0\x00",  # Truncated literal/length.
    b"\x04\x01\x00", b"\x04\x01\x01",  # Zero offset/back-reference before any output.
    b"\x02\x00x\x01\x01", b"\x01\x00xy",  # Declared output overflow/trailing bytes.
    b"\x02\x00x\x02", b"\x02\x00x\x03\x01\x00",  # Truncated two/four-byte offsets.
])
def test_public_snappy_refuses_malformed_or_expanding_bytes_before_admission(tmp_path, raw):
    namespace = deployer(tmp_path)
    with pytest.raises(ValueError,match="deploy_scene_retirement_runtime_unproven"):
        namespace["_scene_source_snappy"](raw,deadline=time.monotonic()+5,cap=1023)


def test_public_snappy_respects_the_same_absolute_install_deadline(tmp_path,monkeypatch):
    namespace = deployer(tmp_path)
    raw = snappy_literal(b"prefix")+b"\x00x"*100
    ticks = iter([0.0,0.0,2.0])
    monkeypatch.setattr(time,"monotonic",lambda:next(ticks))
    with pytest.raises(ValueError,match="deploy_scene_retirement_runtime_unproven"):
        namespace["_scene_source_snappy"](raw,deadline=1.0,cap=1024)


def public_transport(value, *, change=None, copies=1, reverse=False, storage="inline"):
    signed = statement(value)
    commit = value["sources"][0]["commit"]
    signed["subject"].append({"name": "source-commit-selector.json",
                              "digest": {"sha256": hashlib.sha256(selector_bytes(commit)).hexdigest()}})
    if change == "subject":
        signed["subject"][0]["digest"] = {"sha256": "0" * 64}
    elif change == "selector":
        signed["subject"][1]["digest"] = {"sha256": "0" * 64}
    elif change == "extra-subject":
        signed["subject"].append(dict(signed["subject"][1]))
    elif change == "duplicate-manifest":
        signed["subject"][1] = dict(signed["subject"][0])
    elif change == "duplicate-selector":
        signed["subject"][0] = dict(signed["subject"][1])
    elif change == "single-subject":
        signed["subject"].pop()
    elif change == "predicate-type":
        signed["predicateType"] = "untrusted/predicate"
    bundle = {
        "mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json",
        "dsseEnvelope": {
            "payloadType": "application/vnd.in-toto+json",
            "payload": base64.b64encode(canonical(signed)).decode(),
        },
    }
    requests = []
    blob_url = "https://tmaproduction.blob.core.windows.net/attestations/123/2026/10/06/456.json.sn?sig=synthetic-fixture"

    def open_request(request, timeout):
        url = request.full_url
        assert 0 < timeout <= 30 and not request.has_header("Authorization")
        requests.append(url)
        if url == blob_url:
            response = Response(url, bundle)
            if storage == "snappy-url":
                response.headers["Content-Type"] = "application/x-snappy"
                response.body = io.BytesIO(snappy_literal(canonical(bundle)))
            return response
        assert url.startswith("https://api.github.com/repos/ognjhunt/BlueprintCapturePipeline/")
        assert "/actions/" not in url, "Actions run/artifact retention must not gate source proof"
        result = {
            "attestations": [
                ({"bundle": bundle | {"synthetic_signature": index}} if storage == "inline" else
                 {"bundle": None, "bundle_url": blob_url}) for index in (reversed(range(copies)) if reverse else range(copies))
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
    assert path.stat().st_ino == original.st_ino and len(requests) == 2
    assert not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


@pytest.mark.parametrize("change", ["selector", "subject", "extra-subject", "duplicate-manifest", "duplicate-selector", "single-subject", "predicate-type"])
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


@pytest.mark.parametrize("commit", [COMMIT, "b" * 40], ids=["fresh-install", "rollback"])
@pytest.mark.parametrize("storage", ["inline", "blob-url", "snappy-url"])
def test_public_source_lookup_survives_deleted_actions_artifacts_and_runs(tmp_path, monkeypatch, commit, storage):
    namespace = deployer(tmp_path)
    value = manifest()
    value["sources"][0]["commit"] = commit
    open_request, requests = public_transport(value, storage=storage)
    expected_url = "https://api.github.com/repos/ognjhunt/BlueprintCapturePipeline/attestations/sha256:" + hashlib.sha256(selector_bytes(commit)).hexdigest() + "?per_page=20"

    def crypto_boundary(source_commit, *, deadline, _proof_root):
        assert source_commit == commit and time.monotonic() < deadline
        raw = (_proof_root / "source-sha256-manifest.json").read_bytes()
        proof = (_proof_root / "source-provenance.sigstore.json").read_bytes()
        assert raw == canonical(value)
        return value, raw, proof  # Only the protected crypto boundary can admit this data.

    namespace["_scene_source_attestation"] = crypto_boundary
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: type("Opener", (), {"open": staticmethod(open_request)})())
    namespace["_scene_source_delivery"](commit, deadline=time.monotonic() + 10)
    assert requests[0] == expected_url and len(requests) == (1 if storage == "inline" else 2)
    assert namespace["_scene_source_selector_bytes"](commit) == selector_bytes(commit)
    assert (namespace["_SCENE_SOURCE_ATTESTATIONS"] / commit / "source-sha256-manifest.json").read_bytes() == canonical(value)
    assert not namespace["_SCENE_RUNTIME_BOOT_ROOT"].exists()


@pytest.mark.parametrize("url", [
    "http://tmaproduction.blob.core.windows.net/attestations/123/2026/10/06/456.json.sn",
    "https://other.invalid/attestations/123/2026/10/06/456.json.sn",
    "https://credential@tmaproduction.blob.core.windows.net/attestations/123/2026/10/06/456.json.sn",
    "https://tmaproduction.blob.core.windows.net:443/attestations/123/2026/10/06/456.json.sn",
    "https://tmaproduction.blob.core.windows.net/attestations/../456.json.sn",
    "https://tmaproduction.blob.core.windows.net/attestations/123/2026/10/06/456.json.sn#fragment",
    "https://[malformed/attestations/123/2026/10/06/456.json.sn",
])
def test_public_bundle_url_refuses_unsupported_origins_or_paths_before_request(tmp_path, monkeypatch, url):
    namespace = deployer(tmp_path)
    calls = []
    def request(req, timeout):
        calls.append(req.full_url)
        assert req.full_url.startswith("https://api.github.com/")
        return Response(req.full_url, {"attestations": [{"bundle": None, "bundle_url": url}]})
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: type("Opener", (), {"open": staticmethod(request)})())
    with pytest.raises(ValueError, match="deploy_scene_retirement_runtime_unproven"):
        namespace["_scene_source_delivery"](COMMIT, deadline=time.monotonic() + 10)
    assert len(calls) == 1 and not namespace["_SCENE_SOURCE_ATTESTATIONS"].exists()


def test_public_source_locator_candidate_limit_refuses_before_publication(tmp_path, monkeypatch):
    namespace = deployer(tmp_path)
    request, _ = public_transport(manifest(), copies=21)
    namespace["_scene_source_attestation"] = lambda *a, **kw: pytest.fail("response bound precedes crypto")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: type("Opener", (), {"open": staticmethod(request)})())
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
    "change", ["none", "paired", "paired-reverse", "wrong-selector", "duplicate-selector", "extra-subject", "certificate", "timestamp", "subject", "predicate", "wrong-contract", "wrong-repo", "wrong-commit"]
)
def test_direct_bootstrap_crypto_and_policy_refusal_preserves_no_installer(
    tmp_path, monkeypatch, change
):
    namespace = deployer(tmp_path)
    value = manifest()
    if change == "wrong-contract":
        value["sources"][1]["commit"] = "f" * 40
    elif change == "wrong-repo":
        value["sources"][0]["repository"] = "other/repository"
    elif change == "wrong-commit":
        value["sources"][0]["commit"] = "f" * 40
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
    if change in {"paired", "paired-reverse", "wrong-selector", "duplicate-selector", "extra-subject"}:
        verified["statement"]["subject"].append({"name": "source-commit-selector.json",
            "digest": {"sha256": hashlib.sha256(selector_bytes()).hexdigest()}})
        if change == "paired-reverse":
            verified["statement"]["subject"].reverse()
        elif change == "wrong-selector":
            verified["statement"]["subject"][1]["digest"] = {"sha256": "0" * 64}
        elif change == "duplicate-selector":
            verified["statement"]["subject"][0] = verified["statement"]["subject"][1]
        elif change == "extra-subject":
            verified["statement"]["subject"].append(verified["statement"]["subject"][1])
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
    if change in {"none", "paired", "paired-reverse"}:
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
        "_SourceDeadlineSocket",
        "_SourceDeadlineHTTPSConnection",
        "_SourceDeadlineHTTPSHandler",
        "_SourceClosingHTTPErrorProcessor",
        "_SourceNoRedirect",
        "_source_response_opener",
        "_scene_source_snappy",
        "_scene_source_selector_bytes",
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
        if isinstance(node, (ast.FunctionDef,ast.ClassDef)) and node.name in names
    }
    right = {
        node.name: ast.dump(node, include_attributes=False)
        for node in wrapper.body
        if isinstance(node, (ast.FunctionDef,ast.ClassDef)) and node.name in names
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


@pytest.mark.parametrize("consumer",["api","sdk"])
@pytest.mark.parametrize("fault",["redirect","error","status","header","body","chunk","trailer"])
def test_actual_source_application_dispatch_closes_and_bounds_all_http_framing(
    tmp_path,monkeypatch,consumer,fault
):
    """Unmodified urllib/HTTPResponse/SocketIO; only connect supplies fake sockets."""
    clock,calls,responses = [0.0],[],[]
    body = b'tiny synthetic SDK bytes' if consumer == 'sdk' else b'{"workflow_runs":[]}'
    ordinary = b'Content-Length: '+str(len(body)).encode()+b'\r\n\r\n'+body
    if fault == 'redirect':
        prefix = b'HTTP/1.1 302 Synthetic\r\nLocation: https://other.invalid/a\r\nContent-Length: 9\r\n\r\nforbidden'
        suffix = b''
    elif fault == 'error':
        prefix = b'HTTP/1.1 401 Synthetic\r\nContent-Length: 9\r\n\r\nforbidden'
        suffix = b''
    elif fault == 'status':
        prefix,suffix = b'',b'HTTP/1.1 200 OK\r\n'+ordinary
    elif fault == 'header':
        prefix,suffix = b'HTTP/1.1 200 OK\r\n',b'X-Slow: drip\r\n'+ordinary
    elif fault == 'body':
        prefix = b'HTTP/1.1 200 OK\r\nContent-Length: '+str(len(body)).encode()+b'\r\n\r\n'
        suffix = body
    elif fault == 'chunk':
        prefix = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n'
        suffix = b'1;slow=metadata\r\nx\r\n0\r\n\r\n'
    else:
        prefix = b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n'
        suffix = b'X-Slow-Trailer: drip\r\n\r\n'
    class Peer:
        def __init__(self,timeout):
            self.prefix,self.suffix = prefix,bytearray(suffix)
            self.timeouts,self.request,self.timeout = [],bytearray(),timeout
            calls.append(self)
        def settimeout(self,value):
            self.timeout = value
            self.timeouts.append(value)
        def sendall(self,value,*args):
            self.request.extend(value)
        def recv_into(self,target,*args):
            if self.prefix:
                count = len(self.prefix)
                target[:count],self.prefix = self.prefix,b''
                return count
            if self.timeout < 4:
                clock[0] += self.timeout
                raise TimeoutError('synthetic source remaining raw budget')
            clock[0] += 4
            target[0] = self.suffix.pop(0)
            return 1
        def makefile(self,*args,**kwargs):
            return io.BufferedReader(socket.SocketIO(self,'r'))
        def _decref_socketios(self):
            pass
        def close(self):
            pass
    actual = http.client.HTTPConnection.response_class
    class TracedResponse(actual):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            responses.append(self)
    monkeypatch.setattr(http.client.HTTPConnection,'response_class',TracedResponse)
    monkeypatch.setattr(http.client.HTTPSConnection,'connect',lambda connection:setattr(connection,'sock',Peer(connection.timeout)))
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    if consumer == 'api':
        namespace = deployer(tmp_path)
        def execute():
            return namespace['_scene_source_delivery'](COMMIT,deadline=10)
    else:
        module = installer()
        module._OWNER = os.getuid()
        module._FREE_FLOOR = 0
        module._sdk_root = lambda:tmp_path/'sdk'
        row = {'url':'https://files.pythonhosted.org/fixture.whl','size':len(body),
               'hash':'sha256:'+hashlib.sha256(body).hexdigest()}
        def execute():
            return module._sdk_artifact(row,None,10)
    exception = urllib.error.HTTPError if fault == 'error' else ValueError if fault == 'redirect' else TimeoutError
    with pytest.raises(exception) as held:
        execute()
    assert len(calls) == 1 and len(responses) == 1 and responses[0].closed
    if fault == 'error':
        assert held.value.fp.closed
    elif fault not in {'error','redirect'}:
        distinct = [value for index,value in enumerate(calls[0].timeouts)
                    if index == 0 or value != calls[0].timeouts[index-1]]
        assert clock[0] == 10 and distinct == [10,6,2]
    assert b'Authorization:' not in calls[0].request
    assert not (tmp_path/'runtime').exists()


@pytest.mark.parametrize("consumer",["api","sdk"])
def test_source_transports_clamp_connection_timeout_before_any_late_connect(tmp_path,monkeypatch,consumer):
    namespace = deployer(tmp_path) if consumer == 'api' else vars(installer())
    clock,calls = [95.0],[]
    monkeypatch.setattr(time,'monotonic',lambda:clock[0])
    def connect(connection):
        calls.append(connection.timeout)
        connection.sock = object()
    monkeypatch.setattr(http.client.HTTPSConnection,'connect',connect)
    connection = namespace['_SourceDeadlineHTTPSConnection']('synthetic.invalid',deadline=100,timeout=30)
    connection.connect()
    assert calls == [5] and connection.sock._maximum_timeout == 5
    clock[0] = 100
    with pytest.raises(TimeoutError,match='source_transfer_deadline'):
        namespace['_SourceDeadlineHTTPSConnection']('synthetic.invalid',deadline=100,timeout=30).connect()
    assert calls == [5]


def test_source_transports_are_embedded_identically_without_new_bandit_suppressions():
    names = {'_SourceDeadlineSocket','_SourceDeadlineHTTPSConnection','_SourceDeadlineHTTPSHandler',
             '_SourceClosingHTTPErrorProcessor','_SourceNoRedirect','_source_response_opener'}
    def definitions(raw):
        return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(raw).body
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names}
    deployed = definitions((SCRIPTS/'deploy_control_plane_commit.py').read_text())
    assert len(deployed) == len(names)
    assert definitions((SCRIPTS/'install_scene_retirement_runtime.py').read_text()) == deployed
    shell = (SCRIPTS/'install_live_pipeline_control_plane.sh').read_text()
    embedded = shell.split("<<'PY_RUNTIME'\n",1)[1].split('\nPY_RUNTIME',1)[0]
    assert definitions(embedded) == deployed
    from collections import Counter
    for name in ('install_scene_retirement_runtime.py','deploy_control_plane_commit.py','install_live_pipeline_control_plane.sh'):
        baseline = subprocess.check_output(['/usr/bin/git','show',
            '20159f08a9470b26e8c1f988f39fa076ff82fb7f:scripts/'+name],cwd=SCRIPTS.parent).decode()
        def comments(raw):
            return Counter(line[line.index('# nosec'):].strip() for line in raw.splitlines() if '# nosec' in line)
        assert not comments((SCRIPTS/name).read_text())-comments(baseline)
