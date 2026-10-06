"""Synthetic source/crypto-boundary contracts; no signing or live deployment."""

from __future__ import annotations

import copy
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from scripts import release_source_manifest as source


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def _repo(root: Path, files: dict[str, bytes]) -> str:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "fixture@example.invalid")
    _git(root, "config", "user.name", "Source fixture")
    for path, content in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "scripts/install_scene_retirement_runtime.py").chmod(0o755) if "scripts/install_scene_retirement_runtime.py" in files else None
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "synthetic source")
    return _git(root, "rev-parse", "HEAD").decode().strip()


@pytest.fixture
def sources(tmp_path, monkeypatch):
    pipeline = tmp_path / "pipeline"
    contracts = tmp_path / "contracts"
    files = {
        "src/blueprint_pipeline/__init__.py": b"raw\x00pipeline\n",
        "scripts/install_scene_retirement_runtime.py": b"# installer\n",
        "scripts/deploy_control_plane_commit.py": b"# deployer\n",
        "scripts/release_source_manifest.py": b"# verifier\n",
        "deploy/systemd/control.service": b"[Service]\n",
        "uv.lock": b"version = 1\n",
        "pyproject.toml": b"[project]\nname='fixture'\n",
    }
    commit = _repo(pipeline, files)
    contracts_commit = _repo(contracts, {"src/blueprint_contracts/__init__.py": b"contracts\n"})
    monkeypatch.setattr(source, "CONTRACTS_COMMIT", contracts_commit)
    return pipeline, contracts, commit, files


def _manifest(sources):
    pipeline, contracts, commit, _ = sources
    return source.build_manifest(pipeline, contracts, commit)


def test_manifest_binds_complete_committed_raw_bytes_path_mode_and_both_sources(sources):
    pipeline, _, commit, files = sources
    raw = _manifest(sources)
    manifest = source.validate_manifest_bytes(raw, commit)
    rows = source.source_inventory(manifest, source.PIPELINE_REPOSITORY, commit)
    assert set(rows) == set(files)
    for path, content in files.items():
        assert rows[path]["sha256"] == hashlib.sha256(content).hexdigest()
        assert rows[path]["size"] == len(content)
    assert rows["scripts/install_scene_retirement_runtime.py"]["mode"] == "100755"
    assert manifest["sources"][0]["tree"] == _git(pipeline, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    assert manifest["sources"][1]["commit"] == source.CONTRACTS_COMMIT
    # Mutable working files and staging cannot change the selected source bytes.
    (pipeline / "scripts/install_scene_retirement_runtime.py").write_bytes(b"uncommitted replacement")
    _git(pipeline, "add", ".")
    assert _manifest(sources) == raw


def test_builder_batches_objects_instead_of_spawning_per_leaf(sources, monkeypatch):
    pipeline, contracts, _, _ = sources
    for number in range(50):
        (pipeline / f"scripts/fixture_{number:02}.py").write_bytes(str(number).encode())
    _git(pipeline, "add", ".")
    _git(pipeline, "commit", "-qm", "batch inventory")
    commit = _git(pipeline, "rev-parse", "HEAD").decode().strip()
    calls = []
    original = source._run_bounded

    def record(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(source, "_run_bounded", record)
    source.build_manifest(pipeline, contracts, commit)
    assert sum(command[-1] == "--batch" for command in calls) == math.ceil(57 / 16) + 1
    assert len(calls) == 4 + math.ceil(57 / 16) + 1
    assert all("--no-replace-objects" in command for command in calls)


def test_builder_refuses_symlink_under_privileged_root(sources):
    pipeline, contracts, _, _ = sources
    os.symlink("install_scene_retirement_runtime.py", pipeline / "scripts/unsafe.py")
    _git(pipeline, "add", ".")
    _git(pipeline, "commit", "-qm", "unsafe source fixture")
    commit = _git(pipeline, "rev-parse", "HEAD").decode().strip()
    with pytest.raises(source.ManifestError, match="leaf refused"):
        source.build_manifest(pipeline, contracts, commit)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(schema_version="untrusted.v1"),
    lambda value: value["sources"][0].update(repository="other/repo"),
    lambda value: value["sources"][0].update(commit="a" * 40),
    lambda value: value["sources"][1].update(commit="b" * 40),
    lambda value: value["sources"][0].update(roots=["scripts"]),
    lambda value: value["sources"][0]["files"].reverse(),
    lambda value: value["sources"][0]["files"].append(value["sources"][0]["files"][-1]),
    lambda value: value["sources"][0]["files"][0].update(path="../escape"),
    lambda value: value["sources"][0]["files"][0].update(mode="120000"),
    lambda value: value["sources"][0]["files"][0].update(mode={"unsafe": "100644"}),
    lambda value: value["sources"][0]["files"][0].update(sha256="a" * 40),
    lambda value: value["sources"][0]["files"][0].update(size=True),
    lambda value: value["sources"][0]["files"][0].update(size=source.MAX_TOTAL_BYTES + 1),
    lambda value: value["sources"][0]["files"].pop(0),
])
def test_schema_fails_closed_on_unbound_or_unsafe_inventory(sources, mutation):
    value = json.loads(_manifest(sources))
    mutation(value)
    with pytest.raises(source.ManifestError):
        source.validate_manifest_bytes(source.canonical_manifest_bytes(value), sources[2])


def test_noncanonical_duplicate_truncated_and_overbudget_json_refused(sources, monkeypatch):
    raw = _manifest(sources)
    for unsafe in (json.dumps(json.loads(raw), indent=2).encode(), raw[:-5],
                   raw.replace(b'"schema_version":', b'"schema_version":"duplicate","schema_version":')):
        with pytest.raises(source.ManifestError):
            source.validate_manifest_bytes(unsafe, sources[2])
    monkeypatch.setattr(source, "MAX_MANIFEST_BYTES", len(raw) - 1)
    with pytest.raises(source.ManifestError, match="size refused"):
        source.validate_manifest_bytes(raw, sources[2])


@pytest.mark.parametrize("budget", ["files", "bytes"])
def test_source_budget_is_combined_across_pipeline_and_contracts(sources, monkeypatch, budget):
    raw = _manifest(sources)
    value = json.loads(raw)
    if budget == "files":
        monkeypatch.setattr(source, "MAX_FILES", len(value["sources"][0]["files"]))
    else:
        total = sum(row["size"] for inventory in value["sources"] for row in inventory["files"])
        monkeypatch.setattr(source, "MAX_TOTAL_BYTES", total - 1)
    with pytest.raises(source.ManifestError, match="inventory budget"):
        source.validate_manifest_bytes(raw, sources[2])
    with pytest.raises(source.ManifestError, match="combined"):
        _manifest(sources)


def _verified(raw):
    return [{"verificationResult": {
        "signature": {"certificate": {"issuer": "fixture verified by fake crypto boundary"}},
        "verifiedTimestamps": [{"type": "transparency-log", "timestamp": "2026-10-05T00:00:00Z"}],
        "statement": {"_type": "https://in-toto.io/Statement/v1",
                      "predicateType": source.PREDICATE_TYPE,
                      "predicate": json.loads(raw),
                      "subject": [{"name": "source-sha256-manifest.json", "digest": {"sha256": hashlib.sha256(raw).hexdigest()}}]},
    }}]


@pytest.fixture
def evidence(sources, tmp_path):
    raw = _manifest(sources)
    manifest = tmp_path / "manifest.json"
    bundle = tmp_path / "bundle.json"
    gh = tmp_path / "trusted-gh"
    manifest.write_bytes(raw)
    bundle.write_bytes(b'{"synthetic":"crypto is mocked, this is not a signed proof"}\n')
    gh.write_bytes(b"synthetic executable fixture")
    gh.chmod(0o755)
    return raw, manifest, bundle, gh


def _verify(sources, evidence):
    _, manifest, bundle, gh = evidence
    return source.verify_manifest_attestation(manifest, bundle, sources[2], gh_executable=gh, deadline=time.monotonic() + 5)


def test_verifier_uses_all_strict_crypto_flags_and_protected_snapshots(sources, evidence, monkeypatch):
    raw, manifest, bundle, gh = evidence

    def verify(command, **kwargs):
        assert command[:3] == [str(gh), "attestation", "verify"]
        # This standalone boolean flag prevents pair-wise interpretation.
        for flag, expected in {
            "--repo": source.PIPELINE_REPOSITORY,
            "--cert-identity": f"https://github.com/{source.PIPELINE_REPOSITORY}/.github/workflows/ci.yml@refs/heads/main",
            "--cert-oidc-issuer": "https://token.actions.githubusercontent.com",
            "--source-ref": "refs/heads/main",
            "--source-digest": sources[2], "--signer-digest": sources[2],
            "--digest-alg": "sha256", "--predicate-type": source.PREDICATE_TYPE,
            "--format": "json", "--hostname": "github.com",
        }.items():
            assert command[command.index(flag) + 1] == expected
        assert "--deny-self-hosted-runners" in command
        assert "--signer-workflow" not in command  # Mutually exclusive with exact SAN.
        snapshot = Path(command[3])
        proof = Path(command[command.index("--bundle") + 1])
        assert snapshot != manifest and proof != bundle
        assert snapshot.read_bytes() == raw
        assert snapshot.parent.stat().st_mode & 0o077 == 0
        assert snapshot.stat().st_mode & 0o222 == 0
        return json.dumps(_verified(raw)).encode()

    monkeypatch.setattr(source, "_run_bounded", verify)
    assert _verify(sources, evidence)["sources"][0]["commit"] == sources[2]


@pytest.mark.parametrize("mutation", [
    lambda value: value.clear(),
    lambda value: value.append(copy.deepcopy(value[0])),
    lambda value: value[0].pop("verificationResult"),
    lambda value: value[0]["verificationResult"].pop("signature"),
    lambda value: value[0]["verificationResult"].update(verifiedTimestamps=[]),
    lambda value: value[0]["verificationResult"]["statement"].update(predicateType="unsigned/custom"),
    lambda value: value[0]["verificationResult"]["statement"]["subject"][0].update(digest={"sha256": "f" * 64}),
    lambda value: value[0]["verificationResult"]["statement"].update(subject=[]),
    lambda value: value[0]["verificationResult"]["statement"]["predicate"].update(schema_version="tampered"),
])
def test_verifier_refuses_missing_truncated_wrong_or_ambiguous_verified_proof(sources, evidence, monkeypatch, mutation):
    result = _verified(evidence[0])
    mutation(result)
    monkeypatch.setattr(source, "_run_bounded", lambda *args, **kwargs: json.dumps(result).encode())
    with pytest.raises(source.ManifestError):
        _verify(sources, evidence)


@pytest.mark.parametrize("failure", ["wrong repository", "wrong workflow", "wrong ref", "wrong release source",
                                      "wrong issuer", "self-hosted runner", "expired certificate without valid trusted signing timestamp",
                                      "tampered signature", "truncated bundle"])
def test_crypto_failure_never_becomes_source_authority(sources, evidence, monkeypatch, failure):
    # Signature policy is implemented by trusted gh, not caller-created JSON.
    def refuse(*args, **kwargs):
        raise source.ManifestError(f"cryptographic verification refused: {failure}")
    monkeypatch.setattr(source, "_run_bounded", refuse)
    with pytest.raises(source.ManifestError, match="cryptographic"):
        _verify(sources, evidence)


def test_verifier_refuses_manifest_or_bundle_replacement_during_crypto(sources, evidence, monkeypatch):
    raw, manifest, _, _ = evidence
    def replace(*args, **kwargs):
        original = manifest.read_bytes()
        manifest.unlink()
        manifest.write_bytes(original)
        return json.dumps(_verified(raw)).encode()
    monkeypatch.setattr(source, "_run_bounded", replace)
    with pytest.raises(source.ManifestError, match="changed during verification"):
        _verify(sources, evidence)


def test_verifier_refuses_missing_tool_symlinks_and_unparseable_output(sources, evidence, monkeypatch, tmp_path):
    raw, manifest, bundle, gh = evidence
    gh.unlink()
    with pytest.raises(source.ManifestError, match="unavailable"):
        _verify(sources, evidence)
    gh.write_bytes(b"fixture")
    gh.chmod(0o755)
    real_manifest = tmp_path / "real-manifest"
    manifest.rename(real_manifest)
    manifest.symlink_to(real_manifest)
    with pytest.raises(source.ManifestError, match="unsafe"):
        _verify(sources, evidence)
    manifest.unlink()
    manifest.write_bytes(raw)
    monkeypatch.setattr(source, "_run_bounded", lambda *args, **kwargs: b'[{"verificationResult":')
    with pytest.raises(source.ManifestError, match="truncated"):
        _verify(sources, evidence)


def test_command_bounds_output_and_absolute_deadline():
    with pytest.raises(source.ManifestError, match="output budget"):
        source._run_bounded([sys.executable, "-c", "print('x' * 100000)"], deadline=time.monotonic() + 5, stdout_cap=100)
    with pytest.raises(source.ManifestError, match="deadline"):
        source._run_bounded([sys.executable, "-c", "import time; time.sleep(5)"], deadline=time.monotonic() + 0.05, stdout_cap=100)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), 0.0])
def test_expired_or_nonfinite_budget_refuses_before_source_acquisition(sources, deadline):
    with pytest.raises(source.ManifestError, match="deadline"):
        source.build_manifest(sources[0], sources[1], sources[2], deadline=deadline)


def test_ci_attests_source_after_unchanged_security_and_license_gates():
    workflow = yaml.safe_load((Path(__file__).parents[1] / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["supply-chain"]
    assert {"impacted-gate", "lint", "typecheck", "sast", "source-governance", "dependency-security", "container-contract"} <= set(job["needs"])
    assert job["permissions"] == {"contents": "read", "id-token": "write", "attestations": "write", "artifact-metadata": "write"}
    steps = job["steps"]
    source_attest = next(step for step in steps if step.get("id") == "attest-source")
    assert source_attest["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    assert source_attest["uses"] == "actions/attest@a1948c3f048ba23858d222213b7c278aabede763"
    assert source_attest["with"]["subject-path"].endswith("/source-sha256-manifest.json")
    assert source_attest["with"]["predicate-path"] == source_attest["with"]["subject-path"]
    assert source_attest["with"]["predicate-type"] == source.PREDICATE_TYPE
    license_step = next(step for step in steps if "build_supply_chain_evidence.py" in step.get("run", ""))
    assert steps.index(license_step) < steps.index(source_attest)
    checkout = next(step for step in steps if step.get("with", {}).get("repository") == source.CONTRACTS_REPOSITORY)
    assert checkout["with"]["ref"] == "7708a4e4c5dedeeb39cc73d3f6869304de295b81"
    assert checkout["with"]["persist-credentials"] is False
    assert "source-provenance.sigstore.json" in next(step["run"] for step in steps if step["name"] == "Preserve keyless attestation bundles")
    locator = next(step for step in steps if step["name"] == "Publish public source attestation digest locator")
    assert locator["with"]["name"] == "source-sha256-${{ github.sha }}-${{ steps.source-manifest.outputs.sha256 }}"
    assert "steps.attest-source.outcome == 'success'" in locator["if"]


def _candidate_bundle(raw):
    statement = _verified(raw)[0]["verificationResult"]["statement"]
    return {"dsseEnvelope": {"payloadType": "application/vnd.in-toto+json",
                             "payload": base64.b64encode(json.dumps(statement).encode()).decode(),
                             "signatures": [{"sig": "not a signature; candidate data only"}]}}


def test_public_bundle_embeds_exact_candidate_but_extraction_does_not_authenticate(sources, evidence, monkeypatch):
    raw = evidence[0]
    bundle = json.dumps(_candidate_bundle(raw)).encode()
    extracted = source.extract_untrusted_manifest(bundle, sources[2], hashlib.sha256(raw).hexdigest())
    assert extracted == raw
    evidence[2].write_bytes(bundle)
    def refuse(*args, **kwargs):
        raise source.ManifestError("unsigned candidate refused by trusted crypto")
    monkeypatch.setattr(source, "_run_bounded", refuse)
    with pytest.raises(source.ManifestError, match="unsigned"):
        _verify(sources, evidence)


@pytest.mark.parametrize("fault", ["digest", "release", "payload", "type", "predicate", "truncated"])
def test_public_candidate_extraction_refuses_wrong_binding_or_malformed_payload(sources, evidence, fault):
    raw = evidence[0]
    bundle = _candidate_bundle(raw)
    digest = hashlib.sha256(raw).hexdigest()
    commit = sources[2]
    if fault == "digest":
        digest = "a" * 64
    elif fault == "release":
        commit = "b" * 40
    elif fault == "payload":
        bundle["dsseEnvelope"]["payload"] = "%%%"
    elif fault == "type":
        bundle["dsseEnvelope"]["payloadType"] = "text/plain"
    elif fault == "predicate":
        statement = _verified(raw)[0]["verificationResult"]["statement"]
        statement["predicate"]["sources"][0]["files"][0]["sha256"] = "f" * 64
        bundle["dsseEnvelope"]["payload"] = base64.b64encode(json.dumps(statement).encode()).decode()
    encoded = json.dumps(bundle).encode()
    if fault == "truncated":
        encoded = encoded[:-10]
    with pytest.raises(source.ManifestError):
        source.extract_untrusted_manifest(encoded, commit, digest)
