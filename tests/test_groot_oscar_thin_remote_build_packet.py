from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

from blueprint_pipeline.groot_oscar_thin_remote_build_packet import (
    REQUIRED_IMAGE_FILES,
    REQUIRED_ROOT_FILES,
    _versioned_ref_blockers,
    prepare_remote_build_packet,
)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    image_root = root / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop"
    image_root.mkdir(parents=True)
    for filename in REQUIRED_ROOT_FILES:
        (root / filename).write_text(filename + "\n", encoding="utf-8")
    for filename in REQUIRED_IMAGE_FILES:
        (image_root / filename).write_text(filename + "\n", encoding="utf-8")
    package = root / "src/blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "thin_release_image_contract.py").write_text(
        "def build_thin_release_contract(*a, **k): return {'status':'passed'}\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Tests"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_packet_binds_minimal_context_and_exact_build_flow(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/blueprint/foundation:20260714",
        release_ref="registry.example/blueprint/release:20260714",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
        generated_at="2026-07-14T00:00:00+00:00",
    )
    assert result["status"] == "ready"
    assert result["provider_launch_performed_by_packet"] is False
    assert result["tarball_sha256"] == hashlib.sha256(
        Path(result["tarball_path"]).read_bytes()
    ).hexdigest()
    assert result["supported_execution_planes"] == {
        "native_linux_amd64_docker_builder": True,
        "runpod_pod": False,
    }
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert script.count("docker buildx build") == 2
    assert "FOUNDATION_IMAGE=$foundation_exact" in script
    assert 'FOUNDATION_MODEL_ASSETS=external' in script
    assert "thin_release_image_contract" in script
    assert 'release.get("required_cuda_version")' in script
    assert '"required_cuda_version":"12.6"' not in script
    subprocess.run(
        ["bash", "-n", result["run_script_path"]],
        check=True,
        capture_output=True,
        text=True,
    )
    python_heredocs = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    assert len(python_heredocs) == 3
    for index, source in enumerate(python_heredocs):
        compile(source, f"generated_remote_build_heredoc_{index}.py", "exec")
    assert '+"\\n",encoding="utf-8")' in script
    assert "--push" in script
    foundation_build_at = script.index('-t "$foundation_candidate_ref" --push')
    release_build_at = script.index('-t "$release_candidate_ref" --push')
    validation_at = script.index("validate-thin-release")
    release_promotion_at = script.index(
        'docker buildx imagetools create --tag "$release_ref" "$release_exact"'
    )
    foundation_promotion_at = script.index(
        'docker buildx imagetools create --tag "$foundation_ref" "$foundation_exact"'
    )
    contract_at = script.index(
        'PYTHONPATH="$context_dir/src" python3 - "$script_dir" "$validation_result"'
    )
    result_at = script.index('mv "$validation_result" "$result"')
    assert (
        foundation_build_at
        < release_build_at
        < validation_at
        < contract_at
        < release_promotion_at
        < foundation_promotion_at
        < result_at
    )
    assert '-t "$release_ref" --push' not in script
    assert '-t "$foundation_ref" --push' not in script
    assert '[[ "$promoted_release_digest" == "$release_digest" ]]' in script
    assert '[[ "$promoted_foundation_digest" == "$foundation_digest" ]]' in script
    assert "hf_token" not in script
    assert "snapshot_download" not in script
    assert script.count("--attest type=sbom --attest type=provenance,mode=max") == 2
    assert '"registry:$release_exact"' in script
    assert "release_sbom.spdx.json" in script
    assert "release_provenance.json" in script
    assert "release_supply_chain_manifest.json" in script
    assert "release_supply_chain_disk_admission.json" in script
    assert "syft_1.44.0_linux_amd64.tar.gz" in script
    assert "0e91737aee2b5baf1d255b959630194a302335d848ff97bb07921eb6205b5f5a" in script
    assert "serverless_worker_contract" in script
    assert "runpod_sdk_exactly_pinned" in script
    packet = Path(result["packet_dir"])
    context_manifest = json.loads(
        (packet / "context_manifest.json").read_text(encoding="utf-8")
    )
    paths = {row["path"] for row in context_manifest["files"]}
    assert "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile" in paths
    assert (
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "apt_transport_hardening.conf"
        in paths
    )
    assert (
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "oscar_cpu_import_probe.py"
        in paths
    )
    assert "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile" in paths
    assert (
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_oscar_foundation.lock"
        in paths
    )
    assert (
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_runpod_serverless.lock"
        in paths
    )
    assert (
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_runpod_serverless_sdk.lock"
        in paths
    )
    assert "src/blueprint_pipeline/thin_release_image_contract.py" in paths
    with tarfile.open(result["tarball_path"], "r:gz") as archive:
        names = set(archive.getnames())
    assert (
        "groot_oscar_thin_remote_build/remote_build_groot_oscar_thin_images.sh"
        in names
    )


def test_release_removes_build_only_serverless_environment_files() -> None:
    release_dockerfile = (
        Path(__file__).resolve().parents[1]
        / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile"
    ).read_text(encoding="utf-8")

    assert "site-packages/pip*" in release_dockerfile
    assert "-name __pycache__" in release_dockerfile


def test_packet_can_reuse_exact_foundation_and_build_only_thin_release(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    foundation = "registry.example/blueprint/foundation@sha256:" + "f" * 64
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref=foundation,
        release_ref="registry.example/blueprint/release:20260715-serverless",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
        reuse_foundation_exact=True,
    )

    assert result["status"] == "ready"
    assert result["reuse_foundation_exact"] is True
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert script.count("docker buildx build") == 1
    assert 'foundation_exact="$foundation_ref"' in script
    assert "Foundation.Dockerfile" not in script
    assert "reused foundation digest changed" in script
    assert 'imagetools create --tag "$foundation_ref"' not in script
    subprocess.run(["bash", "-n", result["run_script_path"]], check=True)


def test_packet_can_bind_an_exact_embedded_model_foundation(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    foundation = "registry.example/blueprint/foundation@sha256:" + "e" * 64
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref=foundation,
        release_ref="registry.example/blueprint/release:embedded-qualification",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
        reuse_foundation_exact=True,
        foundation_model_assets="embedded",
    )

    assert result["status"] == "ready"
    assert result["foundation_model_assets"] == "embedded"
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert 'FOUNDATION_MODEL_ASSETS=embedded' in script
    assert 'foundation_model_assets=\'embedded\'' in script
    assert '"models_embedded":contract.get("models_embedded_in_foundation") is True' in script


def test_packet_refuses_embedded_assets_on_a_new_unverified_foundation(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/blueprint/foundation:versioned",
        release_ref="registry.example/blueprint/release:embedded-qualification",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
        foundation_model_assets="embedded",
    )
    assert result["status"] == "blocked"
    assert "embedded_foundation_assets_require_exact_reuse" in result["blockers"]


def test_packet_build_outputs_require_promotable_tags() -> None:
    digest_ref = "registry.example/blueprint/release@sha256:" + "a" * 64
    assert _versioned_ref_blockers(digest_ref, "release") == [
        "release_image_ref_must_use_tag"
    ]


def test_packet_refuses_dirty_source_and_unstable_refs(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/foundation:latest",
        release_ref="registry.example/release:dev",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=True,
    )
    assert result["status"] == "blocked"
    assert "remote_release_packet_requires_clean_source_worktree" in result["blockers"]
    assert "foundation_image_ref_refuses_unstable_tag" in result["blockers"]
    assert "release_image_ref_refuses_unstable_tag" in result["blockers"]
    assert Path(result["tarball_path"]).is_file()


def test_packet_rejects_and_shell_quotes_malicious_image_refs(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    malicious = 'registry.example/foundation:$(touch /tmp/owned)'
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref=malicious,
        release_ref='registry.example/release:v1";touch /tmp/owned;#',
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
    )
    assert result["status"] == "blocked"
    assert "foundation_image_ref_invalid" in result["blockers"]
    assert "release_image_ref_invalid" in result["blockers"]
    script = Path(result["run_script_path"]).read_text(encoding="utf-8")
    assert "foundation_ref='registry.example/foundation:$(touch /tmp/owned)'" in script
    subprocess.run(["bash", "-n", result["run_script_path"]], check=True)


def test_packet_refuses_commit_that_only_matches_head_prefix(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    head = _head(root)
    mistyped = head[:8] + ("0" if head[8] != "0" else "1") + head[9:]
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/foundation:versioned",
        release_ref="registry.example/release:versioned",
        source_commit=mistyped,
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
    )
    assert result["status"] == "blocked"
    assert "source_commit_not_exact_head" in result["blockers"]


def test_packet_reports_missing_required_context(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "README.md").unlink()
    result = prepare_remote_build_packet(
        output_dir=tmp_path / "out",
        repo_root=root,
        foundation_ref="registry.example/foundation:versioned",
        release_ref="registry.example/release:versioned",
        source_commit=_head(root),
        source_patch_sha256=hashlib.sha256(b"").hexdigest(),
        source_worktree_dirty=False,
    )
    assert result["status"] == "blocked"
    assert "remote_context_source_missing:README.md" in result["blockers"]
