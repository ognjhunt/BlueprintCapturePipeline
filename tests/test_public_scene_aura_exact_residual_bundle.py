from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_aura_exact_residual_bundle import (
    AURA_COMMIT,
    AURA_REPOSITORY,
    AURA_TREE,
    ENTRYPOINT,
    AuraExactResidualBundleError,
    build_aura_exact_residual_bundle,
)
from blueprint_pipeline.public_scene_aura_exact_residual_vast import (
    validate_aura_exact_residual_bundle,
)
from blueprint_pipeline.adp_aura_author_smoke_vast import (
    WONDERWORLD_MARIGOLD_RUNTIME_FILES,
)
from blueprint_pipeline.public_scene_aura_exact_residual_preflight import (
    materialize_aura_exact_residual_preflight,
)
from blueprint_pipeline.provider_runtime_bundle_contract import provider_runtime_contract_blockers
from blueprint_pipeline.vast_provider_adapter import _blueprint_bundle_preflight
from tests.test_public_scene_aura_exact_residual_preflight import _packet


def _git(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True)


def _init_release(tmp_path: Path, *, name: str, commit: str, tree: str) -> Path:
    """Create a tiny git tree and patch identity lookups for bundle-only tests."""

    root = tmp_path / name
    root.mkdir()
    _git(["git", "init", "-q"], root)
    _git(["git", "config", "user.email", "test@example.invalid"], root)
    _git(["git", "config", "user.name", "Bundle Test"], root)
    (root / "LICENSE").write_text("release\n", encoding="utf-8")
    if name == "aura":
        for relative in (
            "train.py",
            "remove.py",
            "inpaint.py",
            "submodules/diff-surfel-rasterization/CMakeLists.txt",
            "submodules/simple-knn/CMakeLists.txt",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# release member\n", encoding="utf-8")
    elif name == "lama":
        path = root / "bin" / "predict.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# release member\n", encoding="utf-8")
        (root / "requirements.txt").write_text("\n", encoding="utf-8")
    else:
        for relative in WONDERWORLD_MARIGOLD_RUNTIME_FILES.values():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# {relative}\n", encoding="utf-8")
    _git(["git", "add", "."], root)
    _git(["git", "commit", "-qm", "release"], root)
    return root


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _preflight(tmp_path: Path) -> Path:
    packet = _packet(tmp_path)
    value = json.loads(packet.read_text(encoding="utf-8"))
    backend_path = Path(value["backend_admission"]["path"])
    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    source_archive = tmp_path / "aura-source-admission-anchor.zip"
    with zipfile.ZipFile(source_archive, "w") as archive:
        archive.writestr("LICENSE", "Apache-2.0 fixture source anchor\n")
    backend.update(
        {
            "source_archive": {
                "path": str(source_archive),
                "size_bytes": source_archive.stat().st_size,
                "sha256": _sha256(source_archive),
            },
            "source_archive_sha256": _sha256(source_archive),
            "source_repository": AURA_REPOSITORY,
            "source_revision": AURA_COMMIT,
            "source_tree": AURA_TREE,
        }
    )
    backend["receipt_digest"] = canonical_digest(backend, digest_field="receipt_digest")
    _write(backend_path, backend)
    value["backend_admission"].update(
        {
            "size_bytes": backend_path.stat().st_size,
            "sha256": _sha256(backend_path),
            "receipt_digest": backend["receipt_digest"],
        }
    )
    value["packet_digest"] = canonical_digest(value, digest_field="packet_digest")
    _write(packet, value)
    path = tmp_path / "preflight.json"
    materialize_aura_exact_residual_preflight(input_packet_path=packet, output_path=path)
    return path


def _fake_identity(monkeypatch: pytest.MonkeyPatch, wonderworld: Path) -> None:
    import blueprint_pipeline.public_scene_aura_exact_residual_bundle as subject

    original = subject._verified_git_release

    def verified(root: str | Path, *, repository: str, commit: str, tree: str, code: str):
        # Keep the filesystem/clean-tree behavior real in the actual materializer;
        # fixture commits cannot reproduce the published commit IDs.
        path = Path(root)
        assert (path / ".git").is_dir()
        return {
            "repository": repository,
            "commit": commit,
            "tree": tree,
            "tracked_files_clean": True,
        }

    monkeypatch.setattr(subject, "_verified_git_release", verified)
    monkeypatch.setattr(
        subject,
        "WONDERWORLD_MARIGOLD_LICENSE_SHA256",
        _sha256(wonderworld / "marigold_module" / "LICENSE.txt"),
    )
    assert original is not None


def test_seals_shared_camera_bundle_and_rehearses_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _preflight(tmp_path)
    aura = _init_release(tmp_path, name="aura", commit="unused", tree="unused")
    lama = _init_release(tmp_path, name="lama", commit="unused", tree="unused")
    wonderworld = _init_release(tmp_path, name="wonderworld", commit="unused", tree="unused")
    repo = Path(__file__).resolve().parents[1]
    _fake_identity(monkeypatch, wonderworld)

    receipt = build_aura_exact_residual_bundle(
        preflight_path=preflight,
        aura_source_directory=aura,
        lama_source_directory=lama,
        wonderworld_source_directory=wonderworld,
        output_root=tmp_path / "bundle",
        repo_root=repo,
    )

    assert receipt["replacement_object_count"] == 2
    assert receipt["shared_camera_count"] == 2
    assert receipt["task_count"] == 2
    assert receipt["private_derived_upload_only"] is True
    assert receipt["raw_interiorgs_bytes_included"] is False
    assert receipt["stock_inpaint360gs_code_or_author_data_included"] is False
    rehearsal = receipt["exact_bundle_entrypoint_rehearsal"]
    assert rehearsal["status"] == "passed"
    assert rehearsal["provider_mutations_performed"] == 0
    assert rehearsal["gpu_runtime_started"] is False
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        request = json.loads(archive.read("provider_runtime/aura_exact_residual_runtime_request.json"))
        assert request["commands"]["excluded_stock_entrypoints"] == [
            "utils/sam2_utils.py",
            "utils/LeftRefill/sdedit_utils.py",
        ]
        assert all(
            row["calibration"]["intrinsics"]["model"] == "PINHOLE"
            for row in request["camera_inputs"]
        )
        assert all(
            plan["task_scene"].startswith("../data/Other-360/")
            for plan in request["task_plans"]
        )
        train_config = archive.read("provider_runtime/configs/train.config").decode("utf-8")
        assert '"../data/Other-360/shared_retained_scene"' in train_config
        assert '"../work/model"' in train_config
        for plan in request["task_plans"]:
            inpaint_config = archive.read(
                f"provider_runtime/configs/inpaint_{plan['task_id']}.config"
            ).decode("utf-8")
            assert f'"{plan["task_scene"]}"' in inpaint_config
            assert 'skip_train = false' in inpaint_config
        assert request["commands"]["task_initialization_checkpoint"] == (
            "one_shared_removal_point_cloud_digest_verified_before_and_after_each_task"
        )
        assert request["wonderworld_marigold_runtime"]["license"] == "Apache-2.0"
        assert len(request["wonderworld_marigold_runtime"]["files"]) == 4
        assert [model["repository"] for model in request["marigold_runtime_models"]] == [
            "prs-eth/marigold-depth-v1-0",
            "prs-eth/marigold-v1-0",
        ]
        assert all(
            f"provider_runtime/runtime_dependencies/{relative}" in archive.namelist()
            for relative in WONDERWORLD_MARIGOLD_RUNTIME_FILES
        )
        assert ENTRYPOINT in archive.namelist()


def test_rejects_a_source_release_tree_with_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _preflight(tmp_path)
    aura = _init_release(tmp_path, name="aura", commit="unused", tree="unused")
    lama = _init_release(tmp_path, name="lama", commit="unused", tree="unused")
    wonderworld = _init_release(tmp_path, name="wonderworld", commit="unused", tree="unused")
    (aura / "escape").symlink_to("/etc/passwd")
    _git(["git", "add", "escape"], aura)
    _git(["git", "commit", "-qm", "tracked symlink"], aura)
    _fake_identity(monkeypatch, wonderworld)

    with pytest.raises(AuraExactResidualBundleError, match="release_symlink_forbidden"):
        build_aura_exact_residual_bundle(
            preflight_path=preflight,
            aura_source_directory=aura,
            lama_source_directory=lama,
            wonderworld_source_directory=wonderworld,
            output_root=tmp_path / "bundle",
            repo_root=Path(__file__).resolve().parents[1],
        )


def test_excludes_untracked_release_tree_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _preflight(tmp_path)
    aura = _init_release(tmp_path, name="aura", commit="unused", tree="unused")
    lama = _init_release(tmp_path, name="lama", commit="unused", tree="unused")
    wonderworld = _init_release(tmp_path, name="wonderworld", commit="unused", tree="unused")
    (aura / "untracked-secret-like-scratch.txt").write_text("not a release input\n")
    _fake_identity(monkeypatch, wonderworld)

    receipt = build_aura_exact_residual_bundle(
        preflight_path=preflight,
        aura_source_directory=aura,
        lama_source_directory=lama,
        wonderworld_source_directory=wonderworld,
        output_root=tmp_path / "bundle",
        repo_root=Path(__file__).resolve().parents[1],
    )

    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        assert (
            "provider_runtime/AuraFusion360_official/untracked-secret-like-scratch.txt"
            not in archive.namelist()
        )


def test_exact_residual_runtime_has_a_distinct_fail_closed_provider_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    entrypoint = (root / "scripts/run_public_scene_aura_exact_residual.sh").read_text()
    runner = (root / "scripts/public_scene_aura_exact_residual_runner.py").read_text()
    assert provider_runtime_contract_blockers(
        provider_bundle_kind="adp_aura_exact_residual",
        entrypoint_text=entrypoint,
        runner_text=runner,
    ) == []
    assert provider_runtime_contract_blockers(
        provider_bundle_kind="adp_inpaint360_interiorgs",
        entrypoint_text=entrypoint,
        runner_text=runner,
    )


def test_vast_preflight_accepts_only_the_exact_residual_bundle_layout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "exact-residual.zip"
    entries = {
        "provider_runtime/run_public_scene_aura_exact_residual.sh": (
            root / "scripts/run_public_scene_aura_exact_residual.sh"
        ).read_bytes(),
        "provider_runtime/public_scene_aura_exact_residual_runner.py": (
            root / "scripts/public_scene_aura_exact_residual_runner.py"
        ).read_bytes(),
        "provider_runtime/aura_exact_residual_bundle_manifest.json": b"{}",
        "provider_runtime/aura_exact_residual_runtime_request.json": b"{}",
        "provider_runtime/input/preflight.json": b"{}",
        "provider_runtime/input/backend_admission.json": b"{}",
        "provider_runtime/input/shared_retained_scene.ply": b"ply\n",
        "provider_runtime/big-lama.zip": b"x",
    }
    with zipfile.ZipFile(bundle, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    result = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="2026-08-12T00:00:00Z",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_aura_exact_residual",
        bundle_path=bundle,
        provider_bundle_url="https://example.invalid/bundle.zip",
        provider_output_put_url="https://example.invalid/out.zip",
    )
    assert result["status"] == "passed"
    assert result["blockers"] == []


def test_vast_bundle_validator_reads_the_packet_not_digest_shaped_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = _preflight(tmp_path)
    aura = _init_release(tmp_path, name="aura", commit="unused", tree="unused")
    lama = _init_release(tmp_path, name="lama", commit="unused", tree="unused")
    wonderworld = _init_release(tmp_path, name="wonderworld", commit="unused", tree="unused")
    _fake_identity(monkeypatch, wonderworld)
    receipt = build_aura_exact_residual_bundle(
        preflight_path=preflight,
        aura_source_directory=aura,
        lama_source_directory=lama,
        wonderworld_source_directory=wonderworld,
        output_root=tmp_path / "bundle",
        repo_root=Path(__file__).resolve().parents[1],
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_aura_exact_residual_vast._authority",
        lambda *_args, **_kwargs: (
            {"authority_digest": "sha256:" + "a" * 64, "paid_compute": {"external_instance_allowlist": [47373597]}},
            tmp_path / "authority.json",
        ),
    )
    validated = validate_aura_exact_residual_bundle(
        tmp_path / "bundle" / "aura_exact_residual_bundle_receipt.json"
    )
    assert validated["bundle_sha256"] == receipt["bundle_sha256"]
    assert validated["preflight_digest"] == receipt["preflight_digest"]
    assert validated["replacement_object_count"] == 2
    receipt_path = tmp_path / "bundle" / "aura_exact_residual_bundle_receipt.json"
    value = json.loads(receipt_path.read_text())
    request_path = Path(value["runtime_request"]["path"])
    request_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime_request"):
        validate_aura_exact_residual_bundle(tmp_path / "bundle" / "aura_exact_residual_bundle_receipt.json")
