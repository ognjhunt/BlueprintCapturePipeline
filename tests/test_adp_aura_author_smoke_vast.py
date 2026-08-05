from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import adp_aura_author_smoke_vast as aura
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    real_root = Path(__file__).resolve().parents[1]
    for name in (
        "run_adp_aura_author_smoke_provider_runtime.sh",
        "adp_aura_author_smoke_provider_runner.py",
    ):
        (scripts / name).write_bytes((real_root / "scripts" / name).read_bytes())
    source = tmp_path / "AuraFusion360_official"
    sam2 = tmp_path / "sam2"
    (source / "submodules/diff-surfel-rasterization/third_party/glm").mkdir(parents=True)
    (source / "submodules/simple-knn").mkdir(parents=True)
    (source / "README.md").write_text("author source", encoding="utf-8")
    (source / "submodules/diff-surfel-rasterization/LICENSE.md").write_text(
        "license", encoding="utf-8"
    )
    (source / "submodules/diff-surfel-rasterization/third_party/glm/copying.txt").write_text(
        "license", encoding="utf-8"
    )
    (source / "submodules/simple-knn/LICENSE.md").write_text("license", encoding="utf-8")
    sam2.mkdir()
    (sam2 / "LICENSE").write_text("sam2 license", encoding="utf-8")

    def fake_git(path: Path, *args: str) -> str:
        if args == ("status", "--porcelain"):
            return ""
        if args == ("rev-parse", "HEAD^{tree}"):
            if path == sam2:
                return aura.SAM2_SOURCE_TREE
            return aura.SOURCE_TREE
        if args == ("rev-parse", "HEAD"):
            if path == sam2:
                return aura.SAM2_SOURCE_COMMIT
            for relative, revision in aura.SUBMODULES.items():
                if path == source / relative:
                    return revision
            return aura.SOURCE_COMMIT
        raise AssertionError((path, args))

    monkeypatch.setattr(aura, "_git", fake_git)
    monkeypatch.setattr(
        aura,
        "_source_files",
        lambda _source: [
            ("README.md", source / "README.md"),
            (
                "submodules/diff-surfel-rasterization/LICENSE.md",
                source / "submodules/diff-surfel-rasterization/LICENSE.md",
            ),
            (
                "submodules/diff-surfel-rasterization/third_party/glm/copying.txt",
                source / "submodules/diff-surfel-rasterization/third_party/glm/copying.txt",
            ),
            ("submodules/simple-knn/LICENSE.md", source / "submodules/simple-knn/LICENSE.md"),
        ],
    )
    monkeypatch.setattr(aura, "SAM2_LICENSE_SHA256", aura._sha256(sam2 / "LICENSE"))
    original_tracked_files = aura._tracked_files
    monkeypatch.setattr(
        aura,
        "_tracked_files",
        lambda repo, prefix="": (
            [("LICENSE", sam2 / "LICENSE")]
            if repo == sam2
            else original_tracked_files(repo, prefix)
        ),
    )
    snapshots = [
        {"artifact_id": artifact_id, "rights_established": True}
        for artifact_id in (
            "aurafusion360_sunflower_author_scene",
            "aurafusion360_sunflower_expected_output",
            "aurafusion360_sam2_hiera_large",
            "aurafusion360_marigold_depth_v1_0",
            "aurafusion360_sd2_inpainting_exact_checkpoint",
        )
    ]
    receipt: dict[str, object] = {
        "methods": {
            "aurafusion360_quality_challenger": {
                "checkpoint_rights_established": True,
                "author_data_rights_established": True,
                "remote_snapshots": snapshots,
            }
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    monkeypatch.setattr(aura, "PREREQUISITE_RECEIPT_DIGEST", receipt["receipt_digest"])
    prerequisite = tmp_path / "prerequisite.json"
    _write_json(prerequisite, receipt)
    return {
        "repo": repo,
        "source": source,
        "sam2": sam2,
        "prerequisite": prerequisite,
        "job": tmp_path / "job",
    }


def test_bundle_derives_source_and_rights_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        prerequisite_receipt_path=paths["prerequisite"],
        job_dir=paths["job"],
        generated_at="2026-08-04T00:00:00+00:00",
    )
    assert receipt["status"] == "ready"
    assert receipt["retry_cap"] == 0
    assert receipt["depth_anything3_used"] is False
    assert receipt["smoke_scope"] == "unchanged_author_inpaint_init_stage_only"
    assert Path(receipt["bundle_path"]).is_file()
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        runner = archive.read(
            "provider_runtime/adp_aura_author_smoke_provider_runner.py"
        ).decode()
    assert 'filename=expected["expected_ply_path"]' in runner
    assert 'allow_patterns=[expected["path_prefix"] + "*"]' not in runner


def test_bundle_rejects_missing_publisher_rights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(paths["prerequisite"].read_text())
    receipt["methods"]["aurafusion360_quality_challenger"][
        "author_data_rights_established"
    ] = False
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    monkeypatch.setattr(aura, "PREREQUISITE_RECEIPT_DIGEST", receipt["receipt_digest"])
    _write_json(paths["prerequisite"], receipt)
    with pytest.raises(ValueError, match="adp_aura_prerequisite_rights_missing"):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            prerequisite_receipt_path=paths["prerequisite"],
            job_dir=paths["job"],
        )


def test_bundle_rejects_changed_sam2_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    original_git = aura._git

    def changed_git(path: Path, *args: str) -> str:
        if path == paths["sam2"] and args == ("status", "--porcelain"):
            return " M sam2/modeling/sam2_base.py"
        return original_git(path, *args)

    monkeypatch.setattr(aura, "_git", changed_git)
    with pytest.raises(ValueError, match="adp_aura_sam2_source_identity_mismatch"):
        aura.build_aura_author_smoke_vast_bundle(
            repo_root=paths["repo"],
            aura_root=paths["source"],
            sam2_root=paths["sam2"],
            prerequisite_receipt_path=paths["prerequisite"],
            job_dir=paths["job"],
        )


def test_dry_run_requires_exact_bundle_without_provider_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        prerequisite_receipt_path=paths["prerequisite"],
        job_dir=paths["job"],
    )
    result = aura.run_aura_author_smoke_vast(
        job_dir=tmp_path / "run",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=receipt,
    )
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0


def test_provider_contract_rejects_generic_import_smoke() -> None:
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_aura_smoke",
        entrypoint_text="adp_aura_smoke_runner_failed_without_runtime_result "
        "blocked_adp_aura_smoke_process_exited_without_result "
        "--no-build-isolation setuptools==80.9.0 torch==2.5.1",
        runner_text="import torch",
    )
    assert blockers == ["provider_runner_missing_adp_aura_smoke_runtime_contract"]


def test_provider_contract_rejects_isolated_native_extension_build() -> None:
    blockers = provider_runtime_contract_blockers(
        provider_bundle_kind="adp_aura_smoke",
        entrypoint_text=(
            "adp_aura_smoke_runner_failed_without_runtime_result "
            "blocked_adp_aura_smoke_process_exited_without_result torch==2.5.1"
        ),
        runner_text=(
            "adp_aura_author_smoke_result.json inpaint_init_executed "
            "author_source_modified published_expected_output_bound "
            "depth_anything3_used"
        ),
    )
    assert blockers == ["provider_entrypoint_missing_runtime_result_crash_fallback"]


def test_vast_adapter_preflights_dedicated_aura_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        prerequisite_receipt_path=paths["prerequisite"],
        job_dir=paths["job"],
    )
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_aura_smoke",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_aura_smoke",
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.test",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_aura_smoke",
    )
    assert "run_adp_aura_author_smoke_provider_runtime.sh" in script
    assert "adp_aura_provider_runtime_output.zip" in script
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        entrypoint = archive.read(
            "provider_runtime/run_adp_aura_author_smoke_provider_runtime.sh"
        ).decode()
    assert "--no-build-isolation" in entrypoint
    assert "setuptools==80.9.0" in entrypoint
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_aura_smoke",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.test/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.test/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"


def _allocator_args(
    tmp_path: Path, bundle_receipt: Path, *, execute: bool
) -> list[str]:
    args = [
        "gpu-canary",
        "--probe-kind",
        aura.PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-aura-smoke",
        "--adp-aura-bundle-receipt",
        str(bundle_receipt),
        "--adp-job-dir",
        str(tmp_path / "run"),
        "--adp-max-hourly-rate-usd",
        "1.50",
        "--adp-max-spend-usd",
        "5.00",
        "--adp-hard-ttl-seconds",
        "10800",
    ]
    if execute:
        args.append("--execute")
    return args


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_aura_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    receipt = aura.build_aura_author_smoke_vast_bundle(
        repo_root=paths["repo"],
        aura_root=paths["source"],
        sam2_root=paths["sam2"],
        prerequisite_receipt_path=paths["prerequisite"],
        job_dir=paths["job"],
    )
    receipt_path = paths["job"] / "adp_aura_author_smoke_bundle_receipt.json"
    monkeypatch.setattr(
        allocator,
        "ADP_AURA_PREREQUISITE_RECEIPT_DIGEST",
        receipt["prerequisite_receipt_digest"],
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    observed: dict = {}

    def fake_run(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_aura_author_smoke_vast", fake_run)
    assert allocator.main(_allocator_args(tmp_path, receipt_path, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["aura_inpaint_init_author_smoke_only"] is True
    assert admission["full_author_workflow_claimed"] is False
    assert admission["hard_cap_usd"] == 5.0


def test_output_inspector_recognizes_aura_runtime_result(tmp_path: Path) -> None:
    output = tmp_path / "output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "adp_aura_author_smoke_result.json",
            json.dumps({"status": "blocked", "blockers": ["typed_runtime_failure"]}),
        )
    inspection = inspect_provider_runtime_output_zip(output)
    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_blockers"] == ["typed_runtime_failure"]
