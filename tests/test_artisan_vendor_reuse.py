"""ADP-009/day 28: admit real temporary bytes without copying the SDK again."""
import json
import shutil

import pytest

from blueprint_pipeline.artisan_vendor_reuse import (
    PACKAGE_PREFIX, prepare_vendor_reuse, vendor_copy_function,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.immutable_file_digest import sha256_file


def setup_vendor(tmp_path):
    runtime = tmp_path / "runtime"
    prior = runtime / "scene-configuration" / ("a" * 40)
    source = tmp_path / "installed"
    files = []
    for kind, name in [("site_packages", "module.py"), ("blender", "blender")]:
        current = source / kind / name
        current.parent.mkdir(parents=True)
        current.write_bytes((kind + " installed bytes\n").encode() * 100)
        current.chmod(0o755 if kind == "blender" else 0o644)
        relative = PACKAGE_PREFIX + f"runtime/{kind}/{name}"
        published = prior / relative
        published.parent.mkdir(parents=True)
        published.write_bytes(current.read_bytes())
        published.chmod(0o555 if kind == "blender" else 0o444)
        files.append({"relative_path": relative, "size_bytes": published.stat().st_size,
                      "sha256": sha256_file(published)})
    files.append({"relative_path": "other/new-source.py", "size_bytes": 1000, "sha256": "unused"})
    manifest = {"files": files}
    manifest["toolchain_digest"] = canonical_digest(manifest, digest_field="toolchain_digest")
    (prior / "task_evaluation_scene_configuration_toolchain.v1.json").write_text(json.dumps(manifest))
    args = {"toolchain_root": prior, "site_packages_root": source / "site_packages",
            "blender_root": source / "blender", "staging_device": tmp_path.stat().st_dev}
    return runtime, source, prior, args


def test_full_byte_reuse_preserves_installed_sdk_and_published_bytes(tmp_path):
    _, source, prior, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    assert plan and plan["shared_bytes"] == sum(row["size_bytes"] for row in plan["files"])
    output = tmp_path / "new"
    for kind in ("site_packages", "blender"):
        copy = vendor_copy_function(plan, kind=kind, source_root=source / kind)
        shutil.copytree(source / kind, output / kind, copy_function=copy)
        copy.require_complete()
        for row in [r for r in plan["files"] if r["kind"] == kind]:
            target = output / kind / row["relative_path"]
            assert target.stat().st_ino == (prior / row["published_path"]).stat().st_ino
            assert not target.stat().st_mode & 0o222
    assert (source / "site_packages/module.py").stat().st_mode & 0o200
    shutil.rmtree(output)
    assert all(sha256_file(prior / r["published_path"]) == r["sha256"] for r in plan["files"])


@pytest.mark.parametrize("mutation", ["source_bytes", "published_bytes", "writable", "cross_device"])
def test_unverifiable_reuse_does_not_discount_the_full_copy(tmp_path, mutation):
    _, source, prior, args = setup_vendor(tmp_path)
    published = prior / (PACKAGE_PREFIX + "runtime/site_packages/module.py")
    if mutation == "source_bytes":
        (source / "site_packages/module.py").write_bytes(b"changed installation")
    elif mutation == "published_bytes":
        published.chmod(0o644)
        published.write_bytes(b"changed published bytes")
        published.chmod(0o444)
    elif mutation == "writable":
        published.chmod(0o644)
    else:
        args["staging_device"] += 1
    assert prepare_vendor_reuse(**args) is None


def test_changed_sdk_after_admission_fails_without_copy_fallback(tmp_path):
    _, source, _, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    (source / "site_packages/module.py").write_bytes(b"changed after admission")
    target = tmp_path / "new.py"
    with pytest.raises(ValueError, match="bytes_changed"):
        vendor_copy_function(plan, kind="site_packages", source_root=source / "site_packages")(
            source / "site_packages/module.py", target)
    assert not target.exists()


def test_missing_sdk_file_after_admission_refuses_incomplete_copy(tmp_path):
    _, source, _, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    (source / "site_packages/module.py").unlink()
    copy = vendor_copy_function(plan, kind="site_packages", source_root=source / "site_packages")
    shutil.copytree(source / "site_packages", tmp_path / "new", copy_function=copy)
    with pytest.raises(ValueError, match="source_set_changed"):
        copy.require_complete()


def test_deployment_reserves_unshared_bytes_and_binds_required_copy_plan(tmp_path):
    from scripts import deploy_control_plane_commit as deploy
    runtime, source, prior, _ = setup_vendor(tmp_path)
    estimate = deploy._scene_configuration_staging_estimate(runtime, "b" * 40)
    full = estimate["bytes"]
    plan_path = deploy._bind_artisan_vendor_reuse(estimate, runtime_root=runtime,
        state_root=tmp_path / "state", site_packages_root=source / "site_packages",
        blender_root=source / "blender")
    plan = json.loads(plan_path.read_text())
    assert plan["toolchain_root"] == str(prior)
    assert estimate["full_copy_bytes"] == full
    assert estimate["vendor_reuse"]["digest"] == plan["digest"]
    assert estimate["target_root"] == str(prior.parent)
    assert estimate["bytes"] == 1050 + 3 * 4096 + deploy.RELEASE_ESTIMATE_OVERHEAD_BYTES


def test_target_release_temporary_files_use_the_reserved_sdk_filesystem(tmp_path, monkeypatch):
    import subprocess
    from scripts import deploy_control_plane_commit as deploy
    _, _, _, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    path = tmp_path / "reuse.json"
    path.write_text(json.dumps(plan))
    observed = {}

    def run(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
            "status": "ready", "source_commit": "b" * 40, "environment": {}}))

    monkeypatch.setattr(deploy.subprocess, "run", run)
    deploy._provision_scene_configuration_from_release(repository_root=tmp_path,
        source_commit="b" * 40, readback_user="blueprint", artisan_vendor_reuse_plan=path)
    assert observed["env"]["TMPDIR"] == plan["staging_root"]
