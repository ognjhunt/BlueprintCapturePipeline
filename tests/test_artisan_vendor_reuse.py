"""ADP-009/day 28: admit real temporary bytes without copying the SDK again."""
import json
import shutil
import time

import pytest

from blueprint_pipeline import artisan_vendor_reuse as vendor_reuse
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


def _recorded_reads(monkeypatch):
    from blueprint_pipeline import immutable_file_digest
    reads = []
    original = immutable_file_digest._read_digest
    monkeypatch.setattr(immutable_file_digest, "_read_digest",
                        lambda stream: reads.append(stream.name) or original(stream))
    return reads


def _sources(source):
    return sorted(str(path) for path in source.rglob("*") if path.is_file())


def _write_plan(directory, plan):
    directory.mkdir(exist_ok=True)
    path = directory / (plan["digest"][7:] + ".json")
    path.write_text(json.dumps(plan))
    path.chmod(0o444)
    return path


@pytest.fixture()
def settled(monkeypatch):
    """Every test file was written just now; a real vendor tree is days old."""
    real = time.time_ns
    monkeypatch.setattr(vendor_reuse.time, "time_ns", lambda: real() + 60_000_000_000)


def test_build_does_not_hash_a_source_again_when_its_identity_held_since_planning(tmp_path, monkeypatch, settled):
    _, source, _, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    assert plan["source_files_hashed"] == 2
    reads = _recorded_reads(monkeypatch)
    copy = vendor_copy_function(plan, kind="site_packages", source_root=source / "site_packages")
    copy(source / "site_packages/module.py", tmp_path / "new.py")
    assert str(source / "site_packages/module.py") not in reads


def test_next_deploys_planning_skips_only_sources_whose_identity_held(tmp_path, monkeypatch, settled):
    _, source, _, args = setup_vendor(tmp_path)
    plans = tmp_path / "plans"
    _write_plan(plans, prepare_vendor_reuse(**args))
    reads = _recorded_reads(monkeypatch)

    next_plan = prepare_vendor_reuse(**args, previous_plans=plans)

    assert next_plan["source_files_hashed"] == 0 and next_plan["source_bytes_hashed"] == 0
    assert not set(reads) & set(_sources(source))
    assert [row["sha256"] for row in next_plan["files"]] == [row["sha256"] for row in prepare_vendor_reuse(**args)["files"]]

    moved = source / "blender/blender"
    moved.chmod(0o775)
    reads.clear()
    after_chmod = prepare_vendor_reuse(**args, previous_plans=plans)
    assert after_chmod["source_files_hashed"] == 1
    assert reads.count(str(moved)) == 1 and str(source / "site_packages/module.py") not in reads


def test_changed_source_bytes_are_refused_although_an_older_plan_remembers_them(tmp_path, settled):
    _, source, _, args = setup_vendor(tmp_path)
    plans = tmp_path / "plans"
    _write_plan(plans, prepare_vendor_reuse(**args))
    (source / "site_packages/module.py").write_bytes(b"changed installation")
    assert prepare_vendor_reuse(**args, previous_plans=plans) is None


def test_source_changed_with_the_same_size_after_planning_stops_the_build(tmp_path, settled):
    _, source, _, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    original = source / "site_packages/module.py"
    original.write_bytes(bytes(reversed(original.read_bytes())))
    with pytest.raises(ValueError, match="bytes_changed"):
        vendor_copy_function(plan, kind="site_packages", source_root=source / "site_packages")(
            original, tmp_path / "new.py")


def test_a_file_written_just_now_is_not_remembered_by_its_identity(tmp_path):
    """A second write in the same clock tick would leave the identity as it was."""
    _, _, _, args = setup_vendor(tmp_path)
    plan = prepare_vendor_reuse(**args)
    assert plan is not None and {row["source_identity"] for row in plan["files"]} == {None}


@pytest.mark.parametrize("damage", ["writable", "edited", "other_sources", "other_schema", "not_json", "symlink"])
def test_an_untrustworthy_plan_only_means_the_bytes_are_read_again(tmp_path, damage, settled):
    _, source, _, args = setup_vendor(tmp_path)
    plans = tmp_path / "plans"
    plan = prepare_vendor_reuse(**args)
    path = _write_plan(plans, plan)
    path.chmod(0o644)
    if damage == "edited":
        plan["files"][0]["sha256"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(plan))
    elif damage == "other_sources":
        plan["sources"]["blender"] = "/elsewhere"
        plan["digest"] = canonical_digest(plan, digest_field="digest")
        path.write_text(json.dumps(plan))
    elif damage == "other_schema":
        plan["schema_version"] = "artisan_vendor_reuse.v0"
        plan["digest"] = canonical_digest(plan, digest_field="digest")
        path.write_text(json.dumps(plan))
    elif damage == "not_json":
        path.write_text("{")
    elif damage == "symlink":
        target = tmp_path / "elsewhere.json"
        target.write_text(path.read_text())
        target.chmod(0o444)
        path.unlink()
        path.symlink_to(target)
    if damage not in {"writable", "symlink"}:
        path.chmod(0o444)
    assert prepare_vendor_reuse(**args, previous_plans=plans)["source_files_hashed"] == 2


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


def test_second_deploy_plans_from_the_first_deploys_plan_and_reports_it(tmp_path, settled):
    from scripts import deploy_control_plane_commit as deploy
    runtime, source, _, _ = setup_vendor(tmp_path)

    def bind():
        estimate = deploy._scene_configuration_staging_estimate(runtime, "b" * 40)
        deploy._bind_artisan_vendor_reuse(estimate, runtime_root=runtime, state_root=tmp_path / "state",
                                          site_packages_root=source / "site_packages",
                                          blender_root=source / "blender")
        return estimate["vendor_reuse"]

    first = bind()
    assert first["source_files_hashed"] == 2 and first["source_bytes_hashed"] > 0
    assert first["planning_seconds"] >= 0
    second = bind()
    assert second["source_files_hashed"] == 0 and second["source_bytes_hashed"] == 0
    assert second["shared_bytes"] == first["shared_bytes"] and second["digest"] != first["digest"]


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


def test_planning_reads_every_source_but_not_inodes_the_toolchain_readback_proved(tmp_path, monkeypatch):
    from blueprint_pipeline import immutable_file_digest
    from blueprint_pipeline.immutable_file_digest import sealed_identity
    from blueprint_pipeline.readback_ledger import write_ledger

    _, source, prior, args = setup_vendor(tmp_path)
    manifest_path = prior / "task_evaluation_scene_configuration_toolchain.v1.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["readback_actor"] = "service-account:blueprint"
    manifest["toolchain_digest"] = canonical_digest(manifest, digest_field="toolchain_digest")
    manifest_path.write_text(json.dumps(manifest))
    published = [row["relative_path"] for row in manifest["files"] if (prior / row["relative_path"]).exists()]
    reads = []
    original = immutable_file_digest._read_digest
    monkeypatch.setattr(immutable_file_digest, "_read_digest",
                        lambda stream: reads.append(stream.name) or original(stream))
    assert prepare_vendor_reuse(**args) is not None
    assert len(reads) == 4  # without a ledger: two sources, two published files
    reads.clear()
    assert write_ledger(prior, tree_digest=manifest["toolchain_digest"], readback_actor="service-account:blueprint",
                        inodes={name: sealed_identity((prior / name).stat()) for name in published})["status"] == "written"
    assert prepare_vendor_reuse(**args) is not None
    assert sorted(reads) == sorted(str(path) for path in source.rglob("*") if path.is_file())
