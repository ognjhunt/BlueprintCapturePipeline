"""Pinned source retention: real byte identity, bounded I/O and safe retries."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest


def _blob_sha(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    recipe = assets.recipe_for("kuka_lbr_iiwa_14")
    catalog = assets.asset_recipes()
    catalog["recipes"] = [recipe, assets.recipe_for("yaskawa_gp20"),
                          assets.recipe_for("droid_franka_robotiq_2f85")]
    catalog.pop("recipes_digest")
    data_path = tmp_path / "recipes.json"
    data_path.write_text(json.dumps(catalog))
    monkeypatch.setattr(assets, "DATA", data_path)
    content = {
        "kuka_iiwa_14/iiwa14.xml": b'<mujoco model="development_only"/>\n',
        "kuka_iiwa_14/LICENSE": b"Development fixture license bytes\n",
        "kuka_iiwa_14/assets/link.obj": b"v 0 0 0\n",
    }
    tree = {"truncated": False, "tree": [
        {"path": name, "type": "blob", "mode": "100644", "size": len(value), "sha": _blob_sha(value)}
        for name, value in content.items()
    ]}
    calls = []

    def get(url, *, maximum):
        calls.append(url)
        if url.startswith("https://api.github.com/"):
            assert f'/git/trees/{recipe["source"]["revision"]}?recursive=1' in url
            result = json.dumps(tree).encode()
        else:
            prefix = ("https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/"
                      f'{recipe["source"]["revision"]}/')
            assert url.startswith(prefix)
            result = content[url.removeprefix(prefix)]
        assert len(result) <= maximum
        return result

    return SimpleNamespace(recipe=recipe, data=catalog, data_path=data_path, content=content,
                           tree=tree, calls=calls, get=get, destination=tmp_path / "source-bundle")


def _materialize(case):
    return assets.materialize_sources(case.recipe["embodiment_id"], case.destination, get=case.get)


def _rewrite_receipt(case, update):
    path = case.destination / assets.RECEIPT
    value = json.loads(path.read_text())
    update(value)
    value["bundle_digest"] = cross_runtime_canonical_digest(value, digest_field="bundle_digest")
    path.write_text(json.dumps(value))


def test_public_fetch_uses_required_user_agent_and_existing_https_boundary(monkeypatch):
    calls = []

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(body=b"{}")

    monkeypatch.setattr(assets, "fetch_bounded_https", fetch)
    assert assets._get("https://api.github.com/repos/example/robot/git/trees/commit", maximum=1024) == b"{}"
    _, options = calls[0]
    assert options["headers"]["User-Agent"]
    assert options["allowed_origins"] == ["https://api.github.com"]
    assert options["max_redirects"] == 0 and options["max_bytes"] == 1024


def test_materialize_exact_git_blobs_sha256_and_license_first_then_retry_without_fetch(source_case):
    case = source_case
    result = _materialize(case)
    assert result["claim_ceiling"] == "source_asset_not_qualified"
    assert result["execution_authorized"] is False
    assert result["source_arm_count"] == 1
    assert case.calls[1].endswith("/kuka_iiwa_14/LICENSE")
    assert result["bundle_digest"] == cross_runtime_canonical_digest(result, digest_field="bundle_digest")
    for record in result["files"]:
        data = (case.destination / "source" / record["path"]).read_bytes()
        assert data == case.content[record["path"]]
        assert record["git_blob_sha"] == _blob_sha(data)
        assert record["sha256"] == "sha256:" + hashlib.sha256(data).hexdigest()
    case.calls.clear()
    assert _materialize(case) == result
    assert case.calls == []


@pytest.mark.parametrize("defect", ["hash", "size", "lfs", "symlink", "submodule", "truncated", "escape", "no_license"])
def test_tree_and_blob_integrity_refusals_leave_no_partial_bundle(source_case, defect):
    case = source_case
    item = case.tree["tree"][0]
    expected = "asset_source_"
    if defect == "hash":
        item["sha"] = "0" * 40
        expected += "git_blob_changed"
    elif defect == "size":
        item["size"] += 1
        expected += "size_changed"
    elif defect == "lfs":
        case.content[item["path"]] = b"version https://git-lfs.github.com/spec/v1\noid sha256:unmaterialized\n"
        item["size"] = len(case.content[item["path"]])
        item["sha"] = _blob_sha(case.content[item["path"]])
        expected += "lfs_bytes_not_materialized"
    elif defect in {"symlink", "submodule"}:
        item.update(mode="120000" if defect == "symlink" else "160000",
                    type="blob" if defect == "symlink" else "commit")
        expected += "symlink_or_submodule_refused"
    elif defect == "truncated":
        case.tree["truncated"] = True
        expected += "tree_incomplete"
    elif defect == "escape":
        item["path"] = "../outside"
        expected = "asset_source_path_invalid"
    else:
        case.tree["tree"] = [row for row in case.tree["tree"] if row["path"] != "kuka_iiwa_14/LICENSE"]
        expected += "entry_or_license_missing"
    with pytest.raises(assets.AssetPreparationError, match=expected):
        _materialize(case)
    assert not case.destination.exists()
    assert not list(case.destination.parent.glob(".blueprint-assets-*"))


@pytest.mark.parametrize("bound", ["MAX_FILES", "MAX_BYTES", "MAX_FILE_BYTES"])
def test_declared_bounds_refuse_before_any_source_file_is_fetched(source_case, monkeypatch, bound):
    case = source_case
    monkeypatch.setattr(assets, bound, 1)
    with pytest.raises(assets.AssetPreparationError, match="bound_exceeded"):
        _materialize(case)
    assert len(case.calls) == 1  # Only the bounded repository-tree read.
    assert not case.destination.exists()


@pytest.mark.parametrize("defect", ["file_changed", "orphan", "file_symlink", "directory_symlink", "extra_symlink"])
def test_retry_rejects_changed_bytes_and_every_symlink_without_refetch(source_case, defect):
    case = source_case
    _materialize(case)
    case.calls.clear()
    source = case.destination / "source"
    path = source / "kuka_iiwa_14/iiwa14.xml"
    if defect == "file_changed":
        path.write_bytes(b"changed")
    elif defect == "orphan":
        (source / "orphan").write_text("unbound data")
    elif defect == "file_symlink":
        external = case.destination.parent / "outside"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)
    elif defect == "directory_symlink":
        external = case.destination.parent / "outside-model"
        (source / "kuka_iiwa_14").rename(external)
        (source / "kuka_iiwa_14").symlink_to(external, target_is_directory=True)
    else:
        (source / "unrecorded-link").symlink_to(case.destination.parent, target_is_directory=True)
    with pytest.raises(assets.AssetPreparationError, match="asset_bundle_"):
        _materialize(case)
    assert case.calls == []


@pytest.mark.parametrize("defect", ["empty_inventory", "license", "source_revision", "execution", "git_hash"])
def test_even_redigested_receipts_must_preserve_source_binding_and_file_truth(source_case, defect):
    case = source_case
    _materialize(case)

    def update(value):
        if defect == "empty_inventory":
            value["files"] = []
        elif defect == "license":
            value["license"]["identifier"] = "MIT"
        elif defect == "source_revision":
            value["revision"] = "a" * 40
        elif defect == "execution":
            value["execution_authorized"] = True
        else:
            value["files"][0]["git_blob_sha"] = "a" * 40

    _rewrite_receipt(case, update)
    with pytest.raises(assets.AssetPreparationError, match="asset_bundle_"):
        assets.verify_source_bundle(case.destination)


def test_recipe_change_refuses_old_cache_instead_of_silently_replacing_it(source_case):
    case = source_case
    result = _materialize(case)
    data = deepcopy(case.data)
    data["recipes"][0]["composition_required"].append("new_exact_tool_required")
    case.data_path.write_text(json.dumps(data))
    case.calls.clear()
    with pytest.raises(assets.AssetPreparationError, match="asset_bundle_recipe_changed"):
        _materialize(case)
    assert json.loads((case.destination / assets.RECEIPT).read_text()) == result
    assert case.calls == []


def test_empty_destination_created_during_fetch_is_never_replaced(source_case, monkeypatch):
    case = source_case
    publish = assets.publish_staged_immutable_directory

    def race(**kwargs):
        Path(kwargs["destination"]).mkdir()
        return publish(**kwargs)

    monkeypatch.setattr(assets, "publish_staged_immutable_directory", race)
    with pytest.raises(assets.AssetPreparationError, match="asset_destination_appeared"):
        _materialize(case)
    assert case.destination.is_dir()
    assert list(case.destination.iterdir()) == []


@pytest.mark.parametrize("identity", ["yaskawa_gp20", "droid_franka_robotiq_2f85", "unknown_robot"])
def test_unavailable_or_existing_runtime_does_not_fetch_or_fabricate_source(source_case, identity):
    case = source_case
    with pytest.raises(assets.AssetPreparationError, match="asset_(source|recipe)_"):
        assets.materialize_sources(identity, case.destination, get=case.get)
    assert case.calls == []
    assert not case.destination.exists()
