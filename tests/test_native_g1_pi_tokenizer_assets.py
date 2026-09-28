from __future__ import annotations

import hashlib
import json
import urllib.error
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_pi_tokenizer_assets as assets
from blueprint_pipeline.native_g1_provider_bundle import _verify_embedded_pi_tokenizer


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "assets"
    root.mkdir()
    rows = []
    for name in sorted(assets.EXPECTED_FILES):
        content = ("pinned-" + name).encode()
        (root / name).write_bytes(content)
        if name in {"tokenizer.json", "tokenizer.model"}:
            identity = {"sha256": hashlib.sha256(content).hexdigest()}
        else:
            identity = {
                "git_blob_sha1": hashlib.sha1(
                    f"blob {len(content)}\0".encode() + content
                ).hexdigest()
            }
        rows.append({"path": name, "size_bytes": len(content), **identity})
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "g1_paligemma_tokenizer_inventory.v1",
                "source_repository": assets.SOURCE_REPOSITORY,
                "source_revision": assets.SOURCE_REVISION,
                "model_license": "gemma",
                "rights_review_required": True,
                "files": rows,
            }
        )
    )
    return inventory, root


def test_verified_assets_require_each_exact_file_and_reject_tampering(tmp_path: Path) -> None:
    inventory, root = _fixture(tmp_path)
    receipt = assets.verify_tokenizer_assets(inventory_path=inventory, asset_dir=root)
    assert receipt["status"] == "tokenizer_bytes_verified"
    assert len(receipt["files"]) == 6
    (root / "tokenizer_config.json").write_text("changed")
    with pytest.raises(ValueError, match="size_mismatch"):
        assets.verify_tokenizer_assets(inventory_path=inventory, asset_dir=root)
    original = b"pinned-tokenizer_config.json"
    (root / "tokenizer_config.json").write_bytes(b"x" + original[1:])
    with pytest.raises(ValueError, match="digest_mismatch"):
        assets.verify_tokenizer_assets(inventory_path=inventory, asset_dir=root)


def test_real_inventory_pins_upstream_revision_and_six_files() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/g1_paligemma_tokenizer_inventory.v1.json"
    value = assets._inventory(path)
    assert value["source_revision"] == assets.SOURCE_REVISION
    assert {row["path"] for row in value["files"]} == assets.EXPECTED_FILES


def test_provider_stage_uses_verified_bundle_bytes_and_refuses_changed_destination(
    tmp_path: Path,
) -> None:
    inventory, root = _fixture(tmp_path)
    target = tmp_path / "compat"
    receipt = assets.stage_provider_tokenizer(
        inventory_path=inventory,
        bundled_dir=root,
        destination=target,
    )
    assert receipt["status"] == "tokenizer_compat_path_ready"
    assert receipt["compat_path"] == str(target)
    (target / "tokenizer.json").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="size_mismatch"):
        assets.stage_provider_tokenizer(
            inventory_path=inventory,
            bundled_dir=root,
            destination=target,
        )


def test_bundle_rechecks_embedded_tokenizer_bytes(tmp_path: Path) -> None:
    inventory, root = _fixture(tmp_path)
    receipt = assets.verify_tokenizer_assets(inventory_path=inventory, asset_dir=root)
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(inventory, "configs/" + inventory.name)
        for name in sorted(assets.EXPECTED_FILES):
            archive.write(root / name, "provider_runtime/inputs/pi_tokenizer/" + name)
    with zipfile.ZipFile(archive_path) as archive:
        _verify_embedded_pi_tokenizer(
            archive,
            {"pi_tokenizer_assets": receipt},
            inventory_path=inventory,
        )
    corrupt_archive = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(corrupt_archive, "w") as archive:
        archive.write(inventory, "configs/" + inventory.name)
        for name in sorted(assets.EXPECTED_FILES):
            content = b"changed" if name == "tokenizer.model" else (root / name).read_bytes()
            archive.writestr("provider_runtime/inputs/pi_tokenizer/" + name, content)
    with zipfile.ZipFile(corrupt_archive) as archive:
        with pytest.raises(ValueError, match="pi_tokenizer_file_size_mismatch"):
            _verify_embedded_pi_tokenizer(
                archive,
                {"pi_tokenizer_assets": receipt},
                inventory_path=inventory,
            )


@pytest.mark.parametrize("candidate", list(assets.PUBLISHER_TOKENIZER_REFS))
def test_pi_policy_rejects_a_new_publisher_local_tokenizer_reference(
    tmp_path: Path,
    candidate: str,
) -> None:
    policy = tmp_path / "policy"
    policy.mkdir()
    preprocessor = policy / "policy_preprocessor.json"
    preprocessor.write_text(
        json.dumps(
            {"steps": [{"config": {"tokenizer_name": assets.PUBLISHER_TOKENIZER_REFS[candidate]}}]}
        )
    )
    assets.require_policy_tokenizer_reference(policy, candidate)
    preprocessor.write_text(
        json.dumps({"steps": [{"config": {"tokenizer_name": "/publisher/new/private/path"}}]})
    )
    with pytest.raises(ValueError, match="publisher_reference_changed"):
        assets.require_policy_tokenizer_reference(policy, candidate)


def test_access_check_reports_403_without_exposing_credential(tmp_path: Path, monkeypatch) -> None:
    inventory, _ = _fixture(tmp_path)
    credential = tmp_path / "token"
    credential.write_text("secret-test-token")
    credential.chmod(0o640)
    monkeypatch.setattr(
        assets,
        "_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            urllib.error.HTTPError("https://huggingface.co/", 403, "Forbidden", None, None)
        ),
    )
    with pytest.raises(ValueError, match="g1_pi_tokenizer_access_http_403") as exc:
        assets.preflight_tokenizer_access(inventory_path=inventory, token_file=credential)
    assert "secret-test-token" not in str(exc.value)
