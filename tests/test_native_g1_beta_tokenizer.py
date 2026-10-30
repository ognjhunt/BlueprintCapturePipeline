"""Retained model-only Pi assets need no manual environment edit or credentials."""
import hashlib
import shutil

import pytest

from blueprint_pipeline import native_g1_beta_tokenizer as binding
from blueprint_pipeline import native_g1_pi_tokenizer_assets as assets
from blueprint_pipeline.native_g1_beta_assets import ORDER
from tests.test_native_g1_pi_tokenizer_assets import _fixture


def fixture(tmp_path, monkeypatch):
    inventory, root = _fixture(tmp_path)
    monkeypatch.delenv(binding.TOKENIZER_DIR_ENV, raising=False)
    monkeypatch.setattr(binding, "INVENTORY", inventory)
    monkeypatch.setattr(binding, "PI_AUXILIARY", {"inventory_file_sha256": "sha256:" + hashlib.sha256(inventory.read_bytes()).hexdigest()})
    monkeypatch.setattr(binding, "RETAINED_TOKENIZER_DIR", root)
    monkeypatch.setattr(assets, "_token", lambda *args: pytest.fail("credential read"))
    monkeypatch.setattr(assets, "_request", lambda *args, **kwargs: pytest.fail("network request"))
    return inventory, root


@pytest.mark.parametrize("candidate", [candidate for candidate in ORDER if "_pi05_" in candidate])
def test_pi_uses_reviewed_retained_default_and_verifies_all_six_files(tmp_path, monkeypatch, candidate):
    _, root = fixture(tmp_path, monkeypatch)
    directory, receipt = binding.resolve_selected_tokenizer([candidate])
    assert directory == root
    assert receipt["status"] == "tokenizer_bytes_verified"
    assert {row["path"] for row in receipt["files"]} == assets.EXPECTED_FILES
    assert receipt["inference_executed"] is False


@pytest.mark.parametrize("mutation,reason", [("directory", "unavailable"), ("missing", "invalid"),
    ("tampered", "invalid"), ("inventory", "invalid"), ("symlink", "invalid")])
def test_unavailable_or_changed_retained_assets_have_bounded_individual_reason(tmp_path, monkeypatch, mutation, reason):
    inventory, root = fixture(tmp_path, monkeypatch)
    if mutation == "directory":
        monkeypatch.setattr(binding, "RETAINED_TOKENIZER_DIR", tmp_path / "absent")
    elif mutation == "missing":
        (root / "tokenizer.model").unlink()
    elif mutation == "tampered":
        path = root / "tokenizer.model"
        path.write_bytes(b"x" * path.stat().st_size)
    elif mutation == "inventory":
        inventory.write_text(inventory.read_text() + " ")
    else:
        path = root / "tokenizer.model"
        other = tmp_path / "outside-model"
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(ValueError, match="^g1_beta_selected_pi_tokenizer_" + reason + "$"):
        binding.resolve_selected_tokenizer([ORDER[1]])


def test_explicit_and_environment_overrides_are_respected_without_silent_fallback(tmp_path, monkeypatch):
    _, root = fixture(tmp_path, monkeypatch)
    override = tmp_path / "override"
    shutil.copytree(root, override)
    monkeypatch.setenv(binding.TOKENIZER_DIR_ENV, str(override))
    assert binding.resolve_selected_tokenizer([ORDER[1]])[0] == override
    monkeypatch.setenv(binding.TOKENIZER_DIR_ENV, str(tmp_path / "missing-override"))
    with pytest.raises(ValueError, match="unavailable"):
        binding.resolve_selected_tokenizer([ORDER[1]])
    assert binding.resolve_selected_tokenizer([ORDER[1]], explicit_dir=root)[0] == root


def test_inaccessible_files_return_an_operational_reason_without_exception_details(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    def refused(**kwargs):
        raise PermissionError("private host path details")
    monkeypatch.setattr(binding, "verify_tokenizer_assets", refused)
    with pytest.raises(ValueError, match="^g1_beta_selected_pi_tokenizer_unavailable$"):
        binding.resolve_selected_tokenizer([ORDER[1]])


@pytest.mark.parametrize("candidate", [candidate for candidate in ORDER if "_dp_" in candidate])
def test_dp_does_not_read_tokenizer_even_when_override_is_invalid(tmp_path, monkeypatch, candidate):
    monkeypatch.setenv(binding.TOKENIZER_DIR_ENV, str(tmp_path / "missing"))
    monkeypatch.setattr(binding, "verify_tokenizer_assets", lambda **kwargs: pytest.fail("DP read Pi inputs"))
    assert binding.resolve_selected_tokenizer([candidate]) == (None, None)
