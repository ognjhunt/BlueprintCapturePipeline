"""Real clean Git and file hashing survive a controller-to-provider relocation."""
from copy import deepcopy
import hashlib
import json
import shutil
import subprocess

import pytest

from blueprint_pipeline import native_g1_beta_runtime as runtime
from blueprint_pipeline import native_g1_publisher_source_stage as stage
from blueprint_pipeline import native_g1_policy_server_supervisor as supervisor
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def relocated_source(tmp_path, monkeypatch):
    controller = tmp_path / "controller-source"
    source = controller / "source"
    source.mkdir(parents=True)
    rows = {
        "lerobot/scripts/serve_lerobot_vla_http.py": ("PINNED_POLICY_SERVER_SHA256", b"# Exact CPU fixture server\n"),
        "isaaclab_twist2_g1/action_provider/action_provider_sonic.py": ("PINNED_ACTION_PROVIDER_SHA256", b"# Exact CPU fixture controller\n"),
        "lerobot/pyproject.toml": ("PYPROJECT_SHA256", b"[project]\nname='fixture'\n"),
    }
    (source / "lerobot/src").mkdir(parents=True)
    (source / "lerobot/src/__init__.py").write_text("")
    for name, (constant, data) in rows.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        monkeypatch.setattr(stage, constant, hashlib.sha256(data).hexdigest())
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "remote", "add", "origin", stage.SOURCE_REPOSITORY], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "exact source"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(stage, "PINNED_SOURCE_REVISION", revision)
    monkeypatch.setattr(supervisor, "PINNED_SOURCE_REVISION", revision)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"schema_version": "g1_humanoidarena_checkpoint_inventory.v1",
        "source_repository": stage.SOURCE_REPOSITORY, "source_revision": revision}))
    monkeypatch.setattr(stage, "INVENTORY", inventory)
    original = stage.verify_g1_publisher_source(controller)
    (controller / (stage.SCHEMA + ".json")).write_text(json.dumps(original))
    root = tmp_path / "provider"
    shutil.copytree(controller, root / "publisher-source")
    return root, original


def test_actual_source_verification_relocates_only_path_and_derived_digest(tmp_path, monkeypatch):
    root, original = relocated_source(tmp_path, monkeypatch)
    observed = runtime._verify_relocated_publisher(root)
    assert {key for key in original if original[key] != observed[key]} == {"source_root", "receipt_digest"}
    assert observed["source_root"] == str(root / "publisher-source/source")
    assert json.loads((root / "publisher-source" / (stage.SCHEMA + ".json")).read_text()) == original


@pytest.mark.parametrize("mutation", ["stale_digest", "extra_metadata", "revision", "modified_source"])
def test_relocation_cannot_hide_changed_provenance_or_publisher_bytes(tmp_path, monkeypatch, mutation):
    root, original = relocated_source(tmp_path, monkeypatch)
    changed = deepcopy(original)
    if mutation == "modified_source":
        (root / "publisher-source/source/lerobot/scripts/serve_lerobot_vla_http.py").write_text("modified\n")
    else:
        changed["source_revision" if mutation == "revision" else "unexpected_metadata"] = "changed"
        if mutation != "stale_digest":
            changed["receipt_digest"] = canonical_digest(changed, digest_field="receipt_digest")
        (root / "publisher-source" / (stage.SCHEMA + ".json")).write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        runtime._verify_relocated_publisher(root)
