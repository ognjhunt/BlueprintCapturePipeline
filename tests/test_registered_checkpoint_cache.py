"""ADP-009D/day28: authenticated needed-cache birth, held reads and staging.

Tiny real ALLFOUR inventories exercise the actual fetcher/verifier/WAM paths.
Mac metadata fixtures are not the required foreign-UID Linux acceptance proof.
"""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_registered_checkpoint_cache.py
#   scripts/fetch_g1_humanoidarena_checkpoint.py
#   src/blueprint_pipeline/native_g1_checkpoint_cache.py
#   src/blueprint_pipeline/wam_provider_object_store.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from blueprint_pipeline.native_g1_development_pair import PAIR_ORDER


def raw_ref(path):
    raw = Path(path).read_bytes()
    return {"sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def tiny_inventory():
    rows, payloads = [], {}
    for ci, candidate in enumerate(PAIR_ORDER):
        files = []
        for index in range(6):
            data = f"candidate {ci} file {index}\n".encode()
            name = f"file-{index}.bin"
            files.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)))
            payloads[f"candidate-{ci}/{name}"] = data
        rows.append(dict(candidate_id=candidate, subdirectory=f"candidate-{ci}", files=files,
                         inventory_digest="sha256:" + hashlib.sha256(json.dumps(
                             files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()))
    return dict(schema_version="g1_humanoidarena_checkpoint_inventory.v1", candidates=rows), payloads


@pytest.fixture
def cache_installation(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    config, settings, _, policy = installation
    state = Path(settings["state_root"])
    private = state / "requests/needed-checkpoint-cache-records"
    public = state / "needed-checkpoint-cache-registration"
    authority = state / "needed-checkpoint-cache-authority"
    for path, mode in ((private, 0o700), (public, 0o755), (authority, 0o750)):
        path.mkdir(mode=mode)
    for path, name, mode in ((private, ".cache-store.lock", 0o600),
                             (authority, ".authority.lock", 0o640)):
        (path / name).write_bytes(b"")
        (path / name).chmod(mode)
    root = Path(settings["lane_scratch_work_root"])
    root.mkdir(parents=True, mode=0o750)
    (root / ".lane-scratch.lock").write_bytes(b"")
    (root / ".lane-scratch.lock").chmod(0o600)
    (root / "g1-checkpoint").mkdir(mode=0o750)
    inventory, payloads = tiny_inventory()
    inventory_path = config.parent / "installed-inventory.json"
    inventory_path.write_bytes(encoded(inventory))
    inventory_path.chmod(0o600)
    settings.update(needed_checkpoint_cache_creation_enabled=True,
                    needed_checkpoint_cache_inventory_file=str(inventory_path))
    config.write_bytes(encoded(settings))
    monkeypatch.setattr(cache, "_blueprint_identity", lambda: (0, 0))
    return dict(config=config, settings=settings, private=private, public=public,
                authority=authority, inventory_path=inventory_path, payloads=payloads, policy=policy)


def issue_cache(value, **changes):
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    options = dict(principal="operator", owner="owner", name="needed-models", reference_kind="run_ref",
                   reference_value="run1", lease_ttl_seconds=1800, size_budget_bytes=8 * 1024 * 1024,
                   inventory_raw_sha256=raw_ref(value["inventory_path"])["sha256"],
                   inventory_raw_size_bytes=raw_ref(value["inventory_path"])["size_bytes"],
                   installed_config_path=value["config"], now=lambda: 1000)
    return cache.issue_needed_checkpoint_cache_intent(**(options | changes))


def test_actual_inventory_all_four_resource_shape_without_large_payload():
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    inventory = json.loads((Path(__file__).parents[1] /
                            "configs/g1_humanoidarena_checkpoint_inventory.v1.json").read_bytes())
    fill = cache.derive_checkpoint_flow_resources(inventory, "fill")
    hit = cache.derive_checkpoint_flow_resources(inventory, "stage_hit")
    miss = cache.derive_checkpoint_flow_resources(inventory, "stage_miss")
    assert fill["file_count"] == 24 and fill["logical_bytes"] == 20_810_319_953
    assert fill["quantum_count"] == 19_868
    assert fill["payload_windows"] == miss["payload_windows"] == 59_604
    assert hit["payload_windows"] == 39_736
    assert miss["part_count"] == 2504 and fill["range_count"] == 156
    assert fill["maximum_checks"] == 499_120 and miss["maximum_checks"] == 504_128
    # Reusing one complete candidate does not silently skip the fetcher's second hash.
    present = {row["subdirectory"] + "/" + f["path"] for row in inventory["candidates"][:1]
               for f in row["files"]}
    resume = cache.derive_checkpoint_flow_resources(inventory, "resume", present_paths=present)
    assert resume["payload_windows"] == 2 * 1008 + 3 * (19868 - 1008)


def test_root_issue_is_real_private_grant_and_no_payload_birth(cache_installation):
    value = cache_installation
    grant = issue_cache(value)
    path = value["private"] / (grant["intent_id"] + ".json")
    intent = json.loads(path.read_bytes())
    assert raw_ref(path) == grant["intent"]
    assert intent["schema_version"] == "control_plane_needed_cache_creation_intent.v1"
    assert intent["issuer_uid"] == 0 and intent["principal"] == "operator"
    assert intent["generation"] != intent["intent_id"]
    assert intent["class_intent"] == "cache" and intent["cleanup"] == "owner_review"
    assert intent["size_budget_bytes"] == 8 * 1024 * 1024
    assert list((Path(value["settings"]["lane_scratch_work_root"]) / "g1-checkpoint").iterdir()) == []


@pytest.mark.parametrize("refusal", ["disabled", "unmapped_owner", "wrong_inventory", "budget_bool"])
def test_creation_refusals_precede_claim_directory_or_network(cache_installation, refusal):
    value, changes = cache_installation, {}
    if refusal == "disabled":
        value["config"].write_bytes(encoded(value["settings"] |
            {"needed_checkpoint_cache_creation_enabled": False}))
    elif refusal == "unmapped_owner":
        changes["owner"] = "different-owner"
    elif refusal == "wrong_inventory":
        changes["inventory_raw_sha256"] = "sha256:" + "0" * 64
    else:
        changes["size_budget_bytes"] = True
    with pytest.raises(ValueError):
        issue_cache(value, **changes)
    assert list(value["private"].glob("*.json")) == []


@pytest.mark.parametrize("entrypoint", ["fetcher", "verify", "wam_hash", "wam_stage"])
def test_missing_authority_in_reserved_namespace_never_falls_back(
        tmp_path, monkeypatch, entrypoint):
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import native_g1_checkpoint_cache as native
    from blueprint_pipeline import wam_provider_object_store as wam
    reserved = Path("/mnt/blueprint-work/lanes/g1-checkpoint/unregistered")
    monkeypatch.setattr(wam, "_read_first_file", lambda **kw: pytest.fail("credentials before authority"))
    if entrypoint == "fetcher":
        with pytest.raises(ValueError, match="needed_cache"):
            native._fetcher().materialize_candidate(inventory_path=tmp_path / "missing.json",
                candidate_id=PAIR_ORDER[0], output_dir=reserved)
    elif entrypoint == "verify":
        with pytest.raises(ValueError, match="needed_cache"):
            native.verify_local_g1_checkpoint_cache(reserved)
    elif entrypoint == "wam_hash":
        with pytest.raises(ValueError, match="needed_cache"):
            wam._sha256_file(reserved / "payload.bin")
    else:
        with pytest.raises(ValueError, match="needed_cache"):
            wam.stage_cached_runtime_dependency_object_store(job_dir=tmp_path / "job",
                dependency_path=reserved / "payload.bin", expected_sha256="sha256:" + "a" * 64,
                key_prefix="fixture", expiration_seconds=60)
        assert not (tmp_path / "job").exists()
    assert cache.is_registered_checkpoint_path(reserved)


def test_wrong_private_authority_rejected_without_callbacks(tmp_path):
    from blueprint_pipeline import native_g1_checkpoint_cache as native
    from blueprint_pipeline import wam_provider_object_store as wam
    calls = []
    class Lookalike:
        def check(self, *args, **kwargs):
            calls.append("check")
    with pytest.raises(ValueError, match="needed_cache"):
        native.verify_local_g1_checkpoint_cache(tmp_path, _cache_use=Lookalike())
    with pytest.raises(ValueError, match="needed_cache"):
        wam._sha256_file(tmp_path / "missing", _cache_use=Lookalike())
    assert calls == []
