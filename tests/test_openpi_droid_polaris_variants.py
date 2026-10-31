"""Public OpenPI DROID checkpoints beside pi05_droid, on its unchanged runtime path.

The three PolaRiS joint-position variants are the remaining entries of the
frozen GCS inventory pi05_droid already pins, served by the same pinned OpenPI
source, environment, server and client. Every other public DROID checkpoint
stays visible with the exact reason that path cannot serve it.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_policy_candidate_admission import EXPECTED_CANDIDATES
from blueprint_pipeline.adp009d_policy_provisioning import PolicyProvisioningError, build_provisioning_script
from blueprint_pipeline.adp009d_policy_server_worker import (
    CANDIDATE_TRANSPORTS,
    TRANSPORT_OPENPI_WEBSOCKET,
    build_serve_command,
)
from blueprint_pipeline.beta_policy_candidate_catalog import catalog_candidate, load_default_catalog
from blueprint_pipeline.beta_policy_screening_capabilities import _interface_matches
from blueprint_pipeline.franka_inspection_learned_policy_lane import OFFICIAL_OPENPI_DROID_BASELINES
from blueprint_pipeline.native_task_arena_policy_bundle import (
    OPENPI_CHECKPOINT_INVENTORY_PATH,
    _candidate_runtime_binding,
)
from blueprint_pipeline.native_task_runtime_contract import FROZEN_CANDIDATES
from blueprint_pipeline.openpi_droid_policy_runtime import (
    ARENA_CANDIDATE_POLICY_IDS,
    BETA_POLARIS_VARIANT_CANDIDATE_IDS,
    OPENPI_POLICY_ACTION_CHUNK_ROWS,
    OPENPI_SOURCE_REVISION,
    canonical_sha256,
    validate_beta_polaris_variant_binding,
    validate_beta_polaris_variant_execution_spec,
)

COHORT = (Path(__file__).resolve().parents[1]
          / "docs/experiments/policy_ranking_thesis_20260726/warehouse_policy_cohort_v2_joint_position.json")
VELOCITY = "openpi_droid_runtime_action_space_unsupported:joint_velocity"
PREFIX = "openpi_droid_runtime_requires_polaris_checkpoint_prefix"
NOT_FROZEN = "checkpoint_not_in_frozen_openpi_polaris_inventory"
NO_CONFIG = "openpi_train_config_absent_at_pinned_revision=" + OPENPI_SOURCE_REVISION
HELD = {
    "pi0_droid": ("checkpoints/pi0_droid", [VELOCITY, PREFIX, NOT_FROZEN]),
    "pi0_fast_droid": ("checkpoints/pi0_fast_droid", [VELOCITY, PREFIX, NOT_FROZEN]),
    "pi0_droid_jointpos": ("checkpoints/pi0_droid_jointpos", [PREFIX, NOT_FROZEN, NO_CONFIG]),
    "pi0_droid_jointpos_100k": ("checkpoints/pi0_droid_jointpos_100k", [PREFIX, NOT_FROZEN, NO_CONFIG]),
    "pi0_fast_droid_jointpos": ("checkpoints/pi0_fast_droid_jointpos", [PREFIX, NOT_FROZEN, NO_CONFIG]),
    "pi05_droid_jointpos": ("checkpoints/pi05_droid_jointpos", [PREFIX, NOT_FROZEN, NO_CONFIG]),
    "paligemma_binning_droid_jointpos": ("checkpoints/paligemma_binning_droid_jointpos", [PREFIX, NOT_FROZEN]),
    "paligemma_binning_droid_jointpos_polaris": (
        "checkpoints/polaris/paligemma_binning_droid_jointpos_polaris",
        [NOT_FROZEN, "openpi_config_checkpoint_pairing_unstated:paligemma_binning_droid_jointpos"]),
    **{name: ("checkpoints/roboarena/" + name, [VELOCITY, PREFIX, NOT_FROZEN]) for name in (
        "paligemma_binning_droid", "paligemma_fast_droid", "paligemma_fast_specialist_droid",
        "paligemma_vq_droid", "paligemma_diffusion_droid")},
}


def _inventory() -> dict:
    return json.loads(OPENPI_CHECKPOINT_INVENTORY_PATH.read_text(encoding="utf-8"))


def _entry(policy_id: str) -> dict:
    [entry] = [row for row in _inventory()["entries"] if row["policy_id"] == policy_id]
    return entry


def test_variants_are_the_rest_of_pi05s_frozen_inventory_and_never_adp_candidates():
    inventory = _inventory()
    payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    assert inventory["status"] == "frozen" and not inventory["blockers"]
    assert inventory["inventory_sha256"] == canonical_sha256(payload)
    assert inventory["openpi_revision"] == OPENPI_SOURCE_REVISION
    assert ARENA_CANDIDATE_POLICY_IDS == {"pi05_droid": "pi05_droid_jointpos_polaris",
                                          **{variant: variant for variant in BETA_POLARIS_VARIANT_CANDIDATE_IDS}}
    assert {row["policy_id"] for row in inventory["entries"]} == set(ARENA_CANDIDATE_POLICY_IDS.values())
    cohort = json.loads(COHORT.read_text(encoding="utf-8"))
    assert cohort["checkpoint_inventory"]["inventory_sha256"] == inventory["inventory_sha256"]
    assert {row["policy_id"]: row["action_horizon"] for row in cohort["primary_cohort"]} == OPENPI_POLICY_ACTION_CHUNK_ROWS
    assert not set(BETA_POLARIS_VARIANT_CANDIDATE_IDS) & (set(EXPECTED_CANDIDATES) | set(FROZEN_CANDIDATES))
    assert {key for key, transport in CANDIDATE_TRANSPORTS.items()
            if transport == TRANSPORT_OPENPI_WEBSOCKET} == {
        "pi05_droid", "cosmos3_edge_policy_droid", *BETA_POLARIS_VARIANT_CANDIDATE_IDS}


def test_enabled_rows_reuse_pi05_identity_license_runtime_and_interface_exactly():
    catalog = load_default_catalog()
    pi05 = catalog_candidate(catalog, "pi05_droid")
    for variant in BETA_POLARIS_VARIANT_CANDIDATE_IDS:
        row = catalog_candidate(catalog, variant)
        assert row["policy"] == {**pi05["policy"], "source": _entry(variant)["checkpoint_uri"]}
        assert row["policy"]["source"] == "gs://openpi-assets/checkpoints/polaris/" + variant
        assert row["eligibility"] == pi05["eligibility"]
        assert row["runtime"] == pi05["runtime"]
        assert row["interface"] == {**pi05["interface"], "action_horizon": OPENPI_POLICY_ACTION_CHUNK_ROWS[variant]}
        assert row["supported_task_kinds"] == pi05["supported_task_kinds"]
        assert "deployment" not in row


def test_unservable_public_droid_checkpoints_stay_visible_with_exact_reasons():
    catalog = load_default_catalog()
    pi05 = catalog_candidate(catalog, "pi05_droid")
    openpi_rows = {row["candidate_id"] for row in catalog["candidates"]
                   if row["policy"]["source"].startswith("gs://openpi-assets/")}
    assert openpi_rows == {"pi05_droid", *BETA_POLARIS_VARIANT_CANDIDATE_IDS, *HELD}
    roboarena = {row["checkpoint_uri"]: row["checkpoint_generation_manifest_sha256"]
                 for row in OFFICIAL_OPENPI_DROID_BASELINES}
    for key, (path, blockers) in HELD.items():
        row = catalog_candidate(catalog, key)
        assert row["policy"]["source"] == "gs://openpi-assets/" + path
        assert row["eligibility"] == {**pi05["eligibility"], "enabled": False, "blockers": blockers}
        assert row["interface"] is None and row["runtime"] is None and row["supported_task_kinds"] == []
        generation = row["policy"]["digest"].removeprefix("sha256:")
        assert row["policy"]["digest_kind"] == "hub_tree_listing"
        assert row["policy"]["revision"] == "gcs-generation-manifest:" + generation
        if row["policy"]["source"] in roboarena:
            # The same listing digest the inspection lane pinned on 2026-08-03.
            assert generation == roboarena[row["policy"]["source"]]


@pytest.mark.parametrize("variant", BETA_POLARIS_VARIANT_CANDIDATE_IDS)
def test_runtime_binding_serves_each_variant_from_its_frozen_entry(variant):
    pi05_policy, pi05_endpoint, pi05_identity = _candidate_runtime_binding("pi05_droid")
    policy, endpoint, identity = _candidate_runtime_binding(variant)
    entry = _entry(variant)
    assert (policy.policy_id, policy.config_name, policy.checkpoint_uri) == (variant, variant, entry["checkpoint_uri"])
    assert (policy.checkpoint_object_count, policy.checkpoint_size_bytes) == (entry["object_count"], entry["size_bytes"])
    assert policy.checkpoint_object_manifest_sha256 == entry["legacy_object_manifest_sha256"]
    assert policy.checkpoint_generation_manifest_sha256 == entry["generation_manifest_sha256"]
    assert policy.checkpoint_inventory_sha256 == pi05_policy.checkpoint_inventory_sha256
    assert policy.action_chunk_rows == 10 and policy.open_loop_horizon == pi05_policy.open_loop_horizon == 8
    assert (endpoint, identity) == (pi05_endpoint, pi05_identity)
    validate_beta_polaris_variant_binding(candidate_id=variant, spec=policy)
    with pytest.raises(ValueError, match="action_chunk_rows_mismatch"):
        validate_beta_polaris_variant_binding(candidate_id=variant, spec=replace(policy, action_chunk_rows=15))
    with pytest.raises(ValueError, match="variant_unknown"):
        validate_beta_polaris_variant_binding(candidate_id="pi05_droid", spec=pi05_policy)
    with pytest.raises(ValueError, match="candidate_mismatch"):
        validate_beta_polaris_variant_binding(candidate_id=variant, spec=pi05_policy)


def test_variant_execution_spec_must_be_the_local_openpi_robolab_interface():
    policy, _, _ = _candidate_runtime_binding("pi0_fast_droid_jointpos_polaris")
    spec = {"candidate_id": "pi0_fast_droid_jointpos_polaris", "policy_interface_id": "openpi_droid.v1",
            "evaluation_backend": "robolab.v1", "policy_spec": asdict(policy), "open_loop_horizon": 8}
    assert validate_beta_polaris_variant_execution_spec(spec) == policy
    for change, code in ((("policy_interface_id", "openpi_droid_https.v1"), "interface_invalid"),
                         (("evaluation_backend", "task_family_native.v1"), "interface_invalid"),
                         (("open_loop_horizon", 10), "open_loop_horizon_mismatch"),
                         (("policy_spec", {**asdict(policy), "command": "x"}), "policy_spec_invalid"),
                         (("candidate_id", "pi0_droid_jointpos_polaris"), "candidate_mismatch")):
        with pytest.raises(ValueError, match=code):
            validate_beta_polaris_variant_execution_spec({**spec, change[0]: change[1]})


@pytest.mark.parametrize("stop_after_handshake", [False, True])
def test_provisioning_is_pi05s_script_with_only_the_candidate_and_checkpoint_changed(stop_after_handshake):
    pi05_uri = _entry("pi05_droid_jointpos_polaris")["checkpoint_uri"]
    pi05 = build_provisioning_script("pi05_droid", stop_after_handshake=stop_after_handshake)
    assert pi05.count(pi05_uri) == 1
    for variant in BETA_POLARIS_VARIANT_CANDIDATE_IDS:
        expected = (pi05.replace(pi05_uri, "@checkpoint@").replace("pi05_droid", variant)
                    .replace("@checkpoint@", _entry(variant)["checkpoint_uri"]))
        assert build_provisioning_script(variant, stop_after_handshake=stop_after_handshake) == expected
    with pytest.raises(PolicyProvisioningError):
        build_provisioning_script("pi0_droid")


def test_policy_server_uses_the_identity_bound_openpi_server_for_variants():
    command = build_serve_command(candidate_id="pi0_droid_jointpos_100k_polaris", python="/venv/bin/python",
                                  source_root="/src", checkpoint_root="/ckpt", port=8000, policy_spec_path="/spec.json",
                                  checkpoint_inventory_path="/inventory.json", runtime_dir="/runtime")
    assert command == ["/venv/bin/python", "/runtime/openpi_droid_policy_runtime.py", "--policy-spec", "/spec.json",
                       "--checkpoint-dir", "/ckpt", "--checkpoint-inventory", "/inventory.json",
                       "--host", "127.0.0.1", "--port", "8000"]


def test_capability_admits_variants_only_on_the_local_openpi_interface_and_exact_horizon():
    catalog = load_default_catalog()
    row = catalog_candidate(catalog, "pi0_droid_jointpos_polaris")
    policy, _, _ = _candidate_runtime_binding("pi0_droid_jointpos_polaris")
    spec = {"candidate_id": "pi0_droid_jointpos_polaris", "policy_interface_id": "openpi_droid.v1",
            "open_loop_horizon": 8, "policy_spec": asdict(policy)}
    assert _interface_matches(row, spec)
    assert not _interface_matches({**row, "interface": {**row["interface"], "action_horizon": 15}}, spec)
    assert not _interface_matches(row, {**spec, "candidate_id": "pi0_droid"})
    assert not _interface_matches(row, {**spec, "policy_interface_id": "groot_n17_droid.v1"})
