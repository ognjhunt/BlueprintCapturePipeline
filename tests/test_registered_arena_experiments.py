"""ADP-009D/day28: actual authenticated Arena owner-review adoption."""

# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py
#   src/blueprint_pipeline/control_plane_arena_scratch.py
import json
from pathlib import Path

import pytest

from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth
from tests.test_registered_experiment_retirement_flow import retirement_installation  # noqa: F401


def _arena(installation):
    (Path(installation[1]["lane_scratch_inputs_root"]) / "arena").mkdir(mode=0o750)
    return issue(
        installation,
        root="inputs",
        reference_value="arena-launch-r33",
        participant_profile="arena_owner_review.v1",
    )


def test_authentic_arena_birth_binds_fixed_tag_and_current_owner(retirement_installation):
    setup = retirement_installation
    grant = _arena(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    assert target.parent == Path(setup[1]["lane_scratch_inputs_root"]) / "arena"
    lease = json.loads((target / ".lane-scratch.v1.json").read_bytes())
    assert lease["owner"] == "owner" and lease["lane"] == "arena"
    assert lease["run_ref"] == "arena-launch-r33"
    assert lease["class_intent"] == "evidence" and lease["cleanup"] == "owner_review"
    selector = setup[2].parents[1] / "experiment-authority/arena-selection-r33.json"
    selected = json.loads(selector.read_bytes())
    assert selected["tag"] == "r33" and selected["intent_id"] == grant["intent_id"]
    assert selected["birth"] == born["birth"] and selected["generation"] == born["generation"]
    assert selected["target_identity"]["ino"] == target.stat().st_ino


def test_arena_tag_is_consumed_before_another_fresh_identity(retirement_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    setup = retirement_installation
    first = _arena(setup)
    claim = setup[2] / "arena-launch-r33.arena-claim.json"
    original_claim = claim.read_bytes()
    original = root.secrets.token_hex

    def no_new_id(size):
        if size == 16:
            pytest.fail("duplicate canonical Arena tag generated a fresh grant identity")
        return original(size)

    monkeypatch.setattr(root.secrets, "token_hex", no_new_id)
    with pytest.raises(ValueError, match="experiment_arena_tag_claimed"):
        issue(
            setup,
            root="inputs",
            reference_value="arena-launch-r33",
            participant_profile="arena_owner_review.v1",
        )
    assert claim.read_bytes() == original_claim
    assert json.loads(original_claim)["intent_id"] == first["intent_id"]
    assert len(list(setup[2].glob("[0-9a-f]" * 32 + ".json"))) == 1


@pytest.mark.parametrize(
    "reference",
    ["arena-launch-r0", "arena-launch-r00", "arena-launch-r033", "arena-launch-r1000000"],
)
def test_arena_new_authority_rejects_noncanonical_tag_before_grant(
    retirement_installation, reference
):
    with pytest.raises(ValueError, match="experiment_arena_tag_invalid"):
        issue(
            retirement_installation,
            root="inputs",
            reference_value=reference,
            participant_profile="arena_owner_review.v1",
        )
    assert not list(retirement_installation[2].glob("*.arena-claim.json"))


def test_private_arena_claim_recovers_only_original_grant(retirement_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    setup = retirement_installation
    first = _arena(setup)
    intent_path = setup[2] / (first["intent_id"] + ".json")
    original = intent_path.read_bytes()
    intent_path.unlink()  # The durable tag claim preceded an interrupted intent publication.
    monkeypatch.setattr(
        root, "_issue", lambda *args, **kw: pytest.fail("recovery issued a new grant")
    )
    recovered = root.recover_arena_issue(
        "r33", principal="operator", owner="owner", installed_config_path=setup[0], now=lambda: 1001
    )
    assert recovered == first and intent_path.read_bytes() == original


def test_private_arena_claim_expiry_never_reissues(retirement_installation):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    setup = retirement_installation
    _arena(setup)
    with pytest.raises(ValueError, match="experiment_arena_claim_invalid"):
        root.recover_arena_issue(
            "r33",
            principal="operator",
            owner="owner",
            installed_config_path=setup[0],
            now=lambda: 2800,
        )


def test_actual_arena_selector_admits_same_live_target_sh(retirement_installation, monkeypatch):
    import fcntl
    import os
    from blueprint_pipeline import control_plane_arena_scratch as arena
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer

    setup = retirement_installation
    grant = _arena(setup)
    born = birth(setup, grant)
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", setup[2].parents[1] / "experiment-authority")
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    use = arena.admit_registered_arena_attempt("r33", now=lambda: 1102)
    try:
        assert use.path == Path(born["path"]) and use.entry["intent_id"] == grant["intent_id"]
        foreign = os.open(use.path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(foreign, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(foreign)
    finally:
        use.close()


def test_reserved_arena_lease_never_enters_legacy_mutation(retirement_installation):
    from blueprint_pipeline import control_plane_lane_scratch as scratch

    setup = retirement_installation
    grant = _arena(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    original = (target / scratch.LEASE_FILE).read_bytes()
    with pytest.raises(
        scratch.LaneScratchError, match="lane_scratch_registered_authority_required"
    ):
        scratch.renew_lane_scratch(
            lane="arena",
            name=target.name,
            owner="owner",
            expected_digest=json.loads(original)["lease_digest"],
            ttl_seconds=1800,
            root=Path(setup[1]["lane_scratch_inputs_root"]),
            now=lambda: 1002,
        )
    assert (target / scratch.LEASE_FILE).read_bytes() == original


def test_expired_hardlinked_arena_owner_review_is_actual_gc_keep(
    retirement_installation, monkeypatch
):
    import os
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from tests.test_registered_experiment_retirement_flow import _gc

    setup = retirement_installation
    grant = _arena(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    (target / "evidence").write_bytes(b"preserved")
    os.link(target / "evidence", target / "copied-evidence")
    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in target.iterdir()}
    monkeypatch.setattr(
        actions, "_manifest", lambda *a, **kw: pytest.fail("owner review inspected payload")
    )
    monkeypatch.setattr(
        actions, "_hash_manifest", lambda *a, **kw: pytest.fail("owner review hashed payload")
    )
    selected = root.issue_experiment_action_intent(
        grant["intent_id"],
        principal="operator",
        owner="owner",
        action="owner_review",
        expires_at_epoch=3500,
        installed_config_path=setup[0],
        now=lambda: 2900,
    )
    action = json.loads((setup[2] / (selected["action_id"] + ".action.json")).read_bytes())
    assert action["manifest"] is None
    outcome = _gc(setup)["registered_experiments"]["outcomes"][0]
    assert outcome["action_id"] == selected["action_id"] and outcome["decision"] == "kept"
    assert outcome["reason"] == "owner_review" and outcome["removed_logical_bytes"] == 0
    assert {p.name: (p.stat().st_ino, p.read_bytes()) for p in target.iterdir()} == before


@pytest.mark.parametrize("action", ["delete", "offload"])
def test_arena_unknown_descendants_never_gain_removal_authority(retirement_installation, action):
    from blueprint_pipeline import control_plane_lane_experiment_retirement as root

    setup = retirement_installation
    grant = _arena(setup)
    born = birth(setup, grant)
    target = Path(born["path"])
    (target / "payload").write_bytes(b"keep")
    with pytest.raises(ValueError, match="experiment_action_profile_unsupported"):
        root.issue_experiment_action_intent(
            grant["intent_id"],
            principal="operator",
            owner="owner",
            action=action,
            expires_at_epoch=3500,
            installed_config_path=setup[0],
            now=lambda: 2900,
        )
    assert (target / "payload").read_bytes() == b"keep"


def test_actual_arena_payload_uses_same_current_lifetime_without_legacy_lease(
    retirement_installation, monkeypatch
):
    from blueprint_pipeline import control_plane_arena_scratch as arena
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer

    setup = retirement_installation
    grant = _arena(setup)
    born = birth(setup, grant)
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", setup[2].parents[1] / "experiment-authority")
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    use = arena.admit_registered_arena_attempt("r33", now=lambda: 1102)
    monkeypatch.setattr(
        arena.LeasedScratchDirectory,
        "open",
        lambda *a, **kw: pytest.fail("unproved legacy lease fallback"),
    )
    try:
        assert arena.prepare_arena_attempt("r33", _registered_use=use) == Path(born["path"])
        made = arena.mkdir_arena_payload("r33", "arena_packet/nested", _registered_use=use)
        assert made.is_dir() and made == Path(born["path"]) / "arena_packet/nested"
        use.now = lambda: 2800
        with pytest.raises(ValueError, match="experiment_consumer_inactive"):
            arena.mkdir_arena_payload("r33", "after-expiry", _registered_use=use)
        assert not (Path(born["path"]) / "after-expiry").exists()
    finally:
        use.close()


def test_unregistered_arena_new_writer_refuses_caller_owner_as_grant(retirement_installation):
    from blueprint_pipeline import control_plane_arena_scratch as arena

    setup = retirement_installation
    inputs = Path(setup[1]["lane_scratch_inputs_root"]).parent
    (inputs / "lanes/arena").mkdir(mode=0o750)
    with pytest.raises(
        arena.ArenaScratchError, match="arena_scratch_registered_authority_required"
    ):
        arena.prepare_arena_attempt(
            "r33",
            owner="owner",
            run_ref="arena-launch-r33",
            ttl_seconds=1800,
            inputs_root=inputs,
            lane_root=inputs / "lanes",
            now=lambda: 1000,
        )
    assert not (inputs / "lanes/arena/arena-launch-r33").exists()
