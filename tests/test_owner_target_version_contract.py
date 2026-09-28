"""ADP-009D/day28: independently expected versions never manufacture authority."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_target_versions.py

import copy
import json

import pytest

from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def expected():
    directory = dict(dev=1, ino=2, type="directory")
    return dict(
        schema_version="control_plane_lane_expected_target_version.v1", root="work",
        lane="lane", name="cache", root_identity=directory,
        lane_identity=directory | dict(ino=3), folder_identity=directory | dict(ino=4),
        lease_file_identity=dict(dev=1, ino=5, type="regular", mode=0o600,
                                 uid=501, gid=20, nlink=1, size_bytes=400,
                                 mtime_ns=100, ctime_ns=101),
        lease_raw_sha256="sha256:" + "a" * 64, lease_raw_size_bytes=400,
        lease_digest="sha256:" + "b" * 64,
        lease=dict(owner="owner", reference_kind="run_ref", reference_value="run1",
                   reason="cache", class_intent="cache", cleanup="owner_review",
                   consumer_lifetime_contract="leased_scratch_use.v1",
                   created_at_epoch=1000, expires_at_epoch=1100, renewed_at_epoch=1000,
                   released_at_epoch=None, size_budget_bytes=8192))


def parsed(value=None):
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    budget = ReferenceCollectionBudget(monotonic=lambda: 0, values_limit=10_000)
    return t._expected_version(json.dumps(expected() if value is None else value).encode(), budget)


def pair(action="register"):
    decision = dict(path="/mnt/blueprint-work/lanes/lane/cache", owner="owner",
                    action=action, references=["pin"])
    if action == "register":
        decision.update(lane="lane", name="cache", run_ref="run1", reason="cache",
                        class_intent="cache", cleanup="owner_review", ttl_seconds=100,
                        size_budget_bytes=8192)
    elif action == "keep":
        decision["expires_at_epoch"] = 1100
    return dict(decision=decision, census_row=dict(path=decision["path"], references=["pin"]))


def matching(value=None, selected=None, **changes):
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    args = dict(expected=parsed(value), selected=pair() if selected is None else selected,
                policy=dict(owners=["owner"], allowed_actions=["keep", "register"],
                            max_consent_seconds=100), consent_issued=1000,
                consent_expires=1090, expires_at_epoch=1080, now=1001,
                budget=ReferenceCollectionBudget(monotonic=lambda: 0, values_limit=10_000))
    args.update(changes)
    return t._match_intent(**args)


def test_expected_tuple_literal_and_authentic_register_budget():
    assert parsed() == expected()
    result = matching()
    assert result["approved_size_budget_bytes"] == 8192
    assert result["budget_source"] == "protected_register_intent"
    assert result["historical_references"] == ["pin"]
    assert result["execution_authorized"] is result["registration_applied"] is False
    assert result["target_generation_bound"] is result["action_generation_bound"] is False


def test_keep_never_upgrades_lease_budget_to_approval():
    result = matching(selected=pair("keep"))
    assert result["approved_size_budget_bytes"] is None
    assert result["budget_source"] == "lease_metadata_only"
    assert "cache_budget_owner_approval_missing" in result["kept_reasons"]


@pytest.mark.parametrize("change", [
    {"root": "legacy"}, {"lane": "../bad"}, {"name": True}, {"unknown": "PRIVATE"},
    {"lease_raw_size_bytes": True}, {"lease_raw_size_bytes": 2**63},
    {"lease_raw_sha256": "bad"}, {"lease_digest": "sha256:" + "A" * 64},
    {"folder_identity": dict(dev=True, ino=4, type="directory")},
    {"folder_identity": dict(dev=1, ino=2**64, type="directory")},
    {"folder_identity": dict(dev=1, ino=4, type="regular")},
])
def test_expected_top_level_and_identity_refusals(change):
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_expected_invalid"):
        parsed(expected() | change)


@pytest.mark.parametrize(("key", "value"), [
    ("size_budget_bytes", True), ("size_budget_bytes", 0),
    ("created_at_epoch", True), ("expires_at_epoch", 10**400),
    ("released_at_epoch", False), ("renewed_at_epoch", 999),
    ("reference_kind", "path"), ("consumer_lifetime_contract", None),
    ("owner", "bad owner"), ("reason", "PRIVATE\n"),
])
def test_expected_projection_refusals(key, value):
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    data = expected()
    data["lease"][key] = value
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_expected_invalid"):
        parsed(data)


def test_duplicate_and_unknown_nested_fields_refuse():
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    data = expected()
    data["lease"]["extra"] = "PRIVATE"
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_expected_invalid"):
        parsed(data)
    raw = json.dumps(expected()).replace('"root": "work"', '"root": "work", "root": "work"')
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_expected_invalid"):
        t._expected_version(raw.encode(), ReferenceCollectionBudget(monotonic=lambda: 0))


@pytest.mark.parametrize(("field", "value"), [
    ("owner", "other"), ("lane", "other"), ("name", "other"), ("run_ref", "other"),
    ("reason", "other"), ("class_intent", "scratch"), ("cleanup", "delete"),
    ("size_budget_bytes", 8193), ("path", "/mnt/blueprint-work/lanes/lane/other"),
])
def test_selected_approved_metadata_cannot_be_substituted(field, value):
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    selected = copy.deepcopy(pair())
    selected["decision"][field] = value
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_intent_mismatch"):
        matching(selected=selected)


@pytest.mark.parametrize("changes", [
    dict(policy=dict(owners=["other"], allowed_actions=["register"], max_consent_seconds=100)),
    dict(expires_at_epoch=True), dict(expires_at_epoch=1091), dict(now=1100),
    dict(expires_at_epoch=1000),
])
def test_current_policy_and_all_expiry_ceilings_refuse(changes):
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    with pytest.raises(t.OwnerTargetVersionError):
        matching(**changes)


def test_inactive_and_unsupported_intent_cannot_gain_approval():
    from blueprint_pipeline import control_plane_lane_owner_target_versions as t
    data = expected()
    data["lease"]["released_at_epoch"] = 1001
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_inactive"):
        matching(value=data)
    with pytest.raises(t.OwnerTargetVersionError, match="owner_target_intent_unsupported"):
        matching(selected=pair("delete"))
