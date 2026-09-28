"""Actual root-issued birth precedes current authority and producer payload."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_scratch.py

import json
import os
from pathlib import Path

import pytest

from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401


def prepare(installation):  # noqa: F811
    _, settings, store, _ = installation
    state = store.parents[1]
    public = state / "experiment-authority"
    public.mkdir(mode=0o750)
    (public / ".authority.lock").write_bytes(b"")
    (public / ".authority.lock").chmod(0o640)
    for root in (settings["lane_scratch_work_root"], settings["lane_scratch_inputs_root"]):
        path = Path(root)
        path.mkdir(parents=True, mode=0o750)
        (path / ".lane-scratch.lock").write_bytes(b"")
        (path / ".lane-scratch.lock").chmod(0o600)
        (path / "g1").mkdir(mode=0o750)
    return public


def birth(installation, result, **options):  # noqa: F811
    from blueprint_pipeline.control_plane_lane_experiment_birth import create_registered_experiment
    return create_registered_experiment(result["intent_id"], expected_intent=result["intent"],
        installed_config_path=installation[0], now=options.pop("now", lambda: 1100), **options)


def test_root_birth_uses_actual_native_constructor_and_current_authority(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_scratch as native
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    public = prepare(installation)
    grant = issue(installation)
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)
    original = native.create_lane_scratch
    calls = []
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    monkeypatch.setattr(native, "create_lane_scratch", constructor)
    result = birth(installation, grant)
    target = Path(installation[1]["lane_scratch_work_root"]) / "g1" / ("registered-" + grant["intent_id"])
    assert result["path"] == str(target) and len(calls) == 1
    assert calls[0][0][1].startswith("create-")
    lease = json.loads((target / native.LEASE_FILE).read_bytes())
    assert lease["name"] == target.name and lease["owner"] == "owner"
    assert lease["class_intent"] == "scratch" and lease["cleanup"] == "delete"
    assert lease["expires_at_epoch"] == 2800
    marker = json.loads((target / ".registered-experiment.v1.json").read_bytes())
    assert marker["generation"] == result["generation"]
    head = json.loads((public / "HEAD.json").read_bytes())
    authority = json.loads((public / head["record_name"]).read_bytes())
    entry = authority["enrollments"][0]
    assert entry["intent_id"] == grant["intent_id"] and entry["state"] == "active"
    assert entry["completion"] is entry["restoration"] is None
    assert entry["target_identity"]["ino"] == target.stat().st_ino
    assert len(list(target.iterdir())) == 2  # The actual target inode carries the lifetime flock.


def test_birth_reissue_refuses_prior_claim_before_native_constructor(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_scratch as native
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    prepare(installation)
    grant = issue(installation)
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)
    first = birth(installation, grant)
    monkeypatch.setattr(native, "create_lane_scratch", lambda *a, **kw: pytest.fail("reissued birth"))
    with pytest.raises(ValueError, match="experiment_creation_already_claimed"):
        birth(installation, grant)
    assert Path(first["path"]).is_dir()


@pytest.mark.parametrize("drift", ["intent_selector", "expired", "policy_revoke"])
def test_birth_current_authority_checks_precede_claim_or_folder(installation, monkeypatch, drift):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    prepare(installation)
    grant = issue(installation)
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    if drift == "intent_selector":
        grant["intent"]["sha256"] = "sha256:" + "0" * 64
    if drift == "policy_revoke":
        path = installation[3]
        policy = json.loads(path.read_bytes())
        policy["enabled"] = False
        path.write_text(json.dumps(policy))
    options = {"now": lambda: 2900} if drift == "expired" else {}
    with pytest.raises(ValueError):
        birth(installation, grant, **options)
    assert not list(installation[2].glob("*.claim.json"))
    root = Path(installation[1]["lane_scratch_work_root"]) / "g1"
    assert list(root.iterdir()) == []


def test_second_actual_issuance_and_birth_updates_current_head_without_losing_first(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    public = prepare(installation)
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)
    first = issue(installation)
    born_first = birth(installation, first)
    original = (public / "HEAD.json").read_bytes()
    second = issue(installation)
    born_second = birth(installation, second)
    head = json.loads((public / "HEAD.json").read_bytes())
    old_head = json.loads(original)
    authority = json.loads((public / head["record_name"]).read_bytes())
    assert head["version"] == 1 and head["authority_epoch_id"] == old_head["authority_epoch_id"]
    assert authority["previous_record"] == old_head["record"]
    assert {row["intent_id"] for row in authority["enrollments"]} == {first["intent_id"], second["intent_id"]}
    assert all(row["state"] == "active" for row in authority["enrollments"])
    assert Path(born_first["path"]).is_dir() and Path(born_second["path"]).is_dir()
    assert (public / old_head["record_name"]).is_file()


def test_missing_existing_head_never_bootstraps_a_new_authority(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    public = prepare(installation)
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)
    first = issue(installation)
    birth(installation, first)
    (public / "HEAD.json").unlink()
    second = issue(installation)
    with pytest.raises(ValueError, match="experiment_authority_missing"):
        birth(installation, second)
    assert not (installation[2] / (second["intent_id"] + ".claim.json")).exists()


def test_current_head_substitution_preserves_foreign_bytes_before_cas(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    from blueprint_pipeline import control_plane_lane_experiment_publication as publication
    public = prepare(installation)
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: None)
    birth(installation, issue(installation))
    second = issue(installation)
    original = publication._guard
    changed = False
    def guard(files, state, **kwargs):
        nonlocal changed
        if state.previous is not None and not changed:
            foreign = public / "foreign.json"
            foreign.write_bytes(b"FOREIGN")
            foreign.chmod(0o640)
            os.rename(foreign, public / "HEAD.json")
            changed = True
        return original(files, state, **kwargs)
    monkeypatch.setattr(publication, "_guard", guard)
    with pytest.raises(ValueError, match="experiment_head_changed"):
        birth(installation, second)
    assert changed and (public / "HEAD.json").read_bytes() == b"FOREIGN"


def test_legacy_directory_handle_cannot_read_registered_authority_by_lease_only(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_birth as root
    from blueprint_pipeline.control_plane_leased_scratch import LeasedScratchDirectory
    prepare(installation)
    grant = issue(installation)
    monkeypatch.setattr(root, '_blueprint_identity', lambda:(0,0))
    monkeypatch.setattr(os, 'fchown', lambda *a:None)
    born = birth(installation, grant)
    with pytest.raises(ValueError, match='lane_scratch_registered_authority_required'):
        LeasedScratchDirectory.open(lane='g1', name=Path(born['path']).name, owner='owner',
            run_ref='run1', root=installation[1]['lane_scratch_work_root'], now=lambda:1200)
