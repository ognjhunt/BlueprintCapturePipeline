import base64
import json
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_revoked_bundle_retention as retention
from blueprint_pipeline.task_evaluation_scene_intake import _seal


def _json(path, value, field):
    path.write_text(json.dumps(_seal(value, field)))


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(retention, "_not_open", lambda path: None)
    old_id, new_id = "scene-" + "1" * 64, "scene-" + "2" * 64
    old, new = root / "old-intent", root / "new-intent"
    old.mkdir()
    new.mkdir()
    request = {
        "owner": {"user_id": "owner", "organization_id": "org"},
        "source": {"content_digest": "source"},
        "task": {"subject": "book"},
    }
    for directory, identity in [(old, old_id), (new, new_id)]:
        _json(
            directory / "intent.json", {"intent_id": identity, "request": request}, "intent_digest"
        )
    intent = json.loads((old / "intent.json").read_text())
    _json(
        old / "revoked.json",
        {"status": "revoked", "intent_digest": intent["intent_digest"], "owner": request["owner"]},
        "receipt_digest",
    )
    launch = root / (old_id[:26] + "-launch")
    launch.mkdir()
    (launch / "launch_receipt.json").write_text('{"status":"blocked"}')
    (launch / "post_teardown_provider_zero_receipt.json").write_text(
        '{"status":"provider_zero_confirmed","blockers":[]}'
    )
    activation = root / "activations"
    paths = []
    for identity, unique in [(old_id, b"original unique input"), (new_id, b"new code")]:
        path = (
            activation / (identity[:26] + "-attempt") / "launch-set/bundle" / retention.BUNDLE_NAME
        )
        path.parent.mkdir(parents=True)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("input/shared.bin", b"source-observation" * 100)
            archive.writestr("input/unique.json", unique)
        paths.append(path)
    return dict(
        obsolete=paths[0],
        retained=paths[1],
        activation_root=activation,
        revoked_intent_root=old,
        successor_intent_root=new,
        terminal_launch_root=launch,
        destination=root / "plan.json",
    )


def test_reclaims_only_packaging_and_recovers_every_member(case):
    with zipfile.ZipFile(case["obsolete"]) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    plan = retention.plan_retention(**case)
    assert case["obsolete"].is_file()
    with zipfile.ZipFile(case["retained"]) as archive:
        recovered = {
            m["name"]: archive.read(m["name"])
            if m["retained_identical_member"]
            else base64.b64decode(m["preserved_base64"])
            for m in plan["members"]
        }
    assert recovered == original
    assert not plan["original_zip_container_identity_recoverable"]
    receipt = retention.apply_retention(
        plan_path=case["destination"],
        receipt_path=case["destination"].with_name("receipt.json"),
        acknowledgement=retention.ACK,
    )
    assert receipt["status"] == "reclaimed"
    assert not case["obsolete"].exists() and case["retained"].is_file()
    assert (case["revoked_intent_root"] / "revoked.json").is_file()


def test_changed_successor_refuses_deletion(case):
    retention.plan_retention(**case)
    case["retained"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="input_changed"):
        retention.apply_retention(
            plan_path=case["destination"],
            receipt_path=case["destination"].with_name("receipt.json"),
            acknowledgement=retention.ACK,
        )
    assert case["obsolete"].is_file()


@pytest.mark.parametrize("failure", ["not_closed", "owner", "open"])
def test_unsafe_scope_refuses_plan(case, monkeypatch, failure):
    if failure == "not_closed":
        (case["terminal_launch_root"] / "launch_receipt.json").write_text('{"status":"running"}')
    elif failure == "owner":
        path = case["successor_intent_root"] / "intent.json"
        value = json.loads(path.read_text())
        value["request"]["owner"]["user_id"] = "someone_else"
        _json(path, value, "intent_digest")
    else:

        def refuse(path):
            raise ValueError("open_reader")

        monkeypatch.setattr(retention, "_not_open", refuse)
    with pytest.raises(ValueError):
        retention.plan_retention(**case)
    assert case["obsolete"].exists() and not case["destination"].exists()
