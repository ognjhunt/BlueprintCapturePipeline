"""Private beta model delivery consumes only owner-pinned, byte-exact artifacts."""
# Covers: src/blueprint_pipeline/team_robot_model_delivery.py
# Covers: src/blueprint_pipeline/team_candidate_quick10_launch.py
# Covers: src/blueprint_pipeline/team_candidate_quick10_registry.py
import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from blueprint_pipeline import team_robot_model_delivery as delivery


@pytest.fixture
def selected(tmp_path):
    raw = b'#usda 1.0\ndef Xform "Robot" {}\n'
    row = {"team_id": "team-one", "checkpoint_id": "checkpoint-one",
           "model_url": "https://storage.googleapis.com/approved-bucket/team-models/robot.usd",
           "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
           "uri": "gs://approved-bucket/team-models/robot.usd", "generation": "12345",
           "size_bytes": len(raw)}
    return SimpleNamespace(raw=raw, row=row, target=tmp_path / "robot.usd",
        manifest={"model": {"format": "usd", "url": row["model_url"], "sha256": row["sha256"]}},
        profile={"private_robot_models": {"canonical_uri_prefixes": ["gs://approved-bucket/team-models/"],
                                         "artifacts": [row]}})


def run(selected, client):
    return delivery.stage_private_robot_model(profile=selected.profile, manifest=selected.manifest,
        team_id="team-one", checkpoint_id="checkpoint-one", target=selected.target, storage_client=client)


class Client:
    def __init__(self, selected, *, body=None, generation=None, size=None, error=None):
        self.selected, self.calls = selected, []
        self.body = selected.raw if body is None else body
        self.generation = generation if generation is not None else selected.row["generation"]
        self.size = selected.row["size_bytes"] if size is None else size
        self.error = error

    def bucket(self, name):
        self.calls.append(("bucket", name))
        return self

    def blob(self, name, **kwargs):
        self.calls.append(("blob", name, kwargs))
        return self

    def reload(self, **kwargs):
        self.calls.append(("reload", kwargs))
        if self.error:
            raise self.error

    def download_to_file(self, writer, **kwargs):
        self.calls.append(("download", kwargs))
        for offset in range(0, len(self.body), 5):
            writer.write(self.body[offset:offset + 5])


def test_exact_generation_download_keeps_private_bytes_and_provenance(selected):
    client = Client(selected)
    receipt = run(selected, client)
    assert selected.target.read_bytes() == selected.raw
    assert selected.target.stat().st_mode & 0o777 == 0o600
    assert client.calls == [
        ("bucket", "approved-bucket"), ("blob", "team-models/robot.usd", {"generation": 12345}),
        ("reload", {"if_generation_match": 12345, "timeout": 30}),
        ("download", {"if_generation_match": 12345, "timeout": 300, "checksum": "auto", "raw_download": True})]
    assert all(receipt[key] == value for key, value in selected.row.items())
    assert receipt["public_access_required"] is False


@pytest.mark.parametrize("field,value", [
    ("uri", "gs://another-bucket/team-models/robot.usd"),
    ("uri", "gs://approved-bucket/team-models/../other/robot.usd"),
    ("uri", "gs://approved-bucket/team-models/robot.usd?token=SECRET"),
    ("model_url", "https://foreign.example/model.usd"),
    ("generation", "latest"), ("generation", "0"), ("generation", 12345),
    ("size_bytes", 0), ("size_bytes", True), ("size_bytes", delivery.MAX_MODEL_BYTES + 1),
    ("sha256", "not-a-digest"), ("team_id", "../another-team")])
def test_invalid_operator_selection_never_uses_adc(selected, field, value):
    selected.row[field] = value
    client = Client(selected)
    with pytest.raises(delivery.RobotModelDeliveryError, match="configuration_invalid"):
        run(selected, client)
    assert not client.calls and not selected.target.exists()


@pytest.mark.parametrize("change", ["different_team", "different_checkpoint", "missing_object", "different_digest"])
def test_binding_mismatch_never_downloads_or_falls_back(selected, change):
    if change == "different_team":
        selected.row["team_id"] = "other-team"
    elif change == "different_checkpoint":
        selected.row["checkpoint_id"] = "other-checkpoint"
    elif change == "missing_object":
        selected.manifest["model"]["url"] = selected.row["model_url"].replace("robot.usd", "missing.usd")
    else:
        selected.manifest["model"]["sha256"] = "sha256:" + "0" * 64
    client = Client(selected)
    with pytest.raises(delivery.RobotModelDeliveryError, match="(owner_binding_missing|digest_binding_mismatch)"):
        run(selected, client)
    assert not client.calls and not selected.target.exists()


def test_ambiguous_operator_mapping_refuses_before_access(selected):
    selected.profile["private_robot_models"]["artifacts"].append(deepcopy(selected.row))
    client = Client(selected)
    with pytest.raises(delivery.RobotModelDeliveryError, match="configuration_invalid"):
        run(selected, client)
    assert not client.calls


@pytest.mark.parametrize("override,code", [
    ({"generation": 54321}, "generation_mismatch"), ({"size": 1}, "size_mismatch"),
    ({"body": b"#usda 1.0\ndef Xform \"Wrong\" {}\n"}, "digest_mismatch"),
    ({"body": b"too short"}, "size_mismatch"), ({"body": b"x" * 500}, "size_mismatch")])
def test_metadata_bytes_and_actual_stream_bounds_are_checked(selected, override, code):
    client = Client(selected, **override)
    with pytest.raises(delivery.RobotModelDeliveryError, match=code):
        run(selected, client)
    if selected.target.exists():
        assert selected.target.stat().st_size <= selected.row["size_bytes"]
    if "generation" in override or "size" in override:
        assert not any(call[0] == "download" for call in client.calls)


def test_provider_error_cannot_expose_private_response(selected):
    client = Client(selected, error=RuntimeError("SECRET-PROVIDER-RESPONSE"))
    with pytest.raises(delivery.RobotModelDeliveryError) as raised:
        run(selected, client)
    assert str(raised.value) == "team_robot_model_private_download_unavailable"
    assert raised.value.__suppress_context__ is True


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_local_bytes_are_never_replaced(selected, symlink):
    original = selected.target.parent / "retained.usd"
    original.write_bytes(b"retained")
    if symlink:
        selected.target.symlink_to(original)
    else:
        selected.target.write_bytes(b"retained")
    with pytest.raises(delivery.RobotModelDeliveryError, match="download_unavailable"):
        run(selected, Client(selected))
    assert selected.target.read_bytes() == original.read_bytes() == b"retained"


def test_unmapped_public_model_keeps_existing_fetch_without_storage_access(selected):
    selected.manifest["model"]["url"] = "https://team.example/robot.usd"
    client = Client(selected)
    assert run(selected, client) is None
    assert not client.calls


def test_launch_private_selection_cannot_call_anonymous_fetch(selected, monkeypatch):
    from blueprint_pipeline import team_candidate_quick10_launch as launch
    calls = []
    monkeypatch.setattr(launch, "verify_model_file", lambda manifest, path: {"sha256": selected.row["sha256"]})
    monkeypatch.setattr(launch, "read_model_structure", lambda path: {})
    monkeypatch.setattr(launch, "team_embodiment_binding", lambda *args, **kwargs: {"checked": True})
    monkeypatch.setattr(launch, "_read", lambda path: {})
    selected.manifest["base"] = {"mode": "fixed"}
    binding = launch._stage_robot_model(job_dir=selected.target.parent / "job", manifest=selected.manifest,
        facts={"packet_dir": str(selected.target.parent)}, fetch=lambda *args: calls.append(args),
        profile=selected.profile, team_id="team-one", checkpoint_id="checkpoint-one", storage_client=Client(selected))
    assert binding == {"checked": True} and not calls
    assert (selected.target.parent / "job/team_robot_model_delivery.json").is_file()


def test_derived_profile_cannot_choose_private_model_authority():
    from blueprint_pipeline.team_candidate_quick10_registry import DERIVED_FIELDS, OPERATOR_FIELDS
    assert "private_robot_models" in OPERATOR_FIELDS
    assert "private_robot_models" not in DERIVED_FIELDS
