"""Actual synthetic producer bytes through native staging and existing readers."""

import base64
import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline import capture_original_owner_observer as observer
from blueprint_pipeline import pubsub_handoff_listener as listener
from blueprint_pipeline.frames_layout import load_frames_layout, iter_frame_payloads
from blueprint_pipeline.local_capture import resolve_local_capture_context
from blueprint_pipeline.task_evaluation_scene_retirement_generations import capture_birth_input_path
from tests.test_scene_retirement_real_participants import access_fixture


FIXTURES = Path(__file__).parent / "fixtures"


class SelectedBlob:
    def __init__(self, name, version):
        self.name = name
        self.generation = int(version["generation"])
        self.size = version["size_bytes"]
        self.crc32c = version["crc32c"]
        self.bytes = base64.b64decode(version["bytes_base64"])

    def reload(self, **kwargs):
        assert kwargs["if_generation_match"] == self.generation

    def download_as_bytes(self, **kwargs):
        assert kwargs["if_generation_match"] == self.generation
        return self.bytes

    def download_to_file(self, destination, **kwargs):
        assert kwargs["if_generation_match"] == self.generation
        destination.write(self.bytes)


class SelectedStorage:
    def __init__(self, bundle):
        self.objects = {(row["name"], int(version["generation"])): SelectedBlob(row["name"], version)
                        for row in bundle["objects"] for version in row["versions"]}

    def bucket(self, name):
        assert name == "test-bucket"
        return self

    def blob(self, name, generation):
        return self.objects[(name, generation)]

    def list_blobs(self, *_args, **_kwargs):
        raise AssertionError("selected staging must not discover mutable latest objects")


def staged_capture(tmp_path, monkeypatch, *, withdrawn=False, policy_kind='enabled', purpose=None):
    access, policy, root = access_fixture(tmp_path, monkeypatch)
    bundle = json.loads((FIXTURES / "capture-delivery-browser-web-bundle.json").read_text())
    assert bundle["synthetic"] is True
    owner = json.loads((FIXTURES / ("capture-delivery-browser-web-owner-revoked.json"
                                   if withdrawn else "capture-delivery-browser-web-owner-granted.json")).read_text())
    # Refresh only this synthetic observation's test clock. Producer identities,
    # rights and member bytes stay exactly those emitted by the real fixtures.
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    if purpose:
        from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
        owner['purpose'] = purpose
        owner['capture_owner'] = None
        owner['source_projection_digest'] = cross_runtime_canonical_digest({
            key: owner[key] for key in (*observer._SOURCE_KEYS, 'purpose')})
    owner["observed_at_epoch"] = int(time.time())
    owner["valid_until_epoch"] = owner["observed_at_epoch"] + 60
    owner["observation_digest"] = canonical_digest(owner, digest_field="observation_digest")
    monkeypatch.setattr(observer, "load_original_owner_observation", lambda **_kwargs: owner)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(tmp_path / "disk"))
    handoff = listener.parse_handoff_payload(bundle["published"])
    if policy_kind != 'enabled':
        from tests.test_capture_generation_birth import _without_retirement_policy
        target_path = root / handoff.bucket / 'scenes' / handoff.scene_id / 'captures' / handoff.capture_id
        _without_retirement_policy(access, policy, target_path, tmp_path, monkeypatch, policy_kind)
    storage = SelectedStorage(bundle)
    target = listener.stage_handoff_capture(handoff, storage_root=root, storage_client=storage,
                                           **({'expected_purpose': purpose} if purpose else {}))
    return root, target, handoff, storage


def test_actual_web_owner_fixture_satisfies_strict_consumer_source_digest():
    import copy
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    owner = json.loads((FIXTURES / "capture-delivery-browser-web-owner-granted.json").read_text())
    def validate(value):
        return observer.validate_observation(value, bucket=value["bucket"], scene_id=value["scene_id"],
            capture_id=value["capture_id"], marker_generation=value["completion_marker"]["generation"],
            now_epoch=value["observed_at_epoch"])
    assert validate(owner) == owner
    for field in ("completion_marker", "producer_delivery"):
        changed = copy.deepcopy(owner)
        target = changed[field] if field == "completion_marker" else changed[field]["raw_video"]
        if field == "completion_marker":
            target["sha256"] = "sha256:" + "0" * 64
        else:
            target["crc32c"] = "AAAAAA=="
        changed["observation_digest"] = cross_runtime_canonical_digest(changed, digest_field="observation_digest")
        with pytest.raises(ValueError, match="capture_owner_observation_digest_invalid"):
            validate(changed)


@pytest.mark.parametrize('policy_kind', ['enabled', 'absent', 'disabled'])
def test_real_selected_staging_feeds_existing_capture_and_frame_readers(tmp_path, monkeypatch, policy_kind):
    root, target, handoff, storage = staged_capture(tmp_path, monkeypatch, policy_kind=policy_kind)
    context = resolve_local_capture_context(target)
    assert "/deliveries/" in context.descriptor_uri
    assert context.descriptor_path == capture_birth_input_path(target, "capture_descriptor.json")
    assert context.descriptor_path.is_file()
    assert capture_birth_input_path(target, "qa_report.json").is_file()
    layout = load_frames_layout(target / "frames")
    frames = list(iter_frame_payloads(target / "frames"))
    assert len(layout.records) == len(frames) == 5
    assert sum(len(data) for _record, data in frames) == 8875
    assert not (target / "frames").exists(), "read selected members in place; do not make untracked copies"
    assert listener.stage_handoff_capture(handoff, storage_root=root, storage_client=storage) == target
    assert list(iter_frame_payloads(target / "frames")) == frames


def test_actual_selected_source_staging_rejects_current_withdrawal_before_payload_writes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="capture_owner"):
        staged_capture(tmp_path, monkeypatch, withdrawn=True)
    assert not list(tmp_path.rglob("capture_descriptor.json"))


def test_actual_selected_descriptor_rechecks_canonical_late_withdrawal_before_preparation(tmp_path, monkeypatch):
    from blueprint_pipeline.capture_orchestrator import PipelineConfig
    from blueprint_pipeline.common import PipelineError
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.site_package_orchestrator import run_qualification_pipeline
    root, target, _, _ = staged_capture(tmp_path, monkeypatch)
    tombstone = target.parent.parent / "website_withdrawal/tombstone.json"
    tombstone.parent.mkdir()
    value = {"schema_version": "website_capture_withdrawal_tombstone.v1",
             "request_id": "r1", "scene_id": "site-r1",
             "consent_revoked_at": "2026-10-07T00:00:00Z"}
    value["digest"] = canonical_digest(value, digest_field="digest")
    tombstone.write_text(json.dumps(value))
    def forbidden(**_kwargs):
        pytest.fail("withdrawn synthetic source cannot reach context, sponsorship or providers")
    monkeypatch.setattr("blueprint_pipeline.site_package_orchestrator.load_current_website_task_context", forbidden)
    monkeypatch.setattr("blueprint_pipeline.site_package_orchestrator.load_website_scene_sponsorship", forbidden)
    context = resolve_local_capture_context(target)
    with pytest.raises(PipelineError, match="website_capture_withdrawn"):
        run_qualification_pipeline(descriptor_gcs_uri=context.descriptor_uri,
                                   config=PipelineConfig(gcs_root=root))


@pytest.mark.parametrize("damage", ["index_bytes", "descriptor_bytes", "symlink", "missing"])
@pytest.mark.parametrize('policy_kind', ['enabled', 'absent', 'disabled'])
def test_selected_input_damage_refuses_canonical_fallback(tmp_path, monkeypatch, damage, policy_kind):
    _, target, _, _ = staged_capture(tmp_path, monkeypatch, policy_kind=policy_kind)
    relative = "capture_descriptor.json" if damage == "descriptor_bytes" else "frames/index.jsonl"
    selected = capture_birth_input_path(target, relative)
    if damage.endswith("bytes"):
        selected.write_bytes(b"changed")
    elif damage == "symlink":
        selected.unlink()
        selected.symlink_to(tmp_path / "outside")
    else:
        selected.unlink()
    canonical = target / relative
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("old canonical artifact")
    with pytest.raises((ValueError, OSError)):
        capture_birth_input_path(target, relative)


def test_selected_lookup_rejects_unregistered_or_cross_namespace_paths(tmp_path, monkeypatch):
    _, target, _, _ = staged_capture(tmp_path, monkeypatch)
    for relative in ("../other/capture_descriptor.json", "raw/manifest.json", "frames/../index.jsonl", "frames/unknown.tar"):
        with pytest.raises(ValueError):
            capture_birth_input_path(target, relative)


def test_preparation_null_owner_historical_members_feed_readers_without_execution_authority(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access

    _, target, handoff, _ = staged_capture(tmp_path, monkeypatch, policy_kind='absent', purpose='scene_preparation')
    with pytest.raises(ValueError, match='capture_owner'):
        generations.capture_birth_source_projection(target)
    source = generations.capture_birth_source_projection(target, expected_purpose='scene_preparation')
    assert source['capture_owner_user_id'] is None
    assert source['capture_observation_purpose'] == 'scene_preparation' and access._policy() is None
    context = resolve_local_capture_context(target)
    assert generations.read_selected_capture_input_bytes(context.descriptor_path)
    assert len(list(iter_frame_payloads(target / 'frames'))) == 5
    owner = json.loads(Path(source['owner_observation_raw_ref']['path']).read_bytes())
    with pytest.raises(ValueError, match='capture_owner'):
        observer.validate_observation(owner, bucket=handoff.bucket, scene_id=handoff.scene_id,
            capture_id=handoff.capture_id, marker_generation=handoff.source_finalize['generation'],
            now_epoch=owner['observed_at_epoch'])


def test_disabled_selected_read_holds_fixed_fence_through_actual_byte_read(tmp_path, monkeypatch):
    import fcntl
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations
    from blueprint_pipeline import task_evaluation_scene_retirement_access as access

    _, target, _, _ = staged_capture(tmp_path, monkeypatch, policy_kind='absent')
    original = generations._capture_birth_selected_bytes
    calls = []
    def checked(selected, row):
        with access._opened(tmp_path / 'coordinator', directory=True, protected=True) as (fd, _):
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        calls.append(selected)
        return original(selected, row)
    monkeypatch.setattr(generations, '_capture_birth_selected_bytes', checked)
    assert generations.capture_birth_input_bytes(target, 'capture_descriptor.json')
    assert len(calls) == 1


def test_ordinary_relative_capture_keeps_legacy_input_reads_without_native_admission(tmp_path, monkeypatch):
    from blueprint_pipeline import task_evaluation_scene_retirement_generations as generations

    monkeypatch.chdir(tmp_path)
    root = Path('ordinary-legacy-capture')
    root.mkdir()
    (root / 'capture_descriptor.json').write_bytes(b'{"legacy":true}')
    assert generations.capture_birth_source_projection(root) is None
    assert generations.capture_birth_input_bytes(root, 'capture_descriptor.json') == b'{"legacy":true}'
