"""Only real activation consumers pin retained metadata; ordinary release owns it."""
import json
from pathlib import Path
import time

import pytest

from blueprint_pipeline import control_plane_storage_pins as pins
from blueprint_pipeline import task_evaluation_partial_astra_transport as transport
from blueprint_pipeline import task_evaluation_launch_activation_worker as worker
from blueprint_pipeline.control_plane_storage_gc import build_derived_directory_manifest
from tests.test_partial_astra_transport import retained, seal, write  # noqa: F401


@pytest.fixture
def consumer(retained, tmp_path, monkeypatch):  # noqa: F811
    args, source = retained
    base = tmp_path / 'launch-activations'
    metadata = base / 'retained-source-activation'
    owned = base / 'current-scene-configuration-activation-auto'
    owned.mkdir(parents=True)
    profile = json.loads((source / 'launch_profile.json').read_text())
    for row in profile['immutable_inputs']:
        original = Path(row['path'])
        target = metadata / 'launch-set' / original.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        row['path'] = str(target)
    write(source / 'launch_profile.json', seal(profile, 'profile_digest'))
    launch = json.loads((source / 'launch_receipt.json').read_text())
    launch['launch_profile_digest'] = profile['profile_digest']
    write(source / 'launch_receipt.json', seal(launch, 'receipt_digest'))
    zero = json.loads((source / 'post_teardown_provider_zero_receipt.json').read_text())
    zero.update(launch_profile_digest=profile['profile_digest'], receipt_digest=launch['receipt_digest'])
    write(source / 'post_teardown_provider_zero_receipt.json', seal(zero, 'provider_zero_receipt_digest'))
    args['output_root'] = owned / 'partial_astra_successor'
    request = {'activation_id': owned.name, 'team_namespace': args['envelope']['team_namespace'],
               'preparation': {'preparation_id': 'current-preparation'}}
    pins_root = tmp_path / 'pins'
    monkeypatch.setenv(pins.PINS_ROOT_ENV, str(pins_root))
    return args, request, owned, metadata, pins_root


def select(consumer, created):
    args, request, owned, _, pins_root = consumer
    return transport.select_partial_astra_source(**args, activation_request=request,
        activation_root=owned, pins_root=pins_root, on_pin_created=created.append)


def test_live_source_metadata_is_protected_and_repeat_does_not_refresh_or_add_pins(consumer):
    args, request, owned, metadata, pins_root = consumer
    created = []
    selected = select(consumer, created)
    assert selected.is_file() and len(created) == 1
    pin = created[0]
    assert set(pin['paths']) == {str(owned), str(metadata)}
    assert pin['depends_on'] == [{'kind': 'compilation', 'owner_id': 'current-preparation'},
                                 {'kind': 'preparation', 'owner_id': 'current-preparation'}]
    report = build_derived_directory_manifest(derived_roots=[owned.parent], pins_root=pins_root,
        queue_roots=[], minimum_age_seconds=0, classifier=lambda *_a, **_k: None)
    assert report['candidates'] == [] and report['retained_counts']['pinned'] == 2
    assert select(consumer, created) == selected
    assert len(created) == 1
    assert json.loads(pins.pin_path(pins_root, 'activation', request['activation_id']).read_text()) == pin
    assert len(list(pins_root.rglob('*.json'))) == 1


def test_offline_selector_does_not_pin_even_with_configured_production_pin_environment(consumer):
    args, _, _, _, pins_root = consumer
    assert transport.select_partial_astra_source(**args).is_file()
    assert not pins_root.exists()


@pytest.mark.parametrize('legacy', ['insufficient', 'released', 'expired'])
def test_existing_insufficient_or_closed_pin_is_not_overwritten_or_released(consumer, legacy):
    _, request, owned, metadata, pins_root = consumer
    existing = pins.pin_activation_best_effort(request, owned.parent, pins_root=pins_root,
        retained_paths=[] if legacy == 'insufficient' else [metadata])
    path = pins.pin_path(pins_root, 'activation', request['activation_id'])
    if legacy == 'released':
        pins.release_storage_pin(pins_root=pins_root, kind='activation', owner_id=request['activation_id'])
        existing = json.loads(path.read_text())
    if legacy == 'expired':
        existing['expires_at_epoch'] = time.time() - 10
        path.write_text(json.dumps(existing))
    created = []
    with pytest.raises(transport.PartialAstraTransportError, match='known_partial_source_unusable'):
        select(consumer, created)
    assert created == []
    worker._release_created_early_storage_pins(request, created)
    assert json.loads(path.read_text()) == existing


def test_failure_after_early_pin_releases_only_this_invocations_pin_and_dependencies(consumer, monkeypatch):
    _, request, owned, metadata, pins_root = consumer
    for kind in ('preparation', 'compilation'):
        pins.write_storage_pin(pins_root=pins_root, kind=kind, owner_id='current-preparation', paths=[owned/kind])
    unrelated = pins.write_storage_pin(pins_root=pins_root, kind='activation', owner_id='unrelated', paths=[owned.parent/'unrelated'])
    def refuse(*args, **kwargs):
        raise ValueError('fixture disk reservation refusal')
    monkeypatch.setattr(transport, 'reserve_control_plane_disk', refuse)
    created = []
    with pytest.raises(transport.PartialAstraTransportError):
        select(consumer, created)
    assert len(created) == 1 and str(metadata) in pins.live_pinned_paths(pins_root)
    worker._release_created_early_storage_pins(request, created)
    assert pins.live_pinned_paths(pins_root) == set(unrelated['paths'])
    assert metadata.is_dir()


def test_source_metadata_is_revalidated_after_pin_before_packing(consumer, monkeypatch):
    args, request, _, metadata, pins_root = consumer
    original = pins.pin_activation_best_effort
    def pin_then_change(*args, **kwargs):
        value = original(*args, **kwargs)
        (metadata/'launch-set/source-authority.json').write_text('{}')
        return value
    monkeypatch.setattr(pins, 'pin_activation_best_effort', pin_then_change)
    created = []
    with pytest.raises(transport.PartialAstraTransportError):
        select(consumer, created)
    assert len(created) == 1 and not args['output_root'].exists()
    worker._release_created_early_storage_pins(request, created)
    assert not pins.live_pinned_paths(pins_root)


def test_ineligible_candidate_never_creates_a_pin(consumer):
    args, _, _, _, pins_root = consumer
    args['envelope']['stage_configuration_references'][0]['digest'] = 'sha256:' + 'f' * 64
    created = []
    assert select(consumer, created) is None
    assert not created and not pins_root.exists()


def test_offline_cached_selection_with_missing_original_archive_cannot_pin(consumer):
    args, _, _, _, pins_root = consumer
    selected = transport.select_partial_astra_source(**args)
    value = json.loads(selected.read_text())
    Path(value['source_archive']['path']).unlink()
    created = []
    with pytest.raises((OSError, transport.PartialAstraTransportError)):
        select(consumer, created)
    assert not created and not pins_root.exists()


def test_pin_published_between_gc_targets_protects_later_target(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager
    import threading
    from blueprint_pipeline import control_plane_storage_gc as gc

    base = tmp_path / 'launch-activations'
    disposable, source = base / 'a-disposable', base / 'z-source'
    for path in (disposable, source):
        path.mkdir(parents=True)
        (path / 'metadata.json').write_text('unchanged source metadata')
    pins_root = tmp_path / 'pins'
    report = gc.build_derived_directory_manifest(derived_roots=[base], pins_root=pins_root,
        queue_roots=[], minimum_age_seconds=0, classifier=lambda *_a, **_k: None)
    assert len(report['candidates']) == 2
    first_target_finished = threading.Event()
    pin_published = threading.Event()
    real_guard = pins.storage_pin_guard
    exclusive_claims = []

    @contextmanager
    def interleaved_guard(root, *, exclusive):
        with real_guard(root, exclusive=exclusive):
            if exclusive:
                exclusive_claims.append(str(root))
            yield
        if exclusive and len(exclusive_claims) == 1:
            first_target_finished.set()
            assert pin_published.wait(5)

    monkeypatch.setattr(pins, 'storage_pin_guard', interleaved_guard)

    def publish_pin():
        assert first_target_finished.wait(5)
        pins.write_storage_pin(pins_root=pins_root, kind='activation', owner_id='successor', paths=[source])
        pin_published.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        published = pool.submit(publish_pin)
        result = gc.apply_derived_directory_manifest(report, ack=gc.DERIVED_ACK, pins_root=pins_root,
            queue_roots=[], classifier=lambda *_a, **_k: None)
        published.result(timeout=5)
    assert len(exclusive_claims) == 2
    assert result['removed_count'] == 1 and not disposable.exists()
    assert (source / 'metadata.json').read_text() == 'unchanged source metadata'
    assert result['skipped'] == [{'name': source.name, 'reason': 'candidate_changed'}]
