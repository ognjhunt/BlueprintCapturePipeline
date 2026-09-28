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
    authority = public / "authority"
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


def fill_cache(value, monkeypatch):
    import io
    import os
    from urllib.parse import unquote
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import control_plane_disk_budget as ledger
    from blueprint_pipeline import native_g1_checkpoint_cache as native
    fetcher = native._fetcher()
    calls = []
    class Response(io.BytesIO):
        def __init__(self, url):
            path = unquote(url.removeprefix(fetcher.MODEL_BASE))
            super().__init__(value['payloads'][path])
            self.url = url
        def geturl(self):
            return self.url
    def response(url, **kwargs):
        calls.append(url)
        return Response(url)
    monkeypatch.setattr(fetcher, '_open_https', response)
    monkeypatch.setattr(native, '_fetcher', lambda: fetcher)
    monkeypatch.setattr(cache, '_process_identity', lambda: dict(boot_id='aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
        pid=os.getpid(), start_ticks=1, pid_namespace_inode=1), raising=False)
    reserve = ledger.reserve_control_plane_disk
    reservations = []
    def local_reserve(role, **kwargs):
        from types import SimpleNamespace
        result = reserve(role, reservation_root=value['config'].parent / 'reservations',
            disk_usage=lambda path: SimpleNamespace(total=100*1024**3, free=90*1024**3),
            now=lambda: 1000, **kwargs)
        reservations.append(result)
        return result
    monkeypatch.setattr(cache, 'reserve_control_plane_disk', local_reserve, raising=False)
    grant = issue_cache(value)
    result = cache.fill_needed_checkpoint_cache(grant['intent_id'],
        expected_sha256=grant['intent']['sha256'], expected_size_bytes=grant['intent']['size_bytes'],
        installed_config_path=value['config'], now=lambda: 1100)
    return grant, result, fetcher, calls, reservations


def test_actual_new_fill_uses_one_live_reservation_and_real_all_four_verification(
        cache_installation, monkeypatch):
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import native_g1_checkpoint_cache as native
    value = cache_installation
    grant, result, fetcher, calls, reservations = fill_cache(value, monkeypatch)
    target = Path(result['path'])
    assert result['status'] == 'cache_ready' and result['target_ready_observed'] is True
    assert len(reservations) == 1 and reservations[0].role == 'g1_checkpoint_cache'
    assert reservations[0].released is True
    assert len(calls) == 24
    assert {path: (target / path).read_bytes() for path in value['payloads']} == value['payloads']
    lease = json.loads((target / '.lane-scratch.v1.json').read_bytes())
    assert lease['class_intent'] == 'cache' and lease['cleanup'] == 'owner_review'
    assert lease['size_budget_bytes'] == 8 * 1024 * 1024 and lease['expires_at_epoch'] == 2800
    birth_raw = (value['public'] / (grant['intent_id'] + '.birth.json')).read_bytes()
    assert b'principal' not in birth_raw and b'policy' not in birth_raw
    actual = fetcher.materialize_candidate
    candidates = []
    def verify(**kwargs):
        assert kwargs['_cache_use'] is use
        candidates.append(kwargs['candidate_id'])
        return actual(**kwargs)
    monkeypatch.setattr(fetcher, 'materialize_candidate', verify)
    with cache.NeededCheckpointCacheUse.open_registered(target, mode='read',
            installed_config_path=value['config'], now=lambda: 1200) as use:
        rows = native.verify_local_g1_checkpoint_cache(target, _cache_use=use)
        assert candidates == list(PAIR_ORDER) and len(rows) == 24
        assert use.resource_counters['roles']['verify']['bytes'] == sum(map(len, value['payloads'].values()))
    assert use.closed is True


def fake_wam(monkeypatch, *, hit):
    import sys
    from types import SimpleNamespace
    from blueprint_pipeline import wam_provider_object_store as wam
    events, objects, pending = [], {}, {}
    class NotFound(Exception):
        response = {'ResponseMetadata': {'HTTPStatusCode': 404}}
    class Client:
        def head_object(self, *, Bucket, Key):
            events.append(('head', Key))
            if Key not in objects:
                if not hit:
                    raise NotFound()
                sha = Key.rsplit('/', 1)[-1].removesuffix('.bin')
                data = next(data for data in fake_wam.payloads.values() if hashlib.sha256(data).hexdigest() == sha)
                objects[Key] = (data, sha)
            data, sha = objects[Key]
            return {'ContentLength': len(data), 'Metadata': {'sha256': sha}}
        def create_multipart_upload(self, *, Bucket, Key, Metadata):
            events.append(('create', Key))
            pending[Key] = [b'', Metadata['sha256']]
            return {'UploadId': 'owned-' + hashlib.sha256(Key.encode()).hexdigest()[:16]}
        def upload_part(self, *, Bucket, Key, UploadId, PartNumber, Body):
            assert isinstance(Body, bytes) and len(Body) <= 8*1024*1024
            pending[Key][0] += Body
            events.append(('part', Key))
            return {'ETag': '"fixture-part"'}
        def complete_multipart_upload(self, *, Bucket, Key, UploadId, MultipartUpload):
            events.append(('complete', Key))
            objects[Key] = tuple(pending.pop(Key))
            return {}
        def abort_multipart_upload(self, *, Bucket, Key, UploadId):
            events.append(('abort', Key))
            pending.pop(Key, None)
        def generate_presigned_url(self, *args, **kwargs):
            events.append(('presign', kwargs['Params']['Key']))
            return 'https://fixture.invalid/owned'
        def upload_file(self, *args, **kwargs):
            pytest.fail('enrolled bytes reached unguarded SDK pathname upload')
        def upload_fileobj(self, *args, **kwargs):
            pytest.fail('enrolled bytes reached unguarded SDK filelike upload')
        def close(self):
            events.append(('close', None))
    def client(*args, **kwargs):
        assert kwargs['config'].connect_timeout == kwargs['config'].read_timeout == 45
        assert kwargs['config'].retries == {'total_max_attempts': 1}
        return Client()
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, 'boto3', SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules, 'botocore.client', SimpleNamespace(Config=Config))
    monkeypatch.setattr(wam, '_read_first_file', lambda **kw: ('fixture', {'available': True}))
    return events


@pytest.mark.parametrize('hit', [True, False])
def test_real_verifier_and_wam_callee_share_lifetime_through_all_24_files(
        cache_installation, monkeypatch, hit):
    import fcntl
    import os
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    from blueprint_pipeline import native_g1_checkpoint_cache as native
    value = cache_installation
    _, result, _, _, _ = fill_cache(value, monkeypatch)
    target = Path(result['path'])
    fake_wam.payloads = value['payloads']
    events = fake_wam(monkeypatch, hit=hit)
    with cache.NeededCheckpointCacheUse.open_registered(target, mode='read',
            installed_config_path=value['config'], now=lambda: 1200) as use:
        proof = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(proof, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(proof)
        staged = native.stage_g1_checkpoint_cache(cache_root=target, job_dir=value['config'].parent / 'staged',
            key_prefix='fixture', expiration_seconds=60, _cache_use=use)
        assert staged['status'] == 'completed' and staged['file_count'] == 24
        assert staged['cache_hit_count'] == (24 if hit else 0)
        assert staged['upload_count'] == (0 if hit else 24)
        assert use.closed is False
        roles = use.resource_counters['roles']
        assert roles['verify']['bytes'] == roles['wam_hash']['bytes'] == sum(map(len, value['payloads'].values()))
        if not hit:
            assert roles['upload']['bytes'] == roles['verify']['bytes']
        assert sum(kind == 'close' for kind, _ in events) == 24
    assert use.closed


def test_root_revoke_during_actual_hash_stops_next_payload_read(cache_installation, monkeypatch):
    import os
    from blueprint_pipeline import control_plane_registered_checkpoint_cache as cache
    value = cache_installation
    grant, result, _, _, _ = fill_cache(value, monkeypatch)
    target = Path(result['path'])
    data_reads, original = [], os.pread
    def read(fd, amount, offset):
        block = original(fd, amount, offset)
        data_reads.append(block)
        if len(data_reads) == 1:
            cache.update_needed_checkpoint_cache_authority(operation='revoke', intent_id=grant['intent_id'],
                installed_config_path=value['config'], now=lambda: 1201)
        return block
    with cache.NeededCheckpointCacheUse.open_registered(target, mode='read',
            installed_config_path=value['config'], now=lambda: 1200) as use:
        monkeypatch.setattr(os, 'pread', read)
        with pytest.raises(ValueError, match='needed_cache'):
            use.hash_file(target / next(iter(value['payloads'])), role='wam_hash')
        assert len(data_reads) == 1 and use.failure is not None
    assert {path: (target / path).read_bytes() for path in value['payloads']} == value['payloads']
