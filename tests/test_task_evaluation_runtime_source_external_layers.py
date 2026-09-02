"""Runtime-source wrappers carry their multi-gigabyte payload as an external layer.

Every release re-bound the same 4.29 GB runtime packet into a fresh wrapper
whose only new bytes were its identity bindings, and the preparation content
store, keyed by whole-wrapper digest, kept one 4.29 GB blob per release.  A v2
wrapper stores members above a size threshold once, by digest, and references
them by URI; preparation fetches each layer once, and the adapter resolves it
through the compile-side member store with the same digest verification the
embedded path always had.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline import task_evaluation_activation_runtime_layers as activation_layers
from blueprint_pipeline.task_evaluation_episode_compilation_worker import (
    COMPILATION_RESERVATION_MARGIN_BYTES,
    _expected_compilation_bytes,
)
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    stage_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
    EXTERNAL_LAYER_TRANSPORT,
    MANIFEST_NAME,
    RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX,
    TaskEvaluationNativeArenaAdapterError,
    build_task_evaluation_runtime_source_bundle,
    main as adapter_main,
    materialize_native_arena_adapter,
    read_runtime_source_external_layers,
)
from tests.test_native_task_arena_bundle import _runtime_source_packet
from tests.test_task_evaluation_configured_scene_object_store import (
    _ContentAddressedClient,
)
from tests.test_task_evaluation_launch_preparation_worker import (
    SERVICE_ACCOUNT,
    fake_adapter,
    fetcher,
    request_with_fetchable_bytes,
)
from tests.test_task_evaluation_native_arena_preparation_adapter import (
    _bundles,
    _identity,
)


BUCKET = "blueprint-production-inputs"
LAYER_PREFIX = (
    f"s3://{BUCKET}/{store.LARGE_ARTIFACT_KEY_PREFIX}/{store.EXTERNAL_LAYER_ARTIFACT_KIND}"
)
PACKET_NAME = "native_task_runtime_sources.zip"


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _v2_wrapper(
    tmp_path: Path,
    value: dict,
    *,
    layer_store: Path,
    name: str = "runtime-source-v2.zip",
    prefix: str = LAYER_PREFIX,
) -> tuple[dict, Path, Path]:
    """Build a wrapper whose runtime packet, and only that packet, is external."""

    receipt_path = _runtime_source_packet(tmp_path)
    source_root = receipt_path.parent
    packet = source_root / PACKET_NAME
    others = [
        path for path in source_root.rglob("*") if path.is_file() and path != packet
    ]
    # The fixture packet must be the largest member for the threshold below to
    # externalize exactly one file; fail loudly here rather than silently later.
    assert all(path.stat().st_size < packet.stat().st_size for path in others)
    built = build_task_evaluation_runtime_source_bundle(
        source_root=source_root,
        output_path=tmp_path / name,
        expected_production_commit=value["expected_production_commit"],
        runtime_identity=value["runtime"]["identity"],
        external_layer_store_root=layer_store,
        external_layer_uri_prefix=prefix,
        external_layer_min_bytes=packet.stat().st_size,
    )
    return built, Path(built["path"]), packet


def test_builder_externalizes_the_runtime_packet_and_keeps_the_wrapper_small(
    tmp_path: Path,
) -> None:
    value, _configured, _construction, v1_wrapper = _bundles(tmp_path)
    layer_store = tmp_path / "layer-store"

    built, wrapper, packet = _v2_wrapper(tmp_path, value, layer_store=layer_store)

    hex_digest = _sha(packet).removeprefix("sha256:")
    assert built["external_layer_count"] == 1
    layer = built["external_layers"][0]
    assert layer == {
        "relative_path": f"payload/{PACKET_NAME}",
        "sha256": _sha(packet),
        "size_bytes": packet.stat().st_size,
        "uri": f"{LAYER_PREFIX}/sha256/{hex_digest}/{PACKET_NAME}",
        "store_path": str(layer_store / "sha256" / hex_digest / PACKET_NAME),
    }
    assert Path(layer["store_path"]).read_bytes() == packet.read_bytes()
    assert wrapper.stat().st_size < packet.stat().st_size
    with zipfile.ZipFile(wrapper) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read(MANIFEST_NAME))
    assert f"payload/{PACKET_NAME}" not in names
    assert MANIFEST_NAME in names
    external_rows = [row for row in manifest["entries"] if "external_layer" in row]
    assert [row["external_layer"] for row in external_rows] == [
        {"transport": EXTERNAL_LAYER_TRANSPORT, "uri": layer["uri"]}
    ]
    assert [row["sha256"] for row in external_rows] == [_sha(packet)]

    value["execution_adapter"]["runtime_source_bundle"] = _identity(wrapper)
    assert read_runtime_source_external_layers(bundle_path=wrapper, request=value) == [
        {
            "relative_path": f"payload/{PACKET_NAME}",
            "sha256": _sha(packet),
            "size_bytes": packet.stat().st_size,
            "uri": layer["uri"],
        }
    ]
    value_v1 = copy.deepcopy(value)
    value_v1["execution_adapter"]["runtime_source_bundle"] = _identity(v1_wrapper)
    assert (
        read_runtime_source_external_layers(bundle_path=v1_wrapper, request=value_v1)
        == []
    )

    # Rebuilding for the same release reuses the stored layer and reproduces
    # the wrapper bytes exactly; the store never holds a second copy.
    again, _again_path, _packet = _v2_wrapper(
        tmp_path, value, layer_store=layer_store, name="runtime-source-v2-again.zip"
    )
    assert again["sha256"] == built["sha256"]
    assert [path.name for path in (layer_store / "sha256").iterdir()] == [hex_digest]


def test_materialize_resolves_external_layers_through_the_member_store(
    tmp_path: Path,
) -> None:
    value, configured, construction_bundle, _v1 = _bundles(tmp_path)
    built, wrapper, packet = _v2_wrapper(tmp_path, value, layer_store=tmp_path / "layers")
    value["execution_adapter"]["runtime_source_bundle"] = _identity(wrapper)
    layers = {row["sha256"]: Path(row["store_path"]) for row in built["external_layers"]}
    member_store = tmp_path / "compiled-content" / "sha256"

    results = [
        materialize_native_arena_adapter(
            request=value,
            compiled_episode_packet_path=construction_bundle,
            compiled_episode_packet_reference=_identity(construction_bundle),
            configured_revision=configured,
            runtime_source_bundle_path=wrapper,
            output_root=tmp_path / f"adapter-{index}",
            content_store_root=member_store,
            external_layers=layers,
        )
        for index in range(2)
    ]

    cached = member_store / _sha(packet).removeprefix("sha256:")
    for result in results:
        assert result["status"] == "native_arena_adapter_materialized"
        materialized = Path(result["runtime_source_receipt"]).parent / PACKET_NAME
        assert materialized.read_bytes() == packet.read_bytes()
        assert materialized.stat().st_ino == cached.stat().st_ino
    assert cached.stat().st_nlink == 3

    # Without a member store the layer is copied into place and verified the
    # same way; nothing links back into the layer store.
    plain = materialize_native_arena_adapter(
        request=value,
        compiled_episode_packet_path=construction_bundle,
        compiled_episode_packet_reference=_identity(construction_bundle),
        configured_revision=configured,
        runtime_source_bundle_path=wrapper,
        output_root=tmp_path / "adapter-plain",
        external_layers=layers,
    )
    materialized = Path(plain["runtime_source_receipt"]).parent / PACKET_NAME
    assert materialized.read_bytes() == packet.read_bytes()
    assert materialized.stat().st_nlink == 1


def test_missing_or_tampered_external_layer_is_refused_before_exposure(
    tmp_path: Path,
) -> None:
    value, configured, construction_bundle, _v1 = _bundles(tmp_path)
    built, wrapper, packet = _v2_wrapper(tmp_path, value, layer_store=tmp_path / "layers")
    value["execution_adapter"]["runtime_source_bundle"] = _identity(wrapper)
    digest = built["external_layers"][0]["sha256"]
    member_store = tmp_path / "compiled-content" / "sha256"

    def attempt(layers: dict, output: Path) -> dict:
        return materialize_native_arena_adapter(
            request=value,
            compiled_episode_packet_path=construction_bundle,
            compiled_episode_packet_reference=_identity(construction_bundle),
            configured_revision=configured,
            runtime_source_bundle_path=wrapper,
            output_root=output,
            content_store_root=member_store,
            external_layers=layers,
        )

    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match=f"task_evaluation_adapter_external_layer_missing:{PACKET_NAME}",
    ):
        attempt({}, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()

    short = tmp_path / "short.zip"
    short.write_bytes(packet.read_bytes()[:-1])
    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match=f"task_evaluation_adapter_external_layer_invalid:{PACKET_NAME}",
    ):
        attempt({digest: short}, tmp_path / "short-out")
    assert not (tmp_path / "short-out").exists()

    tampered = tmp_path / "tampered.zip"
    data = bytearray(packet.read_bytes())
    data[-1] ^= 0xFF
    tampered.write_bytes(bytes(data))
    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_bundle_member_readback_mismatch",
    ):
        attempt({digest: tampered}, tmp_path / "tampered-out")
    assert not (tmp_path / "tampered-out").exists()
    assert not (member_store / digest.removeprefix("sha256:")).exists()
    assert not list(member_store.glob(".*.partial-*"))


def test_wrapper_that_embeds_a_declared_external_layer_is_refused(
    tmp_path: Path,
) -> None:
    value, _configured, _construction, _v1 = _bundles(tmp_path)
    _built, wrapper, packet = _v2_wrapper(tmp_path, value, layer_store=tmp_path / "layers")
    doubled = tmp_path / "doubled.zip"
    with zipfile.ZipFile(wrapper) as source, zipfile.ZipFile(doubled, "w") as target:
        for info in source.infolist():
            target.writestr(info, source.read(info.filename))
        target.writestr(f"payload/{PACKET_NAME}", packet.read_bytes())
    value["execution_adapter"]["runtime_source_bundle"] = _identity(doubled)

    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_bundle_external_layer_invalid",
    ):
        read_runtime_source_external_layers(bundle_path=doubled, request=value)


def test_publish_runtime_source_external_layers_binds_the_embedded_uri(
    tmp_path: Path,
) -> None:
    value, _configured, _construction, _v1 = _bundles(tmp_path)
    built, _wrapper, _packet = _v2_wrapper(tmp_path, value, layer_store=tmp_path / "layers")
    client = _ContentAddressedClient()

    published = store.publish_runtime_source_external_layers(
        built, client=client, bucket=BUCKET
    )

    assert published["status"] == "remote_verified"
    assert published["wrapper_sha256"] == built["sha256"]
    assert published["layer_count"] == 1
    assert published["layers"][0]["uri"] == built["external_layers"][0]["uri"]
    assert published["layers"][0]["digest"] == built["external_layers"][0]["sha256"]
    assert published["layers"][0]["upload_performed"] is True
    again = store.publish_runtime_source_external_layers(
        built, client=client, bucket=BUCKET
    )
    assert again["layers"][0]["cache_hit"] is True
    assert client.upload_count == 1

    # A wrapper built against another bucket names URIs this store cannot
    # produce; refuse instead of publishing bytes nobody will fetch.
    other_built, _other, _packet = _v2_wrapper(
        tmp_path,
        value,
        layer_store=tmp_path / "layers-other",
        name="other.zip",
        prefix=LAYER_PREFIX.replace(BUCKET, "other-bucket"),
    )
    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_runtime_source_layer_uri_mismatch",
    ):
        store.publish_runtime_source_external_layers(
            other_built, client=_ContentAddressedClient(), bucket=BUCKET
        )
    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_runtime_source_receipt_invalid",
    ):
        store.publish_runtime_source_external_layers(
            {**built, "role": "construction_packet"}, client=client, bucket=BUCKET
        )


def test_cli_builds_and_publishes_a_layered_runtime_source_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    value, _configured, _construction, _v1 = _bundles(tmp_path)
    receipt_path = _runtime_source_packet(tmp_path)
    packet = receipt_path.parent / PACKET_NAME
    build_receipt = tmp_path / "build-receipt.json"

    code = adapter_main(
        [
            "build-runtime-source",
            "--source-root",
            str(receipt_path.parent),
            "--output",
            str(tmp_path / "cli-wrapper.zip"),
            "--expected-production-commit",
            value["expected_production_commit"],
            "--runtime-id",
            value["runtime"]["identity"]["id"],
            "--runtime-version",
            value["runtime"]["identity"]["version"],
            "--external-layer-store-root",
            str(tmp_path / "cli-store"),
            "--external-layer-uri-prefix",
            LAYER_PREFIX,
            "--external-layer-min-bytes",
            str(packet.stat().st_size),
            "--receipt-out",
            str(build_receipt),
        ]
    )
    assert code == 0
    built = json.loads(build_receipt.read_text(encoding="utf-8"))
    assert built["external_layer_count"] == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == built

    client = _ContentAddressedClient()
    monkeypatch.setattr(store, "_artifact_object_store_client", lambda: (client, BUCKET))
    code = adapter_main(["publish-runtime-source-layers", "--receipt", str(build_receipt)])
    assert code == 0
    published = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert published["layer_count"] == 1
    assert published["layers"][0]["uri"] == built["external_layers"][0]["uri"]
    assert client.upload_count == 1

    code = adapter_main(
        ["publish-runtime-source-layers", "--receipt", str(tmp_path / "absent.json")]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "blocked"


def test_preparation_fetches_runtime_source_layers_once_into_the_content_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, payloads = request_with_fetchable_bytes()
    built, wrapper, packet = _v2_wrapper(
        tmp_path, value, layer_store=tmp_path / "publisher-store"
    )
    wrapper_uri = f"s3://{BUCKET}/runtime-source-v2.zip"
    value["execution_adapter"]["runtime_source_bundle"] = {
        "uri": wrapper_uri,
        "digest": built["sha256"],
        "size_bytes": built["size_bytes"],
    }
    payloads[wrapper_uri] = wrapper.read_bytes()
    layer = built["external_layers"][0]
    payloads[layer["uri"]] = packet.read_bytes()
    fetches: list[str] = []

    def counted(uri: str, destination: Path, maximum_bytes: int) -> None:
        fetches.append(uri)
        fetcher(payloads)(uri, destination, maximum_bytes)

    reservations: list[dict[str, object]] = []

    class _Reservation:
        def release(self) -> None:
            return None

    def recorded_reserve(role: str, **kwargs: object) -> _Reservation:
        reservations.append({"role": role, "expected_bytes": kwargs.get("expected_bytes")})
        return _Reservation()

    monkeypatch.setattr(worker, "reserve_control_plane_disk", recorded_reserve)
    second = copy.deepcopy(value)
    second["preparation_id"] = value["preparation_id"] + "-b"
    second["run_id"] = value["run_id"] + "-b"
    queue = tmp_path / "queue"

    for request_value in (value, second):
        stage_launch_preparation_request(
            value=request_value, queue_root=queue, submitted_by="blueprint-webapp"
        )
        run = worker.process_launch_preparation_queue(
            queue_root=queue,
            input_root=tmp_path / "inputs",
            allowed_uri_prefixes=[f"s3://{BUCKET}/"],
            service_account=SERVICE_ACCOUNT,
            source_commit=request_value["expected_production_commit"],
            fetcher=counted,
            adapter_materializer=fake_adapter,
            episode_compilation_queue_root=tmp_path / "episode-compilation",
            disk_reservation_root=tmp_path / "reservations",
        )
        result = run["results"][0]
        assert result["status"] != "blocked", result.get("blockers")
        rows = [
            row
            for row in result["references"]
            if row["contract_path"].startswith(RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX)
        ]
        assert [row["contract_path"] for row in rows] == [
            f"{RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX}0"
        ]
        assert rows[0]["digest"] == layer["sha256"]
        assert rows[0]["uri"] == layer["uri"]
        assert rows[0]["size_bytes"] == packet.stat().st_size
        assert rows[0]["full_byte_service_account_readback_passed"] is True
        materialized = Path(rows[0]["materialized_path"])
        assert materialized.read_bytes() == packet.read_bytes()
        assert materialized.parent == tmp_path / "inputs" / request_value["preparation_id"]
        wrapper_row = next(
            row
            for row in result["references"]
            if row["contract_path"]
            == "execution_adapter.runtime_source_bundle"
        )
        derived = activation_layers.derive_runtime_source_external_layer_references(
            request=request_value,
            wrapper_path=Path(wrapper_row["materialized_path"]),
        )
        assert derived == [
            {
                "contract_path": rows[0]["contract_path"],
                "uri": rows[0]["uri"],
                "digest": rows[0]["digest"],
                "size_bytes": rows[0]["size_bytes"],
            }
        ]
        assert result["reference_count"] == len(result["references"])
        # The handoff to the compiler carries the layer row like any other
        # verified reference.
        envelopes = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (tmp_path / "episode-compilation" / "pending").glob("*.json")
        ]
        compilation = next(
            envelope
            for envelope in envelopes
            if envelope["preparation_id"] == request_value["preparation_id"]
        )
        assert rows[0] in compilation["materialized_references"]

    assert fetches.count(layer["uri"]) == 1
    assert fetches.count(wrapper_uri) == 1
    cas = tmp_path / "inputs" / "content-addressed" / "sha256"
    stored_layer = cas / layer["sha256"].removeprefix("sha256:")
    assert stored_layer.stat().st_nlink == 3  # store entry plus two projections
    # Disk is reserved for the layer only on the miss; the second preparation
    # finds it in the store and reserves nothing for it.
    layer_reservations = [
        row for row in reservations if row["expected_bytes"] == packet.stat().st_size
    ]
    assert len(layer_reservations) == 1
    assert all(row["role"] == "launch_preparation" for row in reservations)


def test_compilation_reserves_only_the_runtime_members_the_member_store_lacks(
    tmp_path: Path,
) -> None:
    value, _configured, _construction, v1_wrapper = _bundles(tmp_path)
    _built, wrapper, _packet = _v2_wrapper(tmp_path, value, layer_store=tmp_path / "layers")
    store_root = tmp_path / "member-store" / "sha256"

    def references(path: Path) -> dict[str, dict[str, object]]:
        return {
            "execution_adapter.runtime_source_bundle": {
                "materialized_path": str(path),
                "size_bytes": path.stat().st_size,
            }
        }

    with zipfile.ZipFile(wrapper) as archive:
        v2_entries = json.loads(archive.read(MANIFEST_NAME))["entries"]
    assert _expected_compilation_bytes(
        references(wrapper), content_store_root=store_root
    ) == COMPILATION_RESERVATION_MARGIN_BYTES + sum(row["size_bytes"] for row in v2_entries)

    store_root.mkdir(parents=True)
    for row in v2_entries:
        (store_root / row["sha256"].removeprefix("sha256:")).write_bytes(b"present")
    assert (
        _expected_compilation_bytes(references(wrapper), content_store_root=store_root)
        == COMPILATION_RESERVATION_MARGIN_BYTES
    )

    with zipfile.ZipFile(v1_wrapper) as archive:
        v1_entries = json.loads(archive.read(MANIFEST_NAME))["entries"]
    missing_v1 = sum(
        row["size_bytes"]
        for row in v1_entries
        if not (store_root / row["sha256"].removeprefix("sha256:")).is_file()
    )
    assert (
        _expected_compilation_bytes(references(v1_wrapper), content_store_root=store_root)
        == COMPILATION_RESERVATION_MARGIN_BYTES + missing_v1
    )

    opaque = tmp_path / "opaque.bin"
    opaque.write_bytes(b"not a wrapper archive")
    assert (
        _expected_compilation_bytes(references(opaque), content_store_root=store_root)
        == COMPILATION_RESERVATION_MARGIN_BYTES + opaque.stat().st_size
    )
    assert (
        _expected_compilation_bytes({}, content_store_root=store_root)
        == COMPILATION_RESERVATION_MARGIN_BYTES
    )


def test_preparation_refuses_a_wrapper_archive_it_cannot_validate(tmp_path: Path) -> None:
    """A real archive with a broken manifest fails before any layer is fetched."""

    value, payloads = request_with_fetchable_bytes()
    bogus = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bogus, "w") as archive:
        archive.writestr(
            MANIFEST_NAME,
            json.dumps(
                {
                    "schema_version": "wrong",
                    "entries": [
                        {
                            "relative_path": f"payload/{PACKET_NAME}",
                            "size_bytes": 1,
                            "sha256": "sha256:" + "0" * 64,
                            "external_layer": {
                                "transport": EXTERNAL_LAYER_TRANSPORT,
                                "uri": f"s3://{BUCKET}/layers/x",
                            },
                        }
                    ],
                }
            ),
        )
    wrapper_uri = f"s3://{BUCKET}/bogus.zip"
    value["execution_adapter"]["runtime_source_bundle"] = {
        "uri": wrapper_uri,
        "digest": _sha(bogus),
        "size_bytes": bogus.stat().st_size,
    }
    payloads[wrapper_uri] = bogus.read_bytes()
    queue = tmp_path / "queue"
    stage_launch_preparation_request(
        value=value, queue_root=queue, submitted_by="blueprint-webapp"
    )

    run = worker.process_launch_preparation_queue(
        queue_root=queue,
        input_root=tmp_path / "inputs",
        allowed_uri_prefixes=[f"s3://{BUCKET}/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=value["expected_production_commit"],
        fetcher=fetcher(payloads),
        adapter_materializer=fake_adapter,
        episode_compilation_queue_root=tmp_path / "episode-compilation",
    )

    result = run["results"][0]
    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "launch_preparation_runtime_source_bundle_invalid:"
        "task_evaluation_adapter_bundle_manifest_invalid"
    ]
    pending = tmp_path / "episode-compilation" / "pending"
    assert not pending.exists() or not list(pending.glob("*.json"))

    # A wrapper archive that declares no external layer keeps the pre-existing
    # contract: preparation does not validate it; the compile step does.
    legacy, legacy_payloads = request_with_fetchable_bytes()
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as archive:
        archive.writestr(MANIFEST_NAME, json.dumps({"schema_version": "wrong"}))
    plain_uri = f"s3://{BUCKET}/plain.zip"
    legacy["execution_adapter"]["runtime_source_bundle"] = {
        "uri": plain_uri,
        "digest": _sha(plain),
        "size_bytes": plain.stat().st_size,
    }
    legacy_payloads[plain_uri] = plain.read_bytes()
    legacy_queue = tmp_path / "legacy-queue"
    stage_launch_preparation_request(
        value=legacy, queue_root=legacy_queue, submitted_by="blueprint-webapp"
    )
    legacy_run = worker.process_launch_preparation_queue(
        queue_root=legacy_queue,
        input_root=tmp_path / "legacy-inputs",
        allowed_uri_prefixes=[f"s3://{BUCKET}/"],
        service_account=SERVICE_ACCOUNT,
        source_commit=legacy["expected_production_commit"],
        fetcher=fetcher(legacy_payloads),
        adapter_materializer=fake_adapter,
        episode_compilation_queue_root=tmp_path / "legacy-episode-compilation",
    )
    assert legacy_run["results"][0]["status"] != "blocked", legacy_run["results"][0]


def test_layer_prefix_is_derived_from_the_object_store_contract(tmp_path: Path) -> None:
    """A typed prefix that deviates from the publisher's key shape fails at build time."""

    from blueprint_pipeline.task_evaluation_native_arena_preparation_adapter import (
        external_layer_uri_prefix_for_bucket,
    )

    value, _configured, _construction, _v1 = _bundles(tmp_path)
    receipt_path = _runtime_source_packet(tmp_path)
    packet = receipt_path.parent / PACKET_NAME
    assert external_layer_uri_prefix_for_bucket(BUCKET) == LAYER_PREFIX

    derived, _wrapper, _packet = _v2_wrapper(tmp_path, value, layer_store=tmp_path / "derived", name="derived.zip")
    by_bucket = build_task_evaluation_runtime_source_bundle(
        source_root=receipt_path.parent,
        output_path=tmp_path / "by-bucket.zip",
        expected_production_commit=value["expected_production_commit"],
        runtime_identity=value["runtime"]["identity"],
        external_layer_store_root=tmp_path / "by-bucket-store",
        external_layer_bucket=BUCKET,
        external_layer_min_bytes=packet.stat().st_size,
    )
    assert by_bucket["external_layers"][0]["uri"] == derived["external_layers"][0]["uri"]
    assert by_bucket["sha256"] == derived["sha256"]

    for bad_prefix in (
        LAYER_PREFIX.replace("native-runtime-source-layer", "native-runtime-source-layers"),
        f"s3://{BUCKET}/native-runtime-source-layer",
        LAYER_PREFIX + "/extra",
    ):
        with pytest.raises(
            TaskEvaluationNativeArenaAdapterError,
            match="task_evaluation_adapter_external_layer_prefix_contract_mismatch",
        ):
            build_task_evaluation_runtime_source_bundle(
                source_root=receipt_path.parent,
                output_path=tmp_path / "bad.zip",
                expected_production_commit=value["expected_production_commit"],
                runtime_identity=value["runtime"]["identity"],
                external_layer_store_root=tmp_path / "bad-store",
                external_layer_uri_prefix=bad_prefix,
                external_layer_min_bytes=packet.stat().st_size,
            )
        assert not (tmp_path / "bad.zip").exists()
    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_external_layer_prefix_contract_mismatch",
    ):
        build_task_evaluation_runtime_source_bundle(
            source_root=receipt_path.parent,
            output_path=tmp_path / "mismatch.zip",
            expected_production_commit=value["expected_production_commit"],
            runtime_identity=value["runtime"]["identity"],
            external_layer_store_root=tmp_path / "mismatch-store",
            external_layer_bucket="other-bucket",
            external_layer_uri_prefix=LAYER_PREFIX,
            external_layer_min_bytes=packet.stat().st_size,
        )
    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_external_layer_bucket_invalid",
    ):
        external_layer_uri_prefix_for_bucket("Not A Bucket")
    with pytest.raises(
        TaskEvaluationNativeArenaAdapterError,
        match="task_evaluation_adapter_external_layer_configuration_invalid",
    ):
        build_task_evaluation_runtime_source_bundle(
            source_root=receipt_path.parent,
            output_path=tmp_path / "nothing.zip",
            expected_production_commit=value["expected_production_commit"],
            runtime_identity=value["runtime"]["identity"],
            external_layer_store_root=tmp_path / "nothing-store",
            external_layer_min_bytes=packet.stat().st_size,
        )

    code = adapter_main(
        [
            "build-runtime-source",
            "--source-root",
            str(receipt_path.parent),
            "--output",
            str(tmp_path / "cli-bucket.zip"),
            "--expected-production-commit",
            value["expected_production_commit"],
            "--runtime-id",
            value["runtime"]["identity"]["id"],
            "--runtime-version",
            value["runtime"]["identity"]["version"],
            "--external-layer-store-root",
            str(tmp_path / "cli-bucket-store"),
            "--external-layer-bucket",
            BUCKET,
            "--external-layer-min-bytes",
            str(packet.stat().st_size),
        ]
    )
    assert code == 0
