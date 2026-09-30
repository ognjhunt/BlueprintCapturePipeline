# Covers (for impacted-test selection):
#   src/blueprint_pipeline/arena_provider_output_streaming.py
#   src/blueprint_pipeline/adp_isaac_lab_arena_vast.py
#   src/blueprint_pipeline/native_task_arena_vast.py
#   src/blueprint_pipeline/policy_canary_output_members.py
#   src/blueprint_pipeline/provider_output_promotion.py
#   src/blueprint_pipeline/provider_output_range_ingestion.py
#   src/blueprint_pipeline/provider_output_remote_collection.py
#   src/blueprint_pipeline/control_plane_disk_budget.py
#   src/blueprint_pipeline/task_evaluation_policy_canary_dispatcher.py
#   tests/provider_output_fixtures.py
"""Stream Quick-10 provider output behind BLUEPRINT_POLICY_CANARY_OUTPUT_DELIVERY (plan 15, 15.C3).

The lane runs for real: staging, the Vast adapter and the watchdog are the only
doubles. Spaces (staging) and B2 (the artifact store) are served over the real
range transport by ``StagedSpaces`` and ``VirtualCasClient``; the fake adapter
drives the real ``RemoteProviderOutputCollector`` inside its paid window.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from blueprint_pipeline import adp_isaac_lab_arena_vast as arena
from blueprint_pipeline import arena_provider_output_streaming as streaming
from blueprint_pipeline import native_task_arena_paired_witness_staging as paired
from blueprint_pipeline import native_task_arena_vast as native
from blueprint_pipeline import provider_output_promotion_records as records
from blueprint_pipeline import provider_output_range_transport as transport
from blueprint_pipeline import task_evaluation_configured_scene_object_store as scene_store
from blueprint_pipeline.common import write_json
from blueprint_pipeline.control_plane_disk_usage import tree_usage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import PROVIDER_RESULT_FILENAME
from blueprint_pipeline.policy_canary_output_members import DELIVERY_ENV, POLICY_CANARY_OUTPUT_CONTRACT
from blueprint_pipeline.wam_provider_object_store import SCHEMA_VERSION, STAGING_MANIFEST_FILENAME
from tests.provider_output_fixtures import (
    DEFLATED,
    STORED,
    Entry,
    Zeros,
    build_zip,
    quick10_production_shape,
    virtual_sha256,
)
from tests.test_provider_output_promotion import World

RESULT = PROVIDER_RESULT_FILENAME
PREFIX = "native_task_arena_policy_canary_session"
INSTANCE = 49_247_792
CHUNK = 1024**2


def _aggregate(**extra) -> dict:
    value = {"schema_version": "native_task_arena_policy_canary_session_result.v1",
             "status": "runtime_completed_unqualified_pending_closeout", "run_kind": "internal_policy_canary",
             "claim_ceiling": "diagnostic_policy_execution", "learned_policy_rollout_count": 20,
             "candidate_policy_queried": True, "scene_promotion_performed": False,
             "official_ranking_performed": False, "episodes": [], "artifact_inventory": [], "blockers": [],
             **extra, "result_digest": ""}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def quick10_run_archive(*, frames_per_camera=45, png_bytes=60_000, mp4_bytes=5_000_000,
                        policy_requests_per_episode=45, policy_request_bytes=100_000, aggregate=None):
    """A Quick-10-shaped output at production member counts, about 1 GB: real JSON, zero-run bulk.

    The layout is ``quick10_production_shape``'s (120 MP4s, 5,400 PNG frames, 900 policy
    requests); the JSON members are small real documents, the aggregate a transport-complete
    result. Plan 15's acceptance allows 1 GB with the member counts kept.
    """
    shape = quick10_production_shape(frames_per_camera=frames_per_camera, png_bytes=png_bytes,
                                     mp4_bytes=mp4_bytes, policy_requests_per_episode=policy_requests_per_episode)
    payloads: dict[str, bytes | Zeros] = {}
    for name, size in shape.items():
        parts = PurePosixPath(name).parts
        if name == RESULT:
            payloads[name] = json.dumps(aggregate or _aggregate(), sort_keys=True).encode()
        elif "policy-requests" in parts:
            payloads[name] = Zeros(policy_request_bytes)
        elif name.endswith(".json"):
            payloads[name] = json.dumps({"member": name, "episode_rows": list(range(24))}).encode()
        else:
            payloads[name] = Zeros(size)
    archive = build_zip([Entry(name, data, method=STORED if isinstance(data, Zeros) else DEFLATED)
                         for name, data in payloads.items()])
    return archive, payloads


class Lane:
    """One arena lane with its paid-window seams replaced; Spaces and B2 behind the real transport."""

    def __init__(self, tmp_path: Path, monkeypatch):
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self.world = World(tmp_path / "doubles", monkeypatch, witness=False)
        self.ledger = tmp_path / "ledger"
        self.samples: list[tuple[str, int]] = []
        self.staged: list[bool] = []
        self.events: list[str] = []
        self.adapter_calls: list[dict] = []
        self.adapter_phase_ranges: list = []
        monkeypatch.setenv(streaming.RESERVATION_ROOT_ENV, str(self.ledger))
        monkeypatch.setattr(scene_store, "_artifact_object_store_client", lambda: (self.world.cas, self.world.cas.bucket))
        monkeypatch.setattr(transport, "_open_with_policy", self._route)
        monkeypatch.setattr(streaming, "disk_usage_provider", self._usage)
        monkeypatch.setattr(arena, "utc_now_iso", lambda: "2026-09-28T00:00:00Z")
        monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", self._stage)
        monkeypatch.setattr(arena, "require_pre_spend_preflight", lambda **_kwargs: {"status": "PASS", "blockers": []})
        monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)
        handle = SimpleNamespace(pod_name_prefix="blueprint-native-task-policy-canary-",
                                 started_instance_id_path=tmp_path / "started_vast_instance_id.txt")
        monkeypatch.setattr(arena, "arm_independent_vast_watchdog",
                            lambda **_kwargs: ({"status": "armed", "blockers": []}, handle))
        monkeypatch.setattr(arena, "close_independent_vast_watchdog", self._close_watchdog)
        monkeypatch.setattr(paired, "paired_witness_secret_paths", lambda *_args, **_kwargs: {})
        bundle = tmp_path / "bundle.zip"
        bundle.write_bytes(b"provider bundle bytes")
        self.bundle = {"status": "ready", "bundle_path": str(bundle), "bundle_sha256": arena._file_sha256(bundle),
                       "protocol_digest": "sha256:" + "b" * 64, "container_image": "immutable-image",
                       "implementation_commit": "c" * 40}

    # -- doubles ----------------------------------------------------------------
    def _route(self, request, timeout, policy):
        served = self.world.cas if urlparse(request.full_url).hostname == "b2.example.invalid" else self.world.spaces
        return served.opener(request, timeout, policy)

    def _usage(self, path):
        used = sum(tree_usage(self.attempt / name).apparent_bytes
                   for name in ("immutable_execution", ".provider_output_ingestion")) if self.attempt else 0
        self.samples.append((Path(path).name, used))
        return SimpleNamespace(total=2 * 1024**4, used=used, free=2 * 1024**4 - used)

    def _stage(self, *, job_dir, output_promotion_required=False, **_kwargs):
        staging = Path(job_dir)
        staging.mkdir(parents=True)
        self.staged.append(output_promotion_required)
        keys = self.world.keys
        manifest = {"schema_version": SCHEMA_VERSION, "status": "completed",
                    "object_store": {"key_prefix": "blueprint/task"}, "bundle_key": keys["bundle"],
                    "output_key": keys["output"],
                    **({"output_promotion_required": True} if output_promotion_required else {})}
        (staging / STAGING_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for name, role in (("provider_bundle_url.txt", "bundle"), ("provider_output_put_url.txt", "output"),
                           ("provider_output_get_url.txt", "output")):
            (staging / name).write_text(self.world.spaces.url(keys[role]) + "\n", encoding="utf-8")
        self.world.spaces.put(keys["bundle"], b"provider bundle bytes", '"bundle"')
        return {"status": "completed"}

    def _close_watchdog(self, **kwargs):
        self.events.append("watchdog_closed")
        return {"status": "provider_terminal", "instance_ids": [INSTANCE], "provider_absence_confirmed": True}

    @property
    def attempt(self) -> Path | None:
        attempts = sorted((self.tmp_path / "job" / "attempts").glob("attempt_*"))
        return attempts[-1] if attempts else None

    def adapter(self, archive=None, *, upload=True, ssh_zip: bytes | None = None, download_bytes=None):
        """The Vast adapter's contract: the provider uploads, the output is collected, then teardown."""

        def fake(*, job_dir, **kwargs):
            self.adapter_calls.append(kwargs)
            provider = Path(job_dir)
            zip_path = Path(kwargs["provider_runtime_output_zip"])
            output_key = self.world.keys["output"]
            if upload:
                self.world.stage("output", archive)
            collector = kwargs.get("provider_output_collector")
            transfer = None
            if collector is not None:
                transfer = collector(url=kwargs["provider_output_get_url"], output_path=zip_path,
                                     minimum_free_bytes=0)
                self.adapter_phase_ranges = [row["range"] for row in self.world.spaces.requests(output_key)]
            elif upload:
                zip_path.write_bytes(download_bytes if download_bytes is not None else archive.to_bytes())
            command = {"provider_bundle_kind": PREFIX, "provider_runtime_output_zip_path": str(zip_path)}
            if transfer is not None and transfer["status"] == "completed":
                command["provider_output_remote_observation"] = dict(transfer["remote_object"])
            elif ssh_zip is not None:
                zip_path.write_bytes(ssh_zip)  # pinned SSH recovery lands a whole, unguarded ZIP
                command["provider_output_download_manifest"] = {"ssh_recovery": {"status": "completed"}}
            write_json(provider / "vast_provider_command_result.json", command)
            write_json(provider / "vast_provider_adapter_result.json",
                       {"status": "completed", "vast_instance_ids": [INSTANCE], "continuing_spend_from_this_run": False})
            write_json(provider / "vast_teardown_manifest.json",
                       {"status": "completed", "vast_instance_ids": [INSTANCE], "continuing_spend_from_this_run": False,
                        "runner_gpu_teardown_completed": True, "generated_at": "2026-09-28T01:00:00+00:00"})
            self.events.append("adapter_returned_after_teardown")
            return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.4, "vast_instance_ids": [INSTANCE],
                    "continuing_spend_from_this_run": False, "provider_create_attempted": True,
                    "generated_at": "2026-09-28T00:10:00+00:00"}

        return fake

    def run_session(self, adapter, *, stream=True, job_dir=None):
        """The real Quick-10 session and lane, its authority, bundle and witness checks stubbed."""
        monkeypatch = self.monkeypatch
        if stream:
            monkeypatch.setenv(DELIVERY_ENV, "stream")
        else:
            monkeypatch.delenv(DELIVERY_ENV, raising=False)
        monkeypatch.setattr(arena, "run_vast_provider_adapter", adapter)
        monkeypatch.setattr(native, "validate_policy_canary_session_authority", lambda value: value)
        monkeypatch.setattr(native, "validate_policy_canary_provider_bundle", lambda *_args, **_kwargs: self.bundle)
        monkeypatch.setattr(native, "consume_session_authority_once",
                            lambda *_args, **_kwargs: {"status": "consumed", "blockers": []})
        monkeypatch.setattr(native, "_policy_provider_transfer_byte_budget", lambda _candidate: (100, 20))
        monkeypatch.setattr(paired, "build_paired_witness_binding",
                            lambda *_args, **_kwargs: {"maximum_archive_bytes": 1_000_000})
        authority = {"hard_cap_usd": 4.0, "hard_ttl_seconds": 14_400, "authority_digest": "sha256:" + "a" * 64,
                     "resource_name": "blueprint-native-task-policy-canary-" + "a" * 32}
        return native.run_native_task_arena_policy_canary_session_vast(
            job_dir=job_dir or self.tmp_path / "job", prepared_bundle=self.bundle, session_authority=authority,
            paid_resource_admission_grant=object(), execute=True, hard_cap_usd=4.0, hard_ttl_seconds=14_400,
            provider_runtime_environment={"BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "640x360"})

    def history(self) -> list[dict]:
        path = self.ledger / "history" / "policy_canary_output.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


@pytest.fixture
def lane(tmp_path, monkeypatch):
    return Lane(tmp_path, monkeypatch)


def _members(root: Path) -> dict[str, int]:
    return {path.relative_to(root).as_posix(): path.stat().st_size for path in root.rglob("*") if path.is_file()}


def _overlaps(first, last, row) -> bool:
    return first < row["record_end_offset"] and last >= row["local_header_offset"]


def test_quick10_streams_needed_members_without_zip_or_mp4_copy(lane, tmp_path):
    archive, payloads = quick10_run_archive()
    assert 0.9e9 < archive.size < 1.2e9 and sum(name.endswith(".mp4") for name in payloads) == 120

    result = lane.run_session(lane.adapter(archive))

    assert result["status"] == "completed", result["blockers"]
    assert result["provider_output_delivery"] == "stream" and result["archive_durable"] is True
    assert lane.staged == [True] and lane.events == ["adapter_returned_after_teardown", "watchdog_closed"]
    attempt = Path(result["attempt_root"])
    evidence = attempt / "immutable_execution"
    index = json.loads((attempt / "provider_output_member_index.v1.json").read_text())
    rows = {row["path"]: row for row in index["members"] if row["kind"] == "file"}
    needed = POLICY_CANARY_OUTPUT_CONTRACT.paths(index)
    # The evidence root holds exactly the contract's set, byte for byte the archive's members.
    assert _members(evidence) == {path: rows[path]["size"] for path in needed}
    assert sum(_members(evidence).values()) == POLICY_CANARY_OUTPUT_CONTRACT.needed_bytes(index)
    assert result["native_control_result_path"] == str(evidence / RESULT)
    assert result["native_control_result_digest"] == json.loads(payloads[RESULT])["result_digest"]
    # No ZIP, no MP4 copy, and no file carrying any bulk member's bytes.
    bulk_sizes = {data.size for data in payloads.values() if isinstance(data, Zeros)}
    for path in tmp_path.rglob("*"):
        assert path.name != "vast_provider_runtime_output.zip" and "vast_provider_runtime_output_videos" not in path.parts
        assert not path.name.endswith((".mp4", ".png")), path
        if path.is_file() and path.stat().st_size in bulk_sizes:
            assert path.read_bytes() != bytes(path.stat().st_size), path
    # B2: no ingestion range touched a bulk member's record.
    output_key = urlparse(index["archive"]["durable_reference"]["uri"]).path.lstrip("/")
    ingested = [span for key, span in lane.world.cas.ranged_requests() if key == output_key]
    assert ingested
    for first, last in ingested:
        assert not any(_overlaps(first, last, rows[name]) for name, data in payloads.items() if isinstance(data, Zeros))
    # Spaces: two whole-object GETs (index and promotion); the adapter's ranges read only the
    # end records, the central directory and the small result record the inspection opens.
    spaces_key = lane.world.keys["output"]
    assert lane.world.spaces.whole_object_gets(spaces_key) == 2
    directory = index["directory"]
    allowed = [(directory["offset"], archive.size - 1), (rows[RESULT]["local_header_offset"], rows[RESULT]["record_end_offset"])]
    phase = [span for span in lane.adapter_phase_ranges if span not in (None, (0, 0))]
    assert phase and all(any(first <= end and last >= start for start, end in allowed) for first, last in phase)
    # Host growth during ingestion never exceeded the needed set, one chunk and the metadata.
    metadata = sum(_members(attempt / ".provider_output_ingestion").values())
    ingestion = [used for name, used in lane.samples if name == "immutable_execution"]
    assert ingestion and max(ingestion) - ingestion[0] <= POLICY_CANARY_OUTPUT_CONTRACT.needed_bytes(index) + CHUNK + metadata
    # The hold was forecast before the run, shrunk after indexing, and released with the outcome.
    [sample] = lane.history()
    assert (sample["workload"], sample["outcome"]) == ("quick10_needed_members", "completed")
    assert sample["observed_bytes"] == POLICY_CANARY_OUTPUT_CONTRACT.needed_bytes(index)
    assert sample["reserved_bytes"] == result["provider_output_needed_set"]["hold_bytes"] < 1024**3
    assert list(lane.ledger.glob("*.json")) == []
    # Promotion is bound to this staging manifest; every staged object is gone.
    receipt = records.load_promotion_receipt(attempt / "object_store_staging",
                                             staging_manifest_sha256=records.staging_manifest_sha256(
                                                 attempt / "object_store_staging"))
    assert receipt["status"] == "promoted" and receipt["source"] == "remote_observation"
    assert result["all_staged_objects_absent"] is True and result["provider_closeout"]["all_staged_objects_absent"]
    # M1 (T1): the attempt's host bytes stay within the needed set, the index, the metadata and 32 MiB.
    host = result["provider_output_host_bytes"]
    index_bytes = (attempt / "provider_output_member_index.v1.json").stat().st_size
    assert host["attempt_root_allocated_bytes"] <= (
        POLICY_CANARY_OUTPUT_CONTRACT.needed_bytes(index) + index_bytes + metadata + 32 * CHUNK)
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text())
    assert manifest["status"] == "completed" and {
        "provider_output_member_index", "provider_output_promotion", "provider_output_ingestion_receipt",
        "provider_output_member_view"} <= set(manifest["required_roles"])
    remote_rows = [row for row in manifest["files"] if row.get("location") == "archive_member"]
    assert len(remote_rows) == len(rows) - len(needed)


def _small_archive(**extra):
    archive, payloads = quick10_run_archive(frames_per_camera=2, png_bytes=4_096, mp4_bytes=32_768,
                                            policy_requests_per_episode=2, policy_request_bytes=2_048, **extra)
    return archive, payloads


def test_promotion_failure_keeps_the_staged_output_and_blocks(lane):
    archive, _ = _small_archive()

    def refused(**_kwargs):
        raise RuntimeError("b2 unavailable")

    lane.monkeypatch.setattr(lane.world.cas, "create_multipart_upload", refused)

    result = lane.run_session(lane.adapter(archive))

    assert result["status"] == "blocked" and result["archive_durable"] is False
    assert f"{PREFIX}_provider_output_promotion_failed" in result["blockers"]
    assert f"{PREFIX}_object_store_provider_zero_not_proven" in result["blockers"]
    # Nothing durable, so nothing deleted: the staged output waits for the door's resume.
    assert lane.world.keys["output"] in lane.world.spaces.stores
    assert result["all_staged_objects_absent"] is False
    assert result["native_control_result_path"] is None and result["native_control_result_digest"] is None
    attempt = Path(result["attempt_root"])
    assert not (attempt / "immutable_execution").exists()
    receipt = json.loads((attempt / "object_store_staging" / records.RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "failed" and receipt["blockers"]
    assert result["provider_output_promotion"]["status"] == "failed"
    assert lane.history() == [] and list(lane.ledger.glob("*.json")) == []  # released, no sample


@pytest.mark.parametrize("limit", ["needed_set_budget", "forecast_hold"])
def test_needed_set_over_budget_blocks_after_run_with_the_archive_durable(lane, limit):
    from blueprint_pipeline import policy_canary_output_members as output_members

    archive, _ = _small_archive()
    if limit == "needed_set_budget":
        lane.monkeypatch.setattr(output_members, "POLICY_CANARY_OUTPUT_CONTRACT",
                                 output_members.PolicyCanaryOutputContract(needed_set_budget_bytes=1_000))
        code = f"{PREFIX}_provider_output_needed_set_over_budget"
    else:
        # The operator declared a footprint smaller than this run needs: growth is never admitted.
        lane.monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_POLICY_CANARY_OUTPUT_BYTES", str(4 * CHUNK))
        code = f"{PREFIX}_provider_output_disk_budget_exceeded_after_run"

    result = lane.run_session(lane.adapter(archive))

    assert result["status"] == "blocked" and code in result["blockers"]
    # The paid output is durable in B2 and the staged objects are gone; the door can ingest later.
    assert result["archive_durable"] is True and result["provider_output_promotion"]["status"] == "promoted"
    assert result["all_staged_objects_absent"] is True
    reference = result["provider_output_promotion"]["durable_reference"]
    assert urlparse(reference["uri"]).path.lstrip("/") in lane.world.cas.objects
    attempt = Path(result["attempt_root"])
    assert not (attempt / "immutable_execution").exists() and lane.world.cas.ranged_requests() == []
    assert result["native_control_result_path"] is None  # review I8: never a path to a partial tree
    needed = result["provider_output_needed_set"]
    assert needed["bytes"] > (1_000 if limit == "needed_set_budget" else 0)
    assert list(lane.ledger.glob("*.json")) == []


IDENTITY = "cell_runs/00/policy_canary_static_startup_preflight.v1.json"  # each cell's worker seals one


@pytest.mark.parametrize("tampered", [False, True])
def test_ingestion_records_the_native_inventory_outcome_and_never_blocks_on_it(lane, tampered):
    """Review minor 6 (design 4): once ingested, the native inventory is checked against Blueprint's
    index -- the identity document bound to the run, every inventory row by digest, bulk members
    included, without their bytes. The outcome is sealed into the ingestion receipt and shown in
    the lane result; it is never a blocker (download mode never runs it, delivery re-verifies)."""
    run_id, inputs_digest = "scene-839873-canary-1", "sha256:" + "4" * 64
    identity = {"schema_version": "policy_canary_static_startup_preflight.v1", "status": "passed",
                "run_id": run_id, "runtime_inputs_digest": inputs_digest, "result_digest": ""}
    identity["result_digest"] = canonical_digest(identity, digest_field="result_digest")
    frame, request = b"\x89PNG" + bytes(4_000), json.dumps({"observation": [0.5] * 8}).encode()
    frame_path, request_path = "episodes/media/e/frames/external/000000.png", "cell_runs/00/policy-requests/000000.json"
    rows = [{"role": role, "relative_path": path, "size_bytes": len(data),
             "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
            for role, path, data in (("lossless_frame", frame_path, frame), ("policy_request", request_path, request))]
    if tampered:
        rows[0]["sha256"] = "sha256:" + "0" * 64
    aggregate = _aggregate(artifact_inventory=rows, artifact_inventory_digest=canonical_digest({"value": rows}))
    archive = build_zip([Entry(RESULT, json.dumps(aggregate, sort_keys=True).encode()),
                         Entry(IDENTITY, json.dumps(identity, sort_keys=True).encode()),
                         Entry(frame_path, frame, method=STORED), Entry(request_path, request)])
    lane.bundle.update(runtime_inputs_digest=inputs_digest, static_startup_preflight={"run_id": run_id})

    result = lane.run_session(lane.adapter(archive))

    assert result["status"] == "completed", result["blockers"]
    assert result["provider_output_native_inventory_binding"] == {
        "identity_document": IDENTITY, "result_document": RESULT, "run_id": run_id,
        "runtime_inputs_digest": inputs_digest}
    attempt = Path(result["attempt_root"])
    receipt = json.loads((attempt / ".provider_output_ingestion" / "receipt.json").read_text())
    native = receipt["native_inventory"]
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert result["provider_output_ingestion"]["native_inventory"] == native
    if tampered:
        assert native == {"status": "failed", "code": "provider_output_native_artifact_digest_mismatch"}
    else:
        assert native == {"status": "verified", "identity_document_digest": identity["result_digest"],
                          "result_document_digest": aggregate["result_digest"], "verified_native_file_count": 2,
                          "episode_qualification_performed": False, "image_qualification_performed": False,
                          "scientific_finalization_pending": True}
    # The frame and the request stayed in the archive: checked by their index digests, never fetched.
    assert sorted(_members(attempt / "immutable_execution")) == sorted([IDENTITY, RESULT])


@pytest.mark.parametrize("step", ["hold_resize", "view_descriptor"])
def test_an_untyped_failure_after_the_paid_run_still_seals_a_blocked_lane_result(lane, step):
    """Review minor 7: once the paid run is over every failure is evidence. An OSError from the hold's
    resize or the view descriptor's write -- neither a typed refusal -- seals the lane result
    blocked with ``…_provider_output_ingestion_failed:<Type>`` and the archive still durable."""
    from blueprint_pipeline import control_plane_disk_budget as budget
    from blueprint_pipeline import provider_output_member_view as view

    def failed(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    if step == "hold_resize":
        lane.monkeypatch.setattr(budget.DiskReservation, "resize", failed)
    else:
        lane.monkeypatch.setattr(view, "write_member_view_descriptor", failed)
    archive, _ = _small_archive()

    result = lane.run_session(lane.adapter(archive))

    assert result["status"] == "blocked"
    assert f"{PREFIX}_provider_output_ingestion_failed:PermissionError" in result["blockers"]
    assert result["archive_durable"] is True and result["provider_output_promotion"]["status"] == "promoted"
    assert result["native_control_result_path"] is None and result["all_staged_objects_absent"] is True
    sealed = json.loads((Path(result["attempt_root"]) / "adp_arena_vast_result.json").read_text())
    assert sealed["blockers"] == result["blockers"]
    assert sealed["visual_evidence"]["media_gap"] == {
        "type": "provider_output_not_ingested", "reason": f"{PREFIX}_provider_output_ingestion_failed:PermissionError"}
    assert list(lane.ledger.glob("*.json")) == []  # the hold was released


def test_ssh_fallback_in_stream_mode_publishes_then_ingests_by_range(lane, tmp_path):
    archive, payloads = _small_archive()

    # The provider's upload never landed: the collector sees 404 and pinned SSH recovery runs.
    result = lane.run_session(lane.adapter(upload=False, ssh_zip=archive.to_bytes()))

    assert result["status"] == "completed", result["blockers"]
    attempt = Path(result["attempt_root"])
    promotion = result["provider_output_promotion"]
    assert (promotion["status"], promotion["source"]) == ("promoted", "ssh_local_zip")
    assert promotion["archive_sha256"] == virtual_sha256(archive)
    # Published to B2 with a full readback, then removed behind its pointer.
    assert not (attempt / "vast_provider_run" / "vast_provider_runtime_output.zip").exists()
    receipt = json.loads((attempt / "object_store_staging" / records.RECEIPT_FILENAME).read_text())
    assert receipt["local_copy_removed_after_verified_promotion"] is True
    assert ("upload_file", urlparse(promotion["durable_reference"]["uri"]).path.lstrip("/")) in lane.world.cas.calls
    # The needed members then came from B2 by range, one request each.
    index = json.loads((attempt / "provider_output_member_index.v1.json").read_text())
    needed = POLICY_CANARY_OUTPUT_CONTRACT.paths(index)
    assert _members(attempt / "immutable_execution") == {
        row["path"]: row["size"] for row in index["members"] if row["path"] in needed}
    assert len(lane.world.cas.ranged_requests()) == len(needed)
    assert not any(path.name == "vast_provider_runtime_output.zip" for path in tmp_path.rglob("*"))


def test_forecast_hold_is_taken_before_consumption_and_refused_without_room(lane):
    """Review I3: the needed members' hold is admitted before the authority is consumed, so a
    host without room refuses before any spend; during the run it is live beside the dispatch hold."""
    archive, _ = _small_archive()
    seen = []

    def adapter(**kwargs):
        seen.extend(json.loads(path.read_text())["expected_bytes"] for path in lane.ledger.glob("*.json"))
        return lane.adapter(archive)(**kwargs)

    assert lane.run_session(adapter)["status"] == "completed"
    # The budget's hold (review minor 5), not the role's 1 GiB ceiling, from before consumption
    # through the paid window.
    assert seen == [POLICY_CANARY_OUTPUT_CONTRACT.forecast_hold_bytes()] and seen[0] < 1024**3

    full = SimpleNamespace(total=100 * 1024**3, used=99 * 1024**3, free=1024**3)
    lane.monkeypatch.setattr(streaming, "disk_usage_provider", lambda _path: full)
    lane.monkeypatch.setattr(native, "consume_session_authority_once",
                             lambda *_args, **_kwargs: pytest.fail("consumed before the hold was admitted"))
    lane.monkeypatch.setattr(native, "run_arena_native_control_vast",
                             lambda **_kwargs: pytest.fail("the lane ran without the hold"))
    refused = native.run_native_task_arena_policy_canary_session_vast(
        job_dir=lane.tmp_path / "job-2", prepared_bundle=lane.bundle,
        session_authority={"hard_cap_usd": 4.0, "hard_ttl_seconds": 14_400, "authority_digest": "sha256:" + "a" * 64,
                           "resource_name": "blueprint-native-task-policy-canary-" + "a" * 32},
        paid_resource_admission_grant=object(), execute=True, hard_cap_usd=4.0, hard_ttl_seconds=14_400,
        provider_runtime_environment={"BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "640x360"})
    assert refused["blockers"] == ["policy_canary_output_disk_admission_refused"]
    assert refused["provider_mutations_performed"] == 0
    assert refused["provider_output_admission"]["reason"].startswith(
        "control_plane_disk_budget_exceeded:policy_canary_output:")
    assert list(lane.ledger.glob("*.json")) == []


# -- Download mode is today's path, byte for byte ------------------------------------------

GOLDEN_LANE_RESULT_SHA256 = "ba00b7bbcfa3f4a9e6ee522767d9ddef64af4d99e81b3071b4dcd1ffabc96d76"
GOLDEN_ARTIFACT_MANIFEST_SHA256 = "3eeb482399c882929c5cb4fad547fe36f9f5875f48df4fc7e1574c33e4379afa"


def _golden_zip() -> bytes:
    import io
    import zipfile

    native_result = _aggregate()
    files = {RESULT: json.dumps(native_result, sort_keys=True).encode(),
             "cell_runs/00/" + RESULT: b'{"cell": 0}',
             "cell_runs/00/episodes/e.score_receipt.json": b'{"score": 1}',
             "cell_runs/00/episodes/media/e/external.mp4": b"\x00" * 4096,
             "cell_runs/00/episodes/media/e/frames/external/000000.png": b"\x89PNG" + b"\x00" * 100,
             "cell_runs/00/episodes/media/e/policy-requests/000000.json": b'{"request": 0}'}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 28, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            zipped.writestr(info, data)
    return buffer.getvalue()


@pytest.mark.parametrize("explicit", [False, True])
def test_download_mode_lane_result_and_manifest_are_byte_identical(tmp_path, monkeypatch, explicit):
    """The golden digests were taken from PR B's head (87ec78833) before this lane changed."""
    archive = _golden_zip()
    calls = {}

    def fake_stage(*, job_dir, **kwargs):
        calls["stage"] = kwargs
        staging = Path(job_dir)
        staging.mkdir(parents=True)
        for name in ("provider_bundle_url.txt", "provider_output_put_url.txt", "provider_output_get_url.txt"):
            (staging / name).write_text("https://example.invalid/object\n")
        return {"status": "completed"}

    def fake_adapter(*, job_dir, **kwargs):
        calls["adapter"] = kwargs
        provider = Path(job_dir)
        write_json(provider / "vast_provider_adapter_result.json", {"status": "completed", "vast_instance_ids": [7]})
        write_json(provider / "vast_teardown_manifest.json",
                   {"continuing_spend_from_this_run": False, "generated_at": "2026-09-28T01:00:00Z"})
        Path(kwargs["provider_runtime_output_zip"]).write_bytes(archive)
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.1, "vast_instance_ids": [7],
                "continuing_spend_from_this_run": False, "provider_create_attempted": True}

    handle = SimpleNamespace(pod_name_prefix="blueprint-watchdog-", started_instance_id_path=tmp_path / "started.txt")
    monkeypatch.setattr(arena, "utc_now_iso", lambda: "2026-09-28T00:00:00Z")
    monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr(arena, "cleanup_staged_wam_provider_objects", lambda _path: {"all_objects_absent": True})
    monkeypatch.setattr(arena, "run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(arena, "require_pre_spend_preflight", lambda **_kwargs: {"status": "PASS", "blockers": []})
    monkeypatch.setattr(arena, "arm_independent_vast_watchdog",
                        lambda **_kwargs: ({"status": "armed", "blockers": []}, handle))
    monkeypatch.setattr(arena, "close_independent_vast_watchdog", lambda **_kwargs: {"status": "provider_terminal"})
    monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 60)
    monkeypatch.setattr(streaming, "promote", lambda **_kwargs: pytest.fail("download mode promoted"))
    bundle_path = tmp_path / "bundle.zip"
    bundle_path.write_bytes(b"bundle")
    result = arena.run_arena_native_control_vast(
        approval_path=".", job_dir=tmp_path / "job", paid_resource_admission_grant=object(), execute=True,
        prepared_bundle={"status": "ready", "bundle_path": str(bundle_path),
                         "bundle_sha256": arena._file_sha256(bundle_path), "protocol_digest": "sha256:" + "b" * 64},
        hard_cap_usd=4.0, hard_ttl_seconds=14_400, expected_output_filename=RESULT,
        provider_bundle_kind=PREFIX, result_schema_version="native_task_arena_policy_canary_session_result.v1",
        blocker_prefix=PREFIX, candidate_policy_query_expected=True, require_independent_watchdog=True,
        **({"provider_output_delivery": "download", "provider_output_member_contract": None,
            "provider_output_reservation": None} if explicit else {}))

    assert result["status"] == "completed", result["blockers"]
    attempt = Path(result["attempt_root"])
    lane_bytes = (attempt / "adp_arena_vast_result.json").read_bytes().replace(str(tmp_path).encode(), b"<tmp>")
    assert hashlib.sha256(lane_bytes).hexdigest() == GOLDEN_LANE_RESULT_SHA256
    assert hashlib.sha256((attempt / "artifact_manifest.json").read_bytes()).hexdigest() == (
        GOLDEN_ARTIFACT_MANIFEST_SHA256)
    assert "output_promotion_required" not in calls["stage"] and "provider_output_collector" not in calls["adapter"]
    assert not any(key.startswith(("provider_output_", "archive_durable")) for key in result)
    # The whole archive was extracted, as always.
    assert (attempt / "immutable_execution/cell_runs/00/episodes/media/e/external.mp4").is_file()


# -- Parity through the dispatcher (plan 15, design 7) --------------------------------------


def _inventory_row(path: Path, root: Path, role: str) -> dict:
    return {"role": role, "relative_path": path.relative_to(root).as_posix(),
            "media_type": {".mp4": "video/mp4", ".png": "image/png", ".jsonl": "application/x-ndjson"}.get(
                path.suffix, "application/json"),
            "size_bytes": path.stat().st_size, "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def _parity_evidence(root: Path, *, controls_monkeypatch=None) -> tuple[dict, dict]:
    """A Quick-10 provider tree every consumer can read: twenty interpretable, projection-complete
    episodes, per-cell child results, policy requests and telemetry (download mode's bytes). With
    ``controls_monkeypatch``, also twenty sealed strict controls under ``control_runs/NN/``: their
    receipts, lossless PNG frames and review videos (review important 3)."""
    from tests.test_policy_canary_control_result_delivery import write_native_controls
    from tests.test_policy_canary_episode_interpretation_closeout import _session as interpretable_session

    root.mkdir(parents=True)
    data, result = interpretable_session(root)
    evidence = data["root"]
    extra = {}
    for role, payload in (("reset_state", {"reset": "exact"}),
                          ("policy_query_receipt", {"candidate_policy_queried": True}),
                          ("action_sequence", [{"step_index": 1, "target_joint_positions_rad": [0.1] * 7}]),
                          ("action_delivery_readback", {"actions_reached_robot": True}),
                          ("task_object_trajectory", {"samples": []})):
        path = evidence / "episodes" / f"episode.{role}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        extra[role] = _inventory_row(path, evidence, role)
    rows = list(extra.values())
    for cell in range(10):
        cell_root = evidence / "cell_runs" / f"{cell:02d}"
        (cell_root / "policy-requests").mkdir(parents=True)
        (cell_root / RESULT).write_text(json.dumps({"selected_cell_index": cell, "status": (
            "runtime_selected_cell_completed_pending_aggregation")}), encoding="utf-8")
        request = cell_root / "policy-requests" / "000000.json"
        request.write_text(json.dumps({"observation": [cell] * 64}), encoding="utf-8")
        rows.append(_inventory_row(request, evidence, "policy_request"))
    telemetry = evidence / "policy_canary_telemetry.jsonl"
    telemetry.write_text('{"episode": "episode-0"}\n', encoding="utf-8")
    rows.append(_inventory_row(telemetry, evidence, "indexed_episode_telemetry"))
    for episode in result["episodes"]:
        episode["evidence_artifacts"].update(extra)
        episode.update(checkpoint_digest="sha256:" + "c" * 64, runtime_identity_digest="sha256:" + "d" * 64,
                       candidate_policy_queried=True, actions_reached_robot=True, arm_moved=True,
                       policy_outcome_interpretable=True)
    result["artifact_inventory"].extend(rows)
    if controls_monkeypatch is not None:
        fields, control_rows = write_native_controls(evidence, controls_monkeypatch)
        result.update(fields)
        result["artifact_inventory"].extend(control_rows)
    result.update(schema_version="native_task_arena_policy_canary_session_result.v1",
                  status="runtime_completed_unqualified_pending_closeout", learned_policy_rollout_count=20,
                  candidate_policy_queried=True, scene_promotion_performed=False, official_ranking_performed=False,
                  blockers=[])
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (evidence / RESULT).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return data, result


def _rebind_contract(activation_result: Path, setup_path: Path, contract: dict) -> None:
    """Point the dispatcher test inputs at ``contract`` (the evidence's), re-sealing every digest."""
    from tests.test_task_evaluation_policy_canary_dispatcher import _record

    binding = {"task_success_contract": contract, "task_success_contract_digest": contract["contract_digest"]}

    def reseal(path: Path, field: str, **updates) -> dict:
        value = {**json.loads(path.read_text()), **binding, **updates}
        value[field] = canonical_digest(value, digest_field=field)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return value

    result = json.loads(activation_result.read_text())
    runtime_path = Path(result["policy_canary_runtime_inputs_path"])
    activation = reseal(runtime_path.parent / "task_evaluation_policy_campaign_activation.v1.json",
                        "activation_digest")
    reseal(runtime_path, "runtime_inputs_digest", activation_digest=activation["activation_digest"])
    reseal(activation_result, "result_digest")
    setup = json.loads(setup_path.read_text())
    for name in ("pi05_execution_spec", "groot_execution_spec"):
        spec = Path(setup["records"][name]["path"])
        spec.write_text(json.dumps({**json.loads(spec.read_text()), **binding}, sort_keys=True) + "\n")
        setup["records"][name] = _record(spec)
    setup_path.write_text(json.dumps(setup, sort_keys=True) + "\n", encoding="utf-8")
    reseal(setup_path, "setup_digest", activation_digest=activation["activation_digest"])


def _relative(value, root: Path):
    return json.loads(json.dumps(value).replace(str(root.resolve()), "<run>").replace(str(root), "<run>"))


def _provider_zero() -> dict:
    zero = {"schema_version": "task_evaluation_policy_canary_vast_provider_zero.v1",
            "status": "provider_zero_confirmed", "api_confirmed": True, "provider_zero_verified": True,
            "live_instance_count": 0, "blockers": [], "receipt_digest": ""}
    zero["receipt_digest"] = canonical_digest(zero, digest_field="receipt_digest")
    return zero


def _dispatch(root: Path, monkeypatch, *, archive, data, rights, stream: bool, post_billing) -> dict:
    """One Quick-10 through the real dispatcher, the real session and lane underneath it."""
    from blueprint_pipeline import task_evaluation_policy_canary_dispatcher as dispatcher
    from blueprint_pipeline.episode_interpretation import DeterministicFixtureInterpreter
    from tests.test_episode_interpretation import _output
    from tests.test_policy_canary_episode_interpretation_closeout import _FixtureRunner
    from tests.test_task_evaluation_policy_canary_dispatcher import COMMIT, _echoing_website, _inputs

    lane = Lane(root, monkeypatch)
    activation_result, setup_path, _ = _inputs(root)
    _rebind_contract(activation_result, setup_path, data["contract"])
    monkeypatch.setattr(dispatcher, "_materialize_official_billing_if_posted", post_billing)
    monkeypatch.setattr(dispatcher, "validate_vast_official_same_goal_reconciliation", lambda _path: {})
    monkeypatch.setattr(dispatcher, "build_policy_canary_session_bundle", lambda **kwargs: write_json(
        Path(kwargs["job_dir"]) / "native_task_arena_policy_canary_session_bundle_receipt.v1.json",
        {"bundle_sha256": "sha256:" + "b" * 64}) or {"bundle_sha256": "sha256:" + "b" * 64})
    monkeypatch.setattr(dispatcher, "validate_provider_bundle", lambda value, **_kwargs: value)
    output = root / "dispatch"

    def allocator(argv):
        lane_result = lane.run_session(lane.adapter(archive), stream=stream,
                                       job_dir=Path(argv[argv.index("--adp-job-dir") + 1]))
        write_json(Path(argv[argv.index("--adapter-output") + 1]), lane_result)
        return 0

    runner = _FixtureRunner(DeterministicFixtureInterpreter(_output(data)))
    zero = _provider_zero()
    receipt = dispatcher.dispatch_policy_canary_activation(
        activation_result_path=activation_result, execution_setup_path=setup_path, output_root=output,
        implementation_commit=COMMIT, execute=True, allocator_runner=allocator,
        provider_zero_collector=lambda: zero, sync_runner=_echoing_website(monkeypatch),
        progress_sync_runner=lambda **_kwargs: {"status": "succeeded", "response": {"status": "recorded"}},
        episode_interpretation_runner=runner, episode_interpretation_rights_root=rights)
    return {"root": output, "receipt": receipt, "lane": json.loads((output / "allocator_result.json").read_text()),
            "world": lane.world, "interpreter_calls": runner.interpreter.call_count}


def test_streamed_and_downloaded_quick10_seal_identical_evidence(tmp_path, monkeypatch):
    """Design 7: the same provider output, downloaded or streamed, seals the same evidence through
    the real dispatcher -- native result, joined terminal result, registry, public delivery and
    projection, interpretation, billing verdict, teardown, provider-zero and cleanup rows."""
    from blueprint_pipeline import vast_official_billing_extractor as billing
    from blueprint_pipeline.episode_interpretation import DeterministicFixtureInterpreter
    from tests.provider_output_fixtures import zip_tree
    from tests.test_episode_interpretation import _output
    from tests.test_policy_canary_episode_interpretation_closeout import _rights_for_all

    data, result = _parity_evidence(tmp_path / "source", controls_monkeypatch=monkeypatch)
    archive = zip_tree(data["root"])
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, DeterministicFixtureInterpreter(_output(data)))

    terminal_evidence = {}

    def post_billing(**kwargs):
        # The real terminal-evidence gate decides whether the charge can post. Its full record binds
        # the lane result and the artifact manifest, which differ by design (documented), so the
        # sealed receipt here keeps what must match and the full evidence is compared below.
        evidence = billing._terminal_evidence(instance_id=INSTANCE, terminal_result_path=kwargs["adapter_result_path"])
        terminal_evidence[Path(kwargs["output_path"]).parent.parent.name] = evidence
        write_json(Path(kwargs["output_path"]), {
            "status": "reconciled_official_posted_charges", "official_total_usd": 0.38, "instance_ids": [INSTANCE],
            "terminal_verdict": {key: evidence[key] for key in (
                "terminal_status", "provider_absence_confirmed", "provider_zero_verified",
                "continuing_spend_from_this_run", "retry_cap")},
            "bound_records": {key: {field: evidence[key][field] for field in ("sha256", "size_bytes")}
                              for key in ("provider_adapter_result", "teardown_manifest", "post_teardown_provider_zero")}})
        return True

    sealed = {mode: _dispatch(tmp_path / mode, monkeypatch, archive=archive, data=data, rights=rights,
                              stream=mode == "stream", post_billing=post_billing)
              for mode in ("download", "stream")}

    download, stream = sealed["download"], sealed["stream"]
    assert download["receipt"]["status"] == stream["receipt"]["status"] == "completed_unqualified", (
        download["receipt"], stream["receipt"])
    assert stream["lane"]["provider_output_delivery"] == "stream" and "provider_output_delivery" not in download["lane"]
    # The native result and its digest.
    assert download["lane"]["native_control_result_digest"] == stream["lane"]["native_control_result_digest"]
    assert Path(download["lane"]["native_control_result_path"]).read_bytes() == (
        Path(stream["lane"]["native_control_result_path"]).read_bytes())
    # The joined terminal result: blockers, episode rows, interpretation summary and all.
    terminal = {mode: json.loads((value["root"] / "policy_canary_terminal_result.json").read_text())
                for mode, value in sealed.items()}
    assert terminal["stream"] == terminal["download"]
    assert terminal["stream"]["episode_interpretation"]["receipt_count"] == 20
    assert download["interpreter_calls"] == stream["interpreter_calls"] == 20
    # Registry rows (ids, roles, digests, sizes; evidence roots compared relative), delivery, projection.
    registries = {mode: _relative(json.loads((value["root"] / "artifacts/result_delivery/artifact_registry.json")
                                             .read_text()), value["root"]) for mode, value in sealed.items()}
    for registry in registries.values():
        registry.pop("registry_digest")  # over absolute evidence roots, which differ by run root
    assert registries["stream"] == registries["download"]
    for name in ("artifacts/result_delivery/delivery.json",
                 "artifacts/result_delivery/policy_canary_result_projection.json",
                 "artifacts/result_delivery/policy_canary_webapp_sync.json"):
        paths = [value["root"] / name for value in sealed.values()]
        assert paths[0].read_bytes() == paths[1].read_bytes(), name
    for key in ("result_delivery_digest", "policy_canary_projection_digest", "notification_delivery",
                "official_billing", "provider_zero"):
        assert _relative(download["receipt"][key], download["root"]) == _relative(stream["receipt"][key],
                                                                                   stream["root"]), key
    # Interpretation receipts, byte for byte, beside each mode's evidence.
    receipts = {mode: {path.name: path.read_bytes() for path in (
        Path(value["lane"]["native_control_result_path"]).parent / "episode_interpretation/receipts").glob("*.json")}
        for mode, value in sealed.items()}
    assert receipts["stream"] == receipts["download"] and len(receipts["stream"]) == 20
    # Billing: verdict, charges, instance ids and the adapter, teardown and provider-zero records it
    # binds are identical; only the lane result and artifact manifest records differ (documented).
    bills = [(value["root"] / "official_billing_reconciliation.json").read_bytes() for value in sealed.values()]
    assert bills[0] == bills[1]
    assert terminal_evidence["download"]["terminal_status"] == "completed"
    for key in ("terminal_result", "artifact_manifest"):
        assert terminal_evidence["stream"][key]["sha256"] != terminal_evidence["download"][key]["sha256"], key
    # Teardown, provider-zero, provider closeout and the cleanup rows.
    for mode_value in sealed.values():
        assert mode_value["lane"]["all_staged_objects_absent"] is True
    closeouts = {mode: {key: (value if not isinstance(value, dict) else
                              {field: item for field, item in value.items() if field != "path"})
                        for key, value in sealed[mode]["lane"]["provider_closeout"].items()} for mode in sealed}
    assert closeouts["stream"] == closeouts["download"]
    cleanups = {}
    for mode, value in sealed.items():
        cleanup = json.loads(Path(value["lane"]["object_store_cleanup_path"]).read_text())
        cleanups[mode] = sorted((row["key_sha256"], row["absence"]["absence_confirmed"]) for row in cleanup["objects"])
    assert cleanups["stream"] == cleanups["download"] and all(absent for _, absent in cleanups["stream"])
    zeros = [(value["root"] / "post_teardown_global_provider_zero.json").read_bytes() for value in sealed.values()]
    assert zeros[0] == zeros[1]
    teardowns = [Path(value["lane"]["teardown_manifest_path"]).read_bytes() for value in sealed.values()]
    assert teardowns[0] == teardowns[1]
    # Strict controls (review important 3): every cell's control archive -- which stream mode builds
    # by streaming the frames and videos it never held through the member view -- is download
    # mode's, byte for byte, and all twenty controls verify in both.
    cells = {mode: sorted((value["root"] / "artifacts/result_delivery/controls").glob("cell-*.zip"))
             for mode, value in sealed.items()}
    assert [path.name for path in cells["stream"]] == [path.name for path in cells["download"]] == [
        f"cell-{index:02d}.zip" for index in range(10)]
    for downloaded, streamed in zip(cells["download"], cells["stream"]):
        assert downloaded.read_bytes() == streamed.read_bytes(), streamed.name
    delivered = json.loads((stream["root"] / "artifacts/result_delivery/delivery.json").read_text())
    assert delivered["controls_summary"] == {"expected_count": 20, "recorded_count": 20, "completed_count": 20,
                                             "passed_count": 20, "verified_cell_count": 10}
    control_media = [path for path in (Path(stream["lane"]["attempt_root"]) / "immutable_execution/control_runs")
                     .rglob("*") if path.suffix in {".png", ".mp4"}]
    referenced = json.loads((stream["root"] / "artifacts/result_delivery/archive_member_references.v1.json")
                            .read_text())["members"]
    assert control_media == [] and {PurePosixPath(path).suffix for path in referenced
                                    if "/immutable_execution/control_runs/" in path} >= {".png", ".mp4"}
    # Streamed delivery registered the members it did not hold by archive reference (download: none).
    references = "artifacts/result_delivery/archive_member_references.v1.json"
    assert not (download["root"] / references).exists()
    remote = json.loads((stream["root"] / references).read_text())["members"]
    assert any(path.endswith(".mp4") for path in remote) and any("/policy-requests/" in path for path in remote)
    # Host bytes: no ZIP or bulk member on the streamed host.
    streamed_attempt = Path(stream["lane"]["attempt_root"])
    assert not any(path.suffix in {".zip", ".mp4", ".png"} for path in streamed_attempt.rglob("*"))
    assert any(path.suffix == ".mp4" for path in Path(download["lane"]["attempt_root"]).rglob("*"))


def test_a_streamed_run_blocked_after_its_paid_run_is_not_sealed_as_before_first_observation(tmp_path, monkeypatch):
    """Review important 2: a streamed run whose output arrived -- observed and promoted, durable in
    B2 -- but was never ingested (here its needed set is over budget) has no execution receipt on
    the host. That is not "before first observation": the lane records a
    ``provider_output_not_ingested`` gap carrying the stream blocker, claims nothing about the
    policy, and the dispatcher's gap path seals and delivers that gap instead of twenty episodes
    that never observed, never queried and never moved the arm."""
    from blueprint_pipeline import policy_canary_output_members as output_members
    from blueprint_pipeline.episode_interpretation import DeterministicFixtureInterpreter
    from tests.provider_output_fixtures import zip_tree
    from tests.test_episode_interpretation import _output
    from tests.test_policy_canary_episode_interpretation_closeout import _rights_for_all

    data, result = _parity_evidence(tmp_path / "source")
    rights = tmp_path / "rights"
    _rights_for_all(data, result, rights, DeterministicFixtureInterpreter(_output(data)))
    monkeypatch.setattr(output_members, "POLICY_CANARY_OUTPUT_CONTRACT",
                        output_members.PolicyCanaryOutputContract(needed_set_budget_bytes=1_000))

    def post_billing(**kwargs):
        write_json(Path(kwargs["output_path"]), {"status": "reconciled_official_posted_charges",
                                                  "official_total_usd": 0.38, "instance_ids": [INSTANCE]})
        return True

    sealed = _dispatch(tmp_path / "stream", monkeypatch, archive=zip_tree(data["root"]), data=data, rights=rights,
                       stream=True, post_billing=post_billing)

    stream_blocker = f"{PREFIX}_provider_output_needed_set_over_budget"
    lane = sealed["lane"]
    assert lane["archive_durable"] is True and stream_blocker in lane["blockers"]
    assert lane["visual_evidence"] == {"status": "provider_output_not_ingested", "media_gap": {
        "type": "provider_output_not_ingested", "reason": stream_blocker}}
    for claim in ("candidate_policy_queried", "first_observation_reached", "scientific_attempt_started"):
        assert claim not in lane, claim
    root = sealed["root"]
    gap = json.loads((root / "preprovider_evidence/typed_media_gap.json").read_text())
    assert gap == {"schema_version": "task_evaluation_policy_canary_media_gap.v1",
                   "type": "provider_output_not_ingested", "reason": stream_blocker,
                   "candidate_policy_queried": None, "archive_durable": True}
    terminal = json.loads((root / "policy_canary_terminal_result.json").read_text())
    assert terminal["status"] == "blocked" and terminal["candidate_policy_queried"] is None
    assert "policy_canary_episode_failure:provider_output_not_ingested" in terminal["blockers"]
    assert not any("before_first_observation" in blocker for blocker in terminal["blockers"])
    assert len(terminal["episodes"]) == 20
    for episode in terminal["episodes"]:
        assert episode["typed_harness_failure"] == "provider_output_not_ingested"
        assert episode["visual_evidence"]["media_gap"] == {"type": "provider_output_not_ingested",
                                                           "reason": stream_blocker}
        for claim in ("candidate_policy_queried", "actions_reached_robot", "arm_moved"):
            assert episode[claim] is None, claim
    # What the owner and the Website see names the gap, and nothing says the run never observed.
    delivered = (root / "artifacts/result_delivery/delivery.json").read_text()
    assert {episode["failure"]["code"] for episode in json.loads(delivered)["episodes"]} == {
        "provider_output_not_ingested"}
    # Both deliveries leave the three execution claims unknown (null), never false.
    for name in ("delivery.json", "website_delivery.json"):
        episodes = json.loads((root / "artifacts/result_delivery" / name).read_text())["episodes"]
        assert len(episodes) == 20
        assert {(episode["policy_query"]["candidate_policy_queried"], episode["action_delivery"]["actions_reached_robot"],
                 episode["action_delivery"]["arm_moved"]) for episode in episodes} == {(None, None, None)}, name
    projection = json.loads((root / "artifacts/result_delivery/policy_canary_result_projection.json").read_text())
    assert {row["failure_taxonomy"] for row in projection["episodes"]} == {"provider_output_not_ingested"}
    assert {
        tuple(row[claim] for claim in ("candidate_policy_queried", "actions_reached_robot", "arm_moved"))
        for row in projection["episodes"]
    } == {(None, None, None)}
    assert {row["actions_delivered_episode_count"] for row in projection["candidate_results"]} == {0}
    for name in ("delivery.json", "policy_canary_result_projection.json", "policy_canary_webapp_sync.json"):
        assert "before_first_observation" not in (root / "artifacts/result_delivery" / name).read_text(), name


def test_resume_of_a_completed_attempt_keeps_the_manifest_bound_receipt(lane):
    """Review minor 4: the sealed artifact manifest binds the promotion receipt by sha256 as a
    required role, and the absence proof binds it by digest. The door run again on a completed
    attempt reuses the receipt unchanged, so it must not rewrite it."""
    from blueprint_pipeline import provider_output_promotion as promotion

    archive, _ = _small_archive()
    result = lane.run_session(lane.adapter(archive))
    assert result["status"] == "completed", result["blockers"]
    attempt = Path(result["attempt_root"])
    manifest = json.loads(Path(result["artifact_manifest_path"]).read_text())
    [row] = [row for row in manifest["files"] if "provider_output_promotion" in row.get("roles", [])]
    receipt = attempt / row["relative_path"]
    before = receipt.read_bytes()
    assert row["sha256"] == "sha256:" + hashlib.sha256(before).hexdigest()

    resumed = promotion.resume_provider_output_promotion(attempt, ingest=True)

    assert resumed["status"] == "completed" and resumed["ingestion"]["short_circuited"] is True
    assert receipt.read_bytes() == before
    assert resumed["promotion"]["receipt_digest"] == json.loads(before)["receipt_digest"]


def test_resume_ingests_a_durable_archive_once_and_short_circuits_after_readers_write(lane, capsys):
    """The door's --ingest materializes a run blocked after promotion; once readers have written
    into the evidence root (partial recovery, interpretation), a later resume short-circuits on
    the materialized receipt instead of refusing the tree it no longer owns (review I8)."""
    from blueprint_pipeline import policy_canary_output_members as output_members
    from blueprint_pipeline import provider_output_promotion as promotion

    archive, _ = _small_archive()
    with lane.monkeypatch.context() as patch:
        patch.setattr(output_members, "POLICY_CANARY_OUTPUT_CONTRACT",
                      output_members.PolicyCanaryOutputContract(needed_set_budget_bytes=1_000))
        blocked = lane.run_session(lane.adapter(archive))
    assert f"{PREFIX}_provider_output_needed_set_over_budget" in blocked["blockers"]
    attempt = Path(blocked["attempt_root"])
    assert not (attempt / "immutable_execution").exists()

    resumed = promotion.resume_provider_output_promotion(attempt, ingest=True)

    assert resumed["status"] == "completed", resumed["blockers"]
    assert resumed["ingestion"]["status"] == "materialized" and resumed["ingestion"]["short_circuited"] is False
    # The native inventory is checked with the binding the sealed lane result recorded (this fixture's
    # identity document is filler JSON): an outcome, not a blocker.
    assert resumed["ingestion"]["ingestion"]["native_inventory"] == {
        "status": "failed", "code": "provider_output_native_identity_mismatch"}
    index = json.loads((attempt / "provider_output_member_index.v1.json").read_text())
    needed = POLICY_CANARY_OUTPUT_CONTRACT.paths(index)
    assert sorted(_members(attempt / "immutable_execution")) == sorted(needed)
    assert (attempt / "immutable_execution.member_view.v1.json").is_file()
    fetched = len(lane.world.cas.ranged_requests())
    assert fetched == len(needed)
    [sample] = lane.history()[-1:]
    assert (sample["outcome"], sample["observed_bytes"]) == ("completed", POLICY_CANARY_OUTPUT_CONTRACT.needed_bytes(index))
    # The sealed lane result is never rewritten.
    assert json.loads((attempt / "adp_arena_vast_result.json").read_text())["native_control_result_path"] is None

    # A reader writes into the evidence root, as partial recovery does.
    gap = attempt / "immutable_execution" / "partial_terminal_evidence" / "typed_media_gap.json"
    gap.parent.mkdir()
    gap.write_text("{}\n", encoding="utf-8")

    assert promotion.main(["resume", "--attempt-root", str(attempt), "--ingest"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed" and printed["ingestion"]["short_circuited"] is True
    assert len(lane.world.cas.ranged_requests()) == fetched and gap.is_file()
