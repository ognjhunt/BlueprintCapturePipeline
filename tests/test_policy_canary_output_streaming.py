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

Unset, the setting means auto: stream only when promotion would accept the dedicated B2 store and
the host has recorded a needed-set measurement within budget, else download, with the lane result
recording which and why.

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
from blueprint_pipeline import policy_canary_output_members as output_members
from blueprint_pipeline import provider_output_range_transport as transport
from blueprint_pipeline import task_evaluation_configured_scene_object_store as scene_store
from blueprint_pipeline.common import write_json
from blueprint_pipeline.control_plane_disk_usage import tree_usage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import PROVIDER_RESULT_FILENAME
from blueprint_pipeline.policy_canary_output_members import DELIVERY_ENV
from blueprint_pipeline.wam_provider_object_store import SCHEMA_VERSION, STAGING_MANIFEST_FILENAME
from tests.provider_output_fixtures import (
    DEFLATED,
    STORED,
    Entry,
    Zeros,
    build_zip,
    quick10_production_shape,
)
from tests.test_provider_output_promotion import World

RESULT = PROVIDER_RESULT_FILENAME
PREFIX = "native_task_arena_policy_canary_session"
# The lane-result field recording the effective delivery mode and why it was chosen.
RESOLUTION = "provider_output_delivery_resolution"
INSTANCE = 49_247_792
CHUNK = 1024**2


@pytest.fixture(autouse=True)
def _isolate_supervisor_context(monkeypatch) -> None:
    # Provider/watchdog doubles do not run inside the CI runner's service.
    monkeypatch.delenv("INVOCATION_ID", raising=False)


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
        # The host's needed-set measurement record, redirected; none is written until a test measures.
        self.measurement = tmp_path / "policy-canary-output" / "needed-set-measurement.v1.json"
        monkeypatch.setattr(output_members, "MEASUREMENT_PATH", self.measurement)
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

    def run_session(self, adapter, *, delivery: str | None = "stream", job_dir=None):
        """The real Quick-10 session and lane, its authority, bundle and witness checks stubbed.

        ``delivery`` is the setting's value, or None to leave it unset. The world configures the
        dedicated B2 store, so unset streams (auto) and downloading takes an explicit ``download``.
        """
        monkeypatch = self.monkeypatch
        if delivery is None:
            monkeypatch.delenv(DELIVERY_ENV, raising=False)
        else:
            monkeypatch.setenv(DELIVERY_ENV, delivery)
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


def _small_archive(**extra):
    archive, payloads = quick10_run_archive(frames_per_camera=2, png_bytes=4_096, mp4_bytes=32_768,
                                            policy_requests_per_episode=2, policy_request_bytes=2_048, **extra)
    return archive, payloads


IDENTITY = "cell_runs/00/policy_canary_static_startup_preflight.v1.json"  # each cell's worker seals one


def _failing_b2_ranges(lane, *, status, first_failure, failures):
    """Answer B2 member range requests ``first_failure`` .. ``first_failure + failures - 1`` (1-based,
    ETag probes excluded) with HTTP ``status``; every other request is served as before."""
    import urllib.error

    served = {"ranges": 0}

    def route(request, timeout, policy):
        if (urlparse(request.full_url).hostname == "b2.example.invalid"
                and request.get_header("Range") not in (None, "bytes=0-0")):
            served["ranges"] += 1
            if first_failure <= served["ranges"] < first_failure + failures:
                raise urllib.error.HTTPError("redacted", status, "injected", {}, None)
        return lane._route(request, timeout, policy)

    lane.monkeypatch.setattr(transport, "_open_with_policy", route)
    lane.monkeypatch.setattr(streaming, "INGESTION_RETRY_SLEEP_SECONDS", 0)
    return served


def _ingestion_failures(attempt: Path) -> list[str]:
    path = attempt / ".provider_output_ingestion" / "failures.jsonl"
    return [json.loads(line)["code"] for line in path.read_text().splitlines()] if path.exists() else []


# -- Download mode is today's path, byte for byte ------------------------------------------

GOLDEN_LANE_RESULT_SHA256 = "ba00b7bbcfa3f4a9e6ee522767d9ddef64af4d99e81b3071b4dcd1ffabc96d76"
GOLDEN_ARTIFACT_MANIFEST_SHA256 = "3eeb482399c882929c5cb4fad547fe36f9f5875f48df4fc7e1574c33e4379afa"
# The same doubles through the real Quick-10 session (setting unset, no B2 store), taken from main
# at 2812a2720 before unset meant auto: the session's own lane arguments make it differ from the above.
GOLDEN_SESSION_LANE_RESULT_SHA256 = "6752a164c166408b4c25bca8172bd33fbe8d95c7358a26663539410060f23d10"


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


def _golden_doubles(tmp_path, monkeypatch) -> tuple[dict, dict]:
    """The byte-identity tests' seams (staging, the adapter landing the golden ZIP, the watchdog), and
    their prepared bundle. Returns ``(calls, bundle)``; ``calls`` collects the staging and adapter kwargs."""
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
    return calls, {"status": "ready", "bundle_path": str(bundle_path),
                   "bundle_sha256": arena._file_sha256(bundle_path), "protocol_digest": "sha256:" + "b" * 64}


def _golden_lane(tmp_path, bundle, *, execute=True, **delivery) -> dict:
    """The arena lane called directly, as the golden digests were taken."""
    return arena.run_arena_native_control_vast(
        approval_path=".", job_dir=tmp_path / "job", paid_resource_admission_grant=object(), execute=execute,
        prepared_bundle=bundle, hard_cap_usd=4.0, hard_ttl_seconds=14_400, expected_output_filename=RESULT,
        provider_bundle_kind=PREFIX, result_schema_version="native_task_arena_policy_canary_session_result.v1",
        blocker_prefix=PREFIX, candidate_policy_query_expected=True, require_independent_watchdog=True,
        **delivery)


def _golden_session(tmp_path, monkeypatch, bundle) -> tuple[dict, dict]:
    """The real Quick-10 session over the golden doubles, its authority, bundle and witness checks
    stubbed and its lane call spied on. Returns ``(result, lane_kwargs)``."""
    lane_kwargs = {}
    real_lane = native.run_arena_native_control_vast

    def spy(**kwargs):
        lane_kwargs.update(kwargs)
        return real_lane(**kwargs)

    monkeypatch.setattr(native, "run_arena_native_control_vast", spy)
    monkeypatch.setattr(native, "validate_policy_canary_session_authority", lambda value: value)
    monkeypatch.setattr(native, "validate_policy_canary_provider_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(native, "consume_session_authority_once",
                        lambda *_args, **_kwargs: {"status": "consumed", "blockers": []})
    monkeypatch.setattr(native, "_policy_provider_transfer_byte_budget", lambda _candidate: (100, 20))
    monkeypatch.setattr(paired, "build_paired_witness_binding",
                        lambda *_args, **_kwargs: {"maximum_archive_bytes": 1_000_000})
    monkeypatch.setattr(paired, "paired_witness_secret_paths", lambda *_args, **_kwargs: {})
    authority = {"hard_cap_usd": 4.0, "hard_ttl_seconds": 14_400, "authority_digest": "sha256:" + "a" * 64,
                 "resource_name": "blueprint-native-task-policy-canary-" + "a" * 32}
    result = native.run_native_task_arena_policy_canary_session_vast(
        job_dir=tmp_path / "job", prepared_bundle={**bundle, "container_image": "immutable-image"},
        session_authority=authority, paid_resource_admission_grant=object(), execute=True, hard_cap_usd=4.0,
        hard_ttl_seconds=14_400, provider_runtime_environment={"BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "640x360"})
    return result, lane_kwargs


def _lane_bytes(value: dict, tmp_path) -> bytes:
    """A lane result's bytes as ``write_json`` seals them, with the run's temporary root elided."""
    return json.dumps(value, indent=2).encode().replace(str(tmp_path).encode(), b"<tmp>")


EXPLICIT_DOWNLOAD = {"provider_output_delivery": "download", "provider_output_member_contract": None,
                     "provider_output_reservation": None}


@pytest.mark.parametrize("explicit", [False, True])
def test_download_mode_lane_result_and_manifest_are_byte_identical(tmp_path, monkeypatch, explicit):
    """The golden digests were taken from PR B's head (87ec78833) before this lane changed."""
    calls, bundle = _golden_doubles(tmp_path, monkeypatch)
    result = _golden_lane(tmp_path, bundle, **(EXPLICIT_DOWNLOAD if explicit else {}))

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


@pytest.mark.parametrize("sink", ["dry_run", "live_window_below_minimum", "pre_spend_preflight_blocked",
                                  "staging_blocked", "runtime_dependency_blocked", "watchdog_not_armed",
                                  "completed"])
def test_every_lane_result_records_the_resolution_it_is_given(tmp_path, monkeypatch, sink):
    """Whichever result the lane seals, before the paid window or after it, carries the record."""
    _, bundle = _golden_doubles(tmp_path, monkeypatch)
    if sink == "live_window_below_minimum":
        monkeypatch.setattr(arena, "_remaining_session_live_minutes", lambda **_kwargs: 10)
    elif sink == "pre_spend_preflight_blocked":
        def closed(**_kwargs):
            raise arena.PreSpendPreflightBlocked({"status": "BLOCKED", "blockers": ["spend_gate_closed"]})

        monkeypatch.setattr(arena, "require_pre_spend_preflight", closed)
    elif sink == "staging_blocked":
        monkeypatch.setattr(arena, "stage_wam_provider_bundle_object_store",
                            lambda **_kwargs: {"status": "blocked", "blockers": ["staging_refused"]})
    elif sink == "runtime_dependency_blocked":
        bundle = {**bundle, "runtime_source_packet": {"transport": "content_addressed_external_layer.v1"}}
        monkeypatch.setattr(arena, "stage_cached_runtime_dependency_object_store",
                            lambda **_kwargs: {"status": "blocked", "blockers": ["runtime_layer_refused"]})
        monkeypatch.setattr(arena, "close_cached_runtime_dependency_staging", lambda _path: {"status": "closed"})
    elif sink == "watchdog_not_armed":
        monkeypatch.setattr(arena, "arm_independent_vast_watchdog", lambda **_kwargs: ({"status": "refused"}, None))
    why = {"mode": "download", "reason": "explicit"}

    result = _golden_lane(tmp_path, bundle, execute=sink != "dry_run", **EXPLICIT_DOWNLOAD,
                          provider_output_delivery_resolution=why)

    assert result["status"] == {"dry_run": "dry_run_ready", "completed": "completed"}.get(sink, "blocked")
    assert result[RESOLUTION] == why
    assert json.loads((tmp_path / "job" / "adp_arena_vast_result.json").read_text())[RESOLUTION] == why


@pytest.mark.parametrize("resolution", [
    {"mode": "stream", "reason": "explicit"},  # not the mode the lane runs
    {"mode": "download", "reason": "auto"},  # not a reason
    {"mode": "download", "reason": "auto_needed_set_within_budget"},  # a reason that only explains a stream
    {"mode": "download", "reason": "explicit", "setting": "download"},  # not the record's shape
])
def test_the_lane_refuses_a_resolution_that_does_not_describe_its_mode(tmp_path, monkeypatch, resolution):
    _, bundle = _golden_doubles(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="^adp_arena_provider_output_delivery_resolution_invalid$"):
        _golden_lane(tmp_path, bundle, **EXPLICIT_DOWNLOAD, provider_output_delivery_resolution=resolution)

    assert not (tmp_path / "job").exists()


# -- The default is auto: the host's measurement record, not the deploy, is the switch --------


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
        # The world configures the B2 store, so downloading is asked for explicitly (unset would stream).
        lane_result = lane.run_session(lane.adapter(archive), delivery="stream" if stream else "download",
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


