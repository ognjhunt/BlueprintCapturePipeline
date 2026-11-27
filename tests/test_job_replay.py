"""One command replays a real job's remaining website stages offline, flags every assumed step and buys nothing.

The stages here run on the repository's own synthetic builders: the Atlas preview job of
``tests.test_atlas_preview_replay`` (the real controller, up to its first purchase), and the room job of
``tests.test_website_scene_handoff_navigation`` (the real scene handoff and the real company package builder; its
Atlas lane and mask binding are stood in, as there). ``inline`` replays run the stages in this process on a working
copy standing at the capture's own path while the built capture is set aside and digested, as ``replay_set`` does;
the namespace tests run the whole command, door pull included, against a stand-in door over a local folder.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import job_replay
from blueprint_pipeline.atlas_preview_replay import _namespace_prefix, _tree_digest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_atlas_preview_replay import _controller, site  # noqa: F401 - the Atlas fixture

ROOT = Path(__file__).resolve().parents[1]


def _in_place(capture: Path, tmp_path: Path, **options):
    """``run_stages`` on a copy at the capture's own path; the built capture is set aside and must not change."""
    evidence = capture.with_name(capture.name + ".evidence")
    capture.rename(evidence)
    before = _tree_digest(evidence)
    try:
        shutil.copytree(evidence, capture, symlinks=True)
        result = job_replay.run_stages(capture, run_dir=tmp_path / "run", inline=True, **options)
    finally:
        shutil.rmtree(capture, ignore_errors=True)
        evidence.rename(capture)
    assert _tree_digest(capture) == before
    return result


def _stage(result, name):
    return next(row for row in result["stages"] if row["stage"] == name)


def test_a_purchase_stops_as_would_buy_with_its_provider_operation_and_ceiling(site, tmp_path):  # noqa: F811 - the fixture
    result = _in_place(site.capture, tmp_path)
    atlas = _stage(result, "atlas_preview")
    assert atlas["status"] == "ran" and atlas["stop"]["kind"] == "would_buy"
    [bought] = atlas["would_buy"]
    assert (bought["provider"], bought["resource_class"], bought["maximum_cost_usd"]) == (
        "openai", "openai_api_candidate", .5)
    assert bought["call_site"].startswith("atlas_scene_planner.")
    assert atlas["maximum_usd"] == .5 and atlas["reads"]["count"] > 0
    assert atlas["duration_seconds"] >= 0 and atlas["peak_rss_bytes"] > 0
    # A stop ends the chain: the later stages say they did not run and why.
    for name in ("preparation", "scene_handoff", "company_simulation"):
        row = _stage(result, name)
        assert row["status"] == "skipped" and row["reason"].startswith("after_atlas_preview:would_buy:")
    assert result["clean"] and result["breaches"] == [] and result["assumed"] == []


def test_past_marks_the_step_assumed_and_the_next_stage_runs_on_the_files_as_left(site, tmp_path):  # noqa: F811 - the fixture
    configuration = site.job.parent / "configuration.json"
    configuration.write_text(json.dumps({**site.configuration, "remaining_cost_usd": .9}))
    plain = _in_place(site.capture, tmp_path / "plain")
    assert _stage(plain, "atlas_preview")["stop"]["code"] == "atlas_runtime_budget_insufficient"
    assert _stage(plain, "preparation")["status"] == "skipped"

    result = _in_place(site.capture, tmp_path / "past", past=["atlas_runtime_budget_insufficient"])
    atlas = _stage(result, "atlas_preview")
    assert atlas["status"] == "assumed" and atlas["stop"]["code"] == "atlas_runtime_budget_insufficient"
    assert result["assumed"] == [{"stage": "atlas_preview", "code": "atlas_runtime_budget_insufficient",
                                  "kind": "refusal", "how": "assumed_passed", "note": result["assumed"][0]["note"]}]
    assert not result["clean"]
    # The real preparation ran next, offline, and stopped on its own: this fixture has no descriptor.
    preparation = _stage(result, "preparation")
    assert preparation["status"] == "ran" and preparation["stop"]["kind"] in ("refusal", "error", "missing_input")
    table = job_replay.table({**result, "evidence": {"files": 1, "unchanged": True}, "boundaries": {}})
    assert "ASSUMED atlas_preview: atlas_runtime_budget_insufficient (assumed_passed)" in table
    assert "NOT A CLEAN RUN" in table


def test_a_network_attempt_fails_the_run(site, tmp_path, monkeypatch):  # noqa: F811 - the fixture
    def step(**_options):
        socket.create_connection(("203.0.113.7", 443), timeout=1)
        raise AssertionError("the replay reached the network")
    _controller(monkeypatch, step)
    result = _in_place(site.capture, tmp_path)
    atlas = _stage(result, "atlas_preview")
    assert atlas["stop"]["kind"] == "error" and atlas["stop"]["code"] == "unstubbed_network_call:203.0.113.7"
    assert result["network_attempts"] and result["breaches"] == ["network_attempted"]


def test_a_settlement_is_recorded_never_sent_and_answered_as_no_answer(site, tmp_path, monkeypatch):  # noqa: F811
    """A settlement only releases what a bought call held beyond its cost; every caller keeps the hold when the
    WebApp does not answer, so the replay records it and the stage goes on along that path."""
    from blueprint_pipeline.website_task_context import preparation_webapp_request

    kept = []

    def step(*, task_context, **_options):
        command = {"provider": "meta", "provider_charge_amount_usd": 0.0025, "completed_request_count": 1,
                   "allocation_binding_digest": "sha256:" + "3" * 64}
        try:
            preparation_webapp_request(task_context=task_context, operation="preparation-settlement",
                                       payload={"request_id": "req1", "scene_id": "site-req1", "settlement": command})
        except ValueError as error:
            kept.append(str(error))  # the reservation keeps its hold, as the real callers do
        raise ValueError("atlas_test_stop_after_the_settlement")
    _controller(monkeypatch, step)
    atlas = _stage(_in_place(site.capture, tmp_path), "atlas_preview")
    assert kept == [job_replay.SETTLEMENT_NOT_SENT]
    assert atlas["stop"]["code"] == "atlas_test_stop_after_the_settlement"
    [unsent] = atlas["unsent_settlements"]
    assert (unsent["operation"], unsent["provider"], unsent["provider_charge_amount_usd"]) == (
        "preparation-settlement", "meta", 0.0025)


def _room_job(tmp_path, monkeypatch):
    """The navigation room job, with the task context the WebApp would answer kept beside it, as a pull has it."""
    from blueprint_pipeline import run_e2e
    from tests.test_website_company_simulation import SHARED
    from tests.test_website_scene_handoff_navigation import room_job

    job, context, _result = room_job(tmp_path, monkeypatch)
    retained = {**{key: value for key, value in context.items() if key != "context_digest"},
                "capture_rights": {"consent_status": "granted", "consent_revoked": False, "statement_version": SHARED}}
    retained["context_digest"] = canonical_digest(retained, digest_field="context_digest")
    (job.capture / "pipeline/website_task_context.json").write_text(json.dumps(retained))
    monkeypatch.setenv("BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT", str(job.queue))
    # The preparation before the handoff is this builder's own stand-in (its clean plate is the fixture's): the
    # handoff itself is the real one, entered through the module as ``website_capture_preparation`` enters it.
    monkeypatch.setattr(run_e2e, "run_end_to_end", lambda **_: {"pipeline_status": "completed", "handoff": job.rerun()})
    return job


def test_each_later_stage_runs_offline_the_handoff_and_the_company_package(tmp_path, monkeypatch):
    job = _room_job(tmp_path, monkeypatch)
    result = _in_place(job.capture, tmp_path)
    assert _stage(result, "atlas_preview")["status"] == "skipped"  # this job has no Atlas preview job
    preparation, handoff, company = (_stage(result, name) for name in ("preparation", "scene_handoff",
                                                                         "company_simulation"))
    assert preparation["stop"] == {"kind": "completed", "stage": "preparation", "code": "reached_scene_handoff"}
    assert handoff["status"] == "ran" and handoff["stop"]["code"] == "room_package_written"
    assert handoff["handoff"]["status"] == "room_package_written" and handoff["reads"]["count"] > 0
    # The package was compiled; its upload to the private store is an effect the replay never makes.
    assert company["status"] == "ran"
    assert company["stop"]["kind"] == "external_effect"
    assert company["stop"]["code"] == "artifact_store:website-company-simulation"
    assert company["package"]["digest"].startswith("sha256:")
    assert {"README.txt", "scene.usda", "success_rule.json"} <= set(company["package"]["members"])
    assert result["clean"] and result["breaches"] == [] and result["network_attempts"] == []
    for row in (preparation, handoff, company):
        assert row["duration_seconds"] >= 0 and row["peak_rss_bytes"] > 0 and row["would_buy"] == []


def test_keep_package_keeps_the_zip_the_package_stage_builds(tmp_path, monkeypatch):
    """The door does not serve a package holding an object (``object.usdz`` is a nested archive), so a replay that
    rebuilds the host's bytes keeps them to be checked (stress test, 2026-11-27). Without the flag nothing is kept."""
    import hashlib

    job = _room_job(tmp_path, monkeypatch)
    kept_dir = tmp_path / "run"
    kept_dir.mkdir()
    monkeypatch.setattr(job_replay, "_RUN_DIR", kept_dir)
    company = _stage(_in_place(job.capture, tmp_path / "first"), "company_simulation")
    assert "kept" not in company["package"] and not (kept_dir / "package.zip").exists()
    monkeypatch.setenv(job_replay.KEEP_PACKAGE_ENV, "1")
    company = _stage(_in_place(job.capture, tmp_path / "second"), "company_simulation")
    kept = Path(company["package"]["kept"])
    assert kept == kept_dir / "package.zip"
    assert "sha256:" + hashlib.sha256(kept.read_bytes()).hexdigest() == company["package"]["digest"]
    assert not job_replay._credential_name(job_replay.KEEP_PACKAGE_ENV)  # it reaches the worker's environment


def test_the_masks_substitution_steps_past_the_recomputation_with_the_hosts_own_masks(tmp_path, monkeypatch):
    from contextlib import ExitStack

    from blueprint_pipeline import website_task_masks as masks
    from blueprint_pipeline.website_track_mask_codec import write_track_json

    binding = {"source_frames_digest": "sha256:" + "1" * 64, "task_targets": []}
    completed = {"schema_version": "website_task_masks.v1", "status": "completed", "binding": binding, "targets": []}
    path = tmp_path / "task_masks" / canonical_digest(binding)[7:23] / "task_masks.json"
    path.parent.mkdir(parents=True)
    write_track_json(path, completed)

    def refuses(**_options):
        raise ValueError("website_retained_static_masks_manifest_changed")
    monkeypatch.setattr(masks, "complete_retained_static_task_masks", refuses)
    record = []
    with ExitStack() as stack:
        job_replay._install_substitutions(stack, ["website_retained_static_masks_manifest_changed"], record,
                                          lambda: "scene_handoff")
        value = masks.complete_retained_static_task_masks(
            plan={}, source_geometry={}, task_masks={"binding": binding}, output_root=tmp_path / "task_masks",
            view_plan_root=tmp_path, compact_runs=True)
    assert value == completed
    [row] = record
    assert (row["stage"], row["how"], row["path"]) == ("scene_handoff", "retained_output", str(path.resolve()))
    assert masks.complete_retained_static_task_masks is refuses  # nothing outlives the replay


def test_the_gap_survey_substitution_takes_the_completions_retained_report(tmp_path, monkeypatch):
    from contextlib import ExitStack

    from blueprint_pipeline import atlas_scene_completion as completion

    report = {"status": "needs_input", "reason": "atlas_completion_nothing_to_fill", "digest": "sha256:" + "2" * 64}
    (tmp_path / "completion.json").write_text(json.dumps(report))

    def refuses(**_options):
        raise ValueError("atlas_completion_gap_survey_changed")
    monkeypatch.setattr(completion, "complete_scene", refuses)
    record = []
    with ExitStack() as stack:
        job_replay._install_substitutions(stack, ["atlas_completion_gap_survey_changed"], record, lambda: "atlas_preview")
        assert completion.complete_scene(root=tmp_path) == (None, report)
        (tmp_path / "completion.json").write_text(json.dumps({**report, "status": "passed"}))
        with pytest.raises(ValueError, match="atlas_completion_gap_survey_changed"):
            completion.complete_scene(root=tmp_path)  # a passed report is the code's own to adopt: never assumed
    assert [row["how"] for row in record] == ["retained_output", "no_retained_output"]


def test_a_stage_worker_runs_with_the_thread_counts_its_host_unit_sets(tmp_path, monkeypatch):
    """The Atlas preview unit runs on one BLAS thread; its replayed workers do too, whatever the shell sets."""
    unit = (ROOT / "deploy/systemd/blueprint-website-atlas-preview.service").read_text()
    line = next(row for row in unit.splitlines() if row.startswith("Environment=OPENBLAS_NUM_THREADS="))
    assert dict(item.split("=", 1) for item in line.removeprefix("Environment=").split()) == job_replay.UNIT_THREADS
    for name in job_replay.UNIT_THREADS:
        monkeypatch.setenv(name, "4")
    launch = job_replay.subprocess.Popen

    def worker(argv, **options):
        # The worker the replay starts, as a process that only says which thread counts it was given.
        group, run_dir = argv[argv.index("--worker") + 1], argv[argv.index("--run-dir") + 1]
        report = ("import json, os, sys; json.dump({'rows': [], 'threads': {name: os.environ.get(name) "
                  "for name in sys.argv[3:]}}, open(os.path.join(sys.argv[1], 'threads-' + sys.argv[2] + '.json'), 'w'))")
        return launch([sys.executable, "-c", report, run_dir, group, *job_replay.UNIT_THREADS], **options)
    monkeypatch.setattr(job_replay.subprocess, "Popen", worker)
    seen = {}
    for group, *_ in job_replay.GROUPS:
        job_replay._run_group(group, tmp_path / "capture", run_dir=tmp_path, past=(), commit="", steps=None,
                              watched=str(tmp_path), inline=False)
        seen[group] = json.loads((tmp_path / f"threads-{group}.json").read_text())["threads"]
    one = dict.fromkeys(job_replay.UNIT_THREADS, "1")
    # The preparation runs in the handoff listener, whose unit sets none: it keeps the shell's.
    assert seen == {"atlas_preview": one, "preparation": dict.fromkeys(job_replay.UNIT_THREADS, "4"),
                    "company_simulation": one}


def test_the_door_is_only_read():
    door = job_replay.DoorReader(client=object())
    for route in ("/requests", "/units", "/status"):
        with pytest.raises(ValueError, match="job_replay_door_route_not_read"):
            door._get(route, {})


class _LocalDoor:
    """The door's two read routes over this machine's own folders: a host stand-in."""

    def __init__(self):
        self.calls, self.reads = 0, []

    def list(self, path):
        self.calls += 1
        target = Path(path)
        if not target.exists():
            raise ValueError("refused: not_found")
        if target.is_file():
            info = target.stat()
            return {"type": "file", "size": info.st_size, "mtime": int(info.st_mtime), "entries": None}
        entries = []
        for item in sorted(target.iterdir()):
            info = item.lstat()
            kind = "symlink" if item.is_symlink() else "dir" if item.is_dir() else "file"
            entries.append({"name": item.name, "type": kind, "mtime": int(info.st_mtime),
                            **({"size": info.st_size} if kind == "file" else {})})
        return {"type": "dir", "entries": entries, "truncated": False}

    def read(self, path, offset, length):
        self.calls += 1
        self.reads.append(path)
        data = Path(path).read_bytes()
        return data[offset:offset + length], len(data), offset + length >= len(data)


NAMESPACE = shutil.which("unshare") is not None and os.name == "posix"


def _namespace_available(tmp_path):
    probe = tmp_path / "probe"
    probe.mkdir()
    return NAMESPACE and _namespace_prefix(probe) is not None


# Needs a private mount and network namespace (``unshare``), which hosted CI runners do not grant: the hermetic lanes
# deselect it instead of reporting a skip (they refuse skips). It runs wherever the replay itself runs.
@pytest.mark.external_runtime
def test_the_command_pulls_once_replays_in_its_namespace_and_never_changes_the_evidence(site, tmp_path):  # noqa: F811 - the fixture
    if not _namespace_available(tmp_path):
        pytest.skip("this host has no private mount and network namespace for the replay")
    door, scratch = _LocalDoor(), tmp_path / "scratch"
    before = _tree_digest(site.storage)
    report = job_replay.replay_job(str(site.capture), door=door, scratch=scratch, rounds=1)
    assert _tree_digest(site.storage) == before  # the host stand-in was only read
    assert report["passed"] and report["breaches"] == [] and report["evidence"]["unchanged"]
    assert report["evidence"]["digest_before"] == report["evidence"]["digest_after"]
    atlas = _stage(report, "atlas_preview")
    assert atlas["stop"]["kind"] == "would_buy" and atlas["would_buy"][0]["maximum_cost_usd"] == .5
    assert report["boundaries"]["network_probe"]["interfaces"] == ["lo"]
    assert report["boundaries"]["work_view"] in ("overlay", "copy")  # the stages wrote to their own layer only
    assert report["pull"]["pulled_files"] == len(door.reads) > 0
    # Nothing changed on the host: the next replay reuses every pulled file.
    door.reads.clear()
    again = job_replay.replay_job(str(site.capture), door=door, scratch=scratch, rounds=0)
    assert door.reads == [] and again["pull"]["pulled_files"] == 0 and again["pull"]["reused_files"] > 0
    # One record changed on the host: that one is pulled again, and only it.
    survey = site.job.parent / "survey/atlas_video_survey.json"
    survey.write_text(survey.read_text() + " ")
    os.utime(survey, (survey.stat().st_atime, survey.stat().st_mtime + 5))
    third = job_replay.replay_job(str(site.capture), door=door, scratch=scratch, rounds=0)
    assert door.reads == [str(survey)] and third["pull"]["pulled_files"] == 1


def test_without_a_private_namespace_nothing_is_replayed(site, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setattr(job_replay, "namespace_prefix", lambda _probe: None)
    door = _LocalDoor()
    report = job_replay.replay_job(str(site.capture), door=door, scratch=tmp_path / "scratch")
    assert report["replay_unsupported"] == "private_namespace_unavailable"
    assert report["stages"] == [] and door.calls == 0 and not report["passed"]
    assert job_replay.main([str(site.capture), "--offline", "--scratch", str(tmp_path / "scratch")]) == 5
    assert "not replayed: private_namespace_unavailable" in capsys.readouterr().err


def test_a_named_bulk_file_brings_its_frame_set_and_a_listed_folder_its_small_files(tmp_path):
    host = tmp_path / "host/captures/walkthrough-x"
    for relative, size in (("run.json", 10), ("raw/walkthrough.mp4", 400_000), ("frames/0001.png", 300_000),
                           ("frames/0002.png", 300_000), ("frames/notes.bin", 300_000),
                           ("outputs/000000/depth.bin", 300_000), ("outputs/000001/image.png", 300_000),
                           ("other/big.ply", 300_000)):
        (host / relative).parent.mkdir(parents=True, exist_ok=True)
        (host / relative).write_bytes(b"x" * size)
    door = _LocalDoor()
    pull = job_replay.Pull(door, job_replay.PullCache(tmp_path / "cache"), str(host))
    pull.list_roots([str(host)])
    pull.select_records()
    assert set(pull.wanted) == {f"{host}/run.json"}
    added = pull.expand([f"{host}/frames/0001.png", f"{host}/outputs/000000/depth.bin", f"{host}/absent.json"])
    assert sorted(path[len(str(host)) + 1:] for path in added) == [
        "frames/0001.png", "frames/0002.png", "outputs/000000/depth.bin", "outputs/000001/image.png"]
    assert [path[len(str(host)) + 1:] for path in pull.listed([f"{host}/raw"])] == ["raw/walkthrough.mp4"]
    pull.fetch("test")
    assert all(pull.cache.local(path).read_bytes() == Path(path).read_bytes() for path in pull.wanted)


def test_a_dropped_door_connection_is_read_again(monkeypatch):
    import io

    class Flaky:
        calls = 0

        def _request(self, method, route, query):
            assert (method, route) == ("GET", "/fs/list")
            Flaky.calls += 1
            if Flaky.calls < 3:
                error = RuntimeError("cannot reach the door")
                error.exit_code = 4  # the door client's DoorError code for a dropped connection
                raise error
            return io.BytesIO(b'{"type": "dir", "entries": []}')
    monkeypatch.setattr(job_replay.time, "sleep", lambda _seconds: None)
    door = job_replay.DoorReader(client=Flaky())
    assert door.list("/var/lib/blueprint") == {"type": "dir", "entries": []}
    assert door.retries == 2


def test_a_replay_out_of_rounds_says_its_stops_may_be_the_pulls():
    row = {"stage": "atlas_preview", "status": "ran", "would_buy": [], "duration_seconds": 1.0,
           "peak_rss_bytes": 2 << 20, "stop": {"kind": "refusal", "code": "website_source_geometry_artifact_changed"}}
    report = {"stages": [row], "clean": True, "converged": False, "named_not_pulled": {"files": 43},
              "evidence": {"files": 9, "unchanged": True}, "boundaries": {"network_attempts": []}}
    text = job_replay.table(report)
    assert text.splitlines()[0].split() == ["stage", "ran/skipped/assumed", "stop", "kind:code", "would", "buy",
                                            "(max", "USD)", "duration", "peak", "RSS"]
    assert "refusal:website_source_geometry_artifact_changed" in text and "2.0 MiB" in text
    assert "NOT CONVERGED: the last replay named 43 host files not pulled yet" in text
    assert "NOT CONVERGED" not in job_replay.table({**report, "converged": True})


def _scripted_rounds(monkeypatch, results):
    """``_replay_once`` answering from ``results`` in order, in a namespace stand-in; returns the rounds asked."""
    asked = []

    def once(pull, *, scratch, past, commit, steps, namespace, keep, index):
        asked.append(index)
        run_dir = Path(scratch) / "runs" / f"scripted-{index}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "stages": [], "assumed": [], "clean": True, **results[index]}

    monkeypatch.setattr(job_replay, "namespace_prefix", lambda _probe: [])
    monkeypatch.setattr(job_replay, "_replay_once", once)
    return asked


def test_a_round_that_reaches_the_network_ends_the_replay_and_its_report_keeps_the_breach(site, tmp_path, monkeypatch):  # noqa: F811
    """Codex, #3733: a later round that pulls more and reaches no network must not overwrite the breach."""
    bulk = site.job.parent / "survey/frames.bin"  # over the record bound: pulled only once a round names it
    bulk.write_bytes(b"\0" * (job_replay.SMALL_BYTES + 1))
    named = str(bulk)
    attempts = [{"kind": "connect", "address": "203.0.113.9:443"}]
    asked = _scripted_rounds(monkeypatch, [
        {"network_attempts": attempts, "breaches": ["network_attempt"], "wanted": [named]},
        {"network_attempts": [], "breaches": [], "wanted": []}])
    report = job_replay.replay_job(str(site.capture), door=_LocalDoor(), scratch=tmp_path / "scratch", rounds=2)
    assert asked == [0]  # nothing more is pulled or replayed after the breach
    assert report["passed"] is False and "network_attempt" in report["breaches"]
    assert report["boundaries"]["network_attempts"] == attempts


def test_a_request_id_replayed_online_replays_offline_by_the_same_id(site, tmp_path, monkeypatch):  # noqa: F811
    """Codex, #3733: ``--offline`` with the request id finds the listing the online replay saved."""
    _scripted_rounds(monkeypatch, [{"network_attempts": [], "breaches": [], "wanted": []}] * 2)
    request_id = site.capture.parents[1].name.removeprefix("site-")
    monkeypatch.setattr(job_replay, "resolve_capture", lambda _door, target: str(site.capture)
                        if target == request_id else target)
    scratch = tmp_path / "scratch"
    online = job_replay.replay_job(request_id, door=_LocalDoor(), scratch=scratch, rounds=0)
    assert online["capture_root"] == str(site.capture)
    offline = job_replay.replay_job(request_id, scratch=scratch, rounds=0, offline=True)
    assert "replay_unsupported" not in offline and offline["capture_root"] == str(site.capture)
    # An id never replayed online here is still refused, as before.
    unknown = job_replay.replay_job("itest-" + "0" * 40, scratch=scratch, rounds=0, offline=True)
    assert unknown["replay_unsupported"] == "offline_without_an_earlier_pull"
