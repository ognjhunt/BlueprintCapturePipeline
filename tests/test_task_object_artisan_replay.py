"""The local ARTISAN replay runs the paid driver's own glue on a finished run, at $0.

capture-447ae4f4 (2026-11-02) found its stage-3 export bugs one GPU hour at a
time. The replay runs the same post-ARTISAN code on a run's retained files.
"""
import json
import shutil

import pytest

from blueprint_pipeline import task_object_artisan_replay as replay
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_authoring_contracts import file_record

from .test_task_object_artisan_exports import construction_request, export_fixture, usd_fixture


def finished_producer(tmp_path, *, phase="done"):
    producer = tmp_path / "producer"
    workspace = producer / "artisan_runtime/workspace/data/cache/runs/pl_fixture/workspace"
    workspace.mkdir(parents=True)
    terminal, manifest = export_fixture(workspace)
    exported = workspace / "exports/object.usd"
    shutil.copyfile(usd_fixture(workspace / "geometry"), exported)
    record = file_record(exported)
    manifest["artifacts"]["object.usd"].update(sha256=record["sha256"][7:], size_bytes=record["size_bytes"])
    (workspace / "exports/export_manifest.json").write_text(json.dumps(manifest))
    terminal.update(phase=phase, paths={**terminal["paths"], "root": str(workspace)})
    # As the GPU wrote it: absolute paths under the provider bundle's runtime output.
    terminal["paths"] = {key: value.replace(str(producer), replay.GPU_PRODUCER_ROOT)
                         for key, value in terminal["paths"].items()}
    (producer / replay.TERMINAL).write_text(json.dumps(terminal))
    request = construction_request()
    identity = {"id": "fixture-tray", "version": "v1"}
    request["configuration"]["replacement_identity"] = identity
    source = {"path": "/workspace/stays-on-host.glb", "sha256": "sha256:" + "0" * 64, "size_bytes": 1,
              "digest": "sha256:" + "0" * 64}
    request.update(replacement_identity=identity, source_frames=[source], source_candidate=source)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    (producer / replay.REQUEST).write_text(json.dumps(request))
    return producer


def test_a_finished_run_replays_through_stage_four(tmp_path):
    report = replay.replay(finished_producer(tmp_path), tmp_path / "out")
    assert report["status"] == "passed", report
    assert [row["step"] for row in report["steps"]][-3:] == [
        "package_upstream_candidate", "verify_artisan_candidate", "stage4_static_qualification"]
    assert (tmp_path / "out/static_qualification_receipt.v1.json").is_file()
    assert json.loads((tmp_path / "out/replay_report.json").read_text())["status"] == "passed"


def test_the_first_stop_is_reported_typed(tmp_path):
    # The bug #3247 fixed: an export check that expected the enum name, not its value.
    report = replay.replay(finished_producer(tmp_path, phase="DONE"), tmp_path / "out")
    assert report["status"] == "failed"
    assert report["first_failure"]["step"] == "package_upstream_candidate"
    assert report["first_failure"]["reason"] == "artisan_not_done_and_registered"


def test_a_path_outside_the_producer_is_refused():
    with pytest.raises(ValueError, match="artisan_replay_path_outside_producer:root"):
        replay.rebase_terminal({"paths": {"root": "/elsewhere/workspace"}},
                               original_root=replay.GPU_PRODUCER_ROOT, local_root="/tmp/producer")


def test_a_launch_pull_reads_only_what_the_export_reads(tmp_path):
    source = finished_producer(tmp_path)
    requested = []

    def pull(host_path, local_path):
        requested.append(host_path)
        relative = host_path.split(replay.PRODUCER_RELATIVE + "/", 1)[1]
        origin = source / relative
        if not origin.is_file():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, local_path)
        return True
    assert replay.pull_launch("launch-x", tmp_path / "pulled", pull=pull) == []
    assert all(path.startswith(f"{replay.HOST_LAUNCH_ROOT}/launch-x/{replay.PRODUCER_RELATIVE}/")
               for path in requested)
    # Source frames and source geometry are never requested; the site's images stay on the host.
    assert not any("stays-on-host" in path or "stage-2" in path for path in requested)
    assert replay.replay(tmp_path / "pulled", tmp_path / "out")["status"] == "passed"
