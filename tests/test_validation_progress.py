import json

import pytest

from blueprint_pipeline import validation_file_digests as digests
from blueprint_pipeline import validation_progress as progress


def test_unbound_heartbeat_is_a_no_op(tmp_path):
    assert progress.heartbeat("prefix_phase", through_phase="calibrated_views") is None
    assert list(tmp_path.iterdir()) == []


def test_bound_heartbeat_reports_step_elapsed_and_digest_counters(tmp_path, capsys):
    artifact = tmp_path / "source_standard.ply"
    artifact.write_bytes(b"a" * digests.MINIMUM_BYTES)
    artifact.chmod(0o600)
    sink = tmp_path / "scene" / progress.FILENAME
    sink.parent.mkdir()
    with progress.progress_sink(sink, intent_id="scene-1", source_commit="c" * 40), digests.file_digest_scope():
        first = progress.heartbeat("factory_start")
        digests.sha256_file(artifact)
        digests.sha256_file(artifact)
        second = progress.heartbeat("prefix_candidate", candidate_index=1, candidate_total=14)
    record = json.loads(sink.read_text())
    assert record == second
    assert record["schema_version"] == progress.SCHEMA
    assert record["intent_id"] == "scene-1" and record["source_commit"] == "c" * 40
    assert record["step"] == "prefix_candidate" and record["candidate_total"] == 14
    assert first["heartbeat_sequence"] == 1 and record["heartbeat_sequence"] == 2
    assert record["elapsed_seconds"] >= first["elapsed_seconds"] >= 0
    assert first["digests"]["files_hashed"] == 0
    assert record["digests"]["files_hashed"] == 1
    assert record["digests"]["bytes_hashed"] == digests.MINIMUM_BYTES
    assert record["digests"]["cache_hits"] == 1
    assert record["digests"]["bytes_reused"] == digests.MINIMUM_BYTES
    assert record["digests"]["last_path"] == str(artifact)
    err = capsys.readouterr().err
    assert "step=prefix_candidate" in err and "cache_hits=1" in err and "candidate_index=1" in err
    assert not sink.with_name(sink.name + ".tmp").exists()


def test_nested_sink_is_a_no_op_and_outer_survives_exceptions(tmp_path):
    outer, inner = tmp_path / "outer.json", tmp_path / "inner.json"
    with pytest.raises(RuntimeError), progress.progress_sink(outer, intent_id="a"):
        with progress.progress_sink(inner, intent_id="b"):
            progress.heartbeat("prefix_phase", through_phase="calibrated_views")
        raise RuntimeError("validation refused")
    assert outer.exists() and not inner.exists()
    assert json.loads(outer.read_text())["intent_id"] == "a"
    assert progress.heartbeat("after") is None


def test_heartbeat_without_digest_scope_reports_null_digests(tmp_path):
    with progress.progress_sink(tmp_path / progress.FILENAME):
        record = progress.heartbeat("recover")
    assert record["digests"] is None
