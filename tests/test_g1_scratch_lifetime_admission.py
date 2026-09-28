# Covers (for impacted-test selection):
#   src/blueprint_pipeline/native_g1_development_worker.py
#   src/blueprint_pipeline/native_g1_development_pair.py
#   src/blueprint_pipeline/control_plane_scratch_lifetime.py
"""Fake direct workers prove lifetime admission before every output path check."""

import json

import pytest

from blueprint_pipeline import control_plane_scratch_lifetime as lifetime
from blueprint_pipeline import native_g1_development_worker as worker
from tests.test_leased_scratch_lifetime import folder, open_use


def test_standalone_enrolled_worker_refuses_before_path_access_or_mkdir(tmp_path, monkeypatch):
    root, path = folder(tmp_path)
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    monkeypatch.setattr(worker, "_request", lambda value: value)
    monkeypatch.setattr(worker, "_run_g1_development_worker", lambda **kwargs: pytest.fail("worker reached"), raising=False)
    with pytest.raises(Exception, match="lifetime_required"):
        worker.run_g1_development_worker(request={}, output_dir=path / "candidate")
    assert not (path / "candidate").exists()


def test_supported_worker_holds_target_authority_through_fake_terminal_write(tmp_path, monkeypatch):
    root, path = folder(tmp_path)
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    monkeypatch.setattr(worker, "_request", lambda value: value)
    def fake(**kwargs):
        with pytest.raises(Exception, match="consumer_busy"):
            lifetime.LeasedScratchUse.probe(path, now=lambda: 110)
        kwargs["output_dir"].mkdir()
        (kwargs["output_dir"] / "terminal.json").write_text(json.dumps({"status": "fake"}))
        return {"status": "fake"}
    monkeypatch.setattr(worker, "_run_g1_development_worker", fake, raising=False)
    with open_use(root) as use:
        assert worker.run_g1_development_worker(request={}, output_dir=path / "candidate", scratch_lifetime=use) == {"status": "fake"}


@pytest.mark.parametrize("fault", ["closed", "escape", "changed"])
def test_invalid_worker_authority_never_reaches_output_writer(tmp_path, monkeypatch, fault):
    root, path = folder(tmp_path)
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    monkeypatch.setattr(worker, "_request", lambda value: value)
    monkeypatch.setattr(worker, "_run_g1_development_worker", lambda **kwargs: pytest.fail("worker reached"), raising=False)
    with open_use(root) as use:
        output = path / "candidate"
        if fault == "closed":
            use.close()
        elif fault == "escape":
            output = path / ".." / "foreign"
        else:
            path.rename(path.with_name("moved"))
            path.mkdir()
        with pytest.raises(Exception, match="lane_scratch"):
            worker.run_g1_development_worker(request={}, output_dir=output, scratch_lifetime=use)
        assert not output.exists()


def test_opted_pair_holds_authority_from_worker_through_terminal_receipt(tmp_path, monkeypatch):
    from blueprint_pipeline import native_g1_development_pair as pair
    root = tmp_path / "lanes"
    root.mkdir()
    output = root / "g1" / "pair"
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root,))
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    requests = [{"candidate_id": str(index), "request_digest": "sha256:" + str(index) * 64,
                 **{key: str(tmp_path / key) for key in pair.PATH_FIELDS}} for index in (1, 2)]
    monkeypatch.setattr(pair, "validate_g1_development_pair", lambda _: {"candidate_ids": ["1", "2"], "scene_plan_digest": "scene", "objective_id": "fake"})
    monkeypatch.setattr(pair, "_sealed_json", lambda p: requests[paths.index(p)])
    worker_result = {"status": "blocked", "result_digest": "digest"}
    def fake(**kwargs):
        with pytest.raises(Exception, match="consumer_busy"):
            lifetime.LeasedScratchUse.probe(output)
        kwargs["output_dir"].mkdir()
        return worker_result
    monkeypatch.setattr(worker, "_run_g1_development_worker", fake)
    monkeypatch.setattr(pair, "_read_result", lambda *args, **kwargs: worker_result)
    original_write = type(output).write_text
    observed = []
    def write(self, *args, **kwargs):
        if self == output / (pair.SCHEMA + ".json"):
            with pytest.raises(Exception, match="consumer_busy"):
                lifetime.LeasedScratchUse.probe(output)
            observed.append("terminal_guarded")
        return original_write(self, *args, **kwargs)
    monkeypatch.setattr(type(output), "write_text", write)
    result = pair.run_g1_development_pair(request_paths=paths, output_dir=output,
                                         scratch_owner="owner", scratch_run_ref="run", scratch_ttl_seconds=100,
                                         cooperating_lifetime=True)
    assert result["status"] == "blocked" and observed == ["terminal_guarded"]
    assert result["consumer_lifetime_scope"] == "coordinator_and_direct_worker_only"
    assert "child_consumer_participation_unproven" in result["consumer_lifetime_omissions"]
    with lifetime.LeasedScratchUse.probe(output):
        pass


def test_enrolled_pair_rejects_unknown_local_adapter_before_output_creation(tmp_path, monkeypatch):
    from blueprint_pipeline import native_g1_development_pair as pair
    root = tmp_path / "lanes"
    root.mkdir()
    monkeypatch.setattr(pair, "LANE_SCRATCH_ROOTS", (root,))
    monkeypatch.setattr(pair, "validate_g1_development_pair", lambda _: {})
    with pytest.raises(ValueError, match="participation_unproven"):
        pair.run_g1_development_pair(request_paths=[], output_dir=root / "g1" / "pair",
                                     scratch_owner="owner", scratch_run_ref="run", scratch_ttl_seconds=100,
                                     local_runner=lambda **kwargs: pytest.fail("called"), cooperating_lifetime=True)
    assert not (root / "g1").exists()


@pytest.mark.slow
def test_cli_malformed_request_accounts_for_all_passed_descriptors(tmp_path):
    import os
    request = tmp_path / "request.json"
    request.write_text("malformed private request")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    input_read, input_write = os.pipe()
    output_read, output_write = os.pipe()
    try:
        with pytest.raises(Exception, match="handshake"):
            worker.main(["--request", str(request), "--output-dir", str(tmp_path / "output"),
                         "--lifetime-fd", str(descriptor), "--lifetime-input-fd", str(input_read),
                         "--lifetime-output-fd", str(output_write)])
        for fd in (descriptor, input_read, output_write):
            with pytest.raises(OSError):
                os.fstat(fd)
        assert not (tmp_path / "output").exists()
    finally:
        for fd in (descriptor, input_read, input_write, output_read, output_write):
            try:
                os.close(fd)
            except OSError:
                pass
