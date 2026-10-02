"""ADP-009D/day28: real fixed syscalls/closure; CPU metadata boundary only."""
import json
import os
from pathlib import Path

import pytest

from tests.test_registered_experiment_issuer import installation, issue, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.test_registered_experiment_birth import prepare, birth
from tests.test_registered_disk_diagnostic_birth import _selector


def enrolled(installation, monkeypatch, profile="root_disk_diagnostic_disposable.v1", *, expiry=None):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline import control_plane_lane_experiment_birth as code
    config, settings, store, _ = installation
    public = prepare(installation)
    for key in ("work", "inputs"):
        (Path(settings["lane_scratch_" + key + "_root"]) / "diagnostics").mkdir(mode=0o750)
    request = store.parent / "diagnostic-request.json"
    request.write_bytes(encoded(diagnostic.build_request(installed_config_path=config, run_ref="run1")))
    request.chmod(0o600)
    grant = issue(installation, participant_profile=profile,
                  expires_at_epoch=expiry,
                  request_records=((request, _selector(request.read_bytes())),))
    monkeypatch.setattr(code, "_blueprint_identity", lambda: (0, 0))
    monkeypatch.setattr(os, "fchown", lambda *_: None)
    born = birth(installation, grant)
    return public, request, grant, born


def current(public):
    head = json.loads((public / "HEAD.json").read_bytes())
    return json.loads((public / head["record_name"]).read_bytes())["enrollments"][0]


def test_diagnostic_request_binds_exact_compatible_transport_source(installation):  # noqa: F811
    import hashlib
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    prepare(installation)
    request = diagnostic.build_request(installed_config_path=installation[0], run_ref="run1")
    raw = (Path(diagnostic.__file__).parent / "s3_compatible_transport.py").read_bytes()
    assert request["installed_sources"]["s3_compatible_transport"] == (
        "sha256:" + hashlib.sha256(raw).hexdigest())


def test_changed_compatible_transport_after_producer_close_cannot_mint_completion(
        installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    # Mirror only the bounded source closure; preserve every real checkout byte.
    source = Path(diagnostic.__file__).parent
    # Protected source cannot live beneath the writable pytest runner ancestor.
    # Reuse only this fixture's private protected installation, keeping the
    # actual ancestor checks unchanged on Linux as well as the Mac.
    mirror = installation[0].parent / "source-mirror"
    mirror.mkdir(mode=0o700)
    for name in diagnostic.SOURCE_MODULES | {"s3_compatible_transport"}:
        (mirror / (name + ".py")).write_bytes((source / (name + ".py")).read_bytes())
        (mirror / (name + ".py")).chmod(0o600)
    monkeypatch.setattr(diagnostic, "__file__", str(mirror / Path(diagnostic.__file__).name))
    public, request, grant, born = enrolled(installation, monkeypatch)
    actual_finish = _BirthFiles.finish
    closed = []
    def changed_after_close(files):
        actual_finish(files)
        closed.append(files)
        if len(closed) == 1:
            transport = mirror / "s3_compatible_transport.py"
            transport.write_bytes(transport.read_bytes() + b"\n# compatible-byte source drift\n")
    monkeypatch.setattr(_BirthFiles, "finish", changed_after_close)
    with pytest.raises(ValueError):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 1200)
    assert current(public)["completion"] is None
    assert not (installation[2] / (grant["intent_id"] + ".producer-completion.json")).exists()
    assert (Path(born["path"]) / diagnostic.REPORT_NAME).exists()
    assert closed and all(not files.owned and not files.unresolved for files in closed)


@pytest.mark.parametrize("profile", ["root_disk_diagnostic_disposable.v1", "root_disk_diagnostic_evidence.v1"])
def test_fixed_producer_observes_real_root_fds_then_closes_before_completion(installation, monkeypatch, profile):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    public, request, grant, born = enrolled(installation, monkeypatch, profile)
    observed = []
    actual = os.fstatvfs
    def root_observation(fd):
        info = os.fstat(fd)
        observed.append((info.st_dev, info.st_ino))
        return actual(fd)
    monkeypatch.setattr(os, "fstatvfs", root_observation)
    finished = []
    original_finish = _BirthFiles.finish
    def finish(files):
        original_finish(files)
        finished.append(files)
    monkeypatch.setattr(_BirthFiles, "finish", finish)
    result = diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
        request_path=request, installed_config_path=installation[0], now=lambda: 1200)
    report_path = Path(born["path"]) / diagnostic.REPORT_NAME
    report = json.loads(report_path.read_bytes())
    assert report == result["report"]
    assert report["intent_id"] == grant["intent_id"] and report["generation"] == born["generation"]
    roots = [Path(installation[1]["lane_scratch_" + key + "_root"]) for key in ("work", "inputs")]
    assert observed == [(path.stat().st_dev, path.stat().st_ino) for path in roots]
    assert [row["root"] for row in report["observations"]] == ["work", "inputs"]
    assert all(row["started_at_epoch"] <= row["finished_at_epoch"] == 1200
               and row["statvfs"]["f_blocks"] > 0 for row in report["observations"])
    assert len(finished) == 2 and all(not files.owned and not files.unresolved for files in finished)
    assert current(public)["completion"] == result["completion"] is not None
    assert {path.name for path in Path(born["path"]).iterdir()} == {
        ".lane-scratch.v1.json", ".registered-experiment.v1.json", diagnostic.REPORT_NAME}
    before = {path.name: path.read_bytes() for path in Path(born["path"]).iterdir()}
    with pytest.raises(ValueError, match="diagnostic_producer_inactive"):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 1300)
    assert {path.name: path.read_bytes() for path in Path(born["path"]).iterdir()} == before


@pytest.mark.parametrize("drift", ["foreign_payload", "changed_request", "expired"])
def test_unadmitted_diagnostic_never_mints_completion(installation, monkeypatch, drift):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    public, request, grant, born = enrolled(installation, monkeypatch)
    if drift == "foreign_payload":
        (Path(born["path"]) / "foreign").write_bytes(b"preserve")
    elif drift == "changed_request":
        request.write_bytes(b"{}")
    with pytest.raises(ValueError):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 2900 if drift == "expired" else 1200)
    assert current(public)["completion"] is None
    assert not (Path(born["path"]) / diagnostic.REPORT_NAME).exists()


@pytest.mark.parametrize("failure", ["report_changed", "foreign_after_close", "report_hardlink",
                                    "loader_changed", "policy_revoked", "unknown_close", "forged_closed_proof"])
def test_distinct_finalizer_requires_genuine_closure_and_unchanged_exact_tree(installation, monkeypatch, failure):  # noqa: F811
    from dataclasses import replace
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline import control_plane_lane_experiment_completion as completion
    from blueprint_pipeline import control_plane_lane_owner_consents as owners
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    public, request, grant, born = enrolled(installation, monkeypatch)
    target = Path(born["path"])
    original_finish = _BirthFiles.finish
    closed = []
    def finish(files):
        original_finish(files)
        closed.append(files)
        if len(closed) != 1:
            return
        assert not files.owned and not files.unresolved
        if failure == "report_changed":
            (target / diagnostic.REPORT_NAME).write_bytes(b"{}")
        elif failure == "foreign_after_close":
            (target / "foreign").write_bytes(b"preserve")
        elif failure == "report_hardlink":
            os.link(target / diagnostic.REPORT_NAME, request.parent / "external-report-alias")
        elif failure == "loader_changed":
            path = owners.INSTALLED_PACKAGE_ROOT / "operator_door/config.py"
            path.write_bytes(path.read_bytes() + b"\n# changed compatible loader\n")
        elif failure == "policy_revoked":
            path = installation[3]
            value = json.loads(path.read_bytes())
            value["enabled"] = False
            path.write_bytes(encoded(value))
        elif failure == "unknown_close":
            raise ValueError("diagnostic_producer_closure_unproven")
    monkeypatch.setattr(_BirthFiles, "finish", finish)
    if failure == "forged_closed_proof":
        actual = completion.complete_disk_diagnostic
        def cloned(proof, **options):
            return actual(replace(proof), **options)
        monkeypatch.setattr(completion, "complete_disk_diagnostic", cloned)
    with pytest.raises(ValueError):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 1200)
    assert current(public)["completion"] is None
    assert not (installation[2] / (grant["intent_id"] + ".producer-completion.json")).exists()
    assert closed and all(not files.owned and not files.unresolved for files in closed)
    assert (target / diagnostic.REPORT_NAME).exists()


@pytest.mark.parametrize("completed", [False, True])
def test_generic_reader_cannot_reopen_diagnostic_generation(installation, monkeypatch, completed):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline import control_plane_lane_experiment_consumer as consumer
    public, request, grant, born = enrolled(installation, monkeypatch)
    if completed:
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 1200)
    monkeypatch.setattr(consumer, "AUTHORITY_ROOT", public)
    monkeypatch.setattr(consumer, "_blueprint_gid", lambda: 0)
    with pytest.raises(ValueError, match="diagnostic_consumer_unsupported"):
        consumer.RegisteredExperimentUse.admit(Path(born["path"]), now=lambda: 1300)


@pytest.mark.parametrize("boundary", ["completion", "head", "late_named_guard"])
def test_real_elapsed_expiry_during_publication_cannot_publish_completion_or_head(installation, monkeypatch, boundary):  # noqa: F811
    import time
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    public, request, grant, _ = enrolled(installation, monkeypatch, expiry=1201.5)
    old_head = (public / "HEAD.json").read_bytes()
    origin = time.monotonic()
    actual_write = os.write
    head_writes = 0
    delayed = False
    def delay():
        nonlocal delayed
        delayed = True
        # Genuine elapsed time crosses this fixture's original signed expiry.
        # The producer's five-second monotonic budget remains unchanged.
        time.sleep(max(0, origin + 1.5 - time.monotonic()) + 0.03)
    def write(fd, payload):
        nonlocal head_writes
        result = actual_write(fd, payload)
        raw = bytes(payload)
        if b'"schema_version":"control_plane_lane_experiment_head.v1"' in raw:
            head_writes += 1
        selected = (boundary == "completion" and b'"schema_version":"control_plane_lane_disk_diagnostic_completion.v1"' in raw
                    or boundary == "head" and head_writes == 2)
        if selected and not delayed:
            delay()
        return result
    monkeypatch.setattr(os, "write", write)
    if boundary == "late_named_guard":
        actual_stat = os.stat
        def stat(path, **options):
            result = actual_stat(path, **options)
            if path == "HEAD.json" and head_writes == 2 and not delayed:
                delay()
            return result
        monkeypatch.setattr(os, "stat", stat)
    with pytest.raises(ValueError, match="diagnostic_producer_inactive"):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 1200 + time.monotonic() - origin)
    assert delayed and (public / "HEAD.json").read_bytes() == old_head
    assert current(public)["completion"] is None
    if boundary == "completion":
        assert not (installation[2] / (grant["intent_id"] + ".producer-completion.json")).exists()


def test_finalizer_cannot_reset_original_producer_epoch_floor(installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    from blueprint_pipeline.control_plane_lane_experiment_publication import _BirthFiles
    public, request, grant, born = enrolled(installation, monkeypatch)
    clock = [1200.0]
    def now():
        clock[0] += 0.001
        return clock[0]
    original_finish = _BirthFiles.finish
    rolled = []
    def finish(files):
        original_finish(files)
        if rolled:
            return
        report = json.loads((Path(born["path"]) / diagnostic.REPORT_NAME).read_bytes())
        clock[0] = report["observations"][-1]["finished_at_epoch"] + 0.0001
        assert clock[0] < files.last_epoch
        rolled.append(True)
    monkeypatch.setattr(_BirthFiles, "finish", finish)
    with pytest.raises(ValueError, match="diagnostic_producer_inactive"):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=now)
    assert rolled and current(public)["completion"] is None


def test_expiry_during_temp_birth_guard_performs_no_creation_syscall(installation, monkeypatch):  # noqa: F811
    import time
    from blueprint_pipeline import control_plane_lane_disk_diagnostic as diagnostic
    public, request, grant, _ = enrolled(installation, monkeypatch, expiry=1201.5)
    origin = time.monotonic()
    actual_stat, actual_open = os.stat, os.open
    delayed, created = [], []
    def stat(path, **options):
        try:
            return actual_stat(path, **options)
        except FileNotFoundError:
            if isinstance(path, str) and path.startswith(".target-version-") and not delayed:
                delayed.append(True)
                time.sleep(max(0, origin + 1.5 - time.monotonic()) + 0.03)
            raise
    def open(path, flags, *args, **options):
        if flags & os.O_CREAT:
            created.append(path)
        return actual_open(path, flags, *args, **options)
    monkeypatch.setattr(os, "stat", stat)
    monkeypatch.setattr(os, "open", open)
    with pytest.raises(ValueError, match="diagnostic_producer_inactive"):
        diagnostic.run_registered_disk_diagnostic(grant["intent_id"], expected_intent=grant["intent"],
            request_path=request, installed_config_path=installation[0], now=lambda: 1200 + time.monotonic() - origin)
    assert delayed and created == []
    assert current(public)["completion"] is None
