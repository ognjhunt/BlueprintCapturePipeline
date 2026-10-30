"""Real process exits and retained development USD; no Isaac, GPU or policy run."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_native_smoke as smoke
from blueprint_pipeline import deployment_embodiment_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.deployment_embodiment_preparation_contract import sealed, write_once
from tests import test_embodiment_native_smoke as smoke_fixtures

configured = smoke_fixtures.configured
inputs = smoke_fixtures.inputs


# Exercise the real import_one ordering, replacing only native import/startup.
# close() exits the interpreter immediately, as default Isaac fast shutdown does.
IMPORT_CHILD = r'''
import json, os, shutil, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from blueprint_pipeline import deployment_embodiment_preparation_worker as worker
from blueprint_pipeline import deployment_embodiment_runtime_configuration as config
from blueprint_pipeline import deployment_embodiment_asset_import as importer
from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_isaac_launch as launch
source, prepared, body, identity, recipe = sys.argv[1:]
def prepare(_source, output):
    shutil.copytree(source, output)
    return config.verify_runtime_configuration(output)
def convert(_source, output, **_options):
    shutil.copytree(prepared, output)
    return importer.verify_prepared_asset_directory(output)
with patch.object(worker, "verify_inputs", return_value={"request": {"embodiment_ids": [identity]}}), \
     patch.object(assets, "recipe_for", return_value=json.loads(recipe)), \
     patch.object(config, "prepare_runtime_configuration", side_effect=prepare), \
     patch.object(importer, "import_source_asset", side_effect=convert), \
     patch.object(launch, "launch_embodiment_isaaclab", return_value=(
         SimpleNamespace(close=lambda: os._exit(0)), {"development_only_fake_sdk": True})):
    worker.import_one(Path("unused-development-runtime"), Path(body), identity)
raise AssertionError("SDK close must terminate before this statement")
'''


@pytest.fixture
def batch(inputs, monkeypatch, tmp_path):
    identity = inputs.binding["robot_embodiment"]["embodiment_id"]
    clock = SimpleNamespace(monotonic=0., wall=1000.)
    value = {"inputs_digest": "sha256:" + "1" * 64,
        "request": {"embodiment_ids": [identity], "request_digest": "sha256:" + "2" * 64,
                    "expires_at_epoch": 10000.},
        "sources": [{"embodiment_id": identity, "status": "source_ready"}]}
    monkeypatch.setattr(worker, "verify_inputs", lambda _path: value)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "10000")
    monkeypatch.setenv("BLUEPRINT_EMBODIMENT_PROVISIONING_RECEIPT", "/unused-development-receipt")
    return SimpleNamespace(inputs=inputs, identity=identity, clock=clock, value=value,
                           output=tmp_path / "batch-output", runtime=tmp_path / "runtime",
                           recipe=json.dumps(assets.recipe_for(identity)))


def execute(batch, run):
    return worker.run_batch(batch.runtime, batch.output, run=run,
                           now=lambda: batch.clock.monotonic, wall=lambda: batch.clock.wall)


def checkpoint_body(batch, body):
    """Seal real development source/import bytes, never a native execution claim."""
    body.mkdir()
    shutil.copytree(batch.inputs.configuration, body / "configuration")
    shutil.copytree(batch.inputs.model.parent, body / "prepared")
    binding = batch.inputs.binding
    provenance = binding["model_record"]["source_import_provenance"]
    row = {"embodiment_id": batch.identity, "status": "imported", "native_import_executed": True,
        "configuration_digest": binding["configuration_digest"], "import_digest": provenance["import_digest"],
        "source_bundle_digest": provenance["source_bundle_digest"], "model_sha256": binding["model_record"]["sha256"],
        "task_executed": False, "policy_executed": False, "development_only_fixture": True}
    write_once(body / "binding-inputs.json", binding)
    write_once(body / "import-checkpoint.json", sealed({
        "schema_version": "embodiment_import_checkpoint.v1", "body_result": row,
        "binding_digest": canonical_digest(binding)}, "checkpoint_digest"))


def no_native_diagnostic(body, *, wall_seconds):
    return ORIGINAL_BOUNDED_PROCESS(body, wall_seconds=wall_seconds,
                                    run=lambda *_args, **_kwargs: SimpleNamespace(returncode=23))


def test_sdk_os_exit_preserves_checkpoint_and_parent_starts_separate_diagnostic(batch, monkeypatch):
    calls = []

    def importer_process(command, **options):
        body = Path(command[command.index("--output") + 1])
        result = subprocess.run([sys.executable, "-c", IMPORT_CHILD, str(batch.inputs.configuration),
            str(batch.inputs.model.parent), str(body), batch.identity, batch.recipe], **options)
        assert result.returncode == 0, (batch.output / (batch.identity + ".log")).read_text()
        assert (body / "import-checkpoint.json").is_file()
        assert not (body / "body-result.json").exists()
        calls.append("importer_exited")
        return result

    def diagnostic(body, *, wall_seconds):
        assert worker.verified_import_checkpoint(body, batch.identity)["status"] == "imported"
        assert not (body / "body-result.json").exists()
        calls.append("verified_parent_diagnostic")
        return no_native_diagnostic(body, wall_seconds=wall_seconds)

    monkeypatch.setattr(smoke, "run_bounded_smoke_process", diagnostic)
    result = execute(batch, importer_process)
    assert calls == ["importer_exited", "verified_parent_diagnostic"]
    row = result["results"][0]
    assert row["status"] == "imported"
    assert row["native_controller_smoke"]["status"] == "failed"
    assert row["native_controller_smoke"]["reasons"] == ["native_controller_process_failed:23"]
    body = batch.output / "bodies" / batch.identity
    assert json.loads((body / "body-result.json").read_text()) == row
    assert (body / "prepared/robot.usda").is_file()
    assert "bodies/" + batch.identity + "/native-controller-process.v1.json" in {r["path"] for r in result["files"]}


ORIGINAL_BOUNDED_PROCESS = smoke.run_bounded_smoke_process


def test_import_exception_survives_sdk_os_exit_in_retained_log(batch, monkeypatch):
    child = IMPORT_CHILD.replace('shutil.copytree(prepared, output)',
                                 'raise ValueError("development_only_converter_failure")')

    def importer_process(command, **options):
        body = Path(command[command.index("--output") + 1])
        return subprocess.run([sys.executable, "-c", child, str(batch.inputs.configuration),
            str(batch.inputs.model.parent), str(body), batch.identity, batch.recipe], **options)

    monkeypatch.setattr(smoke, "run_bounded_smoke_process", lambda *_a, **_k: pytest.fail("failed import diagnostic"))
    result = execute(batch, importer_process)
    assert result["results"][0]["status"] == "import_failed"
    assert "ValueError: development_only_converter_failure" in (batch.output / (batch.identity + ".log")).read_text()


@pytest.mark.parametrize("fault", ["checkpoint", "model", "binding"])
def test_changed_completed_import_never_starts_parent_diagnostic(batch, monkeypatch, fault):
    def importer_process(command, **_options):
        body = Path(command[command.index("--output") + 1])
        checkpoint_body(batch, body)
        target = {"checkpoint": "import-checkpoint.json", "model": "prepared/robot.usda",
                  "binding": "binding-inputs.json"}[fault]
        path = body / target
        path.chmod(0o600)
        path.write_text("changed retained bytes")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(smoke, "run_bounded_smoke_process", lambda *_a, **_k: pytest.fail("unverified import diagnostic"))
    result = execute(batch, importer_process)
    assert result["results"][0]["status"] == "import_failed"
    assert result["results"][0]["checkpoint_recovery"].startswith("unverified:")
    assert not (batch.output / "bodies" / batch.identity).exists()


@pytest.mark.parametrize("elapsed,expected", [(1140., 55.), (1198., 0.)])
def test_parent_diagnostic_uses_original_body_deadline(batch, monkeypatch, elapsed, expected):
    observed = []

    def importer_process(command, **options):
        checkpoint_body(batch, Path(command[command.index("--output") + 1]))
        assert options["timeout"] == 1200.
        assert float(options["env"]["BLUEPRINT_EMBODIMENT_BODY_DEADLINE_EPOCH"]) == 2200.
        batch.clock.monotonic += elapsed
        batch.clock.wall += elapsed
        return SimpleNamespace(returncode=0)

    def diagnostic(body, *, wall_seconds):
        observed.append(wall_seconds)
        return ORIGINAL_BOUNDED_PROCESS(body, wall_seconds=wall_seconds,
            run=lambda *_a, **_k: SimpleNamespace(returncode=23))

    monkeypatch.setattr(smoke, "run_bounded_smoke_process", diagnostic)
    result = execute(batch, importer_process)
    assert observed == [expected]
    assert result["results"][0]["status"] == "imported"
    if expected == 0:
        assert result["results"][0]["native_controller_smoke"]["reasons"] == ["native_controller_process_budget_exhausted"]


def test_parent_diagnostic_io_failure_keeps_verified_import(batch, monkeypatch):
    def importer_process(command, **_options):
        checkpoint_body(batch, Path(command[command.index("--output") + 1]))
        return SimpleNamespace(returncode=0)

    def diagnostic(*_a, **_k):
        raise OSError("development-only diagnostic write failure")

    monkeypatch.setattr(smoke, "run_bounded_smoke_process", diagnostic)
    result = execute(batch, importer_process)
    row = result["results"][0]
    assert row["status"] == "imported"
    assert row["native_controller_smoke"]["status"] == "failed"
    assert row["native_controller_smoke"]["checkpoint_recovered"] is True
    assert (batch.output / "bodies" / batch.identity / "prepared/robot.usda").is_file()


def test_diagnostic_child_seals_measured_fixture_evidence_before_sdk_os_exit(batch, tmp_path):
    body = tmp_path / "diagnostic-body"
    checkpoint_body(batch, body)
    child = r'''
import json, os, sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch
from blueprint_pipeline import deployment_embodiment_native_smoke as smoke
from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_isaac_launch as launch
from tests.test_embodiment_native_smoke import Stepper
run = partial(smoke.run_native_controller_smoke, backend_factory=Stepper)
with patch.object(smoke, "run_native_controller_smoke", run), \
     patch.object(assets, "recipe_for", return_value=json.loads(sys.argv[2])), \
     patch.object(launch, "launch_embodiment_isaaclab", return_value=(
         SimpleNamespace(close=lambda: os._exit(0)), {"development_only_fake_sdk": True})):
    smoke.main(["--body", sys.argv[1], "--wall-seconds", "60"])
raise AssertionError("SDK close must terminate")
'''

    def native_process(_command, **options):
        return subprocess.run([sys.executable, "-c", child, str(body), batch.recipe], **options)

    result = ORIGINAL_BOUNDED_PROCESS(body, wall_seconds=60., run=native_process)
    assert result["status"] == "passed", (body / "native-controller-smoke.log").read_text()
    measured = json.loads((body / smoke.FILENAME).read_text())
    assert measured["runtime_readback"]["device"] == "cpu-test-fixture"
    assert measured["physics_ticks"] == 400
    assert measured["smoke_digest"] == result["smoke_digest"]
    assert measured["candidate_policy_queried"] is False
    assert measured["task_executed"] is False
