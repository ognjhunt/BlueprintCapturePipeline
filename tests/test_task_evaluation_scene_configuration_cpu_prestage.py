"""Stages 1-4 run on the control plane; the paid run adopts them at the same paths.

The prestage runs the sealed bundle's own entrypoint at the paid run's
logical paths with the paid run's composed environment. Its only product is
the runner's completed-prefix checkpoint plus a transport record, carried as
a digest-bound capsule the runtime verifies and restores at those exact
paths before executing stage 5.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_cpu_prestage as prestage
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_output_archive import preserve_stage_prefix

RUN_ID = "configure-scene-v1"
COMMIT = "a" * 40
BUNDLE = prestage.BUNDLE_DIRNAME
ENTRYPOINT = "run_task_evaluation_scene_configuration_provider.sh"


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _bundle(tmp_path: Path, *, backend: str = "astra_cad_blender_v1") -> dict:
    archive = tmp_path / "bundle.zip"
    if not archive.exists():
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr(prestage.ENTRYPOINT, "#!/usr/bin/env bash\necho fixture\n")
            zipped.writestr(prestage.MANIFEST_NAME, json.dumps({"replacement_authoring_backend": backend}))
            zipped.writestr("provider_runtime/task_evaluation_scene_configuration_provider_runner.py", "")
            zipped.writestr("provider_runtime/toolchain/manifest.json", "{}")
            zipped.writestr("provider_runtime/input/portable_construction_envelope.v1.json", "{}")
    return {"bundle_path": str(archive), "bundle_sha256": prestage._sha(archive), "run_id": RUN_ID,
            "source_commit": COMMIT, "carried_completed_stage_count": 0}


def _write_result(env: dict, *, status: str, stage_ids: list[str]) -> None:
    output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT"])
    chain = {"status": status, "stage_limit": env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"],
             "whole_run_completed": False, "stage_count": len(stage_ids),
             "stage_results": [{"stage_id": stage_id} for stage_id in stage_ids]}
    result = {"schema_version": "task_evaluation_scene_configuration_provider_result.v1", "status": status,
              "run_id": RUN_ID, "source_commit": COMMIT, "stage_chain": chain, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output / prestage.RESULT_NAME).write_text(canonical_json(result) + "\n")


def _fake_entrypoint(stage_ids: list[str], *, status: str = "completed_prefix", seen: dict | None = None):
    """Behave like the bundle entrypoint: seal a prefix at the given paths."""
    def run(argv, *, cwd, env, stdout, stderr, check):
        if seen is not None:
            seen.update(env)
        assert argv[0] == "bash" and argv[1].endswith(ENTRYPOINT) and cwd == str(Path(argv[1]).parent)
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT"])
        # The real wrapper seals the unpacked toolchain before execution.
        Path(cwd, "toolchain/manifest.json").chmod(0o444)
        Path(cwd, "toolchain").chmod(0o555)
        stages = output / "stages"
        stages.mkdir()
        binding = {"schema_version": "astra_split_stage_resume_binding.v1", "output_root": str(stages),
                   "prefix_stage_limit": env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"],
                   "parent_deadline_epoch": float(env["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"])}
        binding["binding_digest"] = canonical_digest(binding, digest_field="binding_digest")
        (stages / "astra_same_run_resume_binding.json").write_text(canonical_json(binding) + "\n")
        for stage_id in stage_ids:
            (stages / stage_id / "adapter").mkdir(parents=True)
            (stages / stage_id / "completed_stage_checkpoint.json").write_text("{}")
            (stages / stage_id / "adapter/artifact.usda").write_text("#usda 1.0\n")
        (stages / "stage-5").mkdir()
        (stages / "stage-5/partial.json").write_text("{}")
        (stages / "stage-3/official_openai_cost").mkdir(parents=True, exist_ok=True)
        (stages / "stage-3/official_openai_cost/reservation.json").write_text('{"reserved_usd": 3}')
        preserve_stage_prefix(output_root=output, completed_results=[{"stage_id": s} for s in stage_ids],
                              checkpoint_path=Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_CHECKPOINT_PATH"]))
        _write_result(env, status=status, stage_ids=stage_ids)
        stdout.write(b"entrypoint ran\n")
        return type("Completed", (), {"returncode": 0})()
    return run


def _prepare(tmp_path: Path, runner, **overrides):
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    work = tmp_path / "workspace"
    work.mkdir(exist_ok=True)
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin/python3").write_text("")
    key = tmp_path / "key"
    key.write_text("scoped-test-key")
    key.chmod(0o600)
    kwargs = dict(bundle_receipt=_bundle(tmp_path), authority={"authority_digest": "sha256:" + "a" * 64},
                  job_dir=job, environment={"OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(tmp_path / "key"),
                                            "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "3",
                                            "BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH": "9999999999",
                                            "PATH": "/usr/bin:/bin"},
                  secret_file_paths={"OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(key)},
                  stage_limit="stage-4", runner=runner, work_dir=work, python_bin_dir=tmp_path / "bin",
                  now=lambda: 1_000.0, reservation_root=tmp_path / "reservations", which=lambda name: "/usr/bin/" + name,
                  disk_usage=lambda _p: type("Usage", (), {"total": 400 * 1024**3, "used": 0,
                                                            "free": 400 * 1024**3})())
    kwargs.update(overrides)
    return prestage.prepare_stage_prefix_before_gpu(**kwargs), job


def test_prefix_runs_the_bundle_entrypoint_at_the_paid_paths_and_archives_only_completed_stages(tmp_path):
    seen: dict = {}
    receipt, job = _prepare(tmp_path, _fake_entrypoint(["stage-1", "stage-2", "stage-3", "stage-4"], seen=seen))
    work = tmp_path / "workspace"
    assert receipt["status"] == "completed_prefix_before_gpu_allocation"
    assert receipt["completed_stage_ids"] == ["stage-1", "stage-2", "stage-3", "stage-4"]
    assert receipt["gpu_execution_performed"] is False and receipt["execution_site"] == "control_plane"
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    capsule = Path(receipt["capsule_path"])
    assert capsule == job / "cpu_prestage_capsule.zip" and prestage._sha(capsule) == receipt["capsule_sha256"]
    # The paid run's paths and environment; the prefix gets its own bounded deadline.
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_RUNTIME_ROOT"] == str(work / BUNDLE / "provider_runtime")
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT"] == str(work / BUNDLE / "runtime_output")
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_STAGE_CHECKPOINT_PATH"] == str(work / prestage.CHECKPOINT_NAME)
    assert seen["BLUEPRINT_VAST_WORK_DIR"] == str(work)
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"] == "stage-4"
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"] == "3"
    assert seen["OPENAI_CONTENT_AGENTS_API_KEY_FILE"] == str(tmp_path / "key")
    assert float(seen["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"]) == 1_000.0 + prestage.DEFAULT_TTL_SECONDS
    assert seen["PATH"].startswith(str(tmp_path / "bin") + ":")
    assert seen["BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS"] == "1"
    # The capsule: marker, transport, the resume binding and the completed stages only.
    with zipfile.ZipFile(capsule) as zipped:
        names = set(zipped.namelist())
        transport = json.loads(zipped.read("cpu_prestage_transport.json"))
    assert "stages/astra_same_run_resume_binding.json" in names
    assert "stages/stage-4/adapter/artifact.usda" in names and not any(n.startswith("stages/stage-5/") for n in names)
    assert transport["output_root"] == str(work / BUNDLE / "runtime_output") and transport["run_id"] == RUN_ID
    assert transport["gpu_execution_performed"] is False and transport["new_paid_allocation_authorized"] is False
    # The work dir is returned to the paid run's initial state; the log stays with the job.
    assert sorted(p.name for p in work.iterdir()) == [".cpu-prestage.lock"]
    assert (job / "cpu_prestage_entrypoint.log").read_bytes() == b"entrypoint ran\n"
    with zipfile.ZipFile(job / "cpu_prestage_output.zip") as archived:
        assert "stages/stage-3/official_openai_cost/reservation.json" in archived.namelist()
    # Idempotent: the retained receipt is returned without a second execution.
    again, _ = _prepare(tmp_path, lambda *a, **k: pytest.fail("prefix executed twice"))
    assert again == receipt


def test_prefix_drops_host_raw_credentials_and_preserves_scoped_secret_files(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
        _RAW_SECRET_ENVIRONMENT_NAMES,
    )

    environment = {name: "host-only-secret" for name in _RAW_SECRET_ENVIRONMENT_NAMES}
    environment.update({
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(tmp_path / "scoped-key"),
        "OPENAI_ADMIN_API_KEY_FILE": str(tmp_path / "admin-key"),
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "3",
    })
    seen = {}
    entrypoint = _fake_entrypoint(["stage-1", "stage-2", "stage-3", "stage-4"], seen=seen)

    def checked_entrypoint(*args, **kwargs):
        assert not set(_RAW_SECRET_ENVIRONMENT_NAMES).intersection(kwargs["env"])
        return entrypoint(*args, **kwargs)

    secrets = {name: value for name, value in environment.items() if name.endswith("_FILE")}
    for value in secrets.values():
        Path(value).write_text("scoped-test-key")
        Path(value).chmod(0o600)
    _prepare(tmp_path, checked_entrypoint, environment=environment, secret_file_paths=secrets)
    assert seen["OPENAI_CONTENT_AGENTS_API_KEY_FILE"] == environment["OPENAI_CONTENT_AGENTS_API_KEY_FILE"]
    assert seen["OPENAI_ADMIN_API_KEY_FILE"] == environment["OPENAI_ADMIN_API_KEY_FILE"]
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"] == "3"
    assert environment["OPENAI_API_KEY"] == "host-only-secret"


def test_an_entrypoint_that_did_not_complete_the_prefix_yields_no_capsule_and_clears_the_work_dir(tmp_path):
    with pytest.raises(prestage.CpuPrestageError, match="cpu_prestage_prefix_not_completed:blocked"):
        _prepare(tmp_path, _fake_entrypoint(["stage-1"], status="blocked"))
    assert not (tmp_path / "job/cpu_prestage_capsule.zip").exists()
    assert not (tmp_path / "job/cpu_prestage_receipt.json").exists()
    assert (tmp_path / "job/cpu_prestage_provider_result.json").is_file()
    with zipfile.ZipFile(tmp_path / "job/cpu_prestage_output.zip") as archived:
        assert "stages/stage-3/official_openai_cost/reservation.json" in archived.namelist()
    assert sorted(p.name for p in (tmp_path / "workspace").iterdir()) == [".cpu-prestage.lock"]
    with pytest.raises(prestage.CpuPrestageError, match="prior_attempt_requires_reconciliation"):
        _prepare(tmp_path, lambda *a, **k: pytest.fail("failed paid attempt repeated"))


def test_prefix_cannot_extend_the_consumed_parent_authority_deadline(tmp_path):
    seen = {}
    receipt, _ = _prepare(tmp_path, _fake_entrypoint(["stage-1", "stage-2", "stage-3", "stage-4"], seen=seen),
                          environment={"BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH": "9000"})
    assert receipt["prefix_deadline_epoch"] == 9000
    assert seen["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"] == "9000.0"


def test_the_prestage_refuses_a_missing_work_dir_or_a_secret_inside_it(tmp_path):
    with pytest.raises(prestage.CpuPrestageError, match="work_dir_unavailable"):
        _prepare(tmp_path, _fake_entrypoint(["stage-1"]), work_dir=tmp_path / "missing")
    with pytest.raises(prestage.CpuPrestageError, match="secret_inside_work_dir"):
        _prepare(tmp_path, _fake_entrypoint(["stage-1"]),
                 secret_file_paths={"OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(tmp_path / "workspace/key")})


def test_a_limit_that_reaches_a_gpu_stage_or_a_non_resumable_backend_is_refused(tmp_path, monkeypatch):
    stages = [{"stage_id": f"stage-{i}", "adapter": {"id": adapter}} for i, adapter in enumerate((
        "website_prepared_appearance", "website_prepared_collision", "content_agents_rigid_replacement",
        "simready_static_rigid_qualification", "simready_native_import_qualification", "native_task_scene_assembly"), 1)]
    from blueprint_pipeline import task_evaluation_scene_configuration_bundle as bundle
    monkeypatch.setattr(bundle, "portable_construction_envelope", lambda _r: {"recipe": {"stage_sequence": stages}})
    receipt = _bundle(tmp_path)
    assert prestage.prestage_stage_limit(receipt, {}) is None
    assert prestage.prestage_stage_limit(receipt, {prestage.STAGE_LIMIT_ENV: "stage-4"}) == "stage-4"
    with pytest.raises(prestage.CpuPrestageError, match="stage_limit_includes_gpu_stage"):
        prestage.prestage_stage_limit(receipt, {prestage.STAGE_LIMIT_ENV: "stage-5"})
    with pytest.raises(prestage.CpuPrestageError, match="stage_limit_invalid"):
        prestage.prestage_stage_limit(receipt, {prestage.STAGE_LIMIT_ENV: "stage-9"})
    with pytest.raises(prestage.CpuPrestageError, match="carried_prefix_conflict"):
        prestage.prestage_stage_limit({**receipt, "carried_completed_stage_count": 2},
                                      {prestage.STAGE_LIMIT_ENV: "stage-4"})
    other = _bundle(tmp_path / "other", backend="content_agents") if (tmp_path / "other").mkdir() is None else None
    with pytest.raises(prestage.CpuPrestageError, match="backend_not_resumable"):
        prestage.prestage_stage_limit(other, {prestage.STAGE_LIMIT_ENV: "stage-4"})
    # The prefix deadline funds only the scheduled stages: authoring, no Isaac allowance.
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
        GPU_STAGE_TIMEOUT_SECONDS, OUTPUT_AND_CLOSURE_RESERVE_SECONDS,
    )
    assert prestage.prestage_ttl_seconds(receipt, "stage-4") == (
        GPU_STAGE_TIMEOUT_SECONDS["content_agents_rigid_replacement"] + OUTPUT_AND_CLOSURE_RESERVE_SECONDS
        + prestage.DEFAULT_CLOSURE_RESERVE_SECONDS + prestage.PRESTAGE_START_MARGIN_SECONDS)
    assert prestage.prestage_ttl_seconds(receipt, "stage-4") <= prestage.DEFAULT_TTL_SECONDS


def _capsule(path: Path, stage_ids: list[str], *, output_root: Path, extra: str | None = None,
             status: str = "completed_prefix_only", transport_overrides: dict | None = None) -> None:
    transport = {"schema_version": prestage.TRANSPORT_SCHEMA, "output_root": str(output_root), "run_id": RUN_ID,
                 "completed_stage_ids": stage_ids, "gpu_execution_performed": False,
                 "new_paid_allocation_authorized": False, **(transport_overrides or {})}
    transport["transport_digest"] = canonical_digest(transport, digest_field="transport_digest")
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("provider_output_zip_exclusions.json", "{}")
        zipped.writestr("completed_stage_checkpoint.json", json.dumps({
            "schema_version": prestage.MARKER_SCHEMA, "status": status, "completed_stage_ids": stage_ids,
            "whole_run_completed": False, "qualification_authority_granted": False}))
        zipped.writestr("cpu_prestage_transport.json", canonical_json(transport))
        zipped.writestr("stages/astra_same_run_resume_binding.json", "{}")
        for stage_id in stage_ids:
            zipped.writestr(f"stages/{stage_id}/completed_stage_checkpoint.json", "{}")
            zipped.writestr(f"stages/{stage_id}/producer/artifact.usda", "#usda 1.0\n")
        if extra:
            zipped.writestr(extra, "x")


def _consume_environment(capsule: Path) -> dict:
    payload = capsule.read_bytes()
    return {prestage.PREFIX_URL_ENV: "https://store.example/capsule.zip",
            prestage.PREFIX_SHA_ENV: prestage._sha(capsule), prestage.PREFIX_BYTES_ENV: str(len(payload))}


@pytest.mark.parametrize("fault", ["digest", "bytes", "member", "marker", "recursion", "occupied", "path", "run"])
def test_the_runtime_restores_only_a_verified_prefix_bound_to_its_own_run_and_paths(tmp_path: Path, fault: str):
    output = (tmp_path / "runtime_output").resolve()
    (output / "stages").mkdir(parents=True)
    capsule = tmp_path / "capsule.zip"
    _capsule(capsule, ["stage-1", "stage-2"], output_root=tmp_path / "elsewhere" if fault == "path" else output,
             extra="stages/stage-9/adapter/x.json" if fault == "member" else None,
             status="whole_run" if fault == "marker" else "completed_prefix_only",
             transport_overrides={"run_id": "other-run"} if fault == "run" else None)
    payload = capsule.read_bytes()
    environment = _consume_environment(capsule)
    if fault == "digest":
        environment[prestage.PREFIX_SHA_ENV] = "sha256:" + "0" * 64
    if fault == "bytes":
        environment[prestage.PREFIX_BYTES_ENV] = str(len(payload) - 1)
    if fault == "recursion":
        environment["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"] = "stage-4"
    if fault == "occupied":
        (output / "stages/stage-1").mkdir()
    with pytest.raises(prestage.CpuPrestageError):
        prestage.consume_stage_prefix_capsule(environment=environment, output_root=output, expected_run_id=RUN_ID,
                                              opener=lambda url, timeout: _Response(payload))
    assert not (output / "stages/stage-2").exists() and not (output / "stages/stage-9").exists()
    assert not (output / "cpu_prestage_restore.json").exists()


def test_the_runtime_restores_the_prefix_at_its_exact_paths(tmp_path: Path, capsys) -> None:
    output = (tmp_path / "runtime_output").resolve()
    output.mkdir()
    capsule = tmp_path / "capsule.zip"
    _capsule(capsule, ["stage-1", "stage-2", "stage-3", "stage-4"], output_root=output)
    payload = capsule.read_bytes()
    value = prestage.consume_stage_prefix_capsule(environment=_consume_environment(capsule), output_root=output,
                                                  expected_run_id=RUN_ID,
                                                  opener=lambda url, timeout: _Response(payload))
    assert value["completed_stage_ids"] == ["stage-1", "stage-2", "stage-3", "stage-4"]
    assert value["restored_member_count"] == 9
    assert (output / "stages/stage-4/producer/artifact.usda").read_text() == "#usda 1.0\n"
    assert (output / "stages/astra_same_run_resume_binding.json").is_file()
    assert not (output / "completed_stage_checkpoint.json").exists()
    assert not (output / "cpu_prestage_transport.json").exists()
    assert json.loads((output / "cpu_prestage_restore.json").read_text())["restore_digest"] == value["restore_digest"]
    assert "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_RESTORED" in capsys.readouterr().out
    assert prestage.consume_stage_prefix_capsule(environment={}, output_root=output, expected_run_id=RUN_ID) is None


def test_rehearsal_a_real_prefix_sealed_on_the_host_is_adopted_by_the_paid_chain_at_the_same_paths(tmp_path, capsys):
    """End to end with the real stage chain, archive writer and resume binding."""
    from types import SimpleNamespace
    from blueprint_pipeline.task_evaluation_scene_configuration_provider_runtime import (
        execute_scene_configuration_stage_chain,
    )
    from tests.test_task_evaluation_scene_configuration_provider_runtime import _astra_inputs, _producers, _registry

    envelope, configurations = _astra_inputs(tmp_path)
    envelope["expected_production_commit"] = COMMIT
    observed_host: list[str] = []
    produced: list[str] = []
    producer = _producers()

    def produce(**kwargs):
        produced.append(kwargs["stage"]["stage_id"])
        return producer.execute(**kwargs)

    def entrypoint(argv, *, cwd, env, stdout, stderr, check):
        output = Path(env["BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_ROOT"])
        stages = output / "stages"
        stages.mkdir()
        checkpoint = Path(env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_CHECKPOINT_PATH"])
        chain = execute_scene_configuration_stage_chain(
            envelope=envelope, configurations=configurations, output_root=stages,
            registry=_registry(observed_host, real_artifacts=True),
            producer_registry=SimpleNamespace(execute=produce),
            stage_limit=env["BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT"],
            parent_deadline_epoch=float(env["BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH"]),
            checkpoint_callback=lambda results: preserve_stage_prefix(
                output_root=output, completed_results=results, checkpoint_path=checkpoint))
        _write_result(env, status=chain["status"], stage_ids=[r["stage_id"] for r in chain["stage_results"]])
        return type("Completed", (), {"returncode": 0})()

    # The fixture recipe is the canonical ArtiFixer chain (stage 1 funds 12 000 s
    # of training); the transport under test is recipe-independent.
    receipt, _ = _prepare(tmp_path, entrypoint, now=time.time, ttl_seconds=7 * 3600)
    assert observed_host == ["stage-1", "stage-2", "stage-3", "stage-4"] and produced == ["stage-1", "stage-3"]
    work = tmp_path / "workspace"
    output = work / BUNDLE / "runtime_output"
    assert not output.exists()
    # The paid run: fresh container, same paths, capsule restored, chain continues from stage 5.
    output.mkdir(parents=True)
    capsule = Path(receipt["capsule_path"])
    payload = capsule.read_bytes()
    restored = prestage.consume_stage_prefix_capsule(
        environment={prestage.PREFIX_URL_ENV: "https://store.example/c.zip", prestage.PREFIX_SHA_ENV: receipt["capsule_sha256"],
                     prestage.PREFIX_BYTES_ENV: str(receipt["capsule_bytes"])},
        output_root=output.resolve(), expected_run_id=RUN_ID, opener=lambda url, timeout: _Response(payload))
    assert restored["completed_stage_ids"] == ["stage-1", "stage-2", "stage-3", "stage-4"]
    from blueprint_pipeline.task_evaluation_astra_stage_resume import completed_astra_prefix
    native_deadline = time.time() + 4 * 3600
    assert completed_astra_prefix(output.resolve() / "stages", envelope, configurations, native_deadline) is True
    observed_paid = ["stage-1", "stage-2", "stage-3", "stage-4"]
    capsys.readouterr()
    full = execute_scene_configuration_stage_chain(
        envelope=envelope, configurations=configurations, output_root=output.resolve() / "stages",
        registry=_registry(observed_paid, real_artifacts=True), producer_registry=SimpleNamespace(execute=produce),
        parent_deadline_epoch=native_deadline)
    assert full["status"] == "completed" and full["whole_run_completed"] is True
    assert observed_paid == [f"stage-{i}" for i in range(1, 7)] and produced == ["stage-1", "stage-3", "stage-5"]
    assert capsys.readouterr().out.count("BLUEPRINT_SCENE_CONFIGURATION_STAGE_ADOPTED") == 4
    assert (output.resolve() / "stages/astra_prefix_continuation_binding.json").is_file()


def test_ambient_credentials_beyond_the_named_list_are_dropped_but_file_references_survive():
    """The producer's named list is the floor, not the whole rule.

    The paid container is handed scoped secret *files*. Any host variable
    whose name is credential-shaped and whose value is a literal rather than
    a path would reach the stage as a raw credential, so it is dropped even
    though the producer does not name it.
    """
    values, dropped = prestage._provider_environment(
        {
            "HUGGINGFACE_TOKEN": "hf_literal",
            "AWS_SECRET_ACCESS_KEY": "literal",
            "BLUEPRINT_REGISTRY_PASSWORD": "literal",
            "OPENAI_CONTENT_AGENTS_API_KEY_FILE": "/run/secrets/openai",
            "OPENAI_CONTENT_AGENTS_API_KEY_ID": "key_abc123",
            "OPENAI_API_KEY_PATH": "/run/secrets/openai",
            "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "3",
            "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST": "sha256:" + "a" * 64,
            "EMPTY_TOKEN": "",
        },
        {"PATH": "/usr/bin"},
    )
    assert dropped == ["AWS_SECRET_ACCESS_KEY", "BLUEPRINT_REGISTRY_PASSWORD", "HUGGINGFACE_TOKEN"]
    # An identifier is a reference, not a credential.
    assert values["OPENAI_CONTENT_AGENTS_API_KEY_ID"] == "key_abc123"
    # File references and ordinary settings cross unchanged.
    assert values["OPENAI_CONTENT_AGENTS_API_KEY_FILE"] == "/run/secrets/openai"
    assert values["OPENAI_API_KEY_PATH"] == "/run/secrets/openai"
    assert values["BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD"] == "3"
    assert values["BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST"].startswith("sha256:")
    assert values["PATH"] == "/usr/bin" and values["EMPTY_TOKEN"] == ""
    assert "hf_literal" not in json.dumps(values)


def test_every_name_the_producer_forbids_is_refused_however_it_arrives():
    """Derived from the producer's own tuple, so the two cannot drift.

    A name added to the producer's list is covered here without editing this
    test -- including when it arrives as a caller override rather than from
    the ambient host environment.
    """
    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import (
        _RAW_SECRET_ENVIRONMENT_NAMES,
    )

    values, dropped = prestage._provider_environment(
        {name: "literal-credential" for name in _RAW_SECRET_ENVIRONMENT_NAMES}, {})
    assert sorted(dropped) == sorted(_RAW_SECRET_ENVIRONMENT_NAMES) and values == {}
    for name in _RAW_SECRET_ENVIRONMENT_NAMES:
        values, _ = prestage._provider_environment({}, {name: "literal-credential"})
        assert name not in values


def test_an_unusable_environment_refuses_before_the_bundle_is_unpacked(tmp_path):
    """Refuse in milliseconds, not after stages 1-2 have run.

    The first live prestage spent the bundle unpack and two stages before the
    stage-3 producer rejected its environment. Everything that can be judged
    from the inputs alone is now judged before the work dir is touched.
    """
    work = tmp_path / "workspace"
    with pytest.raises(prestage.CpuPrestageError, match="parent_deadline_exhausted"):
        _prepare(tmp_path, lambda *a, **k: pytest.fail("entrypoint ran"),
                 environment={"BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH": "1001"})
    # Not even the work-dir lock is taken: nothing about the host changed.
    assert list(work.iterdir()) == []


def test_the_receipt_records_which_ambient_names_were_dropped_and_never_their_values(tmp_path):
    receipt, _ = _prepare(tmp_path, _fake_entrypoint(["stage-1", "stage-2", "stage-3", "stage-4"]),
                          environment={"OPENAI_API_KEY": "sk-should-never-appear",
                                       "OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(tmp_path / "key")})
    assert "OPENAI_API_KEY" in receipt["dropped_ambient_environment_names"]
    assert "sk-should-never-appear" not in json.dumps(receipt)
    assert "sk-should-never-appear" not in (tmp_path / "job/cpu_prestage_entrypoint.log").read_text()


def test_every_name_the_openai_stage_gate_requires_survives_composition():
    """Derived from the gate's own scope table, so the rule cannot over-drop.

    Stage 3's gate binds an exclusive key file *and* its key id (the
    attestation scope). A credential-shaped filter that swallowed the id
    would break authoring in a new way, visible only in production.
    """
    from blueprint_pipeline.task_evaluation_scene_configuration_openai_gate import _STAGE_SCOPE_ENV

    required = {name: ("/run/secrets/k" if name.endswith("_FILE") else "key_abc123")
                for scope in _STAGE_SCOPE_ENV.values() for name in scope.values()}
    values, dropped = prestage._provider_environment(required, {})
    assert dropped == [] and values == required


def test_only_attempt_staged_files_cross_the_cpu_boundary(tmp_path):
    from blueprint_pipeline.task_evaluation_scene_configuration_builtin_producers import _secret_values

    seen = {}
    fake = _fake_entrypoint(["stage-1", "stage-2", "stage-3", "stage-4"], seen=seen)
    environment = {
        "OPENAI_API_KEY_FILE": "/unreadable/host-default",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE": "/unreadable/inactive-stage",
        "OPENAI_CONTENT_AGENTS_API_KEY_FILE": "/unreadable/ambient-active-alias",
        "OPENAI_CONTENT_AGENTS_API_KEY_ID": "key_scope_identity",
        "PIPELINE_SYNC_TOKEN_FILE": "/unreadable/unrelated-token",
    }

    def entrypoint(*args, **kwargs):
        assert _secret_values(kwargs["env"]) == ["scoped-test-key"]
        return fake(*args, **kwargs)

    _prepare(tmp_path, entrypoint, environment=environment)
    assert seen["OPENAI_CONTENT_AGENTS_API_KEY_FILE"] == str(tmp_path / "key")
    assert seen["OPENAI_CONTENT_AGENTS_API_KEY_ID"] == "key_scope_identity"
    assert not {"OPENAI_API_KEY_FILE", "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_FILE",
                "PIPELINE_SYNC_TOKEN_FILE"}.intersection(seen)


@pytest.mark.parametrize("kind", ["missing", "empty", "shared", "symlink"])
def test_invalid_staged_key_refuses_before_unpack_or_any_stage(tmp_path, kind):
    key = tmp_path / "invalid-key"
    if kind != "missing":
        key.write_text("" if kind == "empty" else "scoped-test-key")
        key.chmod(0o640 if kind == "shared" else 0o600)
    if kind == "symlink":
        link = tmp_path / "key-link"
        link.symlink_to(key)
        key = link
    with pytest.raises(RuntimeError, match="scene_configuration_secret_file_invalid"):
        _prepare(tmp_path, lambda *a, **k: pytest.fail("stage entered"),
                 secret_file_paths={"OPENAI_CONTENT_AGENTS_API_KEY_FILE": str(key)})
    assert list((tmp_path / "workspace").iterdir()) == []
