"""The steps of one test job variation, built from the repository's own test fixtures (synthetic, $0, no network).

Used by ``blueprint_pipeline.test_job_matrix``; not collected by pytest (no ``test_`` prefix). Each step runs the
real code under test and returns a row: ``status`` (``passed``, ``stopped``, ``not_applicable`` or
``not_run_here``), its ``stop`` when it stopped, ``fed_from`` (what it starts from), and what it produced. Nothing
here is a pass the code did not give; a step that cannot run on this job's own inputs says ``not_run_here``.

Fixtures reused, not copied (a re-base only has to keep these names):

- the website recipe: ``tests/test_website_native_inputs.packet`` (the real ``prepare_construction_stages``);
- the scene chain: ``tests/test_scene_configuration_whole_chain_on_host`` ``_host_lane`` (the real lane with its
  doubles for stages 1-4), run on that recipe's stages;
- the website half: ``tests/test_team_scene_packet._scene``, ``tests/test_team_scene_packet_owner_confirmation._job``
  and the package's own helpers in ``tests/test_website_scene_package_robot_absent`` (``_not_run``,
  ``_stance_raises``, ``_packet_raises``, ``_sealed_doc``);
- the zip: ``tests/test_website_company_simulation`` ``Job``, ``World``, ``_package``;
- the room package (navigation, an unknown kind): ``tests/test_website_scene_handoff_navigation.room_job``, the real
  handoff on real fixture geometry onto that ``Job``'s capture.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

STEPS = ("routing", "front_half", "scene_chain", "publication", "completion_enqueue", "website_compile",
         "company_zip")
STOPPING = {"refusal", "missing_input", "error", "would_buy", "would_rent_gpu", "replay_mismatch"}


def _row(status: str, *, stop: Mapping[str, Any] | None = None, fed_from: str = "", **produced: Any) -> dict:
    return {"status": status, "stop": dict(stop) if stop else None, "fed_from": fed_from, **produced}


def _stop(kind: str, code: str, **detail: Any) -> dict[str, Any]:
    return {"kind": kind, "code": code, **detail}


def routing(row: Mapping[str, Any]) -> dict[str, Any]:
    """The native route the job's reading picks (``website_task_semantics._operation_profile``)."""
    from blueprint_pipeline.task_profile_contracts import task_profile_for
    from blueprint_pipeline.website_task_semantics import _operation_profile

    reading, feed = row["reading"], "fixture:matrix row reading (the Atlas review that writes it is a paid model call)"
    profile_id = _operation_profile(reading["task_family"], reading["steps"])
    if profile_id is None:
        # No task profile: not a stop since 2026-11-16, the handoff ships the room package (``write_room_package``).
        return _row("passed", fed_from=feed, profile_id=None, route="room_package")
    return _row("passed", fed_from=feed, profile_id=profile_id,
                native_task_kind=task_profile_for(profile_id).native_task_kind)


def replayed(records: list[str], results: Mapping[str, Any], *, missing: str) -> dict[str, Any]:
    """Rows of the replay set (``blueprint_pipeline.replay_set``), run once before the variations; read only."""
    if not records:
        return _row("not_run_here", reason=missing)
    rows = [results[record] for record in records]
    feed = "fixture:replay_set_v1 " + ",".join(records) + " (synthetic records with their own retained inputs)"
    failed = [r for r in rows if not r["passed"]]
    if failed:
        return _row("stopped", stop=_stop("replay_mismatch", f"{failed[0]['record_id']}:{failed[0]['actual']}"),
                    fed_from=feed, records=rows)
    stops = [r for r in rows if r["actual"].get("kind") in STOPPING]
    if stops:
        actual = stops[0]["actual"]
        return _row("stopped", stop=_stop(actual["kind"], str(actual["code"]), record=stops[0]["record_id"]),
                    fed_from=feed, records=rows)
    return _row("passed", fed_from=feed, records=rows,
                flags=[r["actual"]["code"] for r in rows if r["actual"].get("kind") == "estimate"])


# ----------------------------------------------------------------------------------------- scene chain

def _articulated(patches: pytest.MonkeyPatch) -> None:
    """Stage 3 says it authors an articulated object (no articulated website fixture exists; this stand-in lacks
    the mechanism a real one carries); the recipe is then the real ``prepare_construction_stages`` reading of it."""
    from blueprint_pipeline import task_evaluation_scene_configuration_submission_records as records
    from blueprint_pipeline import website_native_background as background

    derived = background.derived_stage_three_configuration
    patches.setattr(background, "derived_stage_three_configuration", lambda **kw: {
        **derived(**kw), "schema_version": records.ARTICULATED_AUTHORING_SCHEMA_VERSION,
        "mechanism": {"estimated_front_normal_world": [1.0, 0.0, 0.0]}})


def scene_chain(row: Mapping[str, Any], work: Path, patches: pytest.MonkeyPatch) -> dict[str, Any]:
    """The real website recipe (``prepare_construction_stages``), its real stage validation and host plan, then the
    real lane on exactly those stages."""
    from blueprint_pipeline.task_evaluation_scene_configuration_cpu_prestage import host_chain_plan
    from blueprint_pipeline.task_evaluation_scene_configuration_stage_configuration import (
        TaskEvaluationSceneConfigurationStageConfigurationError, validate_immutable_stage_configurations,
    )
    from tests.test_website_native_inputs import packet

    mode = row["scene"]
    if mode == "none":
        return _row("not_applicable", reason="a room package has no subject of its own: no scene build")
    if mode == "isaac":
        return _row("not_run_here", reason="an Isaac stage 5 runs only on a rented GPU")
    feed = "fixture:tests/test_website_native_inputs.packet (the real prepare_construction_stages)"
    if mode == "articulated":
        _articulated(patches)
        feed += ", its stage 3 a stand-in articulated record (so stage validation is not run on it)"
    envelope, configs = packet(work / "packet")
    if mode == "destination":
        envelope["recipe"]["supplemental_destination"] = {"identity": {"id": "tray", "version": "v1"}}
    recipe = [s["adapter"]["id"] for s in envelope["recipe"]["stage_sequence"]]
    try:
        if mode != "articulated":
            validate_immutable_stage_configurations(envelope=envelope, configurations=configs)
    except TaskEvaluationSceneConfigurationStageConfigurationError as exc:
        return _row("stopped", stop=_stop("refusal", str(exc)[:200]), fed_from=feed, recipe=recipe)
    on_host, _limit = host_chain_plan(envelope, {"carried_completed_stage_count": 0}, "stage-4",
                                      diagnostic_only=False)
    if not on_host:
        return _row("stopped", stop=_stop("would_rent_gpu", "host_chain_plan:false:stage-5=" + recipe[4]),
                    fed_from=feed, recipe=recipe)
    from tests import test_scene_configuration_whole_chain_on_host as whole_chain
    from blueprint_pipeline.website_scene_without_simulator import stage_five_receipt_kind

    stages = [{key: s[key] for key in ("stage_id", "capability", "adapter", "execution_class", "depends_on")}
              for s in envelope["recipe"]["stage_sequence"]]
    patches.setattr(whole_chain, "HOST_ADAPTERS", tuple(recipe))  # the lane runs exactly this recipe
    patches.setattr(whole_chain, "_host_stages", lambda: [dict(s) for s in stages])
    lane = whole_chain._host_lane(work / "lane", patches)
    result = lane.run(cpu_prestage_stage_limit="stage-4")
    gpu = [name for name, _ in lane.events if name in {"adapter", "stage"}]
    produced = {"recipe": recipe, "cpu_prestage": (result.get("cpu_prestage") or {}).get("status"),
                "gpu_events": gpu}
    if result["status"] != "completed":
        return _row("stopped", stop=_stop("refusal", str((result.get("blockers") or ["blocked"])[0])[:200]),
                    fed_from=feed, **produced)
    if gpu:
        return _row("stopped", stop=_stop("would_rent_gpu", "lane_events:" + ",".join(gpu)), fed_from=feed,
                    **produced)
    execution = json.loads((lane.job / "immutable_execution" / whole_chain.RESULT_NAME).read_text())
    by_role = {r["role"]: r for stage in execution["stage_chain"]["stage_results"] for r in stage["output_artifacts"]}
    receipt = json.loads((lane.job / "immutable_execution" / by_role["native_import_qualification_receipt"][
        "provider_output_relative_path"]).read_text())
    revision = json.loads(Path(result["configured_scene_revision_path"]).read_text())
    return _row("passed", fed_from=feed, **produced, stage_5_kind=stage_five_receipt_kind(receipt),
                published=result.get("configured_scene_published"),
                physics_authority=revision["replacement"].get("physics_authority"))


def publication(chain: Mapping[str, Any]) -> dict[str, Any]:
    if chain["status"] != "passed":
        return _row("not_applicable", reason="the scene chain published nothing here")
    if chain.get("published") is not True:
        return _row("stopped", stop=_stop("refusal", "configured_scene_not_published"), fed_from="scene_chain")
    return _row("passed", fed_from="scene_chain: the lane's configured revision",
                physics_authority=chain["physics_authority"])


# -------------------------------------------------------------------------------------- website half

def _restage_five(patches: pytest.MonkeyPatch, change: Mapping[str, Any]) -> None:
    """The fixture revision (as already patched) with its stage-5 record changed by ``change`` and sealed again."""
    from tests import test_team_scene_packet as packet_tests
    from tests.test_task_evaluation_rigid_relocation_native_adapter import NATIVE_IMPORT, _rewrite
    from tests.test_website_scene_package_robot_absent import _sealed_doc

    original = packet_tests.rigid_case

    def rigid_case(tmp_path: Path, **options: Any):
        configured, references, docs, spec = original(tmp_path, **options)
        document = _sealed_doc({**docs[NATIVE_IMPORT], **change})
        _rewrite(tmp_path=tmp_path, configured=configured, references=references, contract_path=NATIVE_IMPORT,
                 document=document)
        return configured, references, {**docs, NATIVE_IMPORT: document}, spec

    patches.setattr(packet_tests, "rigid_case", rigid_case)


def _case(case: str, patches: pytest.MonkeyPatch) -> None:
    from tests import test_website_scene_package_robot_absent as package_tests

    if case == "no_stance":
        from blueprint_pipeline.team_embodiment import TeamEmbodimentError
        package_tests._stance_raises(patches, TeamEmbodimentError(["team_embodiment_task_motion_no_clear_stance"]))
    elif case == "quality_gate":
        from blueprint_pipeline import native_task_arena_packet as arena
        from blueprint_pipeline import team_scene_robot_absent as absent
        package_tests._packet_raises(patches, arena.NativeTaskArenaPacketError([absent.APPEARANCE_QUALITY]))


def _recorded(compiled: Mapping[str, Any], work: Path, patches: pytest.MonkeyPatch) -> dict[str, Any]:
    """The real ``record_team_scene`` on the compiled half, its registry write and its probe enqueue recorded."""
    from blueprint_pipeline import rigid_robot_free_probe_runner as probes
    from blueprint_pipeline import team_candidate_quick10_registry as registry
    from blueprint_pipeline import team_scene_packet as tsp

    enqueued: list[Any] = []
    patches.setattr(registry, "write_derived_profile", lambda row: {"status": "written"})
    patches.setattr(probes, "enqueue_rigid_robot_free_probe",
                    lambda root: enqueued.append(str(root)) or {"status": "queued", "queued": True})
    capture = work / "capture"
    capture.mkdir(parents=True)
    (capture / "capture_descriptor.json").write_text(json.dumps({"scene_id": "matrix-site"}))
    recorded = tsp.record_team_scene(compiled, run_root=work / "launch", capture_root=capture,
                                     scenario_id="nominal", scene_revision_digest="sha256:" + "1" * 64)
    return {"status": recorded["robot_free_probe"]["status"], "enqueued": enqueued}


def website_compile(row: Mapping[str, Any], work: Path, patches: pytest.MonkeyPatch) -> dict[str, Any]:
    """The website worker's real compile (``WEBSITE_COMPILE``) of the fixture revision named by the row."""
    from blueprint_pipeline import team_scene_packet as tsp
    from blueprint_pipeline import team_scene_packet_worker as worker
    from blueprint_pipeline import team_scene_robot_absent as absent
    from blueprint_pipeline import website_company_simulation as sim
    from blueprint_pipeline import website_owner_success_assumption as assumption
    from tests import test_team_scene_packet as packet_tests
    from tests import test_website_scene_package_robot_absent as package_tests

    spec = row["compile"]
    if spec.get("handoff") == "room_package":
        return room_handoff(row, work, patches)
    stage_5 = spec["stage_5"]
    articulated = spec["strategy"] == "articulated_open_close"
    if articulated:
        feed = ("fixture:tests/test_team_scene_packet._articulated_scene (the drawer cabinet) with the not-run "
                "record bound to its own published object")
    elif stage_5 in ("not_run", "not_run_static_passed", "not_run_claiming_isaac"):
        package_tests._not_run(patches, static_passed=stage_5 == "not_run_static_passed")
        feed = ("fixture:rigid revision with the no-GPU chain's not-run record bound to its own stage-4 record "
                "(the chain's record above names another object, so it feeds nothing here)")
        if stage_5 == "not_run_claiming_isaac":
            _restage_five(patches, {"native_isaac_executed": True})
            feed += ", resealed claiming Isaac ran"
    elif stage_5 == "isaac_qualified":
        # The fixture's own stage-5 record is the Isaac import result, passed; it lacks only the flag that Isaac ran.
        _restage_five(patches, {"native_isaac_executed": True})
        feed = "fixture:rigid revision whose own passed Isaac record also says Isaac ran"
    else:
        feed = "fixture:the revision as its test fixture publishes it"
    _case(spec.get("case", "default"), patches)
    inputs_dir = work / "compile-inputs"
    inputs_dir.mkdir(parents=True)
    if spec.get("case") == "owner":
        from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
        from tests.test_team_scene_packet_owner_confirmation import _job

        job, base = _job(inputs_dir, patches)
        documents = rigid.fetch_revision_documents(revision=job.revision, output_root=inputs_dir / "fetched",
                                                   fetcher=lambda reference: job.store[reference["uri"]])
        inputs = {**base, "revision": job.revision, "documents": documents,
                  "owner_success_confirmation_scope": {"intent": job.intent, "directory": Path(job.directory)}}
        feed = "fixture:the proposal_only job of tests/test_team_scene_packet_owner_confirmation (thresholds unconfirmed)"
    else:
        scene = (packet_tests._articulated_scene(inputs_dir, not_run=stage_5 == "not_run") if articulated
                 else packet_tests._scene(inputs_dir, strategy=spec["strategy"]))
        inputs = {k: v for k, v in scene.items() if k not in ("profile", "store", "revision_reference")}
    if spec.get("rule_withheld"):
        # The handoff's sealed reason, as the worker reads it (``team_scene_packet_worker._rule_withheld``; the
        # handoff's route is tested in tests/test_website_scene_handoff.py).
        inputs["rule_withheld"] = {"code": spec["rule_withheld"]}
        feed += ", compiled with the rule withheld (" + spec["rule_withheld"] + ")"
    try:
        compiled = tsp.materialize_team_scene(**inputs, output_root=work / "half" / "scene",
                                              **dict(worker.WEBSITE_COMPILE))
    except Exception as exc:  # noqa: BLE001 - the compile's own refusal is the result
        codes = getattr(exc, "errors", None) or getattr(exc, "blockers", None) or [str(exc)[:200]]
        return _row("stopped", stop=_stop("refusal", str(list(codes)[0])[:200]), fed_from=feed)
    probe = _recorded(compiled, work, patches)
    record = json.loads(Path(compiled["path"]).read_text())
    source = sim._half(Path(compiled["path"]).parent, {"team_scene_inputs_path": compiled["path"]})
    if probe["enqueued"]:
        return _row("stopped", stop=_stop("would_buy", "record_team_scene_queued_the_robot_free_probe"),
                    fed_from=feed)
    return _row("passed", fed_from=feed, half_record_path=compiled["path"],
                robot_station_absent=(record.get(absent.FIELD) or {}).get("reason"),
                robot_free_probe=probe["status"],
                assumed_thresholds=bool(record.get(assumption.FIELD)),
                physics_check=(source or {}).get("physics_check"))


def room_handoff(row: Mapping[str, Any], work: Path, patches: pytest.MonkeyPatch) -> dict[str, Any]:
    """The real scene handoff of a room package job (``website_scene_handoff.write_room_package``) on the real
    fixture geometry, written onto the company builder's own test capture."""
    from tests.test_website_scene_handoff_navigation import room_job

    reading = row["reading"]
    _job, _context, result = room_job(work, patches, family=reading["task_family"],
                                     description=reading["steps"][0]["task_quote"],
                                     not_found=bool(row["compile"].get("nothing_found")))
    feed = ("fixture:tests/test_website_scene_handoff_navigation.room_job (the real handoff on "
            "tests/test_website_task_preparation geometry and a real splat)")
    if result.get("status") != "room_package_written":
        return _row("stopped", stop=_stop("refusal", str((result.get("blockers") or [result.get("status")])[0])),
                    fed_from=feed)
    record = json.loads(Path(result["room_package"]["path"]).read_text())
    return _row("passed", fed_from=feed, half_record_path=result["room_package"]["path"], job_root=str(work),
                task_kind=record["task_kind"], success_rule_defined=record["success_rule_defined"],
                success_rule_assumed=record["success_rule_assumed"], robot_free_probe=record["robot_free_probe"],
                robot_station_absent=None, assumed_thresholds=False, physics_check=None)


def readme_variant(readme: str, physics_check: str | None, sentences: Mapping[str, Any]) -> list[str]:
    """What the README's "How it was made" section says, read exactly: the matrix file's fixed text, then the
    sentences of one physics check, then the notes, in the package's order, and nothing else. The compile's
    ``physics_check`` when its sentences are the ones there, then each note; any other text is named in full. The
    sentences are the matrix file's own literals, never the package's constants, so a changed or added sentence
    fails a row until its expectation is changed on purpose."""
    section = readme.split("\nHow it was made\n", 1)[-1].split("\nTerms\n", 1)[0]
    rest = " ".join(section.split())
    if not rest.startswith(sentences["made"]):
        return [f"unexpected_text:{rest}"]
    rest = rest[len(sentences["made"]):].strip()
    physics = sentences["physics_check"]
    said = max((" ".join(words) for words in physics.values() if rest.startswith(" ".join(words))), key=len)
    rest = rest[len(said):].strip()
    agrees = said == " ".join(physics.get(physics_check or "none", ()))
    variant = [(physics_check or "no_check_sentence") if agrees else f"readme_disagrees_with:{physics_check}"]
    for note, sentence in sentences["notes"].items():
        found = re.match(r"(\d+ )?" + re.escape(sentence), rest)
        if found:
            variant.append(note)
            rest = rest[found.end():].strip()
    return variant + ([f"unexpected_text:{rest}"] if rest else [])


def company_zip(row: Mapping[str, Any], compiled: Mapping[str, Any], work: Path) -> dict[str, Any]:
    from tests.test_website_company_simulation import Job

    if compiled.get("job_root"):
        # A room package: the handoff wrote it on this capture, which no team scene status names.
        job = Job(Path(compiled["job_root"]) / "download")
        for path in (job.queue / "status").glob("*.json"):
            path.unlink()
        return _zipped(row, compiled, job, "website_compile:the room package the handoff wrote")
    job = Job(work / "download")
    status = next((job.queue / "status").glob("*.json"))
    record_path = Path(compiled["half_record_path"])
    feed = "website_compile:the compiled half's record"
    zip_spec = row.get("zip") or {}
    change: dict[str, Any] = {}
    if zip_spec.get("furniture_moved"):
        # The compile's own furniture path needs a measurement and the site owner's sealed approval
        # (tests/test_robot_access_furniture.py).
        change["robot_access_removal"] = {"status": "applied", "station_after": {"stance_digest": "matrix"},
                                          "items": [{"label": f"item {i}"} for i in range(zip_spec["furniture_moved"])]}
        feed += (" + the half's sealed record replaced with one carrying an applied removal (the measurement and "
                 "approval path is not built here)")
    if zip_spec.get("support_held"):
        # This fixture's support is unchecked; the CPU rest check's own reasons (``support_rest_check``) are named.
        change["task_support_pad"] = {"status": "held", "reasons": list(zip_spec["support_held"])}
        feed += " + the half's sealed record replaced with one whose support record is held for " + ",".join(
            zip_spec["support_held"])
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    plan_path = record_path.parent / "native-task-packet" / "native_task_arena_scene_plan.v1.json"
    if zip_spec.get("rule_dropped"):
        # A rigid half with no rule anywhere: its plan and its sealed record carry no success contract.
        plan = json.loads(plan_path.read_text())
        plan["task_spec"].pop("task_success_contract", None)
        plan_path.unlink()
        plan_path.write_text(json.dumps(plan))
        change["task_success_contract"] = None
        feed += " + the half's plan and sealed record without their success contract"
    if zip_spec.get("reason"):
        change["rule_withheld"] = {"reason": zip_spec["reason"]}
        feed += " + a reason no handoff gives (" + zip_spec["reason"] + ")"
    if change:
        value = {**json.loads(record_path.read_text()), **change, "inputs_digest": ""}
        value = {key: item for key, item in value.items() if item is not None or key not in change}
        value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
        # The compile seals its record read-only (0444): replace it as a new sealed file, never write through it
        # (only root could, which is why this passed as root and failed on CI's non-root runner).
        record_path.unlink()
        record_path.write_text(json.dumps(value))
        record_path.chmod(0o444)
    status.write_text(json.dumps({**json.loads(status.read_text()), "team_scene_inputs_path": str(record_path)}))
    return _zipped(row, compiled, job, feed)


def _zipped(row: Mapping[str, Any], compiled: Mapping[str, Any], job: Any, feed: str) -> dict[str, Any]:
    from tests.test_website_company_simulation import World, _package

    world = World(job)
    published = world.publish()
    if published.get("status") != "published":
        return _row("stopped", stop=_stop("refusal", str(published.get("reason") or published.get("status"))),
                    fed_from=feed)
    package = _package(job)
    with zipfile.ZipFile(package) as archive:
        members = [[i.filename, i.file_size] for i in archive.infolist()]
        readme_bytes = archive.read("README.txt")
    readme = readme_bytes.decode()
    return _row("passed", fed_from=feed, zip_built=True,
                zip_sha256="sha256:" + hashlib.sha256(package.read_bytes()).hexdigest(),
                zip_members=members, readme=readme, readme_sha256="sha256:" + hashlib.sha256(readme_bytes).hexdigest(),
                readme_variant=readme_variant(readme, compiled.get("physics_check"), row["readme_sentences"]))


def run_variation(row: Mapping[str, Any], work: Path, replay: Mapping[str, Any]) -> dict[str, Any]:
    """Every step in order; the first stop ends the variation and later steps read ``not_reached``."""
    steps: dict[str, dict[str, Any]] = {}
    seconds: dict[str, float] = {}
    patches = pytest.MonkeyPatch()
    stopped_at = None
    downstream = list(row.get("downstream") or [])
    plan: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("routing", lambda: routing(row)),
        ("front_half", lambda: replayed(list(row.get("front_half") or []), replay,
                                        missing="no replay record for this job kind's front half")),
        ("scene_chain", lambda: scene_chain(row, work / "scene", patches)),
        ("publication", lambda: publication(steps["scene_chain"])),
        ("completion_enqueue", lambda: replayed(downstream, replay, missing=(
            "no completion harness takes this chain's own revision; the no-probe rule is read on the compile")
            if row["scene"] == "host_chain" else "no replay record for this job kind's completion")),
        ("website_compile", lambda: website_compile(row, work / "compile", patches)),
        ("company_zip", lambda: company_zip(row, steps["website_compile"], work / "zip")),
    ]
    try:
        for name, call in plan:
            started = time.monotonic()
            steps[name] = call()
            seconds[name] = round(time.monotonic() - started, 3)
            if steps[name]["status"] == "stopped":
                stopped_at = name
                break
    finally:
        patches.undo()
    for name in STEPS:
        steps.setdefault(name, _row("not_reached"))
    reached = [n for n in STEPS if steps[n]["status"] in {"passed", "stopped"}]
    last = stopped_at or reached[-1]
    stop = steps[last].get("stop") or {"kind": "completed", "code": "zip_published"}
    compiled, zipped = steps["website_compile"], steps["company_zip"]
    return {"label": row["label"], "kind": row["kind"], "last_stage": last, "first_stop": stop,
            "zip_built": bool(zipped.get("zip_built")), "zip_sha256": zipped.get("zip_sha256"),
            "zip_members": zipped.get("zip_members"), "readme_variant": zipped.get("readme_variant"),
            "readme_sha256": zipped.get("readme_sha256"), "readme": zipped.get("readme"),
            "physics_authority": steps["publication"].get("physics_authority"),
            **{field: compiled.get(field) for field in ("physics_check", "robot_station_absent", "robot_free_probe",
                                                        "assumed_thresholds")},
            "steps": steps, "seconds": seconds}
