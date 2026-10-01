"""Joined developmental scene stages for the no-paid Plan 13c harness.

ADP-009D/day 28. Test data is generated afresh at the checked-out source. No
production validator is replaced. Contract-fixture release records are exposed
explicitly; a host acceptance run must supply its actual deployment binding.
"""

from __future__ import annotations

import json
import os
import pwd
import hashlib
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from blueprint_pipeline import task_evaluation_scene_configuration_submission_publication as publication
from blueprint_pipeline import task_evaluation_scene_progression as progression
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_host_input_intake import _verified_checkout_head
from blueprint_pipeline.task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
from scripts.control_plane_concurrency_fixture import FilesystemObjectStore


@contextmanager
def fixture_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def advance_fixture_intake(*, host_root: Path, object_root: Path, source_commit: str,
                           release_binding: dict | None = None) -> dict:
    if _verified_checkout_head() != source_commit:
        raise ValueError("harness_checkout_source_mismatch")
    if host_root.exists() or host_root.is_symlink():
        raise ValueError("harness_scene_root_exists")
    host_root.mkdir(mode=0o700)
    # These are data builders, not test execution or patched production
    # functions. Each harness child owns its module globals and environment.
    from tests import test_task_evaluation_completed_scene_progression as fixture
    from tests import test_task_evaluation_scene_configuration_submission as template

    queue = ensure_launch_preparation_queue_root(host_root / "preparation-queue")
    values = {
        "BLUEPRINT_AGENT_EXECUTION_CONFIG": str(host_root / "no-live-agent-config.json"),
        "BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT": str(host_root / "prepared-references"),
        "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT": str(host_root.parent / "reservations"),
        "BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT": str(host_root.parent / "pins"),
    }

    class FixtureBuilderEnvironment:
        def setenv(self, key: str, value: str) -> None:
            values[key] = value

        def setattr(self, obj, key: str, value) -> None:
            # The old data builder requests a test-only validator patch. Keep
            # the real Git validator and require it to agree with new inputs.
            if key != "_verified_checkout_head" or getattr(obj, key)() != source_commit:
                raise ValueError("harness_production_validator_override_refused")

    old_fixture, old_template = fixture.SHA, template.SHA
    fixture.SHA = template.SHA = source_commit
    try:
        config, intent_id, intents, now = fixture._config(host_root, FixtureBuilderEnvironment(),
            submission_enabled=True, existing_support=True,
            extra={"preparation_queue_root": str(queue), "publication_lock_root": str(host_root / "locks"),
                   "submission_transport": "local_owned_queue",
                   "service_account": pwd.getpwuid(os.geteuid()).pw_name})
    finally:
        fixture.SHA, template.SHA = old_fixture, old_template
    if release_binding is not None:
        if (release_binding.get("source_commit") != source_commit
                or release_binding.get("release_digest") != canonical_digest(release_binding, digest_field="release_digest")):
            raise ValueError("harness_release_binding_mismatch")
        (host_root / "release.json").write_text(json.dumps(release_binding))
    store = FilesystemObjectStore(object_root)

    def publish(**kwargs):
        return publication.publish_scene_configuration_submission(**kwargs, client=store)

    with fixture_environment(values):
        result = progression.process_scene_intents(config_path=config, publisher=publish,
                                                  only_intent_id=intent_id, now=now)
    intent_path = intents / intent_id / "intent.json"
    intent = json.loads(intent_path.read_text())
    state = json.loads((intents / intent_id / "progression.json").read_text())
    factory_record = state.get("state", {}).get("factory")
    output = {"claim_ceiling": "development_only", "source_commit": source_commit,
              "release_is_contract_fixture": release_binding is None,
              "config_path": config, "intent": intent, "intent_path": intent_path,
              "progression": result, "preparation_queue": queue,
              "environment": values, "object_bytes_uploaded": store.uploaded_bytes,
              "object_bytes_read_back": store.read_bytes}
    if factory_record:
        factory = json.loads(Path(factory_record["path"]).read_text())
        output["factory"] = factory
        output["request_path"] = Path(factory["submission_request"]["path"])
    publication_record = state.get("state", {}).get("publication")
    if publication_record:
        output["publication"] = json.loads(Path(publication_record["path"]).read_text())
    return output


def _fixture_render_inputs(*, envelope: dict, stage_one_configuration: dict, output_root: Path) -> dict:
    """External renderer fixture; source/reference admission stays in the worker."""
    from PIL import Image
    output_root.mkdir(parents=True)
    seed = canonical_digest({"envelope": envelope["envelope_digest"], "stage": stage_one_configuration})
    frames = []
    for index in range(8):
        path = output_root / f"fixture-{index:02d}.png"
        Image.new("RGB", (32, 32), (index * 20, 80, 160)).save(path)
        frames.append({"path": str(path), "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                       "size_bytes": path.stat().st_size})
    value = {"schema_version": "task_evaluation_scene_configuration_render_inputs.v1",
             "status": "derived_method_inputs_materialized", "run_id": envelope["request"]["run_id"],
             "fixture_provider": True, "fixture_input_digest": seed, "claim_ceiling": "development_only",
             "derived_frames": frames, "derived_frame_count": len(frames),
             "renderer_qualified": False, "physical_truth_claimed": False,
             "raw_interiorgs_bytes_in_provider_packet": False,
             "provider_mutation_performed": False, "paid_execution_requested": False}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def advance_fixture_preparation(*, intake: dict, object_root: Path,
                                reservation_root: Path, pins_root: Path) -> dict:
    from blueprint_pipeline.task_evaluation_launch_preparation_worker import (
        default_reference_fetcher, process_launch_preparation_queue,
    )
    from blueprint_pipeline.task_evaluation_owner_source_store import PREFIX
    store = FilesystemObjectStore(object_root)
    host = intake["config_path"].parent
    construction = host / "construction-queue"

    def fetch(uri: str, destination: Path, maximum_bytes: int) -> None:
        if uri.startswith(PREFIX):
            default_reference_fetcher(uri, destination, maximum_bytes)
            return
        parsed = urlsplit(uri)
        if parsed.scheme != "s3" or parsed.query or parsed.fragment:
            raise ValueError("harness_fixture_reference_uri_refused")
        size = store.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["ContentLength"]
        if size > maximum_bytes:
            raise ValueError("harness_fixture_reference_size_exceeded")
        with store.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"] as source:
            with destination.open("xb") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)

    with fixture_environment(intake["environment"]):
        run = process_launch_preparation_queue(queue_root=intake["preparation_queue"],
            input_root=host.parent / "prepared-references",
            allowed_uri_prefixes=["s3://blueprint/task-evaluation/"],
            service_account=pwd.getpwuid(os.geteuid()).pw_name, source_commit=intake["source_commit"],
            fetcher=fetch, scene_render_input_materializer=_fixture_render_inputs,
            construction_queue_root=construction, episode_compilation_queue_root=host / "episode-compilation",
            disk_reservation_root=reservation_root, storage_pins_root=pins_root,
            installed_source_environment={})
    rows = list((construction / "pending").glob("*.json"))
    return {"run": run, "construction_queue": construction,
            "object_bytes_fetched": store.read_bytes,
            "construction_envelope": json.loads(rows[0].read_text()) if len(rows) == 1 else None}


def advance_fixture_configuration(*, preparation: dict, object_root: Path, output_root: Path,
                                  provider=None) -> dict:
    from blueprint_pipeline.task_evaluation_scene_configuration_publication import publish_configured_scene_revision
    from blueprint_pipeline.task_evaluation_scene_construction_queue import finalize_scene_construction
    from scripts.control_plane_concurrency_provider import fixture_publisher, fixture_scene_artifacts
    envelope = preparation["construction_envelope"]
    stage_results = (provider or fixture_scene_artifacts)(envelope=envelope, output_root=output_root / "fixture-provider")
    if any(row.get("fixture_input_digest") != envelope["envelope_digest"]
           or row.get("actual_provider_calls") != 0 for row in stage_results):
        raise ValueError("harness_fixture_provider_input_mismatch")
    (output_root / "publication").mkdir(mode=0o700)
    publication = publish_configured_scene_revision(envelope=envelope, stage_results=stage_results,
        output_root=output_root / "publication", publisher=fixture_publisher(object_root))
    revision = json.loads(Path(publication["configured_scene_revision"]["path"]).read_text())
    terminal = {"schema_version": "task_evaluation_scene_configuration_vast_result.v1",
                "status": "completed", "run_id": envelope["run_id"],
                "source_commit": envelope["expected_production_commit"],
                "fixture_provider": True, "actual_provider_calls": 0, "claim_ceiling": "development_only",
                "configuration_completed": True, "configured_scene_published": True,
                "configured_scene_revision_digest": revision["revision_digest"],
                "publication_result_digest": publication["result_digest"],
                "full_byte_service_account_readback_passed": True, "provider_mutations_performed": 1,
                "retry_cap": 0, "evaluation_episode_executed": False, "candidate_policy_queried": False,
                "continuing_spend_from_this_run": False, "blockers": []}
    finalization = finalize_scene_construction(queue_root=preparation["construction_queue"],
        envelope={**envelope, "control_plane_envelope_digest": envelope["envelope_digest"]}, terminal_result=terminal)
    terminal["scene_construction_queue_finalization"] = finalization
    terminal["result_digest"] = canonical_digest(terminal, digest_field="result_digest")
    return {"publication": publication, "revision": revision, "terminal": terminal}
