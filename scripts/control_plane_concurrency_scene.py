"""Joined developmental scene stages for the no-paid Plan 13c harness.

ADP-009D/day 28. Test data is generated afresh at the checked-out source. No
production validator is replaced. Contract-fixture release records are exposed
explicitly; a host acceptance run must supply its actual deployment binding.
"""

from __future__ import annotations

import json
import os
import pwd
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
