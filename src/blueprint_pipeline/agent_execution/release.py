"""Drain reasoning sessions before changing their installed controller release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from ..common import write_json
from .contracts import AgentExecutionError, digest
from .journal import AgentJournal, TERMINAL_STATES

MARKER = "release_drain.json"


def pending_cleanup(journal):
    # Query the population rather than mistaking a bounded listing for zero.
    with journal._connect() as connection:
        rows = connection.execute("SELECT task_id FROM tasks WHERE state NOT IN ('completed','failed','cancelled') "
            "OR cleanup_state != 'deleted' ORDER BY task_id").fetchall()
    return [journal.task(row["task_id"]) for row in rows]


def drain(service, *, target_source_commit: str, timeout_seconds: float = 30, drive_worker: bool = True):
    if not 0 < timeout_seconds <= 1800:
        raise AgentExecutionError("agent_release_drain_timeout_invalid")
    import re
    if re.fullmatch(r"[0-9a-f]{40}", target_source_commit) is None:
        raise AgentExecutionError("agent_release_target_invalid")
    marker = {"schema_version": "blueprint_agent_release_drain.v1", "source_commit": service.config.source_commit,
              "target_source_commit": target_source_commit, "config_digest": service.config_digest}
    path = service.journal.root / MARKER
    with service.journal.own_task("release-adoption"):
        if path.exists() and json.loads(path.read_text()) != marker:
            raise AgentExecutionError("agent_release_drain_owner_conflict")
        write_json(path, marker)
    end = time.monotonic() + timeout_seconds
    errors = {}
    while True:
        remaining = pending_cleanup(service.journal)
        if not remaining or time.monotonic() >= end:
            break
        for task in remaining:
            task_id = task["task_id"]
            try:
                if task["state"] not in TERMINAL_STATES:
                    service.service.cancel(task_id)
                elif service.journal.successor(task_id) is None and not service.journal.unsettled_operations(task_id):
                    service.service.request_cleanup(task_id)
            except AgentExecutionError as exc:
                errors[task_id] = str(exc)
        # The root deployment controller must let the installed service account
        # read its private task records and provider credential. Root may queue
        # cleanup through the journal, but is not an admitted task-record owner.
        if drive_worker:
            service.service.tick()
        time.sleep(0.1)
    value = {**marker, "status": "drained" if not remaining else "reconciliation_pending",
             "remaining_task_ids": [task["task_id"] for task in remaining], "errors": errors,
             "inference_admission_disabled": True, "provider_resource_release_inferred": False,
             "historical_results_preserved": True}
    value["receipt_digest"] = digest(value)
    write_json(service.journal.root / "release-drains" / (target_source_commit + ".json"), value)
    return value


def adopt_drained_config(config_path, *, expected_commit: str):
    """Called with the worker quiesced by the canonical deployment controller."""
    from .production import ProductionConfig, _read_private

    path = Path(config_path)
    config = ProductionConfig.model_validate_json(_read_private(path))
    journal = AgentJournal(config.state_root)
    with journal.own_task("release-adoption"):
        marker_path = journal.root / MARKER
        if config.source_commit == expected_commit:
            if marker_path.exists():
                marker = json.loads(_read_private(marker_path))
                saved_path = journal.root / "release-adoptions" / (expected_commit + ".json")
                if marker.get("target_source_commit") == expected_commit and saved_path.exists():
                    saved = json.loads(_read_private(saved_path))
                    if (saved.get("receipt_digest") != digest({k: v for k, v in saved.items() if k != "receipt_digest"})
                            or saved.get("next_config_digest") != digest(config.model_dump(mode="json"))
                            or digest(saved.get("previous_config")) != marker.get("config_digest")
                            or pending_cleanup(journal)):
                        raise AgentExecutionError("agent_release_adoption_recovery_invalid")
                    marker_path.unlink()
            return {"status": "already_bound", "source_commit": expected_commit}
        if pending_cleanup(journal):
            raise AgentExecutionError("agent_release_requires_drain")
        if marker_path.exists():
            marker = json.loads(_read_private(marker_path))
            if marker.get("config_digest") != digest(config.model_dump(mode="json")) or marker.get("target_source_commit") != expected_commit:
                raise AgentExecutionError("agent_release_drain_binding_mismatch")
        # Empty historical stores need no provider operation. Retain both
        # configurations before switching the future execution identity.
        previous = config.model_dump(mode="json")
        following = {**previous, "source_commit": expected_commit}
        ProductionConfig.model_validate(following)
        receipt = {"schema_version": "blueprint_agent_configuration_adoption.v1",
            "previous_config": previous, "next_config_digest": digest(following), "source_commit": expected_commit,
            "prior_source_commit": config.source_commit, "all_prior_sessions_cleaned": True,
            "historical_task_records_modified": False, "provider_mutation_performed": False}
        receipt["receipt_digest"] = digest(receipt)
        write_json(journal.root / "release-adoptions" / (expected_commit + ".json"), receipt)
        metadata = path.stat()
        write_json(path, following)
        os.chown(path, metadata.st_uid, metadata.st_gid)
        path.chmod(metadata.st_mode & 0o777)
        if marker_path.exists():
            marker_path.unlink()
        return {"status": "adopted", "source_commit": expected_commit, "receipt_digest": receipt["receipt_digest"]}


def main(argv=None):
    from .production import ProductionAgentService, ProductionConfig, _read_private
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-source-commit", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    config = ProductionConfig.model_validate_json(_read_private(args.config))
    service = ProductionAgentService(args.config, source_commit=config.source_commit)
    value = drain(service, target_source_commit=args.target_source_commit, timeout_seconds=args.timeout_seconds)
    print(json.dumps(value, sort_keys=True))
    return 0 if value["status"] == "drained" else 2


if __name__ == "__main__":
    raise SystemExit(main())
