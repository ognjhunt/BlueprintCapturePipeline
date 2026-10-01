"""Plan 13c real activation graphs with bounded external storage fixtures.

ADP-009D/day 28. No paid execution. All provider inventory and spend receipts
below are fictional boundary inputs; they confer no production authority.
"""

from __future__ import annotations

import importlib
import hashlib
import io
import json
import os
import pwd
import grp
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from scripts.control_plane_concurrency_fixture import FilesystemObjectStore
from scripts.control_plane_concurrency_scene import fixture_environment


@contextmanager
def _fixture_s3(store):
    """Replace only external transport construction for actual CLI publication."""
    import boto3
    original = boto3.client
    def client(service, **kwargs):
        if service != "s3":
            raise PermissionError("harness_external_client_denied")
        return store
    boto3.client = client
    try:
        yield
    finally:
        boto3.client = original


def canonical_fixture_preparer(*, lane, context_path, receipt_path, repository_root,
                               object_root):
    """Execute original fixed preparation commands in the isolated child.

    The runner invokes CLI main functions unchanged, never returns fabricated
    step receipts. This retains the no-subprocess/no-credential child fences.
    """
    from scripts import prepare_paid_lane_launch as prepare
    context = (prepare._load_scene_configuration_context(context_path, expected_lane=lane)
               if lane == "task_evaluation_scene_configuration"
               else prepare._load_native_context(context_path, expected_lane=lane))
    allowed = {tuple(step.argv[1:3]) for step in prepare.LANES[lane]}
    store = FilesystemObjectStore(object_root)

    def runner(argv):
        if (argv[0] != sys.executable or "--execute" in argv
                or tuple(argv[1:3]) not in {
                    tuple(prepare._render(value, context) for value in pair) for pair in allowed}):
            raise PermissionError("harness_preparation_command_denied")
        if argv[1] == "-m":
            module, arguments = importlib.import_module(argv[2]), list(argv[3:])
        else:
            script = Path(argv[1]).resolve()
            if script.parent != Path(repository_root).resolve() / "scripts":
                raise PermissionError("harness_preparation_script_outside_checkout")
            module, arguments = importlib.import_module("scripts." + script.stem), list(argv[2:])
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr), _fixture_s3(store):
                status = module.main(arguments)
        except SystemExit as exc:
            status = exc.code
        except Exception as exc:
            # A real subprocess would exit nonzero with this original error.
            # Retain that failure in the canonical graph's normal step log.
            status = 1
            stderr.write(type(exc).__name__ + ":" + str(exc))
        return subprocess.CompletedProcess(list(argv), status or 0, stdout.getvalue(), stderr.getvalue())

    receipt = prepare.prepare_paid_lane_launch(lane, context, runner=runner)
    with Path(receipt_path).open("x") as stream:
        json.dump(receipt, stream)
    return receipt


def advance_fixture_configuration_activation(*, intake, preparation, object_root,
                                              output_root, reservation_root):
    from blueprint_pipeline import task_evaluation_scene_progression as progression
    from blueprint_pipeline import task_evaluation_scene_configuration_activation_automation as activation
    from blueprint_pipeline.task_evaluation_launch_activation_worker import process_launch_activation_queue
    from scripts.control_plane_concurrency_provider import fixture_publisher
    from tests.test_task_evaluation_scene_configuration_activation_automation import _provider_zero
    from tests.test_task_evaluation_scene_configuration_bundle import _toolchain
    from tests.test_project_spend_reconciliation import _human_baseline
    from blueprint_pipeline.project_spend_reconciliation import materialize_project_spend_reconciliation
    output_root.mkdir(mode=0o700)
    source = intake["source_commit"]
    account = pwd.getpwuid(os.geteuid())
    group = grp.getgrgid(account.pw_gid).gr_name
    spend = output_root / "fixture-project-spend.json"
    baseline_path, baseline = _human_baseline(output_root / "fixture-baseline.json")
    text = "FICTIONAL no-paid boundary fixture. This is not human production authorization. Opening fixture exposure is zero."
    baseline.update(authorization_text=text,
        authorization_text_sha256="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        opening_project_exposure_usd=0.0, aggregate_project_ceiling_usd=100.0,
        maximum_bounded_exposure_after_full_attempt_reserve_usd=0.75,
        minimum_guaranteed_headroom_after_full_attempt_reserve_usd=99.25,
        fixture_provider=True, claim_ceiling="development_only")
    baseline_path.write_text(json.dumps(baseline))
    materialize_project_spend_reconciliation(baseline_authority_path=baseline_path,
        posted_reconciliation_paths=[], expected_coverage_ids=[],
        completeness_reference="FICTIONAL no-paid development fixture",
        authorized_by="fixture-coordinator", authorized_on=datetime.now(timezone.utc).isoformat(),
        output_path=spend)
    current = {"schema_version": "task_evaluation_project_spend_current.v1",
        "path": str(spend), "digest": "sha256:" + hashlib.sha256(spend.read_bytes()).hexdigest(),
        "observed_at_epoch": time.time(), "receipt_digest": ""}
    current["receipt_digest"] = canonical_digest(current, digest_field="receipt_digest")
    current_path = output_root / "fixture-project-spend-current.json"
    current_path.write_text(json.dumps(current))
    config = json.loads(intake["config_path"].read_text())
    (output_root / "launch-executions").mkdir(mode=0o700)
    (output_root / "configured-controls-intents").mkdir(mode=0o700)
    config.update(activation_enabled=True, activation_intent_root=str(output_root / "intents"),
                  project_spend_current_path=str(current_path), service_group=group,
                  launch_execution_root=str(output_root / "launch-executions"))
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    intake["config_path"].write_text(json.dumps(config))
    toolchain = _toolchain(output_root / "toolchain", source)
    now = datetime.now(timezone.utc)
    zero = _provider_zero(now)
    zero.update(fixture_provider=True, actual_provider_calls=0, claim_ceiling="development_only")
    zero["provider_zero_digest"] = canonical_digest(zero, digest_field="provider_zero_digest")
    queue = output_root / "activation-queue"
    receipts = []

    def preparer(**kwargs):
        receipt = canonical_fixture_preparer(**kwargs, object_root=object_root)
        receipts.append(receipt)
        return receipt

    store = FilesystemObjectStore(object_root)
    def fetch(uri, destination, maximum_bytes):
        parsed = urlsplit(uri)
        if parsed.scheme != "s3" or parsed.query or parsed.fragment:
            raise ValueError("harness_activation_fixture_uri_invalid")
        size = store.head_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["ContentLength"]
        if size > maximum_bytes:
            raise ValueError("harness_activation_fixture_reference_size_exceeded")
        with store.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"] as body:
            with destination.open("xb") as stream:
                shutil.copyfileobj(body, stream, 1024 * 1024)

    values = {**intake["environment"],
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT": str(toolchain)}
    with fixture_environment(values):
        progressed = progression.process_scene_intents(config_path=intake["config_path"],
            only_intent_id=intake["intent"]["intent_id"],
            activation_provisioner=activation.provision_scene_configuration_activation_intent)
        if progressed["results"][0]["phase"] != "scene_configuration":
            raise ValueError("harness_configuration_activation_blocked:" + json.dumps(progressed))
        result_files = list((intake["preparation_queue"] / "results").glob("*.json"))
        if len(result_files) != 1:
            raise ValueError("harness_configuration_preparation_result_ambiguous")
        staged = activation.advance_scene_configuration_activation(preparation_result_path=result_files[0],
            preparation_queue_root=intake["preparation_queue"], activation_queue_root=queue,
            progression_root=output_root / "progression", intent_root=output_root / "intents",
            provider_zero_collector=lambda: zero,
            lineage_publisher_factory=lambda: fixture_publisher(object_root),
            release_window_publisher_factory=lambda: fixture_publisher(object_root), now=now, running_commit=source)
        worker = process_launch_activation_queue(queue_root=queue,
            preparation_queue_root=intake["preparation_queue"],
            preparation_input_root=intake["config_path"].parent.parent / "prepared-references",
            activation_root=output_root / "activated", allowed_uri_prefixes=["s3://blueprint/"],
            service_account=account.pw_name, service_group=group,
            repository_root=Path(__file__).resolve().parents[1],
            destination_prefix="s3://blueprint/task-evaluation/fixture-activated",
            release_window_prefix="s3://blueprint/task-evaluation/",
            profile_dir=output_root / "profiles", webapp_catalog=output_root / "catalog.json",
            standing_authorization_dir=output_root / "authorizations",
            scene_construction_queue_root=preparation["construction_queue"],
            scene_configuration_toolchain_root=toolchain, source_commit=source,
            configured_controls_autostart_intent_root=output_root / "configured-controls-intents",
            fetcher=fetch, preparer=preparer, disk_reservation_root=reservation_root)
    return {"progression": progressed, "staged": staged, "worker": worker,
            "preparer": receipts[0] if len(receipts) == 1 else None,
            "fixture_provider": True, "claim_ceiling": "development_only", "actual_provider_calls": 0}
