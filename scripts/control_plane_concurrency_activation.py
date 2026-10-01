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
    from blueprint_pipeline.task_evaluation_supervisor.openai_cost_authority import (
        OpenAIOrganizationCostsClient,
    )

    original_cost_transport = OpenAIOrganizationCostsClient.__dict__["_default_transport"]

    def costs(url, headers, timeout):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "api.openai.com"
            or parsed.path != "/v1/organization/costs"
        ):
            raise PermissionError("harness_cost_fixture_endpoint_denied")
        return {"object": "page", "data": [], "has_more": False, "fixture_provider": True}

    OpenAIOrganizationCostsClient._default_transport = staticmethod(costs)
    original = boto3.client

    def client(service, **kwargs):
        if service != "s3":
            raise PermissionError("harness_external_client_denied")
        if (
            kwargs.get("endpoint_url") != "https://object-store.fixture.invalid"
            or kwargs.get("aws_access_key_id") != "fixture-access"
            or kwargs.get("aws_secret_access_key") != "fixture-secret"
        ):
            raise PermissionError("harness_object_store_fixture_binding_invalid")
        return store

    boto3.client = client
    try:
        with fixture_environment({
            "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL": "https://object-store.fixture.invalid",
            "AWS_ACCESS_KEY_ID": "fixture-access",
            "AWS_SECRET_ACCESS_KEY": "fixture-secret",
        }):
            yield
    finally:
        boto3.client = original
        OpenAIOrganizationCostsClient._default_transport = original_cost_transport


def canonical_fixture_preparer(*, lane, context_path, receipt_path, repository_root, object_root):
    """Execute original fixed preparation commands in the isolated child.

    The runner invokes CLI main functions unchanged, never returns fabricated
    step receipts. This retains the no-subprocess/no-credential child fences.
    """
    from scripts import prepare_paid_lane_launch as prepare

    context = (
        prepare._load_scene_configuration_context(context_path, expected_lane=lane)
        if lane == "task_evaluation_scene_configuration"
        else prepare._load_native_context(context_path, expected_lane=lane)
    )
    allowed = {tuple(step.argv[1:3]) for step in prepare.LANES[lane]}
    store = FilesystemObjectStore(object_root)

    def runner(argv):
        if (
            argv[0] != sys.executable
            or "--execute" in argv
            or tuple(argv[1:3])
            not in {tuple(prepare._render(value, context) for value in pair) for pair in allowed}
        ):
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
        return subprocess.CompletedProcess(
            list(argv), status or 0, stdout.getvalue(), stderr.getvalue()
        )

    receipt = prepare.prepare_paid_lane_launch(lane, context, runner=runner)
    with Path(receipt_path).open("x") as stream:
        json.dump(receipt, stream)
    return receipt


def advance_fixture_configuration_activation(
    *, intake, preparation, object_root, output_root, reservation_root
):
    from blueprint_pipeline import task_evaluation_scene_progression as progression
    from blueprint_pipeline import (
        task_evaluation_scene_configuration_activation_automation as activation,
    )
    from blueprint_pipeline.task_evaluation_launch_activation_worker import (
        process_launch_activation_queue,
    )
    from scripts.control_plane_concurrency_provider import fixture_publisher
    from tests.test_task_evaluation_scene_configuration_activation_automation import _provider_zero
    from tests.test_task_evaluation_scene_configuration_bundle import _toolchain
    from tests.test_project_spend_reconciliation import _human_baseline
    from blueprint_pipeline.project_spend_reconciliation import (
        materialize_project_spend_reconciliation,
    )

    output_root.mkdir(mode=0o700)
    source = intake["source_commit"]
    account = pwd.getpwuid(os.geteuid())
    group = grp.getgrgid(account.pw_gid).gr_name
    spend = output_root / "fixture-project-spend.json"
    baseline_path, baseline = _human_baseline(output_root / "fixture-baseline.json")
    text = "FICTIONAL no-paid boundary fixture. This is not human production authorization. Opening fixture exposure is zero."
    baseline.update(
        authorization_text=text,
        authorization_text_sha256="sha256:" + hashlib.sha256(text.encode()).hexdigest(),
        opening_project_exposure_usd=0.0,
        aggregate_project_ceiling_usd=100.0,
        maximum_bounded_exposure_after_full_attempt_reserve_usd=0.75,
        minimum_guaranteed_headroom_after_full_attempt_reserve_usd=99.25,
        fixture_provider=True,
        claim_ceiling="development_only",
    )
    baseline_path.write_text(json.dumps(baseline))
    materialize_project_spend_reconciliation(
        baseline_authority_path=baseline_path,
        posted_reconciliation_paths=[],
        expected_coverage_ids=[],
        completeness_reference="FICTIONAL no-paid development fixture",
        authorized_by="fixture-coordinator",
        authorized_on=datetime.now(timezone.utc).isoformat(),
        output_path=spend,
    )
    current = {
        "schema_version": "task_evaluation_project_spend_current.v1",
        "path": str(spend),
        "digest": "sha256:" + hashlib.sha256(spend.read_bytes()).hexdigest(),
        "observed_at_epoch": time.time(),
        "receipt_digest": "",
    }
    current["receipt_digest"] = canonical_digest(current, digest_field="receipt_digest")
    current_path = output_root / "fixture-project-spend-current.json"
    current_path.write_text(json.dumps(current))
    config = json.loads(intake["config_path"].read_text())
    (output_root / "launch-executions").mkdir(mode=0o700)
    (output_root / "configured-controls-intents").mkdir(mode=0o700)
    config.update(
        activation_enabled=True,
        activation_intent_root=str(output_root / "intents"),
        project_spend_current_path=str(current_path),
        service_group=group,
        launch_execution_root=str(output_root / "launch-executions"),
    )
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

    from blueprint_pipeline.task_evaluation_scene_configuration_openai_runtime_scope import (
        OPENAI_RUNTIME_FILE_ENVS,
        OPENAI_RUNTIME_VALUE_ENVS,
    )

    fixture_scopes = {}
    scopes = output_root / "fictional-openai-scopes"
    scopes.mkdir(mode=0o700)
    for index, name in enumerate(OPENAI_RUNTIME_FILE_ENVS):
        path = scopes / (name + ".fixture")
        path.write_text("FICTIONAL_NON_CREDENTIAL_VALUE_" + str(index))
        path.chmod(0o600)
        fixture_scopes[name] = str(path)
    for index, name in enumerate(OPENAI_RUNTIME_VALUE_ENVS):
        fixture_scopes[name] = "fictional-scope-" + str(index)
    values = {
        **intake["environment"],
        **fixture_scopes,
        "BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_TOOLCHAIN_ROOT": str(toolchain),
    }
    with fixture_environment(values):
        progressed = progression.process_scene_intents(
            config_path=intake["config_path"],
            only_intent_id=intake["intent"]["intent_id"],
            activation_provisioner=activation.provision_scene_configuration_activation_intent,
        )
        if progressed["results"][0]["phase"] != "scene_configuration":
            raise ValueError("harness_configuration_activation_blocked:" + json.dumps(progressed))
        result_files = list((intake["preparation_queue"] / "results").glob("*.json"))
        if len(result_files) != 1:
            raise ValueError("harness_configuration_preparation_result_ambiguous")
        staged = activation.advance_scene_configuration_activation(
            preparation_result_path=result_files[0],
            preparation_queue_root=intake["preparation_queue"],
            activation_queue_root=queue,
            progression_root=output_root / "progression",
            intent_root=output_root / "intents",
            provider_zero_collector=lambda: zero,
            lineage_publisher_factory=lambda: fixture_publisher(object_root),
            release_window_publisher_factory=lambda: fixture_publisher(object_root),
            now=now,
            running_commit=source,
        )
        worker = process_launch_activation_queue(
            queue_root=queue,
            preparation_queue_root=intake["preparation_queue"],
            preparation_input_root=intake["config_path"].parent.parent / "prepared-references",
            activation_root=output_root / "activated",
            allowed_uri_prefixes=["s3://blueprint/"],
            service_account=account.pw_name,
            service_group=group,
            repository_root=Path(__file__).resolve().parents[1],
            destination_prefix="s3://blueprint/task-evaluation/fixture-activated",
            release_window_prefix="s3://blueprint/task-evaluation/",
            profile_dir=output_root / "profiles",
            webapp_catalog=output_root / "catalog.json",
            standing_authorization_dir=output_root / "authorizations",
            scene_construction_queue_root=preparation["construction_queue"],
            scene_configuration_toolchain_root=toolchain,
            source_commit=source,
            configured_controls_autostart_intent_root=output_root / "configured-controls-intents",
            fetcher=fetch,
            preparer=preparer,
            disk_reservation_root=reservation_root,
        )
    return {
        "progression": progressed,
        "staged": staged,
        "worker": worker,
        "queue_root": queue,
        "request_roots": [
            output_root / "launch-executions",
            output_root / "configured-controls-intents",
        ],
        "preparer": receipts[0] if len(receipts) == 1 else None,
        "fixture_provider": True,
        "claim_ceiling": "development_only",
        "actual_provider_calls": 0,
    }


def advance_fixture_native_activation(
    *,
    intake,
    episode,
    preparation,
    compiled,
    configuration_activation,
    object_root,
    output_root,
    reservation_root,
):
    """Bind the real compiler to the original native construction launch graph."""
    from blueprint_pipeline.task_evaluation_configured_controls_progression import (
        build_configured_controls_activation_request,
        stage_configured_controls_activation,
    )
    from blueprint_pipeline.task_evaluation_shared_mutation_window import (
        materialize_shared_mutation_window,
    )
    from blueprint_pipeline.task_evaluation_launch_activation_worker import (
        process_launch_activation_queue,
    )
    from scripts.control_plane_concurrency_provider import fixture_publisher

    output_root.mkdir(mode=0o700)
    for name in ("launch-executions", "configured-controls-intents"):
        (output_root / name).mkdir(mode=0o700)
    source = intake["source_commit"]
    if (
        compiled.get("status") != "compiled_for_production_launch"
        or compiled.get("result_digest") != canonical_digest(compiled, digest_field="result_digest")
        or compiled.get("source_commit") != source
    ):
        raise ValueError("harness_native_compilation_invalid")
    first_queue = Path(configuration_activation["queue_root"])
    rows = list((first_queue / "prepared").glob("*.json"))
    if len(rows) != 1:
        raise ValueError("harness_native_configuration_activation_ambiguous")
    first = json.loads(rows[0].read_text())["request"]
    authorization = dict(first["authorization"])
    from blueprint_pipeline.task_evaluation_scene_intake import reserve_scene_attempt
    from blueprint_pipeline.task_evaluation_scene_execution_authority import bind_scene_attempt
    from blueprint_pipeline.task_evaluation_scene_owner_attempt_profiles import (
        make_owner_attempt_record,
    )

    request = episode["episode_preparation_request"]
    with fixture_environment(intake["environment"]):
        attempt = reserve_scene_attempt(
            queue_root=intake["intent_path"].parent.parent,
            intent_id=intake["intent"]["intent_id"],
            attempt_id="fixture-native-construction-"
            + episode["episode_preparation_request_digest"][7:31],
            source_commit=source,
            runtime_digest=request["execution_adapter"]["runtime_source_bundle"]["digest"],
            input_digest=episode["episode_preparation_request_digest"],
            provider=request["spend"]["selected_provider"],
            maximum_spend_usd=request["spend"]["hard_cap_usd"],
        )
    authorization["scene_owner_attempt"] = make_owner_attempt_record(
        owner_fields=bind_scene_attempt(attempt),
        phase="construction",
        team_namespace=request["team_namespace"],
        scene_id=request["scene"]["identity"]["id"],
        task_id=request["task"]["identity"]["id"],
        runtime_source_bundle_digest=request["execution_adapter"]["runtime_source_bundle"][
            "digest"
        ],
    )
    lineage = first["lineage"]
    prep_result = preparation["run"]["results"][0]
    base = build_configured_controls_activation_request(
        progression=episode,
        preparation_result=prep_result,
        release_window=first["release_window"],
        lineage=lineage,
        authorization=authorization,
        lane="native_task_arena_construction",
    )
    template = {
        "schema_version": "task_evaluation_configured_controls_release_window_template.v1",
        "status": "authorized_for_dynamic_release",
        "team_namespace": base["team_namespace"],
        "expected_production_commit": source,
        "allowed_mutations": [
            "catalog_synchronization",
            "profile_publication",
            "standing_authorization",
        ],
        "provider_allowlist": ["vast"],
        "maximum_hard_cap_usd": 12.0,
        "valid_for_seconds": 3600,
        "released_by": "fictional-fixture-coordinator",
        "release_reference": "ADP-009D development_only local fixture",
        "provider_resource_allocation_allowed": False,
        "paid_request_allowed": False,
    }
    template["template_digest"] = canonical_digest(template, digest_field="template_digest")
    window = materialize_shared_mutation_window(
        template, activation_request=base, provider_allowlist=["vast"], hard_cap_usd=12.0
    )
    path = output_root / "fixture-release-window.json"
    path.write_text(json.dumps(window))
    published = fixture_publisher(object_root)(
        path=path, object_name="fixture-native-release-window.json"
    )
    reference = {key: published[key] for key in ("uri", "digest", "size_bytes")}
    queue = output_root / "activation-queue"
    staged = stage_configured_controls_activation(
        progression=episode,
        preparation_result=prep_result,
        release_window=reference,
        lineage=lineage,
        authorization=authorization,
        lane="native_task_arena_construction",
        queue_root=queue,
        submitted_by="fictional-development-fixture",
    )
    account = pwd.getpwuid(os.geteuid())
    store = FilesystemObjectStore(object_root)

    def fetch(uri, destination, maximum_bytes):
        parsed = urlsplit(uri)
        if parsed.scheme != "s3" or parsed.query or parsed.fragment:
            raise ValueError("harness_native_reference_uri_invalid")
        kwargs = {"Bucket": parsed.netloc, "Key": parsed.path.lstrip("/")}
        if store.head_object(**kwargs)["ContentLength"] > maximum_bytes:
            raise ValueError("harness_native_reference_size_exceeded")
        with store.get_object(**kwargs)["Body"] as body, destination.open("xb") as target:
            shutil.copyfileobj(body, target, 1024 * 1024)

    receipts = []

    def prepare(**kwargs):
        result = canonical_fixture_preparer(**kwargs, object_root=object_root)
        receipts.append(result)
        return result

    with fixture_environment(intake["environment"]):
        worker = process_launch_activation_queue(
            queue_root=queue,
            preparation_queue_root=intake["preparation_queue"],
            preparation_input_root=intake["config_path"].parent.parent / "prepared-references",
            episode_compilation_queue_root=preparation["compilation_queue"],
            episode_compilation_output_root=Path(compiled["adapter_result_path"]).parent.parent,
            activation_root=output_root / "activated",
            allowed_uri_prefixes=["s3://blueprint/"],
            service_account=account.pw_name,
            service_group=grp.getgrgid(account.pw_gid).gr_name,
            repository_root=Path(__file__).resolve().parents[1],
            destination_prefix="s3://blueprint/task-evaluation/fixture-native-activated",
            release_window_prefix="s3://blueprint/task-evaluation/",
            profile_dir=output_root / "profiles",
            webapp_catalog=output_root / "catalog.json",
            standing_authorization_dir=output_root / "authorizations",
            source_commit=source,
            fetcher=fetch,
            preparer=prepare,
            disk_reservation_root=reservation_root,
        )
    return {
        "staged": staged,
        "worker": worker,
        "queue_root": queue,
        "request_roots": [
            output_root / "launch-executions",
            output_root / "configured-controls-intents",
        ],
        "preparer": receipts[0] if len(receipts) == 1 else None,
        "fixture_provider": True,
        "actual_provider_calls": 0,
        "claim_ceiling": "development_only",
    }
