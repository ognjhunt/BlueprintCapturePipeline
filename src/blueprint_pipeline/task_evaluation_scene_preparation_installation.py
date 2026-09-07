"""Provision release-independent completed-scene preparation, with execution off.

One operator-owned bootstrap binds the existing asset catalog and storage
roots. Deployment recompiles it; scenes and release SHAs are never hand-edited
into the worker configuration. This module neither starts units nor submits a
run, and the installed configuration cannot activate a paid execution.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import tempfile

from .decision_evidence_contracts import canonical_digest
from .project_spend_reconciliation import validate_project_spend_reconciliation
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path

BOOTSTRAP_SCHEMA = "task_evaluation_scene_preparation_bootstrap.v1"
MANAGED_BY = "blueprint_pipeline.task_evaluation_scene_preparation_installation"
DEFAULT_BOOTSTRAP = "/etc/blueprint/task-evaluation-scene-preparation-bootstrap.json"
DEFAULT_PHYSICS_BOUNDS = {"mass_kg_bounds": [0.1, 1.2], "static_friction_bounds": [0.4, 0.8],
                        "dynamic_friction_bounds": [0.3, 0.6], "restitution_bounds": [0.0, 0.1]}


def validate_destination_catalog(rows):
    """Requalify the existing exact USDs; catalog flags alone are not admission."""
    from .task_evaluation_scene_configuration_submission import _destination
    from .task_evaluation_scene_configuration_static_qualification import qualify_scene_configuration_rigid_asset_static
    from .task_evaluation_scene_configuration_supplemental_destination import _requalification_comparable
    require(isinstance(rows, list) and 1 <= len(rows) <= 64, "destination_catalog_invalid")
    names, aliases = set(), set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"binding_id", "owner_description_aliases", "simready_result"},
                "destination_catalog_row_invalid")
        name = row["binding_id"]
        require(isinstance(name, str) and name and name not in names, "destination_catalog_identity_invalid")
        names.add(name)
        labels = row["owner_description_aliases"]
        require(isinstance(labels, list) and labels and len(labels) <= 16
                and all(isinstance(label, str) and 0 < len(label.strip()) <= 160 for label in labels),
                "destination_catalog_alias_invalid")
        normalized = {label.strip().casefold() for label in labels}
        require(len(normalized) == len(labels) and not aliases.intersection(normalized), "destination_catalog_alias_ambiguous")
        aliases.update(normalized)
        reference = row["simready_result"]
        result, static, paths = _destination(checked_file(reference["path"], reference), [0, 0, 0], [1e-9] * 3)
        authoring = read(paths["authoring_receipt"], digest_field="result_digest")
        identity = result["destination_identity"]
        graph = {"schema_version": "task_evaluation_rigid_replacement_graph.v1", "asset_id": identity["id"],
            "asset_version": identity["version"], "articulation_graph": {"joints": []}, "single_rigid_candidate": True,
            "physics_bounds": authoring["candidate_physics_completion"]["physics_bounds"], "physics_authority_granted": False}
        with tempfile.TemporaryDirectory(prefix="scene-catalog-check-") as temporary:
            actual = qualify_scene_configuration_rigid_asset_static(asset_path=paths["asset"], graph_spec=graph,
                authoring_receipt=authoring, replacement_identity=identity,
                output_path=Path(temporary) / "static.json")
        require(_requalification_comparable(actual) == _requalification_comparable(static),
                "destination_catalog_static_requalification_changed")
    return rows


#: Owner-upload sources the preparation service admits by default.
OWNER_UPLOAD_SOURCE_KINDS = ["mesh", "gaussian_splat"]
#: With public_scene_enabled the service additionally admits rights-admitted
#: public-scene persistent intents (Spec A: the legacy public-scene path for
#: scenes such as 841757). It stays a separate no-spend preparation phase; the
#: per-scene public-source binding and public-scene machinery are materialized
#: by the public-scene provisioner, not manufactured here.
PUBLIC_SCENE_SOURCE_KINDS = ["mesh", "gaussian_splat", "public_scene"]
PROJECT_SPEND_MONITOR_SCHEMA = "task_evaluation_scene_project_spend_monitor.v1"


def build_bootstrap(*, destination_catalog, config_root="/etc/blueprint",
                    state_root="/var/lib/blueprint/pipeline-control-plane",
                    inputs_root="/var/lib/blueprint/task-evaluation-inputs",
                    capture_store_root="/var/lib/blueprint/capture-intake",
                    running_repo_root="/opt/blueprint/task-evaluation-control-plane", service_account="blueprint",
                    public_scene_enabled=False, activation_authorized=False,
                    project_spend_reconciliation_path=None):
    rows = validate_destination_catalog(destination_catalog)
    roots = {key: str(safe_path(value)) for key, value in {
        "config_root": config_root, "state_root": state_root, "inputs_root": inputs_root,
        "capture_store_root": capture_store_root}.items()}
    require(Path(running_repo_root).is_absolute(), "scene_preparation_running_repo_invalid")
    pwd.getpwnam(service_account)
    project_spend_seed = None
    if project_spend_reconciliation_path is not None:
        seed = safe_path(project_spend_reconciliation_path)
        require(seed.is_file() and not seed.is_symlink(), "project_spend_seed_missing")
        seed_reference = record(seed)
        try:
            spend, _ = validate_project_spend_reconciliation(seed)
        except (OSError, ValueError, TypeError):
            require(False, "project_spend_seed_invalid")
        require(spend.get("provider_mutation_performed") is False, "project_spend_seed_scope_invalid")
        project_spend_seed = seed_reference
    value = {"schema_version": BOOTSTRAP_SCHEMA, "managed_by": MANAGED_BY, **roots,
        "running_repo_root": str(running_repo_root), "service_account": service_account,
        "destination_catalog": rows, "simulation_physics_bounds": {key: list(value) for key, value in DEFAULT_PHYSICS_BOUNDS.items()},
        "execution_activation_enabled": False,
        # A3: the SEPARATELY-ADMITTED activation on-ramp. Off by default keeps the
        # installed service preparation-only (no scene-configuration activation is
        # produced). When the owner authorizes activation, the installer emits an
        # activation config (activation_enabled True + activation_intent_root +
        # project_spend_current_path) that the service runs as a second no-spend
        # pass. This never arms paid dispatch; the paid GPU flip stays separate.
        "activation_authorized": bool(activation_authorized),
        "supported_source_kinds": PUBLIC_SCENE_SOURCE_KINDS if public_scene_enabled else OWNER_UPLOAD_SOURCE_KINDS}
    if project_spend_seed is not None:
        value["project_spend_seed"] = project_spend_seed
    value["bootstrap_digest"] = canonical_digest(value, digest_field="bootstrap_digest")
    return value


def _managed_json(path, value, account):
    path = safe_path(path)
    unchanged = False
    if path.exists():
        old = read(path)
        require(old.get("managed_by") == MANAGED_BY, "scene_preparation_unmanaged_file")
        unchanged = old == value
    if not unchanged:
        atomic_json(path, value)
    if os.geteuid() == 0:
        os.chown(path, 0, account.pw_gid)
    path.chmod(0o640)


def install_scene_preparation(*, bootstrap_path):
    bootstrap_path = safe_path(bootstrap_path)
    require(not bootstrap_path.stat().st_mode & 0o022, "scene_preparation_bootstrap_writable")
    bootstrap = read(bootstrap_path, digest_field="bootstrap_digest")
    require(bootstrap.get("schema_version") == BOOTSTRAP_SCHEMA
            and bootstrap.get("managed_by") == MANAGED_BY
            and bootstrap.get("execution_activation_enabled") is False
            and bootstrap.get("supported_source_kinds") in (OWNER_UPLOAD_SOURCE_KINDS, PUBLIC_SCENE_SOURCE_KINDS),
            "scene_preparation_bootstrap_scope_invalid")
    validate_destination_catalog(bootstrap["destination_catalog"])
    from .task_evaluation_scene_configuration_content_agents_driver import _physics_bounds
    _physics_bounds({"required_output": bootstrap["simulation_physics_bounds"]})
    account = pwd.getpwnam(bootstrap["service_account"])
    state, inputs, config_root = (safe_path(bootstrap[key]) for key in ("state_root", "inputs_root", "config_root"))
    owner_queue = state / "task-evaluation-owned-scene-preparations"
    public_scene_enabled = "public_scene" in bootstrap["supported_source_kinds"]
    public_binding_root = inputs / "public-source-bindings"
    project_spend_seed = bootstrap.get("project_spend_seed")
    seed_path = None
    if project_spend_seed is not None:
        require(isinstance(project_spend_seed, dict), "project_spend_seed_invalid")
        seed_path = checked_file(project_spend_seed.get("path", ""), project_spend_seed)
        try:
            seed, _ = validate_project_spend_reconciliation(seed_path)
        except (OSError, ValueError, TypeError):
            require(False, "project_spend_seed_invalid")
        require(seed.get("provider_mutation_performed") is False, "project_spend_seed_scope_invalid")
    project_spend_root = state / "scene-project-spend"
    directories = [state / "task-evaluation-scene-intents", owner_queue,
        inputs / "owner-source-store", inputs / "completed-scene-preparation",
        inputs / "completed-scene-preparation-inputs", state / "scene-preparation-release-bindings",
        state / "scene-preparation-service", state / "submission-publication-locks",
        state / "disk-reservations", state / "storage-pins", project_spend_root,
        inputs / "task-evaluation-terminal-results"]
    if public_scene_enabled:
        # The public-scene binding directory the per-scene provisioner writes
        # <binding_id>.json into; the scene-progression _source resolver reads
        # public_source_binding_root/<binding_id>.json for a public_scene intent.
        directories.append(public_binding_root)
    for directory in directories:
        safe_path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, mode=0o750)
            if os.geteuid() == 0:
                os.chown(directory, account.pw_uid, account.pw_gid)
    from .task_evaluation_launch_preparation_queue import ensure_launch_preparation_queue_root
    ensure_launch_preparation_queue_root(owner_queue)
    for directory in owner_queue.iterdir():
        if directory.is_dir() and os.geteuid() == 0:
            os.chown(directory, account.pw_uid, account.pw_gid)
    machinery = {"schema_version": "task_evaluation_completed_scene_machinery.v1", "managed_by": MANAGED_BY,
        "maximum_preparation_spend_usd": 0, "provider": "control_plane",
        "destination_catalog": bootstrap["destination_catalog"], "simulation_physics_bounds": bootstrap["simulation_physics_bounds"]}
    machinery["machinery_digest"] = canonical_digest(machinery, digest_field="machinery_digest")
    machinery_path = config_root / "task-evaluation-completed-scene-machinery.json"
    _managed_json(machinery_path, machinery, account)
    config = {"schema_version": "task_evaluation_scene_progression_config.v1", "managed_by": MANAGED_BY,
        "intent_root": str(directories[0]), "capture_store_root": bootstrap["capture_store_root"],
        "factory_output_root": str(inputs / "completed-scene-preparation"),
        "completed_source_machinery_path": str(machinery_path),
        # A6: where the terminal-result index files a paid run's sealed receipts.
        # The reconciler hook is additionally gated on an activation record, so this
        # is inert until the activation on-ramp runs and a run is indexed here.
        "terminal_result_root": str(inputs / "task-evaluation-terminal-results"),
        "deployment_receipt_root": str(state / "deploy-receipts"),
        "release_binding_root": str(state / "scene-preparation-release-bindings"),
        "running_repo_root": bootstrap["running_repo_root"], "runtime_publication_root": str(inputs / "system-runtimes"),
        "trusted_clients": ["blueprint-webapp"], "supported_source_kinds": bootstrap["supported_source_kinds"],
        "maximum_intents_per_pass": 16, "maximum_http_submission_attempts": 2,
        "submission_enabled": True, "submission_transport": "local_owned_queue", "activation_enabled": False,
        "service_account": bootstrap["service_account"], "preparation_queue_root": str(owner_queue),
        "publication_lock_root": str(state / "submission-publication-locks"),
        "service_status_path": str(state / "scene-preparation-service/latest.json"),
        "preparation_worker": {"input_root": str(inputs / "completed-scene-preparation-inputs"),
            "construction_queue_root": str(state / "task-evaluation-scene-constructions"),
            "disk_reservation_root": str(state / "disk-reservations"), "storage_pins_root": str(state / "storage-pins"),
            "max_messages": 1, "allowed_uri_prefixes": ["s3://blueprint/task-evaluation/production-inputs/",
                                                         "s3://blueprint/task-evaluation/host-only-owner-sources/"]}}
    if public_scene_enabled:
        # public_source_binding_root + machinery_path let the scene-progression
        # _source resolver bind a public_scene intent. The public-scene machinery
        # (task_evaluation_public_scene_machinery.v1) is release/provider bound and
        # is materialized per scene by the public-scene provisioner, not here — the
        # worker only reads machinery_path once a public_scene intent exists.
        config["public_source_binding_root"] = str(public_binding_root)
        config["machinery_path"] = str(config_root / "task-evaluation-public-scene-machinery.json")
    if bool(bootstrap.get("activation_authorized")):
        # A3: the separately-admitted activation on-ramp. The production progression
        # timer (blueprint-task-evaluation-scene-progression.service) execs
        # `task_evaluation_scene_progression --config <THIS file>` directly, so
        # activation is enabled by setting activation_enabled True on the ONE config
        # that service reads -- a typed, separately-admitted mode gated by the
        # owner's --activation-authorized, not a second wrapper pass. _advance_intent
        # then provisions the scene-configuration activation intent (no allocation)
        # the configured-controls worker consumes. project_spend_current_path is
        # where the project-spend monitor (capacity/funding) publishes the fresh
        # reconciliation _activation requires; activation fails closed on a
        # stale/absent pointer (never allocates). Off by default -> preparation-only.
        # R4: the ONE canonical activation-intent registry -- producer-writable
        # (scene-progression.service ReadWritePaths cover /var/lib/blueprint, not
        # /etc/blueprint) and consumer-readable, and DISTINCT from the activation
        # materialization root. The configured-controls consumer unit + preflight +
        # activation automation DEFAULT_INTENT_ROOT all point at this same path.
        activation_intent_root = state / "task-evaluation-scene-configuration-activation-intents"
        if not activation_intent_root.exists():
            activation_intent_root.mkdir(parents=True, mode=0o750)
            if os.geteuid() == 0:
                os.chown(activation_intent_root, account.pw_uid, account.pw_gid)
        config["activation_enabled"] = True
        config["activation_intent_root"] = str(activation_intent_root)
        config["project_spend_current_path"] = str(state / "scene-project-spend" / "current.json")
    if project_spend_seed is not None:
        # The monitor is a retained-input reader and conservative reservation
        # publisher. It never calls a provider. Keep its seed reference digest
        # bound in the installer-managed config, and expose only the config path
        # through the shared service environment.
        monitor = {"schema_version": PROJECT_SPEND_MONITOR_SCHEMA,
            "managed_by": MANAGED_BY,
            "scene_root": str(directories[0]),
            "seed_reconciliation_path": str(seed_path),
            "seed_reconciliation_reference": dict(project_spend_seed),
            "output_root": str(project_spend_root / "outputs"),
            "current_path": str(project_spend_root / "current.json")}
        monitor["config_digest"] = canonical_digest(monitor, digest_field="config_digest")
        monitor_path = project_spend_root / "monitor.json"
        _managed_json(monitor_path, monitor, account)
        config["project_spend_monitor_config_path"] = str(monitor_path)
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    config_path = config_root / "task-evaluation-scene-progression.json"
    _managed_json(config_path, config, account)
    environment = {"BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT": config["intent_root"],
        "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS": "blueprint-webapp",
        "BLUEPRINT_TASK_EVALUATION_OWNER_SOURCE_STORE_ROOT": str(inputs / "owner-source-store"),
        "PIPELINE_CAPTURE_INTAKE_STORE_ROOT": bootstrap["capture_store_root"]}
    if project_spend_seed is not None:
        environment["BLUEPRINT_SCENE_PROJECT_SPEND_CONFIG"] = config["project_spend_monitor_config_path"]
    require(all(not any(c.isspace() for c in value) for value in environment.values()), "scene_preparation_environment_path_invalid")
    content = "# Managed by " + MANAGED_BY + ".\n" + "".join(f"{key}={value}\n" for key, value in sorted(environment.items()))
    env_path = safe_path(config_root / "task-evaluation-scene-progression.env")
    if env_path.exists():
        require(env_path.read_text().startswith("# Managed by " + MANAGED_BY + ".\n"), "scene_preparation_unmanaged_environment")
    if not env_path.exists() or env_path.read_text() != content:
        with tempfile.NamedTemporaryFile(mode="w", dir=config_root, prefix=".scene-preparation-env-", delete=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(0o640)
        if os.geteuid() == 0:
            os.chown(temporary, 0, account.pw_gid)
        os.replace(temporary, env_path)
    result = {"schema_version": "task_evaluation_scene_preparation_installation.v1", "status": "installed",
        "bootstrap": record(bootstrap_path), "config": record(config_path), "environment": record(env_path),
        "machinery": record(machinery_path), "execution_activation_enabled": False,
        "provider_mutation_performed": False, "service_start_requested": False}
    if project_spend_seed is not None:
        result["project_spend_monitor"] = record(config["project_spend_monitor_config_path"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--destination-simready")
    parser.add_argument("--destination-alias", action="append", default=[])
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--public-scene-enabled", action="store_true",
                        help="Admit rights-admitted public-scene persistent intents (Spec A: the "
                             "legacy public-scene path, e.g. 841757) in addition to owner uploads.")
    parser.add_argument("--activation-authorized", action="store_true",
                        help="Separately admit the scene-configuration activation on-ramp (A3): the "
                             "service runs a second no-spend pass that provisions activation intents. "
                             "Off by default keeps the service preparation-only. Never arms paid dispatch.")
    parser.add_argument("--project-spend-reconciliation",
                        help="Digest-bound retained project-spend reconciliation used by the no-spend monitor.")
    args = parser.parse_args(argv)
    if args.destination_simready:
        path = safe_path(args.destination_simready)
        identity = read(path, digest_field="result_digest")["destination_identity"]
        bootstrap = build_bootstrap(destination_catalog=[{"binding_id": identity["id"] + "-" + identity["version"],
            "owner_description_aliases": args.destination_alias, "simready_result": record(path)}],
            public_scene_enabled=args.public_scene_enabled, activation_authorized=args.activation_authorized,
            project_spend_reconciliation_path=args.project_spend_reconciliation)
        _managed_json(safe_path(args.bootstrap), bootstrap, pwd.getpwnam(bootstrap["service_account"]))
    result = install_scene_preparation(bootstrap_path=args.bootstrap) if args.install else {"bootstrap": record(args.bootstrap)}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
