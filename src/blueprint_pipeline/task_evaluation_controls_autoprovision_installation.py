"""Provision the controls-autoprovision configuration, with execution off.

One operator-owned bootstrap binds the retained Franka USD / camera template /
runtime payload catalog and the owner-store roots.  Deployment recompiles it
into a validated service-owned config plus a service-owned ``EnvironmentFile``;
scenes and release SHAs are never hand-edited into the worker configuration.

Materializing the config *before* its environment pointer is the whole point.
``task_evaluation_controls_autoprovision.progression_owner_scope`` fails closed
when ``BLUEPRINT_TASK_EVALUATION_CONTROLS_AUTOPROVISION_CONFIG`` names a file it
cannot read or parse -- ``unresolved`` becomes true and the progression worker
returns a blocked report, stopping *every* scene.  So this installer writes the
sealed content catalog and the config first, validates the retained assets by
running the real ``resolve_robot_catalog`` resolver, and only then writes the
``EnvironmentFile`` that exports the config path.  The stored catalog is the
release-independent content-catalog schema, so the worker re-binds it to the
active deployed release each tick without this installer mutating source bytes.

This module neither starts units nor submits a run; the installed configuration
cannot activate a paid execution.
"""
from __future__ import annotations

import argparse
import grp
import json
import os
from pathlib import Path
import pwd
import tempfile

from . import task_evaluation_controls_autoprovision as worker
from . import task_evaluation_scene_intake as intake
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_public_scene_attempt_factory import record
from .task_evaluation_scene_configuration_submission_inputs import read
from .task_evaluation_scene_progression_state import atomic_json, require, safe_path

BOOTSTRAP_SCHEMA = "task_evaluation_controls_autoprovision_bootstrap.v1"
CONFIG_SCHEMA = "task_evaluation_controls_autoprovision_config.v1"
INSTALLATION_SCHEMA = "task_evaluation_controls_autoprovision_installation.v1"
MANAGED_BY = "blueprint_pipeline.task_evaluation_controls_autoprovision_installation"
CONFIG_ENV = worker.CONFIG_ENV

DEFAULT_BOOTSTRAP = "/etc/blueprint/task-evaluation-controls-autoprovision-bootstrap.json"
DEFAULT_CONFIG = "/etc/blueprint/task-evaluation-controls-autoprovision.json"
DEFAULT_CATALOG = "/etc/blueprint/task-evaluation-controls-robot-catalog.json"
DEFAULT_ENVIRONMENT = "/etc/blueprint/task-evaluation-controls-autoprovision.env"

#: The canonical registry both the configuration-activation and the controls
#: progression consumer read (``continuation_provisioning.DEFAULT_INTENT_ROOT``).
#: The service worker installs write-once ``0440`` intents here, so the directory
#: itself must be group-writable under the real systemd sandbox.
DEFAULT_INTENT_ROOT = "/etc/blueprint/task-evaluation-configured-controls-intents"
DEFAULT_CONTROLS_ROOT = "/var/lib/blueprint/task-evaluation-inputs/controls-autoprovision"
DEFAULT_PROFILE_DIR = "/etc/blueprint/task-evaluation-launch-profiles"
DEFAULT_SCENE_ROOT = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents"
DEFAULT_PREPARATION_QUEUE_ROOT = (
    "/var/lib/blueprint/pipeline-control-plane/task-evaluation-owned-scene-preparations"
)

#: Placeholder commit used only to run the real asset/runtime validation at
#: install time.  The stored catalog stays the release-independent content
#: schema and the worker re-binds it to the active release each tick.
_VALIDATION_COMMIT = "0" * 40

#: Fields ``task_evaluation_controls_autoprovision.process_config`` reads.
_CONFIG_KEYS = (
    "scene_root", "preparation_queue_root", "controls_root", "intent_root",
    "profile_dir", "robot_catalog_path", "trusted_clients", "service_group",
)


def validate_robot_bindings(bindings, *, service_account="blueprint"):
    """Seal a release-independent content catalog from retained embodiment bytes.

    Runs the production ``resolve_robot_catalog`` resolver so the exact robot
    USD, camera template and runtime payload are re-hashed and refused on any
    changed byte; also requires the placement-credential and spend-pointer
    fields the provisioning path reopens later, so the operator learns about a
    missing binding field at install time rather than at the first tick.
    """
    require(isinstance(bindings, dict) and 1 <= len(bindings) <= 32,
            "controls_autoprovision_bindings_invalid")
    for name, row in bindings.items():
        require(intake._identifier(name) and isinstance(row, dict)
                and "expected_production_commit" not in row,
                "controls_autoprovision_binding_identity_invalid")
        require(isinstance(row.get("openai_project_id"), str) and row["openai_project_id"]
                and isinstance(row.get("openai_api_key_id"), str) and row["openai_api_key_id"],
                "controls_autoprovision_binding_credentials_invalid")
        require(bool(row.get("project_spend_current_path"))
                or ("project_spend_reconciliation" in row and "project_spend_observed_at_epoch" in row),
                "controls_autoprovision_binding_spend_pointer_invalid")
    content = worker._seal(
        {"schema_version": worker.CONTENT_CATALOG_SCHEMA, "managed_by": MANAGED_BY,
         "bindings": bindings},
        "catalog_digest",
    )
    # Real resolver: hashes every asset and the runtime payload and refuses a
    # mismatch.  The resolved result is discarded; the content catalog is stored
    # exactly as sealed here, so its ``catalog_digest`` covers the stored bytes.
    worker.resolve_robot_catalog(content, source_commit=_VALIDATION_COMMIT)
    return content


def build_bootstrap(*, robot_catalog_bindings, scene_root=DEFAULT_SCENE_ROOT,
                    preparation_queue_root=DEFAULT_PREPARATION_QUEUE_ROOT,
                    controls_root=DEFAULT_CONTROLS_ROOT, intent_root=DEFAULT_INTENT_ROOT,
                    profile_dir=DEFAULT_PROFILE_DIR, config_root="/etc/blueprint",
                    trusted_clients=("blueprint-webapp",), service_account="blueprint",
                    service_group="blueprint"):
    """Seal an operator bootstrap after validating the retained embodiment bytes."""
    validate_robot_bindings(robot_catalog_bindings, service_account=service_account)
    require(isinstance(trusted_clients, (list, tuple)) and trusted_clients
            and all(isinstance(client, str) and client for client in trusted_clients),
            "controls_autoprovision_trusted_clients_invalid")
    roots = {key: str(safe_path(value)) for key, value in {
        "config_root": config_root, "scene_root": scene_root,
        "preparation_queue_root": preparation_queue_root, "controls_root": controls_root,
        "intent_root": intent_root, "profile_dir": profile_dir}.items()}
    pwd.getpwnam(service_account)
    grp.getgrnam(service_group)
    value = {"schema_version": BOOTSTRAP_SCHEMA, "managed_by": MANAGED_BY, **roots,
             "robot_catalog_bindings": robot_catalog_bindings,
             "trusted_clients": list(trusted_clients), "service_account": service_account,
             "service_group": service_group, "execution_activation_enabled": False}
    value["bootstrap_digest"] = canonical_digest(value, digest_field="bootstrap_digest")
    return value


def _managed_json(path, value, group, *, mode=0o640):
    path = safe_path(path)
    unchanged = False
    if path.exists():
        old = read(path)
        require(old.get("managed_by") == MANAGED_BY, "controls_autoprovision_unmanaged_file")
        unchanged = old == value
    if not unchanged:
        atomic_json(path, value)
    if os.geteuid() == 0:
        os.chown(path, 0, group.gr_gid)
    path.chmod(mode)


def _managed_environment(path, environment, group):
    path = safe_path(path)
    require(all(not any(c.isspace() for c in str(v)) for v in environment.values()),
            "controls_autoprovision_environment_value_invalid")
    header = "# Managed by " + MANAGED_BY + ".\n"
    content = header + "".join(f"{key}={value}\n" for key, value in sorted(environment.items()))
    if path.exists():
        require(path.read_text().startswith(header), "controls_autoprovision_unmanaged_environment")
    if not path.exists() or path.read_text() != content:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=".controls-autoprovision-env-", delete=False) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        temporary.chmod(0o640)
        if os.geteuid() == 0:
            os.chown(temporary, 0, group.gr_gid)
        os.replace(temporary, path)
    return path


def _ensure_service_directory(path, account, group):
    path = safe_path(path)
    if not path.exists():
        path.mkdir(parents=True, mode=0o750)
    if os.geteuid() == 0:
        os.chown(path, account.pw_uid, group.gr_gid)
    path.chmod(0o750)
    require(path.is_dir() and not path.is_symlink(), "controls_autoprovision_directory_invalid")
    return path


def _ensure_registry_directory(path, group):
    """The registry the service worker installs immutable ``0440`` intents into.

    The directory is group-writable so the ``blueprint`` service account can add
    write-once, read-back-verified intents; the entries themselves stay ``0440``.
    """
    path = safe_path(path)
    if not path.exists():
        path.mkdir(parents=True, mode=0o770)
    if os.geteuid() == 0:
        os.chown(path, 0, group.gr_gid)
    path.chmod(0o770)
    require(path.is_dir() and not path.is_symlink(), "controls_autoprovision_intent_root_invalid")
    return path


def install_controls_autoprovision(*, bootstrap_path):
    """Materialize the validated config, sealed catalog and env pointer, idempotently."""
    bootstrap_path = safe_path(bootstrap_path)
    require(not bootstrap_path.stat().st_mode & 0o022, "controls_autoprovision_bootstrap_writable")
    bootstrap = read(bootstrap_path, digest_field="bootstrap_digest")
    require(bootstrap.get("schema_version") == BOOTSTRAP_SCHEMA
            and bootstrap.get("managed_by") == MANAGED_BY
            and bootstrap.get("execution_activation_enabled") is False,
            "controls_autoprovision_bootstrap_scope_invalid")
    account = pwd.getpwnam(bootstrap["service_account"])
    group = grp.getgrnam(bootstrap.get("service_group") or "blueprint")

    # Validate the retained embodiment bytes and build the release-independent
    # content catalog before touching any host path.
    content = validate_robot_bindings(bootstrap["robot_catalog_bindings"],
                                      service_account=bootstrap["service_account"])

    config_root = safe_path(bootstrap["config_root"])
    scene_root = safe_path(bootstrap["scene_root"])
    preparation_queue_root = safe_path(bootstrap["preparation_queue_root"])
    controls_root = safe_path(bootstrap["controls_root"])
    intent_root = safe_path(bootstrap["intent_root"])
    profile_dir = safe_path(bootstrap["profile_dir"])
    # The owner store and profile registry are provisioned by their own
    # installers; require them rather than minting an empty store here.
    require(scene_root.is_dir() and not scene_root.is_symlink(),
            "controls_autoprovision_scene_root_missing")
    require(preparation_queue_root.is_dir() and not preparation_queue_root.is_symlink(),
            "controls_autoprovision_preparation_queue_root_missing")
    require(profile_dir.is_dir() and not profile_dir.is_symlink(),
            "controls_autoprovision_profile_dir_missing")
    _ensure_service_directory(controls_root, account, group)
    _ensure_registry_directory(intent_root, group)

    # 1) Sealed content catalog (read-only, service-readable).  ``content`` is
    # sealed with ``managed_by`` included, so its ``catalog_digest`` matches the
    # stored bytes and ``_managed_json`` can still detect operator conflicts.
    catalog_path = config_root / Path(DEFAULT_CATALOG).name
    _managed_json(catalog_path, content, group)

    # 2) Config (service-readable) -- written before the env pointer.
    config = {"schema_version": CONFIG_SCHEMA, "managed_by": MANAGED_BY,
              "scene_root": str(scene_root), "preparation_queue_root": str(preparation_queue_root),
              "controls_root": str(controls_root), "intent_root": str(intent_root),
              "profile_dir": str(profile_dir), "robot_catalog_path": str(catalog_path),
              "trusted_clients": list(bootstrap["trusted_clients"]),
              "service_group": bootstrap.get("service_group") or "blueprint",
              "execution_activation_enabled": False}
    require(all(config.get(key) is not None for key in _CONFIG_KEYS),
            "controls_autoprovision_config_incomplete")
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    config_path = config_root / Path(DEFAULT_CONFIG).name
    _managed_json(config_path, config, group)

    # 3) The env pointer -- only now that a readable, valid config exists.
    env_path = _managed_environment(config_root / Path(DEFAULT_ENVIRONMENT).name,
                                    {CONFIG_ENV: str(config_path)}, group)

    return {"schema_version": INSTALLATION_SCHEMA, "status": "installed",
            "bootstrap": record(bootstrap_path), "config": record(config_path),
            "catalog": record(catalog_path), "environment": record(env_path),
            "intent_root": str(intent_root), "controls_root": str(controls_root),
            "config_env": CONFIG_ENV, "execution_activation_enabled": False,
            "provider_mutation_performed": False, "service_start_requested": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args(argv)
    if args.install:
        result = install_controls_autoprovision(bootstrap_path=args.bootstrap)
    else:
        result = {"bootstrap": record(safe_path(args.bootstrap))}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
