"""Adopt a $0 scene preparation made by an earlier, retained release (dark launch).

A deploy used to retire every website preparation in flight and rebuild it on
the new release, because every later stage compares the commit the preparation
names with the commit it runs. When the host switch below is on, the new
release adopts the preparation instead, if all of these hold:

* the code that produces a preparation is byte-identical between the two
  releases (``contract_identity``, recorded with the preparation when it is
  made; a preparation made before this module existed records none and is
  never adopted);
* its inputs are unchanged (intent digest, source binding digest);
* the earlier release is still retained (its release record and runtime
  publications verify; the activation also needs its toolchain tree);
* no paid reservation has been made for it yet.

Progression writes one sealed adoption record per preparation request and
adopting commit under ``<preparation queue>/release-adoptions``. Only the
commit-name comparisons of the preparation worker, the activation staging and
the activation worker accept a request that names the earlier commit, and only
when that record exists, the switch is on and their own code has the same
contract identity. Every other comparison those checks make stays exact.
Records name both commits: the preparation keeps the commit that made it, and
``adopting_commit`` / ``*_worker_source_commit`` name the release that ran on.

Switch off (the default) changes nothing: ``adoption_for`` returns None and
every caller keeps today's refusal.
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest

SWITCH_ENV = "BLUEPRINT_RELEASE_PREPARATION_ADOPTION"
SWITCH_ON = "enabled"
SCHEMA = "release_preparation_adoption.v1"
DIRECTORY = "release-adoptions"
#: Modules that write a website preparation's records: the factory, the
#: submission and construction producers, and the preparation worker.
PRODUCER_MODULES = (
    "website_scene_dispatch",
    "website_native_submission",
    "website_native_background",
    "website_native_inputs",
    "task_evaluation_launch_preparation_worker",
    "release_preparation_adoption",
)
_PACKAGE = Path(__file__).resolve().parent
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return (os.environ if environ is None else environ).get(SWITCH_ENV, "").strip() == SWITCH_ON


def _imports(source: str) -> list[tuple[str, bool]]:
    from .provider_runtime_import_closure import _import_targets
    return _import_targets(source)


def producer_closure(package_dir: str | Path = _PACKAGE, roots: tuple[str, ...] = PRODUCER_MODULES) -> list[str]:
    """Relative paths of the preparation's producer code.

    The roots, every module a root imports (also inside a function), then the
    module-level import closure of all of those. A sub-package that is reached
    is included whole.
    """
    package = Path(package_dir)
    start = set(roots)
    for name in roots:
        path = package / (name + ".py")
        if path.is_file():
            start.update(target for target, _lazy in _imports(path.read_text(encoding="utf-8")))
    seen: set[str] = set()
    files: set[str] = set()
    stack = sorted(start)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        path = package / (name + ".py")
        if path.is_file():
            files.add(path.name)
            try:
                targets = _imports(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            stack.extend(target for target, lazy in targets if not lazy)
        elif (package / name).is_dir():
            files.update(str(item.relative_to(package)) for item in (package / name).rglob("*.py"))
    return sorted(files)


@lru_cache(maxsize=4)
def _identity(package_dir: str, roots: tuple[str, ...]) -> str:
    package = Path(package_dir)
    rows = [{"path": name, "sha256": hashlib.sha256((package / name).read_bytes()).hexdigest()}
            for name in producer_closure(package, roots)]
    return canonical_digest({"schema_version": "release_preparation_contract.v1", "files": rows})


def contract_identity(package_dir: str | Path = _PACKAGE, roots: tuple[str, ...] = PRODUCER_MODULES) -> str:
    """Digest over the source bytes of the preparation's producer code (see ``producer_closure``)."""
    return _identity(str(Path(package_dir).resolve()), tuple(roots))


def record_path(queue_root: str | Path, request_digest: str, adopting_commit: str) -> Path:
    if _DIGEST.fullmatch(str(request_digest)) is None or _COMMIT.fullmatch(str(adopting_commit)) is None:
        raise ValueError("release_adoption_identity_invalid")
    return Path(queue_root) / DIRECTORY / f"{request_digest[7:]}-{adopting_commit}.json"


def record_adoption(*, queue_root: str | Path, request_digest: str, preparation_id: str, prepared_commit: str,
                    adopting_commit: str, identity: str, intent_id: str, attempt_id: str) -> dict[str, Any]:
    """Write (once) the sealed record that lets the adopting release's workers accept the request."""
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_scene_configuration_submission_inputs import read
    if prepared_commit == adopting_commit or _COMMIT.fullmatch(str(prepared_commit)) is None:
        raise ValueError("release_adoption_identity_invalid")
    value = {"schema_version": SCHEMA, "request_digest": request_digest, "preparation_id": preparation_id,
             "prepared_commit": prepared_commit, "adopting_commit": adopting_commit,
             "contract_identity": identity, "intent_id": intent_id, "attempt_id": attempt_id,
             "spend_usd": 0, "paid_authority_granted": False}
    value["record_digest"] = canonical_digest(value, digest_field="record_digest")
    path = record_path(queue_root, request_digest, adopting_commit)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        intake.write_exclusive(path, value)
    except FileExistsError:
        pass
    if read(path, digest_field="record_digest") != value:
        raise ValueError("release_adoption_record_conflict")
    return value


def adoption_for(*, queue_root: str | Path, request_digest: str, prepared_commit: str, running_commit: str,
                 environ: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """The adoption record that lets ``running_commit`` accept a request naming ``prepared_commit``, or None.

    None whenever the switch is off, the commits are equal, the record is absent
    or invalid, or this process's own producer code has another contract identity.
    """
    from .task_evaluation_scene_configuration_submission_inputs import read
    if not enabled(environ) or prepared_commit == running_commit:
        return None
    try:
        path = record_path(queue_root, request_digest, running_commit)
        if not path.is_file() or path.is_symlink():
            return None
        value = read(path, digest_field="record_digest")
    except (OSError, ValueError):
        return None
    if not (value.get("schema_version") == SCHEMA and value.get("request_digest") == request_digest
            and value.get("prepared_commit") == prepared_commit and value.get("adopting_commit") == running_commit
            and value.get("spend_usd") == 0 and value.get("paid_authority_granted") is False
            and value.get("contract_identity") == contract_identity()):
        return None
    return value


def provenance(record: Mapping[str, Any], *, worker_commit: str) -> dict[str, Any]:
    """What a record made under adoption carries: who prepared, who ran this step."""
    return {"prepared_commit": record["prepared_commit"], "adopting_commit": record["adopting_commit"],
            "worker_source_commit": worker_commit, "contract_identity": record["contract_identity"],
            "adoption_record_digest": record["record_digest"]}


def retained_toolchain_root(current_root: str | Path, *, running_commit: str, prepared_commit: str) -> Path:
    """The earlier release's toolchain tree, beside the running one (``.../scene-configuration/<commit>``)."""
    current = Path(current_root)
    if current.name != running_commit or _COMMIT.fullmatch(prepared_commit) is None:
        raise ValueError("release_adoption_toolchain_layout_unknown")
    retained = current.parent / prepared_commit
    if retained.is_symlink() or not retained.is_dir():
        raise ValueError("release_adoption_toolchain_not_retained")
    return retained


def activation_adoption(request: Mapping[str, Any], preparation_queue_root: str | Path,
                        running_commit: str) -> dict[str, Any] | None:
    """The activation worker's lookup: the adoption of the preparation this activation names."""
    preparation = request.get("preparation") if isinstance(request.get("preparation"), Mapping) else {}
    return adoption_for(queue_root=preparation_queue_root, request_digest=str(preparation.get("request_digest")),
                        prepared_commit=str(request.get("expected_production_commit")),
                        running_commit=running_commit)


def adopted_toolchain(root: str | Path, adoption: Mapping[str, Any] | None) -> Path:
    """The toolchain tree an activation binds: the running one, or the adopted release's retained one."""
    if adoption is None:
        return Path(root)
    return retained_toolchain_root(root, running_commit=adoption["adopting_commit"],
                                   prepared_commit=adoption["prepared_commit"])


def with_adoption(result: dict[str, Any], adoption: Mapping[str, Any] | None, worker_commit: str) -> dict[str, Any]:
    """An activation result made under adoption names both commits; otherwise it is returned unchanged."""
    if adoption is None:
        return result
    value = {**result, "release_adoption": provenance(adoption, worker_commit=worker_commit)}
    value["result_digest"] = canonical_digest(value, digest_field="result_digest")
    return value


def blocked_activation(activation_queue_root: Path, activation_id: str) -> list[str] | None:
    """The blockers of a terminal refusal of this activation, or None while none exists."""
    import json
    results = Path(activation_queue_root) / "results"
    for path in sorted(results.glob("*.json")) if results.is_dir() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (isinstance(value, dict) and value.get("activation_id") == activation_id
                and value.get("status") == "blocked" and value.get("provider_mutation_performed") is False):
            return [str(item)[:200] for item in value.get("blockers") or []][:8] or ["activation_blocked"]
    return None


def request_digest_from_queue_name(name: str) -> str | None:
    """``<preparation_id>-<64 hex>.json`` names every preparation queue and result file."""
    match = re.fullmatch(r".+-([0-9a-f]{64})\.json", name)
    return "sha256:" + match.group(1) if match else None


__all__ = ["SWITCH_ENV", "SWITCH_ON", "activation_adoption", "adopted_toolchain", "adoption_for",
           "blocked_activation", "with_adoption", "contract_identity", "enabled", "producer_closure",
           "provenance", "record_adoption", "retained_toolchain_root"]
