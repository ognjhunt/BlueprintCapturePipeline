"""Release superseded team-evaluation holds before any paid placement work.

CPU inputs remain immutable. An inference reservation, an agent output, or a
native plan keeps the entire hold. The active release's holds are never retired.
"""
from pathlib import Path
from typing import Any, Mapping

from .configured_scene_run_identity import scoped_identity
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_retained_controls_evidence import DIRECTORY, _file, _read

SCHEMA = "task_evaluation_unused_team_adoption_cancellation.v1"
FIELDS = ("attempt_id", "attempt_digest", "intent_digest", "provider", "maximum_spend_usd")


def require(value: Any, reason: str) -> None:
    if not value:
        raise ValueError("unused_team_holds_" + reason)


def unpaid_paths(binding: Path, intent_digest: str) -> list[Path]:
    from .task_evaluation_configured_controls_autostart_validation import RESULT_SCHEMA_VERSION
    token = intent_digest.removeprefix("sha256:")[:16]
    return [binding / ("agent-placement-attempts-" + token),
            binding / "agent-official-openai-cost" / ("agent-placement-attempts-" + token),
            binding / ("agent-placement-checkpoint-" + token + ".v1.json"),
            binding / (RESULT_SCHEMA_VERSION + "-" + token + ".json")]


def is_unpaid(binding: Path, intent_digest: str) -> bool:
    paths = unpaid_paths(binding, intent_digest)
    if any(p.is_symlink() for p in (binding, *binding.parents)):
        return False
    for index, root in enumerate(paths):
        if any(p.is_symlink() for p in (root, *root.parents)):
            return False
        if not root.exists():
            continue
        if not root.is_dir():
            return False
        for path in root.rglob("*"):
            if path.is_symlink():
                return False
            if path.is_dir():
                continue
            # A failed credential gate can acquire/release the scope lock;
            # neither receipt reserves money or calls the model.
            if index == 1 and path.name in {
                "openai_scope_lock_acquired.v1.json", "openai_scope_lock_released.v1.json"
            }:
                continue
            return False
    return True


def validate(*, receipt: Mapping[str, Any], attempt: Mapping[str, Any]) -> None:
    ref = receipt.get("source_adoption_intent") or {}
    old_path = Path(str(ref.get("path") or ""))
    old = _read(old_path)
    require(receipt.get("schema_version") == SCHEMA
        and receipt.get("status") == "cancelled_before_paid_materialization"
        and receipt.get("receipt_digest") == canonical_digest(receipt, digest_field="receipt_digest")
        and all(receipt.get(k) == attempt.get(k) for k in FIELDS)
        and _file(old_path) == ref
        and old.get("intent_digest") == canonical_digest(old, digest_field="intent_digest")
        and old.get("expected_production_commit") == attempt.get("source_commit")
        and receipt.get("provider_mutation_performed") is False
        and receipt.get("cpu_artifacts_preserved") is True, "receipt_invalid")
    current_ref = receipt["superseding_adoption_intent"]
    current = _read(Path(current_ref["path"]))
    require(_file(Path(current_ref["path"])) == current_ref
        and current.get("intent_digest") == canonical_digest(current, digest_field="intent_digest")
        and current["expected_production_commit"] != old["expected_production_commit"]
        and current.get("evaluation_authority") is not None
        and old.get("evaluation_authority") == current["evaluation_authority"]
        and old.get("evaluation_run_id") == current.get("evaluation_run_id")
        and old.get("configuration_adoption") == current.get("configuration_adoption"), "source_changed")
    owner = _read(Path(old["phases"]["construction"]["authorization_path"]))["scene_owner_attempt"]["scene_attempt_binding"]
    stem = owner["attempt_id"].removesuffix("-construction")
    require(attempt["attempt_id"] in {stem + "-" + p for p in ("construction", "controls", "placement")}
        and all(owner[k] == attempt[k] for k in ("intent_id", "intent_digest", "source_commit", "input_digest", "runtime_digest"))
        and current["evaluation_authority"]["scene_intent_digest"] == attempt["intent_digest"], "owner_changed")
    binding = Path(receipt["binding_root"])
    require(binding.is_absolute()
        and binding.name == scoped_identity("cpu-robot-binding", old["evaluation_run_id"])
        and binding.parent.name == old["evaluation_authority"]["source_launch_id"]
        and is_unpaid(binding, old["intent_digest"]), "paid_materialization_present")


def retire(*, config: Mapping[str, Any], intent_id: str, current_intent_path: Path,
           dry_run: bool = False) -> list[dict[str, Any]]:
    from . import task_evaluation_scene_intake as intake
    from .task_evaluation_release_identity import running_release_commit
    from .task_evaluation_retained_controls_evidence import validated_cancellation

    current = _read(current_intent_path)
    require(current.get("intent_digest") == canonical_digest(current, digest_field="intent_digest")
        and current.get("evaluation_authority") is not None, "current_intent_invalid")
    require(dry_run or running_release_commit() == current["expected_production_commit"], "running_release_required")
    directory = Path(config["scene_root"]) / intent_id
    owner = intake._read(directory / "intent.json", "intent_digest")
    require(owner["intent_digest"] == current["evaluation_authority"]["scene_intent_digest"], "owner_changed")
    records = sorted((Path(config["controls_root"]) / "terminal-adoptions" / intent_id).glob("*/terminal_adoption_provisioning.json"))
    retained_current = []
    for record_path in records:
        record = intake._read(record_path, "receipt_digest")
        path = Path(record["provisioning"]["intent_path"])
        if _read(path).get("intent_digest") == current["intent_digest"]:
            retained_current.append(path)
    require(len(retained_current) == 1, "current_adoption_missing")
    current_intent_path = retained_current[0]  # Never bind a mutable registry file.
    state = Path(config.get("progression_root") or str(directory.parent.parent / "task-evaluation-configured-controls"))
    binding = state / current["evaluation_authority"]["source_launch_id"] / scoped_identity("cpu-robot-binding", current["evaluation_run_id"])
    results = []
    with intake._lock(Path(config["scene_root"])):
        for p in records:
            provision = intake._read(p, "receipt_digest")
            old_path = Path(provision["provisioning"]["intent_path"])
            old = _read(old_path)
            if old.get("expected_production_commit") == current["expected_production_commit"]:
                continue
            require(provision["execution_source_commit"] == old.get("expected_production_commit"), "source_changed")
            # Reused placement has no inference reservation. Its native holds
            # belong to completed_placement_adoption.retire_unused_native, which
            # verifies native execution history; this unpaid-authoring check
            # must neither invent a placement hold nor release those GPU holds.
            if old.get("completed_placement_adoption") is not None:
                continue
            if not is_unpaid(binding, old["intent_digest"]):
                continue
            old_owner = _read(Path(old["phases"]["construction"]["authorization_path"]))["scene_owner_attempt"]["scene_attempt_binding"]
            stem = old_owner["attempt_id"].removesuffix("-construction")
            for phase in ("construction", "controls", "placement"):
                attempt = intake._read(directory / "attempts" / (stem + "-" + phase + ".json"), "attempt_digest")
                existing = validated_cancellation(directory, attempt)
                if existing is not None:
                    results.append(existing)
                    continue
                receipt = {"schema_version": SCHEMA, "status": "cancelled_before_paid_materialization",
                    **{k: attempt[k] for k in FIELDS}, "source_adoption_intent": _file(old_path),
                    "superseding_adoption_intent": _file(current_intent_path), "binding_root": str(binding),
                    "cpu_artifacts_preserved": True, "provider_mutation_performed": False}
                receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
                validate(receipt=receipt, attempt=attempt)
                if not dry_run:
                    target = directory / DIRECTORY / (attempt["attempt_id"] + ".json")
                    target.parent.mkdir(mode=0o750, exist_ok=True)
                    intake.write_exclusive(target, receipt)
                results.append(receipt)
    return results
