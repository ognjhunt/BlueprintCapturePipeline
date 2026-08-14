"""A file may not claim a schema whose producer could never have emitted it.

`docs/arm_decision_proof_v1/manifests/adp009d_840313_runtime_readiness.v1.json`
declared `task_evaluation_runtime_readiness.v1` and carried three blockers --
`exact_adp009d_runtime_adapter_not_on_protected_main`,
`scripted_positive_control_not_passed`, and
`allocator_artifact_manifest_not_emitted` -- that no code in `src/` or
`scripts/` can produce. The module that owns that schema,
`adp009d_live_readiness`, emits blockers named `live_readiness_*` and an
`observations` block of six keys; the file had three.

The franka lane's dry launch profile pins that file by digest and copies its
blockers into `execution_admission`, so a reader of the deployed profile saw
`allocator_artifact_manifest_not_emitted` and reasonably took it for a
measurement of the allocator. It never was one. The allocator's behaviour for
that lane has never been measured, because measuring it needs a completed paid
allocator result, which a dry-only lane does not produce.

Digest-binding cannot catch this: the file's `receipt_digest` is
self-consistent. It was sealed, just never produced. So the contract is
producibility, not integrity -- and both the blocker vocabulary and the
observation key set are rediscovered from the producer's source, so extending
the producer cannot leave a manifest checked against a stale hand-written list.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "blueprint_pipeline"
DOCS = REPO_ROOT / "docs"

READINESS_SCHEMA = "task_evaluation_runtime_readiness.v1"


def _producer_module(schema: str) -> Path:
    """The module that declares this schema as its own SCHEMA_VERSION."""

    owners = []
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "SCHEMA_VERSION"
                    and isinstance(node.value, ast.Constant)
                    and node.value.value == schema
                ):
                    owners.append(path)
    assert len(owners) == 1, f"expected exactly one producer for {schema}, got {owners}"
    return owners[0]


def _emittable_blockers(module: Path) -> set[str]:
    """Every blocker literal the producer can append, read from its source."""

    tree = ast.parse(module.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "add"}
            and isinstance(node.func.value, ast.Name)
            and "blocker" in node.func.value.id.lower()
        ):
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    found.add(argument.value)
    assert found, f"no blocker literals discovered in {module}"
    return found


def _observation_keys(module: Path) -> set[str]:
    """The keys of the `observations` dict the producer builds."""

    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "observations"
                and isinstance(value, ast.Dict)
            ):
                return {
                    inner.value
                    for inner in value.keys
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
                }
    raise AssertionError(f"no observations dict literal found in {module}")


def _checked_in_manifests(schema: str) -> list[Path]:
    matches = []
    for path in DOCS.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version") == schema:
            matches.append(path)
    return matches


def test_a_checked_in_readiness_manifest_uses_only_its_producers_blockers() -> None:
    """Iterated rather than parametrized: an empty case set must not become a skip.

    Parametrizing over "files declaring this schema" collects nothing once the
    last such file is fixed, and pytest reports that as a SKIP. The CPU full
    lane blocks on any skip, so a contract that had simply run out of work
    turned the whole lane red. Looping keeps the check a real pass.
    """

    producer = _producer_module(READINESS_SCHEMA)
    vocabulary = _emittable_blockers(producer)
    for manifest_path in _checked_in_manifests(READINESS_SCHEMA):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        orphans = sorted(
            blocker
            for blocker in payload.get("blockers", [])
            if isinstance(blocker, str) and blocker not in vocabulary
        )

        assert not orphans, (
            f"{manifest_path.relative_to(REPO_ROOT)} declares {READINESS_SCHEMA} "
            f"but carries blockers {producer.name} cannot emit: {orphans}. A "
            "blocker no code produces is an assertion someone typed."
        )


def test_a_checked_in_readiness_manifest_reports_every_observation() -> None:
    """A partial observation block reads as a full one and hides what was never checked."""

    producer = _producer_module(READINESS_SCHEMA)
    expected = _observation_keys(producer)
    for manifest_path in _checked_in_manifests(READINESS_SCHEMA):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = set(payload.get("observations") or {})

        assert actual == expected, (
            f"{manifest_path.relative_to(REPO_ROOT)} declares {READINESS_SCHEMA} "
            f"but its observations differ from what {producer.name} emits. "
            f"Missing: {sorted(expected - actual)}. "
            f"Unknown: {sorted(actual - expected)}."
        )


def test_the_dry_lane_placeholder_does_not_claim_a_producer_owned_schema() -> None:
    """The franka dry profile admits on declared preconditions, not measurements.

    Keeping the placeholder out of the producer's schema is what stops
    `allocator_artifact_manifest_not_emitted` reappearing in
    `execution_admission` where a reader takes it for an allocator observation.
    """

    placeholder = (
        DOCS / "arm_decision_proof_v1" / "manifests"
        / "adp009d_840313_runtime_readiness.v1.json"
    )
    payload = json.loads(placeholder.read_text(encoding="utf-8"))

    # The blocker strings stay verbatim: they are the frozen evaluation-run
    # spec's own vocabulary, and rewriting a sealed artifact that defines the
    # run to improve three names is not a trade worth making. What changes is
    # the document's claim about itself.
    assert payload["schema_version"] != READINESS_SCHEMA
    assert payload["claim_ceiling"] == "declared_precondition_not_measured"
    assert payload["measurement_source"] is None
    # Never measured is not the same as measured false.
    assert all(value is None for value in payload["observations"].values())
    # The lane's safety posture is unchanged by saying so.
    assert payload["status"] == "blocked"
    assert payload["live_execution_enabled"] is False


def test_the_dry_builder_pins_the_placeholder_schema() -> None:
    """A digest alone did not catch a sealed-but-unproduced file; the schema must be pinned."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_adp009d_840313_launch_profile",
        REPO_ROOT / "scripts" / "build_adp009d_840313_launch_profile.py",
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)

    placeholder = json.loads(
        (
            DOCS / "arm_decision_proof_v1" / "manifests"
            / "adp009d_840313_runtime_readiness.v1.json"
        ).read_text(encoding="utf-8")
    )
    assert builder.EXPECTED_READINESS_SCHEMA == placeholder["schema_version"]
    assert builder.EXPECTED_READINESS_DIGEST == placeholder["receipt_digest"]


def test_the_producibility_contract_still_detects_an_unproducible_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, the contract rots into a no-op the day the last case is fixed.

    The two parametrized checks above draw their cases from the tree, so once no
    checked-in file declares the producer's schema they collect nothing and pass
    by collecting nothing. This proves the mechanism -- producer discovery,
    vocabulary extraction, and the comparison -- still rejects the exact file
    shape that was shipped.
    """

    producer = _producer_module(READINESS_SCHEMA)
    vocabulary = _emittable_blockers(producer)
    observations = _observation_keys(producer)

    # The vocabulary is real and is the one the module actually owns.
    assert "live_readiness_artifact_manifest_invalid" in vocabulary
    assert all(name.startswith("live_readiness_") for name in vocabulary)
    assert "allocator_artifact_manifest_emitted" in observations
    assert len(observations) == 6

    # The shape that shipped, verbatim, is still rejected by both rules.
    shipped_blockers = [
        "exact_adp009d_runtime_adapter_not_on_protected_main",
        "scripted_positive_control_not_passed",
        "allocator_artifact_manifest_not_emitted",
    ]
    assert [b for b in shipped_blockers if b not in vocabulary] == shipped_blockers
    shipped_observations = {
        "exact_runtime_adapter_on_protected_main",
        "scripted_positive_control_passed",
        "allocator_artifact_manifest_emitted",
    }
    assert shipped_observations != observations


def test_every_readiness_schema_has_exactly_one_producer() -> None:
    """Two producers for one schema would make "producible" ambiguous."""

    assert _producer_module(READINESS_SCHEMA).name == "adp009d_live_readiness.py"
