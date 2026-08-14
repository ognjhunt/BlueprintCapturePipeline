"""The SAM 3.1 admission writes a terminal allocator result, so it must seal one.

`prepare_sam31_gpu_canary` writes `adapter_output`, and `adapter_output` is
literally the `terminal_contract.result_path` of the live profile that makes
`semantic-sam31-source-tracks` website-reachable --
`{launch_run_root}/allocator/result.json`. That profile's `required_path_fields`
are read straight off that file, so an admission written there without
`artifact_manifest_path` and `teardown_manifest_path` ends
`allocator_terminal_artifact_missing:` for both, whatever happened on the
provider. That is the failure that cost a paid run on 2026-08-13.

Today the paid lane happens to overwrite that file with a sealed result on the
branches it reaches, so the defect is masked rather than absent: the module's own
terminal write does not satisfy the contract it writes into, and every rescue
depends on a caller ordering nothing pins. These tests pin the file, by the
fields it emits rather than by any helper name.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.sam31_gpu_admission import prepare_sam31_gpu_canary
from blueprint_pipeline.task_evaluation_live_profile import shared_control_surface
from tests.test_sam31_gpu_admission import COMMIT, _preflight, _request


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_BUILDER = REPO_ROOT / "scripts" / "build_sam31_source_tracks_live_profile.py"

#: The launch profile's terminal contract also asks for this, and only a
#: completed execute can name it -- the paid lane sets it from the canary's own
#: normalized output. An admission that never reached a provider must not invent
#: one, so it is the single required field this module cannot seal.
EXECUTE_ONLY_TERMINAL_FIELD = "source_track_import_result_path"


def _prepare(attempt_root: Path) -> dict:
    """Run one admission whose terminal result lands under `attempt_root`."""

    attempt_root.mkdir(parents=True, exist_ok=True)
    request_path = attempt_root / "request.json"
    preflight_path = attempt_root / "preflight.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")
    preflight = _preflight()
    preflight["observed_at_epoch"] = time.time()
    preflight["preflight_digest"] = canonical_digest(
        preflight, digest_field="preflight_digest"
    )
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    prepare_sam31_gpu_canary(
        request_path=request_path,
        preflight_path=preflight_path,
        admission_out=attempt_root / "admission.json",
        bound_request_out=attempt_root / "bound-request.json",
        adapter_output=attempt_root / "result.json",
        provider="vast",
        expected_source_commit=COMMIT,
        checkout_source_commit=COMMIT,
        checkout_clean=True,
        max_spend_usd=1.0,
        hard_ttl_seconds=600,
        retry_cap=0,
        authority_id="design-partner-beta-authorization",
        execute=False,
    )
    return json.loads((attempt_root / "result.json").read_text(encoding="utf-8"))


def _write_retained_evidence(attempt_root: Path) -> Path:
    """Lay out the evidence a real attempt leaves under its allocator root."""

    provider_run = attempt_root / "vast_provider_run"
    provider_run.mkdir(parents=True, exist_ok=True)
    (provider_run / "vast_provider_adapter_result.json").write_text(
        json.dumps({"status": "blocked", "provider_mutations_performed": 0}),
        encoding="utf-8",
    )
    teardown = provider_run / "vast_teardown_manifest.json"
    teardown.write_text(
        json.dumps({"continuing_spend_from_this_run": False}), encoding="utf-8"
    )
    return teardown


def _builder_additional_path_fields() -> tuple[str, ...]:
    """The extra terminal fields this builder adds, read from its own source.

    They are named in the `additional_required_path_fields=` keyword it hands
    to `shared_control_surface`, either as literals or through a module
    constant, so both spellings are resolved here.
    """

    tree = ast.parse(PROFILE_BUILDER.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value.value, str)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "additional_required_path_fields" or not isinstance(
                keyword.value, (ast.Tuple, ast.List)
            ):
                continue
            return tuple(
                item.value
                if isinstance(item, ast.Constant)
                else constants[item.id]
                for item in keyword.value.elts
                if (isinstance(item, ast.Constant) and isinstance(item.value, str))
                or (isinstance(item, ast.Name) and item.id in constants)
            )
    return ()


def _required_path_fields() -> list[str]:
    """Read the live profile's terminal contract from where the builder gets it.

    This used to scrape a `required_path_fields` literal out of the builder.
    There is no longer one to scrape: the builder takes the whole control
    surface from `shared_control_surface` and names only the single field it
    adds. Reading the shared definition plus that addition keeps this test
    pinned to the contract a launch actually reads, rather than to a copy of it
    that drifts the moment the lane is refactored -- which is exactly what
    happened here.
    """

    source = PROFILE_BUILDER.read_text(encoding="utf-8")
    assert "shared_control_surface" in source, (
        f"{PROFILE_BUILDER.name} no longer takes its control surface from the "
        "shared definition, so this test is reading a contract the launch does "
        "not use"
    )
    surface = shared_control_surface(
        additional_required_path_fields=_builder_additional_path_fields()
    )
    return list(surface["terminal_contract"]["required_path_fields"])


def test_the_terminal_result_names_both_manifest_fields(tmp_path: Path) -> None:
    """The file the launch contract reads must name the fields it reads."""

    terminal = _prepare(tmp_path / "allocator")

    assert "artifact_manifest_path" in terminal
    assert "teardown_manifest_path" in terminal


def test_the_terminal_result_names_the_evidence_a_real_attempt_retained(
    tmp_path: Path,
) -> None:
    """A terminal result written over a root that holds retained evidence must
    point at it, rather than leave it on disk with nothing naming it."""

    attempt_root = tmp_path / "allocator"
    attempt_root.mkdir(parents=True)
    teardown = _write_retained_evidence(attempt_root)

    terminal = _prepare(attempt_root)

    assert terminal["teardown_manifest_path"] == str(teardown.resolve())
    manifest_path = Path(terminal["artifact_manifest_path"])
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["binding"]["allocator_lane"] == "semantic_sam31_source_tracks"
    assert {"allocator_adapter_result", "teardown_manifest"}.issubset(
        set(manifest["observed_roles"])
    )


def test_the_seal_uses_the_root_the_lane_writes_its_provider_run_under(
    tmp_path: Path,
) -> None:
    """The #501 precedent: a lane can seal a root its evidence is not under.

    The paid lane derives `attempt_root` from the adapter result's own parent --
    `_write_terminal_result` builds `<adapter_output>.parent/vast_provider_run` --
    so the admission has to seal that same directory. Evidence sitting at the
    launch run root must stay unclaimed rather than be reported as this
    attempt's.
    """

    run_root = tmp_path / "run"
    attempt_root = run_root / "allocator"
    attempt_root.mkdir(parents=True)
    _write_retained_evidence(attempt_root)
    # A decoy one level up, which is where a sloppier seal would look.
    _write_retained_evidence(run_root)

    terminal = _prepare(attempt_root)

    assert terminal["artifact_manifest_path"] == str(
        (attempt_root / "artifact_manifest.json").resolve()
    )
    assert not (run_root / "artifact_manifest.json").exists()
    assert terminal["teardown_manifest_path"] == str(
        (attempt_root / "vast_provider_run" / "vast_teardown_manifest.json").resolve()
    )


def test_a_teardown_path_naming_a_file_nobody_wrote_stays_null(tmp_path: Path) -> None:
    """A dry run that never reached a provider has nothing to inventory, and a
    path to a file nobody wrote reads as evidence gone missing rather than
    evidence that was never produced."""

    terminal = _prepare(tmp_path / "allocator")

    assert terminal["teardown_manifest_path"] is None
    assert terminal["artifact_manifest_path"] is None
    assert terminal["status"] == "dry_run_ready"
    assert terminal["blockers"] == []


def test_the_module_seals_every_manifest_field_its_launch_contract_requires() -> None:
    """Read the requirement off the profile builder, not out of memory."""

    required = set(_required_path_fields())
    assert required, "the sam31 live profile declares no required path fields"

    terminal_source = (
        REPO_ROOT / "src" / "blueprint_pipeline" / "sam31_gpu_admission.py"
    ).read_text(encoding="utf-8")
    unsealed = {field for field in required if f'"{field}"' not in terminal_source}

    assert unsealed == {EXECUTE_ONLY_TERMINAL_FIELD}, (
        "the sam31 admission writes the launch profile's terminal result but does "
        f"not name every field that contract reads off it: {sorted(unsealed)}"
    )


def test_the_admission_receipt_keeps_its_own_digest(tmp_path: Path) -> None:
    """Sealing the terminal result must not rewrite the admission receipt.

    `admission.json` is digest-bound evidence of what was admitted; the terminal
    result is a different artifact that the launch contract reads. Folding seal
    fields into the receipt would break its self-digest, which is the thing that
    makes it evidence.
    """

    attempt_root = tmp_path / "allocator"
    _prepare(attempt_root)

    admission = json.loads((attempt_root / "admission.json").read_text(encoding="utf-8"))
    assert admission["admission_digest"] == canonical_digest(
        admission, digest_field="admission_digest"
    )
    assert "artifact_manifest_path" not in admission
