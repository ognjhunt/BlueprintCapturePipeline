"""Every paid lane must seal the terminal artifacts its profile asks it for.

A production launch profile's terminal contract reads two fields off the
allocator result, `artifact_manifest_path` and `teardown_manifest_path`. Eight
paid lanes emitted neither or only the second, so each of them ended
`allocator_terminal_artifact_missing:` regardless of what happened on the
provider -- "all controls passed" was unreachable by construction, and the
evidence they had retained sat on disk with nothing pointing at it.

Fixing that lane by lane leaves the next lane to rediscover it. This is the test
that makes the shared seal the default: a lane added tomorrow either calls it or
fails here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"

#: Modules that run a paid provider attempt and return an allocator result.
#: Adding a lane here is the point: it is cheaper than discovering at the paid
#: boundary that its evidence was never inventoried.
PAID_LANE_MODULES = (
    "adp009d_aura_native_vast.py",
    "adp009d_franka_vast.py",
    "adp009d_ovrtx_vast.py",
    "adp_aura_author_smoke_vast.py",
    "adp_aura_interiorgs_vast.py",
    "adp_content_agents_vast.py",
    "adp_gaussian_excision_vast.py",
    "adp_inpaint360_interiorgs_vast.py",
    "adp_joint_agent_vast.py",
    "adp_retained_scene_render_vast.py",
    "native_task_arena_vast.py",
    "paired_target_native_import_vast.py",
    "public_scene_aura_exact_residual_vast.py",
    "public_scene_artifixer3d_vast.py",
    "public_scene_simready_isaac_vast.py",
    "sam31_paid_resource_allocator_lane.py",
    "semantic_teacher_image_edit_vast.py",
    "simpler_public_vast.py",
)

#: The shared Arena transport seals on behalf of every lane that runs through
#: it, so delegating to it satisfies the contract as completely as calling the
#: seal directly -- and is how the Franka lane already satisfies it.
DELEGATING_ENTRYPOINTS = ("run_arena_native_control_vast",)

#: These build their manifest directly with the shared builder rather than the
#: convenience seal, which satisfies the same contract.
DIRECT_BUILDER_MODULES = (
    "adp_isaac_lab_arena_vast.py",
    "adp_retained_scene_render_vast.py",
)


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@pytest.mark.parametrize("module", PAID_LANE_MODULES)
def test_a_paid_lane_seals_its_terminal_artifacts(module: str) -> None:
    path = SOURCE_ROOT / module
    assert path.is_file(), f"paid lane module missing: {module}"
    calls = _calls(path)

    sealed = "seal_lane_terminal_artifacts" in calls
    built = "build_task_evaluation_artifact_manifest" in calls
    delegated = any(entrypoint in calls for entrypoint in DELEGATING_ENTRYPOINTS)

    assert sealed or built or delegated, (
        f"{module} returns an allocator result without sealing "
        "artifact_manifest_path/teardown_manifest_path. Call "
        "seal_lane_terminal_artifacts(result, attempt_root=job, lane=...) before "
        "writing the terminal result, or build the manifest directly."
    )


@pytest.mark.parametrize("module", DIRECT_BUILDER_MODULES)
def test_a_direct_builder_lane_uses_the_shared_manifest(module: str) -> None:
    """One schema, so a validator that knows one lane can check another."""

    path = SOURCE_ROOT / module
    assert "build_task_evaluation_artifact_manifest" in _calls(path)


def test_the_lane_list_is_the_set_of_modules_that_run_a_paid_attempt() -> None:
    """The list is only a gate if it stays complete.

    A module that runs a provider attempt is identifiable: it takes a paid
    resource admission grant. Anything matching that and absent from the list
    would slip through the parametrised check above without this.
    """

    discovered = {
        path.name
        for path in sorted(SOURCE_ROOT.glob("*_vast.py"))
        if "paid_resource_admission_grant" in path.read_text(encoding="utf-8")
    }
    known = set(PAID_LANE_MODULES) | set(DIRECT_BUILDER_MODULES)

    assert discovered <= known, (
        "these modules run a paid attempt but are not covered by the terminal "
        f"artifact contract: {sorted(discovered - known)}"
    )


def test_the_seal_leaves_a_dry_run_alone(tmp_path) -> None:
    """A dry run never reaches a provider, so it has no terminal artifacts to
    seal and must not be made to look like a live attempt that lost them."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    result = {"status": "dry_run_ready", "blockers": ["b", "a"]}

    sealed = seal_lane_terminal_artifacts(
        result, attempt_root=tmp_path, lane="adp_example"
    )

    assert sealed == result
    # Blocker order is the lane's own, not re-sorted behind its back.
    assert sealed["blockers"] == ["b", "a"]


def test_the_seal_names_both_artifacts_after_a_real_attempt(tmp_path) -> None:
    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    provider_run = tmp_path / "vast_provider_run"
    provider_run.mkdir()
    (provider_run / "vast_provider_adapter_result.json").write_text("{}", encoding="utf-8")
    (provider_run / "vast_teardown_manifest.json").write_text(
        '{"continuing_spend_from_this_run": false}', encoding="utf-8"
    )
    (tmp_path / "immutable_execution").mkdir()
    (tmp_path / "immutable_execution" / "result.json").write_text("{}", encoding="utf-8")

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "blockers": []},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["status"] == "completed"
    assert sealed["artifact_manifest_path"] == str(tmp_path / "artifact_manifest.json")
    assert sealed["teardown_manifest_path"] == str(
        provider_run / "vast_teardown_manifest.json"
    )
    assert (tmp_path / "artifact_manifest.json").is_file()


def test_an_attempt_that_retained_nothing_is_blocked_by_its_own_manifest(tmp_path) -> None:
    """An attempt that reached a provider and kept no evidence must not report a
    terminal status implying it did."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    (tmp_path / "vast_provider_run").mkdir()

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "blockers": []},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["status"] == "blocked"
    assert sealed["blockers"]


def test_no_paid_lane_reads_a_bundle_path_off_an_unresolved_receipt() -> None:
    """A receipt records the absolute paths of the machine that built it.

    A lane that loads one and reads `bundle_path` straight off it looks for the
    archive where it was authored rather than where it is, so a bundle staged
    host-resident is still reported as a binding failure. Six lanes did this
    after #464 fixed the seventh.
    """

    allocator = (SOURCE_ROOT / "paid_resource_allocator.py").read_text(encoding="utf-8")

    assert "prepared_bundle = _load(receipt_path)" not in allocator, (
        "a lane loads its bundle receipt without resolving it against this host; "
        "use _host_resident_bundle(receipt_path, blockers) so the receipt's "
        "bundle_path is the one that resolved here"
    )
    assert "_host_resident_bundle" in allocator


def test_the_content_agents_preflight_resolves_its_bundle_host_side() -> None:
    """The preflight is a local dry-run receipt, so it has to be producible
    where the bundle now lives. Reading  off the receipt sent it
    looking on the machine that built the bundle."""

    preflight = (
        SOURCE_ROOT / "adp_content_agents_bundle_preflight.py"
    ).read_text(encoding="utf-8")

    assert "bundle_receipt = _read_json(receipt_path)" not in preflight
    assert "_host_resident_receipt" in preflight


def test_a_teardown_path_naming_a_file_nobody_wrote_is_nulled(tmp_path) -> None:
    """Lanes name their teardown path unconditionally as
    `<provider_run>/vast_teardown_manifest.json`. A run that failed before that
    directory existed then reports a path to a file that was never written,
    which reads as teardown evidence gone missing rather than teardown that
    never happened. Observed on the 2026-08-13 content-agents attempt."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    claimed = tmp_path / "vast_provider_run" / "vast_teardown_manifest.json"

    sealed = seal_lane_terminal_artifacts(
        {"status": "blocked", "teardown_manifest_path": str(claimed)},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["teardown_manifest_path"] is None


def test_a_real_teardown_path_survives(tmp_path) -> None:
    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    provider_run = tmp_path / "vast_provider_run"
    provider_run.mkdir()
    teardown = provider_run / "vast_teardown_manifest.json"
    teardown.write_text('{"continuing_spend_from_this_run": false}', encoding="utf-8")
    (provider_run / "vast_provider_adapter_result.json").write_text("{}", encoding="utf-8")
    (tmp_path / "immutable_execution").mkdir()
    (tmp_path / "immutable_execution" / "r.json").write_text("{}", encoding="utf-8")

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "teardown_manifest_path": str(teardown)},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["teardown_manifest_path"] == str(teardown)


def test_a_lane_records_why_a_caught_failure_happened() -> None:
    """Ten lanes recorded `..._failed:{type(exc).__name__}` and discarded the
    message, so a blocked run said `ValueError` and nothing about which value.
    Diagnosing that costs another attempt, and a paid one when the failure only
    happens on the execute path."""

    import ast

    for module in PAID_LANE_MODULES:
        source = (SOURCE_ROOT / module).read_text(encoding="utf-8")
        assert "{type(exc).__name__}" not in source, (
            f"{module} records a caught failure by type alone; use "
            "redacted_failure_detail(exc) so the receipt carries the cause"
        )
        ast.parse(source)


def test_a_failure_detail_carries_the_cause_without_a_secret() -> None:
    from blueprint_pipeline.common import redacted_failure_detail

    assert redacted_failure_detail(ValueError("invalid_vast_instance_label_prefix")) == (
        "ValueError:invalid_vast_instance_label_prefix"
    )
    # A receipt is exactly the artifact that gets copied around later.
    signed = redacted_failure_detail(
        RuntimeError("PUT https://o.example/x?X-Amz-Signature=deadbeefcafe failed")
    )
    assert "X-Amz-Signature" not in signed and "<redacted>" in signed
    assert redacted_failure_detail(RuntimeError("")) == "RuntimeError"


def test_every_live_profile_builder_can_distinguish_a_rebuild() -> None:
    """Published profiles are immutable, so a profile whose inputs changed
    needs its own id rather than a conflicting rewrite.

    Inputs change at a fixed commit more often than they look like they would:
    regenerating a config preflight is enough. Without a discriminator the
    collision surfaces as `launch_profile_immutable_input_digest_mismatch` on a
    later launch rather than at publish time, which is a wasted attempt and a
    confusing one.
    """

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    builders = sorted(scripts.glob("build_*_live_profile.py"))
    assert builders, "no live profile builders found"

    for builder in builders:
        source = builder.read_text(encoding="utf-8")
        assert '"--revision"' in source, (
            f"{builder.name} cannot distinguish a rebuilt profile at the same "
            "commit; add a --revision discriminator"
        )


def test_no_module_resolves_a_credential_only_from_a_developer_home() -> None:
    """Control-plane units run with `ProtectHome=true` and home `/nonexistent`.

    A module that resolves a credential only from `~/.blueprint-secrets` can
    never read one on the deployed host. PR #449 fixed that for provider keys,
    #488 for the config preflight, and it was still live in the content-agents
    lane -- where it surfaced as `adp_content_agents_openai_secret_missing`
    raised *after* paid admission had been granted, which is the worst place to
    discover a missing credential.

    A module may still name the developer path as a last resort; it may not be
    the only thing it consults.
    """

    # Only resolution counts. An argparse `default="~/.blueprint-secrets/..."`
    # is a developer convenience the units override with an explicit path; a
    # `Path("~/.blueprint-secrets/...")` is the module deciding for itself
    # where a secret lives.
    resolves_from_home = re.compile(r'Path\(\s*f?"~/\.blueprint-secrets/')
    home_only: list[str] = []
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not resolves_from_home.search(source):
            continue
        honours_deployment = (
            "_read_secret" in source
            or "_FILE" in source
            or "PROVIDER_SECRETS_DIR" in source
            or "SECRETS_DIR" in source
        )
        if not honours_deployment:
            home_only.append(path.name)

    assert not home_only, (
        "these resolve a credential only from a developer home, which is "
        f"unreadable under ProtectHome=true: {home_only}"
    )
