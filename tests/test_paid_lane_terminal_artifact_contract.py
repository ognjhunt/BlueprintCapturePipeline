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

A gate is only as wide as the set it looks at, though. This one discovered lanes
with ``*_vast.py`` and so never saw `reconstruction_vast_worker_smoke.py`, which
went on to rent a GPU, pass its healthcheck, tear down with provider zero
confirmed, and still report `allocator_terminal_artifact_missing:` -- with
nothing here red. Widening to ``*_vast*.py`` surfaced seven more lanes running a
paid attempt. Each is now accounted for: sealed here, sealed by the wrapper that
writes its result, or named as debt below with a reason that the tests keep
honest.
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
    # Not a `*_vast*.py` transport: it is the fresh-site camera lane's own
    # runner, and #543 gave `new-site-native-camera` a builder, so the launch
    # profile now reads the terminal result it writes.
    "nvidia_warehouse_native_camera_gpu_admission.py",
    # The `*_runpod*.py` half of the provider families. Dual-purpose: the
    # frozen `openpi-policy-ranking` campaign and the live
    # `new-site-diagnostic-canary` #544 made reachable share this runner.
    "openpi_policy_ranking_runpod.py",
    "paired_target_native_import_vast.py",
    "public_scene_aura_exact_residual_vast.py",
    "public_scene_artifixer3d_vast.py",
    "public_scene_simready_isaac_vast.py",
    "reconstruction_vast_worker_smoke.py",
    # Not a `*_vast*.py` transport at all: it writes the launch profile's
    # terminal `result.json` on every non-execute-ready path, so it is a
    # terminal write into the same contract and is verified as one.
    "sam31_gpu_admission.py",
    "sam31_paid_resource_allocator_lane.py",
    "semantic_teacher_image_edit_vast.py",
    "simpler_public_vast.py",
)

#: Lanes whose result is sealed by the allocator wrapper that writes it to the
#: adapter output, rather than by the lane itself. That is a legitimate way to
#: meet the contract -- the profile reads the wrapper's file, not the lane's --
#: but only while the named wrapper really does seal, so the test below checks
#: both halves rather than taking the claim on trust.
SEALED_BY_ALLOCATOR_CALLER = {
    "sam31_vast_source_track_canary.py": "sam31_paid_resource_allocator_lane.py",
}

#: Lanes that run a paid attempt and seal nothing, with why. These all reach the
#: canonical allocator's single unsealed ``write_json(adapter_path, result)``,
#: so each would report `allocator_terminal_artifact_missing:` on a live launch
#: exactly as `reconstruction_vast_worker_smoke` did.
#:
#: This is a ledger of known debt, not a suppression list: an entry here is an
#: admission that the lane is not launchable, not an exemption from fixing it.
#: The tests below refuse a stale entry, refuse an entry that has quietly
#: started sealing, and cap the total so the ledger cannot absorb a new lane.
UNSEALED_LANE_DEBT = {
    "groot_oscar_runpod_canary.py": "frozen_program",
    "groot_oscar_runpod_persistent_carrier_campaign.py": "frozen_program",
    "groot_oscar_runpod_s3_model_cache.py": "frozen_program",
    "measurement_chrono_dem_vast_canary.py": "shares_unsealed_canonical_allocator_write",
    "measurement_dlo_lab_vast_canary.py": "shares_unsealed_canonical_allocator_write",
    "measurement_isaac_vast_canary.py": "shares_unsealed_canonical_allocator_write",
    "reconstruction_isaac_vast_operation.py": "shares_unsealed_canonical_allocator_write",
    "reconstruction_vast_operation.py": "shares_unsealed_canonical_allocator_write",
    "single_g1_kitchen_episode_runpod.py": "frozen_program",
    "unitree_groot_n17_sonic_vast_persistent_session.py": "frozen_program",
    "unitree_unifolm_runpod_server.py": "frozen_program",
}

VALID_DEBT_REASONS = {"shares_unsealed_canonical_allocator_write", "frozen_program"}

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


def _discovered_paid_lanes() -> set[str]:
    """Every module that runs a provider attempt, read from the tree.

    A module that runs one is identifiable: it takes a paid resource admission
    grant. The name pattern is the other half, and it used to be ``*_vast.py``
    -- which missed `reconstruction_vast_worker_smoke.py` entirely, so that
    lane reached a live GPU, passed its healthcheck, tore down cleanly, and was
    still reported `allocator_terminal_artifact_missing:` with nothing in this
    suite red. It is not the only lane named that way, and the next one would
    have slipped through the same gap.

    Widening to ``*_vast*.py`` fixed the pattern and left the *provider* half
    of the hole open: Vast is not the only provider, and every ``*_runpod*.py``
    lane was invisible here for the same reason. Six of them take a grant, and
    one -- `openpi_policy_ranking_runpod.py` -- is the paid execution path for
    `new-site-diagnostic-canary`, which #544 made website-reachable. A gate
    that only knows one provider's naming will keep rediscovering this, so both
    families are globbed and a lane on a third provider is expected to be added
    here rather than to a debt row.
    """

    return {
        path.name
        for pattern in ("*_vast*.py", "*_runpod*.py")
        for path in sorted(SOURCE_ROOT.glob(pattern))
        if "paid_resource_admission_grant" in path.read_text(encoding="utf-8")
    }


def _accounted_for() -> set[str]:
    return (
        set(PAID_LANE_MODULES)
        | set(DIRECT_BUILDER_MODULES)
        | set(SEALED_BY_ALLOCATOR_CALLER)
        | set(UNSEALED_LANE_DEBT)
    )


def test_the_lane_list_is_the_set_of_modules_that_run_a_paid_attempt() -> None:
    """The list is only a gate if it stays complete.

    Anything that runs a paid attempt and is absent from every list here would
    slip through the parametrised check above without this.
    """

    discovered = _discovered_paid_lanes()
    known = _accounted_for()

    assert discovered <= known, (
        "these modules run a paid attempt but are not covered by the terminal "
        f"artifact contract: {sorted(discovered - known)}"
    )


def test_the_widened_discovery_still_sees_the_lanes_it_used_to() -> None:
    """A pattern that stopped matching would empty the gate silently."""

    discovered = _discovered_paid_lanes()

    assert "reconstruction_vast_worker_smoke.py" in discovered
    assert {module for module in PAID_LANE_MODULES if module.endswith("_vast.py")} <= discovered
    # The runpod family, invisible here until a live canary depended on it.
    assert "openpi_policy_ranking_runpod.py" in discovered
    assert sum(1 for module in discovered if "_runpod" in module) >= 6


def test_a_lane_sealed_by_its_caller_really_is_sealed_by_that_caller() -> None:
    """Naming a sealer is a claim, and an unchecked claim is how a lane ends up
    covered by a wrapper that stopped sealing two refactors ago."""

    for lane, sealer in sorted(SEALED_BY_ALLOCATOR_CALLER.items()):
        path = SOURCE_ROOT / sealer
        assert path.is_file(), f"{lane} names a sealer that does not exist: {sealer}"
        source = path.read_text(encoding="utf-8")
        assert lane.removesuffix(".py") in source, (
            f"{sealer} is named as the sealer for {lane} but does not import it"
        )
        assert "seal_lane_terminal_artifacts" in _calls(path), (
            f"{sealer} is named as the sealer for {lane} and seals nothing"
        )


def test_no_named_gap_outlives_the_lane_it_describes() -> None:
    """An entry for a module that no longer runs a paid attempt is stale, and a
    stale entry reads as coverage."""

    discovered = _discovered_paid_lanes()
    stale = sorted((set(SEALED_BY_ALLOCATOR_CALLER) | set(UNSEALED_LANE_DEBT)) - discovered)

    assert not stale, f"these are named here but run no paid attempt: {stale}"


def test_a_lane_that_started_sealing_is_promoted_rather_than_left_as_debt() -> None:
    """Otherwise the ledger keeps reporting debt that has already been paid,
    and the parametrised seal check never guards the lane."""

    promoted = sorted(
        module
        for module in UNSEALED_LANE_DEBT
        if "seal_lane_terminal_artifacts" in _calls(SOURCE_ROOT / module)
    )

    assert not promoted, (
        f"{promoted} now seal their terminal artifacts; move them into "
        "PAID_LANE_MODULES rather than leaving a stale debt entry."
    )


def test_the_unsealed_lane_debt_is_stated_rather_than_implied() -> None:
    """Every entry is a lane that cannot pass its own launch profile. Keeping
    the size visible is what stops the ledger becoming the place new lanes go."""

    assert set(UNSEALED_LANE_DEBT.values()) <= VALID_DEBT_REASONS, (
        f"unrecognized debt reasons: {sorted(set(UNSEALED_LANE_DEBT.values()) - VALID_DEBT_REASONS)}"
    )

    # Split, because one total hid the distinction that matters. A
    # `frozen_program` lane is fenced by the reachability contract and cannot be
    # triggered from the website at all; an unfenced one is a lane a launch can
    # actually reach and then fail to report on. Widening discovery to the
    # runpod family added six frozen rows at once -- lanes that had been running
    # paid attempts unsealed the whole time, simply invisible to a gate that
    # only knew Vast's naming -- and a single total would have read that as the
    # ledger doubling.
    launchable = sorted(
        module
        for module, reason in UNSEALED_LANE_DEBT.items()
        if reason != "frozen_program"
    )

    assert len(launchable) <= 5, (
        f"unsealed lanes that are not fenced by the frozen-program contract "
        f"grew to {len(launchable)}: {launchable}. These are reachable and "
        "cannot report completed after paying for a GPU. Lower this bound as "
        "lanes are sealed; do not raise it to make it pass."
    )
    assert len(UNSEALED_LANE_DEBT) <= 11, (
        f"unsealed paid lanes grew to {len(UNSEALED_LANE_DEBT)}: "
        f"{sorted(UNSEALED_LANE_DEBT)}. Lower this bound as lanes are sealed; "
        "raise it only when discovery widens to a provider family that was "
        "never checked, and say so."
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
    #
    # A *function* default is also the module deciding. `adp_joint_agent_vast`
    # took `secret_root: str | Path = "~/.blueprint-secrets"` and read files
    # under it, and this pattern missed it entirely -- the lane then failed on
    # a rented-adjacent boundary with `model_api_key_missing`, the same defect
    # #492 had just fixed in three siblings. A default is not a convenience
    # when nothing overrides it.
    resolves_from_home = re.compile(
        r'Path\(\s*f?"~/\.blueprint-secrets/'
        r'|secret_root[^=\n]*=\s*"~/\.blueprint-secrets"'
    )
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


def test_a_lane_that_reached_a_provider_cannot_seal_an_empty_root(tmp_path) -> None:
    """The defect the first live SimReady run exposed.

    It rented a GPU, ran the probe, tore the instance down with a 200 from the
    provider API, and reported `completed` with `blockers: []` -- while its own
    terminal contract went unmet, because it sealed the job root and its
    provider run lives under `attempts/attempt_001/`. The sealer found nothing
    and said nothing.
    """

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "blockers": [], "estimated_cost_usd": 0.065277},
        attempt_root=tmp_path,
        lane="contract_probe",
    )

    assert sealed["status"] == "blocked"
    assert any(
        item.startswith("terminal_artifacts_not_found_under_attempt_root:")
        for item in sealed["blockers"]
    )


def test_a_dry_run_that_never_reached_a_provider_still_seals_quietly(tmp_path) -> None:
    """A dry run has nothing to inventory and must not look like a lost one."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    sealed = seal_lane_terminal_artifacts(
        {"status": "dry_run_ready", "blockers": [], "estimated_cost_usd": None},
        attempt_root=tmp_path,
        lane="contract_probe",
    )

    assert sealed["status"] == "dry_run_ready"
    assert sealed["blockers"] == []


def test_every_lane_seals_the_root_its_provider_run_lives_under() -> None:
    """Calling the sealer is not enough; it has to be pointed at the evidence.

    Rediscovered from source: whatever expression a lane builds
    `<root>/vast_provider_run` from is the expression it must pass as
    `attempt_root`.
    """

    mismatched: list[str] = []
    for module in PAID_LANE_MODULES:
        path = SOURCE_ROOT / module
        source = path.read_text(encoding="utf-8")
        if "seal_lane_terminal_artifacts(" not in source:
            continue
        tree = ast.parse(source)
        provider_roots = {
            ast.unparse(node.value.left)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "provider_run" for t in node.targets)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.right, ast.Constant)
            and node.value.right.value == "vast_provider_run"
        }
        if not provider_roots:
            continue
        sealed_roots = {
            ast.unparse(keyword.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "seal_lane_terminal_artifacts"
            for keyword in node.keywords
            if keyword.arg == "attempt_root"
        }
        if sealed_roots and not (sealed_roots & provider_roots):
            mismatched.append(
                f"{module}: seals {sorted(sealed_roots)} but its provider run is "
                f"under {sorted(provider_roots)}"
            )

    assert not mismatched, (
        "these seal a root their terminal evidence is not under, so they report "
        f"a terminal status their artifacts do not support: {mismatched}"
    )


#: Transports the allocator can dispatch to that do NOT seal terminal evidence.
#: Every one is an open defect, not an exemption: a lane here rents a GPU, runs,
#: tears down cleanly, and still reports `allocator_terminal_artifact_missing`.
#: They are recorded so the set cannot grow silently while each is fixed on the
#: paid execution path, which is a change no launch profile can make for it.
#:
#: `sam31_gpu_admission.py` was the urgent one -- `semantic-sam31-source-tracks`
#: is website-reachable today -- and it now seals, so it has moved into
#: `PAID_LANE_MODULES` and the bound below has been ratcheted with it.
#: `nvidia_warehouse_native_camera_gpu_admission.py` followed for the same
#: reason: #543 gave `new-site-native-camera` a builder, which turned a dormant
#: omission into a live one, and it now seals.
#:
#: `openpi_policy_ranking_gpu_admission.py` stays, but read what it is: this
#: module writes no terminal adapter result of its own, and the paid execution
#: for both the frozen campaign and the live `new-site-diagnostic-canary` runs
#: through `openpi_policy_ranking_runpod.py`, which now seals and is checked as
#: a paid lane above. The live exposure is closed; the row remains because this
#: transport still names neither field and the ledger is keyed on the
#: allocator's imports.
TRANSPORTS_MISSING_TERMINAL_EVIDENCE: dict[str, str] = {
    "openpi_policy_ranking_gpu_admission.py": "new-site-diagnostic-canary + frozen openpi",
    "policy_ranking_cosmos_reasoner_gpu_admission.py": "frozen_program",
    "policy_ranking_successor_gpu_admission.py": "frozen_program",
    "reconstruction_gpu_admission.py": "reconstruction-worker-smoke",
    "reconstruction_isaac_vast_operation.py": "reconstruction isaac operation",
    "reconstruction_vast_operation.py": "reconstruction operation",
}


def _allocator_transport_imports() -> set[str]:
    """Every transport module the allocator itself imports, read from its source.

    The list above is hand-written, which is exactly how an entire naming
    family went unchecked: it enumerates `*_vast.py`, and six
    `*_gpu_admission.py` transports -- including one that is website-reachable
    today -- were never in it. A lane outside this contract rents a GPU, runs,
    tears down cleanly, and still reports `allocator_terminal_artifact_missing`,
    which is the failure that cost a paid run on 2026-08-13.

    Rediscovering from the allocator's own imports means a new transport is
    covered the moment the allocator can dispatch to it, whatever it is named.
    """

    source = (SOURCE_ROOT / "paid_resource_allocator.py").read_text(encoding="utf-8")
    found = set()
    for match in re.finditer(r"^from \.([a-z0-9_]+) import", source, re.MULTILINE):
        module = match.group(1)
        if module.endswith(("_vast", "_gpu_admission", "_allocator_lane", "_vast_operation")):
            found.add(f"{module}.py")
    assert found, "no transport imports discovered in the allocator"
    return found


def test_every_transport_the_allocator_imports_is_covered_by_this_contract() -> None:
    classified = (
        set(PAID_LANE_MODULES)
        | set(DIRECT_BUILDER_MODULES)
        | set(TRANSPORTS_MISSING_TERMINAL_EVIDENCE)
    )
    unclassified = sorted(_allocator_transport_imports() - classified)

    assert not unclassified, (
        "the allocator can dispatch to transports this contract never checks: "
        f"{unclassified}. Add each to PAID_LANE_MODULES (so its terminal "
        "evidence is verified), or to NON_PAID_ALLOCATOR_IMPORTS with why it "
        "returns no admissible allocator result. A lane outside this contract "
        "rents a GPU and then cannot report completed."
    )


def test_the_missing_terminal_evidence_set_only_ever_shrinks() -> None:
    """A recorded defect must be a shrinking list, never a growing exemption.

    The bound is ratcheted every time one is fixed, so the room a fix bought
    cannot be spent on the next lane that would rather be excused than sealed.
    """

    assert len(TRANSPORTS_MISSING_TERMINAL_EVIDENCE) <= 6, (
        "a transport was added to TRANSPORTS_MISSING_TERMINAL_EVIDENCE rather "
        "than sealing its terminal evidence. This list records lanes that "
        "cannot report completed after paying for a GPU; it is not an "
        "exemption to widen."
    )


def test_every_recorded_gap_still_lacks_the_evidence_it_claims_to_lack() -> None:
    """Once a transport seals, its entry must go, or the ledger starts lying.

    Checked by the emitted FIELD, never by grepping for the helper that usually
    produces it -- a prior session grepped for `seal_lane_terminal_artifacts`,
    concluded five lanes were broken, and was wrong.
    """

    for module in sorted(TRANSPORTS_MISSING_TERMINAL_EVIDENCE):
        source = (SOURCE_ROOT / module).read_text(encoding="utf-8")
        emits_both = (
            '"artifact_manifest_path"' in source
            and '"teardown_manifest_path"' in source
        )
        assert not emits_both, (
            f"{module} now emits both terminal path fields; remove it from "
            "TRANSPORTS_MISSING_TERMINAL_EVIDENCE and add it to "
            "PAID_LANE_MODULES so it is verified rather than excused."
        )


def test_a_lane_that_rented_one_instance_cannot_seal_nothing_and_stay_completed(
    tmp_path: Path,
) -> None:
    """One rented instance is recorded singular, and that must still count.

    Production regression: the reached-a-provider probe matched only keys
    ending `instance_ids`. Lanes that rent exactly one instance record
    `instance_id`, so a run that really had rented a GPU was judged never to
    have reached a provider. The sealer then took its quiet dry-run path,
    left both manifest paths `None`, and kept the result's `completed` status
    -- the precise silent-wrong-success this probe exists to prevent, and the
    same shape as the SimReady run that tore its instance down correctly and
    sealed nothing.
    """

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    sealed = seal_lane_terminal_artifacts(
        {"status": "completed", "blockers": [], "instance_id": "26104412"},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert sealed["status"] == "blocked"
    assert any(
        blocker.startswith("terminal_artifacts_not_found_under_attempt_root:")
        for blocker in sealed["blockers"]
    ), sealed["blockers"]


def test_a_dry_run_with_no_instance_still_takes_the_quiet_path(tmp_path: Path) -> None:
    """The probe must stay narrow: no rented resource means nothing to seal."""

    from blueprint_pipeline.task_evaluation_artifact_manifest import (
        seal_lane_terminal_artifacts,
    )

    sealed = seal_lane_terminal_artifacts(
        {"status": "blocked", "blockers": ["refused_before_provider"]},
        attempt_root=tmp_path,
        lane="adp_example",
    )

    assert "terminal_artifacts_not_found_under_attempt_root" not in " ".join(
        sealed["blockers"]
    )
