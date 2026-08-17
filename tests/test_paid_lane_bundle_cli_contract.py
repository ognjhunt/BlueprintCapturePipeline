"""A lane that cannot be rebuilt from a command line cannot be launched twice.

The allocator refuses a bundle whose `blueprint_commit` is not the commit the
control plane is running, and every deploy moves that commit. So a lane whose
bundle exists only because someone once called a Python function is launchable
exactly until the next deploy, and then never again.

`adp_gaussian_excision_vast` was in that state: a complete
`build_gaussian_excision_vast_bundle`, no `main()`, and a bundle pinned to a
commit from days earlier. It read as "ready" everywhere and could not be run.

This rediscovers the set from source rather than listing it, so the next lane
added cannot be omitted from a list instead of given an entry point.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: Lanes whose bundle is built by a dedicated script rather than by the lane
#: module itself. Named, because "there is a script somewhere" is exactly the
#: assumption that let a lane sit unbuildable.
BUNDLE_BUILT_BY_SCRIPT = {
    "adp_retained_scene_render_vast.py": "build_retained_scene_render_bundle.py",
    "public_scene_simready_isaac_vast.py": "public_scene_simready_isaac_bundle.py",
}


#: Bundle modules that seal a provider bundle and still have no entry point.
#:
#: Named, not skipped silently. Each is launchable exactly once after the
#: commit its bundle was built at, and then never again -- the same defect
#: #512 fixed in four `*_vast.py` lanes, in modules that scan missed because
#: it only looked at `*_vast.py`.
#:
#: They are listed rather than fixed in one pass because each takes different
#: inputs and an untested CLI is worse than a named gap. Removing a name here
#: means giving that module a `main()`.
BUNDLE_MODULES_WITHOUT_AN_ENTRYPOINT = {
    # Frozen or retired lanes -- no launch path is wanted.
    "public_scene_aura_exact_residual_bundle.py",
    "cosmos_edge_closed_loop_provider_bundle.py",
    # Generic Arena bundle is a shared library, not a launch entry point.
    "native_task_arena_bundle.py",
    # ADP-009D diagnostics and the launch qualification bundle.
    "adp009d_native_microcheck_bundle.py",
    "articulated_isaac_bundle.py",
    "articulated_native_diagnostic_bundle.py",
    "launch_bundle.py",
}


def _bundle_modules() -> list[Path]:
    """Every module that seals a provider bundle, by shape not by name."""

    import ast as _ast

    found: list[Path] = []
    for path in sorted(SOURCE_ROOT.glob("*_bundle.py")):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        if any(
            isinstance(node, _ast.FunctionDef)
            and node.name.startswith("build_")
            and node.name.endswith("_bundle")
            for node in _ast.walk(tree)
        ):
            found.append(path)
    return found


def _paid_lane_modules() -> list[Path]:
    return [
        path
        for path in sorted(SOURCE_ROOT.glob("*_vast.py"))
        if "paid_resource_admission_grant" in path.read_text(encoding="utf-8")
    ]


def _builds_a_bundle(source: str) -> bool:
    """Does this module define the function that seals a provider bundle?"""

    tree = ast.parse(source)
    return any(
        isinstance(node, ast.FunctionDef)
        and node.name.startswith("build_")
        and node.name.endswith(("_bundle", "_vast_bundle"))
        for node in ast.walk(tree)
    )


def _has_entrypoint(source: str) -> bool:
    tree = ast.parse(source)
    has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )
    return has_main and "__main__" in source


def test_there_are_paid_lanes_to_check() -> None:
    """A discovery that matched nothing would make the check below vacuous."""

    assert len(_paid_lane_modules()) >= 10


@pytest.mark.parametrize("path", _paid_lane_modules(), ids=lambda p: p.stem)
def test_a_lane_that_seals_a_bundle_can_be_run_from_a_command_line(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8")
    if path.name in BUNDLE_BUILT_BY_SCRIPT:
        script = SCRIPTS / BUNDLE_BUILT_BY_SCRIPT[path.name]
        alt = SOURCE_ROOT / BUNDLE_BUILT_BY_SCRIPT[path.name]
        assert script.is_file() or alt.is_file(), (
            f"{path.name} is recorded as built by "
            f"{BUNDLE_BUILT_BY_SCRIPT[path.name]}, which does not exist"
        )
        return
    if not _builds_a_bundle(source):
        # This test's premise does not apply to an adapter-only lane. Returning
        # is an explicit pass; the canonical full lane rejects every skip.
        return
    assert _has_entrypoint(source), (
        f"{path.name} seals a provider bundle with no `main()` entry point. The "
        "allocator refuses a bundle built at another commit and every deploy "
        "moves the commit, so this lane is launchable exactly once and then "
        "never again. Give it a CLI, or record it in BUNDLE_BUILT_BY_SCRIPT."
    )


def test_the_gaussian_excision_lane_exposes_both_of_its_steps() -> None:
    """The bundle needs a sealed wheelhouse, and sealing one is its own step."""

    from blueprint_pipeline import adp_gaussian_excision_vast as lane

    assert callable(lane.main)
    assert callable(lane.materialize_gaussian_excision_dependency_wheelhouse)
    assert callable(lane.build_gaussian_excision_vast_bundle)


def test_the_bundle_module_scan_finds_something() -> None:
    """A scan matching nothing would make the check below vacuous."""

    assert len(_bundle_modules()) >= 10


@pytest.mark.parametrize("path", _bundle_modules(), ids=lambda p: p.stem)
def test_a_bundle_module_has_an_entrypoint_or_is_a_named_gap(path: Path) -> None:
    """`*_vast.py` was never the whole set.

    `paired_target_native_import_bundle.py` seals the bundle for the appearance
    path this program bets on, had no `main()`, and its newest receipt read
    `status: ready` while pinned to a commit the host had long since left. The
    lane could not launch and nothing said so, because the earlier scan only
    looked at `*_vast.py`.
    """

    source = path.read_text(encoding="utf-8")
    if _has_entrypoint(source):
        assert path.name not in BUNDLE_MODULES_WITHOUT_AN_ENTRYPOINT, (
            f"{path.name} has an entry point now; remove it from "
            "BUNDLE_MODULES_WITHOUT_AN_ENTRYPOINT"
        )
        return
    assert path.name in BUNDLE_MODULES_WITHOUT_AN_ENTRYPOINT, (
        f"{path.name} seals a provider bundle with no `main()`, so it can be "
        "built exactly once and never rebuilt at a new deployed commit. Give it "
        "a CLI, or add it to BUNDLE_MODULES_WITHOUT_AN_ENTRYPOINT with a reason."
    )


#: Authority materializers with no command line. Named, not skipped.
#:
#: `public_scene_aura_exact_residual_vast` is retired: no launch profile will
#: be built for it and no attempt will be authorized, so it wants no entry
#: point. Its *historical* receipts remain this campaign's spend anchor, which
#: is a reason to keep the artifacts, not a reason to keep the path.
AUTHORITY_MODULES_WITHOUT_AN_ENTRYPOINT = {
    "public_scene_aura_exact_residual_vast.py",
}


def _authority_materializers() -> list[tuple[Path, str]]:
    """Every module that mints a paid attempt authority, by shape not by name."""

    import ast as _ast

    found: list[tuple[Path, str]] = []
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in _ast.walk(tree):
            if (
                isinstance(node, _ast.FunctionDef)
                and node.name.startswith("materialize_")
                and node.name.endswith("_paid_attempt_authority")
            ):
                found.append((path, node.name))
                break
    return found


def test_authority_materializers_are_discoverable() -> None:
    assert len(_authority_materializers()) >= 2


@pytest.mark.parametrize(
    "path,function", _authority_materializers(), ids=lambda v: getattr(v, "stem", "")
)
def test_an_authority_can_be_minted_from_a_command_line(path: Path, function: str) -> None:
    """A third scope for the same defect.

    #512 covered lanes that seal a bundle and #520 covered bundle modules. A
    module that mints an *authority* escaped both, so the ArtiFixer3D campaign
    was authorizable only from a Python session -- which is not a production
    path, and left the appearance chain unlaunchable for a reason nothing
    reported.
    """

    reachable = any(
        function in candidate.read_text(encoding="utf-8")
        for candidate in SCRIPTS.glob("*.py")
    )
    if reachable:
        assert path.name not in AUTHORITY_MODULES_WITHOUT_AN_ENTRYPOINT, (
            f"{path.name} is reachable now; remove it from "
            "AUTHORITY_MODULES_WITHOUT_AN_ENTRYPOINT"
        )
        return
    assert path.name in AUTHORITY_MODULES_WITHOUT_AN_ENTRYPOINT, (
        f"{path.name} mints a paid attempt authority that no script can call, so "
        "its lane cannot be authorized from a production path. Give it a CLI, or "
        "add it to AUTHORITY_MODULES_WITHOUT_AN_ENTRYPOINT with a reason."
    )


#: Builder parameters a bundle CLI deliberately does not offer, and why.
#: Reasons are load-bearing: `generated_at` stamps the receipt and defaulting
#: it to now is the intended behaviour, whereas a parameter that selects *what
#: gets built* is never safe to fix at its default.
CLI_PARAMETERS_DELIBERATELY_NOT_OFFERED: dict[str, str] = {
    "ctrl_world_provider_bundle.generated_at": "receipt timestamp defaults to now",
    "oscar_wam_provider_bundle.generated_at": "receipt timestamp defaults to now",
    "public_scene_simready_isaac_bundle.generated_at": "receipt timestamp defaults to now",
}


def _bundle_cli_calls() -> list[tuple[str, str, set[str], set[str]]]:
    """(module, builder, builder parameters, parameters the CLI can supply)."""

    import ast as _ast

    rows: list[tuple[str, str, set[str], set[str]]] = []
    for path in sorted(SOURCE_ROOT.glob("*_bundle.py")):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        mains = [n for n in tree.body if isinstance(n, _ast.FunctionDef) and n.name == "main"]
        builders = {
            node.name: {arg.arg for arg in node.args.kwonlyargs}
            for node in tree.body
            if isinstance(node, _ast.FunctionDef) and node.name.startswith("build_")
        }
        if not mains or not builders:
            continue
        for call in _ast.walk(mains[0]):
            if (
                isinstance(call, _ast.Call)
                and isinstance(call.func, _ast.Name)
                and call.func.id in builders
            ):
                supplied = {kw.arg for kw in call.keywords if kw.arg}
                # A `**` spread carries its keys as string literals, which may
                # be inline or in a dict built earlier in the same function.
                # Naming a parameter anywhere in `main` is the wiring.
                if any(kw.arg is None for kw in call.keywords):
                    supplied.update(
                        node.value
                        for node in _ast.walk(mains[0])
                        if isinstance(node, _ast.Constant) and isinstance(node.value, str)
                    )
                rows.append((path.stem, call.func.id, builders[call.func.id], supplied))
    return rows


BUNDLE_CLI_CALLS = _bundle_cli_calls()


def _bundle_cli_case_id(row: tuple[str, str, set[str], set[str]]) -> str:
    """Name a CLI contract without unordered parameter-set representations."""

    module, builder, _, _ = row
    return f"{module}-{builder}"


def test_bundle_cli_calls_are_discoverable() -> None:
    assert len(BUNDLE_CLI_CALLS) >= 5


def test_bundle_cli_case_ids_are_unique_and_hash_seed_independent() -> None:
    ids = [_bundle_cli_case_id(row) for row in BUNDLE_CLI_CALLS]

    assert len(ids) == len(set(ids))
    assert all("{" not in case_id and "}" not in case_id for case_id in ids)


@pytest.mark.parametrize(
    "module,builder,parameters,supplied",
    BUNDLE_CLI_CALLS,
    ids=[_bundle_cli_case_id(row) for row in BUNDLE_CLI_CALLS],
)
def test_a_bundle_cli_can_supply_every_parameter_of_what_it_builds(
    module: str, builder: str, parameters: set[str], supplied: set[str]
) -> None:
    """A flag that does not exist is a decision silently fixed at its default.

    `public_scene_artifixer3d_bundle` offered eight flags for a thirteen
    parameter builder, so its CLI could not select an editor backend at all and
    could only ever build whatever the input schema defaulted to -- on the lane
    the appearance approach depends on.
    """

    missing = {
        name
        for name in parameters - supplied
        if f"{module}.{name}" not in CLI_PARAMETERS_DELIBERATELY_NOT_OFFERED
    }

    assert not missing, (
        f"{module}.{builder} cannot be given {sorted(missing)} from its command "
        "line. Add flags, or record each in "
        "CLI_PARAMETERS_DELIBERATELY_NOT_OFFERED with a reason."
    )


def test_no_recorded_exemption_outlives_its_parameter() -> None:
    """An exemption for a parameter that no longer exists hides a real gap."""

    live = {
        f"{module}.{name}"
        for module, _, parameters, _ in BUNDLE_CLI_CALLS
        for name in parameters
    }

    assert not set(CLI_PARAMETERS_DELIBERATELY_NOT_OFFERED) - live


#: Modules that produce a file a bundle module *requires*, and so are one link
#: further up the same chain #512 fixed. A bundle with a `main()` is still
#: unbuildable if the thing it demands as input can only be produced by calling
#: a Python function -- which is how the scene-840920 SimReady probe root was
#: discovered to be unreachable while its bundle module looked healthy.
#:
#: Discovered by shape: read each bundle module's REQUIRED_* tuples, then find
#: which module writes those exact filenames.
def _required_input_filenames() -> set[str]:
    names: set[str] = set()
    for path in _bundle_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not target.id.startswith("REQUIRED_"):
                continue
            for element in ast.walk(node.value):
                if isinstance(element, ast.Constant) and isinstance(element.value, str):
                    if element.value.endswith(".json"):
                        names.add(Path(element.value).name)
    return names


def _modules_writing(names: set[str]) -> dict[str, Path]:
    writers: dict[str, Path] = {}
    for path in sorted(SOURCE_ROOT.glob("*.py")):
        if path.name.endswith("_bundle.py"):
            continue
        source = path.read_text(encoding="utf-8")
        for name in names:
            if name in source and "write_json" in source:
                writers.setdefault(name, path)
    return writers


def test_required_bundle_inputs_are_discovered() -> None:
    """A discovery that matched nothing would make the check below vacuous."""

    assert _required_input_filenames()


def test_a_module_producing_a_required_bundle_input_has_a_command_line() -> None:
    """The bundle's own CLI is not enough if its input has none."""

    writers = _modules_writing(_required_input_filenames())
    assert writers, "no producer modules discovered -- the shape scan is broken"
    missing = sorted(
        f"{path.name} produces {name}"
        for name, path in writers.items()
        if not _has_entrypoint(path.read_text(encoding="utf-8"))
    )
    assert not missing, (
        "These modules produce a file a bundle module requires, but cannot be "
        "run from a command line, so the bundle cannot be rebuilt at the "
        "deployed commit: " + "; ".join(missing)
    )
