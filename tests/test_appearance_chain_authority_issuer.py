"""The appearance chain has to be authorizable from a production path.

`scripts/issue_appearance_chain_paid_attempt_authority.py` is the entry point
for both links of the campaign. The interesting failure it guards is not a
crash: it is a flag that quietly does not exist, because the authority it mints
is single-use and carries the campaign's running spend forward against a shared
cap. A parameter the CLI cannot supply is spend the CLI cannot account for.

That is not hypothetical. The first cut of the script hand-listed its flags and
dropped four of ArtiFixer3D's six predecessor parameters, so a second attempt
could not have accounted for the first attempt's spend.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.paired_target_native_import_vast import MAX_HARD_CAP_USD

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "issue_appearance_chain_paid_attempt_authority.py"


def _load():
    name = "issue_appearance_chain"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclass` resolves annotations through `sys.modules[cls.__module__]`,
    # so a file-loaded module has to be registered before it is executed.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


issuer = _load()


@pytest.mark.parametrize("link", sorted(issuer.LINKS), ids=str)
def test_every_materializer_keyword_has_a_flag(link: str) -> None:
    """Derives the requirement from the signature, so upstream cannot outrun it.

    A new keyword on either materializer -- especially another spend anchor --
    fails here until a flag supplies it, rather than being silently defaulted
    away at the point where a paid attempt is authorized.
    """

    entry = issuer.LINKS[link]
    signature = inspect.signature(entry.materialize)
    upstream = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }

    missing = upstream - set(entry.params)
    assert not missing, (
        f"{link} cannot supply {sorted(missing)} from a command line. Add a flag "
        "to the table; do not let a paid authority default it away."
    )


@pytest.mark.parametrize("link", sorted(issuer.LINKS), ids=str)
def test_no_flag_invents_a_keyword_the_materializer_will_not_take(link: str) -> None:
    entry = issuer.LINKS[link]
    upstream = set(inspect.signature(entry.materialize).parameters)

    assert not set(entry.params) - upstream


@pytest.mark.parametrize("link", sorted(issuer.LINKS), ids=str)
def test_flags_are_distinct_within_a_link(link: str) -> None:
    """Two keywords sharing a flag would let one silently overwrite the other."""

    flags = [param.flag for param in issuer.LINKS[link].params.values()]

    assert len(flags) == len(set(flags))


def test_the_artifixer_predecessor_group_is_all_or_nothing_upstream() -> None:
    """Pins why all four flags exist rather than the one that reads as enough.

    Upstream raises `artifixer3d_predecessor_attempt_incomplete` when some but
    not all are supplied, so exposing a subset produces a flag that cannot be
    used at all.
    """

    source = (
        REPO_ROOT / "src" / "blueprint_pipeline" / "public_scene_artifixer3d_vast.py"
    ).read_text(encoding="utf-8")
    assert "artifixer3d_predecessor_attempt_incomplete" in source

    group = {
        "prior_artifixer_authority_path",
        "prior_artifixer_result_path",
        "prior_artifixer_cleanup_path",
        "prior_artifixer_provider_zero_path",
    }
    assert group <= set(issuer.LINKS["artifixer3d"].params)


def test_the_import_gate_can_pin_which_instances_may_already_be_running() -> None:
    """A concurrent lane's instances must not be admitted as ours by default."""

    param = issuer.LINKS["paired-target"].params["allowed_active_instance_ids"]

    assert param.accumulate and param.type is int
    assert param.default == ()


def test_the_import_authority_default_uses_the_allocator_hard_cap() -> None:
    assert issuer.LINKS["paired-target"].params["hard_cap_usd"].default == (
        MAX_HARD_CAP_USD
    )


def test_a_supplied_repeatable_flag_does_not_crash_the_parser() -> None:
    """The case the table test above cannot see, because it never parses a value.

    `action="append"` appends to whatever default it is given, and a tuple has
    no `append`. So pinning `default == ()` -- which is what the materializer
    needs -- made the *only* invocation that names a running instance die in
    `argparse`, on the paid path where naming one matters.
    """

    args = issuer.build_parser().parse_args(
        [
            "paired-target",
            "--bundle-receipt", "b.json",
            "--prior-artifixer-authority", "a.json",
            "--prior-artifixer-result", "r.json",
            "--prior-artifixer-cleanup", "c.json",
            "--prior-artifixer-provider-zero", "z.json",
            "--authorized-by", "operator",
            "--authority-reference", "goal",
            "--blueprint-commit", "e" * 40,
            "--output", "out.json",
            "--allow-active-instance", "42",
            "--allow-active-instance", "7",
        ]
    )

    assert issuer.call_arguments(issuer.LINKS["paired-target"], args)[
        "allowed_active_instance_ids"
    ] == (42, 7)


def test_an_omitted_repeatable_flag_stays_empty_not_none() -> None:
    """`None` would reach a `Sequence[int]` parameter that iterates it."""

    args = issuer.build_parser().parse_args(
        [
            "paired-target",
            "--bundle-receipt", "b.json",
            "--prior-artifixer-authority", "a.json",
            "--prior-artifixer-result", "r.json",
            "--prior-artifixer-cleanup", "c.json",
            "--prior-artifixer-provider-zero", "z.json",
            "--authorized-by", "operator",
            "--authority-reference", "goal",
            "--blueprint-commit", "e" * 40,
            "--output", "out.json",
        ]
    )

    assert issuer.call_arguments(issuer.LINKS["paired-target"], args)[
        "allowed_active_instance_ids"
    ] == ()


def test_the_authorization_date_defaults_to_today_not_to_empty() -> None:
    args = issuer.build_parser().parse_args(
        [
            "artifixer3d",
            "--bundle-receipt", "b.json",
            "--campaign-start-receipt", "start.json",
            "--authorized-by", "operator",
            "--authority-reference", "goal",
            "--blueprint-commit", "e" * 40,
            "--output", "out.json",
        ]
    )

    authorized_on = issuer.call_arguments(issuer.LINKS["artifixer3d"], args)["authorized_on"]

    assert authorized_on and authorized_on.count("-") == 2


@pytest.mark.parametrize("link", sorted(issuer.LINKS), ids=str)
def test_who_authorized_it_and_what_they_authorized_are_both_required(link: str) -> None:
    """Neither is derivable, and an unattributed paid attempt is not authorized."""

    for flag in ("--authorized-by", "--authority-reference"):
        assert issuer.LINKS[link].params[
            {"--authorized-by": "authorized_by", "--authority-reference": "authorization_reference"}[
                flag
            ]
        ].required


def test_a_missing_anchor_is_refused_without_touching_a_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of this script is refusing before anything is rented."""

    code = issuer.main(
        [
            "artifixer3d",
            "--bundle-receipt", str(tmp_path / "absent.json"),
            "--campaign-start-receipt", str(tmp_path / "absent.json"),
            "--authorized-by", "operator",
            "--authority-reference", "goal",
            "--blueprint-commit", "e" * 40,
            "--output", str(tmp_path / "authority.json"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "blocked"
    assert payload["provider_mutation_performed"] is False
    assert payload["blockers"], "a refusal has to name its cause"
    assert not (tmp_path / "authority.json").exists()
