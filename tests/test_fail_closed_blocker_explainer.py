"""An anonymous fail-closed blocker names the predicates that decided it."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from blueprint_pipeline import fail_closed_blocker_explainer as explainer

VALIDATOR_SOURCE = textwrap.dedent(
    '''
    class PacketError(ValueError):
        pass


    def _require(condition, code):
        if not condition:
            raise PacketError(code)


    def validate(profile, request, frames, commit):
        bindings = request.get("bindings") or {}
        if (
            profile.get("schema_version") != "profile.v1"
            or profile.get("source_commit_sha") != commit
            or request.get("provider_profile") != profile
            or not isinstance(frames, list)
            or len(frames) != request.get("frame_count")
            or bindings.get("digest") is None
        ):
            raise PacketError("packet_configuration_invalid")
        return True


    def admit(cap, ttl, retry_cap):
        _require(cap > 0 and ttl <= 1800 and retry_cap == 0, "admission_bounds_invalid")
        return True
    '''
)


@pytest.fixture
def validators(tmp_path: Path):
    path = tmp_path / "packet_validators.py"
    path.write_text(VALIDATOR_SOURCE, encoding="utf-8")
    namespace: dict = {}
    exec(compile(VALIDATOR_SOURCE, str(path), "exec"), namespace)  # noqa: S102 - test fixture module
    return namespace


def test_or_chain_names_exactly_the_predicates_that_fired(validators) -> None:
    profile = {"schema_version": "profile.v1", "source_commit_sha": "0" * 40}
    request = {"provider_profile": profile, "frame_count": 16, "bindings": {"digest": "sha256:x"}}
    with pytest.raises(validators["PacketError"]) as caught:
        validators["validate"](profile, request, list(range(16)), "1" * 40)

    fired = explainer.fired_predicates(caught.value)

    assert fired == ["profile.get('source_commit_sha') != commit"]
    report = explainer.explain_blocker(caught.value)
    assert report["blocker"] == "packet_configuration_invalid"
    [explanation] = report["explanations"]
    assert explanation["kind"] == "if" and explanation["operator"] == "or"
    assert explanation["predicates_total"] == 6 and explanation["fired_total"] == 1
    assert explanation["function"] == "validate"
    assert (
        explainer.annotate_blocker("packet_configuration_invalid", caught.value)
        == "packet_configuration_invalid:predicates=profile.get('source_commit_sha') != commit"
    )


def test_multiple_fired_predicates_and_requirement_style_calls(validators) -> None:
    profile = {"schema_version": "profile.v0", "source_commit_sha": "0" * 40}
    request = {"provider_profile": {}, "frame_count": 3, "bindings": {}}
    with pytest.raises(validators["PacketError"]) as caught:
        validators["validate"](profile, request, "not-a-list", "0" * 40)
    fired = explainer.fired_predicates(caught.value)
    assert fired == [
        "profile.get('schema_version') != 'profile.v1'",
        "request.get('provider_profile') != profile",
        "not isinstance(frames, list)",
        "len(frames) != request.get('frame_count')",
        "bindings.get('digest') is None",
    ]

    with pytest.raises(validators["PacketError"]) as caught:
        validators["admit"](cap=1.0, ttl=7200, retry_cap=1)
    report = explainer.explain_blocker(caught.value)
    requirement = next(e for e in report["explanations"] if e["kind"] == "require")
    assert requirement["operator"] == "and"
    assert requirement["fired"] == ["ttl <= 1800", "retry_cap == 0"]


def test_unexplainable_failures_leave_the_blocker_code_unchanged(validators) -> None:
    try:
        raise RuntimeError("plain_failure")
    except RuntimeError as exc:
        assert explainer.annotate_blocker("worker_failed:RuntimeError", exc) == "worker_failed:RuntimeError"
        assert explainer.explain_blocker(exc)["explanations"] == []
    # A lone requirement adds nothing the code does not already say.
    with pytest.raises(validators["PacketError"]) as caught:
        validators["_require"](False, "single_requirement_failed")
    assert explainer.annotate_blocker("single_requirement_failed", caught.value) == "single_requirement_failed"


def test_explain_call_reports_instead_of_raising(validators) -> None:
    profile = {"schema_version": "profile.v1", "source_commit_sha": "0" * 40}
    request = {"provider_profile": profile, "frame_count": 1, "bindings": {"digest": "d"}}
    refused = explainer.explain_call(validators["validate"], profile, request, [1], "f" * 40)
    assert refused["status"] == "refused"
    assert refused["explanations"][0]["fired"] == ["profile.get('source_commit_sha') != commit"]
    accepted = explainer.explain_call(validators["validate"], profile, request, [1], "0" * 40)
    assert accepted == {"status": "accepted"}


def test_annotation_is_bounded_and_source_only() -> None:
    long_name = "x" * 400
    namespace: dict = {}
    source = f"def check(value):\n    if (value == '{long_name}' or value is None):\n        raise ValueError('bounded_invalid')\n"
    path = Path(__file__).parent / "_bounded_check_fixture.py"
    try:
        path.write_text(source, encoding="utf-8")
        exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - test fixture module
        with pytest.raises(ValueError) as caught:
            namespace["check"](None)
        annotated = explainer.annotate_blocker("bounded_invalid", caught.value)
    finally:
        path.unlink(missing_ok=True)
    assert annotated.startswith("bounded_invalid:predicates=")
    assert len(annotated) <= explainer.MAX_ANNOTATION_CHARS
    assert "value is None" in annotated
    assert long_name not in annotated  # long predicate text is truncated, never the value


def test_a_comprehension_inside_a_predicate_still_resolves_the_frame_names(tmp_path: Path) -> None:
    source = textwrap.dedent(
        """
        import re
        _DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


        def validate(bindings):
            if (
                not isinstance(bindings, dict)
                or any(_DIGEST.fullmatch(str(bindings.get(f) or "")) is None for f in ("a", "b"))
            ):
                raise ValueError("bindings_invalid")
        """
    )
    path = tmp_path / "comprehension_fixture.py"
    path.write_text(source, encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, str(path), "exec"), namespace)  # noqa: S102 - test fixture module
    with pytest.raises(ValueError) as caught:
        namespace["validate"]({"a": "sha256:" + "0" * 64, "b": "nope"})
    report = explainer.explain_blocker(caught.value)
    [explanation] = report["explanations"]
    assert explanation["evaluation_errors"] == []
    assert explanation["fired"] == [
        "any((_DIGEST.fullmatch(str(bindings.get(f) or '')) is None for f in ('a', 'b')))"
    ]
