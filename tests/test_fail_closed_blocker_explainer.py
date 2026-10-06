"""An anonymous fail-closed blocker names the predicates that decided it."""

from __future__ import annotations

import textwrap
from abc import ABCMeta
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
    # Preserve the validator's short circuit: later operands never ran.
    assert fired == ["profile.get('schema_version') != 'profile.v1'"]

    with pytest.raises(validators["PacketError"]) as caught:
        validators["admit"](cap=1.0, ttl=7200, retry_cap=1)
    report = explainer.explain_blocker(caught.value)
    requirement = next(e for e in report["explanations"] if e["kind"] == "require")
    assert requirement["operator"] == "and"
    assert requirement["fired"] == ["ttl <= 1800"]


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


def test_a_comprehension_with_application_calls_is_declined(tmp_path: Path) -> None:
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
    assert explanation["evaluation_errors"][0].endswith("_UnsafePredicate")
    assert explanation["fired"] == []


def _refusal(tmp_path, source, arguments=()):
    path = tmp_path / 'security_validator.py'
    path.write_text(source)
    scope = {}
    exec(compile(source, str(path), 'exec'), scope)  # noqa: S102 - owned fixture only
    try:
        scope['validate'](*arguments)
    except ValueError as error:
        return scope, error
    raise AssertionError('fixture did not refuse')


@pytest.mark.parametrize('module', ['collections.abc', 'typing'])
@pytest.mark.parametrize('expected', ['RowMapping', '(RowMapping, list)'])
def test_canonical_mapping_alias_reads_exact_dict_without_abc_hooks(tmp_path, monkeypatch, module, expected):
    _, error = _refusal(tmp_path, f'''from {module} import Mapping as RowMapping
def validate(value):
    if not isinstance(value, {expected}) or value.get('schema_version') != 'envelope.v1':
        raise ValueError('refused')
''', ({'schema_version': 'wrong'},))
    events = []
    original_instancecheck = ABCMeta.__instancecheck__
    original_subclasscheck = ABCMeta.__subclasscheck__
    def instancecheck(cls, value):
        events.append('instancecheck')
        return original_instancecheck(cls, value)
    def subclasscheck(cls, value):
        events.append('subclasscheck')
        return original_subclasscheck(cls, value)
    with monkeypatch.context() as patch:
        patch.setattr(ABCMeta, '__instancecheck__', instancecheck)
        patch.setattr(ABCMeta, '__subclasscheck__', subclasscheck)
        report = explainer.explain_blocker(error)
    assert events == []
    assert report['explanations'][0]['fired'] == ["value.get('schema_version') != 'envelope.v1'"]
    assert report['explanations'][0]['evaluation_errors'] == []


@pytest.mark.parametrize(('value', 'expected'), [(True, 'int'), (True, '(RowMapping, int)'), ([], '(RowMapping, list)')])
def test_mapping_tuple_preserves_ordinary_builtin_subtyping(tmp_path, value, expected):
    _, error = _refusal(tmp_path, f'''from collections.abc import Mapping as RowMapping
def validate(value):
    if isinstance(value, {expected}):
        raise ValueError('refused')
''', (value,))
    report = explainer.explain_blocker(error)
    assert report['explanations'][0]['fired'] == [f'isinstance(value, {expected})']
    assert report['explanations'][0]['evaluation_errors'] == []


def test_shadowed_mapping_never_dispatches_metaclass_hooks(tmp_path):
    scope, error = _refusal(tmp_path, '''events = []
class Meta(type):
    def __instancecheck__(cls, value):
        events.append('instancecheck')
        return False
    def __subclasscheck__(cls, value):
        events.append('subclasscheck')
        return False
    def __eq__(cls, other):
        events.append('equality')
        return False
    def __hash__(cls):
        events.append('hash')
        return 1
class Mapping(metaclass=Meta):
    pass
def validate(value):
    if not isinstance(value, Mapping) or value.get('schema_version') != 'envelope.v1':
        raise ValueError('refused')
''', ({'schema_version': 'wrong'},))
    before = list(scope['events'])
    report = explainer.explain_blocker(error)
    assert before == ['instancecheck']
    assert scope['events'] == before
    assert report['explanations'][0]['fired'] == []
    assert report['explanations'][0]['evaluation_errors'][0].endswith('_UnsafePredicate')


@pytest.mark.parametrize('value_type', ['DictSubclass', 'Application'])
def test_canonical_mapping_still_refuses_application_values_without_hooks(tmp_path, value_type):
    scope, error = _refusal(tmp_path, f'''from collections.abc import Mapping
events = []
class Meta(type):
    def __eq__(cls, other):
        events.append('equality')
        return False
    def __hash__(cls):
        events.append('hash')
        return 1
class DictSubclass(dict, metaclass=Meta):
    pass
class Application(metaclass=Meta):
    @property
    def __class__(self):
        events.append('class-property')
        return dict
def validate():
    value = {value_type}()
    if not isinstance(value, Mapping) or True:
        raise ValueError('refused')
''')
    before = list(scope['events'])
    report = explainer.explain_blocker(error)
    assert scope['events'] == before
    assert report['explanations'][0]['fired'] == []
    assert report['explanations'][0]['evaluation_errors'][0].endswith('_UnsafePredicate')


def test_diagnostics_never_visit_original_short_circuited_call(tmp_path):
    scope, error = _refusal(tmp_path, '''events=[]
def effect():
    events.append('must-not-run')
    return True
def validate():
    if True or effect():
        raise ValueError('refused')
''')
    assert scope['events'] == []
    report = explainer.explain_blocker(error)
    assert scope['events'] == []
    assert report['explanations'][0]['fired'] == ['True']
    assert report['explanations'][0]['evaluation_errors'] == []


def test_diagnostics_do_not_repeat_original_application_call(tmp_path):
    scope, error = _refusal(tmp_path, '''events=[]
def effect():
    events.append('original-only')
    return True
def validate():
    if effect() or False:
        raise ValueError('refused')
''')
    report = explainer.explain_blocker(error)
    assert scope['events'] == ['original-only']
    assert report['explanations'][0]['fired'] == []
    assert report['explanations'][0]['evaluation_errors'][0].endswith('_UnsafePredicate')
    assert explainer.annotate_blocker('refused', error) == 'refused'


@pytest.mark.parametrize('operation', ['__bool__', '__eq__', '__len__', '__class__', 'get'])
def test_overloaded_objects_cannot_execute_during_explanation(tmp_path, operation):
    events = []
    class Hostile:
        def __bool__(self):
            events.append('__bool__')
            return True
        def __eq__(self, other):
            events.append('__eq__')
            return True
        def __len__(self):
            events.append('__len__')
            return 1
        @property
        def __class__(self):
            events.append('__class__')
            return list
        def get(self, key):
            events.append('get')
            return True
    conditions = {'__bool__': 'value', '__eq__': 'value == 1', '__len__': 'len(value) == 1',
                  '__class__': 'isinstance(value, list)', 'get': "value.get('x')"}
    _, error = _refusal(tmp_path, f"def validate(value):\n    if {conditions[operation]}:\n        raise ValueError('refused')\n", (Hostile(),))
    before = list(events)
    report = explainer.explain_blocker(error)
    assert events == before
    assert report['explanations'][0]['evaluation_errors'][0].endswith('_UnsafePredicate')


def test_replaced_validator_source_cannot_supply_executable_diagnostics(tmp_path):
    scope, error = _refusal(tmp_path, '''events=[]
def effect():
    events.append('must-not-run')
    return True
def validate():
    if True:
        raise ValueError('refused')
''')
    path = tmp_path / 'security_validator.py'
    path.write_text(path.read_text().replace('if True:', 'if effect():'))
    explainer.explain_blocker(error)
    assert scope['events'] == []


@pytest.mark.parametrize('value', [list(range(1100)), 'x' * 65537, 1 << 257])
def test_data_bounds_decline_without_masking_original_blocker(tmp_path, value):
    _, error = _refusal(tmp_path, "def validate(value):\n    if value or False:\n        raise ValueError('refused')\n", (value,))
    assert explainer.annotate_blocker('refused', error) == 'refused'


def test_unsafe_operand_stops_before_later_pure_truth(tmp_path):
    _, error = _refusal(tmp_path, "def validate(value):\n    if value() or True:\n        raise ValueError('refused')\n", (lambda: True,))
    report = explainer.explain_blocker(error)
    assert report['explanations'][0]['fired'] == []


@pytest.mark.parametrize('condition', ['value', 'values[0]', 'len(value) > 0', 'not isinstance(value, list)'])
def test_type_admission_never_compares_or_hashes_application_metaclasses(tmp_path, condition):
    events = []
    class Meta(type):
        def __eq__(cls, other):
            events.append('metaclass-equality')
            return False
        def __hash__(cls):
            events.append('metaclass-hash')
            return 1
    class Value(metaclass=Meta):
        def __len__(self):
            return 1
    value = Value()
    _, error = _refusal(tmp_path,
        f"def validate(value, values):\n    if {condition}:\n        raise ValueError('refused')\n", (value, [value]))
    assert events == []
    report = explainer.explain_blocker(error)
    assert events == []
    assert report['explanations'][0]['evaluation_errors'][0].endswith('_UnsafePredicate')
    assert explainer.annotate_blocker('refused', error) == 'refused'
    assert events == []


@pytest.mark.parametrize('condition', ["value == ['needle']", "'needle' in value", "value[0] == 'needle'", "value.get('key') == 'needle'"])
def test_mutation_after_validation_cannot_enter_diagnostic_operations(tmp_path, monkeypatch, condition):
    events = []
    class Hostile:
        def __eq__(self, other):
            events.append('must-not-compare')
            return True
    value = {'key': 'needle'} if '.get(' in condition else ['needle']
    _, error = _refusal(tmp_path,
        f"def validate(value):\n    if {condition}:\n        raise ValueError('refused')\n", (value,))
    original_data = explainer._PurePredicate.data
    changed = False
    def mutate_after_validation(reader, data, depth=0):
        nonlocal changed
        snapshot = original_data(reader, data, depth)
        if data is value and not changed:
            changed = True
            if type(value) is dict:
                value['key'] = Hostile()
            else:
                value[0] = Hostile()
        return snapshot
    monkeypatch.setattr(explainer._PurePredicate, 'data', mutate_after_validation)
    report = explainer.explain_blocker(error)
    assert changed
    assert events == []
    assert report['explanations'][0]['fired'] == [condition]
    assert report['explanations'][0]['evaluation_errors'] == []


def test_snapshot_comparisons_preserve_bare_name_alias_chain(tmp_path):
    left, other = ['same'], ['same']
    _, error = _refusal(tmp_path,
        "def validate(left, other, reference):\n    if left == other is reference:\n        raise ValueError('refused')\n", (left, other, other))
    report = explainer.explain_blocker(error)
    assert report['explanations'][0]['fired'] == ['left == other is reference']
    assert report['explanations'][0]['evaluation_errors'] == []


def test_computed_container_identity_is_declined_instead_of_misreported(tmp_path):
    value = [['same']]
    _, error = _refusal(tmp_path,
        "def validate(value):\n    if value[0] is value[0]:\n        raise ValueError('refused')\n", (value,))
    report = explainer.explain_blocker(error)
    assert report['explanations'][0]['fired'] == []
    assert report['explanations'][0]['evaluation_errors'][0].endswith('_UnsafePredicate')


def test_dict_snapshot_never_looks_up_a_key_again_after_validation(tmp_path, monkeypatch):
    events = []
    key = 'needle-key'
    class Colliding:
        def __hash__(self):
            return hash(key)
        def __eq__(self, other):
            events.append('must-not-compare-colliding-key')
            return True
    value = {key: 'needle'}
    _, error = _refusal(tmp_path,
        "def validate(value):\n    if value.get('needle-key') == 'needle':\n        raise ValueError('refused')\n", (value,))
    original_data = explainer._PurePredicate.data
    changed = False
    def replace_key_after_validation(reader, data, depth=0):
        nonlocal changed
        snapshot = original_data(reader, data, depth)
        if data is key and not changed:
            changed = True
            value.clear()
            value[Colliding()] = 'foreign'
        return snapshot
    monkeypatch.setattr(explainer._PurePredicate, 'data', replace_key_after_validation)
    explainer.explain_blocker(error)
    assert changed
    assert events == []
